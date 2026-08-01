# Copyright (c) HelkiQuant contributors.
# Licensed under the MIT License.
"""
CatBoostDEnsemble
-----------------
将 DoubleEnsemble 的样本重加权(SR) + 特征选择(FS) + 多子模型集成机制
移植到 CatBoost 上。

与原版 DEnsembleModel (LightGBM) 的关键差异：
    - 子模型训练：CatBoost(params).fit(Pool(...)) 替代 lgb.train()
    - 损失曲线：model.predict(Pool(X), ntree_start=i, ntree_end=i+1)
    - 数据结构：catboost.Pool 替代 lgb.Dataset
    - 其余算法逻辑（SR/FS/集成）完全保留
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Text, Union

from catboost import Pool, CatBoost
from scipy.stats import mode as sp_mode

try:
    from qlib.model.base import Model
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.model.interpret.base import FeatureInt
    from qlib.log import get_module_logger
except ModuleNotFoundError:
    # DEnsemble only needs these lightweight interfaces when a
    # dataframe-backed adapter is supplied without the research extra.
    import logging

    class Model:
        pass

    class DatasetH:
        pass

    class FeatureInt:
        pass

    class DataHandlerLP:
        DK_L = "learn"
        DK_I = "infer"

    def get_module_logger(name):
        return logging.getLogger(name)
from realtime_output import log_step, setup_realtime_output
from training_frame_utils import collapse_identical_datetime_rows


class CatBoostDEnsemble(Model, FeatureInt):
    """以 CatBoost 为基底的 DoubleEnsemble 模型。

    Parameters
    ----------
    loss : str
        CatBoost 损失函数，"RMSE"、"Logloss" 或 "MultiClass"
    thresholds : list[float] or None
        MultiClass 模式下的分类阈值，默认 [-0.10, 0.10]
    num_models : int
        子模型数量
    enable_sr : bool
        是否启用样本重加权 (Sample Reweighting)
    enable_fs : bool
        是否启用特征选择 (Feature Selection)
    alpha1, alpha2 : float
        SR 中 h-value 的权重系数
    bins_sr : int
        SR 中 h-value 分箱数
    bins_fs : int
        FS 中 g-value 分箱数
    decay : float or None
        SR 中的衰减系数，None 则默认 0.5
    sample_ratios : list[float] or None
        FS 中各箱的特征采样比例
    sub_weights : list[float] or None
        各子模型的集成权重
    **kwargs :
        传递给 CatBoost 的超参（learning_rate, max_depth 等）
    """

    def __init__(
        self,
        loss: str = "RMSE",
        thresholds: list | None = None,
        adaptive_thresholds: str | None = None,
        binary_threshold: float = 0.0,
        num_models: int = 6,
        enable_sr: bool = True,
        enable_fs: bool = True,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        bins_sr: int = 10,
        bins_fs: int = 5,
        decay: float | None = None,
        sample_ratios: list | None = None,
        sub_weights: list | None = None,
        ensemble_eval_rows: int | None = 200000,
        random_seed: int = 42,
        collapse_by_datetime: bool = False,
        **kwargs,
    ):
        if loss not in {"RMSE", "Logloss", "MultiClass"}:
            raise NotImplementedError(f"Unsupported loss: {loss}")
        self._is_multiclass = (loss == "MultiClass")
        self._is_binary = (loss == "Logloss")
        self._is_cls = self._is_multiclass or self._is_binary
        self.thresholds = sorted(thresholds) if thresholds is not None else [-0.10, 0.10]
        # 自适应阈值模式：None | "std_ratio"
        #   std_ratio: 按 instrument 训练段标签 std 动态计算阈值 [-1*std, +1*std]
        self.adaptive_thresholds = adaptive_thresholds
        # 二分类标签阈值（label > threshold 记为 1）
        self.binary_threshold = float(binary_threshold)
        # 训练后填充： instrument → (lower_thr, upper_thr)
        self.instrument_thresholds: dict[str, tuple[float, float]] = {}
        self.loss = loss
        self.num_models = num_models
        self.enable_sr = enable_sr
        self.enable_fs = enable_fs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.bins_sr = bins_sr
        self.bins_fs = bins_fs
        self.decay = decay if decay is not None else 0.5
        self.ensemble_eval_rows = (
            None if ensemble_eval_rows is None else int(ensemble_eval_rows)
        )
        self.random_seed = int(random_seed)
        self.collapse_by_datetime = bool(collapse_by_datetime)

        if sample_ratios is None:
            sample_ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
        if len(sample_ratios) != bins_fs:
            raise ValueError("len(sample_ratios) must equal bins_fs")
        self.sample_ratios = sample_ratios

        if sub_weights is None:
            sub_weights = [1] * num_models
        if len(sub_weights) != num_models:
            raise ValueError("len(sub_weights) must equal num_models")
        self.sub_weights = sub_weights

        self._params = {"loss_function": loss}
        self._params["random_seed"] = self.random_seed
        self._params.update(kwargs)
        self.ensemble: list[CatBoost] = []
        self.sub_features: list[pd.Index] = []
        self._rng = np.random.default_rng(self.random_seed)
        self.logger = get_module_logger("CatBoostDEnsemble")

    # ------------------------------------------------------------------ #
    # 数据准备
    # ------------------------------------------------------------------ #
    def _compute_instrument_thresholds(self, y_series: pd.Series) -> None:
        """在训练开始时，按 instrument 计算标签 std 作为阈值。

        仅在 adaptive_thresholds=="std_ratio" 且为 MultiClass 时有效。
        二分类忘记调用。
        """
        if not self._is_multiclass or self.adaptive_thresholds != "std_ratio":
            return
        # y_series.index 是 MultiIndex， level=1 为 instrument
        try:
            grouped = y_series.groupby(level="instrument")
        except Exception:
            # 退化：可能是单级索引，跳过
            return
        for inst, grp in grouped:
            v = grp.values.astype(np.float64)
            v = v[~np.isnan(v)]
            if len(v) < 30:
                continue  # 样本太少，使用全局阈值
            std = float(np.std(v))
            if std < 1e-6:
                continue
            self.instrument_thresholds[str(inst)] = (-std, std)
        self.logger.info(
            f"adaptive_thresholds: {len(self.instrument_thresholds)} 只股票生成个性化阈值"
        )

    def _bin_labels(self, y, y_index=None) -> np.ndarray:
        """将连续标签离散化为类别。

        Parameters
        ----------
        y : array-like
            原始连续标签。
        y_index : pd.MultiIndex or None
            样本索引（含 instrument 级）。当 adaptive_thresholds 启用时必传。
        """
        y_arr = np.asarray(y).ravel()
        # 二分类：Logloss
        if self._is_binary:
            return (y_arr > self.binary_threshold).astype(int)
        # MultiClass
        if self.adaptive_thresholds == "std_ratio" and y_index is not None \
                and len(self.instrument_thresholds) > 0:
            # 逐样本查找所在 instrument 的阈值并应用
            labels = np.zeros(len(y_arr), dtype=int)
            # 准备 instrument 数组（众多果为 MultiIndex，取 instrument 级）
            try:
                inst_arr = y_index.get_level_values("instrument").astype(str).values
            except Exception:
                inst_arr = np.array([""] * len(y_arr))
            lo_default, hi_default = self.thresholds[0], self.thresholds[1]
            for i, val in enumerate(y_arr):
                lo, hi = self.instrument_thresholds.get(inst_arr[i], (lo_default, hi_default))
                if val < lo:
                    labels[i] = 0
                elif val < hi:
                    labels[i] = 1
                else:
                    labels[i] = 2
            return labels
        # 默认全局阈值
        bins = [-np.inf] + self.thresholds + [np.inf]
        return np.digitize(y_arr, bins[1:-1]).astype(int)

    def _prepare_data(
        self, df_train, df_valid, weights, features
    ) -> tuple[Pool, Pool]:
        x_train = df_train["feature"].loc[:, features]
        y_train = df_train["label"]
        x_valid = df_valid["feature"].loc[:, features]
        y_valid = df_valid["label"]

        y_train_1d = np.squeeze(y_train.values)
        y_valid_1d = np.squeeze(y_valid.values)

        # MultiClass / Binary 需要将连续标签离散化
        if self._is_cls:
            y_train_1d = self._bin_labels(y_train_1d, y_index=df_train.index)
            y_valid_1d = self._bin_labels(y_valid_1d, y_index=df_valid.index)

        w_train = weights.values if weights is not None else None
        train_pool = Pool(data=x_train, label=y_train_1d, weight=w_train)
        valid_pool = Pool(data=x_valid, label=y_valid_1d)
        return train_pool, valid_pool

    # ------------------------------------------------------------------ #
    # 子模型训练
    # ------------------------------------------------------------------ #
    def _train_submodel(
        self, df_train, df_valid, weights, features, epochs: int
    ) -> CatBoost:
        setup_realtime_output()
        import time as _time
        t0 = _time.time()
        train_pool, valid_pool = self._prepare_data(
            df_train, df_valid, weights, features
        )
        params = dict(self._params)
        params["iterations"] = epochs
        # Default to CPU without probing CUDA. Users with a compatible CUDA
        # setup can still explicitly pass task_type="GPU" in the model config.
        params.setdefault("task_type", "CPU")
        # 进度可见性：每 max(1, epochs/20) 轮打印一次 loss
        if "verbose" not in params and "verbose_eval" not in params and \
                "logging_level" not in params:
            params["verbose"] = max(1, epochs // 20)
        log_step(
            f"    [CatBoost] iterations={epochs} task={params['task_type']} "
            f"n_train={train_pool.num_row()} n_valid={valid_pool.num_row()} "
            f"n_features={train_pool.num_col()}"
        )
        model = CatBoost(params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        log_step(
            f"    [CatBoost] done in {_time.time()-t0:.1f}s, "
            f"best_iter={model.get_best_iteration()}, trees={model.tree_count_}"
        )
        return model

    # ------------------------------------------------------------------ #
    # 损失曲线（SR 的核心）
    # ------------------------------------------------------------------ #
    def _retrieve_loss_curve(self, model: CatBoost, df_train, features) -> pd.DataFrame:
        """逐迭代提取损失曲线，shape = (N, T)。

        使用 model.predict(Pool(X), ntree_start=i, ntree_end=i+1)
        逐棵树累加预测，记录每步的样本损失。

        对于 MultiClass：累加的是 logit（原始分数），预测概率通过 softmax 得到。
        """
        x_train = df_train["feature"].loc[:, features]
        y_train = df_train["label"]
        y_1d = self._bin_labels(np.squeeze(y_train.values), y_index=df_train.index) if self._is_cls else np.squeeze(y_train.values)
        N = x_train.shape[0]

        num_trees = model.tree_count_
        pred_pool = Pool(data=x_train)

        loss_curve = pd.DataFrame(np.zeros((N, num_trees), dtype=float))

        import time as _time
        t0 = _time.time()
        log_step = max(1, num_trees // 10)  # 每 10% 打印一次
        print(f"    [loss_curve] N={N} trees={num_trees} cls={self._is_cls} "
              f"binary={self._is_binary}", flush=True)
        if self._is_multiclass:
            # MultiClass: 累加 logit，用 softmax 概率计算交叉熵
            num_classes = model.classes_.shape[0]
            pred_logit = np.zeros((N, num_classes), dtype=float)
            for t in range(num_trees):
                logit_t = model.predict(
                    pred_pool, ntree_start=t, ntree_end=t + 1,
                    prediction_type="RawFormulaVal",
                )
                pred_logit += np.array(logit_t)
                # softmax → 概率
                exp_logit = np.exp(pred_logit - pred_logit.max(axis=1, keepdims=True))
                proba = exp_logit / exp_logit.sum(axis=1, keepdims=True)
                # 真实类别的交叉熵
                prob_true = proba[np.arange(N), y_1d]
                loss_curve.iloc[:, t] = -np.log(np.clip(prob_true, 1e-7, 1.0))
                if (t + 1) % log_step == 0 or t + 1 == num_trees:
                    print(f"    [loss_curve] tree {t+1}/{num_trees} "
                          f"({(t+1)*100/num_trees:.0f}%) elapsed={_time.time()-t0:.1f}s",
                          flush=True)
        elif self._is_binary:
            # Binary Logloss: 累加 logit（1D），sigmoid → P(1)，BCE loss
            pred_logit = np.zeros(N, dtype=float)
            y_bin = y_1d.astype(float)
            for t in range(num_trees):
                logit_t = model.predict(
                    pred_pool, ntree_start=t, ntree_end=t + 1,
                    prediction_type="RawFormulaVal",
                )
                pred_logit += np.array(logit_t).ravel()
                proba1 = 1.0 / (1.0 + np.exp(-pred_logit))
                proba1 = np.clip(proba1, 1e-7, 1 - 1e-7)
                loss_curve.iloc[:, t] = -(y_bin * np.log(proba1) +
                                          (1 - y_bin) * np.log(1 - proba1))
                if (t + 1) % log_step == 0 or t + 1 == num_trees:
                    print(f"    [loss_curve] tree {t+1}/{num_trees} "
                          f"({(t+1)*100/num_trees:.0f}%) elapsed={_time.time()-t0:.1f}s",
                          flush=True)
        else:
            pred_cum = np.zeros(N, dtype=float)
            for t in range(num_trees):
                pred_t = model.predict(
                    pred_pool, ntree_start=t, ntree_end=t + 1
                )
                pred_cum += pred_t
                loss_curve.iloc[:, t] = self._get_loss(y_1d, pred_cum)
                if (t + 1) % log_step == 0 or t + 1 == num_trees:
                    print(f"    [loss_curve] tree {t+1}/{num_trees} "
                          f"({(t+1)*100/num_trees:.0f}%) elapsed={_time.time()-t0:.1f}s",
                          flush=True)
        print(f"    [loss_curve] done in {_time.time()-t0:.1f}s", flush=True)
        return loss_curve

    # ------------------------------------------------------------------ #
    # 样本重加权 (SR) — 框架无关
    # ------------------------------------------------------------------ #
    def _sample_reweight(
        self, loss_curve: pd.DataFrame, loss_values: pd.Series, k_th: int
    ) -> pd.Series:
        N, T = loss_curve.shape
        # 统一使用 RangeIndex 避免 MultiIndex 对齐问题
        loss_curve_reset = loss_curve.reset_index(drop=True)
        loss_values_reset = loss_values.reset_index(drop=True)

        loss_curve_norm = loss_curve_reset.rank(axis=0, pct=True)
        loss_values_norm = (-loss_values_reset).rank(pct=True)

        part = max(int(T * 0.1), 1)
        l_start = loss_curve_norm.iloc[:, :part].mean(axis=1)
        l_end = loss_curve_norm.iloc[:, -part:].mean(axis=1)

        h1 = loss_values_norm
        h2 = (l_end / l_start).rank(pct=True)
        h = pd.DataFrame({"h": self.alpha1 * h1 + self.alpha2 * h2})
        h["bins"] = pd.cut(h["h"], self.bins_sr)
        h_avg = h.groupby("bins", group_keys=False, observed=False)["h"].mean()

        weights = pd.Series(np.zeros(N, dtype=float))
        for b in h_avg.index:
            weights[h["bins"] == b] = 1.0 / (self.decay ** k_th * h_avg[b] + 0.1)
        return weights

    # ------------------------------------------------------------------ #
    # 特征选择 (FS) — 框架无关（只需 submodel.predict）
    # ------------------------------------------------------------------ #
    def _feature_selection(
        self, df_train, loss_values: pd.Series
    ) -> pd.Index:
        x_train = df_train["feature"]
        y_train = df_train["label"]
        features = x_train.columns
        N, F = x_train.shape
        M = len(self.ensemble)

        # 预计算 numpy 数组和子模型特征索引，避免重复 DataFrame 操作
        x_np = x_train.values.copy()
        loss_values_np = loss_values.values
        feat_to_idx = {feat: i for i, feat in enumerate(features)}
        sub_feat_indices = [
            np.array([feat_to_idx[f] for f in self.sub_features[i_s]])
            for i_s in range(M)
        ]

        g_values = np.zeros(F, dtype=float)

        for i_f, feat in enumerate(features):
            # 原地置换特征列
            col_idx = feat_to_idx[feat]
            original = x_np[:, col_idx].copy()
            self._rng.shuffle(x_np[:, col_idx])

            if self._is_cls:
                y_1d = self._bin_labels(np.squeeze(y_train.values), y_index=df_train.index)
                votes = np.zeros((N, M), dtype=int)
                for i_s in range(M):
                    # prediction_type="Class" 返回类别索引，比 probability + argmax 快
                    pred_cls = self.ensemble[i_s].predict(
                        x_np[:, sub_feat_indices[i_s]],
                        prediction_type="Class",
                    )
                    votes[:, i_s] = np.array(pred_cls, dtype=int).ravel()
                # 快速多数投票（替代 sp_mode）
                ensemble_cls = np.apply_along_axis(
                    lambda row: np.bincount(row).argmax(), 1, votes
                )
                loss_feat = (ensemble_cls != y_1d).astype(float)
            else:
                pred = np.zeros(N, dtype=float)
                for i_s in range(M):
                    pred += self.ensemble[i_s].predict(
                        x_np[:, sub_feat_indices[i_s]]
                    ) / M
                loss_feat = self._get_loss(y_train.values.squeeze(), pred)

            diff = loss_feat - loss_values_np
            std_diff = np.std(diff)
            g_values[i_f] = np.mean(diff) / (std_diff + 1e-7)

            # 恢复原始特征列
            x_np[:, col_idx] = original

        # 替换 NaN
        g_values = np.where(np.isnan(g_values), 0, g_values)

        # 分箱采样
        g_series = pd.Series(g_values, index=features)
        g_bins = pd.cut(g_series, self.bins_fs)

        res_feat = []
        sorted_bins = sorted(g_bins.unique(), reverse=True)
        for i_b, b in enumerate(sorted_bins):
            b_feat = features[g_bins == b]
            num_feat = int(np.ceil(self.sample_ratios[i_b] * len(b_feat)))
            res_feat = res_feat + self._rng.choice(
                b_feat, size=num_feat, replace=False
            ).tolist()
        return pd.Index(dict.fromkeys(res_feat))

    # ------------------------------------------------------------------ #
    # 损失函数
    # ------------------------------------------------------------------ #
    def _get_loss(self, label, pred) -> np.ndarray:
        if self.loss == "RMSE":
            return (label - pred) ** 2
        elif self.loss == "Logloss":
            # 稳定性处理
            pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)
            return -(label * np.log(pred_clipped) + (1 - label) * np.log(1 - pred_clipped))
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

    # ------------------------------------------------------------------ #
    # fit — 主训练循环
    # ------------------------------------------------------------------ #
    def fit(self, dataset: DatasetH, epochs: int = 100, **kwargs):
        setup_realtime_output()
        log_step(f"[DEnsemble] dataset.prepare train/valid start epochs={epochs}")
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset")

        if self.collapse_by_datetime:
            train_rows = len(df_train)
            valid_rows = len(df_valid)
            df_train = collapse_identical_datetime_rows(
                df_train, context="CatBoostDEnsemble train"
            )
            df_valid = collapse_identical_datetime_rows(
                df_valid, context="CatBoostDEnsemble valid"
            )
            log_step(
                "[DEnsemble] collapsed identical datetime rows "
                f"train={train_rows}->{len(df_train)} valid={valid_rows}->{len(df_valid)}"
            )

        x_train, y_train = df_train["feature"], df_train["label"]
        N, F = x_train.shape

        # 重置训练状态，确保同一实例重复 fit 可复现且不会混入旧子模型。
        self.ensemble = []
        self.sub_features = []
        self._rng = np.random.default_rng(self.random_seed)
        self.instrument_thresholds = {}
        # 仅 MultiClass + adaptive_thresholds="std_ratio" 时生成个股阈值
        if self._is_multiclass and self.adaptive_thresholds == "std_ratio":
            try:
                y_series = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train
                self._compute_instrument_thresholds(y_series)
            except Exception as _e:
                self.logger.warning(f"adaptive_thresholds 计算失败: {_e}")

        weights = pd.Series(np.ones(N, dtype=float))
        features = x_train.columns
        if self.ensemble_eval_rows is not None and N > self.ensemble_eval_rows:
            rng = np.random.default_rng(self.random_seed)
            eval_pos = np.sort(
                rng.choice(N, size=self.ensemble_eval_rows, replace=False)
            )
            df_ensemble = df_train.iloc[eval_pos]
            print(
                f"  [DEnsemble] SR/FS evaluation subsample: "
                f"{len(df_ensemble)}/{N} rows",
                flush=True,
            )
        else:
            eval_pos = np.arange(N)
            df_ensemble = df_train
        pred_sub = pd.DataFrame(
            np.zeros((len(df_ensemble), self.num_models), dtype=float),
            index=df_ensemble.index,
        )

        import time as _time
        t_layer = _time.time()
        for k in range(self.num_models):
            t_k = _time.time()
            self.sub_features.append(features)
            print(f"  [DEnsemble] === sub-model {k+1}/{self.num_models} === "
                  f"features={len(features)} weighted_N={N}", flush=True)
            self.logger.info(f"Training sub-model ({k + 1}/{self.num_models})")

            model_k = self._train_submodel(
                df_train, df_valid, weights, features, epochs
            )
            self.ensemble.append(model_k)
            print(f"  [DEnsemble] sub-model {k+1} total {_time.time()-t_k:.1f}s "
                  f"(layer elapsed {_time.time()-t_layer:.1f}s)", flush=True)

            if k + 1 == self.num_models:
                break

            # 损失曲线 + 集成预测
            self.logger.info("Computing loss curve...")
            loss_curve = self._retrieve_loss_curve(model_k, df_ensemble, features)

            if self._is_cls:
                # MultiClass: 用概率信号作为集成预测
                proba_k = model_k.predict(
                    df_ensemble["feature"].loc[:, features].values,
                    prediction_type="Probability",
                )
                pred_k = pd.Series(
                    np.array(proba_k)[:, -1] - np.array(proba_k)[:, 0],
                    index=df_ensemble.index,
                )
            else:
                pred_k = pd.Series(
                    model_k.predict(df_ensemble["feature"].loc[:, features].values),
                    index=df_ensemble.index,
                )
            pred_sub.iloc[:, k] = pred_k

            if self._is_cls:
                # MultiClass: 集成投票 + 误判损失
                y_1d = self._bin_labels(
                    np.squeeze(df_ensemble["label"].values),
                    y_index=df_ensemble.index,
                )
                votes = np.zeros((len(df_ensemble), self.num_models), dtype=int)
                for j in range(k + 1):
                    proba_j = self.ensemble[j].predict(
                        df_ensemble["feature"].loc[:, self.sub_features[j]].values,
                        prediction_type="Probability",
                    )
                    votes[:, j] = np.argmax(np.array(proba_j), axis=1)
                # 多数投票
                ensemble_cls = sp_mode(votes[:, : k + 1], axis=1, keepdims=False).mode
                # 确保索引与训练数据一致
                loss_values = pd.Series(
                    (ensemble_cls != y_1d).astype(float), index=df_ensemble.index
                )
            else:
                pred_ensemble = (
                    pred_sub.iloc[:, : k + 1] * self.sub_weights[: k + 1]
                ).sum(axis=1) / np.sum(self.sub_weights[: k + 1])
                loss_values = pd.Series(
                    self._get_loss(
                        df_ensemble["label"].values.squeeze(),
                        pred_ensemble.values,
                    ),
                    index=df_ensemble.index,
                )

            if self.enable_sr:
                self.logger.info("Sample reweighting...")
                eval_weights = self._sample_reweight(loss_curve, loss_values, k + 1)
                # SR is estimated on a deterministic representative subset,
                # then applied only to those rows. Remaining rows keep neutral
                # weight 1.0, avoiding a multi-gigabyte full loss curve.
                weights = pd.Series(np.ones(N, dtype=float))
                weights.iloc[eval_pos] = eval_weights.values

            if self.enable_fs:
                self.logger.info("Feature selection...")
                features = self._feature_selection(df_ensemble, loss_values)

    # ------------------------------------------------------------------ #
    # predict — 集成预测
    # ------------------------------------------------------------------ #
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.ensemble:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        N = x_test.shape[0]

        if self._is_cls:
            # 集成：子模型概率加权平均
            num_classes = self.ensemble[0].classes_.shape[0]
            proba_sum = np.zeros((N, num_classes), dtype=float)
            w_sum = 0.0
            for i_sub, submodel in enumerate(self.ensemble):
                feat_sub = self.sub_features[i_sub]
                proba = submodel.predict(
                    x_test.loc[:, feat_sub].values,
                    prediction_type="Probability",
                )
                proba_sum += np.array(proba) * self.sub_weights[i_sub]
                w_sum += self.sub_weights[i_sub]
            proba_avg = proba_sum / w_sum
            if self._is_binary:
                # Binary: 返回 P(1) ∈ [0, 1]，方便 main.py 直接用 0.55/0.45 阈值
                signal = proba_avg[:, 1]
            else:
                # MultiClass: 返回 P(最后一类) - P(第一类) ∈ [-1, 1]
                signal = proba_avg[:, -1] - proba_avg[:, 0]
            return pd.Series(signal, index=x_test.index)
        else:
            # 回归: 加权平均
            pred = pd.Series(np.zeros(N), index=x_test.index)
            for i_sub, submodel in enumerate(self.ensemble):
                feat_sub = self.sub_features[i_sub]
                pred += (
                    pd.Series(
                        submodel.predict(x_test.loc[:, feat_sub].values),
                        index=x_test.index,
                    )
                    * self.sub_weights[i_sub]
                )
            pred = pred / np.sum(self.sub_weights)
            return pred

    # ------------------------------------------------------------------ #
    # 特征重要性
    # ------------------------------------------------------------------ #
    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        res = []
        for _model, _weight in zip(self.ensemble, self.sub_weights):
            fi = pd.Series(
                _model.get_feature_importance(*args, **kwargs),
                index=_model.feature_names_,
            )
            res.append(fi * _weight)
        return pd.concat(res, axis=1, sort=False).sum(axis=1).sort_values(
            ascending=False
        )
