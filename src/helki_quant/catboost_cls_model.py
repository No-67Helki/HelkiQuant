# Copyright (c) Qlib_helki multi-layer strategy.
# Licensed under the MIT License.
"""
CatBoost 分类模型 - 用于外层行情状态识别（主升浪/震荡/主跌浪）。

与 qlib.contrib.model.catboost_model.CatBoostModel 的区别：
    1. loss 固定为 MultiClass（多分类）
    2. 训练时将连续标签离散化为 3 个类别
    3. 三分类预测时返回 P(上涨) - P(下跌)；二分类预测时返回 P(1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Text, Union

from catboost import Pool, CatBoost

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.interpret.base import FeatureInt
from realtime_output import log_step, setup_realtime_output
from training_frame_utils import collapse_identical_datetime_rows


class CatBoostClsModel(Model, FeatureInt):
    """CatBoost 多分类模型，将连续收益率标签离散化为类别后训练。

    Parameters
    ----------
    num_classes : int
        类别数量，默认 3（主升浪/震荡/主跌浪）
    thresholds : list[float]
        分类阈值，默认 [-0.10, 0.10]（即 < -10% = 主跌浪，> 10% = 主升浪）
    **kwargs :
        传递给 CatBoost 的参数
    """

    def __init__(
        self,
        num_classes: int = 3,
        thresholds: list | None = None,
        collapse_by_datetime: bool = False,
        **kwargs,
    ):
        if thresholds is None:
            thresholds = [-0.10, 0.10]
        self.num_classes = num_classes
        self.thresholds = sorted(thresholds)
        self.collapse_by_datetime = bool(collapse_by_datetime)
        # 过滤掉自定义参数，只保留 CatBoost 认识的参数
        cb_kwargs = {k: v for k, v in kwargs.items()
                     if k not in ("num_classes", "thresholds")}
        self._params = {"loss_function": "MultiClass"}
        self._params.update(cb_kwargs)
        self.model = None

    def _bin_labels(self, y: np.ndarray) -> np.ndarray:
        """将连续收益率转换为类别标签。

        bins: (-inf, t0) → 0, [t0, t1) → 1, [t1, +inf) → 2
        """
        bins = [-np.inf] + self.thresholds + [np.inf]
        return np.digitize(y, bins[1:-1]).astype(int)  # 0, 1, 2

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 20,
        reweighter=None,
        **kwargs,
    ):
        setup_realtime_output()
        log_step("[CatBoostClsModel] dataset.prepare train/valid start")
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        if self.collapse_by_datetime:
            train_rows = len(df_train)
            valid_rows = len(df_valid)
            df_train = collapse_identical_datetime_rows(
                df_train, context="CatBoostClsModel train"
            )
            df_valid = collapse_identical_datetime_rows(
                df_valid, context="CatBoostClsModel valid"
            )
            log_step(
                "[CatBoostClsModel] collapsed identical datetime rows "
                f"train={train_rows}->{len(df_train)} valid={valid_rows}->{len(df_valid)}"
            )

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        log_step(
            "[CatBoostClsModel] dataset ready "
            f"n_train={len(x_train)} n_valid={len(x_valid)} n_features={x_train.shape[1]}"
        )

        y_train_1d = np.squeeze(y_train.values)
        y_valid_1d = np.squeeze(y_valid.values)

        # 离散化标签
        y_train_cls = self._bin_labels(y_train_1d)
        y_valid_cls = self._bin_labels(y_valid_1d)

        train_pool = Pool(data=x_train, label=y_train_cls)
        valid_pool = Pool(data=x_valid, label=y_valid_cls)

        params = dict(self._params)
        params["iterations"] = num_boost_round
        params["early_stopping_rounds"] = early_stopping_rounds
        if not any(key in params for key in ("verbose", "logging_level", "silent", "verbose_eval")):
            params["verbose"] = verbose_eval
        elif not any(key in params for key in ("verbose", "logging_level", "silent")):
            params["verbose_eval"] = verbose_eval
        # Avoid noisy CUDA probing on CPU-only or driver-incompatible hosts.
        # task_type="GPU" remains available when explicitly set in the config.
        params.setdefault("task_type", "CPU")
        self.model = CatBoost(params, **kwargs)
        log_step(
            "[CatBoostClsModel] fit start "
            f"iterations={num_boost_round} early_stopping={early_stopping_rounds} "
            f"task={params.get('task_type')}"
        )
        self.model.fit(train_pool, eval_set=valid_pool, use_best_model=True, **kwargs)
        log_step(
            "[CatBoostClsModel] fit done "
            f"best_iter={self.model.get_best_iteration()} trees={self.model.tree_count_}"
        )

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> pd.Series:
        """预测：返回 P(类别最高) - P(类别最低) 作为连续信号。

        3 类场景：返回 P(上涨) - P(下跌)，值域 [-1, 1]。
        2 类场景：返回 P(1)，用于日内T方向概率。
        """
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        proba = self.model.predict(x_test.values, prediction_type="Probability")
        proba = np.array(proba)
        if proba.ndim == 2 and proba.shape[1] >= 3:
            # 3 类: P(主升浪) - P(主跌浪)
            signal = proba[:, -1] - proba[:, 0]
        elif proba.ndim == 2 and proba.shape[1] == 2:
            signal = proba[:, 1]
        else:
            signal = proba.squeeze()
        return pd.Series(signal, index=x_test.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        return pd.Series(
            data=self.model.get_feature_importance(*args, **kwargs),
            index=self.model.feature_names_,
        ).sort_values(ascending=False)
