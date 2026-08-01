# Copyright (c) Qlib_helki multi-layer strategy.
# Licensed under the MIT License.
"""
三层嵌套机器学习策略 - 训练 + 回测入口

流程：
    1. 初始化 Qlib
    2. 训练外/中层模型（1666 只日频池）与内层模型（80 只分钟池预计算字段）
    3. 生成预测信号
    4. 用 MultiLayerStrategy + 嵌套 TWAP 执行回测
    5. 输出回测报告

用法：
    python workflow.py run_once
    python workflow.py run_once --config config_densemble_v2.yaml --exp_name my_exp
"""
from __future__ import annotations

import json
import sys
import time
import datetime as _dt
from pathlib import Path

import fire
import joblib
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R

DIRNAME = Path(__file__).absolute().resolve().parent
INTRADAY_DIR = DIRNAME.parent / "intraday_t"
if str(DIRNAME) not in sys.path:
    sys.path.insert(0, str(DIRNAME))
if str(INTRADAY_DIR) not in sys.path:
    sys.path.insert(0, str(INTRADAY_DIR))

from realtime_output import log_step, setup_realtime_output

_T0 = time.time()


def _ts():
    """返回 [HH:MM:SS  +XXX.Xs]。"""
    now = _dt.datetime.now().strftime("%H:%M:%S")
    return f"[{now}  +{time.time()-_T0:.1f}s]"


def _load_yaml(path: Path) -> dict:
    yaml = YAML(typ="safe", pure=True)
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f)


def _build_dataset_and_model(cfg: dict):
    """根据配置构建 (model, dataset) 并训练。"""
    setup_realtime_output()
    log_step("[build] init_instance_by_config(model)")
    model = init_instance_by_config(cfg["model"])
    log_step("[build] init_instance_by_config(dataset) - loading data/processors")
    t_d = time.time()
    dataset = init_instance_by_config(cfg["dataset"])
    log_step(f"[build] dataset ready ({time.time()-t_d:.1f}s)")
    fit_params = cfg.get("fit_params", {})
    log_step(f"[build] model.fit start fit_params={fit_params}")
    t_f = time.time()
    model.fit(dataset, **fit_params)
    log_step(f"[build] model.fit done ({time.time()-t_f:.1f}s)")
    return model, dataset


def _generate_prediction(model, dataset, segment="test"):
    """生成预测信号。"""
    log_step(f"[predict] segment={segment} start")
    pred = model.predict(dataset, segment=segment)
    log_step(f"[predict] segment={segment} done rows={len(pred)}")
    if isinstance(pred, pd.Series):
        return pred
    return pd.Series(pred, index=dataset.prepare(segment, col_set="feature").index)


def _export_layer(model, dataset, layer_name: str, out_dir: Path,
                  data_handler_kwargs: dict | None = None):
    """导出某层模型到 out_dir/<layer_name>/。

    产出：
        - submodel_<i>.cbm: 各 CatBoost 子模型
        - sub_features_<i>.json: 各子模型保留的特征列名
        - sub_weights.json: 集成权重
        - meta.json: loss / thresholds / num_models / 全特征列等
        - feature_names.txt: handler 输出的全部特征名（推理时按此对齐）
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = out_dir / layer_name
    layer_dir.mkdir(parents=True, exist_ok=True)

    # 1. 子模型 + 子特征
    for i, sub in enumerate(model.ensemble):
        sub.save_model(str(layer_dir / f"submodel_{i}.cbm"))
        feats = list(model.sub_features[i])
        (layer_dir / f"sub_features_{i}.json").write_text(
            json.dumps(feats, ensure_ascii=False), encoding="utf-8"
        )

    # 2. 元数据
    meta = {
        "loss": model.loss,
        "is_cls": bool(model._is_cls),
        "is_multiclass": bool(getattr(model, "_is_multiclass", False)),
        "is_binary": bool(getattr(model, "_is_binary", False)),
        "thresholds": list(model.thresholds) if model._is_cls else None,
        "binary_threshold": float(getattr(model, "binary_threshold", 0.0)),
        "adaptive_thresholds": getattr(model, "adaptive_thresholds", None),
        # 自适应阈值：MultiClass 时按 instrument 学到的 (low, high)，供推理端可选回放
        "instrument_thresholds": {
            str(k): [float(v[0]), float(v[1])]
            for k, v in getattr(model, "instrument_thresholds", {}).items()
        },
        "num_models": int(model.num_models),
        "sub_weights": list(model.sub_weights),
        "alpha1": model.alpha1,
        "alpha2": model.alpha2,
        "decay": model.decay,
    }
    (layer_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3. 全特征列名（推理 handler 输出顺序基准）
    try:
        x_any = dataset.prepare("test", col_set="feature")
        all_feats = list(x_any.columns)
    except Exception:
        all_feats = list(model.sub_features[0])
    (layer_dir / "feature_names.txt").write_text(
        "\n".join(all_feats), encoding="utf-8"
    )

    # 4. handler 配置（推理重建用）
    if data_handler_kwargs is not None:
        (layer_dir / "handler_kwargs.json").write_text(
            json.dumps(data_handler_kwargs, ensure_ascii=False, indent=2,
                       default=str),
            encoding="utf-8",
        )

    # 4b. 提取 RobustZScoreNorm 参数（在线单股推理使用，跳过 CSZ）
    try:
        handler = dataset.handler
        proc_list = getattr(handler, "_infer_processors", None) or \
                    getattr(handler, "infer_processors", None) or []
        rz_median = None
        rz_mad = None
        rz_cols = None
        for p in proc_list:
            cls_name = type(p).__name__
            if cls_name == "RobustZScoreNorm":
                rz_median = np.asarray(getattr(p, "mean_train", None))
                rz_mad = np.asarray(getattr(p, "std_train", None))
                raw_cols = list(getattr(p, "cols", []))
                # cols 可能是 MultiIndex tuple list (("feature","KMID"))
                # 也可能是单层 string list；统一取最后一级
                rz_cols = [c[-1] if isinstance(c, tuple) else str(c)
                           for c in raw_cols]
                break
        if rz_median is not None and rz_mad is not None and len(rz_cols) > 0:
            np.savez(
                layer_dir / "norm_params.npz",
                median=rz_median.astype(np.float64),
                scaled_mad=rz_mad.astype(np.float64),
            )
            (layer_dir / "norm_cols.txt").write_text(
                "\n".join(str(c) for c in rz_cols), encoding="utf-8"
            )
            print(f"  [export] {layer_name} norm_params: "
                  f"median.shape={rz_median.shape}, "
                  f"mad.shape={rz_mad.shape}, cols={len(rz_cols)}")
        else:
            print(f"  [warn] {layer_name} 未找到 RobustZScoreNorm 参数，"
                  "在线推理将跳过该归一化")
    except Exception as e:
        import traceback as _tb
        print(f"  [warn] {layer_name} norm 参数导出失败: {e}")
        _tb.print_exc()

    # 5. joblib 全量备份（双保险）
    try:
        joblib.dump(model, layer_dir / "model_full.pkl")
    except Exception as e:
        print(f"  [warn] joblib dump failed for {layer_name}: {e}")

    print(f"  [export] {layer_name} -> {layer_dir} "
          f"({len(model.ensemble)} sub-models, {len(all_feats)} features)")


def run_once(
    config: str = str(DIRNAME / "config_densemble_v2.yaml"),
    exp_name: str = "multi_layer_strategy",
    artifacts_dir: str | None = None,
):
    """单次训练三层模型 + 回测。

    Parameters
    ----------
    config : str
        YAML 配置文件路径
    exp_name : str
        MLflow 实验名
    artifacts_dir : str
        三层模型导出目录；None 则取 <multi_layer>/artifacts/<exp_name>/
    """
    setup_realtime_output()
    cfg = _load_yaml(Path(config))

    # ---- 初始化 Qlib ----
    qlib.init(**cfg["qlib_init"])

    stock_id = cfg["stock_id"]
    backtest_cfg = cfg["backtest"]

    if artifacts_dir is None:
        out_dir = DIRNAME / "artifacts" / exp_name
    else:
        out_dir = Path(artifacts_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[artifacts] 三层模型将导出至: {out_dir}")

    # ================================================================ #
    # 1. 训练外层模型（行情状态分类）
    # ================================================================ #
    print("=" * 60)
    print(f"{_ts()} [1/3] 训练外层模型 (CatBoost 分类: 主升浪/震荡/主跌浪)")
    print("=" * 60)
    _t_outer = time.time()
    outer_model, outer_dataset = _build_dataset_and_model(cfg["outer_model"])
    pred_outer = _generate_prediction(outer_model, outer_dataset)
    print(f"{_ts()} [1/3] 外层训练+预测完成 耗时={time.time()-_t_outer:.1f}s")
    print(f"  外层预测样本数: {len(pred_outer)}")
    print(f"  外层信号统计: mean={pred_outer.mean():.4f}, "
          f"std={pred_outer.std():.4f}, "
          f"min={pred_outer.min():.4f}, max={pred_outer.max():.4f}")
    _export_layer(outer_model, outer_dataset, "outer", out_dir,
                  data_handler_kwargs=cfg["outer_model"]["dataset"]["kwargs"]
                                          ["handler"]["kwargs"])

    # ================================================================ #
    # 2. 训练中层模型（短期波段三分类）
    # ================================================================ #
    print("\n" + "=" * 60)
    print(f"{_ts()} [2/3] 训练中层模型 (CatBoost 分类: 短期波段三态)")
    print("=" * 60)
    _t_mid = time.time()
    middle_model, middle_dataset = _build_dataset_and_model(cfg["middle_model"])
    pred_middle = _generate_prediction(middle_model, middle_dataset)
    print(f"{_ts()} [2/3] 中层训练+预测完成 耗时={time.time()-_t_mid:.1f}s")
    print(f"  中层预测样本数: {len(pred_middle)}")
    print(f"  中层信号统计: mean={pred_middle.mean():.4f}, "
          f"std={pred_middle.std():.4f}, "
          f"min={pred_middle.min():.4f}, max={pred_middle.max():.4f}")
    _export_layer(middle_model, middle_dataset, "middle", out_dir,
                  data_handler_kwargs=cfg["middle_model"]["dataset"]["kwargs"]
                                          ["handler"]["kwargs"])

    # ================================================================ #
    # 3. 训练内层模型（日内T方向二分类）
    # ================================================================ #
    print("\n" + "=" * 60)
    print(f"{_ts()} [3/3] 训练内层模型 (CatBoost 分类: 下一交易日日内T收益方向)")
    print("=" * 60)
    _t_inner = time.time()
    inner_model, inner_dataset = _build_dataset_and_model(cfg["inner_model"])
    pred_inner = _generate_prediction(inner_model, inner_dataset)
    print(f"{_ts()} [3/3] 内层训练+预测完成 耗时={time.time()-_t_inner:.1f}s")
    print(f"  内层预测样本数: {len(pred_inner)}")
    print(f"  内层信号统计: mean={pred_inner.mean():.4f}, "
          f"std={pred_inner.std():.4f}, "
          f"min={pred_inner.min():.4f}, max={pred_inner.max():.4f}")
    _export_layer(inner_model, inner_dataset, "inner", out_dir,
                  data_handler_kwargs=cfg["inner_model"]["dataset"]["kwargs"]
                                          ["handler"]["kwargs"])

    # ---- 导出三层测试期 pred 信号 ----
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_outer.rename("pred_outer").to_csv(pred_dir / "pred_outer.csv")
    pred_middle.rename("pred_middle").to_csv(pred_dir / "pred_middle.csv")
    pred_inner.rename("pred_inner").to_csv(pred_dir / "pred_inner.csv")
    print(f"[predictions] 三层测试期信号 -> {pred_dir}")

    # ================================================================ #
    # 4. 回测：用三层信号 + MultiLayerStrategy + 嵌套 TWAP
    # ================================================================ #
    print("\n" + "=" * 60)
    print("[回测] MultiLayerStrategy + 嵌套 TWAP 执行")
    print("=" * 60)

    with R.start(experiment_name=exp_name):
        R.log_params(
            stock_id=stock_id,
            outer_upper=cfg["strategy"]["kwargs"]["outer_upper"],
            outer_lower=cfg["strategy"]["kwargs"]["outer_lower"],
        )

        # 构建策略实例（注入三层预测信号）
        strategy_cfg = cfg["strategy"]
        strategy_instance = init_instance_by_config({
            "class": strategy_cfg["class"],
            "module_path": strategy_cfg["module_path"],
            "kwargs": {
                **strategy_cfg["kwargs"],
                "stock_id": stock_id,
                "pred_outer": pred_outer,
                "pred_middle": pred_middle,
                "pred_inner": pred_inner,
                "level_infra": None,
                "common_infra": None,
                "trade_exchange": None,
            },
        })

        # 构建执行器实例
        executor_instance = init_instance_by_config(cfg["executor"])

        # 执行回测
        from qlib.backtest import backtest as qlib_backtest

        portfolio_metric_dict, indicator_dict = qlib_backtest(
            start_time=backtest_cfg["start_time"],
            end_time=backtest_cfg["end_time"],
            strategy=strategy_instance,
            executor=executor_instance,
            account=backtest_cfg["account"],
            exchange_kwargs=backtest_cfg["exchange_kwargs"],
            benchmark=backtest_cfg.get("benchmark"),
        )

        # 保存结果
        recorder = R.get_recorder()
        artifact_objects = {}
        for freq, (report, positions) in portfolio_metric_dict.items():
            artifact_objects[f"report_{freq}.pkl"] = report
            artifact_objects[f"positions_{freq}.pkl"] = positions
        for freq, (ind_df, ind_obj) in indicator_dict.items():
            artifact_objects[f"indicators_{freq}.pkl"] = ind_df
        R.save_objects(**artifact_objects)

        # 打印关键指标
        for freq, (report, _) in portfolio_metric_dict.items():
            if report is not None and not report.empty:
                print(f"\n--- 回测报告摘要 (freq={freq}) ---")
                for col in ["return", "cost", "turnover"]:
                    if col in report.columns:
                        cum_val = report[col].sum() if col != "return" else (
                            (1 + report[col]).prod() - 1
                        )
                        print(f"  累计 {col}: {cum_val:.4f}")
                print(f"  交易天数: {len(report)}")

    print(f"\n[OK] 实验完成。可用 `mlflow ui` 查看 {exp_name} 结果。")


if __name__ == "__main__":
    fire.Fire({"run_once": run_once})
