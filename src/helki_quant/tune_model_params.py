from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from ruamel.yaml import YAML
from sklearn.metrics import f1_score, log_loss, roc_auc_score

import qlib
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config


DIRNAME = Path(__file__).resolve().parent
INTRADAY_DIR = DIRNAME.parent / "intraday_t"
for path in (DIRNAME, INTRADAY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from catboost_densemble import CatBoostDEnsemble


LAYER_KEYS = {
    "outer": "outer_model",
    "middle": "middle_model",
    "inner": "inner_model",
}

# A compact, deliberately conservative search space. DoubleEnsemble multiplies
# model capacity, so very deep/high-rate CatBoost candidates are intentionally
# excluded.
CANDIDATES = [
    {
        "name": "d4_slow_reg",
        "learning_rate": 0.030,
        "max_depth": 4,
        "max_leaves": 16,
        "l2_leaf_reg": 12.0,
        "random_strength": 1.5,
        "subsample": 0.75,
        "iterations": 240,
    },
    {
        "name": "d5_slow_reg",
        "learning_rate": 0.035,
        "max_depth": 5,
        "max_leaves": 32,
        "l2_leaf_reg": 8.0,
        "random_strength": 1.0,
        "subsample": 0.80,
        "iterations": 220,
    },
    {
        "name": "d5_fast_reg",
        "learning_rate": 0.050,
        "max_depth": 5,
        "max_leaves": 32,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.75,
        "subsample": 0.85,
        "iterations": 180,
    },
    {
        "name": "d6_slow_strong_reg",
        "learning_rate": 0.025,
        "max_depth": 6,
        "max_leaves": 64,
        "l2_leaf_reg": 14.0,
        "random_strength": 1.5,
        "subsample": 0.80,
        "iterations": 260,
    },
    {
        "name": "d6_baseline_like",
        "learning_rate": 0.050,
        "max_depth": 6,
        "max_leaves": 64,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "subsample": 0.85,
        "iterations": 220,
    },
]


def _load_yaml(path: Path) -> dict:
    return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))


def _resolve_provider_uri(config_path: Path, qlib_kwargs: dict) -> dict:
    qlib_kwargs = copy.deepcopy(qlib_kwargs)
    provider_uri = qlib_kwargs.get("provider_uri")
    if isinstance(provider_uri, dict):
        for freq, value in list(provider_uri.items()):
            if isinstance(value, str) and not os.path.isabs(value):
                provider_uri[freq] = str((config_path.parent / value).resolve())
    return qlib_kwargs


def _flatten_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [col[-1] if isinstance(col, tuple) else str(col) for col in out.columns]
    return out


def _sample_train(
    x: pd.DataFrame,
    y_cls: np.ndarray,
    max_rows: int,
    target: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(x) <= max_rows:
        return x, y_cls
    rng = np.random.default_rng(42)
    selected: list[int] = []
    per_class = max_rows // len(np.unique(y_cls))
    for cls in np.unique(y_cls):
        positions = np.flatnonzero(y_cls == cls)
        take = min(len(positions), per_class)
        selected.extend(rng.choice(positions, size=take, replace=False).tolist())

    # Always retain target-stock history because the deployed strategy trades it.
    try:
        target_pos = np.flatnonzero(
            x.index.get_level_values("instrument").astype(str).values == target
        )
        selected.extend(target_pos.tolist())
    except Exception:
        pass
    selected = sorted(set(selected))
    if len(selected) > max_rows:
        selected = sorted(rng.choice(selected, size=max_rows, replace=False).tolist())
    return x.iloc[selected], y_cls[selected]


def _daily_ic(signal: pd.Series, label: pd.Series) -> pd.Series:
    pair = pd.concat([signal.rename("signal"), label.rename("label")], axis=1).dropna()
    return pair.groupby(level="datetime", sort=True).apply(
        lambda part: part["signal"].corr(part["label"], method="spearman")
        if part["signal"].nunique() > 2 and part["label"].nunique() > 2
        else np.nan
    ).dropna()


def _target_ic(signal: pd.Series, label: pd.Series, target: str) -> float:
    try:
        mask = signal.index.get_level_values("instrument").astype(str) == target
    except Exception:
        return np.nan
    pair = pd.concat(
        [signal.loc[mask].rename("signal"), label.loc[mask].rename("label")],
        axis=1,
    ).dropna()
    if len(pair) < 30:
        return np.nan
    return float(pair["signal"].corr(pair["label"], method="spearman"))


def _monthly_ic_std(signal: pd.Series, label: pd.Series) -> float:
    daily = _daily_ic(signal, label)
    if daily.empty:
        return np.nan
    monthly = daily.groupby(daily.index.to_period("M")).mean()
    return float(monthly.std()) if len(monthly) > 1 else 0.0


def _metrics(
    layer: str,
    proba: np.ndarray,
    y_cls: np.ndarray,
    y_raw: pd.Series,
    target: str,
) -> dict:
    if proba.shape[1] == 2:
        signal_arr = proba[:, 1]
    else:
        signal_arr = proba[:, -1] - proba[:, 0]
    signal = pd.Series(signal_arr, index=y_raw.index)
    daily = _daily_ic(signal, y_raw)
    ic_mean = float(daily.mean())
    ic_std = float(daily.std())
    icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
    target_ic = _target_ic(signal, y_raw, target)
    monthly_std = _monthly_ic_std(signal, y_raw)
    ll = float(log_loss(y_cls, proba, labels=np.arange(proba.shape[1])))
    pred_cls = np.argmax(proba, axis=1)
    f1 = float(f1_score(y_cls, pred_cls, average="macro"))
    auc = (
        float(roc_auc_score(y_cls, proba[:, 1]))
        if layer == "inner" and len(np.unique(y_cls)) == 2
        else np.nan
    )

    # Validation-only robust score. IC and its temporal stability dominate;
    # classification metrics keep probability estimates well behaved.
    score = 2.5 * ic_mean + 0.25 * icir - 0.30 * monthly_std
    score += 0.20 * f1 - 0.10 * ll
    if np.isfinite(target_ic):
        score += 0.10 * target_ic
    if np.isfinite(auc):
        score += 0.10 * (auc - 0.5)
    return {
        "score": float(score),
        "daily_ic_mean": ic_mean,
        "daily_ic_std": ic_std,
        "daily_icir": float(icir),
        "monthly_ic_std": float(monthly_std),
        "target_rank_ic": float(target_ic),
        "macro_f1": f1,
        "logloss": ll,
        "auc": float(auc),
        "signal_mean": float(signal.mean()),
        "signal_std": float(signal.std()),
    }


def tune_layer(
    layer: str,
    config_path: Path,
    output_dir: Path,
    max_train_rows: int,
    target: str,
) -> None:
    cfg = _load_yaml(config_path)
    layer_cfg = cfg[LAYER_KEYS[layer]]
    qlib_key = "qlib_init_inner" if layer == "inner" else "qlib_init"
    qlib.init(**_resolve_provider_uri(config_path, cfg.get(qlib_key, cfg["qlib_init"])))
    dataset = init_instance_by_config(layer_cfg["dataset"])
    train, valid, test = dataset.prepare(
        ["train", "valid", "test"],
        col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_L,
    )
    x_train = _flatten_feature_columns(train["feature"])
    x_valid = _flatten_feature_columns(valid["feature"])
    x_test = _flatten_feature_columns(test["feature"])
    y_train_raw = train["label"].iloc[:, 0].astype(float)
    y_valid_raw = valid["label"].iloc[:, 0].astype(float)
    y_test_raw = test["label"].iloc[:, 0].astype(float)

    model_kwargs = copy.deepcopy(layer_cfg["model"]["kwargs"])
    helper = CatBoostDEnsemble(
        loss=model_kwargs["loss"],
        thresholds=model_kwargs.get("thresholds"),
        adaptive_thresholds=model_kwargs.get("adaptive_thresholds"),
        binary_threshold=model_kwargs.get("binary_threshold", 0.0),
        num_models=1,
        enable_sr=False,
        enable_fs=False,
        bins_fs=1,
        sample_ratios=[1.0],
        sub_weights=[1.0],
    )
    if helper._is_multiclass and helper.adaptive_thresholds == "std_ratio":
        helper._compute_instrument_thresholds(y_train_raw)
    y_train_cls = helper._bin_labels(y_train_raw.values, y_index=y_train_raw.index)
    y_valid_cls = helper._bin_labels(y_valid_raw.values, y_index=y_valid_raw.index)
    y_test_cls = helper._bin_labels(y_test_raw.values, y_index=y_test_raw.index)
    x_fit, y_fit = _sample_train(x_train, y_train_cls, max_train_rows, target)

    print(
        f"[{layer}] train={x_train.shape} fit_sample={x_fit.shape} "
        f"valid={x_valid.shape} test={x_test.shape}",
        flush=True,
    )
    results = []
    for candidate in CANDIDATES:
        params = {
            "loss_function": model_kwargs["loss"],
            "task_type": "CPU",
            "thread_count": int(model_kwargs.get("thread_count", 8)),
            "grow_policy": "Lossguide",
            "bootstrap_type": "Bernoulli",
            "allow_writing_files": False,
            "verbose": False,
            "random_seed": 42,
            **{k: v for k, v in candidate.items() if k != "name"},
        }
        print(f"[{layer}] fitting {candidate['name']} ...", flush=True)
        t0 = time.time()
        model = CatBoost(params)
        model.fit(
            Pool(x_fit, label=y_fit),
            eval_set=Pool(x_valid, label=y_valid_cls),
            use_best_model=True,
        )
        valid_proba = np.asarray(model.predict(x_valid, prediction_type="Probability"))
        valid_metrics = _metrics(layer, valid_proba, y_valid_cls, y_valid_raw, target)
        test_proba = np.asarray(model.predict(x_test, prediction_type="Probability"))
        test_metrics = _metrics(layer, test_proba, y_test_cls, y_test_raw, target)
        result = {
            "name": candidate["name"],
            "params": candidate,
            "best_iteration": int(model.get_best_iteration()),
            "tree_count": int(model.tree_count_),
            "elapsed_seconds": float(time.time() - t0),
            "valid": valid_metrics,
            # Test is reported after selection and never enters candidate score.
            "test": test_metrics,
        }
        results.append(result)
        print(
            f"[{layer}] {candidate['name']} valid_score={valid_metrics['score']:+.4f} "
            f"valid_IC={valid_metrics['daily_ic_mean']:+.4f} "
            f"valid_ICIR={valid_metrics['daily_icir']:+.3f} "
            f"test_IC={test_metrics['daily_ic_mean']:+.4f}",
            flush=True,
        )

    results.sort(key=lambda item: item["valid"]["score"], reverse=True)
    summary = {
        "layer": layer,
        "config": str(config_path),
        "selection_data": "train+valid only",
        "test_used_for_selection": False,
        "max_train_rows": max_train_rows,
        "best": results[0],
        "candidates": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"model_tuning_{layer}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{layer}] best={results[0]['name']} -> {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=[*LAYER_KEYS, "all"], default="all")
    parser.add_argument("--config", default=str(DIRNAME / "config_densemble_v2.yaml"))
    parser.add_argument("--output_dir", default=str(DIRNAME / "tuning_results"))
    parser.add_argument("--max_train_rows", type=int, default=300000)
    parser.add_argument("--target", default="SZ301536")
    args = parser.parse_args()

    layers = list(LAYER_KEYS) if args.layer == "all" else [args.layer]
    for layer in layers:
        tune_layer(
            layer,
            config_path=Path(args.config).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            max_train_rows=args.max_train_rows,
            target=args.target,
        )


if __name__ == "__main__":
    main()
