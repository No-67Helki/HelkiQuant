from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

import qlib
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config


DIRNAME = Path(__file__).resolve().parent
INTRADAY_DIR = DIRNAME.parent / "intraday_t"
for path in (DIRNAME, INTRADAY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

LAYER_KEYS = {
    "outer": "outer_model",
    "middle": "middle_model",
    "inner": "inner_model",
}
SEGMENTS = ("train", "valid", "test")


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


def _drop_cross_sectional_processors(dataset_cfg: dict) -> dict:
    """Evaluate factors and labels in the same form available to single-stock live inference."""
    dataset_cfg = copy.deepcopy(dataset_cfg)
    handler_kwargs = dataset_cfg["kwargs"]["handler"]["kwargs"]
    for key in ("infer_processors", "learn_processors"):
        processors = handler_kwargs.get(key, [])
        handler_kwargs[key] = [
            proc for proc in processors if proc.get("class") != "CSZScoreNorm"
        ]
    return dataset_cfg


def _flatten_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [col[-1] if isinstance(col, tuple) else str(col) for col in out.columns]
    return out


def _daily_spearman(feature: pd.Series, label: pd.Series) -> pd.Series:
    pair = pd.concat([feature.rename("x"), label.rename("y")], axis=1).dropna()
    if pair.empty:
        return pd.Series(dtype=float)
    return pair.groupby(level="datetime", sort=True).apply(
        lambda part: part["x"].corr(part["y"], method="spearman")
        if part["x"].nunique() > 2 and part["y"].nunique() > 2
        else np.nan
    ).dropna()


def _safe_corr(feature: pd.Series, label: pd.Series) -> float:
    pair = pd.concat([feature.rename("x"), label.rename("y")], axis=1).dropna()
    if len(pair) < 20 or pair["x"].nunique() < 3 or pair["y"].nunique() < 3:
        return np.nan
    return float(pair["x"].corr(pair["y"], method="spearman"))


def _segment_stats(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for feature in x.columns:
        daily_ic = _daily_spearman(x[feature], y)
        mean_ic = float(daily_ic.mean()) if len(daily_ic) else np.nan
        std_ic = float(daily_ic.std()) if len(daily_ic) > 1 else np.nan
        rows.append(
            {
                "feature": feature,
                "cs_ic_mean": mean_ic,
                "cs_ic_std": std_ic,
                "cs_icir": mean_ic / std_ic if np.isfinite(std_ic) and std_ic > 1e-12 else np.nan,
                "cs_positive_ratio": float((daily_ic > 0).mean()) if len(daily_ic) else np.nan,
                "pooled_rank_ic": _safe_corr(x[feature], y),
                "mean": float(x[feature].mean()),
                "std": float(x[feature].std()),
                "nan_ratio": float(x[feature].isna().mean()),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def _add_stability_scores(stats: pd.DataFrame) -> pd.DataFrame:
    out = stats.copy()
    train_ic = out["train_cs_ic_mean"]
    valid_ic = out["valid_cs_ic_mean"]
    same_sign = np.sign(train_ic) == np.sign(valid_ic)
    out["train_valid_same_sign"] = same_sign
    out["stable_ic"] = np.where(
        same_sign,
        np.minimum(train_ic.abs(), valid_ic.abs()),
        0.0,
    )
    train_std = out["train_std"].replace(0.0, np.nan)
    out["valid_mean_shift"] = (out["valid_mean"] - out["train_mean"]).abs() / train_std
    if "test_mean" in out:
        out["test_mean_shift"] = (out["test_mean"] - out["train_mean"]).abs() / train_std
    out["stability_score"] = (
        out["stable_ic"]
        * np.sqrt(out["train_cs_positive_ratio"].fillna(0.5).clip(0.0, 1.0))
        / (1.0 + out["valid_mean_shift"].fillna(10.0).clip(lower=0.0))
    )
    return out.sort_values("stability_score", ascending=False)


def _make_stable_whitelist(
    x_train: pd.DataFrame,
    scores: pd.DataFrame,
    corr_threshold: float,
    min_features: int,
    max_features: int,
    corr_sample: int,
) -> list[str]:
    candidates = scores[
        scores["train_valid_same_sign"]
        & (scores["stable_ic"] > 0)
        & (scores["valid_mean_shift"] < 3.0)
        & (scores["train_nan_ratio"] < 0.25)
    ].index.tolist()
    if len(candidates) < min_features:
        candidates = scores.head(max(min_features, len(candidates))).index.tolist()

    candidates = candidates[: max_features * 3]
    sample = x_train.loc[:, candidates].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(sample) > corr_sample:
        sample = sample.sample(corr_sample, random_state=42)
    corr = sample.corr(method="spearman").abs()

    kept: list[str] = []
    for feature in candidates:
        if all(corr.loc[feature, other] <= corr_threshold for other in kept):
            kept.append(feature)
        if len(kept) >= max_features:
            break
    if len(kept) < min_features:
        for feature in scores.index:
            if feature not in kept:
                kept.append(feature)
            if len(kept) >= min_features:
                break
    return kept


def evaluate_layer(
    layer: str,
    config_path: Path,
    output_dir: Path,
    corr_threshold: float,
    min_features: int,
    max_features: int,
    corr_sample: int,
    include_test_report: bool = True,
    feature_blacklist: set[str] | None = None,
) -> None:
    cfg = _load_yaml(config_path)
    qlib_key = "qlib_init_inner" if layer == "inner" else "qlib_init"
    qlib.init(**_resolve_provider_uri(config_path, cfg.get(qlib_key, cfg["qlib_init"])))

    dataset_cfg = _drop_cross_sectional_processors(cfg[LAYER_KEYS[layer]]["dataset"])
    dataset = init_instance_by_config(dataset_cfg)
    segment_data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    segment_stats: dict[str, pd.DataFrame] = {}

    report_segments = SEGMENTS if include_test_report else ("train", "valid")
    for segment in report_segments:
        frame = dataset.prepare(
            segment,
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x = _flatten_feature_columns(frame["feature"])
        y = frame["label"].iloc[:, 0].astype(float)
        segment_data[segment] = (x, y)
        segment_stats[segment] = _segment_stats(x, y).add_prefix(f"{segment}_")
        print(
            f"[{layer}] {segment}: rows={len(x)} features={x.shape[1]} "
            f"label_mean={y.mean():+.6f} label_std={y.std():.6f}",
            flush=True,
        )

    stats = pd.concat([segment_stats[segment] for segment in report_segments], axis=1)
    feature_blacklist = set(feature_blacklist or set())
    if feature_blacklist:
        stats = stats.drop(index=[feature for feature in feature_blacklist if feature in stats.index])
    stats = _add_stability_scores(stats)
    kept = _make_stable_whitelist(
        segment_data["train"][0],
        stats,
        corr_threshold=corr_threshold,
        min_features=min_features,
        max_features=max_features,
        corr_sample=corr_sample,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / f"factor_report_{layer}.csv"
    whitelist_path = output_dir / f"feature_whitelist_{layer}_v2.json"
    stats.to_csv(stats_path)
    whitelist_path.write_text(
        json.dumps(
            {
                "layer": layer,
                "selection_segments": ["train", "valid"],
                "evaluation_segment": "test" if include_test_report else None,
                "test_metrics_read_during_selection": include_test_report,
                "cross_sectional_processors_removed": True,
                "corr_threshold": corr_threshold,
                "feature_blacklist": sorted(feature_blacklist),
                "n_total": int(stats.shape[0]),
                "n_kept": len(kept),
                "kept": kept,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{layer}] top stable factors:\n{stats.head(20).to_string()}", flush=True)
    print(f"[{layer}] report -> {stats_path}", flush=True)
    print(f"[{layer}] whitelist ({len(kept)}) -> {whitelist_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=[*LAYER_KEYS, "all"], default="all")
    parser.add_argument("--config", default=str(DIRNAME / "config_densemble_v2.yaml"))
    parser.add_argument("--output_dir", default=str(DIRNAME / "factor_reports"))
    parser.add_argument("--corr_threshold", type=float, default=0.90)
    parser.add_argument("--min_features", type=int, default=30)
    parser.add_argument("--max_features", type=int, default=80)
    parser.add_argument("--corr_sample", type=int, default=100000)
    parser.add_argument(
        "--feature_blacklist",
        nargs="*",
        default=[],
        help="Feature names to exclude before stable factor selection.",
    )
    args = parser.parse_args()

    layers = list(LAYER_KEYS) if args.layer == "all" else [args.layer]
    for layer in layers:
        evaluate_layer(
            layer=layer,
            config_path=Path(args.config).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            corr_threshold=args.corr_threshold,
            min_features=args.min_features,
            max_features=args.max_features,
            corr_sample=args.corr_sample,
            feature_blacklist=set(args.feature_blacklist),
        )


if __name__ == "__main__":
    main()
