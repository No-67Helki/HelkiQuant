from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

import qlib
from qlib.utils import init_instance_by_config


HERE = Path(__file__).resolve().parent
MULTI_LAYER = HERE.parent
if str(MULTI_LAYER) not in sys.path:
    sys.path.insert(0, str(MULTI_LAYER))


def load_yaml(path: Path) -> dict:
    return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        rank_sum = np.bincount(inv, weights=ranks)
        ranks = (rank_sum / counts)[inv]
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def safe_logloss(y_true: pd.Series, score: pd.Series) -> float | None:
    sample = pd.concat([y_true.rename("y"), score.rename("score")], axis=1).dropna()
    if sample.empty:
        return None
    p = sample["score"].clip(1e-6, 1 - 1e-6)
    y = sample["y"].astype(float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def frame_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0}
    y = frame["label"].astype(int)
    score = frame["score"].astype(float)
    return {
        "rows": int(len(frame)),
        "date_start": str(frame.index.min().date()),
        "date_end": str(frame.index.max().date()),
        "label_positive_ratio": float(y.mean()),
        "score_mean": float(score.mean()),
        "score_std": float(score.std()),
        "auc": auc_score(y.to_numpy(), score.to_numpy()),
        "spearman": float(score.corr(y, method="spearman")),
        "pearson": float(score.corr(y, method="pearson")),
        "logloss": safe_logloss(y, score),
        "risk_ge_055": float((score >= 0.55).mean()),
        "risk_ge_060": float((score >= 0.60).mean()),
        "risk_ge_070": float((score >= 0.70).mean()),
    }


def quantile_table(frame: pd.DataFrame, bins: int = 5) -> list[dict]:
    sample = frame.dropna(subset=["score", "label"]).copy()
    if sample.empty:
        return []
    sample["bucket"] = pd.qcut(sample["score"].rank(method="first"), bins, labels=False)
    rows = []
    for bucket, part in sample.groupby("bucket", sort=True):
        rows.append(
            {
                "bucket": int(bucket),
                "days": int(len(part)),
                "score_mean": float(part["score"].mean()),
                "label_positive_ratio": float(part["label"].mean()),
            }
        )
    return rows


def load_fold_frame(config_path: Path, prediction_path: Path) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    qlib.init(**cfg["qlib_init"])
    dataset = init_instance_by_config(cfg["outer_model"]["dataset"])
    labels = dataset.prepare("test", col_set="label")
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    labels = labels.rename("label")
    pred = pd.read_csv(prediction_path, index_col=[0, 1], parse_dates=[0]).iloc[:, 0]
    pred.index.names = labels.index.names
    pred = pred.rename("score")
    frame = pd.concat([pred, labels], axis=1, join="inner").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    daily = frame.groupby(level="datetime").agg({"score": "median", "label": "median"})
    daily["label"] = (daily["label"] > 0.5).astype(int)
    return daily


def load_fold_frame_from_daily_labels(
    prediction_path: Path,
    daily_labels: pd.Series,
) -> pd.DataFrame:
    pred = pd.read_csv(prediction_path, index_col=[0, 1], parse_dates=[0]).iloc[:, 0]
    pred.index.names = ["datetime", "instrument"]
    daily_score = pred.groupby(level="datetime").median().rename("score")
    labels = daily_labels.copy()
    labels.index = pd.to_datetime(labels.index)
    labels.index.name = "datetime"
    frame = pd.concat([daily_score, labels.rename("label")], axis=1, join="inner")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame["label"] = (frame["label"].astype(float) > 0.5).astype(int)
    return frame


def discover_folds(
    config_dir: Path,
    prediction_dir: Path,
    config_name: str,
) -> list[int]:
    config_folds = {
        int(path.parent.name.removeprefix("fold_"))
        for path in config_dir.glob(f"fold_*/{config_name}")
        if path.parent.name.removeprefix("fold_").isdigit()
    }
    prediction_folds = {
        int(path.stem.removeprefix("fold_"))
        for path in prediction_dir.glob("fold_*.csv")
        if path.stem.removeprefix("fold_").isdigit()
    }
    folds = sorted(config_folds & prediction_folds)
    if not folds:
        raise FileNotFoundError(
            f"no matching folds under config={config_dir} prediction={prediction_dir}"
        )
    return folds


def evaluate(
    config_dir: Path,
    prediction_dir: Path,
    output_path: Path,
    variant: str,
    *,
    folds: list[int] | None = None,
    config_name: str = "simple.yaml",
    daily_labels: pd.Series | None = None,
    daily_labels_source: str | None = None,
) -> dict:
    fold_numbers = folds or discover_folds(config_dir, prediction_dir, config_name)
    frames = []
    fold_rows = []
    for fold in fold_numbers:
        config_path = config_dir / f"fold_{fold:02d}" / config_name
        prediction_path = prediction_dir / f"fold_{fold:02d}.csv"
        if not config_path.exists() or not prediction_path.exists():
            raise FileNotFoundError(
                f"missing fold input config={config_path} prediction={prediction_path}"
            )
        frame = (
            load_fold_frame_from_daily_labels(prediction_path, daily_labels)
            if daily_labels is not None
            else load_fold_frame(config_path, prediction_path)
        )
        frame["fold"] = fold
        frames.append(frame)
        metrics = frame_metrics(frame.drop(columns=["fold"]))
        metrics["fold"] = fold
        metrics["quantiles"] = quantile_table(frame.drop(columns=["fold"]))
        fold_rows.append(metrics)
        print(
            f"[outer regime eval] fold={fold} days={metrics['rows']} "
            f"auc={metrics.get('auc')} spearman={metrics.get('spearman')}",
            flush=True,
        )
    all_frame = pd.concat(frames).sort_index()
    overall_frame = all_frame.drop(columns=["fold"])
    overall = frame_metrics(overall_frame)
    report = {
        "status": "outer_regime_oof_evaluated_research_only",
        "variant": variant,
        "config_dir": str(config_dir),
        "prediction_dir": str(prediction_dir),
        "config_name": config_name,
        "fold_numbers": fold_numbers,
        "fold_count": len(fold_numbers),
        "daily_labels_source": daily_labels_source,
        "overall": overall,
        "folds": fold_rows,
        "overall_quantiles": quantile_table(overall_frame),
        "worst_fold_auc": min(
            row["auc"] for row in fold_rows if row.get("auc") is not None
        ),
        "worst_fold_spearman": min(row["spearman"] for row in fold_rows),
        "deployment_allowed": False,
        "interpretation": (
            "This evaluates the outer risk model as a daily adverse-regime "
            "probability. It is not yet an enabled portfolio overlay."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-name", default="simple.yaml")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--daily-labels-csv", default="")
    parser.add_argument("--daily-label-column", default="broad_adverse_loss5_20d")
    args = parser.parse_args()
    prediction_dir = (
        Path(args.prediction_dir).resolve()
        if args.prediction_dir
        else (HERE / "outputs" / "oof" / args.variant / "outer").resolve()
    )
    daily_labels = None
    daily_labels_source = None
    if args.daily_labels_csv:
        daily_labels_path = Path(args.daily_labels_csv).resolve()
        daily_frame = pd.read_csv(daily_labels_path, parse_dates=["datetime"])
        if args.daily_label_column not in daily_frame.columns:
            raise KeyError(
                f"missing daily label column {args.daily_label_column}: {daily_labels_path}"
            )
        daily_labels = daily_frame.set_index("datetime")[args.daily_label_column]
        daily_labels_source = str(daily_labels_path)
    report = evaluate(
        Path(args.config_dir).resolve(),
        prediction_dir,
        Path(args.output).resolve(),
        args.variant,
        folds=args.folds,
        config_name=args.config_name,
        daily_labels=daily_labels,
        daily_labels_source=daily_labels_source,
    )
    print(
        f"[outer regime eval] variant={args.variant} "
        f"days={report['overall']['rows']} auc={report['overall']['auc']} "
        f"spearman={report['overall']['spearman']} worst_auc={report['worst_fold_auc']}"
    )


if __name__ == "__main__":
    main()
