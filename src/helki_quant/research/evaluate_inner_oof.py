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
INTRADAY = MULTI_LAYER.parent / "intraday_t"
for path in (MULTI_LAYER, INTRADAY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


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


def frame_metrics(frame: pd.DataFrame, positive_threshold: float = 0.0) -> dict:
    if frame.empty:
        return {"rows": 0}
    positive = frame["label"] > positive_threshold
    daily_ic = frame.groupby(level="datetime").apply(
        lambda x: x["score"].corr(x["label"], method="spearman") if len(x) >= 5 else np.nan
    )
    return {
        "rows": int(len(frame)),
        "date_start": str(frame.index.get_level_values("datetime").min()),
        "date_end": str(frame.index.get_level_values("datetime").max()),
        "score_mean": float(frame["score"].mean()),
        "score_std": float(frame["score"].std()),
        "label_mean": float(frame["label"].mean()),
        "positive_threshold": float(positive_threshold),
        "label_positive_ratio": float(positive.mean()),
        "spearman": float(frame["score"].corr(frame["label"], method="spearman")),
        "pearson": float(frame["score"].corr(frame["label"], method="pearson")),
        "auc_positive": auc_score(
            (frame["label"].to_numpy() > positive_threshold).astype(int),
            frame["score"].to_numpy(),
        ),
        "daily_ic_mean": float(daily_ic.mean()) if len(daily_ic.dropna()) else None,
        "daily_ic_ir": float(daily_ic.mean() / (daily_ic.std() + 1e-12))
        if len(daily_ic.dropna())
        else None,
        "buy_signal_ratio_055": float((frame["score"] >= 0.55).mean()),
        "sell_signal_ratio_045": float((frame["score"] <= 0.45).mean()),
    }


def load_fold_frame(config_path: Path, prediction_path: Path) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    qlib.init(**cfg.get("qlib_init_inner", cfg["qlib_init"]))
    dataset = init_instance_by_config(cfg["inner_model"]["dataset"])
    labels = dataset.prepare("test", col_set="label")
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    labels = labels.rename("label")
    pred = pd.read_csv(prediction_path, index_col=[0, 1], parse_dates=[0]).iloc[:, 0]
    pred.index.names = labels.index.names
    pred = pred.rename("score")
    return pd.concat([pred, labels], axis=1, join="inner").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()


def evaluate(
    config_dir: Path,
    prediction_dir: Path,
    output_path: Path,
    variant: str,
    positive_threshold: float,
    folds: list[int] | None = None,
) -> dict:
    frames = []
    fold_rows = []
    target_folds = folds or list(range(1, 7))
    for fold in target_folds:
        fold_dir = config_dir / f"fold_{fold:02d}"
        if variant.endswith("_simple") or variant == "inner_exec_simple":
            config_path = fold_dir / "simple.yaml"
        else:
            config_path = fold_dir / "de2_srfs_es.yaml"
            if not config_path.exists():
                config_path = fold_dir / "densemble.yaml"
        prediction_path = prediction_dir / f"fold_{fold:02d}.csv"
        frame = load_fold_frame(config_path, prediction_path)
        frame["fold"] = fold
        frames.append(frame)
        fold_metric = frame_metrics(frame.drop(columns=["fold"]), positive_threshold)
        fold_metric["fold"] = fold
        fold_rows.append(fold_metric)
        print(
            f"[inner oof eval] fold={fold} rows={fold_metric['rows']} "
            f"auc={fold_metric.get('auc_positive')} spearman={fold_metric.get('spearman')}",
            flush=True,
        )
    all_frame = pd.concat(frames).sort_index()
    overall = frame_metrics(all_frame.drop(columns=["fold"]), positive_threshold)
    report = {
        "status": "inner_oof_evaluated",
        "variant": variant,
        "positive_threshold": float(positive_threshold),
        "config_dir": str(config_dir),
        "prediction_dir": str(prediction_dir),
        "evaluated_folds": target_folds,
        "overall": overall,
        "folds": fold_rows,
        "worst_fold_auc": min(
            row["auc_positive"] for row in fold_rows if row.get("auc_positive") is not None
        ),
        "worst_fold_spearman": min(row["spearman"] for row in fold_rows),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_folds(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    folds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            folds.extend(range(int(start), int(end) + 1))
        else:
            folds.append(int(part))
    folds = sorted(set(folds))
    bad = [fold for fold in folds if fold < 1 or fold > 6]
    if bad:
        raise ValueError(f"folds must be in 1..6, got {bad}")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--config-dir", default=str(HERE / "outputs" / "inner_exec_fold_configs")
    )
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--positive-threshold", type=float, default=0.0)
    parser.add_argument("--folds", default=None, help="Optional comma/range list, e.g. 1 or 1,3,5 or 1-3.")
    args = parser.parse_args()
    prediction_dir = (
        Path(args.prediction_dir).resolve()
        if args.prediction_dir
        else (HERE / "outputs" / "oof" / args.variant / "inner").resolve()
    )
    output = (
        Path(args.output).resolve()
        if args.output
        else (HERE / "outputs" / f"{args.variant}_oof_evaluation.json").resolve()
    )
    report = evaluate(
        Path(args.config_dir).resolve(),
        prediction_dir,
        output,
        args.variant,
        args.positive_threshold,
        parse_folds(args.folds),
    )
    overall = report["overall"]
    print(
        f"[inner oof eval] variant={args.variant} rows={overall['rows']} "
        f"auc={overall['auc_positive']} spearman={overall['spearman']} "
        f"worst_auc={report['worst_fold_auc']}"
    )


if __name__ == "__main__":
    main()
