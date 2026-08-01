from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def load_predictions(prediction_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(prediction_dir.glob("fold_*.csv")):
        frame = pd.read_csv(path, parse_dates=["datetime"])
        score_col = [col for col in frame.columns if col not in {"datetime", "instrument"}][0]
        frame = frame.rename(columns={score_col: "score"})
        frame["fold"] = int(path.stem.split("_")[-1])
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no fold predictions in {prediction_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    return out


def metrics(frame: pd.DataFrame, label_col: str, positive_threshold: float) -> dict:
    if frame.empty:
        return {"rows": 0}
    y = frame[label_col].to_numpy() > positive_threshold
    return {
        "rows": int(len(frame)),
        "dates": int(frame["datetime"].nunique()),
        "instruments": int(frame["instrument"].nunique()),
        "score_mean": float(frame["score"].mean()),
        "score_std": float(frame["score"].std()),
        "label_mean": float(frame[label_col].mean()),
        "positive_threshold": float(positive_threshold),
        "label_positive_ratio": float(y.mean()),
        "spearman": float(frame["score"].corr(frame[label_col], method="spearman")),
        "pearson": float(frame["score"].corr(frame[label_col], method="pearson")),
        "auc_positive": auc_score(y.astype(int), frame["score"].to_numpy()),
    }


def evaluate(
    prediction_dir: Path,
    held_context_path: Path,
    output_path: Path,
    label_col: str,
    positive_threshold: float,
) -> dict:
    pred = load_predictions(prediction_dir)
    context = pd.read_csv(held_context_path, parse_dates=["datetime"])
    context["datetime"] = context["datetime"].dt.normalize()
    context["instrument"] = context["instrument"].astype(str).str.upper()
    if label_col not in context.columns:
        raise KeyError(f"label_col not found in held context: {label_col}")
    keep = ["datetime", "instrument", label_col]
    frame = pred.merge(context[keep], on=["datetime", "instrument"], how="inner")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", label_col])
    fold_rows = []
    for fold, part in frame.groupby("fold", sort=True):
        row = metrics(part, label_col, positive_threshold)
        row["fold"] = int(fold)
        fold_rows.append(row)
    overall = metrics(frame, label_col, positive_threshold)
    report = {
        "status": "inner_held_context_oof_evaluated",
        "prediction_dir": str(prediction_dir.resolve()),
        "held_context_path": str(held_context_path.resolve()),
        "label_col": label_col,
        "positive_threshold": float(positive_threshold),
        "overall": overall,
        "folds": fold_rows,
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--held-context", required=True)
    parser.add_argument("--label-col", default="held_t0_sell_open_buy_close_hit")
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(
        Path(args.prediction_dir).resolve(),
        Path(args.held_context).resolve(),
        Path(args.output).resolve(),
        args.label_col,
        args.positive_threshold,
    )
    overall = report["overall"]
    print(
        "[held context eval] "
        f"rows={overall['rows']} auc={overall.get('auc_positive')} "
        f"spearman={overall.get('spearman')} pos={overall.get('label_positive_ratio')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
