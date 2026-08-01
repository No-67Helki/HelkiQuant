from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_outer_regime_history_provider import build_daily_labels
from evaluate_outer_regime_oof import auc_score, safe_logloss


def evaluate(
    prediction_path: Path,
    source_provider: Path,
    raw_daily_dir: Path,
    output_path: Path,
    *,
    history_start: str,
    test_start: str,
    test_end: str,
    horizon: int,
    threshold: float,
    min_listing_days: int,
    min_avg_amount: float,
) -> dict:
    prediction = pd.read_csv(prediction_path, parse_dates=["datetime"])
    prediction["datetime"] = prediction["datetime"].dt.normalize()
    daily_spread = prediction.groupby("datetime")["outer"].agg(lambda values: values.max() - values.min())
    max_daily_spread = float(daily_spread.max()) if not daily_spread.empty else 0.0
    if max_daily_spread > 1e-12:
        raise ValueError(f"outer prediction is not constant within date: spread={max_daily_spread}")
    daily_prediction = prediction.groupby("datetime", as_index=False)["outer"].median()
    daily_prediction = daily_prediction.rename(columns={"outer": "score"})

    labels = build_daily_labels(
        source_provider,
        raw_daily_dir,
        history_start,
        test_end,
        horizons=[horizon],
        min_listing_days=min_listing_days,
        min_avg_amount=min_avg_amount,
    )
    label_col = f"broad_adverse_loss5_{horizon}d"
    forward_col = f"broad_fwd_{horizon}d"
    drawdown_col = f"broad_mdd_{horizon}d"
    wanted_cols = ["datetime", label_col, forward_col, drawdown_col, "eligible_count"]
    sample = daily_prediction.merge(labels[wanted_cols], on="datetime", how="left")
    start = pd.Timestamp(test_start).normalize()
    end = pd.Timestamp(test_end).normalize()
    sample = sample[sample["datetime"].between(start, end)].copy()
    sample["triggered"] = sample["score"] >= float(threshold)
    evaluable = sample.dropna(subset=[label_col]).copy()
    evaluable["label"] = evaluable[label_col].astype(int)

    if evaluable.empty:
        raise ValueError("no forward dates have a fully observed outer label")
    triggered = evaluable["triggered"]
    adverse = evaluable["label"] > 0
    true_positive = triggered & adverse
    false_positive = triggered & ~adverse
    false_negative = ~triggered & adverse
    true_negative = ~triggered & ~adverse
    precision = float(true_positive.sum() / triggered.sum()) if triggered.any() else None
    recall = float(true_positive.sum() / adverse.sum()) if adverse.any() else None

    date_rows = []
    for row in evaluable.itertuples(index=False):
        date_rows.append(
            {
                "datetime": str(pd.Timestamp(row.datetime).date()),
                "score": float(row.score),
                "triggered": bool(row.triggered),
                "label": int(row.label),
                "broad_forward_return": float(getattr(row, forward_col)),
                "broad_forward_max_drawdown": float(getattr(row, drawdown_col)),
                "eligible_count": int(row.eligible_count),
            }
        )

    report = {
        "status": "outer_regime_frozen_forward_evaluated",
        "prediction": str(prediction_path.resolve()),
        "source_provider": str(source_provider.resolve()),
        "raw_daily_dir": str(raw_daily_dir.resolve()),
        "window": {
            "history_start": history_start,
            "test_start": test_start,
            "test_end": test_end,
            "horizon": int(horizon),
            "fully_observed_label_end": str(evaluable["datetime"].max().date()),
        },
        "label": label_col,
        "threshold": float(threshold),
        "prediction_rows": int(len(prediction)),
        "prediction_dates": int(daily_prediction["datetime"].nunique()),
        "max_daily_prediction_spread": max_daily_spread,
        "evaluable_dates": int(len(evaluable)),
        "label_positive_dates": int(adverse.sum()),
        "label_positive_ratio": float(adverse.mean()),
        "score_mean": float(evaluable["score"].mean()),
        "score_std": float(evaluable["score"].std()),
        "auc": auc_score(evaluable["label"].to_numpy(), evaluable["score"].to_numpy()),
        "logloss": safe_logloss(evaluable["label"], evaluable["score"]),
        "confusion": {
            "true_positive": int(true_positive.sum()),
            "false_positive": int(false_positive.sum()),
            "false_negative": int(false_negative.sum()),
            "true_negative": int(true_negative.sum()),
            "precision": precision,
            "recall": recall,
        },
        "triggered_dates": [row for row in date_rows if row["triggered"]],
        "all_evaluable_dates": date_rows,
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--source-provider", required=True)
    parser.add_argument("--raw-daily-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-start", default="2022-01-04")
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-listing-days", type=int, default=250)
    parser.add_argument("--min-avg-amount", type=float, default=100_000_000.0)
    args = parser.parse_args()
    report = evaluate(
        Path(args.prediction).resolve(),
        Path(args.source_provider).resolve(),
        Path(args.raw_daily_dir).resolve(),
        Path(args.output).resolve(),
        history_start=args.history_start,
        test_start=args.test_start,
        test_end=args.test_end,
        horizon=args.horizon,
        threshold=args.threshold,
        min_listing_days=args.min_listing_days,
        min_avg_amount=args.min_avg_amount,
    )
    confusion = report["confusion"]
    print(
        "[outer forward] "
        f"dates={report['evaluable_dates']} positives={report['label_positive_dates']} "
        f"auc={report['auc']} trigger={confusion['true_positive'] + confusion['false_positive']} "
        f"tp={confusion['true_positive']} fp={confusion['false_positive']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
