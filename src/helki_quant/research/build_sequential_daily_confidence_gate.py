from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_sequential_daily_meta_gate import build_daily_meta_frame


def _fold_ranges(
    starts: list[pd.Timestamp], max_date: pd.Timestamp
) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    ordered = sorted(starts)
    ranges = []
    for index, start in enumerate(ordered, start=1):
        end = ordered[index] - pd.Timedelta(days=1) if index < len(ordered) else max_date
        ranges.append((index, start, end))
    return ranges


def build_confidence_gate(
    predictions_path: Path,
    output_predictions: Path,
    output_daily: Path,
    output_json: Path,
    *,
    ranking_col: str,
    realized_edge_col: str,
    evaluation_starts: list[pd.Timestamp],
    history_start: pd.Timestamp,
    daily_top_n: int,
    lookback_dates: int,
    minimum_history_dates: int,
    confidence_quantile: float,
) -> dict:
    if not 0.0 < confidence_quantile < 1.0:
        raise ValueError("confidence_quantile must be inside (0, 1)")
    predictions = pd.read_csv(predictions_path, parse_dates=["datetime", "trade_date"])
    predictions["datetime"] = predictions["datetime"].dt.normalize()
    predictions["trade_date"] = predictions["trade_date"].dt.normalize()
    predictions["instrument"] = predictions["instrument"].astype(str).str.upper()
    predictions["decision_time"] = (
        predictions["decision_time"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )
    daily = build_daily_meta_frame(
        predictions,
        ranking_col=ranking_col,
        realized_edge_col=realized_edge_col,
        daily_top_n=daily_top_n,
    )
    daily = daily[daily["trade_date"] >= history_start].copy()
    output_parts = []
    daily_parts = []
    folds = []
    for fold_id, test_start, test_end in _fold_ranges(
        sorted(evaluation_starts), predictions["trade_date"].max()
    ):
        history = daily[daily["trade_date"] < test_start].tail(lookback_dates).copy()
        test = daily[
            (daily["trade_date"] >= test_start) & (daily["trade_date"] <= test_end)
        ].copy()
        if len(history) < minimum_history_dates:
            raise ValueError(
                f"fold {fold_id} has too little confidence history: {len(history)}"
            )
        threshold = float(history["score_top2_mean"].quantile(confidence_quantile))
        test["confidence_threshold"] = threshold
        test["confidence_gate_margin"] = test["score_top2_mean"] - threshold
        test["confidence_gate_enabled"] = test["confidence_gate_margin"] >= 0.0
        test["fold"] = fold_id
        enabled = test["confidence_gate_enabled"]
        fold = {
            "fold": fold_id,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "history_start": str(history["trade_date"].min().date()),
            "history_end": str(history["trade_date"].max().date()),
            "history_dates": int(len(history)),
            "confidence_quantile": confidence_quantile,
            "confidence_threshold": threshold,
            "test_dates": int(len(test)),
            "enabled_dates": int(enabled.sum()),
            "enabled_fraction": float(enabled.mean()),
            "target_sum_all": float(test["daily_realized_edge"].sum()),
            "target_sum_enabled": float(test.loc[enabled, "daily_realized_edge"].sum()),
            "positive_ratio_enabled": float(test.loc[enabled, "daily_positive"].mean())
            if enabled.any()
            else None,
        }
        folds.append(fold)
        daily_parts.append(test)
        rows = predictions[
            (predictions["trade_date"] >= test_start)
            & (predictions["trade_date"] <= test_end)
        ].copy()
        rows = rows.merge(
            test[
                [
                    "trade_date",
                    "confidence_threshold",
                    "confidence_gate_margin",
                    "confidence_gate_enabled",
                ]
            ],
            on="trade_date",
            how="left",
            validate="many_to_one",
        )
        rows["stock_oof_fold"] = rows["fold"]
        rows["fold"] = fold_id
        rows["selection_score"] = pd.to_numeric(rows[ranking_col], errors="coerce")
        output_parts.append(rows)
        print(
            f"[daily confidence gate] fold={fold_id} threshold={threshold:.6f} "
            f"test={len(test)} enabled={fold['enabled_dates']} "
            f"edge_enabled={fold['target_sum_enabled']:.6f}",
            flush=True,
        )

    output = pd.concat(output_parts, ignore_index=True).sort_values(
        ["trade_date", "instrument"]
    )
    daily_output = pd.concat(daily_parts, ignore_index=True).sort_values("trade_date")
    output_columns = [
        "datetime",
        "trade_date",
        "instrument",
        "decision_time",
        "fold",
        "stock_oof_fold",
        "selection_score",
        "confidence_threshold",
        "confidence_gate_margin",
        "confidence_gate_enabled",
        realized_edge_col,
    ]
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    output[output_columns].to_csv(output_predictions, index=False, encoding="utf-8-sig")
    output_daily.parent.mkdir(parents=True, exist_ok=True)
    daily_output.to_csv(output_daily, index=False, encoding="utf-8-sig")
    enabled = daily_output["confidence_gate_enabled"]
    report = {
        "status": "sequential_daily_confidence_gate_built",
        "predictions": str(predictions_path.resolve()),
        "output_predictions": str(output_predictions.resolve()),
        "output_daily": str(output_daily.resolve()),
        "ranking_col": ranking_col,
        "realized_edge_col": realized_edge_col,
        "daily_top_n": daily_top_n,
        "history_start": str(history_start.date()),
        "lookback_dates": lookback_dates,
        "minimum_history_dates": minimum_history_dates,
        "confidence_quantile": confidence_quantile,
        "folds": folds,
        "overall": {
            "rows": int(len(output)),
            "dates": int(len(daily_output)),
            "enabled_dates": int(enabled.sum()),
            "enabled_fraction": float(enabled.mean()),
            "target_sum_all": float(daily_output["daily_realized_edge"].sum()),
            "target_sum_enabled": float(
                daily_output.loc[enabled, "daily_realized_edge"].sum()
            ),
            "positive_ratio_enabled": float(
                daily_output.loc[enabled, "daily_positive"].mean()
            )
            if enabled.any()
            else None,
        },
        "causality": "Thresholds use only the prior 252 stock-level OOF dates and no outcome labels.",
        "selection_policy": "pre-registered prior-252-day q75 confidence gate and Top-2",
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-daily", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ranking-col", default="raw_score")
    parser.add_argument("--realized-edge-col", required=True)
    parser.add_argument(
        "--evaluation-starts",
        default="2025-06-03,2025-07-22,2025-09-09,2025-10-29,2025-12-17,2026-02-05",
    )
    parser.add_argument("--history-start", default="2023-08-03")
    parser.add_argument("--daily-top-n", type=int, default=2)
    parser.add_argument("--lookback-dates", type=int, default=252)
    parser.add_argument("--minimum-history-dates", type=int, default=120)
    parser.add_argument("--confidence-quantile", type=float, default=0.75)
    args = parser.parse_args()
    build_confidence_gate(
        Path(args.predictions).resolve(),
        Path(args.output_predictions).resolve(),
        Path(args.output_daily).resolve(),
        Path(args.output_json).resolve(),
        ranking_col=args.ranking_col,
        realized_edge_col=args.realized_edge_col,
        evaluation_starts=[
            pd.Timestamp(value.strip()).normalize()
            for value in args.evaluation_starts.split(",")
            if value.strip()
        ],
        history_start=pd.Timestamp(args.history_start).normalize(),
        daily_top_n=args.daily_top_n,
        lookback_dates=args.lookback_dates,
        minimum_history_dates=args.minimum_history_dates,
        confidence_quantile=args.confidence_quantile,
    )


if __name__ == "__main__":
    main()
