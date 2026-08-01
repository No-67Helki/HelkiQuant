from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


META_FEATURES = [
    "held_count",
    "score_max",
    "score_top2_mean",
    "score_median",
    "score_std",
    "score_q75",
    "score_q90",
    "score_top1_top2_spread",
    "score_top2_median_spread",
    "score_above_050_fraction",
]


def build_daily_meta_frame(
    predictions: pd.DataFrame,
    *,
    ranking_col: str,
    realized_edge_col: str,
    daily_top_n: int,
) -> pd.DataFrame:
    rows = []
    for trade_date, part in predictions.groupby("trade_date", sort=True):
        scores = pd.to_numeric(part[ranking_col], errors="coerce")
        if scores.isna().all():
            continue
        ranked = part.assign(_ranking_score=scores).sort_values(
            ["_ranking_score", "instrument"], ascending=[False, True]
        )
        top = ranked.head(daily_top_n)
        top_scores = pd.to_numeric(top["_ranking_score"], errors="coerce")
        top1 = float(top_scores.iloc[0])
        top2_value = float(top_scores.iloc[1]) if len(top_scores) > 1 else top1
        median = float(scores.median())
        target = float(pd.to_numeric(top[realized_edge_col], errors="coerce").sum())
        rows.append(
            {
                "trade_date": pd.Timestamp(trade_date).normalize(),
                "held_count": float(len(part)),
                "score_max": float(scores.max()),
                "score_top2_mean": float(top_scores.mean()),
                "score_median": median,
                "score_std": float(scores.std(ddof=0)),
                "score_q75": float(scores.quantile(0.75)),
                "score_q90": float(scores.quantile(0.90)),
                "score_top1_top2_spread": top1 - top2_value,
                "score_top2_median_spread": float(top_scores.mean()) - median,
                "score_above_050_fraction": float((scores >= 0.5).mean()),
                "daily_realized_edge": target,
                "daily_positive": float(target > 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def _fold_ranges(starts: list[pd.Timestamp], max_date: pd.Timestamp) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    ranges = []
    for index, start in enumerate(sorted(starts), start=1):
        end = starts[index] - pd.Timedelta(days=1) if index < len(starts) else max_date
        ranges.append((index, start, end))
    return ranges


def build_meta_gate(
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
    ridge_alpha: float,
    minimum_training_dates: int,
) -> dict:
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
    required = [ranking_col, realized_edge_col]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise KeyError(f"meta gate prediction columns missing: {missing}")
    daily = build_daily_meta_frame(
        predictions,
        ranking_col=ranking_col,
        realized_edge_col=realized_edge_col,
        daily_top_n=daily_top_n,
    )
    daily = daily[daily["trade_date"] >= history_start].copy()
    starts = sorted(evaluation_starts)
    max_date = predictions["trade_date"].max()
    output_parts = []
    daily_parts = []
    fold_reports = []
    for fold_id, test_start, test_end in _fold_ranges(starts, max_date):
        train_daily = daily[daily["trade_date"] < test_start].copy()
        test_daily = daily[
            (daily["trade_date"] >= test_start) & (daily["trade_date"] <= test_end)
        ].copy()
        if len(train_daily) < minimum_training_dates:
            raise ValueError(
                f"fold {fold_id} has too few earlier OOF dates: "
                f"train={len(train_daily)} minimum={minimum_training_dates}"
            )
        if test_daily.empty:
            raise ValueError(f"fold {fold_id} has no daily rows")
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha)),
            ]
        )
        pipeline.fit(train_daily[META_FEATURES], train_daily["daily_realized_edge"])
        test_daily["meta_gate_score"] = pipeline.predict(test_daily[META_FEATURES])
        test_daily["meta_gate_enabled"] = test_daily["meta_gate_score"] > 0.0
        test_daily["fold"] = fold_id
        ridge = pipeline.named_steps["ridge"]
        fold_report = {
            "fold": fold_id,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "train_start": str(train_daily["trade_date"].min().date()),
            "train_end": str(train_daily["trade_date"].max().date()),
            "train_dates": int(len(train_daily)),
            "test_dates": int(len(test_daily)),
            "enabled_dates": int(test_daily["meta_gate_enabled"].sum()),
            "enabled_fraction": float(test_daily["meta_gate_enabled"].mean()),
            "test_target_sum_all": float(test_daily["daily_realized_edge"].sum()),
            "test_target_sum_enabled": float(
                test_daily.loc[test_daily["meta_gate_enabled"], "daily_realized_edge"].sum()
            ),
            "test_target_positive_ratio_enabled": float(
                test_daily.loc[test_daily["meta_gate_enabled"], "daily_positive"].mean()
            )
            if test_daily["meta_gate_enabled"].any()
            else None,
            "test_score_target_spearman": float(
                test_daily["meta_gate_score"].corr(
                    test_daily["daily_realized_edge"], method="spearman"
                )
            ),
            "test_mae": float(
                mean_absolute_error(
                    test_daily["daily_realized_edge"], test_daily["meta_gate_score"]
                )
            ),
            "ridge_intercept": float(ridge.intercept_),
            "ridge_coefficients": {
                feature: float(coefficient)
                for feature, coefficient in zip(META_FEATURES, ridge.coef_)
            },
        }
        fold_reports.append(fold_report)
        daily_parts.append(test_daily)

        test_rows = predictions[
            (predictions["trade_date"] >= test_start)
            & (predictions["trade_date"] <= test_end)
        ].copy()
        test_rows = test_rows.merge(
            test_daily[["trade_date", "meta_gate_score", "meta_gate_enabled"]],
            on="trade_date",
            how="left",
            validate="many_to_one",
        )
        test_rows["stock_oof_fold"] = test_rows["fold"]
        test_rows["fold"] = fold_id
        test_rows["selection_score"] = pd.to_numeric(test_rows[ranking_col], errors="coerce")
        output_parts.append(test_rows)
        print(
            f"[daily meta gate] fold={fold_id} train_dates={len(train_daily)} "
            f"test_dates={len(test_daily)} enabled={fold_report['enabled_dates']} "
            f"edge_all={fold_report['test_target_sum_all']:.6f} "
            f"edge_enabled={fold_report['test_target_sum_enabled']:.6f}",
            flush=True,
        )

    output = pd.concat(output_parts, ignore_index=True).sort_values(
        ["trade_date", "instrument"]
    )
    output_columns = [
        "datetime",
        "trade_date",
        "instrument",
        "decision_time",
        "fold",
        "stock_oof_fold",
        "selection_score",
        "meta_gate_score",
        "meta_gate_enabled",
        realized_edge_col,
    ]
    daily_output = pd.concat(daily_parts, ignore_index=True).sort_values("trade_date")
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    output[output_columns].to_csv(output_predictions, index=False, encoding="utf-8-sig")
    output_daily.parent.mkdir(parents=True, exist_ok=True)
    daily_output.to_csv(output_daily, index=False, encoding="utf-8-sig")
    enabled = daily_output["meta_gate_enabled"]
    report = {
        "status": "sequential_daily_meta_gate_built",
        "predictions": str(predictions_path.resolve()),
        "output_predictions": str(output_predictions.resolve()),
        "output_daily": str(output_daily.resolve()),
        "ranking_col": ranking_col,
        "realized_edge_col": realized_edge_col,
        "daily_top_n": daily_top_n,
        "history_start": str(history_start.date()),
        "ridge_alpha": ridge_alpha,
        "minimum_training_dates": minimum_training_dates,
        "meta_features": META_FEATURES,
        "folds": fold_reports,
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
        "causality": "Each fold gate is fit only on strictly earlier stock-level OOF dates.",
        "selection_policy": "pre-registered Ridge(alpha=10), predicted edge > 0, daily Top-2",
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
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--minimum-training-dates", type=int, default=120)
    args = parser.parse_args()
    build_meta_gate(
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
        ridge_alpha=args.ridge_alpha,
        minimum_training_dates=args.minimum_training_dates,
    )


if __name__ == "__main__":
    main()
