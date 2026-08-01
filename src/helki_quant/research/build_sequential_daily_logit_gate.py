from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_held_intraday_decision_dataset import normalize_inst
from build_sequential_daily_meta_gate import META_FEATURES, build_daily_meta_frame


CONTEXT_SOURCE_COLUMNS = [
    "held_market_ret_mean",
    "held_market_ret_std",
    "held_market_gap_mean",
    "held_market_vwap_dev_mean",
    "held_market_positive_breadth",
    "target_weight",
    "weight",
    "visible_minute_vol",
    "visible_drawdown_from_high",
    "visible_distance_to_limit_up",
    "held_unrealized_ret_approx",
    "held_abs_weight_gap_to_target",
    "middle",
]

CONTEXT_FEATURES = [
    "ctx_market_ret_mean",
    "ctx_market_ret_std",
    "ctx_market_gap_mean",
    "ctx_market_vwap_dev_mean",
    "ctx_market_positive_breadth",
    "ctx_target_weight_sum",
    "ctx_current_weight_sum",
    "ctx_minute_vol_mean",
    "ctx_drawdown_mean",
    "ctx_limit_up_distance_mean",
    "ctx_unrealized_mean",
    "ctx_weight_gap_abs_mean",
    "ctx_middle_mean",
    "ctx_middle_std",
]

ALL_FEATURES = META_FEATURES + CONTEXT_FEATURES


def aggregate_live_context(
    dataset_paths: tuple[Path, ...],
    *,
    decision_time: str,
) -> pd.DataFrame:
    required = {
        "trade_date",
        "instrument",
        "decision_time",
        *CONTEXT_SOURCE_COLUMNS,
    }
    parts = []
    for path in dataset_paths:
        print(f"[daily logit gate] loading live context {path}", flush=True)
        part = pd.read_csv(path, usecols=lambda column: column in required)
        missing = [column for column in required if column not in part]
        if missing:
            raise KeyError(f"live context columns missing in {path}: {missing}")
        part["trade_date"] = pd.to_datetime(part["trade_date"], errors="raise").dt.normalize()
        part["instrument"] = part["instrument"].map(normalize_inst)
        part["decision_time"] = (
            part["decision_time"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(4)
        )
        part = part[part["decision_time"] == str(decision_time).zfill(4)].copy()
        if part.duplicated(["trade_date", "instrument"]).any():
            raise ValueError(f"duplicate daily live context rows in {path}")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True, sort=False)
    if frame.duplicated(["trade_date", "instrument"]).any():
        raise ValueError("overlapping live context datasets")
    numeric = [column for column in CONTEXT_SOURCE_COLUMNS if column in frame]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby("trade_date", sort=True)
    daily = grouped.agg(
        ctx_market_ret_mean=("held_market_ret_mean", "median"),
        ctx_market_ret_std=("held_market_ret_std", "median"),
        ctx_market_gap_mean=("held_market_gap_mean", "median"),
        ctx_market_vwap_dev_mean=("held_market_vwap_dev_mean", "median"),
        ctx_market_positive_breadth=("held_market_positive_breadth", "median"),
        ctx_target_weight_sum=("target_weight", "sum"),
        ctx_current_weight_sum=("weight", "sum"),
        ctx_minute_vol_mean=("visible_minute_vol", "mean"),
        ctx_drawdown_mean=("visible_drawdown_from_high", "mean"),
        ctx_limit_up_distance_mean=("visible_distance_to_limit_up", "mean"),
        ctx_unrealized_mean=("held_unrealized_ret_approx", "mean"),
        ctx_weight_gap_abs_mean=("held_abs_weight_gap_to_target", "mean"),
        ctx_middle_mean=("middle", "mean"),
        ctx_middle_std=("middle", "std"),
    ).reset_index()
    return daily


def _fold_ranges(
    starts: list[pd.Timestamp], max_date: pd.Timestamp
) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    ranges = []
    ordered = sorted(starts)
    for index, start in enumerate(ordered, start=1):
        end = ordered[index] - pd.Timedelta(days=1) if index < len(ordered) else max_date
        ranges.append((index, start, end))
    return ranges


def _safe_auc(label: pd.Series, score: pd.Series) -> float | None:
    if label.nunique() < 2:
        return None
    return float(roc_auc_score(label, score))


def build_logit_gate(
    predictions_path: Path,
    feature_dataset_paths: tuple[Path, ...],
    output_predictions: Path,
    output_daily: Path,
    output_json: Path,
    *,
    ranking_col: str,
    realized_edge_col: str,
    decision_time: str,
    evaluation_starts: list[pd.Timestamp],
    history_start: pd.Timestamp,
    daily_top_n: int,
    regularization_c: float,
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
    predictions = predictions[
        predictions["decision_time"] == str(decision_time).zfill(4)
    ].copy()
    daily = build_daily_meta_frame(
        predictions,
        ranking_col=ranking_col,
        realized_edge_col=realized_edge_col,
        daily_top_n=daily_top_n,
    )
    context = aggregate_live_context(
        feature_dataset_paths,
        decision_time=decision_time,
    )
    daily = daily.merge(context, on="trade_date", how="left", validate="one_to_one")
    missing_context = int(daily[CONTEXT_FEATURES].isna().all(axis=1).sum())
    if missing_context:
        raise ValueError(f"daily OOF rows missing all live context: {missing_context}")
    daily = daily[daily["trade_date"] >= history_start].copy()

    starts = sorted(evaluation_starts)
    output_parts = []
    daily_parts = []
    fold_reports = []
    for fold_id, test_start, test_end in _fold_ranges(starts, predictions["trade_date"].max()):
        train = daily[daily["trade_date"] < test_start].copy()
        test = daily[
            (daily["trade_date"] >= test_start) & (daily["trade_date"] <= test_end)
        ].copy()
        if len(train) < minimum_training_dates:
            raise ValueError(
                f"fold {fold_id} has too few earlier OOF dates: {len(train)}"
            )
        if train["daily_positive"].nunique() < 2:
            raise ValueError(f"fold {fold_id} gate training requires both classes")
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=regularization_c,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42 + fold_id,
                    ),
                ),
            ]
        )
        pipeline.fit(train[ALL_FEATURES], train["daily_positive"].astype(int))
        test["meta_gate_probability"] = pipeline.predict_proba(test[ALL_FEATURES])[:, 1]
        test["meta_gate_enabled"] = test["meta_gate_probability"] >= 0.5
        test["fold"] = fold_id
        enabled = test["meta_gate_enabled"]
        logit = pipeline.named_steps["logit"]
        fold_report = {
            "fold": fold_id,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "train_start": str(train["trade_date"].min().date()),
            "train_end": str(train["trade_date"].max().date()),
            "train_dates": int(len(train)),
            "train_positive_ratio": float(train["daily_positive"].mean()),
            "test_dates": int(len(test)),
            "test_positive_ratio": float(test["daily_positive"].mean()),
            "enabled_dates": int(enabled.sum()),
            "enabled_fraction": float(enabled.mean()),
            "enabled_positive_ratio": float(test.loc[enabled, "daily_positive"].mean())
            if enabled.any()
            else None,
            "test_target_sum_all": float(test["daily_realized_edge"].sum()),
            "test_target_sum_enabled": float(
                test.loc[enabled, "daily_realized_edge"].sum()
            ),
            "test_auc": _safe_auc(test["daily_positive"], test["meta_gate_probability"]),
            "test_log_loss": float(
                log_loss(
                    test["daily_positive"],
                    test["meta_gate_probability"],
                    labels=[0, 1],
                )
            ),
            "logit_intercept": float(logit.intercept_[0]),
            "logit_coefficients": {
                feature: float(coefficient)
                for feature, coefficient in zip(ALL_FEATURES, logit.coef_[0])
            },
        }
        fold_reports.append(fold_report)
        daily_parts.append(test)
        rows = predictions[
            (predictions["trade_date"] >= test_start)
            & (predictions["trade_date"] <= test_end)
        ].copy()
        rows = rows.merge(
            test[["trade_date", "meta_gate_probability", "meta_gate_enabled"]],
            on="trade_date",
            how="left",
            validate="many_to_one",
        )
        rows["stock_oof_fold"] = rows["fold"]
        rows["fold"] = fold_id
        rows["selection_score"] = pd.to_numeric(rows[ranking_col], errors="coerce")
        output_parts.append(rows)
        print(
            f"[daily logit gate] fold={fold_id} train={len(train)} test={len(test)} "
            f"enabled={fold_report['enabled_dates']} auc={fold_report['test_auc']} "
            f"edge_enabled={fold_report['test_target_sum_enabled']:.6f}",
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
        "meta_gate_probability",
        "meta_gate_enabled",
        realized_edge_col,
    ]
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    output[output_columns].to_csv(output_predictions, index=False, encoding="utf-8-sig")
    output_daily.parent.mkdir(parents=True, exist_ok=True)
    daily_output.to_csv(output_daily, index=False, encoding="utf-8-sig")
    enabled = daily_output["meta_gate_enabled"]
    report = {
        "status": "sequential_daily_logit_gate_built",
        "predictions": str(predictions_path.resolve()),
        "feature_datasets": [str(path.resolve()) for path in feature_dataset_paths],
        "output_predictions": str(output_predictions.resolve()),
        "output_daily": str(output_daily.resolve()),
        "ranking_col": ranking_col,
        "realized_edge_col": realized_edge_col,
        "daily_top_n": daily_top_n,
        "history_start": str(history_start.date()),
        "regularization_c": regularization_c,
        "minimum_training_dates": minimum_training_dates,
        "features": ALL_FEATURES,
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
        "causality": "Each fold classifier is fit only on strictly earlier stock-level OOF dates and pre-decision context.",
        "selection_policy": "pre-registered balanced LogisticRegression(C=0.1), p>=0.5, Top-2",
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--feature-dataset", action="append", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-daily", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ranking-col", default="raw_score")
    parser.add_argument("--realized-edge-col", required=True)
    parser.add_argument("--decision-time", default="1000")
    parser.add_argument(
        "--evaluation-starts",
        default="2025-06-03,2025-07-22,2025-09-09,2025-10-29,2025-12-17,2026-02-05",
    )
    parser.add_argument("--history-start", default="2023-08-03")
    parser.add_argument("--daily-top-n", type=int, default=2)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument("--minimum-training-dates", type=int, default=120)
    args = parser.parse_args()
    build_logit_gate(
        Path(args.predictions).resolve(),
        tuple(Path(path).resolve() for path in args.feature_dataset),
        Path(args.output_predictions).resolve(),
        Path(args.output_daily).resolve(),
        Path(args.output_json).resolve(),
        ranking_col=args.ranking_col,
        realized_edge_col=args.realized_edge_col,
        decision_time=args.decision_time,
        evaluation_starts=[
            pd.Timestamp(value.strip()).normalize()
            for value in args.evaluation_starts.split(",")
            if value.strip()
        ],
        history_start=pd.Timestamp(args.history_start).normalize(),
        daily_top_n=args.daily_top_n,
        regularization_c=args.regularization_c,
        minimum_training_dates=args.minimum_training_dates,
    )


if __name__ == "__main__":
    main()
