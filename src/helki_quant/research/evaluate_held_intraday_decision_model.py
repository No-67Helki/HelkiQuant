from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from held_intraday_factor_engineering import (
    INDUSTRY_ENGINEERED_FEATURES,
    LIMIT_ENGINEERED_FEATURES,
    REALTIME_ENGINEERED_FEATURES,
)


FEATURE_COLS = [
    "shares",
    "mark_price",
    "weight",
    "target_weight",
    "target_shares",
    "held_age_days",
    "held_unrealized_ret_approx",
    "held_prev_day_ret",
    "held_weight_gap_to_target",
    "held_abs_weight_gap_to_target",
    "held_share_gap_to_target",
    "held_target_share_ratio",
    "target_missing",
    "rank",
    "middle",
    "decision_minute",
    "sell_price_decision",
    "visible_ret_from_open",
    "visible_gap_vs_mark",
    "visible_last_vs_mark",
    "visible_range",
    "visible_range_pos",
    "visible_vwap_dev",
    "visible_volume",
    "visible_amount",
    "visible_minute_vol",
    "visible_ret_autocorr",
    "visible_price_vol_corr",
    "visible_first_5m_ret",
    "visible_first_10m_ret",
    "visible_first_15m_ret",
    "visible_first_30m_ret",
    "visible_recent_5m_ret",
    "visible_recent_10m_ret",
    "visible_recent_15m_ret",
    "visible_momentum_accel_5m",
    "visible_trend_slope",
    "visible_drawdown_from_high",
    "visible_rebound_from_low",
    "visible_up_minute_ratio",
    "visible_signed_volume_ratio",
    "visible_recent_5m_volume_share",
    "visible_recent_10m_volume_share",
    "visible_log_volume",
    "visible_log_amount",
    "held_market_value",
    "held_log_market_value",
    "held_lot_count",
    "visible_volume_to_held",
    "visible_amount_to_position",
    "t0_volume_10pct",
    "t0_min_fee_drag_10pct",
    "t0_volume_20pct",
    "t0_min_fee_drag_20pct",
    "t0_volume_30pct",
    "t0_min_fee_drag_30pct",
    "held_universe_size",
    "t0_exec_volume_10pct",
    "t0_exec_tradeable_10pct",
    "t0_exec_volume_20pct",
    "t0_exec_tradeable_20pct",
    "t0_exec_volume_30pct",
    "t0_exec_tradeable_30pct",
    "t0_exec_volume_one_lot_max50",
    "t0_exec_tradeable_one_lot_max50",
    "held_market_ret_mean",
    "held_market_ret_median",
    "held_market_ret_std",
    "held_market_gap_mean",
    "held_market_gap_median",
    "held_market_gap_std",
    "held_market_last_vs_mark_mean",
    "held_market_last_vs_mark_median",
    "held_market_last_vs_mark_std",
    "held_market_vwap_dev_mean",
    "held_market_vwap_dev_median",
    "held_market_vwap_dev_std",
    "held_market_drawdown_mean",
    "held_market_drawdown_median",
    "held_market_drawdown_std",
    "held_market_positive_breadth",
]
FEATURE_COLS.extend(REALTIME_ENGINEERED_FEATURES)

CROSS_SECTIONAL_FEATURE_SOURCES = [
    "held_unrealized_ret_approx",
    "held_weight_gap_to_target",
    "visible_ret_from_open",
    "visible_gap_vs_mark",
    "visible_last_vs_mark",
    "visible_range",
    "visible_range_pos",
    "visible_vwap_dev",
    "visible_minute_vol",
    "visible_recent_5m_ret",
    "visible_recent_10m_ret",
    "visible_trend_slope",
    "visible_drawdown_from_high",
    "visible_rebound_from_low",
    "visible_log_volume",
    "visible_log_amount",
    "visible_volume_to_held",
    "visible_amount_to_position",
    "visible_distance_to_limit_up",
    "visible_distance_from_limit_down",
    "industry_visible_ret_rel",
    "industry_visible_gap_rel",
    "industry_visible_vwap_dev_rel",
]
FEATURE_COLS.extend(
    feature
    for source in CROSS_SECTIONAL_FEATURE_SOURCES
    for feature in (f"cs_{source}_rank", f"cs_{source}_rel")
)

LIVE_UNSTABLE_FEATURES = {
    "held_age_days",
    "held_prev_day_ret",
}


def _feature_family_columns(features: list[str]) -> set[str]:
    result = set(features)
    for source in features:
        result.update({f"cs_{source}_rank", f"cs_{source}_rel"})
    return result


LIMIT_FEATURE_COLUMNS = _feature_family_columns(LIMIT_ENGINEERED_FEATURES)
INDUSTRY_FEATURE_COLUMNS = _feature_family_columns(INDUSTRY_ENGINEERED_FEATURES)

LIVE_COMPACT_CORE_FEATURES = {
    "shares",
    "weight",
    "target_weight",
    "held_unrealized_ret_approx",
    "held_weight_gap_to_target",
    "held_target_share_ratio",
    "target_missing",
    "rank",
    "middle",
    "visible_ret_from_open",
    "visible_gap_vs_mark",
    "visible_range_pos",
    "visible_vwap_dev",
    "visible_minute_vol",
    "visible_first_30m_ret",
    "visible_recent_5m_ret",
    "visible_recent_10m_ret",
    "visible_momentum_accel_5m",
    "visible_trend_slope",
    "visible_drawdown_from_high",
    "visible_rebound_from_low",
    "visible_up_minute_ratio",
    "visible_signed_volume_ratio",
    "visible_recent_10m_volume_share",
    "visible_log_amount",
    "visible_amount_to_position",
    "t0_min_fee_drag_10pct",
    "t0_exec_tradeable_one_lot_max50",
    "held_universe_size",
    "held_market_ret_mean",
    "held_market_ret_std",
    "held_market_gap_mean",
    "held_market_vwap_dev_mean",
    "held_market_positive_breadth",
    "cs_held_unrealized_ret_approx_rank",
    "cs_held_weight_gap_to_target_rank",
    "cs_visible_ret_from_open_rank",
    "cs_visible_gap_vs_mark_rank",
    "cs_visible_vwap_dev_rank",
    "cs_visible_recent_5m_ret_rank",
    "cs_visible_trend_slope_rank",
    "cs_visible_drawdown_from_high_rank",
    "cs_visible_amount_to_position_rank",
}


def select_feature_cols(frame: pd.DataFrame, feature_mode: str = "all") -> list[str]:
    feature_cols = [col for col in FEATURE_COLS if col in frame.columns]
    if feature_mode in {
        "live",
        "live_core",
        "live_limit",
        "live_industry",
        "live_compact_core",
        "live_compact_limit",
    }:
        feature_cols = [col for col in feature_cols if col not in LIVE_UNSTABLE_FEATURES]
        if feature_mode == "live_compact_core":
            feature_cols = [col for col in feature_cols if col in LIVE_COMPACT_CORE_FEATURES]
        elif feature_mode == "live_compact_limit":
            allowed = LIVE_COMPACT_CORE_FEATURES | LIMIT_FEATURE_COLUMNS
            feature_cols = [col for col in feature_cols if col in allowed]
        elif feature_mode == "live_core":
            feature_cols = [
                col
                for col in feature_cols
                if col not in LIMIT_FEATURE_COLUMNS and col not in INDUSTRY_FEATURE_COLUMNS
            ]
        elif feature_mode == "live_limit":
            feature_cols = [col for col in feature_cols if col not in INDUSTRY_FEATURE_COLUMNS]
        elif feature_mode == "live_industry":
            feature_cols = [col for col in feature_cols if col not in LIMIT_FEATURE_COLUMNS]
    elif feature_mode != "all":
        raise ValueError(f"unsupported feature_mode: {feature_mode}")
    return feature_cols


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


def metrics(frame: pd.DataFrame, edge_col: str = "t0_best_edge") -> dict:
    if frame.empty:
        return {"rows": 0}
    edge = frame[edge_col] if edge_col in frame.columns else frame["t0_best_edge"]
    result = {
        "rows": int(len(frame)),
        "dates": int(frame["trade_date"].nunique()),
        "instruments": int(frame["instrument"].nunique()),
        "label_positive_ratio": float(frame["label"].mean()),
        "score_mean": float(frame["score"].mean()),
        "score_std": float(frame["score"].std()),
        "auc": auc_score(frame["label"].to_numpy(), frame["score"].to_numpy()),
        "edge_col": edge_col,
        "edge_mean": float(edge.mean()),
        "spearman_edge": float(frame["score"].corr(edge, method="spearman")),
        "selected_060_rows": int((frame["score"] >= 0.60).sum()),
        "selected_060_edge_mean": float(edge.loc[frame["score"] >= 0.60].mean())
        if (frame["score"] >= 0.60).any()
        else None,
        "selected_060_hit_ratio": float(frame.loc[frame["score"] >= 0.60, "label"].mean())
        if (frame["score"] >= 0.60).any()
        else None,
    }
    if "profit_label" in frame.columns:
        result["profit_positive_ratio"] = float(frame["profit_label"].mean())
        result["selected_060_profit_ratio"] = (
            float(frame.loc[frame["score"] >= 0.60, "profit_label"].mean())
            if (frame["score"] >= 0.60).any()
            else None
        )
    return result


def evaluate(
    input_csv: Path,
    output_json: Path,
    *,
    label_col: str,
    test_start: str,
    decision_time: str | None,
    prediction_csv: Path | None,
) -> dict:
    frame = pd.read_csv(input_csv, parse_dates=["trade_date", "datetime"])
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if decision_time:
        wanted = str(decision_time).zfill(4)
        got = frame["decision_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
        frame = frame[got == wanted].copy()
    frame = frame.dropna(subset=[label_col, "t0_best_edge"])
    frame["label"] = (frame[label_col] > 0.5).astype(int)
    feature_cols = select_feature_cols(frame)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[feature_cols] = frame[feature_cols].fillna(0.0)
    test_start_ts = pd.Timestamp(test_start).normalize()
    train = frame[frame["trade_date"] < test_start_ts].copy()
    test = frame[frame["trade_date"] >= test_start_ts].copy()
    if len(train) < 100 or len(test) < 50:
        raise ValueError(f"not enough rows for split train={len(train)} test={len(test)}")
    model = CatBoostClassifier(
        iterations=120,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        task_type="CPU",
        allow_writing_files=False,
        verbose=50,
    )
    model.fit(
        Pool(train[feature_cols], label=train["label"]),
        eval_set=Pool(test[feature_cols], label=test["label"]),
        use_best_model=True,
    )
    test["score"] = model.predict_proba(test[feature_cols])[:, 1]
    train["score"] = model.predict_proba(train[feature_cols])[:, 1]
    if prediction_csv is not None:
        pred_cols = [
            "datetime",
            "trade_date",
            "instrument",
            "decision_time",
            label_col,
            "label",
            "t0_best_edge",
            "score",
        ]
        pred_cols = [col for col in pred_cols if col in test.columns]
        prediction_csv.parent.mkdir(parents=True, exist_ok=True)
        test[pred_cols].to_csv(prediction_csv, index=False, encoding="utf-8-sig")
    result = {
        "status": "held_intraday_decision_model_evaluated",
        "input_csv": str(input_csv.resolve()),
        "label_col": label_col,
        "decision_time": decision_time,
        "feature_cols": feature_cols,
        "test_start": str(test_start_ts.date()),
        "train": metrics(train),
        "test": metrics(test),
        "best_iteration": model.get_best_iteration(),
        "prediction_csv": str(prediction_csv.resolve()) if prediction_csv else None,
        "deployment_allowed": False,
        "research_only_reason": "Held-only intraday decision model is for offline research and is not connected to GmQuant.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--label-col", default="t0_best_hit")
    parser.add_argument("--test-start", default="2026-02-01")
    parser.add_argument("--decision-time", default=None)
    parser.add_argument("--prediction-csv", default=None)
    args = parser.parse_args()
    report = evaluate(
        Path(args.input_csv).resolve(),
        Path(args.output_json).resolve(),
        label_col=args.label_col,
        test_start=args.test_start,
        decision_time=args.decision_time,
        prediction_csv=Path(args.prediction_csv).resolve() if args.prediction_csv else None,
    )
    print(
        "[held intraday model] "
        f"train_rows={report['train']['rows']} test_rows={report['test']['rows']} "
        f"test_auc={report['test'].get('auc')} test_spearman={report['test'].get('spearman_edge')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
