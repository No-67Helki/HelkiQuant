from __future__ import annotations

import numpy as np
import pandas as pd


LIMIT_ENGINEERED_FEATURES = [
    "board_limit_ratio",
    "board_is_chinext",
    "board_is_star",
    "board_is_beijing",
    "board_is_main",
    "visible_distance_to_limit_up",
    "visible_distance_from_limit_down",
    "visible_limit_up_room_fraction",
    "visible_limit_down_room_fraction",
]

INDUSTRY_ENGINEERED_FEATURES = [
    "industry_known",
    "industry_held_count",
    "industry_visible_ret_mean",
    "industry_visible_ret_std",
    "industry_visible_ret_rel",
    "industry_visible_gap_mean",
    "industry_visible_gap_rel",
    "industry_visible_vwap_dev_mean",
    "industry_visible_vwap_dev_rel",
    "industry_positive_breadth",
    "industry_target_weight_sum",
    "industry_current_weight_sum",
    "industry_target_gap",
    "industry_middle_mean",
    "industry_middle_rel",
]

REALTIME_ENGINEERED_FEATURES = LIMIT_ENGINEERED_FEATURES + INDUSTRY_ENGINEERED_FEATURES


def _instrument_series(frame: pd.DataFrame) -> pd.Series:
    for col in ("instrument", "local_symbol", "symbol"):
        if col in frame.columns:
            return frame[col].astype(str).str.upper()
    return pd.Series("", index=frame.index, dtype=object)


def _stock_codes(frame: pd.DataFrame) -> pd.Series:
    return _instrument_series(frame).str.extract(r"(\d{6})$", expand=False).fillna("")


def _numeric_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def add_board_limit_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    codes = _stock_codes(out)
    is_chinext = codes.str.startswith(("300", "301"))
    is_star = codes.str.startswith(("688", "689"))
    is_beijing = codes.str.startswith(("4", "8", "92"))
    limit_ratio = np.select(
        [is_chinext, is_star, is_beijing],
        [0.20, 0.20, 0.30],
        default=0.10,
    ).astype(float)
    previous_close = _numeric_series(out, "mark_price")
    decision_price = _numeric_series(out, "sell_price_decision")
    valid = (
        np.isfinite(previous_close)
        & np.isfinite(decision_price)
        & (previous_close > 0)
        & (decision_price > 0)
    )
    upper = previous_close * (1.0 + limit_ratio)
    lower = previous_close * (1.0 - limit_ratio)
    out["board_limit_ratio"] = limit_ratio
    out["board_is_chinext"] = is_chinext.astype(float)
    out["board_is_star"] = is_star.astype(float)
    out["board_is_beijing"] = is_beijing.astype(float)
    out["board_is_main"] = (~(is_chinext | is_star | is_beijing)).astype(float)
    out["visible_distance_to_limit_up"] = np.where(
        valid,
        upper / (decision_price + 1e-12) - 1.0,
        np.nan,
    )
    out["visible_distance_from_limit_down"] = np.where(
        valid,
        decision_price / (lower + 1e-12) - 1.0,
        np.nan,
    )
    out["visible_limit_up_room_fraction"] = np.where(
        valid,
        (upper - decision_price) / (previous_close * limit_ratio + 1e-12),
        np.nan,
    )
    out["visible_limit_down_room_fraction"] = np.where(
        valid,
        (decision_price - lower) / (previous_close * limit_ratio + 1e-12),
        np.nan,
    )
    return out


def add_industry_snapshot_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    instruments = _instrument_series(out)
    if "group" in out.columns:
        raw_group = out["group"].astype(str).str.strip()
    else:
        raw_group = pd.Series("", index=out.index, dtype=object)
    known = ~raw_group.str.upper().isin({"", "NAN", "NONE", "UNKNOWN", "OTHER"})
    out["industry_known"] = known.astype(float)
    out["_industry_key"] = raw_group.where(known, "__UNKNOWN__" + instruments)
    for source in (
        "visible_ret_from_open",
        "visible_gap_vs_mark",
        "visible_vwap_dev",
        "target_weight",
        "weight",
        "middle",
    ):
        out[source] = _numeric_series(out, source)
    out["target_weight"] = out["target_weight"].fillna(0.0)
    out["weight"] = out["weight"].fillna(0.0)
    snapshot_cols = [col for col in ("trade_date", "decision_time") if col in out.columns]
    keys = snapshot_cols + ["_industry_key"]
    grouped = out.groupby(keys, sort=False, observed=True, dropna=False)
    out["industry_held_count"] = grouped["_industry_key"].transform("size").astype(float)

    specs = {
        "visible_ret_from_open": "industry_visible_ret",
        "visible_gap_vs_mark": "industry_visible_gap",
        "visible_vwap_dev": "industry_visible_vwap_dev",
    }
    for source, prefix in specs.items():
        values = out[source]
        out[f"{prefix}_mean"] = grouped[source].transform("mean")
        if source == "visible_ret_from_open":
            out[f"{prefix}_std"] = grouped[source].transform("std").fillna(0.0)
        out[f"{prefix}_rel"] = values - out[f"{prefix}_mean"]
    positive = (pd.to_numeric(out["visible_ret_from_open"], errors="coerce") > 0.0).astype(
        float
    )
    out["_industry_positive"] = positive
    out["industry_positive_breadth"] = grouped["_industry_positive"].transform("mean")

    for source, output in (
        ("target_weight", "industry_target_weight_sum"),
        ("weight", "industry_current_weight_sum"),
    ):
        out[output] = grouped[source].transform("sum")
    out["industry_target_gap"] = (
        out["industry_target_weight_sum"] - out["industry_current_weight_sum"]
    )
    out["industry_middle_mean"] = grouped["middle"].transform("mean")
    out["industry_middle_rel"] = out["middle"] - out["industry_middle_mean"]
    return out.drop(columns=["_industry_key", "_industry_positive"])


def add_realtime_reproducible_factors(frame: pd.DataFrame) -> pd.DataFrame:
    return add_industry_snapshot_features(add_board_limit_features(frame))
