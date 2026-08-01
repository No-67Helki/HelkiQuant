from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from held_intraday_live_features import minimum_lot


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


def isotonic_score(raw_score: float, x_thresholds: np.ndarray, y_thresholds: np.ndarray) -> float:
    x = np.asarray(x_thresholds, dtype=float)
    y = np.asarray(y_thresholds, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if not np.isfinite(raw_score) or len(x) < 2 or len(x) != len(y):
        return np.nan
    return float(np.interp(float(raw_score), x, y, left=y[0], right=y[-1]))


def build_daily_meta_values(scores: pd.DataFrame, *, score_col: str = "model_score") -> dict[str, float]:
    if score_col not in scores:
        raise KeyError(f"score column missing: {score_col}")
    symbol_col = "instrument" if "instrument" in scores else "local_symbol"
    if symbol_col not in scores:
        raise KeyError("instrument/local_symbol column missing")
    values = pd.to_numeric(scores[score_col], errors="coerce")
    if values.isna().all():
        raise ValueError("all daily stock scores are missing")
    ranked = scores.assign(_ranking_score=values).sort_values(
        ["_ranking_score", symbol_col],
        ascending=[False, True],
    )
    top = pd.to_numeric(ranked.head(2)["_ranking_score"], errors="coerce")
    top1 = float(top.iloc[0])
    top2 = float(top.iloc[1]) if len(top) > 1 else top1
    median = float(values.median())
    return {
        "held_count": float(len(scores)),
        "score_max": float(values.max()),
        "score_top2_mean": float(top.mean()),
        "score_median": median,
        "score_std": float(values.std(ddof=0)),
        "score_q75": float(values.quantile(0.75)),
        "score_q90": float(values.quantile(0.90)),
        "score_top1_top2_spread": top1 - top2,
        "score_top2_median_spread": float(top.mean()) - median,
        "score_above_050_fraction": float((values >= 0.5).mean()),
    }


def ridge_gate_score(values: Mapping[str, Any], artifact: Mapping[str, Any]) -> float:
    features = list(artifact.get("meta_features", META_FEATURES))
    vector = np.asarray([float(values[name]) for name in features], dtype=float)
    mean = np.asarray(artifact["scaler"]["mean"], dtype=float)
    scale = np.asarray(artifact["scaler"]["scale"], dtype=float)
    coefficient = np.asarray(artifact["ridge"]["coefficient"], dtype=float)
    if not (len(vector) == len(mean) == len(scale) == len(coefficient)):
        raise ValueError("daily Ridge artifact shape mismatch")
    if (scale <= 0).any():
        raise ValueError("daily Ridge artifact contains non-positive scale")
    return float(
        ((vector - mean) / scale) @ coefficient
        + float(artifact["ridge"]["intercept"])
    )


def require_fresh_target(
    today: str | pd.Timestamp,
    source_date: str | pd.Timestamp,
    *,
    max_calendar_age_days: int = 0,
) -> int:
    current = pd.Timestamp(today).normalize()
    source = pd.Timestamp(source_date).normalize()
    age = int((current - source).days)
    if age < 0:
        raise RuntimeError(
            f"target source date is in the future: today={current.date()} source={source.date()}"
        )
    if age > int(max_calendar_age_days):
        raise RuntimeError(
            f"stale target context: today={current.date()} source={source.date()} "
            f"age_days={age} max={int(max_calendar_age_days)}"
        )
    return age


def require_fresh_signal(
    today: str | pd.Timestamp,
    signal_date: str | pd.Timestamp,
    *,
    max_calendar_age_days: int = 4,
) -> int:
    current = pd.Timestamp(today).normalize()
    signal = pd.Timestamp(signal_date).normalize()
    age = int((current - signal).days)
    if age <= 0:
        raise RuntimeError(
            f"daily signal must be from a completed earlier session: "
            f"today={current.date()} signal={signal.date()} age_days={age}"
        )
    if age > int(max_calendar_age_days):
        raise RuntimeError(
            f"stale daily signal: today={current.date()} signal={signal.date()} "
            f"age_days={age} max={int(max_calendar_age_days)}"
        )
    return age


def select_sell_first_candidates(
    scores: pd.DataFrame,
    *,
    component: str,
    daily_top_n: int = 2,
    score_threshold: float | None = None,
    trigger_distance: float = 0.0075,
) -> pd.DataFrame:
    required = {
        "symbol",
        "local_symbol",
        "held_volume",
        "available_volume",
        "decision_price",
        "model_score",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"missing candidate columns: {missing}")
    frame = scores.copy()
    frame["lot"] = frame["symbol"].map(minimum_lot).astype(int)
    frame["eligible_one_lot"] = (
        (pd.to_numeric(frame["held_volume"], errors="coerce") >= 2 * frame["lot"])
        & (pd.to_numeric(frame["available_volume"], errors="coerce") >= frame["lot"])
        & (pd.to_numeric(frame["decision_price"], errors="coerce") > 0)
    )
    eligible = frame[frame["eligible_one_lot"]].copy()
    if score_threshold is not None:
        eligible = eligible[
            pd.to_numeric(eligible["model_score"], errors="coerce") >= score_threshold
        ].copy()
    selected = (
        eligible.sort_values(
            ["model_score", "local_symbol"],
            ascending=[False, True],
        )
        .head(int(daily_top_n))
        .copy()
    )
    selected["component"] = component
    selected["direction"] = "sell_first"
    selected["score"] = pd.to_numeric(selected["model_score"], errors="coerce")
    selected["volume"] = selected["lot"].astype(int)
    selected["entry_limit"] = (
        pd.to_numeric(selected["decision_price"], errors="coerce")
        * (1.0 + float(trigger_distance))
    )
    return selected.reset_index(drop=True)


def combine_primary_secondary(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    max_symbols: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_symbols = set(primary.get("symbol", pd.Series(dtype=str)).astype(str))
    conflicts = secondary[secondary["symbol"].astype(str).isin(primary_symbols)].copy()
    secondary_kept = secondary[~secondary["symbol"].astype(str).isin(primary_symbols)].copy()
    primary_part = primary.copy()
    secondary_part = secondary_kept.copy()
    primary_part["component_priority"] = 0
    secondary_part["component_priority"] = 1
    combined = pd.concat([primary_part, secondary_part], ignore_index=True, sort=False)
    combined = (
        combined.sort_values(
            ["component_priority", "score", "local_symbol"],
            ascending=[True, False, True],
        )
        .head(int(max_symbols))
        .reset_index(drop=True)
    )
    return combined, conflicts.reset_index(drop=True)


def trigger_reached(last_price: float, entry_limit: float) -> bool:
    if not np.isfinite(last_price) or not np.isfinite(entry_limit):
        return False
    if last_price <= 0 or entry_limit <= 0:
        return False
    return bool(last_price >= entry_limit)


def estimate_fee(value: float, rate: float, min_cost: float = 5.0) -> float:
    return max(float(value) * float(rate), float(min_cost)) if value > 0 else 0.0


def build_sell_entry_intent(
    candidate: Mapping[str, Any],
    *,
    trigger_price: float,
    nav: float,
    turnover_used: float,
    max_daily_turnover: float = 0.03,
    sell_cost: float = 0.0025,
    min_cost: float = 5.0,
) -> tuple[dict[str, Any], bool]:
    volume = int(candidate["volume"])
    value = volume * float(trigger_price)
    reserved_roundtrip = 2.0 * value
    available = int(candidate["available_volume"])
    accepted = True
    action = "SELL_FIRST_TRIGGERED"
    if turnover_used + reserved_roundtrip > max(float(nav), 1e-12) * max_daily_turnover:
        accepted = False
        action = "SKIP_TURNOVER_BUDGET"
    elif volume > available:
        accepted = False
        action = "SKIP_SELLABLE_INVENTORY"
    event = {
        **dict(candidate),
        "trigger_price": float(trigger_price),
        "entry_value": value,
        "entry_fee_est": estimate_fee(value, sell_cost, min_cost),
        "reserved_roundtrip_turnover": reserved_roundtrip,
        "action": action,
    }
    return event, accepted


def build_buyback_intent(
    entry: Mapping[str, Any],
    *,
    exit_price: float,
    buy_cost: float = 0.001,
    min_cost: float = 5.0,
) -> dict[str, Any]:
    volume = int(entry["volume"])
    value = volume * float(exit_price)
    entry_price = float(entry["trigger_price"])
    entry_value = volume * entry_price
    entry_fee = float(entry.get("entry_fee_est", estimate_fee(entry_value, 0.0025, min_cost)))
    exit_fee = estimate_fee(value, buy_cost, min_cost)
    virtual_pnl = entry_value - entry_fee - value - exit_fee
    return {
        "symbol": entry["symbol"],
        "local_symbol": entry["local_symbol"],
        "component": entry["component"],
        "direction": "sell_first",
        "score": float(entry.get("score", np.nan)),
        "meta_gate_score": float(entry.get("meta_gate_score", np.nan)),
        "volume": volume,
        "entry_price_ref": entry_price,
        "entry_value_ref": entry_value,
        "entry_fee_est": entry_fee,
        "exit_price_ref": float(exit_price),
        "exit_value_ref": value,
        "exit_fee_est": exit_fee,
        "virtual_pnl": virtual_pnl,
        "virtual_edge": virtual_pnl / max(entry_value, 1e-12),
        "action": "BUYBACK_INTENT",
        "restores_original_inventory": True,
    }
