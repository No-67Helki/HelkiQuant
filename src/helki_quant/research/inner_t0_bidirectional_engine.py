from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from held_intraday_live_features import minimum_lot


def percentile_score(raw_score: float, sorted_calibration: np.ndarray) -> float:
    values = np.asarray(sorted_calibration, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(raw_score) or len(values) == 0:
        return np.nan
    return float(np.searchsorted(values, raw_score, side="right") / len(values))


def select_bidirectional_candidates(
    scores: pd.DataFrame,
    *,
    buy_threshold: float = 0.925,
    sell_threshold: float = 0.975,
    buy_top_n: int = 2,
    sell_top_n: int = 1,
    max_symbols: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "symbol",
        "local_symbol",
        "held_volume",
        "available_volume",
        "decision_price",
        "buy_score",
        "sell_score",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"missing score columns: {missing}")
    frame = scores.copy()
    frame["lot"] = frame["symbol"].map(minimum_lot).astype(int)
    frame["eligible_one_lot"] = (
        (pd.to_numeric(frame["held_volume"], errors="coerce") >= 2 * frame["lot"])
        & (pd.to_numeric(frame["available_volume"], errors="coerce") >= frame["lot"])
        & (pd.to_numeric(frame["decision_price"], errors="coerce") > 0)
    )
    buy = frame[
        frame["eligible_one_lot"]
        & (pd.to_numeric(frame["buy_score"], errors="coerce") >= buy_threshold)
    ].nlargest(buy_top_n, "buy_score")
    sell = frame[
        frame["eligible_one_lot"]
        & (pd.to_numeric(frame["sell_score"], errors="coerce") >= sell_threshold)
    ].nlargest(sell_top_n, "sell_score")
    conflicts = set(buy["symbol"]) & set(sell["symbol"])
    buy = buy[~buy["symbol"].isin(conflicts)].copy()
    sell = sell[~sell["symbol"].isin(conflicts)].copy()
    buy["direction"] = "buy_first"
    buy["score"] = buy["buy_score"]
    sell["direction"] = "sell_first"
    sell["score"] = sell["sell_score"]
    selected = pd.concat([buy, sell], ignore_index=True)
    selected = selected.sort_values(["score", "symbol"], ascending=[False, True]).head(max_symbols)
    selected["volume"] = selected["lot"].astype(int)
    selected["entry_limit"] = np.where(
        selected["direction"].eq("buy_first"),
        selected["decision_price"] * (1.0 - 0.006),
        selected["decision_price"] * (1.0 + 0.0075),
    )
    conflict_rows = frame[frame["symbol"].isin(conflicts)].copy()
    return selected.reset_index(drop=True), conflict_rows.reset_index(drop=True)


def trigger_reached(direction: str, last_price: float, entry_limit: float) -> bool:
    if not np.isfinite(last_price) or not np.isfinite(entry_limit) or last_price <= 0 or entry_limit <= 0:
        return False
    if direction == "buy_first":
        return bool(last_price <= entry_limit)
    if direction == "sell_first":
        return bool(last_price >= entry_limit)
    raise ValueError(f"unsupported direction: {direction}")


def estimate_fee(value: float, rate: float, min_cost: float = 5.0) -> float:
    return max(value * rate, min_cost) if value > 0 else 0.0


def build_entry_intent(
    candidate: dict[str, Any],
    *,
    trigger_price: float,
    nav: float,
    cash_available: float,
    turnover_used: float,
    buy_cash_reserved: float,
    max_daily_turnover: float = 0.03,
    buy_cost: float = 0.001,
    sell_cost: float = 0.0025,
    min_cost: float = 5.0,
) -> tuple[dict[str, Any], bool]:
    direction = str(candidate["direction"])
    volume = int(candidate["volume"])
    value = volume * float(trigger_price)
    entry_rate = buy_cost if direction == "buy_first" else sell_cost
    entry_fee = estimate_fee(value, entry_rate, min_cost)
    reserved_roundtrip = 2.0 * value
    reason = "TRIGGERED"
    accepted = True
    if turnover_used + reserved_roundtrip > max(nav, 1e-12) * max_daily_turnover:
        reason = "SKIP_TURNOVER_BUDGET"
        accepted = False
    elif direction == "buy_first" and buy_cash_reserved + value + entry_fee > cash_available:
        reason = "SKIP_CASH_BUDGET"
        accepted = False
    elif direction == "sell_first" and volume > int(candidate["available_volume"]):
        reason = "SKIP_SELLABLE_INVENTORY"
        accepted = False
    event = {
        **candidate,
        "trigger_price": float(trigger_price),
        "entry_value": value,
        "entry_fee_est": entry_fee,
        "reserved_roundtrip_turnover": reserved_roundtrip,
        "action": reason,
    }
    return event, accepted


def build_exit_intent(
    entry: dict[str, Any],
    *,
    exit_price: float,
    buy_cost: float = 0.001,
    sell_cost: float = 0.0025,
    min_cost: float = 5.0,
) -> dict[str, Any]:
    direction = str(entry["direction"])
    volume = int(entry["volume"])
    value = volume * float(exit_price)
    if direction == "buy_first":
        action = "SELL_OLD_INVENTORY_INTENT"
        exit_rate = sell_cost
    elif direction == "sell_first":
        action = "BUYBACK_INTENT"
        exit_rate = buy_cost
    else:
        raise ValueError(f"unsupported direction: {direction}")
    return {
        "symbol": entry["symbol"],
        "local_symbol": entry["local_symbol"],
        "direction": direction,
        "volume": volume,
        "exit_price_ref": float(exit_price),
        "exit_value_ref": value,
        "exit_fee_est": estimate_fee(value, exit_rate, min_cost),
        "action": action,
        "restores_original_inventory": True,
    }
