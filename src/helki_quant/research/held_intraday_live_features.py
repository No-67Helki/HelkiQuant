from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from held_intraday_factor_engineering import add_realtime_reproducible_factors


DECISION_MINUTES = {"1000": 10 * 60}
CS_SOURCE_COLS = [
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


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def minimum_lot(symbol: object, default_lot: int = 100) -> int:
    code = str(symbol).strip().upper()[-6:]
    return max(200, default_lot) if code.startswith(("688", "689")) else default_lot


def normalize_gm_minute_bars(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize GmQuant minute volume to the lot-like unit used by research data."""
    required = {"eob", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return pd.DataFrame(), {
            "valid": False,
            "reason": f"missing_columns:{','.join(missing)}",
            "volume_scale": None,
        }
    out = frame.copy()
    out["eob"] = pd.to_datetime(out["eob"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["eob", "open", "high", "low", "close"])
    positive = out[(out["volume"] > 0) & (out["amount"] > 0) & (out["close"] > 0)]
    if positive.empty:
        return pd.DataFrame(), {
            "valid": False,
            "reason": "no_positive_volume_amount",
            "volume_scale": None,
        }
    amount_per_volume_price = (
        positive["amount"].sum()
        / (positive["volume"].sum() * positive["close"].iloc[-1] + 1e-12)
    )
    if 0.2 <= amount_per_volume_price <= 5.0:
        volume_scale = 0.01
        unit_source = "gm_shares_to_research_lots"
    elif 20.0 <= amount_per_volume_price <= 200.0:
        volume_scale = 1.0
        unit_source = "already_research_lots"
    else:
        return pd.DataFrame(), {
            "valid": False,
            "reason": "ambiguous_volume_unit",
            "amount_per_volume_price": float(amount_per_volume_price),
            "volume_scale": None,
        }
    out["volume"] = out["volume"] * volume_scale
    return out.sort_values("eob").reset_index(drop=True), {
        "valid": True,
        "reason": "ok",
        "amount_per_volume_price_raw": float(amount_per_volume_price),
        "volume_scale": volume_scale,
        "unit_source": unit_source,
    }


def current_session_bars(
    frame: pd.DataFrame,
    trade_date: object,
    decision_minute: int,
) -> tuple[pd.DataFrame, float]:
    if frame.empty:
        return pd.DataFrame(), np.nan
    wanted = pd.Timestamp(trade_date).normalize()
    dates = frame["eob"].dt.normalize()
    previous = frame[dates < wanted]
    previous_close = finite_float(previous["close"].iloc[-1]) if not previous.empty else np.nan
    current = frame[dates == wanted].copy()
    current["minute_of_day"] = current["eob"].dt.hour * 60 + current["eob"].dt.minute
    current = current[current["minute_of_day"] <= int(decision_minute)].copy()
    return current.sort_values("minute_of_day").reset_index(drop=True), previous_close


def visible_features(day: pd.DataFrame, decision_minute: int, mark_price: float) -> dict[str, float]:
    visible = day[day["minute_of_day"] <= decision_minute].sort_values("minute_of_day")
    if visible.empty or not np.isfinite(mark_price) or mark_price <= 0:
        return {}
    open_price = finite_float(visible["open"].iloc[0])
    last_price = finite_float(visible["close"].iloc[-1])
    high = finite_float(visible["high"].max())
    low = finite_float(visible["low"].min())
    if not all(np.isfinite(value) and value > 0 for value in (open_price, last_price, high, low)):
        return {}
    volume = pd.to_numeric(visible["volume"], errors="coerce").to_numpy(dtype=float)
    amount = pd.to_numeric(visible["amount"], errors="coerce").to_numpy(dtype=float)
    vol_sum = float(np.nansum(volume))
    amount_sum = float(np.nansum(amount))
    vwap = amount_sum / (vol_sum + 1e-12) if vol_sum > 0 else np.nan
    if np.isfinite(vwap) and last_price > 0:
        ratio = vwap / (last_price + 1e-12)
        if ratio > 20:
            vwap /= 100.0
        elif ratio < 0.05:
            vwap *= 100.0
    close = pd.to_numeric(visible["close"], errors="coerce")
    close_values = close.to_numpy(dtype=float)
    minute_values = visible["minute_of_day"].to_numpy(dtype=int)

    def price_at(minute: int) -> float:
        pos = int(np.searchsorted(minute_values, minute, side="right") - 1)
        return float(close_values[pos]) if pos >= 0 else np.nan

    def visible_return(start_minute: int, end_minute: int) -> float:
        start_price = price_at(start_minute)
        end_price = price_at(end_minute)
        if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price <= 0:
            return np.nan
        return end_price / start_price - 1.0

    log_ret = np.diff(np.log(close_values + 1e-12))
    minute_axis = minute_values.astype(float)
    log_close = np.log(close_values + 1e-12)
    trend_slope = np.nan
    finite = np.isfinite(log_close) & np.isfinite(minute_axis)
    if finite.sum() >= 4 and np.nanstd(minute_axis[finite]) > 1e-12:
        slope = np.polyfit(minute_axis[finite], log_close[finite], 1)[0]
        trend_slope = float(slope * max(decision_minute - 570, 1))
    ret_autocorr = np.nan
    if len(log_ret) > 3 and np.nanstd(log_ret[:-1]) > 1e-12 and np.nanstd(log_ret[1:]) > 1e-12:
        ret_autocorr = float(np.corrcoef(log_ret[:-1], log_ret[1:])[0, 1])
    price_vol_corr = np.nan
    if len(log_ret) > 3:
        log_volume = np.log(volume[1:] + 1.0)
        if np.nanstd(log_ret) > 1e-12 and np.nanstd(log_volume) > 1e-12:
            price_vol_corr = float(np.corrcoef(log_ret, log_volume)[0, 1])
    recent_5_mask = minute_values > decision_minute - 5
    recent_10_mask = minute_values > decision_minute - 10
    signed_volume = np.sign(np.r_[0.0, np.diff(close_values)]) * volume
    return {
        "visible_ret_from_open": last_price / (open_price + 1e-12) - 1.0,
        "visible_gap_vs_mark": open_price / (mark_price + 1e-12) - 1.0,
        "visible_last_vs_mark": last_price / (mark_price + 1e-12) - 1.0,
        "visible_range": high / (low + 1e-12) - 1.0,
        "visible_range_pos": (last_price - low) / (high - low + 1e-12),
        "visible_vwap_dev": last_price / (vwap + 1e-12) - 1.0 if np.isfinite(vwap) else np.nan,
        "visible_volume": vol_sum,
        "visible_amount": amount_sum,
        "visible_minute_vol": float(np.nanstd(log_ret) * np.sqrt(240)) if len(log_ret) > 3 else np.nan,
        "visible_ret_autocorr": ret_autocorr,
        "visible_price_vol_corr": price_vol_corr,
        "visible_first_5m_ret": visible_return(570, min(decision_minute, 575)),
        "visible_first_10m_ret": visible_return(570, min(decision_minute, 580)),
        "visible_first_15m_ret": visible_return(570, min(decision_minute, 585)),
        "visible_first_30m_ret": visible_return(570, min(decision_minute, 600)),
        "visible_recent_5m_ret": visible_return(decision_minute - 5, decision_minute),
        "visible_recent_10m_ret": visible_return(decision_minute - 10, decision_minute),
        "visible_recent_15m_ret": visible_return(decision_minute - 15, decision_minute),
        "visible_momentum_accel_5m": (
            visible_return(decision_minute - 5, decision_minute)
            - visible_return(decision_minute - 10, decision_minute - 5)
        ),
        "visible_trend_slope": trend_slope,
        "visible_drawdown_from_high": last_price / (high + 1e-12) - 1.0,
        "visible_rebound_from_low": last_price / (low + 1e-12) - 1.0,
        "visible_up_minute_ratio": float((log_ret > 0).mean()) if len(log_ret) else np.nan,
        "visible_signed_volume_ratio": float(np.nansum(signed_volume) / (vol_sum + 1e-12)),
        "visible_recent_5m_volume_share": float(np.nansum(volume[recent_5_mask]) / (vol_sum + 1e-12)),
        "visible_recent_10m_volume_share": float(np.nansum(volume[recent_10_mask]) / (vol_sum + 1e-12)),
        "visible_log_volume": float(np.log1p(max(vol_sum, 0.0))),
        "visible_log_amount": float(np.log1p(max(amount_sum, 0.0))),
        "sell_price_decision": last_price,
    }


def build_position_features(
    position: dict[str, Any],
    target: dict[str, Any],
    visible: dict[str, float],
    *,
    previous_close: float,
    nav: float,
    buy_cost: float = 0.001,
    sell_cost: float = 0.0025,
    min_cost: float = 5.0,
) -> dict[str, float]:
    shares = finite_float(position.get("volume"), 0.0)
    cost_price = finite_float(position.get("cost_price"), np.nan)
    target_weight = finite_float(target.get("target_weight"), 0.0)
    target_shares = finite_float(target.get("target_shares"), 0.0)
    rank = finite_float(target.get("rank"), np.nan)
    middle = finite_float(target.get("middle"), np.nan)
    decision_price = finite_float(visible.get("sell_price_decision"), np.nan)
    weight = shares * previous_close / max(nav, 1e-12)
    market_value = shares * previous_close
    row = {
        "instrument": str(position.get("local") or position.get("symbol") or "").upper(),
        "group": str(target.get("group") or ""),
        "shares": shares,
        "mark_price": previous_close,
        "weight": weight,
        "target_weight": target_weight,
        "target_shares": target_shares,
        "held_unrealized_ret_approx": (
            previous_close / (cost_price + 1e-12) - 1.0
            if np.isfinite(cost_price) and cost_price > 0
            else 0.0
        ),
        "held_weight_gap_to_target": target_weight - weight,
        "held_abs_weight_gap_to_target": abs(target_weight - weight),
        "held_share_gap_to_target": target_shares - shares,
        "held_target_share_ratio": target_shares / (shares + 1e-12),
        "target_missing": float(not bool(target)),
        "rank": rank,
        "middle": middle,
        "decision_minute": 600.0,
        "held_market_value": market_value,
        "held_log_market_value": float(np.log1p(max(market_value, 0.0))),
        "held_lot_count": shares / 100.0,
        "visible_volume_to_held": visible["visible_volume"] / shares if shares > 0 else np.nan,
        "visible_amount_to_position": visible["visible_amount"] / market_value if market_value > 0 else np.nan,
    }
    row.update(visible)
    for fraction in (0.1, 0.2, 0.3):
        suffix = f"{int(fraction * 100)}pct"
        legacy_volume = np.floor(shares * fraction / 100.0) * 100.0
        value = legacy_volume * decision_price if legacy_volume > 0 and decision_price > 0 else 0.0
        fees = (max(value * sell_cost, min_cost) + max(value * buy_cost, min_cost)) if value > 0 else 0.0
        row[f"t0_volume_{suffix}"] = legacy_volume
        row[f"t0_min_fee_drag_{suffix}"] = fees / value if value > 0 else np.nan
    lot = float(minimum_lot(position.get("symbol") or position.get("local")))
    for fraction in (0.1, 0.2, 0.3):
        suffix = f"{int(fraction * 100)}pct"
        volume = np.floor(shares * fraction / lot) * lot
        valid = volume >= lot and volume <= shares
        row[f"t0_exec_volume_{suffix}"] = volume if valid else 0.0
        row[f"t0_exec_tradeable_{suffix}"] = float(valid)
    valid_one_lot = shares >= lot and lot <= shares * 0.5
    row["t0_exec_volume_one_lot_max50"] = lot if valid_one_lot else 0.0
    row["t0_exec_tradeable_one_lot_max50"] = float(valid_one_lot)
    return row


def add_held_cross_sectional_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = add_realtime_reproducible_factors(frame).copy()
    derived: dict[str, pd.Series | np.ndarray] = {
        "held_universe_size": np.full(len(out), float(len(out))),
    }
    for col in CS_SOURCE_COLS:
        values = pd.to_numeric(out[col], errors="coerce")
        out[col] = values
        derived[f"cs_{col}_rank"] = values.rank(method="average", pct=True)
        derived[f"cs_{col}_rel"] = values - values.median()
    market_specs = {
        "held_market_ret": "visible_ret_from_open",
        "held_market_gap": "visible_gap_vs_mark",
        "held_market_last_vs_mark": "visible_last_vs_mark",
        "held_market_vwap_dev": "visible_vwap_dev",
        "held_market_drawdown": "visible_drawdown_from_high",
    }
    for prefix, col in market_specs.items():
        values = pd.to_numeric(out[col], errors="coerce")
        derived[f"{prefix}_mean"] = np.full(len(out), values.mean())
        derived[f"{prefix}_median"] = np.full(len(out), values.median())
        derived[f"{prefix}_std"] = np.full(
            len(out),
            values.std() if len(values) > 1 else 0.0,
        )
    positive_breadth = (
        pd.to_numeric(out["visible_ret_from_open"], errors="coerce") > 0.0
    ).astype(float).mean()
    derived["held_market_positive_breadth"] = np.full(
        len(out),
        positive_breadth,
    )
    return pd.concat(
        [out, pd.DataFrame(derived, index=out.index)],
        axis=1,
    )
