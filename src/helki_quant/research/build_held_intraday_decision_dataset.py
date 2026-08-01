from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MULTI_LAYER = HERE.parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_minute_staging import (  # noqa: E402
    REQUIRED,
    MinuteSourceIndex,
    build_minute_source_index,
    files_for_instrument,
    read_one,
)
from held_intraday_factor_engineering import add_realtime_reproducible_factors  # noqa: E402


DECISION_MINUTES = {
    "0935": 9 * 60 + 35,
    "0945": 9 * 60 + 45,
    "1000": 10 * 60,
    "1015": 10 * 60 + 15,
}
BUYBACK_WINDOWS = {
    "1420_1430": (14 * 60 + 20, 14 * 60 + 30),
    "1445_1450": (14 * 60 + 45, 14 * 60 + 50),
    "1450_1455": (14 * 60 + 50, 14 * 60 + 55),
}
TRIGGER_WINDOW_END_MINUTE = 11 * 60
TRIGGER_DISTANCES = {
    "buy_first": 0.006,
    "sell_first": 0.0075,
}
TRIGGER_TOUCH_BUFFERS = (0.0, 0.001)
TRIGGER_BUYBACK_WINDOWS = ("1420_1430", "1445_1450")

CONTEXT_FEATURE_COLUMNS = [
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
    "group",
]

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


def normalize_inst(value: object) -> str:
    raw = str(value).upper()
    if raw.startswith("SZSE."):
        return "SZ" + raw.split(".", 1)[1]
    if raw.startswith("SHSE."):
        return "SH" + raw.split(".", 1)[1]
    return raw


def to_raw_inst(inst: str) -> str:
    return normalize_inst(inst).lower()


def minute_vwap(day: pd.DataFrame, start_minute: int, end_minute: int) -> float:
    sub = day[(day["minute_of_day"] >= start_minute) & (day["minute_of_day"] <= end_minute)]
    if sub.empty:
        return np.nan
    volume = pd.to_numeric(sub["volume"], errors="coerce").to_numpy(dtype=float)
    amount = pd.to_numeric(sub["amount"], errors="coerce").to_numpy(dtype=float)
    vol_sum = float(np.nansum(volume))
    if vol_sum <= 0:
        return np.nan
    value = float(np.nansum(amount) / (vol_sum + 1e-12))
    close_ref = float(pd.to_numeric(sub["close"], errors="coerce").dropna().iloc[-1])
    if np.isfinite(value) and np.isfinite(close_ref) and close_ref > 0:
        ratio = value / (close_ref + 1e-12)
        if ratio > 20:
            value /= 100.0
        elif ratio < 0.05:
            value *= 100.0
    return value


def price_at_or_before(day: pd.DataFrame, minute: int) -> float:
    sub = day[day["minute_of_day"] <= minute]
    if sub.empty:
        return np.nan
    return float(sub.sort_values("minute_of_day")["close"].iloc[-1])


def return_between(day: pd.DataFrame, start_minute: int, end_minute: int) -> float:
    start = price_at_or_before(day, start_minute)
    end = price_at_or_before(day, end_minute)
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
        return np.nan
    return end / start - 1.0


def visible_features(day: pd.DataFrame, decision_minute: int, mark_price: float) -> dict[str, float]:
    visible = day[day["minute_of_day"] <= decision_minute].sort_values("minute_of_day")
    if visible.empty:
        return {}
    open_price = float(visible["open"].iloc[0])
    last_price = float(visible["close"].iloc[-1])
    high = float(visible["high"].max())
    low = float(visible["low"].min())
    volume = pd.to_numeric(visible["volume"], errors="coerce").to_numpy(dtype=float)
    amount = pd.to_numeric(visible["amount"], errors="coerce").to_numpy(dtype=float)
    vol_sum = float(np.nansum(volume))
    vwap = float(np.nansum(amount) / (vol_sum + 1e-12)) if vol_sum > 0 else np.nan
    if np.isfinite(vwap) and np.isfinite(last_price) and last_price > 0:
        ratio = vwap / (last_price + 1e-12)
        if ratio > 20:
            vwap /= 100.0
        elif ratio < 0.05:
            vwap *= 100.0
    close = pd.to_numeric(visible["close"], errors="coerce")
    close_values = close.to_numpy(dtype=float)
    minute_values = visible["minute_of_day"].to_numpy(dtype=int)

    def visible_price_at(minute: int) -> float:
        pos = int(np.searchsorted(minute_values, minute, side="right") - 1)
        return float(close_values[pos]) if pos >= 0 else np.nan

    def visible_return(start_minute: int, end_minute: int) -> float:
        start_price = visible_price_at(start_minute)
        end_price = visible_price_at(end_minute)
        if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price <= 0:
            return np.nan
        return end_price / start_price - 1.0

    log_ret = np.diff(np.log(close_values + 1e-12))
    minute_axis = visible["minute_of_day"].to_numpy(dtype=float)
    log_close = np.log(close.to_numpy(dtype=float) + 1e-12)
    trend_slope = np.nan
    if len(log_close) >= 4 and np.nanstd(minute_axis) > 1e-12:
        finite = np.isfinite(log_close) & np.isfinite(minute_axis)
        if finite.sum() >= 4:
            slope = np.polyfit(minute_axis[finite], log_close[finite], 1)[0]
            trend_slope = float(slope * max(decision_minute - 570, 1))
    ret_autocorr = np.nan
    if len(log_ret) > 3 and np.nanstd(log_ret[:-1]) > 1e-12 and np.nanstd(log_ret[1:]) > 1e-12:
        ret_autocorr = float(np.corrcoef(log_ret[:-1], log_ret[1:])[0, 1])
    price_vol_corr = np.nan
    if len(log_ret) > 3:
        log_v = np.log(volume[1:] + 1.0)
        if np.nanstd(log_ret) > 1e-12 and np.nanstd(log_v) > 1e-12:
            price_vol_corr = float(np.corrcoef(log_ret, log_v)[0, 1])
    recent_5_mask = minute_values > decision_minute - 5
    recent_10_mask = minute_values > decision_minute - 10
    recent_5_volume_share = float(np.nansum(volume[recent_5_mask]) / (vol_sum + 1e-12))
    recent_10_volume_share = float(np.nansum(volume[recent_10_mask]) / (vol_sum + 1e-12))
    signed_volume = np.sign(np.r_[0.0, np.diff(close_values)]) * volume
    return {
        "visible_ret_from_open": last_price / (open_price + 1e-12) - 1.0,
        "visible_gap_vs_mark": open_price / (mark_price + 1e-12) - 1.0,
        "visible_last_vs_mark": last_price / (mark_price + 1e-12) - 1.0,
        "visible_range": high / (low + 1e-12) - 1.0,
        "visible_range_pos": (last_price - low) / (high - low + 1e-12),
        "visible_vwap_dev": last_price / (vwap + 1e-12) - 1.0 if np.isfinite(vwap) else np.nan,
        "visible_volume": vol_sum,
        "visible_amount": float(np.nansum(amount)),
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
        "visible_recent_5m_volume_share": recent_5_volume_share,
        "visible_recent_10m_volume_share": recent_10_volume_share,
        "visible_log_volume": float(np.log1p(max(vol_sum, 0.0))),
        "visible_log_amount": float(np.log1p(max(float(np.nansum(amount)), 0.0))),
    }


def add_position_dependent_features(
    frame: pd.DataFrame,
    *,
    sell_cost: float,
    buy_cost: float,
) -> pd.DataFrame:
    """Recompute features that change when the held-position context changes."""

    if frame.empty:
        return frame.copy()
    out = frame.copy()
    shares = pd.to_numeric(out["shares"], errors="coerce")
    mark_price = pd.to_numeric(out["mark_price"], errors="coerce")
    sell_price = pd.to_numeric(out["sell_price_decision"], errors="coerce")
    held_market_value = shares * mark_price
    out["held_market_value"] = held_market_value
    out["held_log_market_value"] = np.log1p(held_market_value.clip(lower=0.0))
    out["held_lot_count"] = shares / 100.0
    out["visible_volume_to_held"] = np.where(
        shares > 0,
        pd.to_numeric(out["visible_volume"], errors="coerce") / shares,
        np.nan,
    )
    out["visible_amount_to_position"] = np.where(
        held_market_value > 0,
        pd.to_numeric(out["visible_amount"], errors="coerce") / held_market_value,
        np.nan,
    )
    for fraction in (0.1, 0.2, 0.3):
        suffix = str(int(fraction * 100))
        volume = np.floor(shares * fraction / 100.0) * 100.0
        valid = (volume > 0) & (sell_price > 0)
        traded_value = volume * sell_price
        sell_fee = np.maximum(traded_value * sell_cost, 5.0)
        buy_fee = np.maximum(traded_value * buy_cost, 5.0)
        out[f"t0_volume_{suffix}pct"] = volume.where(volume > 0, 0.0)
        out[f"t0_min_fee_drag_{suffix}pct"] = np.where(
            valid,
            (sell_fee + buy_fee) / traded_value,
            np.nan,
        )
    return out


def add_cross_sectional_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    group_cols = ["trade_date", "decision_time"]
    grouped = out.groupby(group_cols, sort=False, observed=True)
    out["held_universe_size"] = grouped["instrument"].transform("size").astype(float)
    for col in CS_SOURCE_COLS:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        out[col] = values
        out[f"cs_{col}_rank"] = grouped[col].rank(method="average", pct=True)
        out[f"cs_{col}_rel"] = values - grouped[col].transform("median")
    market_specs = {
        "held_market_ret": "visible_ret_from_open",
        "held_market_gap": "visible_gap_vs_mark",
        "held_market_last_vs_mark": "visible_last_vs_mark",
        "held_market_vwap_dev": "visible_vwap_dev",
        "held_market_drawdown": "visible_drawdown_from_high",
    }
    for prefix, col in market_specs.items():
        if col not in out.columns:
            continue
        out[f"{prefix}_mean"] = grouped[col].transform("mean")
        out[f"{prefix}_median"] = grouped[col].transform("median")
        out[f"{prefix}_std"] = grouped[col].transform("std").fillna(0.0)
    positive = (pd.to_numeric(out["visible_ret_from_open"], errors="coerce") > 0.0).astype(float)
    out["held_market_positive_breadth"] = positive.groupby(
        [out[col] for col in group_cols], sort=False
    ).transform("mean")
    return out


def add_execution_aligned_labels(
    frame: pd.DataFrame,
    *,
    sell_cost: float,
    buy_cost: float,
    slippage: float,
    min_cost: float = 5.0,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    shares = pd.to_numeric(out["shares"], errors="coerce").to_numpy(dtype=float)
    sell_reference = pd.to_numeric(out["sell_price_decision"], errors="coerce").to_numpy(dtype=float)
    codes = out["instrument"].astype(str).str.upper().str[-6:]
    lots = np.where(codes.str.startswith(("688", "689")), 200.0, 100.0)
    sizing: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fraction in (0.1, 0.2, 0.3):
        name = f"{int(fraction * 100)}pct"
        volume = np.floor(shares * fraction / lots) * lots
        valid = (volume >= lots) & (volume <= shares)
        sizing[name] = (volume, valid)
    one_lot = lots.copy()
    one_lot_valid = (shares >= one_lot) & (one_lot <= shares * 0.5)
    sizing["one_lot_max50"] = (one_lot, one_lot_valid)

    for sizing_name, (volume, valid_volume) in sizing.items():
        out[f"t0_exec_volume_{sizing_name}"] = np.where(valid_volume, volume, 0.0)
        out[f"t0_exec_tradeable_{sizing_name}"] = valid_volume.astype(float)
        for buy_name in BUYBACK_WINDOWS:
            buy_col = f"buyback_{buy_name}_price"
            if buy_col not in out.columns:
                continue
            buy_reference = pd.to_numeric(out[buy_col], errors="coerce").to_numpy(dtype=float)
            valid = (
                valid_volume
                & np.isfinite(sell_reference)
                & np.isfinite(buy_reference)
                & (sell_reference > 0)
                & (buy_reference > 0)
            )
            sell_value = volume * sell_reference * (1.0 - slippage)
            buy_value = volume * buy_reference * (1.0 + slippage)
            sell_fee = np.maximum(sell_value * sell_cost, min_cost)
            buy_fee = np.maximum(buy_value * buy_cost, min_cost)
            pnl = sell_value - sell_fee - buy_value - buy_fee
            reference_value = volume * sell_reference
            edge = pnl / (reference_value + 1e-12)
            prefix = f"t0_exec_{buy_name}_{sizing_name}"
            out[f"{prefix}_pnl"] = np.where(valid, pnl, np.nan)
            out[f"{prefix}_edge"] = np.where(valid, edge, np.nan)
            out[f"{prefix}_hit"] = np.where(valid, (pnl > 0).astype(float), np.nan)

            morning_buy_value = volume * sell_reference * (1.0 + slippage)
            afternoon_sell_value = volume * buy_reference * (1.0 - slippage)
            morning_buy_fee = np.maximum(morning_buy_value * buy_cost, min_cost)
            afternoon_sell_fee = np.maximum(afternoon_sell_value * sell_cost, min_cost)
            buy_first_pnl = (
                afternoon_sell_value
                - afternoon_sell_fee
                - morning_buy_value
                - morning_buy_fee
            )
            buy_first_edge = buy_first_pnl / (reference_value + 1e-12)
            buy_first_prefix = f"t0_buy_first_{buy_name}_{sizing_name}"
            out[f"{buy_first_prefix}_pnl"] = np.where(valid, buy_first_pnl, np.nan)
            out[f"{buy_first_prefix}_edge"] = np.where(valid, buy_first_edge, np.nan)
            out[f"{buy_first_prefix}_hit"] = np.where(
                valid,
                (buy_first_pnl > 0).astype(float),
                np.nan,
            )
    return out


def trigger_label_prefix(
    direction: str,
    trigger_distance: float,
    touch_buffer: float,
    buyback_window: str = "1445_1450",
) -> str:
    if direction not in TRIGGER_DISTANCES:
        raise ValueError(f"unsupported trigger direction: {direction}")
    trigger_code = int(round(float(trigger_distance) * 10_000))
    touch_code = int(round(float(touch_buffer) * 10_000))
    return (
        f"trigger_{direction}_{trigger_code:04d}_touch{touch_code:04d}_"
        f"{buyback_window}_one_lot_max50"
    )


def add_trigger_aligned_labels(
    frame: pd.DataFrame,
    *,
    sell_cost: float,
    buy_cost: float,
    slippage: float,
    min_cost: float = 5.0,
    trigger_distances: dict[str, float] | None = None,
    touch_buffers: tuple[float, ...] = TRIGGER_TOUCH_BUFFERS,
    buyback_window: str = "1445_1450",
) -> pd.DataFrame:
    """Label the exact limit-touch execution used by the held-only runtime."""

    if frame.empty:
        return frame
    if "label_trigger_window_low" not in frame or "label_trigger_window_high" not in frame:
        raise KeyError("trigger-window extrema are required for trigger-aligned labels")
    buyback_col = f"buyback_{buyback_window}_price"
    if buyback_col not in frame:
        raise KeyError(f"missing buyback price: {buyback_col}")
    out = frame.copy()
    distances = trigger_distances or TRIGGER_DISTANCES
    shares = pd.to_numeric(out["shares"], errors="coerce").to_numpy(dtype=float)
    decision_price = pd.to_numeric(
        out["sell_price_decision"], errors="coerce"
    ).to_numpy(dtype=float)
    exit_reference = pd.to_numeric(out[buyback_col], errors="coerce").to_numpy(dtype=float)
    window_low = pd.to_numeric(
        out["label_trigger_window_low"], errors="coerce"
    ).to_numpy(dtype=float)
    window_high = pd.to_numeric(
        out["label_trigger_window_high"], errors="coerce"
    ).to_numpy(dtype=float)
    codes = out["instrument"].astype(str).str.upper().str[-6:]
    volume = np.where(codes.str.startswith(("688", "689")), 200.0, 100.0)
    valid_volume = (shares >= volume) & (volume <= shares * 0.5)

    for direction, trigger_distance in distances.items():
        if direction not in {"buy_first", "sell_first"}:
            raise ValueError(f"unsupported trigger direction: {direction}")
        entry_price = decision_price * (
            1.0 - float(trigger_distance)
            if direction == "buy_first"
            else 1.0 + float(trigger_distance)
        )
        valid = (
            valid_volume
            & np.isfinite(decision_price)
            & np.isfinite(entry_price)
            & np.isfinite(exit_reference)
            & np.isfinite(window_low)
            & np.isfinite(window_high)
            & (decision_price > 0)
            & (entry_price > 0)
            & (exit_reference > 0)
        )
        entry_value = volume * entry_price
        if direction == "buy_first":
            exit_price = exit_reference * (1.0 - slippage)
            exit_value = volume * exit_price
            entry_fee = np.maximum(entry_value * buy_cost, min_cost)
            exit_fee = np.maximum(exit_value * sell_cost, min_cost)
            conditional_pnl = exit_value - exit_fee - entry_value - entry_fee
        else:
            exit_price = exit_reference * (1.0 + slippage)
            exit_value = volume * exit_price
            entry_fee = np.maximum(entry_value * sell_cost, min_cost)
            exit_fee = np.maximum(exit_value * buy_cost, min_cost)
            conditional_pnl = entry_value - entry_fee - exit_value - exit_fee
        conditional_edge = conditional_pnl / (entry_value + 1e-12)

        for touch_buffer in touch_buffers:
            if direction == "buy_first":
                touched = window_low <= entry_price * (1.0 - float(touch_buffer))
            else:
                touched = window_high >= entry_price * (1.0 + float(touch_buffer))
            touched = valid & touched
            prefix = trigger_label_prefix(
                direction,
                float(trigger_distance),
                float(touch_buffer),
                buyback_window,
            )
            out[f"{prefix}_entry_price"] = np.where(valid, entry_price, np.nan)
            out[f"{prefix}_touched"] = np.where(valid, touched.astype(float), np.nan)
            out[f"{prefix}_conditional_pnl"] = np.where(touched, conditional_pnl, np.nan)
            out[f"{prefix}_conditional_edge"] = np.where(touched, conditional_edge, np.nan)
            out[f"{prefix}_conditional_hit"] = np.where(
                touched,
                (conditional_pnl > 0).astype(float),
                np.nan,
            )
            realized_pnl = np.where(touched, conditional_pnl, 0.0)
            realized_edge = np.where(touched, conditional_edge, 0.0)
            out[f"{prefix}_realized_pnl"] = np.where(valid, realized_pnl, np.nan)
            out[f"{prefix}_realized_edge"] = np.where(valid, realized_edge, np.nan)
            out[f"{prefix}_realized_hit"] = np.where(
                valid,
                (touched & (conditional_pnl > 0)).astype(float),
                np.nan,
            )
    return out


def edge_from_sell_buy(
    sell_price: float,
    buy_price: float,
    *,
    sell_cost: float,
    buy_cost: float,
    slippage: float,
) -> float:
    if not np.isfinite(sell_price) or not np.isfinite(buy_price) or sell_price <= 0 or buy_price <= 0:
        return np.nan
    sell_net = sell_price * (1.0 - sell_cost - slippage)
    buy_gross = buy_price * (1.0 + buy_cost + slippage)
    return sell_net / (buy_gross + 1e-12) - 1.0


def read_stage_symbol(stage_dir: Path | None, raw_inst: str) -> pd.DataFrame | None:
    if stage_dir is None:
        return None
    candidates = [
        stage_dir / f"{raw_inst}.csv",
        stage_dir / f"{raw_inst}_1m.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None
    frame = pd.read_csv(path, parse_dates=["date"])
    missing = [col for col in REQUIRED if col not in frame.columns]
    if missing:
        raise ValueError(f"missing staged columns in {path}: {missing}")
    for col in REQUIRED[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame[REQUIRED]


def load_minute_for_symbol(
    raw_inst: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    stage_dir: Path | None,
    stage_only: bool,
    source_index: MinuteSourceIndex | None = None,
) -> pd.DataFrame:
    parts = []
    staged = read_stage_symbol(stage_dir, raw_inst)
    if staged is not None:
        sources = [staged]
    elif stage_only:
        sources = []
    else:
        sources = [
            read_one(source)
            for source in files_for_instrument(
                raw_inst,
                source_index=source_index,
                start=start,
                end=end,
            )
        ]
    for frame in sources:
        frame["trade_date"] = frame["date"].dt.normalize()
        frame = frame[frame["trade_date"].between(start, end)].copy()
        if not frame.empty:
            parts.append(frame)
    if not parts:
        return pd.DataFrame()
    out = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    out["minute_of_day"] = out["date"].dt.hour * 60 + out["date"].dt.minute
    out["trade_date"] = out["date"].dt.normalize()
    return out


def build_dataset(
    held_context_path: Path,
    output_csv: Path,
    output_json: Path,
    *,
    start: str | None,
    end: str | None,
    max_instruments: int,
    stage_dir: Path | None,
    stage_only: bool,
    sell_cost: float,
    buy_cost: float,
    slippage: float,
) -> dict:
    held = pd.read_csv(held_context_path, parse_dates=["datetime"])
    held["datetime"] = held["datetime"].dt.normalize()
    if "trade_date" not in held.columns:
        raise ValueError(
            "held context missing execution trade_date; refusing same-day fallback because it leaks close-time holdings into intraday features"
        )
    held["trade_date"] = pd.to_datetime(held["trade_date"]).dt.normalize()
    invalid_execution_date = held["trade_date"].notna() & (held["trade_date"] <= held["datetime"])
    if invalid_execution_date.any():
        raise ValueError("held context execution trade_date must be strictly after holding datetime")
    held["instrument"] = held["instrument"].map(normalize_inst)
    if start:
        held = held[held["trade_date"] >= pd.Timestamp(start).normalize()]
    if end:
        held = held[held["trade_date"] <= pd.Timestamp(end).normalize()]
    held = held.dropna(subset=["trade_date", "mark_price"]).copy()
    held_by_inst = {inst: part.copy() for inst, part in held.groupby("instrument", sort=True)}
    if max_instruments > 0:
        held_by_inst = dict(list(held_by_inst.items())[:max_instruments])
    if not held_by_inst:
        raise ValueError("no held rows after date filtering")
    min_date = held["trade_date"].min()
    max_date = held["trade_date"].max()
    source_index: MinuteSourceIndex | None = None
    if not stage_only:
        missing_stage = [
            inst
            for inst in held_by_inst
            if stage_dir is None or not (stage_dir / f"{to_raw_inst(inst)}.csv").exists()
        ]
        if missing_stage:
            print(
                f"[held intraday] indexing raw minute sources for missing_stage={len(missing_stage)}",
                flush=True,
            )
            source_index = build_minute_source_index()
            print(
                f"[held intraday] minute source index symbols={len(source_index)}",
                flush=True,
            )

    rows = []
    details = []
    context_cols = CONTEXT_FEATURE_COLUMNS
    for pos, (inst, part) in enumerate(held_by_inst.items(), start=1):
        raw_inst = to_raw_inst(inst)
        minute = load_minute_for_symbol(
            raw_inst,
            min_date,
            max_date,
            stage_dir,
            stage_only,
            source_index,
        )
        if minute.empty:
            details.append({"instrument": inst, "status": "missing_minute"})
            continue
        day_map = {date: day for date, day in minute.groupby("trade_date", sort=True)}
        inst_rows = 0
        for held_row in part.itertuples(index=False):
            trade_date = held_row.trade_date
            day = day_map.get(trade_date)
            if day is None or day.empty:
                continue
            for decision_name, decision_minute in DECISION_MINUTES.items():
                sell_price = price_at_or_before(day, decision_minute)
                feat = visible_features(day, decision_minute, float(held_row.mark_price))
                if not feat:
                    continue
                trigger_window = day[
                    (day["minute_of_day"] > decision_minute)
                    & (day["minute_of_day"] <= TRIGGER_WINDOW_END_MINUTE)
                ]
                row = {
                    "datetime": held_row.datetime.strftime("%Y-%m-%d"),
                    "trade_date": trade_date.strftime("%Y-%m-%d"),
                    "instrument": inst,
                    "decision_time": decision_name,
                    "decision_minute": decision_minute,
                    "sell_price_decision": sell_price,
                    # These are label-only future fields. They are intentionally
                    # absent from every live feature whitelist.
                    "label_trigger_window_low": float(
                        pd.to_numeric(trigger_window["low"], errors="coerce").min()
                    )
                    if not trigger_window.empty
                    else np.nan,
                    "label_trigger_window_high": float(
                        pd.to_numeric(trigger_window["high"], errors="coerce").max()
                    )
                    if not trigger_window.empty
                    else np.nan,
                    "label_trigger_window_minutes": int(len(trigger_window)),
                }
                for col in context_cols:
                    row[col] = getattr(held_row, col, np.nan)
                row.update(feat)
                best_edge = np.nan
                best_window = None
                for buy_name, (start_minute, end_minute) in BUYBACK_WINDOWS.items():
                    buy_price = minute_vwap(day, start_minute, end_minute)
                    edge = edge_from_sell_buy(
                        sell_price,
                        buy_price,
                        sell_cost=sell_cost,
                        buy_cost=buy_cost,
                        slippage=slippage,
                    )
                    row[f"buyback_{buy_name}_price"] = buy_price
                    row[f"t0_edge_{buy_name}"] = edge
                    row[f"t0_hit_{buy_name}"] = float(edge > 0) if np.isfinite(edge) else np.nan
                    if np.isfinite(edge) and (not np.isfinite(best_edge) or edge > best_edge):
                        best_edge = edge
                        best_window = buy_name
                row["t0_best_edge"] = best_edge
                row["t0_best_hit"] = float(best_edge > 0) if np.isfinite(best_edge) else np.nan
                row["t0_best_strong_hit"] = float(best_edge > 0.002) if np.isfinite(best_edge) else np.nan
                row["t0_best_buyback_window"] = best_window
                rows.append(row)
                inst_rows += 1
        details.append({"instrument": inst, "status": "ok", "rows": inst_rows})
        if pos % 25 == 0:
            print(f"[held intraday] {pos}/{len(held_by_inst)} {inst} rows={inst_rows}", flush=True)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_position_dependent_features(
            out.replace([np.inf, -np.inf], np.nan),
            sell_cost=sell_cost,
            buy_cost=buy_cost,
        )
        out = add_execution_aligned_labels(
            out,
            sell_cost=sell_cost,
            buy_cost=buy_cost,
            slippage=slippage,
        )
        for buyback_window in TRIGGER_BUYBACK_WINDOWS:
            out = add_trigger_aligned_labels(
                out,
                sell_cost=sell_cost,
                buy_cost=buy_cost,
                slippage=slippage,
                buyback_window=buyback_window,
            )
        out = add_realtime_reproducible_factors(out)
        out = add_cross_sectional_features(out).sort_values(
            ["trade_date", "instrument", "decision_minute"]
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    trigger_summary = {}
    if not out.empty:
        decision_1000 = out[
            out["decision_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
            == "1000"
        ]
        for buyback_window in TRIGGER_BUYBACK_WINDOWS:
            for direction, distance in TRIGGER_DISTANCES.items():
                for touch_buffer in TRIGGER_TOUCH_BUFFERS:
                    prefix = trigger_label_prefix(
                        direction,
                        distance,
                        touch_buffer,
                        buyback_window,
                    )
                    valid = decision_1000[f"{prefix}_touched"].notna()
                    touched = decision_1000.loc[valid, f"{prefix}_touched"] > 0.5
                    trigger_summary[prefix] = {
                        "valid_rows": int(valid.sum()),
                        "touch_ratio": float(touched.mean()) if valid.any() else None,
                        "conditional_hit_ratio": float(
                            decision_1000.loc[
                                touched.index[touched], f"{prefix}_conditional_hit"
                            ].mean()
                        )
                        if touched.any()
                        else None,
                        "realized_edge_mean": float(
                            decision_1000.loc[valid, f"{prefix}_realized_edge"].mean()
                        )
                        if valid.any()
                        else None,
                    }
    report = {
        "status": "held_intraday_decision_dataset_built",
        "held_context_path": str(held_context_path.resolve()),
        "stage_dir": str(stage_dir.resolve()) if stage_dir else None,
        "stage_only": stage_only,
        "output_csv": str(output_csv.resolve()),
        "start": str(min_date.date()),
        "end": str(max_date.date()),
        "decision_times": DECISION_MINUTES,
        "buyback_windows": BUYBACK_WINDOWS,
        "cross_sectional_sources": CS_SOURCE_COLS,
        "execution_aligned_sizing": ["10pct", "20pct", "30pct", "one_lot_max50"],
        "trigger_aligned_labels": {
            "entry_window_end_minute": TRIGGER_WINDOW_END_MINUTE,
            "trigger_distances": TRIGGER_DISTANCES,
            "touch_buffers": list(TRIGGER_TOUCH_BUFFERS),
            "buyback_windows": list(TRIGGER_BUYBACK_WINDOWS),
            "label_only_future_columns": [
                "label_trigger_window_low",
                "label_trigger_window_high",
                "label_trigger_window_minutes",
            ],
            "summary_1000": trigger_summary,
        },
        "rows": int(len(out)),
        "dates": int(out["trade_date"].nunique()) if not out.empty else 0,
        "instruments": int(out["instrument"].nunique()) if not out.empty else 0,
        "edge_mean": float(out["t0_best_edge"].mean()) if not out.empty else None,
        "best_hit_ratio": float(out["t0_best_hit"].mean()) if not out.empty else None,
        "best_strong_hit_ratio": float(out["t0_best_strong_hit"].mean()) if not out.empty else None,
        "details": details,
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-context", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--stage-dir", default=None)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--sell-cost", type=float, default=0.0025)
    parser.add_argument("--buy-cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--max-instruments", type=int, default=0)
    args = parser.parse_args()
    report = build_dataset(
        Path(args.held_context).resolve(),
        Path(args.output_csv).resolve(),
        Path(args.output_json).resolve(),
        start=args.start,
        end=args.end,
        max_instruments=args.max_instruments,
        stage_dir=Path(args.stage_dir).resolve() if args.stage_dir else None,
        stage_only=args.stage_only,
        sell_cost=args.sell_cost,
        buy_cost=args.buy_cost,
        slippage=args.slippage,
    )
    print(
        "[held intraday] "
        f"rows={report['rows']} dates={report['dates']} instruments={report['instruments']} "
        f"edge_mean={report['edge_mean']} hit={report['best_hit_ratio']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
