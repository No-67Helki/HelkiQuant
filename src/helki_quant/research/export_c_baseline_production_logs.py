from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from capital_aware_allocation import ALLOCATION_MODES, allocate_equal_weight_lots
from concentration_constraints import (
    ConcentrationRules,
    concentration_snapshot,
    groups_on_date,
    load_group_metadata,
    select_with_group_cap,
)
from evaluate_daily_topk_grid import load_middle_predictions
from minute_mapped_topk_replay import (
    MappedProfile,
    MappedReplayConfig,
    min_buy_lot,
    order_delta_key,
    prepare_daily_frame,
    round_lot,
    safe_price,
)
from portfolio_experiments import BASE_COST, STRESS_COST, CostScenario
from universe import load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_MIDDLE = HERE / "outputs" / "oof_combined" / "middle_oof_pit_de2_srfs_es.csv"
DEFAULT_GROUP_METADATA = DATA / "industry_theme_pit.csv"
DEFAULT_WINDOWS = DATA / "_research_topk_minute_windows_2025_2026.csv"
DEFAULT_FORBIDDEN = DATA / "forbidden_st_symbols_20260605.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def artifact(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"provenance input not found: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def gm_to_local_symbol(symbol: object) -> str:
    text = str(symbol).upper()
    if "." not in text:
        return text
    exchange, code = text.split(".", 1)
    prefix = "SH" if exchange in {"SHSE", "SH"} else "SZ"
    return f"{prefix}{code}"


def side_name_from_gm(value: object) -> str:
    try:
        side = int(value)
    except Exception:
        return str(value).upper()
    if side == 1:
        return "BUY"
    if side == 2:
        return "SELL"
    return f"UNKNOWN_{side}"


def load_blocked_orders(gm_audit_dir: Path | None) -> dict[tuple[str, str, str], str]:
    if gm_audit_dir is None:
        return {}
    status_path = gm_audit_dir / "order_status.csv"
    if not status_path.exists():
        raise FileNotFoundError(f"missing Gm order status file: {status_path}")
    frame = pd.read_csv(status_path)
    if frame.empty:
        return {}
    rejected = frame[frame["status_name"].astype(str).eq("Rejected")].copy()
    blocked: dict[tuple[str, str, str], str] = {}
    for row in rejected.itertuples(index=False):
        trade_date = str(pd.Timestamp(getattr(row, "event_date")).date())
        instrument = gm_to_local_symbol(getattr(row, "symbol"))
        side = side_name_from_gm(getattr(row, "side"))
        reason = str(getattr(row, "ord_rej_reason_detail", ""))
        blocked[(trade_date, instrument, side)] = reason
    return blocked


def stale_sell_block_reason(
    exact_reason: str | None,
    side: str,
    day_no: int,
    start_days: list[int],
    persist_blocked_sells: bool,
    stale_sell_block_days: int,
) -> str | None:
    if exact_reason or side != "SELL":
        return exact_reason
    if persist_blocked_sells and any(start_day <= day_no for start_day in start_days):
        return "persistent synthetic stale-holding sell block"
    if stale_sell_block_days > 0 and any(
        start_day <= day_no < start_day + stale_sell_block_days
        for start_day in start_days
    ):
        return (
            "finite synthetic stale-holding sell block "
            f"({stale_sell_block_days} trading days)"
        )
    return None


def parse_profile_specs(text: str) -> list[MappedProfile]:
    profiles: list[MappedProfile] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 6:
            raise ValueError(
                "--profile-specs entries must be name,top_k,rebalance,risk,min_amount,industry_cap"
            )
        profiles.append(
            MappedProfile(
                parts[0],
                int(parts[1]),
                int(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
        )
    return profiles


def load_forbidden_instruments(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    frame = pd.read_csv(path, dtype=str).fillna("")
    values: set[str] = set()
    for row in frame.to_dict(orient="records"):
        local = next(
            (
                str(row.get(column, "")).strip().upper()
                for column in ("instrument", "local_instrument")
                if str(row.get(column, "")).strip()
            ),
            "",
        )
        if local and local != "NAN":
            values.add(local)
            continue
        gm_symbol = next(
            (
                row.get(column, "")
                for column in ("gm_symbol", "symbol")
                if str(row.get(column, "")).strip()
            ),
            "",
        )
        values.add(gm_to_local_symbol(gm_symbol))
    return {value for value in values if value and value != "NAN"}


def selected_profiles(profile_specs: str = "") -> list[MappedProfile]:
    if profile_specs.strip():
        return parse_profile_specs(profile_specs)
    return [
        MappedProfile("c_top150_rb45_risk0.80_cap0.30", 150, 45, 0.80, 100_000_000.0, 0.30),
        MappedProfile("c_top150_rb45_risk0.90_cap0.30", 150, 45, 0.90, 100_000_000.0, 0.30),
        MappedProfile("c_top150_rb45_risk1.00_cap0.30", 150, 45, 1.00, 100_000_000.0, 0.30),
        MappedProfile("c_top120_rb45_risk0.80_cap0.30", 120, 45, 0.80, 100_000_000.0, 0.30),
    ]


def load_context(
    middle_path: Path,
    outer_predictions_path: Path | None,
    minute_windows_path: Path,
    group_metadata_path: Path,
    start_signal: str,
    end_signal: str,
    raw_daily_dir: Path,
    price_start: str,
    price_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = load_middle_predictions(middle_path)
    if outer_predictions_path is not None:
        outer = pd.read_csv(outer_predictions_path, parse_dates=["datetime"])
        if "outer" not in outer.columns and "pred_outer" in outer.columns:
            outer = outer.rename(columns={"pred_outer": "outer"})
        if "outer" not in outer.columns:
            raise ValueError(f"{outer_predictions_path} must contain outer or pred_outer")
        predictions = predictions.drop(columns=["outer"], errors="ignore").merge(
            outer[["datetime", "instrument", "outer"]],
            on=["datetime", "instrument"],
            how="left",
        )
        predictions["outer"] = pd.to_numeric(predictions["outer"], errors="coerce").fillna(0.0)
        daily_outer = predictions.groupby("datetime")["outer"].median().rename("outer_risk_probability")
        predictions = predictions.merge(daily_outer, left_on="datetime", right_index=True, how="left")
    predictions = predictions[predictions["datetime"].between(start_signal, end_signal)].copy()
    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(
        raw_daily_dir,
        instruments,
        start=price_start,
        end=price_end,
    )
    minute_windows = pd.read_csv(minute_windows_path, parse_dates=["trade_date"])
    minute_windows["instrument"] = minute_windows["instrument"].astype(str).str.upper()
    group_metadata = load_group_metadata(group_metadata_path, "industry")
    return predictions, prices, minute_windows, group_metadata


def pct_change(nav: pd.Series) -> pd.Series:
    return nav.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    return float(((peak - nav) / peak).max()) if len(nav) else 0.0


def ensure_writer(path: Path, fieldnames: list[str]) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer._handle = handle  # type: ignore[attr-defined]
    return writer


def close_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_handle", None)
    if handle is not None:
        handle.close()


def export_profile_logs(
    frame: pd.DataFrame,
    minute_windows: pd.DataFrame,
    group_metadata: pd.DataFrame,
    profile: MappedProfile,
    cfg: MappedReplayConfig,
    cost: CostScenario,
    output_dir: Path,
    blocked_orders: dict[tuple[str, str, str], str] | None = None,
    outer_risk_threshold: float | None = None,
    outer_risk_floor: float | None = None,
    allocation_mode: str = "fixed_topk",
    pause_buys_on_sell_reject: bool = False,
    sell_first: bool = False,
    persist_blocked_sells: bool = False,
    stale_sell_block_days: int = 0,
    source_provenance: dict[str, object] | None = None,
) -> dict:
    if allocation_mode not in ALLOCATION_MODES:
        raise ValueError(f"unsupported allocation mode: {allocation_mode}")
    if stale_sell_block_days < 0:
        raise ValueError("stale_sell_block_days must be non-negative")
    profile_dir = output_dir / f"{profile.name}_{cost.name}"
    profile_dir.mkdir(parents=True, exist_ok=True)

    target_writer = ensure_writer(
        profile_dir / "targets.csv",
        [
            "trade_date",
            "instrument",
            "rank",
            "middle",
            "target_weight",
            "nominal_target_weight",
            "effective_target_weight",
            "target_value",
            "target_notional_ref",
            "target_shares",
            "mapped",
            "allocated",
            "group",
            "risk_budget",
            "outer_risk_probability",
            "allocation_mode",
        ],
    )
    allocation_writer = ensure_writer(
        profile_dir / "allocations.csv",
        [
            "trade_date",
            "allocation_mode",
            "risk_budget",
            "requested_count",
            "valid_price_count",
            "selected_count",
            "allocated_count",
            "budget_value",
            "allocated_notional",
            "effective_weight",
            "budget_utilization",
            "residual_value",
            "min_name_weight",
            "max_name_weight",
        ],
    )
    order_writer = ensure_writer(
        profile_dir / "orders.csv",
        [
            "trade_date",
            "instrument",
            "side",
            "current_shares",
            "target_shares",
            "delta_shares",
            "order_price",
            "order_notional",
            "reason",
        ],
    )
    fill_writer = ensure_writer(
        profile_dir / "fills.csv",
        [
            "trade_date",
            "instrument",
            "side",
            "shares",
            "order_price",
            "fill_price",
            "gross_value",
            "fee",
            "cash_before",
            "cash_after",
            "nav_ref",
            "turnover_contribution",
        ],
    )
    daily_writer = ensure_writer(
        profile_dir / "daily_account.csv",
        [
            "trade_date",
            "is_rebalance",
            "cash",
            "market_value",
            "nav",
            "gross_exposure",
            "day_turnover",
            "cum_turnover",
            "day_trades",
            "cum_trades",
            "holdings_count",
            "target_count",
            "mapped_count",
            "max_group_fraction",
            "risk_budget",
            "outer_risk_probability",
        ],
    )
    holdings_writer = ensure_writer(
        profile_dir / "holdings.csv",
        [
            "trade_date",
            "instrument",
            "shares",
            "mark_price",
            "market_value",
            "weight",
        ],
    )

    by_date = {date: part.copy() for date, part in frame.groupby("trade_date", sort=True)}
    minute_by_date = {date: part.copy() for date, part in minute_windows.groupby("trade_date", sort=True)}
    minute_pool = set(minute_windows["instrument"].drop_duplicates())
    cash = float(cfg.initial_cash)
    held: dict[str, int] = {}
    previous_full: set[str] = set()
    last_mark: dict[str, float] = {}
    nav_rows: list[tuple[pd.Timestamp, float]] = []
    cash_min = cash
    cum_turnover = 0.0
    cum_trades = 0
    total_fees = 0.0
    nav_mismatch_max = 0.0
    negative_cash_events = 0
    lot_violations = 0
    blocked_orders = blocked_orders or {}
    trade_date_to_day = {
        str(pd.Timestamp(trade_date).date()): day_no
        for day_no, trade_date in enumerate(by_date)
    }
    blocked_sell_start_days: dict[str, list[int]] = {}
    for (blocked_date, instrument, side), _ in blocked_orders.items():
        if side != "SELL" or blocked_date not in trade_date_to_day:
            continue
        blocked_sell_start_days.setdefault(instrument, []).append(
            trade_date_to_day[blocked_date]
        )
    blocked_order_events = 0
    paused_buy_events = 0
    buffer_k = profile.top_k * cfg.buffer_multiple
    desired: dict[str, int] = {}
    active_risk_budget = float(profile.risk_budget)
    active_outer_risk_probability = np.nan
    allocation_records: list[dict] = []

    for day_no, (trade_date, day) in enumerate(by_date.items()):
        trade_date_text = str(pd.Timestamp(trade_date).date())
        minute_day = minute_by_date.get(trade_date)
        if minute_day is None:
            price_open: dict[str, float] = {}
            mark_close: dict[str, float] = {}
        else:
            price_open = minute_day.set_index("instrument")["open_exec"].to_dict()
            mark_close = minute_day.set_index("instrument")["mark_close"].to_dict()
        open_nav = cash + sum(
            volume
            * safe_price(
                price_open.get(inst, np.nan),
                mark_close.get(inst, np.nan),
                last_mark.get(inst, np.nan),
            )
            for inst, volume in held.items()
        )
        is_rebalance = day_no % profile.rebalance_every == 0
        selected_full: list[str] = []
        mapped: list[str] = []
        groups: dict[str, str] | None = None
        day_turnover = 0.0
        day_trades = 0
        max_group_fraction = 0.0
        sell_rejected_today = False

        if is_rebalance:
            outer_risk_probability = (
                float(pd.to_numeric(day.get("outer_risk_probability"), errors="coerce").median())
                if "outer_risk_probability" in day
                else np.nan
            )
            risk_budget = float(profile.risk_budget)
            if (
                outer_risk_threshold is not None
                and outer_risk_floor is not None
                and np.isfinite(outer_risk_probability)
                and outer_risk_probability >= outer_risk_threshold
            ):
                risk_budget = float(outer_risk_floor)
            active_risk_budget = risk_budget
            active_outer_risk_probability = outer_risk_probability
            eligible_mask = day["eligible"].astype("boolean").fillna(False)
            eligible = day[eligible_mask].sort_values("middle", ascending=False)
            ranked = eligible["instrument"].tolist()
            groups = groups_on_date(group_metadata, trade_date)
            selected_full = select_with_group_cap(
                ranked,
                previous_full,
                top_k=profile.top_k,
                buffer_k=buffer_k,
                groups=groups,
                rules=ConcentrationRules(max_group_fraction=profile.industry_cap),
            )
            previous_full = set(selected_full)
            mapped = [
                inst
                for inst in selected_full
                if inst in minute_pool and safe_price(price_open.get(inst, np.nan)) > 0
            ]
            nominal_target_weight = risk_budget / profile.top_k if selected_full else 0.0
            signal_lookup = day.set_index("instrument")
            selected_groups = concentration_snapshot(selected_full, groups)
            max_group_fraction = float(selected_groups.get("max_group_fraction", 0.0) or 0.0)

            allocation_candidates = (
                [inst for inst in mapped if held.get(inst, 0) > 0]
                + [inst for inst in mapped if held.get(inst, 0) <= 0]
                if allocation_mode == "capital_aware"
                else mapped
            )
            allocation = allocate_equal_weight_lots(
                allocation_candidates,
                {inst: safe_price(price_open.get(inst, np.nan)) for inst in mapped},
                {inst: min_buy_lot(inst, cfg.lot_size) for inst in mapped},
                open_nav,
                risk_budget,
                lot_size=cfg.lot_size,
                denominator_count=profile.top_k,
                mode=allocation_mode,
            )
            desired = {inst: int(volume) for inst, volume in allocation["shares"].items()}
            allocation_weights = allocation["weights"]
            allocation_notionals = allocation["notional"]
            allocation_diagnostics = dict(allocation["diagnostics"])
            allocation_diagnostics["retained_candidate_count"] = sum(
                held.get(inst, 0) > 0 for inst in allocation_candidates
            )
            allocation_diagnostics.update(
                {
                    "trade_date": trade_date_text,
                    "allocation_mode": allocation_mode,
                    "risk_budget": risk_budget,
                }
            )
            allocation_records.append(allocation_diagnostics)
            allocation_writer.writerow(
                {
                    key: allocation_diagnostics.get(key)
                    for key in allocation_writer.fieldnames
                }
            )
            for rank, inst in enumerate(selected_full, start=1):
                price = safe_price(price_open.get(inst, np.nan))
                is_mapped = inst in mapped and price > 0
                target_shares = int(desired.get(inst, 0))
                effective_target_weight = float(allocation_weights.get(inst, 0.0))
                target_notional_ref = float(allocation_notionals.get(inst, 0.0))
                target_weight = (
                    nominal_target_weight
                    if allocation_mode == "fixed_topk" and is_mapped
                    else effective_target_weight
                )
                target_value = (
                    open_nav * nominal_target_weight
                    if allocation_mode == "fixed_topk" and is_mapped
                    else target_notional_ref
                )
                target_writer.writerow(
                    {
                        "trade_date": str(pd.Timestamp(trade_date).date()),
                        "instrument": inst,
                        "rank": rank,
                        "middle": float(signal_lookup.loc[inst, "middle"]) if inst in signal_lookup.index else np.nan,
                        "target_weight": target_weight,
                        "nominal_target_weight": nominal_target_weight if is_mapped else 0.0,
                        "effective_target_weight": effective_target_weight,
                        "target_value": target_value,
                        "target_notional_ref": target_notional_ref,
                        "target_shares": target_shares,
                        "mapped": int(is_mapped),
                        "allocated": int(target_shares > 0),
                        "group": groups.get(inst, "__UNKNOWN__") if groups else "__UNKNOWN__",
                        "risk_budget": risk_budget,
                        "outer_risk_probability": outer_risk_probability,
                        "allocation_mode": allocation_mode,
                    }
                )
        else:
            outer_risk_probability = active_outer_risk_probability
            risk_budget = active_risk_budget

            deltas = {inst: desired.get(inst, 0) - held.get(inst, 0) for inst in set(held) | set(desired)}
            has_sell_delta = any(delta < 0 for delta in deltas.values())
            for inst, delta in sorted(deltas.items(), key=order_delta_key):
                price = safe_price(price_open.get(inst, np.nan))
                if price <= 0 or delta == 0:
                    continue
                if delta > 0 and delta < min_buy_lot(inst, cfg.lot_size):
                    continue
                side = "SELL" if delta < 0 else "BUY"
                block_reason = stale_sell_block_reason(
                    blocked_orders.get((trade_date_text, inst, side)),
                    side,
                    day_no,
                    blocked_sell_start_days.get(inst, []),
                    persist_blocked_sells,
                    stale_sell_block_days,
                )
                sell_first_reason = (
                    "buy_deferred_until_sell_targets_complete"
                    if side == "BUY" and sell_first and has_sell_delta
                    else ""
                )
                pause_reason = (
                    "buy_paused_after_sell_rejection"
                    if side == "BUY" and pause_buys_on_sell_reject and sell_rejected_today
                    else ""
                )
                target_shares = desired.get(inst, 0)
                current_shares = held.get(inst, 0)
                order_notional = abs(delta) * price
                order_writer.writerow(
                    {
                        "trade_date": trade_date_text,
                        "instrument": inst,
                        "side": side,
                        "current_shares": current_shares,
                        "target_shares": target_shares,
                        "delta_shares": delta,
                        "order_price": price,
                        "order_notional": order_notional,
                        "reason": (
                            block_reason
                            or pause_reason
                            or sell_first_reason
                            or "rebalance_to_target"
                        ),
                    }
                )
                if pause_reason or sell_first_reason:
                    paused_buy_events += 1
                    continue
                if block_reason:
                    blocked_order_events += 1
                    if side == "SELL":
                        sell_rejected_today = True
                    continue
                nav_ref = cash + sum(
                    volume
                    * safe_price(
                        mark_close.get(code, np.nan),
                        price_open.get(code, np.nan),
                        last_mark.get(code, np.nan),
                    )
                    for code, volume in held.items()
                )
                cash_before = cash
                if delta < 0:
                    volume = min(-delta, held.get(inst, 0))
                    fill_price = price * (1.0 - cost.slippage)
                    gross_value = volume * fill_price
                    fee = max(gross_value * cost.sell_cost, cost.min_cost) if volume > 0 else 0.0
                    cash += gross_value - fee
                    held[inst] = held.get(inst, 0) - volume
                else:
                    fill_price = price * (1.0 + cost.slippage)
                    affordable = round_lot(
                        max(0.0, cash - cost.min_cost) / (fill_price * (1.0 + cost.buy_cost)),
                        cfg.lot_size,
                        min_buy_lot(inst, cfg.lot_size),
                    )
                    volume = min(delta, affordable)
                    gross_value = volume * fill_price
                    fee = max(gross_value * cost.buy_cost, cost.min_cost) if volume > 0 else 0.0
                    cash -= gross_value + fee
                    held[inst] = held.get(inst, 0) + volume
                if volume > 0:
                    turn = gross_value / max(nav_ref, 1e-12)
                    day_turnover += turn
                    cum_turnover += turn
                    day_trades += 1
                    cum_trades += 1
                    total_fees += fee
                    fill_writer.writerow(
                        {
                            "trade_date": str(pd.Timestamp(trade_date).date()),
                            "instrument": inst,
                            "side": side,
                            "shares": volume,
                            "order_price": price,
                            "fill_price": fill_price,
                            "gross_value": gross_value,
                            "fee": fee,
                            "cash_before": cash_before,
                            "cash_after": cash,
                            "nav_ref": nav_ref,
                            "turnover_contribution": turn,
                        }
                    )
                    if volume % cfg.lot_size != 0:
                        lot_violations += 1
                    cash_min = min(cash_min, cash)
                    if cash < -1e-6:
                        negative_cash_events += 1
            held = {inst: volume for inst, volume in held.items() if volume > 0}

        close_market_value = sum(
            volume
            * safe_price(
                mark_close.get(inst, np.nan),
                price_open.get(inst, np.nan),
                last_mark.get(inst, np.nan),
            )
            for inst, volume in held.items()
        )
        close_nav = cash + close_market_value
        recomputed_nav = cash + close_market_value
        nav_mismatch_max = max(nav_mismatch_max, abs(close_nav - recomputed_nav))
        exposure = close_market_value / max(close_nav, 1e-12)
        daily_writer.writerow(
            {
                "trade_date": str(pd.Timestamp(trade_date).date()),
                "is_rebalance": int(is_rebalance),
                "cash": cash,
                "market_value": close_market_value,
                "nav": close_nav,
                "gross_exposure": exposure,
                "day_turnover": day_turnover,
                "cum_turnover": cum_turnover,
                "day_trades": day_trades,
                "cum_trades": cum_trades,
                "holdings_count": len(held),
                "target_count": len(selected_full),
                "mapped_count": len(mapped),
                "max_group_fraction": max_group_fraction,
                "risk_budget": risk_budget,
                "outer_risk_probability": outer_risk_probability,
            }
        )
        for inst, volume in sorted(held.items()):
            mark = safe_price(
                mark_close.get(inst, np.nan),
                price_open.get(inst, np.nan),
                last_mark.get(inst, np.nan),
            )
            market_value = volume * mark
            holdings_writer.writerow(
                {
                    "trade_date": str(pd.Timestamp(trade_date).date()),
                    "instrument": inst,
                    "shares": volume,
                    "mark_price": mark,
                    "market_value": market_value,
                    "weight": market_value / max(close_nav, 1e-12),
                }
            )
        last_mark.update({inst: safe_price(price) for inst, price in mark_close.items() if safe_price(price) > 0})
        nav_rows.append((trade_date, close_nav))

    for writer in (
        target_writer,
        allocation_writer,
        order_writer,
        fill_writer,
        daily_writer,
        holdings_writer,
    ):
        close_writer(writer)

    nav = pd.Series(dict(nav_rows)).sort_index()
    returns = pct_change(nav)
    daily = pd.read_csv(profile_dir / "daily_account.csv")
    audit = {
        "status": "production_log_export_research_only",
        "profile": asdict(profile),
        "config": asdict(cfg),
        "cost": asdict(cost),
        "output_dir": str(profile_dir),
        "source_provenance": dict(source_provenance or {}),
        "days": int(len(nav)),
        "final_nav": float(nav.iloc[-1]) if len(nav) else 0.0,
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1) if len(nav) else 0.0,
        "max_drawdown": max_drawdown(nav),
        "turnover": float(cum_turnover),
        "trades": int(cum_trades),
        "total_fees": float(total_fees),
        "min_cash": float(cash_min),
        "negative_cash_events": int(negative_cash_events),
        "lot_violations": int(lot_violations),
        "blocked_order_events": int(blocked_order_events),
        "paused_buy_events": int(paused_buy_events),
        "blocked_order_keys": len(blocked_orders),
        "outer_risk_threshold": outer_risk_threshold,
        "outer_risk_floor": outer_risk_floor,
        "allocation_mode": allocation_mode,
        "pause_buys_on_sell_reject": bool(pause_buys_on_sell_reject),
        "sell_first": bool(sell_first),
        "persist_blocked_sells": bool(persist_blocked_sells),
        "stale_sell_block_days": int(stale_sell_block_days),
        "rebalance_count": len(allocation_records),
        "avg_rebalance_budget_utilization": float(
            np.mean([row.get("budget_utilization", 0.0) for row in allocation_records])
        )
        if allocation_records
        else 0.0,
        "min_rebalance_budget_utilization": float(
            np.min([row.get("budget_utilization", 0.0) for row in allocation_records])
        )
        if allocation_records
        else 0.0,
        "avg_rebalance_effective_weight": float(
            np.mean([row.get("effective_weight", 0.0) for row in allocation_records])
        )
        if allocation_records
        else 0.0,
        "max_allocated_name_weight": float(
            np.max([row.get("max_name_weight", 0.0) for row in allocation_records])
        )
        if allocation_records
        else 0.0,
        "nav_mismatch_max": float(nav_mismatch_max),
        "max_daily_turnover": float(daily["day_turnover"].max()) if len(daily) else 0.0,
        "max_gross_exposure": float(daily["gross_exposure"].max()) if len(daily) else 0.0,
        "min_gross_exposure": float(daily["gross_exposure"].min()) if len(daily) else 0.0,
        "worst_day_return": float(returns.min()) if len(returns) else 0.0,
        "deployment_allowed": False,
    }
    (profile_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return audit


def run(
    middle_path: Path,
    outer_predictions_path: Path | None,
    minute_windows_path: Path,
    group_metadata_path: Path,
    output_dir: Path,
    start_signal: str,
    end_signal: str,
    cost_name: str,
    initial_cash: float,
    blocked_gm_audit_dir: Path | None = None,
    profile_name: str | None = None,
    profile_specs: str = "",
    raw_daily_dir: Path = DATA / "A_Stock_daily_qfq",
    price_start: str = "2022-01-04",
    price_end: str = "2026-04-28",
    forbidden_path: Path | None = DEFAULT_FORBIDDEN,
    outer_risk_threshold: float | None = None,
    outer_risk_floor: float | None = None,
    allocation_mode: str = "fixed_topk",
    buffer_multiple: int = 2,
    pause_buys_on_sell_reject: bool = False,
    sell_first: bool = False,
    persist_blocked_sells: bool = False,
    stale_sell_block_days: int = 0,
) -> dict:
    if allocation_mode not in ALLOCATION_MODES:
        raise ValueError(f"unsupported allocation mode: {allocation_mode}")
    predictions, prices, minute_windows, group_metadata = load_context(
        middle_path,
        outer_predictions_path,
        minute_windows_path,
        group_metadata_path,
        start_signal,
        end_signal,
        raw_daily_dir,
        price_start,
        price_end,
    )
    forbidden = load_forbidden_instruments(forbidden_path)
    blocked_status_path = (
        blocked_gm_audit_dir / "order_status.csv"
        if blocked_gm_audit_dir is not None
        else None
    )
    source_provenance = {
        "middle_prediction": artifact(middle_path),
        "outer_prediction": artifact(outer_predictions_path),
        "minute_windows": artifact(minute_windows_path),
        "group_metadata": artifact(group_metadata_path),
        "forbidden_symbols": artifact(forbidden_path),
        "blocked_order_status": artifact(blocked_status_path),
    }
    forbidden_prediction_rows = 0
    if forbidden:
        before = len(predictions)
        predictions = predictions[~predictions["instrument"].astype(str).str.upper().isin(forbidden)].copy()
        forbidden_prediction_rows = before - len(predictions)
        print(
            f"[production logs] forbidden instruments={len(forbidden)} "
            f"removed_prediction_rows={forbidden_prediction_rows}",
            flush=True,
        )
    cost = {"base": BASE_COST, "stress": STRESS_COST}[cost_name]
    cfg = MappedReplayConfig(initial_cash=initial_cash, buffer_multiple=buffer_multiple)
    blocked_orders = load_blocked_orders(blocked_gm_audit_dir)
    audits = []
    profiles = selected_profiles(profile_specs)
    if profile_name:
        profiles = [profile for profile in profiles if profile.name == profile_name]
        if not profiles:
            raise ValueError(f"unknown profile_name={profile_name!r}")
    for profile in profiles:
        frame = prepare_daily_frame(predictions, prices, profile, cfg)
        audit = export_profile_logs(
            frame,
            minute_windows,
            group_metadata,
            profile,
            cfg,
            cost,
            output_dir,
            blocked_orders,
            outer_risk_threshold,
            outer_risk_floor,
            allocation_mode,
            pause_buys_on_sell_reject,
            sell_first,
            persist_blocked_sells,
            stale_sell_block_days,
            source_provenance,
        )
        audits.append(audit)
        print(
            f"[production logs] {profile.name}/{cost.name} "
            f"ret={audit['total_return']:+.2%} mdd={audit['max_drawdown']:.2%} "
            f"turn={audit['turnover']:.2f} trades={audit['trades']} "
            f"min_cash={audit['min_cash']:.2f}",
            flush=True,
        )
    summary = {
        "status": "c_baseline_production_logs_research_only",
        "middle_prediction": str(middle_path),
        "outer_predictions": str(outer_predictions_path) if outer_predictions_path else None,
        "minute_windows": str(minute_windows_path),
        "group_metadata": str(group_metadata_path),
        "raw_daily_dir": str(raw_daily_dir),
        "price_start": price_start,
        "price_end": price_end,
        "window": {"start_signal": start_signal, "end_signal": end_signal},
        "cost": cost_name,
        "initial_cash": initial_cash,
        "blocked_gm_audit_dir": str(blocked_gm_audit_dir) if blocked_gm_audit_dir else None,
        "forbidden_path": str(forbidden_path) if forbidden_path else None,
        "forbidden_instruments": int(len(forbidden)),
        "forbidden_prediction_rows": int(forbidden_prediction_rows),
        "blocked_order_keys": len(blocked_orders),
        "outer_risk_threshold": outer_risk_threshold,
        "outer_risk_floor": outer_risk_floor,
        "allocation_mode": allocation_mode,
        "buffer_multiple": buffer_multiple,
        "pause_buys_on_sell_reject": bool(pause_buys_on_sell_reject),
        "sell_first": bool(sell_first),
        "persist_blocked_sells": bool(persist_blocked_sells),
        "stale_sell_block_days": int(stale_sell_block_days),
        "output_dir": str(output_dir),
        "source_provenance": source_provenance,
        "audits": audits,
        "deployment_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "profile",
            "total_return",
            "max_drawdown",
            "turnover",
            "trades",
            "total_fees",
            "min_cash",
            "negative_cash_events",
            "lot_violations",
            "blocked_order_events",
            "paused_buy_events",
            "blocked_order_keys",
            "outer_risk_threshold",
            "outer_risk_floor",
            "allocation_mode",
            "rebalance_count",
            "avg_rebalance_budget_utilization",
            "min_rebalance_budget_utilization",
            "avg_rebalance_effective_weight",
            "max_allocated_name_weight",
            "nav_mismatch_max",
            "max_daily_turnover",
            "max_gross_exposure",
            "min_gross_exposure",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            row = {key: audit.get(key) for key in fields}
            row["profile"] = audit["profile"]["name"]
            writer.writerow(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--middle", default=str(DEFAULT_MIDDLE))
    parser.add_argument("--outer-predictions", default="")
    parser.add_argument("--minute-windows", default=str(DEFAULT_WINDOWS))
    parser.add_argument("--group-metadata", default=str(DEFAULT_GROUP_METADATA))
    parser.add_argument("--output-dir", default=str(HERE / "outputs" / "production_logs_c_baseline"))
    parser.add_argument("--start-signal", default="2025-01-03")
    parser.add_argument("--end-signal", default="2026-04-02")
    parser.add_argument("--cost", choices=["base", "stress"], default="stress")
    parser.add_argument("--initial-cash", type=float, default=500_000.0)
    parser.add_argument("--blocked-gm-audit-dir", default="")
    parser.add_argument("--profile-name", default="")
    parser.add_argument(
        "--profile-specs",
        default="",
        help=(
            "Semicolon separated profile specs: "
            "name,top_k,rebalance,risk,min_amount,industry_cap"
        ),
    )
    parser.add_argument("--raw-daily-dir", default=str(DATA / "A_Stock_daily_qfq"))
    parser.add_argument("--price-start", default="2022-01-04")
    parser.add_argument("--price-end", default="2026-04-28")
    parser.add_argument("--forbidden-symbols", default=str(DEFAULT_FORBIDDEN))
    parser.add_argument("--outer-risk-threshold", type=float, default=None)
    parser.add_argument("--outer-risk-floor", type=float, default=None)
    parser.add_argument(
        "--allocation-mode",
        choices=sorted(ALLOCATION_MODES),
        default="fixed_topk",
    )
    parser.add_argument("--buffer-multiple", type=int, default=2)
    parser.add_argument("--pause-buys-on-sell-reject", action="store_true")
    parser.add_argument("--sell-first", action="store_true")
    parser.add_argument("--persist-blocked-sells", action="store_true")
    parser.add_argument(
        "--stale-sell-block-days",
        type=int,
        default=0,
        help="Block audited sell symbols from the rejection date for N trading days.",
    )
    args = parser.parse_args()
    run(
        Path(args.middle).resolve(),
        Path(args.outer_predictions).resolve() if args.outer_predictions else None,
        Path(args.minute_windows).resolve(),
        Path(args.group_metadata).resolve(),
        Path(args.output_dir).resolve(),
        args.start_signal,
        args.end_signal,
        args.cost,
        args.initial_cash,
        Path(args.blocked_gm_audit_dir).resolve() if args.blocked_gm_audit_dir else None,
        args.profile_name or None,
        args.profile_specs,
        Path(args.raw_daily_dir).resolve(),
        args.price_start,
        args.price_end,
        Path(args.forbidden_symbols).resolve() if args.forbidden_symbols else None,
        args.outer_risk_threshold,
        args.outer_risk_floor,
        args.allocation_mode,
        args.buffer_multiple,
        args.pause_buys_on_sell_reject,
        args.sell_first,
        args.persist_blocked_sells,
        args.stale_sell_block_days,
    )


if __name__ == "__main__":
    main()
