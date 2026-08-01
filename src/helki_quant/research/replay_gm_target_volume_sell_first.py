from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from export_c_baseline_production_logs import load_blocked_orders
from portfolio_experiments import BASE_COST, STRESS_COST, CostScenario
from replay_gm_target_volume import (
    ReplayConfig,
    load_price_panel,
    load_targets,
    min_buy_lot,
    round_lot,
    safe_price,
)


def replay_sell_first(
    target_csv: Path,
    raw_daily_dir: Path,
    gm_audit_dir: Path,
    output_dir: Path,
    cost: CostScenario,
    cfg: ReplayConfig,
    price_start: str,
    price_end: str,
    profile_name: str,
) -> dict[str, object]:
    targets = load_targets(target_csv)
    instruments = set(targets["instrument"])
    prices = load_price_panel(raw_daily_dir, instruments, price_start, price_end)
    price_by_date = {
        date: part.set_index("instrument").copy()
        for date, part in prices.groupby("datetime", sort=True)
    }
    target_by_date = {
        date: part.copy()
        for date, part in targets.groupby("trade_date", sort=True)
    }
    calendar = pd.DatetimeIndex(prices["datetime"].drop_duplicates().sort_values())
    calendar = calendar[
        (calendar >= min(target_by_date)) & (calendar <= pd.Timestamp(price_end))
    ]
    blocked_orders = load_blocked_orders(gm_audit_dir)

    profile_dir = output_dir / "gm_target_volume_sell_first_replay"
    profile_dir.mkdir(parents=True, exist_ok=True)
    orders: list[dict[str, object]] = []
    fills: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []

    cash = float(cfg.initial_cash)
    held: dict[str, int] = {}
    last_mark: dict[str, float] = {}
    pending_target: dict[str, int] | None = None
    pending_weights: dict[str, float] = {}
    pending_buy_names: set[str] = set()
    pending_source_date: str | None = None
    turnover = 0.0
    total_fees = 0.0
    trades = 0
    min_cash = cash
    nav_peak = cash
    max_drawdown = 0.0
    max_daily_turnover = 0.0
    max_gross_exposure = 0.0
    negative_cash_events = 0
    lot_violations = 0
    blocked_events = 0

    def submit(
        trade_date: pd.Timestamp,
        instrument: str,
        target_shares: int,
        price_day: pd.DataFrame,
        reason: str,
    ) -> tuple[bool, float]:
        nonlocal cash, turnover, total_fees, trades, min_cash
        nonlocal negative_cash_events, lot_violations, blocked_events
        current_shares = int(held.get(instrument, 0))
        delta = int(target_shares) - current_shares
        if delta == 0:
            return True, 0.0
        side = "SELL" if delta < 0 else "BUY"
        date_text = trade_date.strftime("%Y-%m-%d")
        minimum_buy = min_buy_lot(instrument, cfg.lot_size)
        if delta > 0 and delta < minimum_buy:
            orders.append(
                {
                    "trade_date": date_text,
                    "instrument": instrument,
                    "side": side,
                    "current_shares": current_shares,
                    "target_shares": int(target_shares),
                    "delta_shares": delta,
                    "order_price": 0.0,
                    "reason": reason,
                    "status": "MIN_BUY_SKIPPED",
                    "block_reason": f"buy_delta_below_{minimum_buy}",
                }
            )
            return False, 0.0
        row = price_day.loc[instrument] if instrument in price_day.index else None
        open_price = row.get("open") if row is not None else None
        close_price = row.get("close") if row is not None else None
        volume_available = row.get("volume") if row is not None else None
        block_reason = blocked_orders.get((date_text, instrument, side), "")
        if not block_reason and (row is None or pd.isna(volume_available) or float(volume_available) <= 0):
            block_reason = "daily_row_missing_or_suspended"
        order_price = safe_price(open_price, close_price, last_mark.get(instrument))
        orders.append(
            {
                "trade_date": date_text,
                "instrument": instrument,
                "side": side,
                "current_shares": current_shares,
                "target_shares": int(target_shares),
                "delta_shares": delta,
                "order_price": order_price,
                "reason": reason,
                "status": "BLOCKED" if block_reason else "FILLED",
                "block_reason": block_reason,
            }
        )
        if block_reason or order_price <= 0:
            blocked_events += 1
            return False, 0.0
        cash_before = cash
        nav_ref = cash + sum(
            shares
            * safe_price(
                price_day.loc[code].get("close") if code in price_day.index else None,
                last_mark.get(code),
            )
            for code, shares in held.items()
        )
        if delta < 0:
            fill_volume = min(-delta, current_shares)
            if 0 < fill_volume < minimum_buy:
                fill_volume = min(minimum_buy, current_shares)
                remainder = current_shares - fill_volume
                if 0 < remainder < minimum_buy:
                    fill_volume = current_shares
            fill_price = order_price * (1.0 - cost.slippage)
            gross_value = fill_volume * fill_price
            fee = max(gross_value * cost.sell_cost, cost.min_cost)
            cash += gross_value - fee
            held[instrument] = current_shares - fill_volume
        else:
            fill_price = order_price * (1.0 + cost.slippage)
            affordable = round_lot(
                max(0.0, cash - cost.min_cost) / (fill_price * (1.0 + cost.buy_cost)),
                cfg.lot_size,
                min_buy_lot(instrument, cfg.lot_size),
            )
            fill_volume = min(delta, affordable)
            if fill_volume <= 0:
                orders[-1]["status"] = "CASH_BLOCKED"
                orders[-1]["block_reason"] = "insufficient_cash"
                blocked_events += 1
                return False, 0.0
            gross_value = fill_volume * fill_price
            fee = max(gross_value * cost.buy_cost, cost.min_cost)
            cash -= gross_value + fee
            held[instrument] = current_shares + fill_volume
        if held.get(instrument, 0) <= 0:
            held.pop(instrument, None)
        turn = gross_value / max(nav_ref, 1e-12)
        turnover += turn
        total_fees += fee
        trades += 1
        min_cash = min(min_cash, cash)
        if cash < -1e-6:
            negative_cash_events += 1
        if fill_volume % cfg.lot_size:
            lot_violations += 1
        fills.append(
            {
                "trade_date": date_text,
                "instrument": instrument,
                "side": side,
                "shares": int(fill_volume),
                "order_price": order_price,
                "fill_price": fill_price,
                "gross_value": gross_value,
                "fee": fee,
                "cash_before": cash_before,
                "cash_after": cash,
                "nav_ref": nav_ref,
                "turnover_contribution": turn,
                "reason": reason,
            }
        )
        return True, turn

    for trade_date in calendar:
        date_text = trade_date.strftime("%Y-%m-%d")
        price_day = price_by_date.get(trade_date)
        if price_day is None:
            continue
        for instrument, value in price_day["close"].items():
            mark = safe_price(value)
            if mark > 0:
                last_mark[instrument] = mark
        day_turnover = 0.0
        is_rebalance = trade_date in target_by_date
        action = "mark"

        if is_rebalance:
            day = target_by_date[trade_date]
            desired = dict(zip(day["instrument"], day["target_shares"]))
            weights = dict(zip(day["instrument"], day["target_weight"]))
            pending_target = None
            pending_weights = {}
            pending_buy_names = set()
            pending_source_date = None
            deltas = {
                instrument: int(desired.get(instrument, 0)) - int(held.get(instrument, 0))
                for instrument in set(held) | set(desired)
            }
            sell_names = sorted(instrument for instrument, delta in deltas.items() if delta < 0)
            buy_names = sorted(instrument for instrument, delta in deltas.items() if delta > 0)
            for instrument in sell_names:
                _, turn = submit(
                    trade_date,
                    instrument,
                    int(desired.get(instrument, 0)),
                    price_day,
                    "target_sell",
                )
                day_turnover += turn
            if sell_names:
                pending_target = {str(k): int(v) for k, v in desired.items()}
                pending_weights = {str(k): float(v) for k, v in weights.items()}
                pending_buy_names = set(buy_names)
                pending_source_date = date_text
                action = "rebalance_sell_first"
            else:
                for instrument in buy_names:
                    _, turn = submit(
                        trade_date,
                        instrument,
                        int(desired[instrument]),
                        price_day,
                        "target_buy",
                    )
                    day_turnover += turn
                action = "rebalance_at_once"
        elif pending_target is not None:
            remaining_sells = sorted(
                instrument
                for instrument in set(held) | set(pending_target)
                if int(held.get(instrument, 0)) > int(pending_target.get(instrument, 0))
            )
            if remaining_sells:
                for instrument in remaining_sells:
                    _, turn = submit(
                        trade_date,
                        instrument,
                        int(pending_target.get(instrument, 0)),
                        price_day,
                        "retry_sell",
                    )
                    day_turnover += turn
                action = "retry_sell"
            else:
                for instrument in sorted(pending_buy_names):
                    target_shares = int(pending_target[instrument])
                    if target_shares <= int(held.get(instrument, 0)):
                        continue
                    _, turn = submit(
                        trade_date,
                        instrument,
                        target_shares,
                        price_day,
                        "pending_buy",
                    )
                    day_turnover += turn
                pending_target = None
                pending_weights = {}
                pending_buy_names = set()
                pending_source_date = None
                action = "pending_buy"

        market_value = sum(
            shares
            * safe_price(
                price_day.loc[instrument].get("close") if instrument in price_day.index else None,
                last_mark.get(instrument),
            )
            for instrument, shares in held.items()
        )
        nav = cash + market_value
        nav_peak = max(nav_peak, nav)
        drawdown = 1.0 - nav / max(nav_peak, 1e-12)
        max_drawdown = max(max_drawdown, drawdown)
        exposure = market_value / max(nav, 1e-12)
        max_daily_turnover = max(max_daily_turnover, day_turnover)
        max_gross_exposure = max(max_gross_exposure, exposure)
        daily_rows.append(
            {
                "trade_date": date_text,
                "action": action,
                "is_rebalance": int(is_rebalance),
                "cash": cash,
                "market_value": market_value,
                "nav": nav,
                "gross_exposure": exposure,
                "daily_turnover": day_turnover,
                "drawdown": drawdown,
                "holdings": len(held),
                "pending_source_date": pending_source_date or "",
            }
        )
        for instrument, shares in sorted(held.items()):
            holdings_rows.append(
                {"trade_date": date_text, "instrument": instrument, "shares": shares}
            )

    daily = pd.DataFrame(daily_rows)
    final_nav = float(daily.iloc[-1]["nav"])
    final_cash = float(daily.iloc[-1]["cash"])
    final_exposure = float(daily.iloc[-1]["gross_exposure"])
    pd.DataFrame(orders).to_csv(profile_dir / "orders.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fills).to_csv(profile_dir / "fills.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(profile_dir / "daily_account.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(holdings_rows).to_csv(
        profile_dir / "holdings.csv", index=False, encoding="utf-8-sig"
    )
    audit = {
        "status": "frozen_gm_target_volume_sell_first_replay",
        "profile": {"name": profile_name},
        "cost": asdict(cost),
        "target_csv": str(target_csv),
        "gm_audit_dir": str(gm_audit_dir),
        "initial_cash": cfg.initial_cash,
        "trades": trades,
        "turnover": turnover,
        "max_daily_turnover": max_daily_turnover,
        "max_gross_exposure": max_gross_exposure,
        "min_cash": min_cash,
        "negative_cash_events": negative_cash_events,
        "lot_violations": lot_violations,
        "blocked_order_events": blocked_events,
        "total_fees": total_fees,
        "final_nav": final_nav,
        "final_cash": final_cash,
        "final_exposure": final_exposure,
        "total_return": final_nav / cfg.initial_cash - 1.0,
        "max_drawdown": max_drawdown,
        "pending_target_at_finish": pending_target is not None,
        "deployment_allowed": False,
    }
    (profile_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-csv", required=True, type=Path)
    parser.add_argument("--raw-daily-dir", required=True, type=Path)
    parser.add_argument("--gm-audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cost", choices=["base", "stress"], default="stress")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--price-start", default="2025-01-07")
    parser.add_argument("--price-end", default="2026-04-03")
    parser.add_argument("--profile-name", default="c_outer_ca_t150_r30_60_20_b4")
    args = parser.parse_args()
    audit = replay_sell_first(
        args.target_csv.resolve(),
        args.raw_daily_dir.resolve(),
        args.gm_audit_dir.resolve(),
        args.output_dir.resolve(),
        {"base": BASE_COST, "stress": STRESS_COST}[args.cost],
        ReplayConfig(initial_cash=args.initial_cash),
        args.price_start,
        args.price_end,
        args.profile_name,
    )
    print(
        "[frozen target SELL_FIRST replay] "
        f"return={audit['total_return']:+.2%} mdd={audit['max_drawdown']:.2%} "
        f"trades={audit['trades']} blocked={audit['blocked_order_events']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
