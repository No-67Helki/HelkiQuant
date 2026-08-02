from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .replay_held_intraday_t0 import (
        concentration_metrics,
        max_drawdown,
        replay_fold_metrics,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from replay_held_intraday_t0 import (
        concentration_metrics,
        max_drawdown,
        replay_fold_metrics,
    )


def filter_component(
    path: Path,
    *,
    trigger_distance: float,
    touch_buffer: float,
    max_lots: int,
) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    frame["trade_date"] = frame["trade_date"].dt.normalize()
    mask = (
        np.isclose(frame["trigger_distance"], trigger_distance, rtol=0.0, atol=1e-12)
        & np.isclose(frame["touch_buffer"], touch_buffer, rtol=0.0, atol=1e-12)
        & (pd.to_numeric(frame["max_lots"], errors="coerce") == max_lots)
    )
    return frame[mask].copy()


def conflict_keys(buy: pd.DataFrame, sell: pd.DataFrame) -> pd.DataFrame:
    keys = ["trade_date", "instrument"]
    return buy[keys].merge(sell[keys], on=keys, how="inner").drop_duplicates()


def combine(
    buy_trades_path: Path,
    sell_trades_path: Path,
    calendar_daily_path: Path,
    daily_account_path: Path,
    output_json: Path,
    output_trades: Path,
    output_daily: Path,
    *,
    buy_trigger: float,
    sell_trigger: float,
    touch_buffer: float,
    max_lots: int,
    max_daily_turnover: float,
    max_symbols_per_day: int,
) -> dict:
    buy = filter_component(
        buy_trades_path,
        trigger_distance=buy_trigger,
        touch_buffer=touch_buffer,
        max_lots=max_lots,
    )
    sell = filter_component(
        sell_trades_path,
        trigger_distance=sell_trigger,
        touch_buffer=touch_buffer,
        max_lots=max_lots,
    )
    if buy.empty or sell.empty:
        raise ValueError(f"empty component buy={len(buy)} sell={len(sell)}")
    conflicts = conflict_keys(buy, sell)
    if not conflicts.empty:
        conflict_index = pd.MultiIndex.from_frame(conflicts)
        buy = buy[
            ~pd.MultiIndex.from_frame(buy[["trade_date", "instrument"]]).isin(conflict_index)
        ].copy()
        sell = sell[
            ~pd.MultiIndex.from_frame(sell[["trade_date", "instrument"]]).isin(conflict_index)
        ].copy()
    candidates = pd.concat([buy, sell], ignore_index=True).sort_values(
        ["trade_date", "score"], ascending=[True, False]
    )

    calendar = pd.read_csv(calendar_daily_path, parse_dates=["trade_date"])
    calendar["trade_date"] = calendar["trade_date"].dt.normalize()
    calendar_mask = (
        np.isclose(calendar["trigger_distance"], buy_trigger, rtol=0.0, atol=1e-12)
        & np.isclose(calendar["touch_buffer"], touch_buffer, rtol=0.0, atol=1e-12)
        & (pd.to_numeric(calendar["max_lots"], errors="coerce") == max_lots)
    )
    calendar = calendar[calendar_mask].drop_duplicates("trade_date", keep="last").copy()
    if calendar.empty:
        raise ValueError("calendar component has no rows for selected buy setting")
    account = pd.read_csv(daily_account_path, parse_dates=["trade_date"])
    account["trade_date"] = account["trade_date"].dt.normalize()
    account = account.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    default_nav = float(account["nav"].iloc[0])
    candidates_by_day = {
        date: part for date, part in candidates.groupby("trade_date", sort=True)
    }

    accepted_rows = []
    daily_rows = []
    cumulative_pnl = 0.0
    cumulative_turnover = 0.0
    rejected_turnover = 0
    rejected_cash = 0
    rejected_symbol_cap = 0
    for day in calendar.sort_values("trade_date").itertuples(index=False):
        trade_date = pd.Timestamp(day.trade_date).normalize()
        if trade_date in account.index:
            account_row = account.loc[trade_date]
            nav = float(account_row["nav"])
            cash = float(account_row["cash"])
        else:
            nav = float(day.base_nav)
            cash = nav
        day_pnl = 0.0
        turnover_value = 0.0
        buy_cash_used = 0.0
        day_symbols: set[str] = set()
        day_accepted = 0
        for row in candidates_by_day.get(trade_date, pd.DataFrame()).itertuples(index=False):
            if len(day_symbols) >= max_symbols_per_day:
                rejected_symbol_cap += 1
                continue
            candidate_turnover = float(row.turnover_value)
            if (turnover_value + candidate_turnover) / max(nav, 1e-12) > max_daily_turnover:
                rejected_turnover += 1
                continue
            if row.trade_direction == "buy_first":
                required_cash = float(row.buy_price) * int(row.t0_volume) + float(row.buy_fee)
                if buy_cash_used + required_cash > cash:
                    rejected_cash += 1
                    continue
                buy_cash_used += required_cash
            accepted = row._asdict()
            accepted["combined_profile"] = "bidirectional_tick_trigger"
            accepted_rows.append(accepted)
            day_symbols.add(row.instrument)
            day_pnl += float(row.pnl)
            turnover_value += candidate_turnover
            day_accepted += 1
        cumulative_pnl += day_pnl
        day_turnover = turnover_value / max(nav, 1e-12)
        cumulative_turnover += day_turnover
        daily_rows.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "fold": int(day.fold),
                "base_nav": nav,
                "incremental_nav": default_nav + cumulative_pnl,
                "day_pnl": day_pnl,
                "cum_pnl": cumulative_pnl,
                "day_turnover": day_turnover,
                "day_trades": 2 * day_accepted,
                "round_trips": day_accepted,
            }
        )
    combined_trades = pd.DataFrame(accepted_rows)
    combined_daily = pd.DataFrame(daily_rows)
    concentration = concentration_metrics(combined_trades, combined_daily)
    folds = replay_fold_metrics(combined_trades, combined_daily)
    pnl = float(combined_trades["pnl"].sum()) if not combined_trades.empty else 0.0
    result = {
        "status": "held_intraday_bidirectional_trigger_combined",
        "components": {
            "buy_first": {
                "path": str(buy_trades_path.resolve()),
                "trigger_distance": buy_trigger,
            },
            "sell_first": {
                "path": str(sell_trades_path.resolve()),
                "trigger_distance": sell_trigger,
            },
            "touch_buffer": touch_buffer,
            "max_lots": max_lots,
        },
        "conflict_policy": "drop_both_same_date_same_instrument",
        "conflicts_dropped": int(len(conflicts)),
        "constraints": {
            "max_daily_turnover": max_daily_turnover,
            "max_symbols_per_day": max_symbols_per_day,
            "buy_first_cash_does_not_use_future_afternoon_sale_proceeds": True,
        },
        "constraint_rejections": {
            "turnover": rejected_turnover,
            "cash": rejected_cash,
            "symbol_cap": rejected_symbol_cap,
        },
        "round_trips": int(len(combined_trades)),
        "orders": int(2 * len(combined_trades)),
        "cum_pnl": pnl,
        "incremental_return": pnl / default_nav,
        "positive_trade_ratio": float((combined_trades["pnl"] > 0).mean())
        if not combined_trades.empty
        else 0.0,
        "cum_turnover": cumulative_turnover,
        "max_daily_turnover": float(combined_daily["day_turnover"].max()),
        "incremental_max_drawdown": max_drawdown(combined_daily["incremental_nav"]),
        "max_overlay_drawdown": max_drawdown(combined_daily["incremental_nav"]),
        **concentration,
        **folds,
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    combined_trades.to_csv(output_trades, index=False, encoding="utf-8-sig")
    combined_daily.to_csv(output_daily, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buy-trades", required=True)
    parser.add_argument("--sell-trades", required=True)
    parser.add_argument("--calendar-daily", required=True)
    parser.add_argument("--daily-account", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-trades", required=True)
    parser.add_argument("--output-daily", required=True)
    parser.add_argument("--buy-trigger", type=float, required=True)
    parser.add_argument("--sell-trigger", type=float, required=True)
    parser.add_argument("--touch-buffer", type=float, default=0.0)
    parser.add_argument("--max-lots", type=int, default=1)
    parser.add_argument("--max-daily-turnover", type=float, default=0.03)
    parser.add_argument("--max-symbols-per-day", type=int, default=3)
    args = parser.parse_args()
    report = combine(
        Path(args.buy_trades).resolve(),
        Path(args.sell_trades).resolve(),
        Path(args.calendar_daily).resolve(),
        Path(args.daily_account).resolve(),
        Path(args.output_json).resolve(),
        Path(args.output_trades).resolve(),
        Path(args.output_daily).resolve(),
        buy_trigger=args.buy_trigger,
        sell_trigger=args.sell_trigger,
        touch_buffer=args.touch_buffer,
        max_lots=args.max_lots,
        max_daily_turnover=args.max_daily_turnover,
        max_symbols_per_day=args.max_symbols_per_day,
    )
    print(
        "[bidirectional trigger] "
        f"pnl={report['cum_pnl']:.2f} return={report['incremental_return']:.4%} "
        f"trips={report['round_trips']} profitable_folds={report['profitable_folds']} "
        f"worst_fold={report['worst_fold_pnl']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
