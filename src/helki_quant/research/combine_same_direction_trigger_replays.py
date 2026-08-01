from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from replay_held_intraday_t0 import (
    concentration_metrics,
    max_drawdown,
    replay_fold_metrics,
)


def _load_component(path: Path, name: str, priority: int) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    required = {
        "trade_date",
        "instrument",
        "trade_direction",
        "score",
        "pnl",
        "turnover_value",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"component {name} missing columns: {missing}")
    frame["trade_date"] = frame["trade_date"].dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame["component"] = name
    frame["component_priority"] = priority
    return frame


def combine(
    primary_trades_path: Path,
    secondary_trades_path: Path,
    calendar_daily_path: Path,
    daily_account_path: Path,
    output_json: Path,
    output_trades: Path,
    output_daily: Path,
    *,
    primary_name: str,
    secondary_name: str,
    max_daily_turnover: float,
    max_symbols_per_day: int,
) -> dict:
    primary = _load_component(primary_trades_path, primary_name, 0)
    secondary = _load_component(secondary_trades_path, secondary_name, 1)
    directions = set(primary["trade_direction"]) | set(secondary["trade_direction"])
    if len(directions) != 1:
        raise ValueError(f"components must have one shared direction, got {directions}")
    conflict_keys = primary[["trade_date", "instrument"]].drop_duplicates()
    conflict_index = pd.MultiIndex.from_frame(conflict_keys)
    secondary_keys = pd.MultiIndex.from_frame(secondary[["trade_date", "instrument"]])
    conflict_mask = secondary_keys.isin(conflict_index)
    conflicts_dropped = int(conflict_mask.sum())
    secondary = secondary.loc[~conflict_mask].copy()
    candidates = pd.concat([primary, secondary], ignore_index=True, sort=False).sort_values(
        ["trade_date", "component_priority", "score", "instrument"],
        ascending=[True, True, False, True],
    )

    calendar = pd.read_csv(calendar_daily_path, parse_dates=["trade_date"])
    calendar["trade_date"] = calendar["trade_date"].dt.normalize()
    calendar = calendar.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    account = pd.read_csv(daily_account_path, parse_dates=["trade_date"])
    account["trade_date"] = account["trade_date"].dt.normalize()
    account = account.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    default_nav = float(account["nav"].iloc[0])
    by_date = {date: part for date, part in candidates.groupby("trade_date", sort=True)}

    accepted = []
    daily_rows = []
    cumulative_pnl = 0.0
    cumulative_turnover = 0.0
    rejected_turnover = 0
    rejected_symbol_cap = 0
    for day in calendar.itertuples(index=False):
        trade_date = pd.Timestamp(day.trade_date).normalize()
        nav = (
            float(account.loc[trade_date, "nav"])
            if trade_date in account.index
            else float(day.base_nav)
        )
        turnover_value = 0.0
        day_pnl = 0.0
        symbols: set[str] = set()
        day_round_trips = 0
        for row in by_date.get(trade_date, pd.DataFrame()).itertuples(index=False):
            if row.instrument in symbols:
                continue
            if len(symbols) >= max_symbols_per_day:
                rejected_symbol_cap += 1
                continue
            candidate_turnover = float(row.turnover_value)
            if (turnover_value + candidate_turnover) / max(nav, 1e-12) > max_daily_turnover:
                rejected_turnover += 1
                continue
            record = row._asdict()
            accepted.append(record)
            symbols.add(row.instrument)
            turnover_value += candidate_turnover
            day_pnl += float(row.pnl)
            day_round_trips += 1
        cumulative_pnl += day_pnl
        day_turnover = turnover_value / max(nav, 1e-12)
        cumulative_turnover += day_turnover
        daily_rows.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "fold": int(day.fold),
                "base_nav": nav,
                "incremental_nav": default_nav + cumulative_pnl,
                "overlay_nav": nav + cumulative_pnl,
                "day_pnl": day_pnl,
                "cum_pnl": cumulative_pnl,
                "day_turnover": day_turnover,
                "day_trades": 2 * day_round_trips,
                "round_trips": day_round_trips,
            }
        )
    trades = pd.DataFrame(accepted)
    daily = pd.DataFrame(daily_rows)
    concentration = concentration_metrics(trades, daily)
    folds = replay_fold_metrics(trades, daily)
    pnl = float(trades["pnl"].sum()) if not trades.empty else 0.0
    base_drawdown = max_drawdown(daily["base_nav"])
    overlay_drawdown = max_drawdown(daily["overlay_nav"])
    result = {
        "status": "same_direction_trigger_components_combined",
        "components": {
            "primary": {
                "name": primary_name,
                "path": str(primary_trades_path.resolve()),
                "input_trades": int(len(primary)),
            },
            "secondary": {
                "name": secondary_name,
                "path": str(secondary_trades_path.resolve()),
                "input_trades": int(len(secondary) + conflicts_dropped),
            },
        },
        "trade_direction": next(iter(directions)),
        "conflict_policy": "keep_primary_drop_secondary_same_trade_date_symbol",
        "conflicts_dropped": conflicts_dropped,
        "constraints": {
            "max_daily_turnover": max_daily_turnover,
            "max_symbols_per_day": max_symbols_per_day,
        },
        "constraint_rejections": {
            "turnover": rejected_turnover,
            "symbol_cap": rejected_symbol_cap,
        },
        "round_trips": int(len(trades)),
        "orders": int(2 * len(trades)),
        "cum_pnl": pnl,
        "incremental_return": pnl / default_nav,
        "positive_trade_ratio": float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0,
        "cum_turnover": cumulative_turnover,
        "max_daily_turnover": float(daily["day_turnover"].max()),
        "base_max_drawdown": base_drawdown,
        "incremental_max_drawdown": max_drawdown(daily["incremental_nav"]),
        "max_overlay_drawdown": overlay_drawdown,
        "overlay_drawdown_delta": overlay_drawdown - base_drawdown,
        **concentration,
        **folds,
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output_trades.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_trades, index=False, encoding="utf-8-sig")
    output_daily.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_daily, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-trades", required=True)
    parser.add_argument("--secondary-trades", required=True)
    parser.add_argument("--calendar-daily", required=True)
    parser.add_argument("--daily-account", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-trades", required=True)
    parser.add_argument("--output-daily", required=True)
    parser.add_argument("--primary-name", default="primary")
    parser.add_argument("--secondary-name", default="secondary")
    parser.add_argument("--max-daily-turnover", type=float, default=0.03)
    parser.add_argument("--max-symbols-per-day", type=int, default=4)
    args = parser.parse_args()
    report = combine(
        Path(args.primary_trades).resolve(),
        Path(args.secondary_trades).resolve(),
        Path(args.calendar_daily).resolve(),
        Path(args.daily_account).resolve(),
        Path(args.output_json).resolve(),
        Path(args.output_trades).resolve(),
        Path(args.output_daily).resolve(),
        primary_name=args.primary_name,
        secondary_name=args.secondary_name,
        max_daily_turnover=args.max_daily_turnover,
        max_symbols_per_day=args.max_symbols_per_day,
    )
    print(
        f"[same-direction combo] pnl={report['cum_pnl']:.2f} "
        f"trips={report['round_trips']} pf={report['profit_factor']} "
        f"conflicts={report['conflicts_dropped']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
