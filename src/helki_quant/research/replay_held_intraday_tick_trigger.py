from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_held_intraday_decision_dataset import load_minute_for_symbol, to_raw_inst  # noqa: E402
from build_minute_staging import build_minute_source_index  # noqa: E402
from replay_held_intraday_t0 import (  # noqa: E402
    concentration_metrics,
    estimate_fee,
    max_drawdown,
    min_lot,
    parse_float_list,
    parse_int_list,
    replay_fold_metrics,
)


def load_trigger_extrema(
    trades: pd.DataFrame,
    stage_dir: Path | None,
    window_end_minute: int,
) -> dict[tuple[str, pd.Timestamp], tuple[float, float]]:
    missing_stage = [
        inst
        for inst in trades["instrument"].drop_duplicates()
        if stage_dir is None or not (stage_dir / f"{to_raw_inst(inst)}.csv").exists()
    ]
    # Building the full 5k-symbol archive index is worthwhile for a portfolio,
    # but needlessly expensive for one or two candidates. With no index,
    # files_for_instrument performs a targeted lookup for that symbol.
    source_index = build_minute_source_index() if len(missing_stage) > 2 else None
    extrema: dict[tuple[str, pd.Timestamp], tuple[float, float]] = {}
    instruments = trades["instrument"].drop_duplicates().tolist()
    for pos, (inst, part) in enumerate(trades.groupby("instrument", sort=True), start=1):
        minute = load_minute_for_symbol(
            to_raw_inst(inst),
            part["trade_date"].min(),
            part["trade_date"].max(),
            stage_dir,
            False,
            source_index,
        )
        if minute.empty:
            continue
        decision_minute = int(part["decision_time"].iloc[0][:2]) * 60 + int(
            part["decision_time"].iloc[0][2:]
        )
        window = minute[
            (minute["minute_of_day"] > decision_minute)
            & (minute["minute_of_day"] <= window_end_minute)
        ]
        for trade_date, day in window.groupby("trade_date", sort=False):
            extrema[(inst, pd.Timestamp(trade_date).normalize())] = (
                float(pd.to_numeric(day["low"], errors="coerce").min()),
                float(pd.to_numeric(day["high"], errors="coerce").max()),
            )
        if pos % 20 == 0:
            print(f"[tick trigger] minute sources {pos}/{len(instruments)}", flush=True)
    return extrema


def select_setting(frame: pd.DataFrame, threshold: float, daily_top_n: int) -> pd.DataFrame:
    threshold_values = pd.to_numeric(frame["threshold"], errors="coerce")
    top_values = pd.to_numeric(frame["daily_top_n"], errors="coerce")
    return frame[
        np.isclose(threshold_values, threshold, rtol=0.0, atol=1e-12)
        & (top_values == daily_top_n)
    ].copy()


def run_trigger_replay(
    base_trades_path: Path,
    base_daily_path: Path,
    daily_account_path: Path,
    output_json: Path,
    output_trades: Path,
    output_daily: Path,
    *,
    threshold: float,
    daily_top_n: int,
    trade_direction: str,
    trigger_distances: list[float],
    max_lots_grid: list[int],
    touch_buffers: list[float],
    stage_dir: Path | None,
    window_end_minute: int,
    max_inventory_fraction: float,
    max_daily_turnover: float,
    lot_size: int,
    buy_cost: float,
    sell_cost: float,
    slippage: float,
    min_cost: float,
) -> dict:
    trades = pd.read_csv(base_trades_path, parse_dates=["trade_date"])
    daily_template = pd.read_csv(base_daily_path, parse_dates=["trade_date"])
    account = pd.read_csv(daily_account_path, parse_dates=["trade_date"])
    for frame in (trades, daily_template, account):
        frame["trade_date"] = frame["trade_date"].dt.normalize()
    trades["instrument"] = trades["instrument"].astype(str).str.upper()
    trades["decision_time"] = (
        trades["decision_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    )
    trades = select_setting(trades, threshold, daily_top_n)
    daily_template = select_setting(daily_template, threshold, daily_top_n)
    if trades.empty or daily_template.empty:
        raise ValueError("selected base replay setting has no trades or daily rows")
    extrema = load_trigger_extrema(trades, stage_dir, window_end_minute)
    nav_by_date = account.drop_duplicates("trade_date", keep="last").set_index("trade_date")["nav"]
    cash_by_date = account.drop_duplicates("trade_date", keep="last").set_index("trade_date")["cash"]
    default_nav = float(account["nav"].iloc[0])
    candidates_by_day = {
        date: part.sort_values("score", ascending=False)
        for date, part in trades.groupby("trade_date", sort=True)
    }

    results = []
    all_trades = []
    all_daily = []
    for trigger_distance in trigger_distances:
        for max_lots in max_lots_grid:
            for touch_buffer in touch_buffers:
                cumulative_pnl = 0.0
                cumulative_turnover = 0.0
                trade_rows = []
                daily_rows = []
                for day_row in daily_template.sort_values("trade_date").itertuples(index=False):
                    trade_date = pd.Timestamp(day_row.trade_date).normalize()
                    nav = float(nav_by_date.get(trade_date, day_row.base_nav))
                    cash = float(cash_by_date.get(trade_date, nav))
                    day_pnl = 0.0
                    turnover_value = 0.0
                    morning_cash_used = 0.0
                    for row in candidates_by_day.get(trade_date, pd.DataFrame()).itertuples(index=False):
                        low_high = extrema.get((row.instrument, trade_date))
                        if low_high is None:
                            continue
                        low, high = low_high
                        minimum_lot = min_lot(row.instrument, lot_size)
                        capacity_lots = int(
                            np.floor(float(row.held_shares) * max_inventory_fraction / minimum_lot)
                        )
                        volume = min(max_lots, capacity_lots) * minimum_lot
                        if volume < minimum_lot or volume > int(row.held_shares):
                            continue
                        if trade_direction == "buy_first":
                            entry_reference = float(row.buy_reference_price)
                            exit_reference = float(row.sell_reference_price)
                            entry_limit = entry_reference * (1.0 - trigger_distance)
                            touched = low <= entry_limit * (1.0 - touch_buffer)
                            buy_price = entry_limit
                            sell_price = exit_reference * (1.0 - slippage)
                        elif trade_direction == "sell_first":
                            entry_reference = float(row.sell_reference_price)
                            exit_reference = float(row.buy_reference_price)
                            entry_limit = entry_reference * (1.0 + trigger_distance)
                            touched = high >= entry_limit * (1.0 + touch_buffer)
                            sell_price = entry_limit
                            buy_price = exit_reference * (1.0 + slippage)
                        else:
                            raise ValueError(f"unsupported trade_direction: {trade_direction}")
                        if not touched or buy_price <= 0 or sell_price <= 0:
                            continue
                        buy_value = volume * buy_price
                        sell_value = volume * sell_price
                        buy_fee = estimate_fee(buy_value, buy_cost, min_cost)
                        sell_fee = estimate_fee(sell_value, sell_cost, min_cost)
                        if trade_direction == "buy_first":
                            if cash - morning_cash_used < buy_value + buy_fee:
                                continue
                            morning_cash_used += buy_value + buy_fee
                        candidate_turnover = buy_value + sell_value
                        if (turnover_value + candidate_turnover) / max(nav, 1e-12) > max_daily_turnover:
                            continue
                        pnl = sell_value - sell_fee - buy_value - buy_fee
                        day_pnl += pnl
                        turnover_value += candidate_turnover
                        trade_rows.append(
                            {
                                "trigger_distance": trigger_distance,
                                "touch_buffer": touch_buffer,
                                "max_lots": max_lots,
                                "trade_direction": trade_direction,
                                "trade_date": trade_date.strftime("%Y-%m-%d"),
                                "fold": int(row.fold),
                                "instrument": row.instrument,
                                "decision_time": row.decision_time,
                                "score": float(row.score),
                                "held_shares": int(row.held_shares),
                                "t0_volume": int(volume),
                                "entry_limit": entry_limit,
                                "window_low": low,
                                "window_high": high,
                                "buy_price": buy_price,
                                "sell_price": sell_price,
                                "buy_fee": buy_fee,
                                "sell_fee": sell_fee,
                                "pnl": pnl,
                                "turnover_value": candidate_turnover,
                            }
                        )
                    cumulative_pnl += day_pnl
                    day_turnover = turnover_value / max(nav, 1e-12)
                    cumulative_turnover += day_turnover
                    daily_rows.append(
                        {
                            "trigger_distance": trigger_distance,
                            "touch_buffer": touch_buffer,
                            "max_lots": max_lots,
                            "trade_direction": trade_direction,
                            "trade_date": trade_date.strftime("%Y-%m-%d"),
                            "fold": int(day_row.fold),
                            "base_nav": nav,
                            "incremental_nav": default_nav + cumulative_pnl,
                            "day_pnl": day_pnl,
                            "cum_pnl": cumulative_pnl,
                            "day_turnover": day_turnover,
                            "day_trades": 2 * sum(
                                row["trade_date"] == trade_date.strftime("%Y-%m-%d")
                                for row in trade_rows
                            ),
                        }
                    )
                replay_trades = pd.DataFrame(trade_rows)
                replay_daily = pd.DataFrame(daily_rows)
                concentration = concentration_metrics(replay_trades, replay_daily)
                fold_stats = replay_fold_metrics(replay_trades, replay_daily)
                pnl = float(replay_trades["pnl"].sum()) if not replay_trades.empty else 0.0
                result = {
                    "trigger_distance": trigger_distance,
                    "touch_buffer": touch_buffer,
                    "max_lots": max_lots,
                    "trade_direction": trade_direction,
                    "round_trips": int(len(replay_trades)),
                    "orders": int(2 * len(replay_trades)),
                    "cum_pnl": pnl,
                    "incremental_return": pnl / default_nav,
                    "profit_factor": concentration["profit_factor"],
                    "positive_trade_ratio": float((replay_trades["pnl"] > 0).mean())
                    if not replay_trades.empty
                    else 0.0,
                    "cum_turnover": cumulative_turnover,
                    "max_daily_turnover": float(replay_daily["day_turnover"].max()),
                    "incremental_max_drawdown": max_drawdown(replay_daily["incremental_nav"]),
                    **{key: value for key, value in concentration.items() if key != "profit_factor"},
                    **fold_stats,
                }
                results.append(result)
                all_trades.extend(trade_rows)
                all_daily.extend(daily_rows)
    ranked = sorted(results, key=lambda row: (row["cum_pnl"], row["round_trips"]), reverse=True)
    report = {
        "status": "held_intraday_tick_trigger_replayed",
        "base_trades": str(base_trades_path.resolve()),
        "base_daily": str(base_daily_path.resolve()),
        "daily_account": str(daily_account_path.resolve()),
        "selected_base_setting": {"threshold": threshold, "daily_top_n": daily_top_n},
        "entry_window_end_minute": window_end_minute,
        "constraints": {
            "max_inventory_fraction": max_inventory_fraction,
            "max_daily_turnover": max_daily_turnover,
            "buy_cost": buy_cost,
            "sell_cost": sell_cost,
            "afternoon_slippage": slippage,
            "min_cost": min_cost,
            "limit_touch_fill_assumption": "fill_at_limit_only_after_minute_extreme_crosses_limit_plus_touch_buffer",
        },
        "grid": {
            "trigger_distances": trigger_distances,
            "max_lots": max_lots_grid,
            "touch_buffers": touch_buffers,
        },
        "best": ranked[0] if ranked else None,
        "results": ranked,
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(all_trades).to_csv(output_trades, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_daily).to_csv(output_daily, index=False, encoding="utf-8-sig")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-trades", required=True)
    parser.add_argument("--base-daily", required=True)
    parser.add_argument("--daily-account", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-trades", required=True)
    parser.add_argument("--output-daily", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--daily-top-n", type=int, required=True)
    parser.add_argument("--trade-direction", choices=["buy_first", "sell_first"], required=True)
    parser.add_argument("--trigger-distances", default="0.004,0.005,0.0075")
    parser.add_argument("--max-lots", default="1,2")
    parser.add_argument("--touch-buffers", default="0.0,0.001")
    parser.add_argument("--stage-dir", default=None)
    parser.add_argument("--window-end-minute", type=int, default=660)
    parser.add_argument("--max-inventory-fraction", type=float, default=0.5)
    parser.add_argument("--max-daily-turnover", type=float, default=0.03)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--buy-cost", type=float, default=0.001)
    parser.add_argument("--sell-cost", type=float, default=0.0025)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--min-cost", type=float, default=5.0)
    args = parser.parse_args()
    report = run_trigger_replay(
        Path(args.base_trades).resolve(),
        Path(args.base_daily).resolve(),
        Path(args.daily_account).resolve(),
        Path(args.output_json).resolve(),
        Path(args.output_trades).resolve(),
        Path(args.output_daily).resolve(),
        threshold=args.threshold,
        daily_top_n=args.daily_top_n,
        trade_direction=args.trade_direction,
        trigger_distances=parse_float_list(args.trigger_distances),
        max_lots_grid=parse_int_list(args.max_lots),
        touch_buffers=parse_float_list(args.touch_buffers),
        stage_dir=Path(args.stage_dir).resolve() if args.stage_dir else None,
        window_end_minute=args.window_end_minute,
        max_inventory_fraction=args.max_inventory_fraction,
        max_daily_turnover=args.max_daily_turnover,
        lot_size=args.lot_size,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        slippage=args.slippage,
        min_cost=args.min_cost,
    )
    best = report.get("best") or {}
    print(
        "[tick trigger] "
        f"trigger={best.get('trigger_distance')} touch={best.get('touch_buffer')} "
        f"max_lots={best.get('max_lots')} pnl={best.get('cum_pnl')} "
        f"trips={best.get('round_trips')} folds={best.get('profitable_folds')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
