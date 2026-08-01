from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_held_intraday_decision_dataset import BUYBACK_WINDOWS, trigger_label_prefix
from replay_held_intraday_t0 import (
    concentration_metrics,
    max_drawdown,
    replay_fold_metrics,
)


def replay(
    predictions_path: Path,
    dataset_path: Path,
    daily_account_path: Path,
    output_json: Path,
    output_trades: Path,
    output_daily: Path,
    *,
    direction: str,
    trigger_distance: float,
    touch_buffer: float,
    buyback_window: str,
    score_column: str,
    gate_score_column: str | None,
    score_threshold: float | None,
    daily_top_n: int,
    max_daily_turnover: float,
) -> dict:
    prefix = trigger_label_prefix(
        direction,
        trigger_distance,
        touch_buffer,
        buyback_window,
    )
    buyback_col = f"buyback_{buyback_window}_price"
    pred = pd.read_csv(predictions_path, parse_dates=["datetime", "trade_date"])
    gate_score_column = gate_score_column or score_column
    for column in (score_column, gate_score_column):
        if column not in pred.columns:
            raise KeyError(f"prediction score column not found: {column}")
    required_data = {
        "datetime",
        "trade_date",
        "instrument",
        "decision_time",
        "t0_exec_volume_one_lot_max50",
        buyback_col,
        f"{prefix}_entry_price",
        f"{prefix}_touched",
        f"{prefix}_realized_pnl",
        f"{prefix}_realized_edge",
    }
    data = pd.read_csv(
        dataset_path,
        usecols=lambda column: column in required_data,
        parse_dates=["datetime", "trade_date"],
    )
    account = pd.read_csv(daily_account_path, parse_dates=["trade_date"])
    for frame in (pred, data, account):
        frame["trade_date"] = frame["trade_date"].dt.normalize()
    for frame in (pred, data):
        frame["instrument"] = frame["instrument"].astype(str).str.upper()
        frame["decision_time"] = (
            frame["decision_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
        )
    keys = ["datetime", "trade_date", "instrument", "decision_time"]
    pred_cols = list(dict.fromkeys(keys + ["fold", score_column, gate_score_column]))
    base = pred[pred_cols].merge(data, on=keys, how="inner")
    base["selection_score"] = pd.to_numeric(base[score_column], errors="coerce")
    base["gate_score"] = pd.to_numeric(base[gate_score_column], errors="coerce")
    if len(base) != len(pred):
        raise ValueError(f"prediction/data merge mismatch predictions={len(pred)} merged={len(base)}")
    account = account.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    default_nav = float(account["nav"].iloc[0])
    eligible = base if score_threshold is None else base[base["gate_score"] >= score_threshold]
    selected = (
        eligible
        .sort_values(
            ["trade_date", "selection_score", "instrument"],
            ascending=[True, False, True],
        )
        .groupby("trade_date", sort=False)
        .head(daily_top_n)
    )
    selected_by_date = {date: part for date, part in selected.groupby("trade_date", sort=True)}

    trades = []
    daily_rows = []
    cumulative_pnl = 0.0
    cumulative_turnover = 0.0
    rejected_cash = 0
    rejected_turnover = 0
    all_dates = pred[["trade_date", "fold"]].drop_duplicates().sort_values("trade_date")
    for day in all_dates.itertuples(index=False):
        trade_date = pd.Timestamp(day.trade_date).normalize()
        if trade_date in account.index:
            nav = float(account.loc[trade_date, "nav"])
            cash = float(account.loc[trade_date, "cash"])
        else:
            nav = default_nav
            cash = default_nav
        day_pnl = 0.0
        turnover_value = 0.0
        buy_cash_used = 0.0
        day_trades = 0
        for row in selected_by_date.get(trade_date, pd.DataFrame()).itertuples(index=False):
            if float(getattr(row, f"{prefix}_touched")) <= 0.5:
                continue
            volume = int(float(row.t0_exec_volume_one_lot_max50))
            entry_price = float(getattr(row, f"{prefix}_entry_price"))
            exit_reference = float(getattr(row, buyback_col))
            exit_price = exit_reference * (0.9995 if direction == "buy_first" else 1.0005)
            candidate_turnover = volume * (entry_price + exit_price)
            if (turnover_value + candidate_turnover) / max(nav, 1e-12) > max_daily_turnover:
                rejected_turnover += 1
                continue
            if direction == "buy_first":
                required_cash = volume * entry_price
                if buy_cash_used + required_cash > cash:
                    rejected_cash += 1
                    continue
                buy_cash_used += required_cash
            pnl = float(getattr(row, f"{prefix}_realized_pnl"))
            edge = float(getattr(row, f"{prefix}_realized_edge"))
            trades.append(
                {
                    "trigger_distance": trigger_distance,
                    "touch_buffer": touch_buffer,
                    "max_lots": 1,
                    "trade_direction": direction,
                    "trade_date": str(trade_date.date()),
                    "fold": int(row.fold),
                    "instrument": row.instrument,
                    "decision_time": row.decision_time,
                    "score": float(row.selection_score),
                    "gate_score": float(row.gate_score),
                    "t0_volume": volume,
                    "entry_limit": entry_price,
                    "buy_price": entry_price if direction == "buy_first" else exit_price,
                    "sell_price": exit_price if direction == "buy_first" else entry_price,
                    "pnl": pnl,
                    "realized_edge": edge,
                    "turnover_value": candidate_turnover,
                }
            )
            day_pnl += pnl
            turnover_value += candidate_turnover
            day_trades += 2
        cumulative_pnl += day_pnl
        day_turnover = turnover_value / max(nav, 1e-12)
        cumulative_turnover += day_turnover
        daily_rows.append(
            {
                "trigger_distance": trigger_distance,
                "touch_buffer": touch_buffer,
                "max_lots": 1,
                "trade_direction": direction,
                "trade_date": str(trade_date.date()),
                "fold": int(day.fold),
                "base_nav": nav,
                "incremental_nav": default_nav + cumulative_pnl,
                "overlay_nav": nav + cumulative_pnl,
                "day_pnl": day_pnl,
                "cum_pnl": cumulative_pnl,
                "day_turnover": day_turnover,
                "day_trades": day_trades,
            }
        )
    trade_frame = pd.DataFrame(trades)
    daily_frame = pd.DataFrame(daily_rows)
    concentration = concentration_metrics(trade_frame, daily_frame)
    folds = replay_fold_metrics(trade_frame, daily_frame)
    pnl = float(trade_frame["pnl"].sum()) if not trade_frame.empty else 0.0
    result = {
        "status": "trigger_aligned_predictions_replayed",
        "predictions": str(predictions_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "daily_account": str(daily_account_path.resolve()),
        "profile": {
            "direction": direction,
            "trigger_distance": trigger_distance,
            "touch_buffer": touch_buffer,
            "buyback_window": buyback_window,
            "score_column": score_column,
            "gate_score_column": gate_score_column,
            "score_threshold": score_threshold,
            "score_gate_enabled": score_threshold is not None,
            "daily_top_n": daily_top_n,
            "max_daily_turnover": max_daily_turnover,
        },
        "candidate_rows": int(len(selected)),
        "candidate_dates": int(selected["trade_date"].nunique()),
        "round_trips": int(len(trade_frame)),
        "orders": int(2 * len(trade_frame)),
        "cum_pnl": pnl,
        "incremental_return": pnl / default_nav,
        "positive_trade_ratio": float((trade_frame["pnl"] > 0).mean())
        if not trade_frame.empty
        else 0.0,
        "cum_turnover": cumulative_turnover,
        "max_daily_turnover": float(daily_frame["day_turnover"].max()),
        "base_max_drawdown": max_drawdown(daily_frame["base_nav"]),
        "incremental_max_drawdown": max_drawdown(daily_frame["incremental_nav"]),
        "max_overlay_drawdown": max_drawdown(daily_frame["overlay_nav"]),
        "overlay_drawdown_delta": (
            max_drawdown(daily_frame["overlay_nav"])
            - max_drawdown(daily_frame["base_nav"])
        ),
        "constraint_rejections": {"cash": rejected_cash, "turnover": rejected_turnover},
        **concentration,
        **folds,
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    trade_frame.to_csv(output_trades, index=False, encoding="utf-8-sig")
    daily_frame.to_csv(output_daily, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--daily-account", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-trades", required=True)
    parser.add_argument("--output-daily", required=True)
    parser.add_argument("--direction", choices=["buy_first", "sell_first"], required=True)
    parser.add_argument("--trigger-distance", type=float, required=True)
    parser.add_argument("--touch-buffer", type=float, default=0.0)
    parser.add_argument(
        "--buyback-window",
        choices=sorted(BUYBACK_WINDOWS),
        default="1445_1450",
    )
    parser.add_argument("--score-column", default="score")
    parser.add_argument(
        "--gate-score-column",
        default=None,
        help="Optional calibrated score used only for the trade/no-trade gate.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--disable-score-gate",
        action="store_true",
        help="Rank the full held cross-section and let the limit-touch condition gate execution.",
    )
    parser.add_argument("--daily-top-n", type=int, required=True)
    parser.add_argument("--max-daily-turnover", type=float, default=0.03)
    args = parser.parse_args()
    report = replay(
        Path(args.predictions).resolve(),
        Path(args.dataset).resolve(),
        Path(args.daily_account).resolve(),
        Path(args.output_json).resolve(),
        Path(args.output_trades).resolve(),
        Path(args.output_daily).resolve(),
        direction=args.direction,
        trigger_distance=args.trigger_distance,
        touch_buffer=args.touch_buffer,
        buyback_window=args.buyback_window,
        score_column=args.score_column,
        gate_score_column=args.gate_score_column,
        score_threshold=None if args.disable_score_gate else args.score_threshold,
        daily_top_n=args.daily_top_n,
        max_daily_turnover=args.max_daily_turnover,
    )
    print(
        "[trigger-aligned replay] "
        f"direction={report['profile']['direction']} pnl={report['cum_pnl']:.2f} "
        f"trips={report['round_trips']} pf={report['profit_factor']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
