from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def min_lot(symbol: str, lot_size: int) -> int:
    code = str(symbol).upper()[-6:]
    if code.startswith(("688", "689")):
        return max(200, lot_size)
    return lot_size


def round_lot(volume: float, lot_size: int) -> int:
    if not np.isfinite(volume) or volume <= 0:
        return 0
    return int(volume // lot_size) * lot_size


def estimate_fee(value: float, rate: float, min_cost: float) -> float:
    return max(value * rate, min_cost) if value > 0 else 0.0


def max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    return float((1.0 - nav / nav.cummax()).max())


def concentration_metrics(trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "profit_factor": None,
            "symbols_traded": 0,
            "active_months": 0,
            "losing_months": 0,
            "top_symbol_positive_pnl_share": None,
            "top3_positive_pnl_share": None,
        }
    gross_pos = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_neg = float(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    symbol_pos = (
        trades.assign(pos_pnl=trades["pnl"].clip(lower=0))
        .groupby("instrument")["pos_pnl"]
        .sum()
        .sort_values(ascending=False)
    )
    month = (
        daily.assign(month=pd.to_datetime(daily["trade_date"]).dt.to_period("M").astype(str))
        .groupby("month")
        .agg(day_pnl=("day_pnl", "sum"), trades=("day_trades", "sum"))
    )
    return {
        "profit_factor": float(gross_pos / abs(gross_neg)) if gross_neg < 0 else None,
        "symbols_traded": int(trades["instrument"].nunique()),
        "active_months": int((month["trades"] > 0).sum()) if len(month) else 0,
        "losing_months": int((month["day_pnl"] < 0).sum()) if len(month) else 0,
        "top_symbol_positive_pnl_share": float(symbol_pos.iloc[0] / gross_pos)
        if gross_pos > 0 and len(symbol_pos)
        else None,
        "top3_positive_pnl_share": float(symbol_pos.head(3).sum() / gross_pos)
        if gross_pos > 0 and len(symbol_pos)
        else None,
    }


def replay_fold_metrics(trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    if "fold" not in daily.columns or daily["fold"].isna().all():
        return {
            "folds": [],
            "fold_count": 0,
            "profitable_folds": 0,
            "worst_fold_pnl": None,
            "worst_fold_return": None,
        }
    rows = []
    for fold, fold_daily in daily.dropna(subset=["fold"]).groupby("fold", sort=True):
        fold_trades = trades[trades["fold"] == fold] if not trades.empty else trades
        pnl = float(fold_daily["day_pnl"].sum())
        base_nav = float(fold_daily["base_nav"].iloc[0])
        rows.append(
            {
                "fold": int(fold),
                "pnl": pnl,
                "return": pnl / max(base_nav, 1e-12),
                "round_trips": int(len(fold_trades)),
                "active_days": int((fold_daily["day_trades"] > 0).sum()),
                "positive_trade_ratio": float((fold_trades["pnl"] > 0).mean())
                if not fold_trades.empty
                else 0.0,
            }
        )
    return {
        "folds": rows,
        "fold_count": len(rows),
        "profitable_folds": sum(row["pnl"] > 0 for row in rows),
        "worst_fold_pnl": min((row["pnl"] for row in rows), default=None),
        "worst_fold_return": min((row["return"] for row in rows), default=None),
    }


def load_daily_nav(daily_account_path: Path | None, dates: pd.Series, default_nav: float) -> pd.DataFrame:
    if daily_account_path and daily_account_path.exists():
        daily = pd.read_csv(daily_account_path, parse_dates=["trade_date"])
        daily["trade_date"] = daily["trade_date"].dt.normalize()
        keep = [col for col in ["trade_date", "nav", "cash"] if col in daily.columns]
        daily = daily[keep].drop_duplicates("trade_date", keep="last")
        if "cash" not in daily.columns:
            daily["cash"] = np.nan
        return daily
    unique_dates = pd.DataFrame({"trade_date": sorted(pd.to_datetime(dates).dt.normalize().unique())})
    unique_dates["nav"] = float(default_nav)
    unique_dates["cash"] = float(default_nav)
    return unique_dates


def run_replay(
    prediction_csv: Path,
    decision_dataset_csv: Path,
    output_json: Path,
    output_trades: Path,
    output_daily: Path,
    *,
    thresholds: list[float],
    trade_fractions: list[float],
    daily_account_path: Path | None,
    default_nav: float,
    lot_size: int,
    buy_cost: float,
    sell_cost: float,
    slippage: float,
    min_cost: float,
    max_daily_turnover: float,
    max_symbols_per_day: int,
    max_round_trips_per_symbol: int,
    selection_mode: str = "threshold",
    daily_top_ns: list[int] | None = None,
    sizing_mode: str = "fraction",
    max_inventory_fraction: float = 0.5,
    trade_direction: str = "sell_first",
    buyback_window: str | None = None,
) -> dict:
    pred = pd.read_csv(prediction_csv, parse_dates=["trade_date", "datetime"])
    data = pd.read_csv(decision_dataset_csv, parse_dates=["trade_date", "datetime"])
    for frame in (pred, data):
        frame["trade_date"] = frame["trade_date"].dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str).str.upper()
        frame["decision_time"] = frame["decision_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    merge_cols = ["trade_date", "instrument", "decision_time"]
    pred_cols = merge_cols + ["score"]
    if "fold" in pred.columns:
        pred_cols.append("fold")
    base = pred[pred_cols].merge(data, on=merge_cols, how="inner")
    base = base.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["score", "shares", "sell_price_decision"]
    )
    daily_nav = load_daily_nav(daily_account_path, base["trade_date"], default_nav)
    daily_nav = daily_nav.sort_values("trade_date")
    nav_by_date = {row.trade_date: float(row.nav) for row in daily_nav.itertuples(index=False)}
    cash_by_date = {row.trade_date: float(row.cash) for row in daily_nav.itertuples(index=False)}

    buyback_cols = [col for col in base.columns if col.startswith("buyback_") and col.endswith("_price")]
    if not buyback_cols:
        raise KeyError("decision dataset has no buyback_*_price columns")
    # The prediction file is already for a fixed label, but the replay may choose the
    # matching buyback price by output name. Fall back to the first non-null buyback
    # only if no explicit column is provided in the CSV name.
    name = prediction_csv.stem
    if buyback_window:
        buyback_col = f"buyback_{buyback_window}_price"
    elif "1420" in name:
        buyback_col = "buyback_1420_1430_price"
    elif "1445" in name:
        buyback_col = "buyback_1445_1450_price"
    elif "1450" in name:
        buyback_col = "buyback_1450_1455_price"
    else:
        buyback_col = buyback_cols[0]
    if buyback_col not in base.columns:
        raise KeyError(f"buyback col not found: {buyback_col}")
    base = base.dropna(subset=[buyback_col]).copy()

    all_results = []
    all_trades = []
    all_daily = []
    by_day = {date: part for date, part in base.groupby("trade_date", sort=True)}
    daily_top_ns = daily_top_ns or []
    if selection_mode == "threshold":
        selectors = [(threshold, None) for threshold in thresholds]
    elif selection_mode == "daily_top_n":
        selectors = [(threshold, top_n) for threshold in thresholds for top_n in daily_top_ns]
    elif selection_mode == "both":
        selectors = [(threshold, None) for threshold in thresholds]
        selectors.extend((threshold, top_n) for threshold in thresholds for top_n in daily_top_ns)
    else:
        raise ValueError(f"unsupported selection_mode: {selection_mode}")
    if not selectors:
        raise ValueError("selection grid is empty")
    for threshold, daily_top_n in selectors:
        for fraction in trade_fractions:
            cum_pnl = 0.0
            cum_turnover = 0.0
            trade_rows = []
            daily_rows = []
            symbol_round_trips: dict[str, int] = {}
            for trade_date in sorted(by_day):
                nav = nav_by_date.get(trade_date, default_nav)
                cash = cash_by_date.get(trade_date, nav)
                candidates = by_day[trade_date]
                selected = candidates[candidates["score"] >= threshold].sort_values("score", ascending=False)
                if daily_top_n is not None:
                    selected = selected.head(daily_top_n)
                day_pnl = 0.0
                day_turnover_value = 0.0
                day_trades = 0
                morning_cash_used = 0.0
                used_symbols = set()
                for row in selected.itertuples(index=False):
                    if len(used_symbols) >= max_symbols_per_day:
                        break
                    if row.instrument in used_symbols:
                        continue
                    if max_round_trips_per_symbol > 0 and symbol_round_trips.get(row.instrument, 0) >= max_round_trips_per_symbol:
                        continue
                    if day_turnover_value / max(nav, 1e-12) >= max_daily_turnover:
                        break
                    min_volume = min_lot(row.instrument, lot_size)
                    if sizing_mode == "one_lot":
                        volume = min_volume
                        if volume > float(row.shares) * max_inventory_fraction:
                            continue
                    elif sizing_mode == "fraction":
                        volume = round_lot(float(row.shares) * fraction, lot_size)
                    else:
                        raise ValueError(f"unsupported sizing_mode: {sizing_mode}")
                    if volume < min_volume or volume > int(row.shares):
                        continue
                    decision_reference_price = float(row.sell_price_decision)
                    afternoon_reference_price = float(getattr(row, buyback_col))
                    if decision_reference_price <= 0 or afternoon_reference_price <= 0:
                        continue
                    if trade_direction == "sell_first":
                        sell_reference_price = decision_reference_price
                        buy_reference_price = afternoon_reference_price
                    elif trade_direction == "buy_first":
                        buy_reference_price = decision_reference_price
                        sell_reference_price = afternoon_reference_price
                    else:
                        raise ValueError(f"unsupported trade_direction: {trade_direction}")
                    sell_price = sell_reference_price * (1.0 - slippage)
                    buy_price = buy_reference_price * (1.0 + slippage)
                    sell_value = volume * sell_price
                    sell_fee = estimate_fee(sell_value, sell_cost, min_cost)
                    buy_value = volume * buy_price
                    buy_fee = estimate_fee(buy_value, buy_cost, min_cost)
                    if trade_direction == "sell_first":
                        available_cash = cash + sell_value - sell_fee + day_pnl
                        if available_cash < buy_value + buy_fee:
                            continue
                    else:
                        available_cash = cash - morning_cash_used
                        if available_cash < buy_value + buy_fee:
                            continue
                    turnover_value = sell_value + buy_value
                    if (day_turnover_value + turnover_value) / max(nav, 1e-12) > max_daily_turnover:
                        continue
                    pnl = sell_value - sell_fee - buy_value - buy_fee
                    if trade_direction == "buy_first":
                        morning_cash_used += buy_value + buy_fee
                    day_pnl += pnl
                    day_turnover_value += turnover_value
                    day_trades += 2
                    used_symbols.add(row.instrument)
                    symbol_round_trips[row.instrument] = symbol_round_trips.get(row.instrument, 0) + 1
                    trade_rows.append(
                        {
                            "threshold": threshold,
                            "selection_mode": "daily_top_n" if daily_top_n is not None else "threshold",
                            "daily_top_n": daily_top_n,
                            "trade_fraction": fraction,
                            "trade_direction": trade_direction,
                            "trade_date": trade_date.strftime("%Y-%m-%d"),
                            "instrument": row.instrument,
                            "decision_time": row.decision_time,
                            "score": float(row.score),
                            "fold": int(row.fold) if hasattr(row, "fold") and pd.notna(row.fold) else np.nan,
                            "held_shares": int(row.shares),
                            "t0_volume": int(volume),
                            "sell_price": sell_price,
                            "buy_price": buy_price,
                            "sell_reference_price": sell_reference_price,
                            "buy_reference_price": buy_reference_price,
                            "sell_fee": sell_fee,
                            "buy_fee": buy_fee,
                            "pnl": pnl,
                            "turnover_value": turnover_value,
                        }
                    )
                cum_pnl += day_pnl
                day_turnover = day_turnover_value / max(nav, 1e-12)
                cum_turnover += day_turnover
                daily_rows.append(
                    {
                        "threshold": threshold,
                        "selection_mode": "daily_top_n" if daily_top_n is not None else "threshold",
                        "daily_top_n": daily_top_n,
                        "trade_fraction": fraction,
                        "trade_date": trade_date.strftime("%Y-%m-%d"),
                        "fold": int(candidates["fold"].iloc[0])
                        if "fold" in candidates.columns and candidates["fold"].notna().any()
                        else np.nan,
                        "base_nav": nav,
                        "overlay_nav": nav + cum_pnl,
                        "incremental_nav": default_nav + cum_pnl,
                        "day_pnl": day_pnl,
                        "cum_pnl": cum_pnl,
                        "day_turnover": day_turnover,
                        "day_trades": day_trades,
                    }
                )
            trades = pd.DataFrame(trade_rows)
            daily = pd.DataFrame(daily_rows)
            conc = concentration_metrics(trades, daily)
            fold_stats = replay_fold_metrics(trades, daily)
            result = {
                "threshold": threshold,
                "selection_mode": "daily_top_n" if daily_top_n is not None else "threshold",
                "daily_top_n": daily_top_n,
                "trade_fraction": fraction,
                "sizing_mode": sizing_mode,
                "max_inventory_fraction": max_inventory_fraction,
                "trade_direction": trade_direction,
                "round_trips": int(len(trades)),
                "orders": int(len(trades) * 2),
                "cum_pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
                "incremental_return": float((trades["pnl"].sum() if not trades.empty else 0.0) / default_nav),
                "cum_turnover": float(cum_turnover),
                "max_daily_turnover": float(daily["day_turnover"].max()) if not daily.empty else 0.0,
                "positive_trade_ratio": float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0,
                "avg_trade_pnl": float(trades["pnl"].mean()) if not trades.empty else 0.0,
                "max_overlay_drawdown": max_drawdown(daily["overlay_nav"]) if not daily.empty else 0.0,
                "incremental_max_drawdown": max_drawdown(daily["incremental_nav"])
                if not daily.empty
                else 0.0,
                **conc,
                **fold_stats,
            }
            all_results.append(result)
            if not trades.empty:
                all_trades.extend(trade_rows)
            all_daily.extend(daily_rows)
    ranked = sorted(
        all_results,
        key=lambda row: (
            row["cum_pnl"],
            row["round_trips"],
            -(row["max_overlay_drawdown"] or 0.0),
        ),
        reverse=True,
    )
    report = {
        "status": "held_intraday_t0_replayed",
        "prediction_csv": str(prediction_csv.resolve()),
        "decision_dataset_csv": str(decision_dataset_csv.resolve()),
        "buyback_col": buyback_col,
        "buyback_window_explicit": buyback_window is not None,
        "base_rows": int(len(base)),
        "thresholds": thresholds,
        "trade_fractions": trade_fractions,
        "selection_mode": selection_mode,
        "daily_top_ns": daily_top_ns,
        "constraints": {
            "lot_size": lot_size,
            "buy_cost": buy_cost,
            "sell_cost": sell_cost,
            "slippage": slippage,
            "min_cost": min_cost,
            "max_daily_turnover": max_daily_turnover,
            "max_symbols_per_day": max_symbols_per_day,
            "max_round_trips_per_symbol": max_round_trips_per_symbol,
            "sizing_mode": sizing_mode,
            "max_inventory_fraction": max_inventory_fraction,
            "trade_direction": trade_direction,
            "no_naked_t0": True,
            "sell_first_buy_back_same_day": True,
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
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--decision-dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-trades", required=True)
    parser.add_argument("--output-daily", required=True)
    parser.add_argument("--daily-account", default="")
    parser.add_argument("--default-nav", type=float, default=1_000_000.0)
    parser.add_argument("--thresholds", default="0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--trade-fractions", default="0.05,0.10,0.20")
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--buy-cost", type=float, default=0.001)
    parser.add_argument("--sell-cost", type=float, default=0.0025)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--min-cost", type=float, default=5.0)
    parser.add_argument("--max-daily-turnover", type=float, default=0.10)
    parser.add_argument("--max-symbols-per-day", type=int, default=20)
    parser.add_argument("--max-round-trips-per-symbol", type=int, default=0)
    parser.add_argument("--selection-mode", choices=["threshold", "daily_top_n", "both"], default="threshold")
    parser.add_argument("--daily-top-ns", default="")
    parser.add_argument("--sizing-mode", choices=["fraction", "one_lot"], default="fraction")
    parser.add_argument("--max-inventory-fraction", type=float, default=0.5)
    parser.add_argument("--trade-direction", choices=["sell_first", "buy_first"], default="sell_first")
    parser.add_argument(
        "--buyback-window",
        choices=["1420_1430", "1445_1450", "1450_1455"],
        default=None,
    )
    args = parser.parse_args()
    report = run_replay(
        Path(args.prediction_csv).resolve(),
        Path(args.decision_dataset).resolve(),
        Path(args.output_json).resolve(),
        Path(args.output_trades).resolve(),
        Path(args.output_daily).resolve(),
        thresholds=parse_float_list(args.thresholds),
        trade_fractions=parse_float_list(args.trade_fractions),
        daily_account_path=Path(args.daily_account).resolve() if args.daily_account else None,
        default_nav=args.default_nav,
        lot_size=args.lot_size,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        slippage=args.slippage,
        min_cost=args.min_cost,
        max_daily_turnover=args.max_daily_turnover,
        max_symbols_per_day=args.max_symbols_per_day,
        max_round_trips_per_symbol=args.max_round_trips_per_symbol,
        selection_mode=args.selection_mode,
        daily_top_ns=parse_int_list(args.daily_top_ns),
        sizing_mode=args.sizing_mode,
        max_inventory_fraction=args.max_inventory_fraction,
        trade_direction=args.trade_direction,
        buyback_window=args.buyback_window,
    )
    best = report.get("best") or {}
    print(
        "[held intraday replay] "
        f"selection={best.get('selection_mode')} threshold={best.get('threshold')} "
        f"top_n={best.get('daily_top_n')} fraction={best.get('trade_fraction')} "
        f"pnl={best.get('cum_pnl')} trips={best.get('round_trips')} "
        f"symbols={best.get('symbols_traded')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
