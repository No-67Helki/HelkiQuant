from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"


def read_calendar(path: Path) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.read_csv(path, header=None, names=["date"], parse_dates=["date"])["date"]
    ).drop_duplicates().sort_values()


def next_trade_date_map(calendar: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp]:
    return dict(zip(calendar[:-1], calendar[1:]))


def load_predictions(prediction_dir: Path, calendar_path: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(prediction_dir.glob("fold_*.csv")):
        if path.stem.endswith("_model"):
            continue
        frame = pd.read_csv(path, parse_dates=["datetime"])
        score_col = [col for col in frame.columns if col not in {"datetime", "instrument"}][0]
        frame = frame.rename(columns={score_col: "score"})
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no fold prediction csv found in {prediction_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["signal_date"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    next_map = next_trade_date_map(read_calendar(calendar_path))
    out["trade_date"] = out["signal_date"].map(next_map)
    out = out.dropna(subset=["trade_date", "score"]).copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    return out[["signal_date", "trade_date", "instrument", "score"]]


def round_lot(volume: float, lot_size: int) -> int:
    if not np.isfinite(volume) or volume <= 0:
        return 0
    return int(volume // lot_size) * lot_size


def min_lot(symbol: str, lot_size: int) -> int:
    code = str(symbol).upper()[-6:]
    if code.startswith(("688", "689")):
        return max(200, lot_size)
    return lot_size


def estimate_buy_fee(value: float, buy_cost: float, min_cost: float) -> float:
    return max(value * buy_cost, min_cost) if value > 0 else 0.0


def estimate_sell_fee(value: float, sell_cost: float, min_cost: float) -> float:
    return max(value * sell_cost, min_cost) if value > 0 else 0.0


def max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    return float((1.0 - nav / nav.cummax()).max())


def concentration_metrics(trade_rows: list[dict], daily_frame: pd.DataFrame) -> dict:
    if not trade_rows:
        return {
            "profit_factor": None,
            "top_symbol_positive_pnl_share": None,
            "top3_positive_pnl_share": None,
            "top3_net_pnl_share": None,
            "active_months": 0,
            "losing_months": 0,
            "worst_month_pnl": 0.0,
            "best_month_pnl": 0.0,
        }
    trades = pd.DataFrame(trade_rows)
    gross_positive = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_negative = float(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    symbol_pnl = trades.groupby("instrument")["pnl"].sum().sort_values(ascending=False)
    symbol_positive = (
        trades.assign(pos_pnl=trades["pnl"].clip(lower=0.0))
        .groupby("instrument")["pos_pnl"]
        .sum()
        .sort_values(ascending=False)
    )
    month = (
        daily_frame.assign(
            month=pd.to_datetime(daily_frame["trade_date"]).dt.to_period("M").astype(str)
        )
        .groupby("month")
        .agg(day_pnl=("day_pnl", "sum"), trades=("day_trades", "sum"))
    )
    pnl_total = float(trades["pnl"].sum())
    top3_net = float(symbol_pnl.head(3).sum()) if len(symbol_pnl) else 0.0
    return {
        "profit_factor": float(gross_positive / abs(gross_negative))
        if gross_negative < 0
        else None,
        "top_symbol_positive_pnl_share": float(symbol_positive.iloc[0] / gross_positive)
        if gross_positive > 0 and len(symbol_positive)
        else None,
        "top3_positive_pnl_share": float(symbol_positive.head(3).sum() / gross_positive)
        if gross_positive > 0 and len(symbol_positive)
        else None,
        "top3_net_pnl_share": float(top3_net / pnl_total) if pnl_total != 0 else None,
        "active_months": int((month["trades"] > 0).sum()) if len(month) else 0,
        "losing_months": int((month["day_pnl"] < 0).sum()) if len(month) else 0,
        "worst_month_pnl": float(month["day_pnl"].min()) if len(month) else 0.0,
        "best_month_pnl": float(month["day_pnl"].max()) if len(month) else 0.0,
    }


def passes_optional_gate(value: float | None, limit: float, higher_is_better: bool) -> bool:
    if limit <= 0:
        return True
    if value is None or not np.isfinite(value):
        return False
    return value >= limit if higher_is_better else value <= limit


def run_replay(
    prediction_dir: Path,
    holdings_path: Path,
    daily_account_path: Path,
    minute_windows_path: Path,
    calendar_path: Path,
    output_json: Path,
    output_trades: Path,
    output_daily: Path,
    sell_price_col: str,
    buy_price_col: str,
    thresholds: list[float],
    trade_fractions: list[float],
    lot_size: int,
    buy_cost: float,
    sell_cost: float,
    slippage: float,
    min_cost: float,
    max_daily_turnover: float,
    max_symbol_round_trips: int,
    min_symbols_traded: int,
    min_active_months: int,
    min_profit_factor: float,
    max_top_symbol_positive_pnl_share: float,
    max_top3_positive_pnl_share: float,
    max_top3_net_pnl_share: float,
) -> dict:
    preds = load_predictions(prediction_dir, calendar_path)
    holdings = pd.read_csv(holdings_path, parse_dates=["trade_date"])
    holdings["signal_date"] = holdings["trade_date"].dt.normalize()
    holdings["instrument"] = holdings["instrument"].astype(str).str.upper()
    holdings = holdings[holdings["shares"] > 0][["signal_date", "instrument", "shares", "weight"]]

    daily = pd.read_csv(daily_account_path, parse_dates=["trade_date"])
    daily["trade_date"] = daily["trade_date"].dt.normalize()
    daily = daily.sort_values("trade_date")

    windows = pd.read_csv(minute_windows_path, parse_dates=["trade_date"])
    windows["trade_date"] = windows["trade_date"].dt.normalize()
    windows["instrument"] = windows["instrument"].astype(str).str.upper()
    missing_cols = [col for col in (sell_price_col, buy_price_col) if col not in windows.columns]
    if missing_cols:
        raise KeyError(f"minute windows missing price columns: {missing_cols}")
    windows = windows.rename(columns={sell_price_col: "sell_price_raw", buy_price_col: "buy_price_raw"})

    base = preds.merge(holdings, on=["signal_date", "instrument"], how="inner")
    base = base.merge(
        windows[["trade_date", "instrument", "sell_price_raw", "buy_price_raw"]],
        on=["trade_date", "instrument"],
        how="inner",
    )
    base = base.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["score", "shares", "sell_price_raw", "buy_price_raw"]
    )
    results = []
    all_trades = []
    all_daily = []
    for threshold in thresholds:
        for fraction in trade_fractions:
            cum_pnl = 0.0
            cum_turnover = 0.0
            trade_count = 0
            symbol_round_trips: dict[str, int] = {}
            daily_rows = []
            trade_rows = []
            by_trade_date = {date: part for date, part in base.groupby("trade_date", sort=True)}
            for _, day_row in daily.iterrows():
                trade_date = day_row["trade_date"]
                nav = float(day_row["nav"])
                cash = float(day_row["cash"])
                day_candidates = by_trade_date.get(trade_date)
                day_pnl = 0.0
                day_turnover_value = 0.0
                day_trades = 0
                if day_candidates is not None and nav > 0:
                    selected = day_candidates[day_candidates["score"] >= threshold].sort_values(
                        "score", ascending=False
                    )
                    for row in selected.itertuples(index=False):
                        if day_turnover_value / nav >= max_daily_turnover:
                            break
                        if max_symbol_round_trips > 0:
                            used = symbol_round_trips.get(row.instrument, 0)
                            if used >= max_symbol_round_trips:
                                continue
                        min_volume = min_lot(row.instrument, lot_size)
                        volume = round_lot(float(row.shares) * fraction, lot_size)
                        if volume < min_volume or volume > int(row.shares):
                            continue
                        sell_price = float(row.sell_price_raw) * (1.0 - slippage)
                        buy_price = float(row.buy_price_raw) * (1.0 + slippage)
                        if sell_price <= 0 or buy_price <= 0:
                            continue
                        sell_value = volume * sell_price
                        sell_fee = estimate_sell_fee(sell_value, sell_cost, min_cost)
                        buy_value = volume * buy_price
                        buy_fee = estimate_buy_fee(buy_value, buy_cost, min_cost)
                        available_cash = cash + sell_value - sell_fee + day_pnl
                        if available_cash < buy_value + buy_fee:
                            continue
                        round_trip_value = sell_value + buy_value
                        if (day_turnover_value + round_trip_value) / nav > max_daily_turnover:
                            continue
                        pnl = sell_value - sell_fee - buy_value - buy_fee
                        day_pnl += pnl
                        day_turnover_value += round_trip_value
                        day_trades += 2
                        symbol_round_trips[row.instrument] = (
                            symbol_round_trips.get(row.instrument, 0) + 1
                        )
                        trade_rows.append(
                            {
                                "threshold": threshold,
                                "trade_fraction": fraction,
                                "trade_date": trade_date.strftime("%Y-%m-%d"),
                                "signal_date": row.signal_date.strftime("%Y-%m-%d"),
                                "instrument": row.instrument,
                                "score": float(row.score),
                                "held_shares": int(row.shares),
                                "t0_volume": int(volume),
                                "sell_price": sell_price,
                                "buy_price": buy_price,
                                "sell_fee": sell_fee,
                                "buy_fee": buy_fee,
                                "pnl": pnl,
                                "turnover_value": round_trip_value,
                            }
                        )
                cum_pnl += day_pnl
                cum_turnover += day_turnover_value / nav if nav > 0 else 0.0
                trade_count += day_trades
                overlay_nav = nav + cum_pnl
                daily_rows.append(
                    {
                        "threshold": threshold,
                        "trade_fraction": fraction,
                        "trade_date": trade_date.strftime("%Y-%m-%d"),
                        "base_nav": nav,
                        "overlay_nav": overlay_nav,
                        "cum_pnl": cum_pnl,
                        "day_pnl": day_pnl,
                        "day_turnover": day_turnover_value / nav if nav > 0 else 0.0,
                        "day_trades": day_trades,
                    }
                )
            daily_frame = pd.DataFrame(daily_rows)
            ret = daily_frame["overlay_nav"].pct_change().dropna()
            base_return = float(daily_frame["base_nav"].iloc[-1] / daily_frame["base_nav"].iloc[0] - 1.0)
            overlay_return = float(
                daily_frame["overlay_nav"].iloc[-1] / daily_frame["overlay_nav"].iloc[0] - 1.0
            )
            trade_frame = pd.DataFrame(trade_rows)
            diagnostics = concentration_metrics(trade_rows, daily_frame)
            result = {
                "threshold": threshold,
                "trade_fraction": fraction,
                "base_return": base_return,
                "overlay_return": overlay_return,
                "incremental_return": overlay_return - base_return,
                "cum_pnl": float(cum_pnl),
                "trades": int(trade_count),
                "round_trips": int(len(trade_rows)),
                "cum_turnover": float(cum_turnover),
                "max_daily_turnover": float(daily_frame["day_turnover"].max()),
                "positive_trade_ratio": float((trade_frame["pnl"] > 0).mean())
                if trade_rows
                else None,
                "avg_trade_pnl": float(trade_frame["pnl"].mean()) if trade_rows else None,
                "symbols_traded": int(trade_frame["instrument"].nunique())
                if trade_rows
                else 0,
                "max_drawdown": max_drawdown(daily_frame["overlay_nav"]),
                "sharpe": float(ret.mean() / (ret.std() + 1e-12) * np.sqrt(252)) if len(ret) else 0.0,
                **diagnostics,
            }
            results.append(result)
            all_trades.extend(trade_rows)
            all_daily.extend(daily_rows)
    ranked = sorted(results, key=lambda row: row["incremental_return"], reverse=True)
    qualified = [
        row
        for row in ranked
        if row["incremental_return"] > 0 and row.get("round_trips", 0) >= 20
        and row.get("symbols_traded", 0) >= min_symbols_traded
        and row.get("active_months", 0) >= min_active_months
        and passes_optional_gate(row.get("profit_factor"), min_profit_factor, True)
        and passes_optional_gate(
            row.get("top_symbol_positive_pnl_share"),
            max_top_symbol_positive_pnl_share,
            False,
        )
        and passes_optional_gate(
            row.get("top3_positive_pnl_share"),
            max_top3_positive_pnl_share,
            False,
        )
        and passes_optional_gate(row.get("top3_net_pnl_share"), max_top3_net_pnl_share, False)
    ]
    report = {
        "status": "inner_t0_overlay_portfolio_replay",
        "prediction_dir": str(prediction_dir.resolve()),
        "holdings_path": str(holdings_path.resolve()),
        "daily_account_path": str(daily_account_path.resolve()),
        "minute_windows_path": str(minute_windows_path.resolve()),
        "execution_window": {
            "sell_price_col": sell_price_col,
            "buy_price_col": buy_price_col,
        },
        "base_rows": int(len(base)),
        "costs": {
            "buy_cost": buy_cost,
            "sell_cost": sell_cost,
            "slippage": slippage,
            "min_cost": min_cost,
        },
        "constraints": {
            "no_naked_t0": True,
            "sell_then_buy_same_day": True,
            "single_position_turnover_fraction_grid": trade_fractions,
            "max_daily_turnover": max_daily_turnover,
            "max_symbol_round_trips": max_symbol_round_trips,
            "min_symbols_traded": min_symbols_traded,
            "min_active_months": min_active_months,
            "min_profit_factor": min_profit_factor,
            "max_top_symbol_positive_pnl_share": max_top_symbol_positive_pnl_share,
            "max_top3_positive_pnl_share": max_top3_positive_pnl_share,
            "max_top3_net_pnl_share": max_top3_net_pnl_share,
            "lot_size": lot_size,
        },
        "best": ranked[0] if ranked else None,
        "best_qualified": qualified[0] if qualified else None,
        "qualified_count": len(qualified),
        "results": ranked,
        "decision": (
            "candidate_for_deeper_inner_replay"
            if qualified
            else "keep_inner_research_only"
        ),
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(all_trades).to_csv(output_trades, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_daily).to_csv(output_daily, index=False, encoding="utf-8-sig")
    return report


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--holdings", required=True)
    parser.add_argument("--daily-account", required=True)
    parser.add_argument("--minute-windows", default=str(DATA / "_canonical_topk_minute_windows_2025_20260605.csv"))
    parser.add_argument("--calendar", default=str(DATA / "cn_data_pool_inner_canonical_20260605" / "calendars" / "day.txt"))
    parser.add_argument("--thresholds", default="0.65,0.70,0.715887,0.75")
    parser.add_argument("--trade-fractions", default="0.10,0.20,0.30")
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--buy-cost", type=float, default=0.001)
    parser.add_argument("--sell-cost", type=float, default=0.0025)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--min-cost", type=float, default=5.0)
    parser.add_argument("--max-daily-turnover", type=float, default=0.20)
    parser.add_argument(
        "--max-symbol-round-trips",
        type=int,
        default=0,
        help="Optional cap on total round trips per symbol for each grid setting; 0 disables.",
    )
    parser.add_argument(
        "--min-symbols-traded",
        type=int,
        default=0,
        help="Minimum unique traded symbols required for a qualified candidate; 0 disables.",
    )
    parser.add_argument(
        "--min-active-months",
        type=int,
        default=0,
        help="Minimum months with at least one round trip required for qualification; 0 disables.",
    )
    parser.add_argument(
        "--min-profit-factor",
        type=float,
        default=0.0,
        help="Minimum gross profit / gross loss ratio required for qualification; 0 disables.",
    )
    parser.add_argument(
        "--max-top-symbol-positive-pnl-share",
        type=float,
        default=0.0,
        help="Maximum single-symbol share of gross positive PnL; 0 disables.",
    )
    parser.add_argument(
        "--max-top3-positive-pnl-share",
        type=float,
        default=0.0,
        help="Maximum top-3-symbol share of gross positive PnL; 0 disables.",
    )
    parser.add_argument(
        "--max-top3-net-pnl-share",
        type=float,
        default=0.0,
        help="Maximum top-3-symbol share of net PnL; 0 disables.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-trades", required=True)
    parser.add_argument("--output-daily", required=True)
    parser.add_argument("--sell-price-col", default="open_exec")
    parser.add_argument("--buy-price-col", default="close_exec")
    args = parser.parse_args()
    report = run_replay(
        Path(args.prediction_dir).resolve(),
        Path(args.holdings).resolve(),
        Path(args.daily_account).resolve(),
        Path(args.minute_windows).resolve(),
        Path(args.calendar).resolve(),
        Path(args.output_json).resolve(),
        Path(args.output_trades).resolve(),
        Path(args.output_daily).resolve(),
        args.sell_price_col,
        args.buy_price_col,
        parse_float_list(args.thresholds),
        parse_float_list(args.trade_fractions),
        args.lot_size,
        args.buy_cost,
        args.sell_cost,
        args.slippage,
        args.min_cost,
        args.max_daily_turnover,
        args.max_symbol_round_trips,
        args.min_symbols_traded,
        args.min_active_months,
        args.min_profit_factor,
        args.max_top_symbol_positive_pnl_share,
        args.max_top3_positive_pnl_share,
        args.max_top3_net_pnl_share,
    )
    best = report.get("best") or {}
    print(
        "[inner t0 replay] "
        f"decision={report['decision']} base_rows={report['base_rows']} "
        f"best_threshold={best.get('threshold')} fraction={best.get('trade_fraction')} "
        f"incremental={best.get('incremental_return')} trades={best.get('trades')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
