from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from concentration_constraints import (
    ConcentrationRules,
    concentration_snapshot,
    groups_on_date,
    load_group_metadata,
    select_with_group_cap,
)
from evaluate_daily_topk_grid import load_middle_predictions
from minute_mapped_topk_replay import MappedProfile, MappedReplayConfig, min_buy_lot, order_delta_key, round_lot
from portfolio_experiments import BASE_COST, STRESS_COST
from universe import UniverseRules, load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_RAW = DATA / "A_Stock_daily_qfq" / "daily_qfq_6.5"
DEFAULT_MIDDLE = HERE / "outputs" / "oof" / "pit_holdout_20260605_de2_srfs_es" / "middle" / "fold_99.csv"
DEFAULT_GROUP_METADATA = DATA / "industry_theme_pit.csv"


def next_date_map(dates: pd.Series) -> dict[pd.Timestamp, pd.Timestamp]:
    unique = pd.DatetimeIndex(pd.to_datetime(dates)).drop_duplicates().sort_values()
    return dict(zip(unique[:-1], unique[1:]))


def prepare_frame(predictions: pd.DataFrame, prices: pd.DataFrame, profile: MappedProfile) -> pd.DataFrame:
    rules = UniverseRules(min_listing_days=250, min_avg_amount=profile.min_avg_amount)
    price_frame = prices.copy().sort_values(["instrument", "datetime"])
    grouped = price_frame.groupby("instrument", sort=False)
    code = price_frame["instrument"].str[-6:]
    price_frame["board_ok"] = code.str.startswith(rules.board_prefixes)
    price_frame["listing_days"] = grouped.cumcount() + 1
    price_frame["avg_amount"] = grouped["amount"].transform(
        lambda values: values.rolling(rules.liquidity_window, min_periods=rules.liquidity_window).mean()
    )
    price_frame["suspend_ratio"] = grouped["volume"].transform(
        lambda values: values.eq(0).rolling(rules.suspend_window, min_periods=rules.suspend_window).mean()
    )
    price_frame["eligible"] = (
        price_frame["board_ok"]
        & (price_frame["listing_days"] >= rules.min_listing_days)
        & (price_frame["avg_amount"] >= rules.min_avg_amount)
        & (price_frame["suspend_ratio"] <= rules.max_suspend_ratio)
        & np.isfinite(price_frame["open"])
        & np.isfinite(price_frame["close"])
        & (price_frame["open"] > 0)
        & (price_frame["close"] > 0)
    )
    calendar = next_date_map(price_frame["datetime"])
    signal = predictions.copy()
    signal["trade_date"] = signal["datetime"].map(calendar)
    signal = signal.dropna(subset=["trade_date", "middle"]).rename(columns={"datetime": "signal_date"})
    known = price_frame[["datetime", "instrument", "eligible", "avg_amount", "listing_days"]].rename(
        columns={"datetime": "signal_date"}
    )
    trade_prices = price_frame[["datetime", "instrument", "open", "close"]].rename(columns={"datetime": "trade_date"})
    return signal.merge(known, on=["signal_date", "instrument"], how="left").merge(
        trade_prices,
        on=["trade_date", "instrument"],
        how="inner",
    )


def metrics(nav: pd.Series, trades: int, turnover: float) -> dict:
    returns = nav.pct_change().dropna()
    sharpe = 0.0 if returns.std() <= 1e-12 else returns.mean() / returns.std() * np.sqrt(252)
    ann = 0.0 if len(nav) < 2 else (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1
    drawdown = (nav.cummax() - nav) / nav.cummax()
    return {
        "days": int(len(nav)),
        "final_nav": float(nav.iloc[-1]) if len(nav) else 0.0,
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1) if len(nav) else 0.0,
        "annualized_return": float(ann),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "trades": int(trades),
        "turnover": float(turnover),
    }


def replay_daily(
    frame: pd.DataFrame,
    group_metadata: pd.DataFrame,
    profile: MappedProfile,
    cfg: MappedReplayConfig,
    cost,
) -> dict:
    by_date = {date: part.copy() for date, part in frame.groupby("trade_date", sort=True)}
    cash = float(cfg.initial_cash)
    held: dict[str, int] = {}
    previous_full: set[str] = set()
    last_close: dict[str, float] = {}
    nav_rows: list[tuple[pd.Timestamp, float]] = []
    target_rows: list[dict] = []
    concentration_rows: list[dict] = []
    trades = 0
    turnover = 0.0
    total_fees = 0.0
    buffer_k = profile.top_k * cfg.buffer_multiple

    for day_no, (trade_date, day) in enumerate(by_date.items()):
        open_prices = day.set_index("instrument")["open"].to_dict()
        close_prices = day.set_index("instrument")["close"].to_dict()
        open_nav = cash + sum(
            volume * open_prices.get(inst, close_prices.get(inst, last_close.get(inst, 0.0)))
            for inst, volume in held.items()
        )
        if day_no % profile.rebalance_every == 0:
            eligible = day[day["eligible"].fillna(False)].sort_values("middle", ascending=False)
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
            target_weight = profile.risk_budget / profile.top_k if selected_full else 0.0
            desired: dict[str, int] = {}
            for inst in selected_full:
                price = float(open_prices.get(inst, 0.0) or 0.0)
                if price <= 0:
                    continue
                shares = round_lot(open_nav * target_weight / price, cfg.lot_size, min_buy_lot(inst, cfg.lot_size))
                if shares > 0:
                    desired[inst] = shares
            deltas = {inst: desired.get(inst, 0) - held.get(inst, 0) for inst in set(held) | set(desired)}
            for inst, delta in sorted(deltas.items(), key=order_delta_key):
                price = float(open_prices.get(inst, 0.0) or 0.0)
                if price <= 0 or delta == 0:
                    continue
                if delta > 0 and delta < min_buy_lot(inst, cfg.lot_size):
                    continue
                nav_ref = cash + sum(
                    volume * close_prices.get(code, open_prices.get(code, last_close.get(code, 0.0)))
                    for code, volume in held.items()
                )
                if delta < 0:
                    volume = min(-delta, held.get(inst, 0))
                    fill_price = price * (1.0 - cost.slippage)
                    value = volume * fill_price
                    fee = max(value * cost.sell_cost, cost.min_cost) if volume > 0 else 0.0
                    cash += value - fee
                    held[inst] = held.get(inst, 0) - volume
                else:
                    fill_price = price * (1.0 + cost.slippage)
                    affordable = round_lot(
                        max(0.0, cash - cost.min_cost) / (fill_price * (1.0 + cost.buy_cost)),
                        cfg.lot_size,
                        min_buy_lot(inst, cfg.lot_size),
                    )
                    volume = min(delta, affordable)
                    value = volume * fill_price
                    fee = max(value * cost.buy_cost, cost.min_cost) if volume > 0 else 0.0
                    cash -= value + fee
                    held[inst] = held.get(inst, 0) + volume
                if volume > 0:
                    trades += 1
                    turnover += value / max(nav_ref, 1e-12)
                    total_fees += fee
            held = {inst: volume for inst, volume in held.items() if volume > 0}
            target_rows.append(
                {
                    "trade_date": str(pd.Timestamp(trade_date).date()),
                    "selected_count": len(selected_full),
                    "executable_count": len(desired),
                    "target_weight_sum": float(len(desired) * target_weight),
                    "holding_count": len(held),
                }
            )
            concentration_rows.append(
                {
                    "trade_date": str(pd.Timestamp(trade_date).date()),
                    "selected": concentration_snapshot(selected_full, groups),
                    "executable": concentration_snapshot(list(desired), groups),
                }
            )
        close_nav = cash + sum(
            volume * close_prices.get(inst, open_prices.get(inst, last_close.get(inst, 0.0)))
            for inst, volume in held.items()
        )
        last_close.update({inst: float(price) for inst, price in close_prices.items() if np.isfinite(price) and price > 0})
        nav_rows.append((trade_date, close_nav))

    nav = pd.Series(dict(nav_rows)).sort_index()
    return {
        "profile": asdict(profile),
        "config": asdict(cfg),
        "cost": asdict(cost),
        "metrics": {**metrics(nav, trades, turnover), "total_fees": float(total_fees)},
        "targets": target_rows,
        "concentration": concentration_rows,
        "nav": {str(k.date()): float(v) for k, v in nav.items()},
    }


def selected_profiles() -> list[MappedProfile]:
    return [
        MappedProfile("c_top150_rb45_risk0.80_cap0.30", 150, 45, 0.80, 100_000_000.0, 0.30),
        MappedProfile("c_top150_rb45_risk0.90_cap0.30", 150, 45, 0.90, 100_000_000.0, 0.30),
        MappedProfile("c_top150_rb45_risk1.00_cap0.30", 150, 45, 1.00, 100_000_000.0, 0.30),
        MappedProfile("c_top120_rb45_risk0.80_cap0.30", 120, 45, 0.80, 100_000_000.0, 0.30),
    ]


def parse_profile_specs(text: str) -> list[MappedProfile]:
    if not text.strip():
        return selected_profiles()
    profiles: list[MappedProfile] = []
    for raw in text.split(";"):
        item = raw.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 6:
            raise ValueError(
                "--profile-spec entries must be name,top_k,rebalance,risk_budget,min_avg_amount,industry_cap"
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


def write_summary(results: list[dict], path: Path) -> None:
    fields = [
        "profile",
        "cost",
        "total_return",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "turnover",
        "trades",
        "total_fees",
        "first_executable_count",
        "first_target_weight_sum",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            metrics_row = row["metrics"]
            first_target = row["targets"][0] if row["targets"] else {}
            writer.writerow(
                {
                    "profile": row["profile"]["name"],
                    "cost": row["cost"]["name"],
                    "total_return": metrics_row["total_return"],
                    "annualized_return": metrics_row["annualized_return"],
                    "sharpe": metrics_row["sharpe"],
                    "max_drawdown": metrics_row["max_drawdown"],
                    "turnover": metrics_row["turnover"],
                    "trades": metrics_row["trades"],
                    "total_fees": metrics_row["total_fees"],
                    "first_executable_count": first_target.get("executable_count"),
                    "first_target_weight_sum": first_target.get("target_weight_sum"),
                }
            )


def run(
    middle_path: Path,
    raw_dir: Path,
    group_metadata_path: Path,
    output_path: Path,
    summary_path: Path,
    start_signal: str,
    end_signal: str,
    price_start: str,
    price_end: str,
    initial_cash: float,
    profiles: list[MappedProfile] | None = None,
) -> dict:
    predictions = load_middle_predictions(middle_path)
    predictions = predictions[predictions["datetime"].between(start_signal, end_signal)].copy()
    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(raw_dir, instruments, start=price_start, end=price_end)
    group_metadata = load_group_metadata(group_metadata_path, "industry")
    cfg = MappedReplayConfig(initial_cash=initial_cash, buffer_multiple=2)
    results = []
    for profile in profiles or selected_profiles():
        frame = prepare_frame(predictions, prices, profile)
        for cost in (BASE_COST, STRESS_COST):
            result = replay_daily(frame, group_metadata, profile, cfg, cost)
            results.append(result)
            m = result["metrics"]
            first = result["targets"][0] if result["targets"] else {}
            print(
                f"[daily holdout C] {profile.name}/{cost.name} "
                f"ret={m['total_return']:+.2%} ann={m['annualized_return']:+.2%} "
                f"sharpe={m['sharpe']:+.2f} mdd={m['max_drawdown']:.2%} "
                f"turn={m['turnover']:.2f} exec={first.get('executable_count')}",
                flush=True,
            )
    report = {
        "status": "extended_daily_holdout_c_baseline_research_only",
        "middle_prediction": str(middle_path),
        "raw_dir": str(raw_dir),
        "group_metadata": str(group_metadata_path),
        "window": {
            "signal_start": start_signal,
            "signal_end": end_signal,
            "price_start": price_start,
            "price_end": price_end,
        },
        "execution_note": (
            "Daily-only approximation: next-day open is used for execution and daily close for marking. "
            "Full minute VWAP replay is still required after minute data is available."
        ),
        "results": results,
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(results, summary_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--middle", default=str(DEFAULT_MIDDLE))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--group-metadata", default=str(DEFAULT_GROUP_METADATA))
    parser.add_argument("--output", default=str(HERE / "outputs" / "extended_daily_holdout_c_baseline_20260605.json"))
    parser.add_argument("--summary", default=str(HERE / "outputs" / "extended_daily_holdout_c_baseline_20260605_summary.csv"))
    parser.add_argument("--start-signal", default="2026-04-21")
    parser.add_argument("--end-signal", default="2026-06-04")
    parser.add_argument("--price-start", default="2022-01-04")
    parser.add_argument("--price-end", default="2026-06-05")
    parser.add_argument("--initial-cash", type=float, default=500_000.0)
    parser.add_argument(
        "--profile-spec",
        default="",
        help="Semicolon-separated entries: name,top_k,rebalance,risk_budget,min_avg_amount,industry_cap",
    )
    args = parser.parse_args()
    run(
        Path(args.middle).resolve(),
        Path(args.raw_dir).resolve(),
        Path(args.group_metadata).resolve(),
        Path(args.output).resolve(),
        Path(args.summary).resolve(),
        args.start_signal,
        args.end_signal,
        args.price_start,
        args.price_end,
        args.initial_cash,
        parse_profile_specs(args.profile_spec),
    )


if __name__ == "__main__":
    main()
