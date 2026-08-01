from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
from minute_cd_replay import load_minute_windows, read_pool
from portfolio_experiments import BASE_COST, STRESS_COST, CostScenario, ExperimentConfig
from universe import UniverseRules, load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_STAGE_CSV = DATA / "_research_1min_pool_csv_2026"
DEFAULT_MIDDLE = HERE / "outputs" / "oof_combined" / "middle_oof_pit_de2_srfs_es.csv"
DEFAULT_GROUP_METADATA = DATA / "industry_theme_pit.csv"


@dataclass(frozen=True)
class MappedProfile:
    name: str
    top_k: int
    rebalance_every: int
    risk_budget: float
    min_avg_amount: float
    industry_cap: float


@dataclass(frozen=True)
class MappedReplayConfig:
    initial_cash: float = 500_000.0
    lot_size: int = 100
    min_listing_days: int = 250
    liquidity_window: int = 20
    buffer_multiple: int = 2


def next_date_map(dates: pd.Series) -> dict[pd.Timestamp, pd.Timestamp]:
    unique = pd.DatetimeIndex(pd.to_datetime(dates)).drop_duplicates().sort_values()
    return dict(zip(unique[:-1], unique[1:]))


def prepare_daily_frame(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    profile: MappedProfile,
    cfg: MappedReplayConfig,
) -> pd.DataFrame:
    rules = UniverseRules(
        min_listing_days=cfg.min_listing_days,
        liquidity_window=cfg.liquidity_window,
        min_avg_amount=profile.min_avg_amount,
    )
    price_frame = prices.copy().sort_values(["instrument", "datetime"])
    eligible = price_frame.copy()
    grouped = eligible.groupby("instrument", sort=False)
    code = eligible["instrument"].str[-6:]
    eligible["board_ok"] = code.str.startswith(rules.board_prefixes)
    eligible["listing_days"] = grouped.cumcount() + 1
    eligible["avg_amount"] = grouped["amount"].transform(
        lambda values: values.rolling(rules.liquidity_window, min_periods=rules.liquidity_window).mean()
    )
    eligible["suspend_ratio"] = grouped["volume"].transform(
        lambda values: values.eq(0).rolling(rules.suspend_window, min_periods=rules.suspend_window).mean()
    )
    eligible["eligible"] = (
        eligible["board_ok"]
        & (eligible["listing_days"] >= rules.min_listing_days)
        & (eligible["avg_amount"] >= rules.min_avg_amount)
        & (eligible["suspend_ratio"] <= rules.max_suspend_ratio)
        & np.isfinite(eligible["open"])
        & np.isfinite(eligible["close"])
        & (eligible["open"] > 0)
        & (eligible["close"] > 0)
    )
    calendar = next_date_map(price_frame["datetime"])
    signal = predictions.copy()
    signal["trade_date"] = signal["datetime"].map(calendar)
    signal = signal.dropna(subset=["trade_date", "middle"]).rename(columns={"datetime": "signal_date"})
    known = eligible[
        ["datetime", "instrument", "eligible", "avg_amount", "listing_days"]
    ].rename(columns={"datetime": "signal_date"})
    return signal.merge(known, on=["signal_date", "instrument"], how="left")


def min_buy_lot(instrument: str, lot_size: int) -> int:
    code = str(instrument).upper()[-6:]
    if code.startswith(("688", "689")):
        return max(lot_size, 200)
    return lot_size


def round_lot(value: float, lot_size: int, min_lot: int | None = None) -> int:
    if not np.isfinite(value):
        return 0
    rounded = max(0, int(value // lot_size) * lot_size)
    if min_lot is not None and rounded < min_lot:
        return 0
    return rounded


def safe_price(*values: float) -> float:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0:
            return number
    return 0.0


def order_delta_key(item: tuple[str, int]) -> tuple[int, int, str]:
    inst, delta = item
    side_rank = 0 if delta < 0 else 1
    return side_rank, delta, inst


def metrics(nav: pd.Series, trades: int, turnover: float, coverage_rows: list[dict]) -> dict:
    returns = nav.pct_change().dropna()
    sharpe = 0.0 if returns.std() <= 1e-12 else returns.mean() / returns.std() * np.sqrt(252)
    peak = nav.cummax()
    drawdown = (peak - nav) / peak
    mapped_counts = np.array([row["mapped_count"] for row in coverage_rows], dtype=float)
    mapped_weight = np.array([row["mapped_weight_fraction"] for row in coverage_rows], dtype=float)
    return {
        "days": int(len(nav)),
        "final_nav": float(nav.iloc[-1]) if len(nav) else 0.0,
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1) if len(nav) else 0.0,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "trades": int(trades),
        "turnover": float(turnover),
        "rebalance_count": int(len(coverage_rows)),
        "avg_mapped_count": float(mapped_counts.mean()) if len(mapped_counts) else 0.0,
        "min_mapped_count": int(mapped_counts.min()) if len(mapped_counts) else 0,
        "avg_mapped_weight_fraction": float(mapped_weight.mean()) if len(mapped_weight) else 0.0,
        "min_mapped_weight_fraction": float(mapped_weight.min()) if len(mapped_weight) else 0.0,
    }


def replay_mapped(
    frame: pd.DataFrame,
    minute_windows: pd.DataFrame,
    minute_pool: set[str],
    group_metadata: pd.DataFrame,
    profile: MappedProfile,
    cfg: MappedReplayConfig,
    cost: CostScenario,
    mapping_policy: str,
    allocation_mode: str = "fixed_topk",
    outer_risk_threshold: float | None = None,
    outer_risk_floor: float | None = None,
) -> dict:
    if mapping_policy not in {"preserve_target_weights", "renormalize_mapped"}:
        raise ValueError("unsupported mapping_policy")
    if allocation_mode not in ALLOCATION_MODES:
        raise ValueError(f"unsupported allocation mode: {allocation_mode}")

    by_date = {date: part.copy() for date, part in frame.groupby("trade_date", sort=True)}
    minute_by_date = {date: part.copy() for date, part in minute_windows.groupby("trade_date", sort=True)}
    cash = float(cfg.initial_cash)
    held: dict[str, int] = {}
    previous_full: set[str] = set()
    last_mark: dict[str, float] = {}
    nav_rows: list[tuple[pd.Timestamp, float]] = []
    coverage_rows: list[dict] = []
    concentration_rows: list[dict] = []
    trades = 0
    turnover = 0.0
    buffer_k = profile.top_k * cfg.buffer_multiple

    for day_no, (trade_date, day) in enumerate(by_date.items()):
        minute_day = minute_by_date.get(trade_date)
        if minute_day is None:
            price_open: dict[str, float] = {}
            mark_close: dict[str, float] = {}
        else:
            price_open = minute_day.set_index("instrument")["open_exec"].to_dict()
            mark_close = minute_day.set_index("instrument")["mark_close"].to_dict()
        open_nav = cash + sum(
            held.get(inst, 0)
            * safe_price(
                price_open.get(inst, np.nan),
                mark_close.get(inst, np.nan),
                last_mark.get(inst, np.nan),
            )
            for inst in held
        )

        if day_no % profile.rebalance_every == 0:
            outer_risk_probability = (
                float(pd.to_numeric(day["outer_risk_probability"], errors="coerce").median())
                if "outer_risk_probability" in day
                else np.nan
            )
            effective_risk_budget = float(profile.risk_budget)
            if (
                outer_risk_threshold is not None
                and outer_risk_floor is not None
                and np.isfinite(outer_risk_probability)
                and outer_risk_probability >= outer_risk_threshold
            ):
                effective_risk_budget = min(effective_risk_budget, float(outer_risk_floor))
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
            allocation_diagnostics: dict = {}
            if allocation_mode == "capital_aware":
                allocation_candidates = [inst for inst in mapped if held.get(inst, 0) > 0]
                allocation_candidates.extend(
                    inst for inst in mapped if held.get(inst, 0) <= 0
                )
                allocation = allocate_equal_weight_lots(
                    allocation_candidates,
                    {inst: safe_price(price_open.get(inst, np.nan)) for inst in mapped},
                    {inst: min_buy_lot(inst, cfg.lot_size) for inst in mapped},
                    open_nav,
                    effective_risk_budget,
                    lot_size=cfg.lot_size,
                    denominator_count=profile.top_k,
                    mode=allocation_mode,
                )
                desired = {inst: int(volume) for inst, volume in allocation["shares"].items()}
                allocation_diagnostics = allocation["diagnostics"]
                allocation_diagnostics["retained_candidate_count"] = sum(
                    held.get(inst, 0) > 0 for inst in allocation_candidates
                )
                mapped_weight_fraction = float(
                    allocation_diagnostics.get("budget_utilization", 0.0)
                )
            else:
                if mapping_policy == "preserve_target_weights":
                    weight_per_name = effective_risk_budget / profile.top_k if selected_full else 0.0
                    mapped_weight_fraction = len(mapped) / max(profile.top_k, 1)
                else:
                    weight_per_name = effective_risk_budget / len(mapped) if mapped else 0.0
                    mapped_weight_fraction = 1.0 if mapped else 0.0

                desired = {}
                for inst in mapped:
                    price = safe_price(price_open.get(inst, np.nan))
                    if price > 0:
                        desired[inst] = round_lot(
                            open_nav * weight_per_name / price,
                            cfg.lot_size,
                            min_buy_lot(inst, cfg.lot_size),
                        )
                allocated_notional = sum(
                    desired.get(inst, 0) * safe_price(price_open.get(inst, np.nan))
                    for inst in desired
                )
                allocation_diagnostics = {
                    "mode": allocation_mode,
                    "budget_value": open_nav * effective_risk_budget,
                    "allocated_notional": allocated_notional,
                    "effective_weight": allocated_notional / max(open_nav, 1e-12),
                    "budget_utilization": allocated_notional
                    / max(open_nav * effective_risk_budget, 1e-12),
                    "allocated_count": sum(volume > 0 for volume in desired.values()),
                }

            deltas = {inst: desired.get(inst, 0) - held.get(inst, 0) for inst in set(held) | set(desired)}
            for inst, delta in sorted(deltas.items(), key=order_delta_key):
                price = safe_price(price_open.get(inst, np.nan))
                if price <= 0 or delta == 0:
                    continue
                if delta > 0 and delta < min_buy_lot(inst, cfg.lot_size):
                    continue
                nav_ref = cash + sum(
                    held.get(code, 0)
                    * safe_price(
                        mark_close.get(code, np.nan),
                        price_open.get(code, np.nan),
                        last_mark.get(code, np.nan),
                    )
                    for code in held
                )
                if delta < 0:
                    volume = min(-delta, held.get(inst, 0))
                    fill = price * (1.0 - cost.slippage)
                    value = volume * fill
                    fee = max(value * cost.sell_cost, cost.min_cost) if volume > 0 else 0.0
                    cash += value - fee
                    held[inst] = held.get(inst, 0) - volume
                else:
                    fill = price * (1.0 + cost.slippage)
                    affordable = round_lot(
                        max(0.0, cash - cost.min_cost) / (fill * (1.0 + cost.buy_cost)),
                        cfg.lot_size,
                        min_buy_lot(inst, cfg.lot_size),
                    )
                    volume = min(delta, affordable)
                    value = volume * fill
                    fee = max(value * cost.buy_cost, cost.min_cost) if volume > 0 else 0.0
                    cash -= value + fee
                    held[inst] = held.get(inst, 0) + volume
                if volume > 0:
                    trades += 1
                    turnover += value / max(nav_ref, 1e-12)
            held = {inst: volume for inst, volume in held.items() if volume > 0}
            coverage_rows.append(
                {
                    "trade_date": str(pd.Timestamp(trade_date).date()),
                    "target_count": len(selected_full),
                    "mapped_count": len(mapped),
                    "mapped_weight_fraction": float(mapped_weight_fraction),
                    "mapped": mapped,
                    "allocation_mode": allocation_mode,
                    "base_risk_budget": profile.risk_budget,
                    "effective_risk_budget": effective_risk_budget,
                    "outer_risk_probability": outer_risk_probability,
                    "outer_overlay_triggered": bool(
                        outer_risk_threshold is not None
                        and np.isfinite(outer_risk_probability)
                        and outer_risk_probability >= outer_risk_threshold
                    ),
                    "allocation": allocation_diagnostics,
                }
            )
            concentration_rows.append(
                {
                    "trade_date": str(pd.Timestamp(trade_date).date()),
                    "full_target": concentration_snapshot(selected_full, groups),
                    "mapped_target": concentration_snapshot(mapped, groups),
                }
            )

        close_nav = cash + sum(
            held.get(inst, 0)
            * safe_price(
                mark_close.get(inst, np.nan),
                price_open.get(inst, np.nan),
                last_mark.get(inst, np.nan),
            )
            for inst in held
        )
        last_mark.update(
            {inst: float(price) for inst, price in mark_close.items() if np.isfinite(price) and price > 0}
        )
        nav_rows.append((trade_date, close_nav))

    nav = pd.Series(dict(nav_rows)).sort_index()
    return {
        "profile": asdict(profile),
        "cost": asdict(cost),
        "mapping_policy": mapping_policy,
        "allocation_mode": allocation_mode,
        "outer_risk_threshold": outer_risk_threshold,
        "outer_risk_floor": outer_risk_floor,
        "metrics": metrics(nav, trades, turnover, coverage_rows),
        "coverage": coverage_rows,
        "concentration": concentration_rows,
        "nav": {str(k.date()): float(v) for k, v in nav.items()},
    }


def fold_summary(results: list[dict], initial_cash: float) -> dict:
    returns = np.array([row["metrics"]["total_return"] for row in results], dtype=float)
    drawdowns = np.array([row["metrics"]["max_drawdown"] for row in results], dtype=float)
    turnovers = np.array([row["metrics"]["turnover"] for row in results], dtype=float)
    mapped = np.array([row["metrics"]["avg_mapped_count"] for row in results], dtype=float)
    return {
        "evaluated_folds": len(results),
        "median_fold_return": float(np.median(returns)) if len(returns) else None,
        "worst_fold_return": float(np.min(returns)) if len(returns) else None,
        "positive_fold_ratio": float((returns > 0).mean()) if len(returns) else None,
        "median_fold_max_drawdown": float(np.median(drawdowns)) if len(drawdowns) else None,
        "median_fold_turnover": float(np.median(turnovers)) if len(turnovers) else None,
        "median_avg_mapped_count": float(np.median(mapped)) if len(mapped) else None,
        "initial_cash": float(initial_cash),
    }


def default_profiles() -> list[MappedProfile]:
    return [
        MappedProfile("growth_top100", 100, 30, 1.0, 100_000_000.0, 0.40),
        MappedProfile("balanced_top150", 150, 30, 0.60, 100_000_000.0, 0.30),
        MappedProfile("defensive_top150", 150, 30, 0.40, 100_000_000.0, 0.30),
    ]


def run(
    stage_dir: Path,
    minute_windows_path: Path | None,
    middle_path: Path,
    folds_path: Path,
    group_metadata_path: Path,
    output_path: Path,
    start_signal: str,
    end_signal: str,
    cfg: MappedReplayConfig,
) -> dict:
    predictions = load_middle_predictions(middle_path)
    predictions = predictions[predictions["datetime"].between(start_signal, end_signal)].copy()
    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(DATA / "A_Stock_daily_qfq", instruments, start="2022-01-04", end="2026-04-28")
    group_metadata = load_group_metadata(group_metadata_path, "industry")
    frames = {
        profile.name: prepare_daily_frame(predictions, prices, profile, cfg)
        for profile in default_profiles()
    }
    all_trade_dates = pd.concat([frame["trade_date"] for frame in frames.values()])
    if minute_windows_path is not None:
        minute_windows = pd.read_csv(minute_windows_path, parse_dates=["trade_date"])
        minute_windows["instrument"] = minute_windows["instrument"].astype(str).str.upper()
        minute_windows = minute_windows[
            minute_windows["trade_date"].between(all_trade_dates.min(), all_trade_dates.max())
        ].copy()
        minute_pool = set(minute_windows["instrument"].drop_duplicates())
    else:
        minute_pool = set(read_pool(stage_dir))
        minute_windows = load_minute_windows(stage_dir, sorted(minute_pool), all_trade_dates.min(), all_trade_dates.max())
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    rows = []
    for profile in default_profiles():
        frame = frames[profile.name]
        for cost in (BASE_COST, STRESS_COST):
            for policy in ("preserve_target_weights", "renormalize_mapped"):
                full = replay_mapped(frame, minute_windows, minute_pool, group_metadata, profile, cfg, cost, policy)
                fold_rows = []
                for fold in folds:
                    lo = max(pd.Timestamp(fold["test_start"]), pd.Timestamp(start_signal))
                    hi = min(pd.Timestamp(fold["test_end"]), pd.Timestamp(end_signal))
                    if lo > hi:
                        continue
                    part = frame[frame["signal_date"].between(lo, hi)].copy()
                    if part.empty:
                        continue
                    fold_result = replay_mapped(
                        part,
                        minute_windows,
                        minute_pool,
                        group_metadata,
                        profile,
                        cfg,
                        cost,
                        policy,
                    )
                    fold_rows.append(
                        {
                            "fold": fold["fold"],
                            "signal_start": str(lo.date()),
                            "signal_end": str(hi.date()),
                            "metrics": fold_result["metrics"],
                        }
                    )
                full["folds"] = fold_rows
                full["fold_summary"] = fold_summary(fold_rows, cfg.initial_cash)
                rows.append(full)
                m = full["metrics"]
                fs = full["fold_summary"]
                print(
                    f"[mapped] {profile.name}/{cost.name}/{policy} "
                    f"return={m['total_return']:+.2%} mdd={m['max_drawdown']:.2%} "
                    f"turn={m['turnover']:.2f} avg_mapped={m['avg_mapped_count']:.1f} "
                    f"fold_worst={fs['worst_fold_return']:+.2%}",
                    flush=True,
                )
    report = {
        "status": "daily_topk_to_minute_pool_mapped_replay_research_only",
        "stage_dir": str(stage_dir),
        "minute_windows": str(minute_windows_path) if minute_windows_path is not None else None,
        "middle_prediction": str(middle_path),
        "folds_path": str(folds_path),
        "group_metadata": str(group_metadata_path),
        "window": {"signal_start": start_signal, "signal_end": end_signal},
        "config": asdict(cfg),
        "minute_pool_size": len(minute_pool),
        "profiles": [asdict(profile) for profile in default_profiles()],
        "results": rows,
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE_CSV))
    parser.add_argument("--minute-windows", default=None)
    parser.add_argument("--middle", default=str(DEFAULT_MIDDLE))
    parser.add_argument("--folds", default=str(HERE / "outputs" / "purged_folds.json"))
    parser.add_argument("--group-metadata", default=str(DEFAULT_GROUP_METADATA))
    parser.add_argument("--output", default=str(HERE / "outputs" / "minute_mapped_topk_replay.json"))
    parser.add_argument("--start-signal", default="2025-01-03")
    parser.add_argument("--end-signal", default="2026-04-02")
    parser.add_argument("--initial-cash", type=float, default=500_000.0)
    args = parser.parse_args()
    run(
        Path(args.stage_dir).resolve(),
        Path(args.minute_windows).resolve() if args.minute_windows else None,
        Path(args.middle).resolve(),
        Path(args.folds).resolve(),
        Path(args.group_metadata).resolve(),
        Path(args.output).resolve(),
        args.start_signal,
        args.end_signal,
        MappedReplayConfig(initial_cash=args.initial_cash),
    )


if __name__ == "__main__":
    main()
