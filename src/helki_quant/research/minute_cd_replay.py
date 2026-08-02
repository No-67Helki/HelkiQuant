from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .concentration_constraints import (
        ConcentrationRules,
        concentration_snapshot,
        groups_on_date,
        load_group_metadata,
        select_with_group_cap,
    )
    from .portfolio_experiments import BASE_COST, STRESS_COST, CostScenario
    from .universe import (
        UniverseRules,
        add_point_in_time_eligibility,
        instrument_to_code,
        load_price_panel,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from concentration_constraints import (
        ConcentrationRules,
        concentration_snapshot,
        groups_on_date,
        load_group_metadata,
        select_with_group_cap,
    )
    from portfolio_experiments import BASE_COST, STRESS_COST, CostScenario
    from universe import (
        UniverseRules,
        add_point_in_time_eligibility,
        instrument_to_code,
        load_price_panel,
    )


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_STAGE_CSV = DATA / "_research_1min_pool_csv_2026"
DEFAULT_MIDDLE = HERE / "outputs" / "oof" / "pit_holdout_de2_srfs_es" / "middle" / "fold_99.csv"
DEFAULT_INNER_SIMPLE = HERE / "outputs" / "oof" / "inner_exec_holdout_simple" / "inner" / "fold_99.csv"
DEFAULT_INNER_DE = HERE / "outputs" / "oof" / "inner_exec_holdout_de2_srfs_es" / "inner" / "fold_99.csv"
DEFAULT_GROUP_METADATA = DATA / "industry_theme_pit.csv"


@dataclass(frozen=True)
class ReplayConfig:
    top_k: int = 5
    buffer_k: int = 10
    rebalance_every: int = 5
    initial_cash: float = 500_000.0
    risk_budget: float = 0.60
    lot_size: int = 100
    min_listing_days: int = 250
    min_avg_amount: float = 100_000_000.0
    inner_low_q: float = 0.30
    inner_high_q: float = 0.70
    max_group_fraction: float = 0.40


def read_pool(stage_dir: Path) -> list[str]:
    return sorted(path.stem.upper() for path in stage_dir.glob("*.csv"))


def load_signal(middle_path: Path, inner_path: Path | None, pool: list[str]) -> pd.DataFrame:
    middle = pd.read_csv(middle_path, parse_dates=["datetime"])
    middle = middle[middle["instrument"].isin(pool)].copy()
    if inner_path is None:
        middle["inner"] = np.nan
    else:
        inner = pd.read_csv(inner_path, parse_dates=["datetime"])
        middle = middle.merge(inner, on=["datetime", "instrument"], how="left")
    return middle


def next_date_map(dates: pd.Series) -> dict[pd.Timestamp, pd.Timestamp]:
    unique = pd.DatetimeIndex(pd.to_datetime(dates)).drop_duplicates().sort_values()
    return dict(zip(unique[:-1], unique[1:]))


def prepare_frame(signal: pd.DataFrame, daily_prices: pd.DataFrame, cfg: ReplayConfig) -> pd.DataFrame:
    rules = UniverseRules(
        min_listing_days=cfg.min_listing_days,
        min_avg_amount=cfg.min_avg_amount,
    )
    eligible = add_point_in_time_eligibility(daily_prices, rules)
    calendar_map = next_date_map(daily_prices["datetime"])
    out = signal.copy()
    out["trade_date"] = out["datetime"].map(calendar_map)
    out = out.dropna(subset=["trade_date", "middle"])
    known = eligible[
        ["datetime", "instrument", "eligible", "avg_amount", "listing_days"]
    ].rename(columns={"datetime": "signal_date"})
    out = out.rename(columns={"datetime": "signal_date"}).merge(
        known,
        on=["signal_date", "instrument"],
        how="left",
    )
    return out


def window_vwap(day: pd.DataFrame, start_minute: int, end_minute: int) -> float:
    mask = (day["minute_of_day"] >= start_minute) & (day["minute_of_day"] <= end_minute)
    if not mask.any():
        return np.nan
    sub = day.loc[mask]
    volume = sub["volume"].to_numpy(dtype=float)
    amount = sub["amount"].to_numpy(dtype=float)
    vol_sum = float(np.nansum(volume))
    if vol_sum <= 0:
        return np.nan
    return float(np.nansum(amount) / (vol_sum + 1e-12))


def load_minute_windows(
    stage_dir: Path,
    instruments: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    for pos, inst in enumerate(instruments, start=1):
        path = stage_dir / f"{inst.lower()}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["date", "close", "volume", "amount"], parse_dates=["date"])
        frame = frame[frame["date"].dt.normalize().between(start, end)].copy()
        if frame.empty:
            continue
        frame["trade_date"] = frame["date"].dt.normalize()
        frame["minute_of_day"] = frame["date"].dt.hour * 60 + frame["date"].dt.minute
        for trade_date, day in frame.groupby("trade_date", sort=True):
            open_vwap = window_vwap(day, 9 * 60 + 31, 9 * 60 + 35)
            close_vwap = window_vwap(day, 14 * 60 + 45, 14 * 60 + 50)
            last_close = float(day.sort_values("minute_of_day")["close"].iloc[-1])
            # Raw A-share minute volume is in lots (hands) while amount is in
            # currency units. Convert amount/volume VWAP back to share price.
            if np.isfinite(last_close) and last_close > 0:
                if np.isfinite(open_vwap) and open_vwap > last_close * 10:
                    open_vwap /= 100.0
                if np.isfinite(close_vwap) and close_vwap > last_close * 10:
                    close_vwap /= 100.0
            rows.append(
                {
                    "trade_date": trade_date,
                    "instrument": inst,
                    "open_exec": open_vwap,
                    "close_exec": close_vwap,
                    "mark_close": last_close,
                }
            )
        if pos % 20 == 0:
            print(f"[minute windows] {pos}/{len(instruments)}", flush=True)
    if not rows:
        raise ValueError("No minute execution windows loaded")
    return pd.DataFrame(rows)


def select_with_buffer(ranked: list[str], previous: set[str], cfg: ReplayConfig) -> list[str]:
    buffer = set(ranked[: cfg.buffer_k])
    retained = [inst for inst in ranked if inst in previous and inst in buffer]
    selected = retained + [inst for inst in ranked if inst not in retained]
    return selected[: cfg.top_k]


def round_lot(value: float, lot_size: int) -> int:
    return max(0, int(value // lot_size) * lot_size)


def metrics(nav: pd.Series, trades: int, turnover: float, holdings_count: list[int]) -> dict:
    returns = nav.pct_change().dropna()
    sharpe = 0.0 if returns.std() <= 1e-12 else returns.mean() / returns.std() * np.sqrt(252)
    peak = nav.cummax()
    drawdown = (peak - nav) / peak
    rolling3 = returns.rolling(3, min_periods=1).sum()
    return {
        "days": int(len(nav)),
        "final_nav": float(nav.iloc[-1]),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1) if len(nav) else 0.0,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "worst_day_return": float(returns.min()) if len(returns) else 0.0,
        "worst_3day_return": float(rolling3.min()) if len(rolling3) else 0.0,
        "trades": int(trades),
        "turnover": float(turnover),
        "avg_holdings": float(np.mean(holdings_count)) if holdings_count else 0.0,
    }


def replay(
    frame: pd.DataFrame,
    minute_windows: pd.DataFrame,
    cfg: ReplayConfig,
    cost: CostScenario,
    mode: str,
    group_metadata: pd.DataFrame | None,
) -> dict:
    if mode not in {"C_open_no_inner", "D_inner_timing"}:
        raise ValueError("unsupported mode")
    merged = frame.merge(minute_windows, on=["trade_date", "instrument"], how="inner")
    by_date = {date: part.copy() for date, part in merged.groupby("trade_date", sort=True)}
    cash = cfg.initial_cash
    held: dict[str, int] = {}
    previous_selection: set[str] = set()
    nav_rows = []
    trades = 0
    turnover = 0.0
    holdings_count = []
    last_mark: dict[str, float] = {}
    concentration_rows = []

    for day_no, (trade_date, day) in enumerate(by_date.items()):
        price_open = day.set_index("instrument")["open_exec"].to_dict()
        price_close = day.set_index("instrument")["close_exec"].to_dict()
        mark_close = day.set_index("instrument")["mark_close"].to_dict()
        open_nav = cash + sum(
            held.get(inst, 0) * price_open.get(inst, mark_close.get(inst, last_mark.get(inst, 0.0)))
            for inst in held
        )
        if day_no % cfg.rebalance_every == 0:
            eligible = day[day["eligible"].fillna(False)].sort_values("middle", ascending=False)
            ranked = eligible["instrument"].tolist()
            groups = groups_on_date(group_metadata, trade_date) if group_metadata is not None else None
            selected = select_with_group_cap(
                ranked,
                previous_selection,
                top_k=cfg.top_k,
                buffer_k=cfg.buffer_k,
                groups=groups,
                rules=ConcentrationRules(max_group_fraction=cfg.max_group_fraction)
                if group_metadata is not None
                else None,
            )
            concentration_rows.append(
                {
                    "trade_date": str(pd.Timestamp(trade_date).date()),
                    "selected": selected,
                    "concentration": concentration_snapshot(selected, groups),
                }
            )
            previous_selection = set(selected)
            target_weight = cfg.risk_budget / len(selected) if selected else 0.0
            desired = {}
            for inst in selected:
                price = price_open.get(inst)
                if np.isfinite(price) and price > 0:
                    desired[inst] = round_lot(open_nav * target_weight / price, cfg.lot_size)
            deltas = {inst: desired.get(inst, 0) - held.get(inst, 0) for inst in set(held) | set(desired)}
            signal = day.set_index("instrument")
            low = day["inner"].quantile(cfg.inner_low_q) if day["inner"].notna().any() else np.nan
            high = day["inner"].quantile(cfg.inner_high_q) if day["inner"].notna().any() else np.nan

            def phase_for(inst: str, delta: int) -> str:
                if mode == "C_open_no_inner" or inst not in signal.index:
                    return "open"
                score = signal.loc[inst, "inner"]
                if not np.isfinite(score) or not np.isfinite(low) or not np.isfinite(high):
                    return "open"
                if delta > 0:
                    return "open" if score >= high else "close"
                return "open" if score <= low else "close"

            for phase, prices in (("open", price_open), ("close", price_close)):
                phase_deltas = {inst: delta for inst, delta in deltas.items() if phase_for(inst, delta) == phase}
                for inst, delta in sorted(phase_deltas.items(), key=lambda item: item[1]):
                    price = prices.get(inst)
                    if not np.isfinite(price) or price <= 0 or delta == 0:
                        continue
                    nav_ref = cash + sum(
                        held.get(code, 0) * mark_close.get(code, prices.get(code, last_mark.get(code, 0.0)))
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

        close_nav = cash + sum(
            held.get(inst, 0) * mark_close.get(inst, price_close.get(inst, last_mark.get(inst, 0.0)))
            for inst in held
        )
        last_mark.update({inst: float(price) for inst, price in mark_close.items() if np.isfinite(price) and price > 0})
        nav_rows.append((trade_date, close_nav))
        holdings_count.append(len(held))

    nav = pd.Series(dict(nav_rows)).sort_index()
    return {
        "mode": mode,
        "cost": asdict(cost),
        "metrics": metrics(nav, trades, turnover, holdings_count),
        "concentration": concentration_rows,
        "nav": {str(k.date()): float(v) for k, v in nav.items()},
    }


def run(
    stage_dir: Path,
    middle_path: Path,
    inner_simple_path: Path,
    inner_de_path: Path,
    output_path: Path,
    cfg: ReplayConfig,
    group_metadata_path: Path | None,
    start_signal: str | None = None,
    end_signal: str | None = None,
) -> dict:
    pool = read_pool(stage_dir)
    signals = {
        "no_inner": load_signal(middle_path, None, pool),
        "inner_simple": load_signal(middle_path, inner_simple_path, pool),
        "inner_densemble": load_signal(middle_path, inner_de_path, pool),
    }
    if start_signal is not None:
        start_ts = pd.Timestamp(start_signal)
        signals = {name: sig[sig["datetime"] >= start_ts].copy() for name, sig in signals.items()}
    if end_signal is not None:
        end_ts = pd.Timestamp(end_signal)
        signals = {name: sig[sig["datetime"] <= end_ts].copy() for name, sig in signals.items()}
    signal_dates = pd.concat([sig["datetime"] for sig in signals.values()])
    daily_prices = load_price_panel(
        DATA / "A_Stock_daily_qfq",
        pool,
        start="2022-01-04",
        end="2026-04-28",
    )
    prepared = {name: prepare_frame(sig, daily_prices, cfg) for name, sig in signals.items()}
    trade_dates = pd.concat([frame["trade_date"] for frame in prepared.values()])
    minute_windows = load_minute_windows(stage_dir, pool, trade_dates.min(), trade_dates.max())
    group_metadata = (
        load_group_metadata(group_metadata_path, "industry")
        if group_metadata_path is not None and group_metadata_path.exists()
        else None
    )
    rows = []
    for cost in (BASE_COST, STRESS_COST):
        rows.append(replay(prepared["no_inner"], minute_windows, cfg, cost, "C_open_no_inner", group_metadata))
        rows[-1]["signal_source"] = "middle_only"
        rows.append(replay(prepared["inner_simple"], minute_windows, cfg, cost, "D_inner_timing", group_metadata))
        rows[-1]["signal_source"] = "inner_simple"
        rows.append(replay(prepared["inner_densemble"], minute_windows, cfg, cost, "D_inner_timing", group_metadata))
        rows[-1]["signal_source"] = "inner_densemble"
    report = {
        "status": "minute_cd_replay_research_only",
        "stage_dir": str(stage_dir),
        "middle_prediction": str(middle_path),
        "inner_simple_prediction": str(inner_simple_path),
        "inner_densemble_prediction": str(inner_de_path),
        "config": asdict(cfg),
        "group_constraint": {
            "metadata_path": str(group_metadata_path) if group_metadata_path is not None else None,
            "enabled": group_metadata is not None,
            "max_group_fraction": cfg.max_group_fraction,
            "warning": None
            if group_metadata is not None
            else "No industry/theme PIT metadata found; group concentration cap was not applied.",
        },
        "window": {
            "signal_start": str(signal_dates.min().date()),
            "signal_end": str(signal_dates.max().date()),
            "trade_start": str(trade_dates.min().date()),
            "trade_end": str(trade_dates.max().date()),
        },
        "results": rows,
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE_CSV))
    parser.add_argument("--middle", default=str(DEFAULT_MIDDLE))
    parser.add_argument("--inner-simple", default=str(DEFAULT_INNER_SIMPLE))
    parser.add_argument("--inner-densemble", default=str(DEFAULT_INNER_DE))
    parser.add_argument("--output", default=str(HERE / "outputs" / "minute_cd_holdout_replay.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--risk-budget", type=float, default=0.60)
    parser.add_argument("--group-metadata", default=str(DEFAULT_GROUP_METADATA))
    parser.add_argument("--max-group-fraction", type=float, default=0.40)
    parser.add_argument("--start-signal", default=None)
    parser.add_argument("--end-signal", default=None)
    args = parser.parse_args()
    cfg = ReplayConfig(
        top_k=args.top_k,
        buffer_k=max(args.top_k * 2, args.top_k),
        rebalance_every=args.rebalance_every,
        risk_budget=args.risk_budget,
        max_group_fraction=args.max_group_fraction,
    )
    report = run(
        Path(args.stage_dir).resolve(),
        Path(args.middle).resolve(),
        Path(args.inner_simple).resolve(),
        Path(args.inner_densemble).resolve(),
        Path(args.output).resolve(),
        cfg,
        Path(args.group_metadata).resolve() if args.group_metadata else None,
        args.start_signal,
        args.end_signal,
    )
    for row in report["results"]:
        metrics_row = row["metrics"]
        print(
            f"[minute {row['signal_source']}/{row['cost']['name']}] "
            f"return={metrics_row['total_return']:+.2%} "
            f"mdd={metrics_row['max_drawdown']:.2%} "
            f"turnover={metrics_row['turnover']:.2f} "
            f"trades={metrics_row['trades']}"
        )


if __name__ == "__main__":
    main()
