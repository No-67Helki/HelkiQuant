from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from universe import (
    UniverseRules,
    add_point_in_time_eligibility,
    instrument_to_code,
    load_price_panel,
)
from concentration_constraints import groups_on_date


@dataclass(frozen=True)
class CostScenario:
    name: str
    buy_cost: float
    sell_cost: float
    slippage: float
    min_cost: float = 5.0


BASE_COST = CostScenario("base", 0.0005, 0.0015, 0.0002)
STRESS_COST = CostScenario("stress", 0.0010, 0.0025, 0.0005)


@dataclass(frozen=True)
class ExperimentConfig:
    top_k: int = 5
    buffer_k: int = 10
    rebalance_every: int = 5
    initial_cash: float = 500_000.0
    lot_size: int = 100
    outer_lookback: int = 40
    outer_gap_reset_days: int = 20
    risk_base: float = 0.60
    risk_slope: float = 0.25
    risk_min: float = 0.20
    risk_max: float = 1.00
    max_board_fraction: float = 1.00
    inner_lower_q: float = 0.30
    inner_upper_q: float = 0.70
    max_group_fraction: float = 1.00


def load_predictions(artifacts_dir: Path) -> pd.DataFrame:
    frames = []
    for layer in ("outer", "middle", "inner"):
        path = artifacts_dir / "predictions" / f"pred_{layer}.csv"
        frame = pd.read_csv(path)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.rename(columns={f"pred_{layer}": layer})
        frames.append(frame[["datetime", "instrument", layer]])
    return frames[0].merge(frames[1], on=["datetime", "instrument"], how="outer").merge(
        frames[2], on=["datetime", "instrument"], how="outer"
    )


def _next_date_map(dates: pd.Series) -> dict[pd.Timestamp, pd.Timestamp]:
    unique = pd.DatetimeIndex(pd.to_datetime(dates)).drop_duplicates().sort_values()
    return dict(zip(unique[:-1], unique[1:]))


def prepare_research_frame(
    predictions: pd.DataFrame,
    price_panel: pd.DataFrame,
    rules: UniverseRules,
) -> pd.DataFrame:
    eligible = add_point_in_time_eligibility(price_panel, rules)
    calendar_map = _next_date_map(price_panel["datetime"])
    signal = predictions.copy()
    signal["trade_date"] = signal["datetime"].map(calendar_map)
    signal = signal.dropna(subset=["trade_date", "middle"])
    known = eligible[
        ["datetime", "instrument", "eligible", "avg_amount", "listing_days"]
    ].rename(columns={"datetime": "signal_date"})
    signal = signal.rename(columns={"datetime": "signal_date"}).merge(
        known,
        on=["signal_date", "instrument"],
        how="left",
    )
    trade_prices = price_panel[
        ["datetime", "instrument", "open", "close"]
    ].rename(columns={"datetime": "trade_date"})
    return signal.merge(trade_prices, on=["trade_date", "instrument"], how="inner")


def add_risk_and_timing_thresholds(frame: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    out = frame.copy().sort_values(["signal_date", "instrument"])
    # The regime must describe the same point-in-time tradable universe used
    # by the portfolio, not newly listed or illiquid names that cannot be held.
    eligible = out[out["eligible"].fillna(False)]
    daily_outer = eligible.groupby("signal_date")["outer"].median().sort_index()
    daily_outer = daily_outer.reindex(out["signal_date"].drop_duplicates().sort_values())
    min_periods = max(10, cfg.outer_lookback // 2)
    gap_group = (
        daily_outer.index.to_series()
        .diff()
        .dt.days.gt(cfg.outer_gap_reset_days)
        .cumsum()
        .to_numpy()
    )
    grouped_outer = daily_outer.groupby(gap_group)
    mean = grouped_outer.transform(
        lambda values: values.rolling(cfg.outer_lookback, min_periods=min_periods).mean()
    )
    std = grouped_outer.transform(
        lambda values: values.rolling(cfg.outer_lookback, min_periods=min_periods).std()
    )
    z = ((daily_outer - mean) / std.replace(0.0, np.nan)).fillna(0.0)
    risk = (
        cfg.risk_base + cfg.risk_slope * z
    ).clip(cfg.risk_min, cfg.risk_max).rename("risk_budget")
    out = out.merge(risk, left_on="signal_date", right_index=True, how="left")

    if out["inner"].notna().any():
        grouped_inner = out.groupby("signal_date")["inner"]
        out["inner_low"] = grouped_inner.transform(
            lambda values: values.quantile(cfg.inner_lower_q)
        )
        out["inner_high"] = grouped_inner.transform(
            lambda values: values.quantile(cfg.inner_upper_q)
        )
    else:
        out["inner_low"] = np.nan
        out["inner_high"] = np.nan
    return out


def _round_lot(value: float, lot_size: int) -> int:
    return max(0, int(value // lot_size) * lot_size)


def _board(instrument: str) -> str:
    code = instrument_to_code(instrument)
    return code[:2]


def _select_with_board_cap(
    ranked: list[str],
    previous_selection: set[str],
    cfg: ExperimentConfig,
) -> list[str]:
    buffer = set(ranked[: cfg.buffer_k])
    retained = [inst for inst in ranked if inst in previous_selection and inst in buffer]
    candidates = retained + [inst for inst in ranked if inst not in retained]
    board_limit = max(1, int(np.ceil(cfg.top_k * cfg.max_board_fraction)))
    selected = []
    board_counts: dict[str, int] = {}
    for inst in candidates:
        board = _board(inst)
        if board_counts.get(board, 0) >= board_limit:
            continue
        selected.append(inst)
        board_counts[board] = board_counts.get(board, 0) + 1
        if len(selected) >= cfg.top_k:
            break
    return selected


def _select_with_group_cap(
    ranked: list[str],
    previous_selection: set[str],
    cfg: ExperimentConfig,
    groups: dict[str, str],
) -> list[str]:
    buffer = set(ranked[: cfg.buffer_k])
    retained = [inst for inst in ranked if inst in previous_selection and inst in buffer]
    candidates = retained + [inst for inst in ranked if inst not in retained]
    group_limit = max(1, int(np.floor(cfg.top_k * cfg.max_group_fraction)))
    selected = []
    group_counts: dict[str, int] = {}
    for inst in candidates:
        group = groups.get(inst, "__UNKNOWN__")
        if group_counts.get(group, 0) >= group_limit:
            continue
        selected.append(inst)
        group_counts[group] = group_counts.get(group, 0) + 1
        if len(selected) >= cfg.top_k:
            break
    return selected


def _metrics(nav: pd.Series, trades: int, turnover: float) -> dict:
    returns = nav.pct_change().dropna()
    sharpe = 0.0 if returns.std() <= 1e-12 else returns.mean() / returns.std() * np.sqrt(252)
    ann_return = 0.0 if len(nav) < 2 else (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1
    peak = nav.cummax()
    mdd = ((peak - nav) / peak).max()
    return {
        "days": int(len(nav)),
        "final_nav": float(nav.iloc[-1]),
        "annualized_return": float(ann_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "trades": int(trades),
        "turnover": float(turnover),
    }


def replay_topk(
    frame: pd.DataFrame,
    *,
    experiment: str,
    cfg: ExperimentConfig,
    cost: CostScenario,
    target: str = "SZ301536",
    group_metadata: pd.DataFrame | None = None,
) -> dict:
    if experiment not in {"A", "B", "C", "D"}:
        raise ValueError("experiment must be A, B, C, or D")
    by_date = {date: part.copy() for date, part in frame.groupby("trade_date", sort=True)}
    cash = float(cfg.initial_cash)
    held: dict[str, int] = {}
    nav_rows = []
    trades = 0
    turnover = 0.0
    previous_selection: set[str] = set()
    last_close: dict[str, float] = {}

    for day_no, (trade_date, day) in enumerate(by_date.items()):
        prices_open = day.set_index("instrument")["open"].to_dict()
        prices_close = day.set_index("instrument")["close"].to_dict()
        open_nav = cash + sum(
            held.get(inst, 0)
            * prices_open.get(inst, prices_close.get(inst, last_close.get(inst, 0.0)))
            for inst in held
        )
        rebalance = day_no % cfg.rebalance_every == 0

        if rebalance:
            eligible = day[day["eligible"].fillna(False)].sort_values("middle", ascending=False)
            if experiment == "A":
                selected = [target] if target in set(eligible["instrument"]) else []
                risk_budget = 0.70
            else:
                ranked = eligible["instrument"].tolist()
                if group_metadata is not None:
                    groups = groups_on_date(group_metadata, trade_date)
                    selected = _select_with_group_cap(
                        ranked,
                        previous_selection,
                        cfg,
                        groups,
                    )
                else:
                    selected = _select_with_board_cap(ranked, previous_selection, cfg)
                risk_budget = 1.0 if experiment == "B" else float(day["risk_budget"].median())
            previous_selection = set(selected)
            target_weight = risk_budget / len(selected) if selected else 0.0

            desired: dict[str, int] = {}
            for inst in selected:
                price = prices_open.get(inst)
                if price and price > 0:
                    desired[inst] = _round_lot(open_nav * target_weight / price, cfg.lot_size)

            deltas = {inst: desired.get(inst, 0) - held.get(inst, 0) for inst in set(held) | set(desired)}
            signal = day.set_index("instrument")

            def timing(inst: str, delta: int) -> str:
                if experiment != "D" or inst not in signal.index:
                    return "open"
                row = signal.loc[inst]
                if not np.isfinite(row.get("inner", np.nan)):
                    return "open"
                if delta > 0:
                    return "open" if row["inner"] >= row["inner_high"] else "close"
                return "open" if row["inner"] <= row["inner_low"] else "close"

            for phase, price_map in (("open", prices_open), ("close", prices_close)):
                phase_deltas = {i: d for i, d in deltas.items() if d < 0 and timing(i, d) == phase}
                phase_deltas.update({i: d for i, d in deltas.items() if d > 0 and timing(i, d) == phase})
                for inst, delta in sorted(phase_deltas.items(), key=lambda item: item[1]):
                    price = price_map.get(inst)
                    if not price or price <= 0 or delta == 0:
                        continue
                    nav_ref = cash + sum(
                        held.get(code, 0)
                        * prices_close.get(code, price_map.get(code, last_close.get(code, 0.0)))
                        for code in held
                    )
                    if delta < 0:
                        volume = min(-delta, held.get(inst, 0))
                        fill_price = price * (1 - cost.slippage)
                        value = volume * fill_price
                        fee = max(value * cost.sell_cost, cost.min_cost)
                        cash += value - fee
                        held[inst] = held.get(inst, 0) - volume
                    else:
                        fill_price = price * (1 + cost.slippage)
                        affordable = _round_lot(
                            max(0.0, cash - cost.min_cost) / (fill_price * (1 + cost.buy_cost)),
                            cfg.lot_size,
                        )
                        volume = min(delta, affordable)
                        value = volume * fill_price
                        fee = max(value * cost.buy_cost, cost.min_cost) if volume > 0 else 0.0
                        cash -= value + fee
                        held[inst] = held.get(inst, 0) + volume
                    if volume > 0:
                        trades += 1
                        turnover += value / max(nav_ref, 1e-12)
                held = {inst: volume for inst, volume in held.items() if volume > 0}

        close_nav = cash + sum(
            held.get(inst, 0)
            * prices_close.get(inst, prices_open.get(inst, last_close.get(inst, 0.0)))
            for inst in held
        )
        last_close.update(
            {
                inst: float(price)
                for inst, price in prices_close.items()
                if np.isfinite(price) and price > 0
            }
        )
        nav_rows.append((trade_date, close_nav))

    nav = pd.Series(dict(nav_rows)).sort_index()
    return {
        "experiment": experiment,
        "cost": asdict(cost),
        "config": asdict(cfg),
        "metrics": _metrics(nav, trades, turnover),
        "nav": nav,
    }
