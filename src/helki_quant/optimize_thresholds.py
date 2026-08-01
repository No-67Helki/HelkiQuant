"""
离线阈值寻优脚本（不走 GMSDK）。

核心原则：
- 读取 artifacts/<exp>/predictions/pred_{outer,middle,inner}.csv 中目标股预测。
- 离线重放时复用 decision_core.build_trade_plan，避免阈值寻优逻辑与实盘/Qlib 策略分叉。
- 默认 scan60：按 GM 交易时段每分钟扫描（60s bar），外/中层日频缓存，T 平仓在 14:45。
- 同时寻优 outer/middle/inner 六个阈值。
- 使用严格时间切分：前段寻优，中段验证，最后一段仅做最终 OOS 评估。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from decision_core import AccountState, StrategyParams, TradeLeg, build_trade_plan, round_unit


TRADE_UNIT = 100
DEFAULT_INITIAL_CASH = 500000.0
DEFAULT_INITIAL_HELD = 0
WAVE_BUY_PCT = 0.20
WAVE_SELL_PCT = 0.30
RANGE_SWING_PCT = 0.10
INTRADAY_T_PCT = 0.30
MAX_POSITION_PCT = 0.70
MIN_CASH_RESERVE = 1000.0
MIN_BUY_LOTS = 1
COMMISSION = 0.00015
SLIPPAGE = 0.0001
MIN_COST = 0.0
# 与 main.py 默认一致：离线重放须施加相同日内约束，避免寻优结果与 GM 回测脱节
MAX_T0_TRADES_PER_DAY = 5
MAX_SWING_TRADES_PER_DAY = 2
SWING_UNLIMITED_IN_RANGE = True
RANGE_REGIME = "B.震荡"
T_CLOSE_HM = "14:45"
SCAN_SESSIONS = (("09:31", "11:30"), ("13:00", "14:44"))
REPO_ROOT = Path(__file__).resolve().parents[2]


def _hm_to_minutes(hm: str) -> int:
    h, m = map(int, hm.split(":"))
    return h * 60 + m


def _intraday_scan_schedule() -> list[str]:
    """与 main.py 交易时段一致：09:31-11:30、13:00-14:44 每分钟一次（60s bar）。"""
    out: list[str] = []
    for start, end in SCAN_SESSIONS:
        cur = _hm_to_minutes(start)
        end_m = _hm_to_minutes(end)
        while cur <= end_m:
            out.append(f"{cur // 60:02d}:{cur % 60:02d}")
            cur += 1
    return out


SCAN_SCHEDULE = _intraday_scan_schedule()


def _scan_price_proxy(open_p: float, close_p: float, hm: str) -> float:
    """用当日 open→close 线性插值近似扫描时刻价格（无分钟行情时的离线代理）。"""
    day_open = 9 * 60 + 30
    day_close = 15 * 60
    cur = _hm_to_minutes(hm)
    frac = (cur - day_open) / max(day_close - day_open, 1)
    frac = float(np.clip(frac, 0.0, 1.0))
    return open_p + (close_p - open_p) * frac


@dataclass(frozen=True)
class ReplayLimits:
    max_swing_per_day: int = MAX_SWING_TRADES_PER_DAY
    max_t0_per_day: int = MAX_T0_TRADES_PER_DAY
    swing_unlimited_in_range: bool = SWING_UNLIMITED_IN_RANGE


def _swing_trade_blocked(day_swing_count: int, regime: str, limits: ReplayLimits) -> bool:
    if limits.swing_unlimited_in_range and regime == RANGE_REGIME:
        return False
    return day_swing_count >= limits.max_swing_per_day


@dataclass(frozen=True)
class ThresholdParams:
    outer_upper: float
    outer_lower: float
    middle_buy: float
    middle_sell: float
    inner_buy: float
    inner_sell: float

    def valid(self) -> bool:
        return (
            np.isfinite(list(asdict(self).values())).all()
            and self.outer_upper > self.outer_lower
            and self.middle_buy > self.middle_sell
            and self.inner_buy > self.inner_sell
        )

    def to_strategy_params(self, *, exit_on_bull_end: bool = False) -> StrategyParams:
        return StrategyParams(
            outer_upper=self.outer_upper,
            outer_lower=self.outer_lower,
            middle_buy_thresh=self.middle_buy,
            middle_sell_thresh=self.middle_sell,
            inner_buy_thresh=self.inner_buy,
            inner_sell_thresh=self.inner_sell,
            trade_unit=TRADE_UNIT,
            wave_buy_pct=WAVE_BUY_PCT,
            wave_sell_pct=WAVE_SELL_PCT,
            range_swing_pct=RANGE_SWING_PCT,
            intraday_t_pct=INTRADAY_T_PCT,
            max_position_pct=MAX_POSITION_PCT,
            min_cash_reserve=MIN_CASH_RESERVE,
            min_buy_lots=MIN_BUY_LOTS,
            exit_on_bull_end=exit_on_bull_end,
        )


@dataclass
class SimState:
    cash: float
    held_volume: int
    held_available: int
    unsettled_buy: int
    buy_cost_rate: float = COMMISSION
    sell_cost_rate: float = COMMISSION
    slippage: float = SLIPPAGE
    min_cost: float = MIN_COST
    previous_regime: str | None = None
    bull_exit_active: bool = False

    def settle_new_day(self) -> None:
        if self.unsettled_buy > 0:
            self.held_available = min(self.held_volume, self.held_available + self.unsettled_buy)
            self.unsettled_buy = 0


def _normalize_inst(inst: str) -> str:
    return str(inst).upper().replace(".", "")


def _prediction_frame(path: Path, target: str, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "instrument" not in df.columns or "datetime" not in df.columns:
        raise ValueError(f"{path} must contain instrument/datetime columns")

    value_col = value_name if value_name in df.columns else df.columns[-1]
    target_norm = _normalize_inst(target)
    inst_norm = df["instrument"].map(_normalize_inst)
    out = df.loc[inst_norm == target_norm, ["datetime", value_col]].copy()
    if out.empty:
        raise ValueError(f"{path} has no prediction rows for target={target}")
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.rename(columns={value_col: value_name})


def load_data(
    artifacts_dir: Path,
    target: str,
    start: str,
    end: str,
    signal_lag_days: int = 1,
) -> pd.DataFrame:
    pred_dir = artifacts_dir / "predictions"
    outer = _prediction_frame(pred_dir / "pred_outer.csv", target, "outer")
    middle = _prediction_frame(pred_dir / "pred_middle.csv", target, "middle")
    inner = _prediction_frame(pred_dir / "pred_inner.csv", target, "inner")
    sig = outer.merge(middle, on="datetime").merge(inner, on="datetime")
    sig = sig.sort_values("datetime").reset_index(drop=True)
    if signal_lag_days < 0:
        raise ValueError("signal_lag_days must be >= 0")
    if signal_lag_days:
        # Daily features are only known after that day's close. Trading at the
        # next open must therefore use the previous trading day's predictions,
        # matching MultiLayerStrategy.generate_trade_decision(shift=1).
        sig[["outer", "middle", "inner"]] = sig[
            ["outer", "middle", "inner"]
        ].shift(signal_lag_days)
        sig = sig.dropna(subset=["outer", "middle", "inner"])

    code = target.replace("SZ", "").replace("SH", "").replace(".", "")
    csv_candidates = [
        REPO_ROOT / "data" / code / "day_data" / f"{code}_daily_qfq.csv",
        REPO_ROOT / "data" / "A_Stock_daily_qfq" / f"{code}_daily_qfq.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(f"daily csv not found for {target}: {csv_candidates}")

    day = pd.read_csv(csv_path)
    day = day.rename(
        columns={
            "日期": "datetime",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
        },
    )
    day["datetime"] = pd.to_datetime(day["datetime"])
    day = day[["datetime", "open", "close", "high", "low"]]

    df = sig.merge(day, on="datetime").sort_values("datetime").reset_index(drop=True)
    df = df[(df["datetime"] >= pd.Timestamp(start)) & (df["datetime"] <= pd.Timestamp(end))]
    return df.reset_index(drop=True)


def _round_volume(volume: float) -> int:
    return round_unit(volume, TRADE_UNIT)


def _execute_leg(
    state: SimState,
    leg: TradeLeg,
    *,
    price: float,
    nav_ref: float,
) -> tuple[int, float]:
    """Execute one leg and return (filled_volume, turnover_value)."""
    volume = _round_volume(leg.volume)
    if volume <= 0 or price <= 0:
        return 0, 0.0

    if leg.action == "BUY":
        buy_px = price * (1.0 + state.slippage)
        affordable = _round_volume(
            max(0.0, state.cash - state.min_cost)
            / (buy_px * (1.0 + state.buy_cost_rate))
        )
        volume = min(volume, affordable)
        if volume <= 0:
            return 0, 0.0
        trade_value = volume * buy_px
        cost = max(trade_value * state.buy_cost_rate, state.min_cost)
        state.cash -= trade_value + cost
        state.held_volume += volume
        state.unsettled_buy += volume
        return volume, trade_value / max(nav_ref, 1e-12)

    sell_px = price * (1.0 - state.slippage)
    volume = min(volume, _round_volume(state.held_available), _round_volume(state.held_volume))
    if volume <= 0:
        return 0, 0.0
    trade_value = volume * sell_px
    cost = max(trade_value * state.sell_cost_rate, state.min_cost)
    state.cash += trade_value - cost
    state.held_volume -= volume
    state.held_available = max(0, state.held_available - volume)
    state.unsettled_buy = min(state.unsettled_buy, state.held_volume)
    return volume, trade_value / max(nav_ref, 1e-12)


def _apply_morning_legs(
    state: SimState,
    plan,
    *,
    price: float,
    limits: ReplayLimits,
    day_swing: int,
    day_t_open: int,
) -> tuple[int, int, int, float, list[TradeLeg], bool]:
    """执行早盘腿；返回 (n_trades, turnover, day_swing, day_t_open, pending_close, t_open_hit)。"""
    n_trades = 0
    turnover = 0.0
    t_open_hit = False
    pending_close: list[TradeLeg] = []
    nav_ref = state.cash + state.held_volume * price

    for leg in plan.morning_legs:
        if leg.tag in {"SWING", "BULL-EXIT"} and _swing_trade_blocked(day_swing, plan.regime, limits):
            continue
        if leg.tag == "T-OPEN" and day_t_open >= limits.max_t0_per_day:
            continue
        filled, turn = _execute_leg(state, leg, price=price, nav_ref=nav_ref)
        if filled > 0:
            n_trades += 1
            turnover += turn
            nav_ref = state.cash + state.held_volume * price
            if leg.tag in {"SWING", "BULL-EXIT"}:
                day_swing += 1
            if leg.tag == "T-OPEN":
                day_t_open += 1
                t_open_hit = True

    if t_open_hit and plan.close_legs:
        pending_close.extend(plan.close_legs)

    return n_trades, turnover, day_swing, day_t_open, pending_close, t_open_hit


def _replay_daily(
    df: pd.DataFrame,
    strategy_params: StrategyParams,
    state: SimState,
    limits: ReplayLimits,
) -> tuple[list[float], int, float, int, int, int, int, int]:
    navs: list[float] = []
    n_trades = 0
    turnover = 0.0
    bull_days = range_days = bear_days = 0
    t_open_count = swing_count = 0

    for _, row in df.iterrows():
        state.settle_new_day()
        open_p = float(row["open"])
        close_p = float(row["close"])
        if not np.isfinite(open_p) or not np.isfinite(close_p) or open_p <= 0 or close_p <= 0:
            navs.append(state.cash + state.held_volume * close_p)
            continue

        plan = build_trade_plan(
            strategy_params,
            float(row["outer"]),
            float(row["middle"]),
            float(row["inner"]),
            AccountState(
                cash_available=state.cash,
                nav=state.cash + state.held_volume * open_p,
                held_volume=state.held_volume,
                held_available=state.held_available,
                price=open_p,
            ),
            previous_regime=state.previous_regime,
            force_bull_exit=state.bull_exit_active,
        )
        if plan.regime == "A.主升浪":
            bull_days += 1
        elif plan.regime == "C.主跌浪":
            bear_days += 1
        else:
            range_days += 1

        nt, turn, day_swing, day_t_open, _, _ = _apply_morning_legs(
            state, plan, price=open_p, limits=limits, day_swing=0, day_t_open=0,
        )
        n_trades += nt
        turnover += turn
        swing_count += day_swing
        t_open_count += day_t_open

        close_nav_ref = state.cash + state.held_volume * close_p
        for leg in plan.close_legs:
            filled, turn = _execute_leg(state, leg, price=close_p, nav_ref=close_nav_ref)
            if filled > 0:
                n_trades += 1
                turnover += turn

        state.bull_exit_active = (
            plan.swing_action == "BULL_EXIT"
            and plan.regime != "A.主升浪"
            and state.held_volume >= TRADE_UNIT
        )
        state.previous_regime = plan.regime
        navs.append(state.cash + state.held_volume * close_p)

    return navs, n_trades, turnover, bull_days, range_days, bear_days, t_open_count, swing_count


def _replay_scan60(
    df: pd.DataFrame,
    strategy_params: StrategyParams,
    state: SimState,
    limits: ReplayLimits,
) -> tuple[list[float], int, float, int, int, int, int, int]:
    """按 60s 扫描重放：外/中层日频缓存，内层用日频预测（与当前 pred CSV 一致）。"""
    navs: list[float] = []
    n_trades = 0
    turnover = 0.0
    bull_days = range_days = bear_days = 0
    t_open_count = swing_count = 0

    for _, row in df.iterrows():
        state.settle_new_day()
        open_p = float(row["open"])
        close_p = float(row["close"])
        outer_s = float(row["outer"])
        middle_s = float(row["middle"])
        inner_s = float(row["inner"])
        if not all(np.isfinite([open_p, close_p, outer_s, middle_s, inner_s])) or open_p <= 0 or close_p <= 0:
            navs.append(state.cash + state.held_volume * (close_p if close_p > 0 else 0.0))
            continue

        day_swing = 0
        day_t_open = 0
        pending_close: list[TradeLeg] = []
        regime_counted = False

        for hm in SCAN_SCHEDULE:
            price = _scan_price_proxy(open_p, close_p, hm)
            plan = build_trade_plan(
                strategy_params,
                outer_s,
                middle_s,
                inner_s,
                AccountState(
                    cash_available=state.cash,
                    nav=state.cash + state.held_volume * price,
                    held_volume=state.held_volume,
                    held_available=state.held_available,
                    price=price,
                ),
                previous_regime=state.previous_regime,
                force_bull_exit=state.bull_exit_active,
            )
            if not regime_counted:
                if plan.regime == "A.主升浪":
                    bull_days += 1
                elif plan.regime == "C.主跌浪":
                    bear_days += 1
                else:
                    range_days += 1
                regime_counted = True

            nt, turn, day_swing, day_t_open, new_close, _ = _apply_morning_legs(
                state,
                plan,
                price=price,
                limits=limits,
                day_swing=day_swing,
                day_t_open=day_t_open,
            )
            n_trades += nt
            turnover += turn
            if new_close:
                pending_close.extend(new_close)

            state.bull_exit_active = (
                plan.swing_action == "BULL_EXIT"
                and plan.regime != "A.主升浪"
                and state.held_volume >= TRADE_UNIT
            )
            state.previous_regime = plan.regime

        swing_count += day_swing
        t_open_count += day_t_open

        close_px = _scan_price_proxy(open_p, close_p, T_CLOSE_HM)
        close_nav_ref = state.cash + state.held_volume * close_px
        for leg in pending_close:
            filled, turn = _execute_leg(state, leg, price=close_px, nav_ref=close_nav_ref)
            if filled > 0:
                n_trades += 1
                turnover += turn
                close_nav_ref = state.cash + state.held_volume * close_px

        navs.append(state.cash + state.held_volume * close_p)

    return navs, n_trades, turnover, bull_days, range_days, bear_days, t_open_count, swing_count


def replay(
    df: pd.DataFrame,
    params: ThresholdParams,
    *,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    initial_held: int = DEFAULT_INITIAL_HELD,
    exit_on_bull_end: bool = False,
    limits: ReplayLimits | None = None,
    scan_mode: str = "scan60",
    strategy_params: StrategyParams | None = None,
    buy_cost_rate: float = COMMISSION,
    sell_cost_rate: float = COMMISSION,
    slippage: float = SLIPPAGE,
    min_cost: float = MIN_COST,
) -> dict:
    if not params.valid() or df.empty:
        return _empty_metrics()

    limits = limits or ReplayLimits()
    strategy_params = strategy_params or params.to_strategy_params(
        exit_on_bull_end=exit_on_bull_end
    )
    initial_held = _round_volume(initial_held)
    state = SimState(
        cash=float(initial_cash),
        held_volume=initial_held,
        held_available=initial_held,
        unsettled_buy=0,
        buy_cost_rate=float(buy_cost_rate),
        sell_cost_rate=float(sell_cost_rate),
        slippage=float(slippage),
        min_cost=float(min_cost),
    )

    if scan_mode == "daily":
        navs, n_trades, turnover, bull_days, range_days, bear_days, t_open_count, swing_count = _replay_daily(
            df, strategy_params, state, limits,
        )
    elif scan_mode == "scan60":
        navs, n_trades, turnover, bull_days, range_days, bear_days, t_open_count, swing_count = _replay_scan60(
            df, strategy_params, state, limits,
        )
    else:
        raise ValueError(f"unknown scan_mode={scan_mode!r}, use daily|scan60")

    out = _metrics(np.asarray(navs, dtype=float), n_trades, turnover, bull_days, range_days, bear_days, t_open_count, swing_count)
    out["scan_mode"] = scan_mode
    out["scans_per_day"] = len(SCAN_SCHEDULE)
    return out


def _empty_metrics() -> dict:
    return {
        "sharpe": -99.0,
        "ann_return": 0.0,
        "mdd": 1.0,
        "n_trades": 0,
        "turnover": 0.0,
        "nav": np.asarray([], dtype=float),
        "bull_days": 0,
        "range_days": 0,
        "bear_days": 0,
        "t_open_count": 0,
        "swing_count": 0,
    }


def _metrics(
    nav: np.ndarray,
    n_trades: int,
    turnover: float,
    bull_days: int,
    range_days: int,
    bear_days: int,
    t_open_count: int,
    swing_count: int,
) -> dict:
    if len(nav) < 2 or not np.isfinite(nav).all():
        out = _empty_metrics()
        out["n_trades"] = n_trades
        out["nav"] = nav
        return out

    rets = np.diff(nav) / nav[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = 0.0 if len(rets) < 2 or rets.std() == 0 else rets.mean() / rets.std() * np.sqrt(252)
    ann_return = (nav[-1] / nav[0]) ** (252 / len(nav)) - 1
    peak = np.maximum.accumulate(nav)
    mdd = float(((peak - nav) / peak).max())
    return {
        "sharpe": float(sharpe),
        "ann_return": float(ann_return),
        "mdd": mdd,
        "n_trades": int(n_trades),
        "turnover": float(turnover),
        "nav": nav,
        "bull_days": int(bull_days),
        "range_days": int(range_days),
        "bear_days": int(bear_days),
        "t_open_count": int(t_open_count),
        "swing_count": int(swing_count),
    }


def score_from_metrics(m: dict, *, n_days: int) -> float:
    if n_days <= 0:
        return -99.0
    score = m["sharpe"] + 0.50 * m["ann_return"]
    score -= 3.0 * max(0.0, m["mdd"] - 0.08)
    score -= 0.02 * max(0.0, m["turnover"] - 6.0)

    trades = m["n_trades"]
    min_trades = max(3, int(n_days * 0.03))
    max_trades = max(20, int(n_days * 1.50))
    if trades < min_trades:
        score -= 0.5 + 0.05 * (min_trades - trades)
    if trades > max_trades:
        score -= 0.01 * (trades - max_trades)
    if m["ann_return"] < 0:
        score += m["ann_return"]
    return float(score)


def split_data(df: pd.DataFrame, opt_ratio: float, valid_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < opt_ratio < 1 or not 0 <= valid_ratio < 1 or opt_ratio + valid_ratio >= 1:
        raise ValueError("Require 0 < opt_ratio < 1 and opt_ratio + valid_ratio < 1")
    n = len(df)
    opt_end = max(2, int(n * opt_ratio))
    valid_end = max(opt_end + 1, int(n * (opt_ratio + valid_ratio)))
    return (
        df.iloc[:opt_end].reset_index(drop=True),
        df.iloc[opt_end:valid_end].reset_index(drop=True),
        df.iloc[valid_end:].reset_index(drop=True),
    )


def _wide_bounds(lo: float, hi: float, *, eps: float = 1e-4) -> tuple[float, float]:
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if hi - lo < eps:
        mid = (lo + hi) / 2
        return mid - eps, mid + eps
    return lo, hi


def quantile_bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        "outer_upper": _wide_bounds(df["outer"].quantile(0.55), df["outer"].quantile(0.90)),
        "outer_lower": _wide_bounds(df["outer"].quantile(0.10), df["outer"].quantile(0.45)),
        "middle_buy": _wide_bounds(df["middle"].quantile(0.55), df["middle"].quantile(0.90)),
        "middle_sell": _wide_bounds(df["middle"].quantile(0.10), df["middle"].quantile(0.45)),
        "inner_buy": _wide_bounds(df["inner"].quantile(0.55), df["inner"].quantile(0.90)),
        "inner_sell": _wide_bounds(df["inner"].quantile(0.10), df["inner"].quantile(0.45)),
    }


def params_from_vector(x: Iterable[float]) -> ThresholdParams:
    vals = list(map(float, x))
    return ThresholdParams(
        outer_upper=vals[0],
        outer_lower=vals[1],
        middle_buy=vals[2],
        middle_sell=vals[3],
        inner_buy=vals[4],
        inner_sell=vals[5],
    )


def objective(
    x: np.ndarray,
    df: pd.DataFrame,
    *,
    initial_cash: float,
    initial_held: int,
    exit_on_bull_end: bool,
    limits: ReplayLimits,
    scan_mode: str,
) -> float:
    params = params_from_vector(x)
    if not params.valid():
        return 99.0

    n = len(df)
    splits = [max(2, int(n * 0.50)), max(3, int(n * 0.75)), n]
    scores = []
    for end in splits:
        sub = df.iloc[:end].reset_index(drop=True)
        m = replay(
            sub,
            params,
            initial_cash=initial_cash,
            initial_held=initial_held,
            exit_on_bull_end=exit_on_bull_end,
            limits=limits,
            scan_mode=scan_mode,
        )
        scores.append(score_from_metrics(m, n_days=len(sub)))
    return -float(np.mean(scores))


def format_metrics(name: str, m: dict) -> str:
    return (
        f"[{name}] score={score_from_metrics(m, n_days=max(len(m['nav']), 1)):+.3f} "
        f"sharpe={m['sharpe']:+.3f} ann_ret={m['ann_return']:+.2%} "
        f"mdd={m['mdd']:.2%} trades={m['n_trades']} "
        f"turnover={m['turnover']:.2f} "
        f"states=A/B/C {m['bull_days']}/{m['range_days']}/{m['bear_days']} "
        f"swing={m['swing_count']} t_open={m['t_open_count']}"
    )


def serializable_metrics(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "nav"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default=str(Path(__file__).parent / "artifacts" / "robust_v2"))
    ap.add_argument("--target", default="SZ301536")
    ap.add_argument("--start", default="2025-11-12")
    ap.add_argument("--end", default="2026-05-10")
    ap.add_argument(
        "--signal_lag_days",
        type=int,
        default=1,
        help="日频预测滞后交易日数；默认 1，避免用当天收盘后信号交易当天开盘",
    )
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opt_ratio", type=float, default=0.60)
    ap.add_argument("--valid_ratio", type=float, default=0.20)
    ap.add_argument("--initial_cash", type=float, default=DEFAULT_INITIAL_CASH)
    ap.add_argument("--initial_held", type=int, default=DEFAULT_INITIAL_HELD)
    ap.add_argument("--exit_on_bull_end", action="store_true")
    ap.add_argument("--max_swing_per_day", type=int, default=MAX_SWING_TRADES_PER_DAY)
    ap.add_argument("--max_t0_per_day", type=int, default=MAX_T0_TRADES_PER_DAY)
    ap.add_argument(
        "--swing_unlimited_in_range",
        action=argparse.BooleanOptionalAction,
        default=SWING_UNLIMITED_IN_RANGE,
    )
    ap.add_argument(
        "--scan_mode",
        choices=("scan60", "daily"),
        default="scan60",
        help="scan60=按 GM 60s 扫描重放；daily=每日开盘一次",
    )
    ap.add_argument("--output", default=None, help="JSON summary path; default: <artifacts_dir>/threshold_optimization.json")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    df = load_data(
        artifacts_dir,
        args.target,
        args.start,
        args.end,
        signal_lag_days=args.signal_lag_days,
    )
    if len(df) < 30:
        raise ValueError(f"Too few trading days for optimization: {len(df)}")

    opt_df, valid_df, test_df = split_data(df, args.opt_ratio, args.valid_ratio)
    limits = ReplayLimits(
        max_swing_per_day=args.max_swing_per_day,
        max_t0_per_day=args.max_t0_per_day,
        swing_unlimited_in_range=args.swing_unlimited_in_range,
    )
    print(f"[INFO] artifacts_dir = {artifacts_dir}")
    print(f"[INFO] loaded {len(df)} days: {df['datetime'].min().date()} -> {df['datetime'].max().date()}")
    print(f"[INFO] signal_lag_days = {args.signal_lag_days}")
    print(f"[INFO] split opt/valid/test = {len(opt_df)}/{len(valid_df)}/{len(test_df)}")
    print(
        f"[INFO] replay limits: max_swing={limits.max_swing_per_day} "
        f"max_t0={limits.max_t0_per_day} range_swing_unlimited={limits.swing_unlimited_in_range}"
    )
    print(f"[INFO] scan_mode={args.scan_mode} scans_per_day={len(SCAN_SCHEDULE)} t_close={T_CLOSE_HM}")

    bounds_d = quantile_bounds(opt_df)
    bounds = [
        bounds_d["outer_upper"],
        bounds_d["outer_lower"],
        bounds_d["middle_buy"],
        bounds_d["middle_sell"],
        bounds_d["inner_buy"],
        bounds_d["inner_sell"],
    ]
    print("[INFO] search bounds from optimization window:")
    for key, val in bounds_d.items():
        print(f"  {key:15s} [{val[0]:+.4f}, {val[1]:+.4f}]")

    baseline = ThresholdParams(
        outer_upper=0.0188,
        outer_lower=-0.0550,
        middle_buy=0.0500,
        middle_sell=-0.0500,
        inner_buy=0.5500,
        inner_sell=0.4500,
    )
    print("\n[BASELINE]")
    for name, part in (("OPT", opt_df), ("VALID", valid_df), ("TEST", test_df), ("ALL", df)):
        m = replay(
            part, baseline,
            initial_cash=args.initial_cash,
            initial_held=args.initial_held,
            exit_on_bull_end=args.exit_on_bull_end,
            limits=limits,
            scan_mode=args.scan_mode,
        )
        print("  " + format_metrics(name, m))

    print("\n[OPTIMIZE] running differential_evolution ...", flush=True)
    obj = partial(
        objective,
        df=opt_df,
        initial_cash=args.initial_cash,
        initial_held=args.initial_held,
        exit_on_bull_end=args.exit_on_bull_end,
        limits=limits,
        scan_mode=args.scan_mode,
    )
    res = differential_evolution(
        obj,
        bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        tol=1e-4,
        polish=True,
        workers=1,
        disp=True,
    )

    best = params_from_vector(res.x)
    print("\n[BEST PARAMS]")
    for key, val in asdict(best).items():
        print(f"  {key:15s} = {val:+.4f}")

    metrics = {}
    for name, part in (("opt", opt_df), ("valid", valid_df), ("test", test_df), ("all", df)):
        metrics[name] = replay(
            part, best,
            initial_cash=args.initial_cash,
            initial_held=args.initial_held,
            exit_on_bull_end=args.exit_on_bull_end,
            limits=limits,
            scan_mode=args.scan_mode,
        )
        print("  " + format_metrics(name.upper(), metrics[name]))

    print("\n[main.py 替换块]")
    print(f"OUTER_UPPER = {best.outer_upper:+.4f}")
    print(f"OUTER_LOWER = {best.outer_lower:+.4f}")
    print(f"MIDDLE_BUY_THRESH  = {best.middle_buy:+.4f}")
    print(f"MIDDLE_SELL_THRESH = {best.middle_sell:+.4f}")
    print(f"INNER_BUY_THRESH   = {best.inner_buy:+.4f}")
    print(f"INNER_SELL_THRESH  = {best.inner_sell:+.4f}")

    print("\n[YAML strategy_params 片段]")
    print(f"outer_upper: {best.outer_upper:+.4f}")
    print(f"outer_lower: {best.outer_lower:+.4f}")
    print(f"middle_buy_thresh: {best.middle_buy:+.4f}")
    print(f"middle_sell_thresh: {best.middle_sell:+.4f}")
    print(f"inner_buy_thresh: {best.inner_buy:+.4f}")
    print(f"inner_sell_thresh: {best.inner_sell:+.4f}")

    out_path = Path(args.output).expanduser().resolve() if args.output else artifacts_dir / "threshold_optimization.json"
    summary = {
        "artifacts_dir": str(artifacts_dir),
        "target": args.target,
        "start": args.start,
        "end": args.end,
        "split": {"opt": len(opt_df), "valid": len(valid_df), "test": len(test_df)},
        "replay_limits": asdict(limits),
        "scan_mode": args.scan_mode,
        "signal_lag_days": args.signal_lag_days,
        "scans_per_day": len(SCAN_SCHEDULE),
        "bounds": bounds_d,
        "best_params": asdict(best),
        "metrics": {k: serializable_metrics(v) for k, v in metrics.items()},
        "optimizer": {"fun": float(res.fun), "nit": int(res.nit), "success": bool(res.success), "message": str(res.message)},
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] summary saved to {out_path}")


if __name__ == "__main__":
    main()
