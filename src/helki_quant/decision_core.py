from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


Action = Literal["BUY", "SELL"]
Phase = Literal["morning", "close"]


@dataclass(frozen=True)
class StrategyParams:
    outer_upper: float
    outer_lower: float
    middle_buy_thresh: float
    middle_sell_thresh: float
    inner_buy_thresh: float
    inner_sell_thresh: float
    trade_unit: int = 100
    wave_buy_pct: float = 0.20
    wave_sell_pct: float = 0.30
    range_swing_pct: float = 0.10
    intraday_t_pct: float = 0.30
    max_position_pct: float = 0.70
    min_cash_reserve: float = 1000.0
    min_buy_lots: int = 1
    exit_on_bull_end: bool = False


@dataclass(frozen=True)
class AccountState:
    cash_available: float
    nav: float
    held_volume: int
    held_available: int
    price: float


@dataclass(frozen=True)
class TradeLeg:
    phase: Phase
    tag: str
    action: Action
    volume: int


@dataclass(frozen=True)
class DecisionPlan:
    regime: str
    middle_state: str
    swing_action: str
    swing_pct: float
    t_action: str
    t_pct: float
    morning_legs: tuple[TradeLeg, ...]
    close_legs: tuple[TradeLeg, ...]


def round_unit(amount: float, unit: int) -> int:
    if amount <= 0:
        return 0
    return int(amount // unit) * unit


def classify_regime(outer_score: float, params: StrategyParams) -> str:
    if outer_score > params.outer_upper:
        return "A.主升浪"
    if outer_score < params.outer_lower:
        return "C.主跌浪"
    return "B.震荡"


def decide_swing(
    regime: str,
    middle_score: float,
    inner_score: float,
    params: StrategyParams,
) -> tuple[str, float]:
    """外层决定风险状态，中层主导波段加减仓。"""
    if regime == "A.主升浪":
        if middle_score > params.middle_buy_thresh:
            return "BUY", params.wave_buy_pct
        if middle_score < params.middle_sell_thresh:
            return "SELL", params.wave_sell_pct
    elif regime == "C.主跌浪":
        if middle_score < params.middle_sell_thresh:
            return "SELL", params.wave_sell_pct
        if middle_score > params.middle_buy_thresh:
            return "BUY", params.wave_buy_pct * 0.5
        # 主跌浪中若短波段不再下跌且内层出现反弹信号，允许极小仓试探。
        if middle_score >= params.middle_sell_thresh and inner_score > params.inner_buy_thresh:
            return "PROBE_BUY", params.wave_buy_pct * 0.25
    else:
        if middle_score > params.middle_buy_thresh:
            return "BUY", params.range_swing_pct
        if middle_score < params.middle_sell_thresh:
            return "SELL", params.range_swing_pct
    return "HOLD", 0.0


def classify_middle_state(middle_score: float, params: StrategyParams) -> str:
    """中层三态：上涨 / 震荡 / 下跌。"""
    if middle_score > params.middle_buy_thresh:
        return "UP"
    if middle_score < params.middle_sell_thresh:
        return "DOWN"
    return "RANGE"


def _inner_t_action(inner_score: float, params: StrategyParams) -> str:
    if inner_score > params.inner_buy_thresh:
        return "BUY_THEN_SELL"
    if inner_score < params.inner_sell_thresh:
        return "SELL_THEN_BUY"
    return "NONE"


def decide_intraday_t(
    regime: str,
    middle_state: str,
    inner_score: float,
    params: StrategyParams,
) -> tuple[str, float]:
    """内层主导日内T方向，中层在震荡行情里给当天三态约束。"""
    inner_action = _inner_t_action(inner_score, params)

    if regime == "A.主升浪":
        if inner_action == "NONE":
            return "NONE", 0.0
        return inner_action, params.intraday_t_pct * 0.5

    if regime == "B.震荡":
        if middle_state == "UP":
            # 当天偏上涨：低买高卖为主；若内层强烈反向则不做T。
            action = "NONE" if inner_action == "SELL_THEN_BUY" else "BUY_THEN_SELL"
        elif middle_state == "DOWN":
            # 当天偏下跌：高抛低吸为主；若内层强烈反向则不做T。
            action = "NONE" if inner_action == "BUY_THEN_SELL" else "SELL_THEN_BUY"
        else:
            # 震荡中性：完全交给内层判断方向。
            action = inner_action
        return action, params.intraday_t_pct if action != "NONE" else 0.0

    # 主跌浪只允许高抛低吸型T；不做低买高卖型逆势T。
    action = inner_action if inner_action == "SELL_THEN_BUY" else "NONE"
    if action == "NONE":
        return "NONE", 0.0

    # 趋势状态下降低做T强度，避免与波段仓位反复打架。
    return action, params.intraday_t_pct * 0.5


def calc_buy_volume(
    account: AccountState,
    params: StrategyParams,
    size_pct: float,
    *,
    max_volume: int | None = None,
) -> int:
    if account.price <= 0 or size_pct <= 0:
        return 0
    budget_cash = max(0.0, account.cash_available - params.min_cash_reserve)
    budget = budget_cash * size_pct
    cur_pos_value = account.held_volume * account.price
    remaining_capacity = max(0.0, account.nav * params.max_position_pct - cur_pos_value)
    budget = min(budget, remaining_capacity, budget_cash)
    min_buy_value = account.price * params.trade_unit * params.min_buy_lots
    if budget < min_buy_value:
        return 0
    volume = round_unit(budget / account.price, params.trade_unit)
    if max_volume is not None:
        volume = min(volume, round_unit(max_volume, params.trade_unit))
    return volume


def build_trade_plan(
    params: StrategyParams,
    outer_score: float,
    middle_score: float,
    inner_score: float,
    account: AccountState,
    previous_regime: str | None = None,
    force_bull_exit: bool = False,
) -> DecisionPlan:
    if not all(isfinite(v) for v in (outer_score, middle_score, inner_score, account.price)):
        return DecisionPlan("INVALID", "RANGE", "HOLD", 0.0, "NONE", 0.0, (), ())

    regime = classify_regime(outer_score, params)
    middle_state = classify_middle_state(middle_score, params)
    swing_action, swing_pct = decide_swing(regime, middle_score, inner_score, params)
    t_action, t_pct = decide_intraday_t(regime, middle_state, inner_score, params)

    cash_left = float(account.cash_available)
    held_volume = int(account.held_volume)
    held_available = int(account.held_available)
    morning_legs: list[TradeLeg] = []
    close_legs: list[TradeLeg] = []

    if params.exit_on_bull_end and regime != "A.主升浪" and (
        previous_regime == "A.主升浪" or force_bull_exit
    ):
        if held_available > 0:
            vol = round_unit(held_available, params.trade_unit)
            if vol > 0:
                morning_legs.append(TradeLeg("morning", "BULL-EXIT", "SELL", vol))
        return DecisionPlan(
            regime=regime,
            middle_state=middle_state,
            swing_action="BULL_EXIT",
            swing_pct=1.0,
            t_action="NONE",
            t_pct=0.0,
            morning_legs=tuple(morning_legs),
            close_legs=(),
        )

    if swing_action in {"BUY", "PROBE_BUY"}:
        vol = calc_buy_volume(
            AccountState(cash_left, account.nav, held_volume, held_available, account.price),
            params,
            swing_pct,
        )
        if vol > 0:
            morning_legs.append(TradeLeg("morning", "SWING", "BUY", vol))
            cash_left -= vol * account.price
            held_volume += vol
    elif swing_action == "SELL":
        vol = min(round_unit(held_available * swing_pct, params.trade_unit), held_available)
        if vol == 0 and held_available >= params.trade_unit and regime == "C.主跌浪":
            vol = round_unit(held_available, params.trade_unit)
        if vol > 0:
            morning_legs.append(TradeLeg("morning", "SWING", "SELL", vol))
            cash_left += vol * account.price
            held_volume = max(0, held_volume - vol)
            held_available = max(0, held_available - vol)

    if t_action == "BUY_THEN_SELL":
        # 低买高卖：买入T腿规模不能超过已有可卖底仓，尾盘卖出底仓闭合。
        vol = calc_buy_volume(
            AccountState(cash_left, account.nav, held_volume, held_available, account.price),
            params,
            t_pct,
            max_volume=held_available,
        )
        if vol > 0:
            morning_legs.append(TradeLeg("morning", "T-OPEN", "BUY", vol))
            close_legs.append(TradeLeg("close", "T-CLOSE", "SELL", vol))
    elif t_action == "SELL_THEN_BUY":
        # 高抛低吸：先卖可卖底仓，尾盘买回同等数量。
        vol = min(round_unit(held_available * t_pct, params.trade_unit), held_available)
        if vol > 0:
            morning_legs.append(TradeLeg("morning", "T-OPEN", "SELL", vol))
            close_legs.append(TradeLeg("close", "T-CLOSE", "BUY", vol))

    return DecisionPlan(
        regime=regime,
        middle_state=middle_state,
        swing_action=swing_action,
        swing_pct=swing_pct,
        t_action=t_action,
        t_pct=t_pct,
        morning_legs=tuple(morning_legs),
        close_legs=tuple(close_legs),
    )
