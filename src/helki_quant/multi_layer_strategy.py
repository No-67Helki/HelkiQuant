# Copyright (c) HelkiQuant contributors.
# Licensed under the MIT License.
"""
MultiLayerStrategy
------------------
三层嵌套机器学习策略：
    外层 (CatBoost 分类) → 行情状态：主升浪 / 震荡 / 主跌浪
    中层 (CatBoost 分类) → 短期波段三态：主导日频加仓 / 减仓
    内层 (CatBoost 分类) → 分钟级日内T方向：主导低买高卖 / 高抛低吸

策略根据三层信号综合决策，生成日级 TradeDecision，
由内层 NestedExecutor + TWAPStrategy 在分钟级拆单执行。

三种行情状态下的交易逻辑：
    主升浪: 外层提高风险预算，中层主导波段加减仓，不做日内T
    震荡  : 中层轻仓调整波段，内层主导日内T
    主跌浪: 外层降低风险预算，中层主导减仓/小仓反弹，内层只做高抛低吸型T
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.strategy.base import BaseStrategy
from qlib.log import get_module_logger

from decision_core import AccountState, StrategyParams, build_trade_plan


class MultiLayerStrategy(BaseStrategy):
    """三层嵌套机器学习日内做T策略。

    Parameters
    ----------
    stock_id : str
        交易标的代码
    pred_outer : pd.Series
        外层模型预测信号（P(主升浪) - P(主跌浪)，值域 [-1, 1]）
    pred_middle : pd.Series
        中层模型预测信号（短期波段三态，P(上涨)-P(下跌)）
    pred_inner : pd.Series
        内层模型预测信号（下一交易日日内T收益为正的概率）
    outer_upper : float
        外层信号 > 此值判定为主升浪
    outer_lower : float
        外层信号 < 此值判定为主跌浪
    middle_buy_thresh : float
        中层信号 > 此值判定为波段低点（加仓信号）
    middle_sell_thresh : float
        中层信号 < 此值判定为波段拐点（减仓信号）
    inner_buy_thresh : float
        内层信号 > 此值判定为日内看涨
    inner_sell_thresh : float
        内层信号 < 此值判定为日内看跌
    trade_unit : int
        最小交易单位（A股=100）
    position_pct : float
        单笔交易使用账户总价值的比例
    wave_buy_pct : float
        波段加仓使用的可用资金比例
    wave_sell_pct : float
        波段减仓的持仓比例
    """

    def __init__(
        self,
        *,
        stock_id: str,
        pred_outer: pd.Series,
        pred_middle: pd.Series,
        pred_inner: pd.Series,
        outer_upper: float = 0.3,
        outer_lower: float = -0.3,
        middle_buy_thresh: float = 0.02,
        middle_sell_thresh: float = -0.02,
        inner_buy_thresh: float = 0.005,
        inner_sell_thresh: float = -0.005,
        trade_unit: int = 100,
        position_pct: float = 0.30,
        wave_buy_pct: float = 0.20,
        wave_sell_pct: float = 0.30,
        range_swing_pct: float = 0.10,
        intraday_t_pct: float = 0.30,
        max_position_pct: float = 0.70,
        min_cash_reserve: float = 1000.0,
        min_buy_lots: int = 1,
        exit_on_bull_end: bool = True,
        level_infra=None,
        common_infra=None,
        trade_exchange=None,
    ):
        super().__init__(
            level_infra=level_infra,
            common_infra=common_infra,
            trade_exchange=trade_exchange,
        )
        if not stock_id:
            raise ValueError("`stock_id` 必须显式指定")

        self.stock_id = stock_id
        self.pred_outer = pred_outer
        self.pred_middle = pred_middle
        self.pred_inner = pred_inner

        self.outer_upper = outer_upper
        self.outer_lower = outer_lower
        self.middle_buy_thresh = middle_buy_thresh
        self.middle_sell_thresh = middle_sell_thresh
        self.inner_buy_thresh = inner_buy_thresh
        self.inner_sell_thresh = inner_sell_thresh
        self.trade_unit = trade_unit
        self.position_pct = position_pct
        self.wave_buy_pct = wave_buy_pct
        self.wave_sell_pct = wave_sell_pct
        self.range_swing_pct = range_swing_pct
        self.intraday_t_pct = intraday_t_pct
        self.max_position_pct = max_position_pct
        self.min_cash_reserve = min_cash_reserve
        self.min_buy_lots = min_buy_lots
        self.exit_on_bull_end = exit_on_bull_end
        self._last_regime: str | None = None
        self._bull_exit_active = False

        self.logger = get_module_logger("MultiLayerStrategy")

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #
    def _round_unit(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return float(int(amount // self.trade_unit) * self.trade_unit)

    def _lookup(self, pred: pd.Series, start_time, end_time) -> float:
        """从预计算的预测序列中提取指定时间范围的信号值。"""
        if pred is None or pred.empty:
            return np.nan
        try:
            idx = pred.index
            if isinstance(idx, pd.MultiIndex):
                dt_level = idx.get_level_values("datetime")
                mask = (dt_level >= start_time) & (dt_level <= end_time)
                sub = pred.loc[mask]
                if not sub.empty:
                    return float(sub.iloc[0])
            else:
                mask = (idx >= start_time) & (idx <= end_time)
                sub = pred.loc[mask]
                if not sub.empty:
                    return float(sub.iloc[0])
        except Exception:
            pass
        return np.nan

    def _get_ref_price(self, start_time, end_time) -> float:
        """获取交易时段的参考价格（收盘价）。"""
        try:
            price = self.trade_exchange.get_close(
                stock_id=self.stock_id,
                start_time=start_time,
                end_time=end_time,
            )
            if price is not None:
                return float(price)
        except Exception:
            pass
        return np.nan

    def _get_position_info(self):
        """返回 (cash, total_value, held_amount, held_available)。"""
        pos = self.trade_position
        try:
            total_value = pos.calculate_value()
        except Exception:
            total_value = pos.get_cash() if pos else 0.0
        cash = pos.get_cash() if pos else 0.0
        try:
            held_amount = pos.get_stock_amount(self.stock_id)
        except Exception:
            held_amount = 0.0
        try:
            held_available = pos.get_stock_available(self.stock_id)
        except Exception:
            held_available = held_amount
        return cash, total_value, held_amount, held_available

    def _make_buy_order(self, budget: float, price: float,
                        start_time, end_time) -> Optional[Order]:
        """创建买入订单。返回 None 如果金额不足。"""
        if not np.isfinite(price) or price <= 0 or budget <= 0:
            return None
        amount = self._round_unit(budget / price)
        if amount <= 0:
            return None
        return Order(
            stock_id=self.stock_id,
            amount=amount,
            start_time=start_time,
            end_time=end_time,
            direction=OrderDir.BUY,
        )

    def _make_sell_order(self, max_amount: float, sell_pct: float,
                         start_time, end_time) -> Optional[Order]:
        """创建卖出订单。sell_pct: 卖出 max_amount 的比例。"""
        if max_amount <= 0:
            return None
        amount = self._round_unit(max_amount * sell_pct)
        amount = min(amount, max_amount)
        if amount <= 0:
            return None
        return Order(
            stock_id=self.stock_id,
            amount=amount,
            start_time=start_time,
            end_time=end_time,
            direction=OrderDir.SELL,
        )

    def _make_buy_amount_order(self, amount: float,
                               start_time, end_time) -> Optional[Order]:
        amount = self._round_unit(amount)
        if amount <= 0:
            return None
        return Order(
            stock_id=self.stock_id,
            amount=amount,
            start_time=start_time,
            end_time=end_time,
            direction=OrderDir.BUY,
        )

    def _make_sell_amount_order(self, amount: float,
                                start_time, end_time) -> Optional[Order]:
        amount = self._round_unit(amount)
        if amount <= 0:
            return None
        return Order(
            stock_id=self.stock_id,
            amount=amount,
            start_time=start_time,
            end_time=end_time,
            direction=OrderDir.SELL,
        )

    def _leg_time_range(self, phase: str, trade_start, trade_end):
        """把共享决策里的 morning/close 腿映射到 Qlib 分钟撮合窗口。"""
        try:
            day = pd.Timestamp(trade_start).date()
            if phase == "morning":
                return pd.Timestamp.combine(day, pd.Timestamp("09:31").time()), \
                    pd.Timestamp.combine(day, pd.Timestamp("09:35").time())
            if phase == "close":
                return pd.Timestamp.combine(day, pd.Timestamp("14:45").time()), \
                    pd.Timestamp.combine(day, pd.Timestamp("14:50").time())
        except Exception:
            pass
        return trade_start, trade_end

    def _strategy_params(self) -> StrategyParams:
        return StrategyParams(
            outer_upper=self.outer_upper,
            outer_lower=self.outer_lower,
            middle_buy_thresh=self.middle_buy_thresh,
            middle_sell_thresh=self.middle_sell_thresh,
            inner_buy_thresh=self.inner_buy_thresh,
            inner_sell_thresh=self.inner_sell_thresh,
            trade_unit=self.trade_unit,
            wave_buy_pct=self.wave_buy_pct,
            wave_sell_pct=self.wave_sell_pct,
            range_swing_pct=self.range_swing_pct,
            intraday_t_pct=self.intraday_t_pct,
            max_position_pct=self.max_position_pct,
            min_cash_reserve=self.min_cash_reserve,
            min_buy_lots=self.min_buy_lots,
            exit_on_bull_end=self.exit_on_bull_end,
        )

    # ------------------------------------------------------------------ #
    # 核心决策逻辑
    # ------------------------------------------------------------------ #
    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start, trade_end = self.trade_calendar.get_step_time(trade_step)
        pred_start, pred_end = self.trade_calendar.get_step_time(trade_step, shift=1)

        # 获取三层信号
        outer_signal = self._lookup(self.pred_outer, pred_start, pred_end)
        middle_signal = self._lookup(self.pred_middle, pred_start, pred_end)
        inner_signal = self._lookup(self.pred_inner, pred_start, pred_end)

        if not (np.isfinite(outer_signal) and np.isfinite(middle_signal)
                and np.isfinite(inner_signal)):
            return TradeDecisionWO(order_list=[], strategy=self)

        # 资金与持仓
        cash, total_value, held_amount, held_available = self._get_position_info()
        price_ref = self._get_ref_price(trade_start, trade_end)
        if not np.isfinite(price_ref) or price_ref <= 0:
            self.logger.warning(f"{trade_start} 无法取得参考价，跳过")
            return TradeDecisionWO(order_list=[], strategy=self)

        plan = build_trade_plan(
            self._strategy_params(),
            outer_signal,
            middle_signal,
            inner_signal,
            AccountState(
                cash_available=cash,
                nav=total_value,
                held_volume=int(held_amount),
                held_available=int(held_available),
                price=price_ref,
            ),
            previous_regime=self._last_regime,
            force_bull_exit=self._bull_exit_active,
        )
        # 当前 step 的成交结果下一次生成决策时才可见；保持退出状态直到持仓实际归零。
        self._bull_exit_active = (
            plan.swing_action == "BULL_EXIT"
            and plan.regime != "A.主升浪"
            and int(held_amount) >= self.trade_unit
        )
        self._last_regime = plan.regime

        order_list: List[Order] = []
        for leg in (*plan.morning_legs, *plan.close_legs):
            leg_start, leg_end = self._leg_time_range(leg.phase, trade_start, trade_end)
            if leg.action == "BUY":
                order = self._make_buy_amount_order(
                    amount=leg.volume,
                    start_time=leg_start,
                    end_time=leg_end,
                )
            else:
                order = self._make_sell_amount_order(
                    amount=leg.volume,
                    start_time=leg_start,
                    end_time=leg_end,
                )
            if order:
                order_list.append(order)

        return TradeDecisionWO(order_list=order_list, strategy=self)
