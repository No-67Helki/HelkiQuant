from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


_EPSILON = 1e-9


def _non_negative(value: float, *, field: str) -> float:
    number = float(value)
    if number < -_EPSILON:
        raise ValueError(f"{field} must be non-negative, got {value}")
    return max(0.0, number)


@dataclass
class PositionState:
    total: float = 0.0
    available: float = 0.0
    frozen: float = 0.0
    unsettled_buy: float = 0.0

    def validate(self) -> None:
        self.total = _non_negative(self.total, field="total")
        self.available = _non_negative(self.available, field="available")
        self.frozen = _non_negative(self.frozen, field="frozen")
        self.unsettled_buy = _non_negative(
            self.unsettled_buy,
            field="unsettled_buy",
        )
        if self.available > self.total + _EPSILON:
            raise ValueError("available shares cannot exceed total shares")
        if self.frozen > self.available + _EPSILON:
            raise ValueError("frozen shares cannot exceed available shares")
        if self.available + self.unsettled_buy > self.total + _EPSILON:
            raise ValueError(
                "available plus unsettled buy shares cannot exceed total shares"
            )

    @property
    def sellable(self) -> float:
        return max(0.0, self.available - self.frozen)


class T1PositionLedger:
    """A-share share ledger with buy-today/sell-next-session settlement."""

    def __init__(
        self,
        initial_positions: Mapping[str, float | Mapping[str, float]] | None = None,
    ) -> None:
        self._positions: dict[str, PositionState] = {}
        for symbol, raw in (initial_positions or {}).items():
            if isinstance(raw, Mapping):
                total = float(raw.get("total", raw.get("amount", 0.0)))
                state = PositionState(
                    total=total,
                    available=float(raw.get("available", total)),
                    frozen=float(raw.get("frozen", 0.0)),
                    unsettled_buy=float(raw.get("unsettled_buy", 0.0)),
                )
            else:
                total = float(raw)
                state = PositionState(total=total, available=total)
            state.validate()
            if state.total > _EPSILON:
                self._positions[str(symbol)] = state

    def state(self, symbol: str) -> PositionState:
        current = self._positions.get(str(symbol))
        if current is None:
            return PositionState()
        return PositionState(**asdict(current))

    def sellable(self, symbol: str) -> float:
        current = self._positions.get(str(symbol))
        return current.sellable if current is not None else 0.0

    def clip_sell(self, symbol: str, requested: float) -> float:
        requested = _non_negative(requested, field="requested sell")
        return min(requested, self.sellable(symbol))

    def buy_filled(self, symbol: str, shares: float) -> None:
        shares = _non_negative(shares, field="buy shares")
        if shares <= _EPSILON:
            return
        key = str(symbol)
        current = self._positions.setdefault(key, PositionState())
        current.total += shares
        current.unsettled_buy += shares
        current.validate()

    def reserve_sell(self, symbol: str, shares: float) -> float:
        shares = _non_negative(shares, field="sell reserve shares")
        key = str(symbol)
        current = self._positions.get(key)
        if current is None:
            return 0.0
        reserved = min(shares, current.sellable)
        current.frozen += reserved
        current.validate()
        return reserved

    def release_sell(self, symbol: str, shares: float) -> float:
        shares = _non_negative(shares, field="sell release shares")
        current = self._positions.get(str(symbol))
        if current is None:
            return 0.0
        released = min(shares, current.frozen)
        current.frozen -= released
        current.validate()
        return released

    def sell_filled(self, symbol: str, shares: float) -> None:
        shares = _non_negative(shares, field="sell fill shares")
        if shares <= _EPSILON:
            return
        key = str(symbol)
        current = self._positions.get(key)
        if current is None or shares > current.frozen + _EPSILON:
            raise ValueError("sell fill exceeds reserved shares")
        current.total -= shares
        current.available -= shares
        current.frozen -= shares
        current.validate()
        if current.total <= _EPSILON:
            self._positions.pop(key, None)

    def settle_next_session(self) -> None:
        for current in self._positions.values():
            current.available = min(
                current.total,
                current.available + current.unsettled_buy,
            )
            current.unsettled_buy = 0.0
            current.validate()

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            symbol: asdict(self._positions[symbol])
            for symbol in sorted(self._positions)
        }
