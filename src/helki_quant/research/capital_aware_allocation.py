from __future__ import annotations

from collections.abc import Mapping, Sequence

import math


ALLOCATION_MODES = {"fixed_topk", "capital_aware"}


def _floor_lot(raw_shares: float, lot_size: int, min_lot: int) -> int:
    if not math.isfinite(raw_shares) or raw_shares <= 0:
        return 0
    shares = int(raw_shares // lot_size) * lot_size
    return shares if shares >= min_lot else 0


def allocate_equal_weight_lots(
    instruments: Sequence[str],
    prices: Mapping[str, float],
    min_lots: Mapping[str, int],
    capital: float,
    risk_budget: float,
    *,
    lot_size: int = 100,
    denominator_count: int | None = None,
    mode: str = "fixed_topk",
) -> dict:
    """Allocate ranked instruments into deterministic whole-lot targets.

    ``fixed_topk`` reproduces the legacy behavior: each ranked name receives
    ``risk_budget / denominator_count`` and unaffordable names are dropped.
    ``capital_aware`` finds the widest affordable equal-weight subset, then
    uses at most one residual lot per name to reduce cash drag.
    """

    if mode not in ALLOCATION_MODES:
        raise ValueError(f"unsupported allocation mode: {mode}")
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be positive")
    if not math.isfinite(risk_budget) or not 0 < risk_budget <= 1:
        raise ValueError("risk_budget must be in (0, 1]")
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")

    ranked: list[str] = []
    seen: set[str] = set()
    for raw in instruments:
        instrument = str(raw).upper()
        if instrument and instrument not in seen:
            ranked.append(instrument)
            seen.add(instrument)

    valid: list[str] = []
    clean_prices: dict[str, float] = {}
    clean_min_lots: dict[str, int] = {}
    for instrument in ranked:
        try:
            price = float(prices.get(instrument, 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price) or price <= 0:
            continue
        min_lot = max(lot_size, int(min_lots.get(instrument, lot_size)))
        valid.append(instrument)
        clean_prices[instrument] = price
        clean_min_lots[instrument] = min_lot

    budget_value = float(capital * risk_budget)
    denominator = int(denominator_count or len(ranked))
    if denominator <= 0 or not valid:
        return {
            "shares": {},
            "notional": {},
            "weights": {},
            "diagnostics": {
                "mode": mode,
                "requested_count": len(ranked),
                "valid_price_count": len(valid),
                "selected_count": 0,
                "allocated_count": 0,
                "budget_value": budget_value,
                "allocated_notional": 0.0,
                "effective_weight": 0.0,
                "budget_utilization": 0.0,
                "ideal_value_per_name": 0.0,
                "residual_value": budget_value,
                "topup_lots": 0,
            },
        }

    if mode == "fixed_topk":
        selected = valid
        ideal_value = budget_value / denominator
    else:
        selected = []
        ideal_value = 0.0
        for count in range(min(len(valid), denominator), 0, -1):
            per_name = budget_value / count
            affordable = [
                instrument
                for instrument in valid
                if clean_prices[instrument] * clean_min_lots[instrument] <= per_name + 1e-9
            ]
            if len(affordable) >= count:
                selected = affordable[:count]
                ideal_value = per_name
                break

    shares: dict[str, int] = {}
    for instrument in selected:
        volume = _floor_lot(
            ideal_value / clean_prices[instrument],
            lot_size,
            clean_min_lots[instrument],
        )
        if volume > 0:
            shares[instrument] = volume

    topup_lots = 0
    if mode == "capital_aware" and shares:
        allocated = sum(shares[inst] * clean_prices[inst] for inst in shares)
        residual = max(0.0, budget_value - allocated)
        topups = []
        for rank, instrument in enumerate(selected):
            if instrument not in shares:
                continue
            lot_value = lot_size * clean_prices[instrument]
            current = shares[instrument] * clean_prices[instrument]
            penalty = abs(current + lot_value - ideal_value) - abs(current - ideal_value)
            topups.append((penalty, rank, lot_value, instrument))
        for _, _, lot_value, instrument in sorted(topups):
            if lot_value <= residual + 1e-9:
                shares[instrument] += lot_size
                residual -= lot_value
                topup_lots += 1

    notionals = {
        instrument: float(volume * clean_prices[instrument])
        for instrument, volume in shares.items()
    }
    weights = {
        instrument: float(value / capital)
        for instrument, value in notionals.items()
    }
    allocated_notional = float(sum(notionals.values()))
    allocated_weights = list(weights.values())
    diagnostics = {
        "mode": mode,
        "requested_count": len(ranked),
        "valid_price_count": len(valid),
        "selected_count": len(selected),
        "allocated_count": len(shares),
        "dropped_invalid_price_count": len(ranked) - len(valid),
        "dropped_unaffordable_count": len(valid) - len(shares),
        "denominator_count": denominator,
        "budget_value": budget_value,
        "allocated_notional": allocated_notional,
        "effective_weight": allocated_notional / capital,
        "budget_utilization": allocated_notional / budget_value,
        "ideal_value_per_name": ideal_value,
        "residual_value": budget_value - allocated_notional,
        "topup_lots": topup_lots,
        "min_name_weight": min(allocated_weights) if allocated_weights else 0.0,
        "max_name_weight": max(allocated_weights) if allocated_weights else 0.0,
    }
    return {
        "shares": shares,
        "notional": notionals,
        "weights": weights,
        "diagnostics": diagnostics,
    }
