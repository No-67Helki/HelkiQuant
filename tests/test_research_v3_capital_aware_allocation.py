from __future__ import annotations

import sys
from pathlib import Path

import pytest


RESEARCH_V3 = Path(__file__).resolve().parents[1] / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH_V3))

from capital_aware_allocation import allocate_equal_weight_lots


def test_fixed_topk_reproduces_drop_behavior() -> None:
    result = allocate_equal_weight_lots(
        ["SZ000001", "SZ000002", "SH688001"],
        {"SZ000001": 10.0, "SZ000002": 80.0, "SH688001": 30.0},
        {"SZ000001": 100, "SZ000002": 100, "SH688001": 200},
        100_000.0,
        0.30,
        denominator_count=3,
        mode="fixed_topk",
    )

    assert result["shares"] == {"SZ000001": 1000, "SZ000002": 100, "SH688001": 300}
    assert result["diagnostics"]["allocated_notional"] == pytest.approx(27_000.0)
    assert result["diagnostics"]["budget_utilization"] == pytest.approx(0.90)


def test_capital_aware_uses_budget_without_exceeding_it() -> None:
    instruments = [f"SZ{i:06d}" for i in range(1, 11)]
    prices = {instrument: price for instrument, price in zip(instruments, range(5, 15))}
    min_lots = {instrument: 100 for instrument in instruments}

    result = allocate_equal_weight_lots(
        instruments,
        prices,
        min_lots,
        100_000.0,
        0.60,
        denominator_count=10,
        mode="capital_aware",
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["allocated_notional"] <= diagnostics["budget_value"]
    assert diagnostics["budget_utilization"] >= 0.90
    assert diagnostics["allocated_count"] == 10
    assert all(volume % 100 == 0 for volume in result["shares"].values())


def test_capital_aware_respects_minimum_initial_lot() -> None:
    result = allocate_equal_weight_lots(
        ["SH688001", "SZ000001"],
        {"SH688001": 50.0, "SZ000001": 10.0},
        {"SH688001": 200, "SZ000001": 100},
        20_000.0,
        0.50,
        denominator_count=2,
        mode="capital_aware",
    )

    assert result["shares"]["SH688001"] == 200
    assert result["shares"]["SH688001"] >= 200
    assert result["diagnostics"]["allocated_notional"] <= 10_000.0


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported allocation mode"):
        allocate_equal_weight_lots(
            ["SZ000001"],
            {"SZ000001": 10.0},
            {"SZ000001": 100},
            100_000.0,
            0.50,
            mode="unknown",
        )
