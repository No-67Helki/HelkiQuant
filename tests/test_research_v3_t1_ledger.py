from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

from execution.t1_ledger import T1PositionLedger  # noqa: E402


def test_buy_today_is_not_sellable_until_next_session() -> None:
    ledger = T1PositionLedger({"SZSE.300001": 300})

    ledger.buy_filled("SZSE.300001", 200)

    state = ledger.state("SZSE.300001")
    assert state.total == 500
    assert state.available == 300
    assert state.unsettled_buy == 200
    assert ledger.clip_sell("SZSE.300001", 500) == 300

    ledger.settle_next_session()

    assert ledger.sellable("SZSE.300001") == 500
    assert ledger.state("SZSE.300001").unsettled_buy == 0


def test_sell_reservation_cancel_and_fill_preserve_available_state() -> None:
    ledger = T1PositionLedger({"SHSE.600000": 500})

    assert ledger.reserve_sell("SHSE.600000", 400) == 400
    assert ledger.sellable("SHSE.600000") == 100
    assert ledger.release_sell("SHSE.600000", 150) == 150
    assert ledger.sellable("SHSE.600000") == 250

    ledger.sell_filled("SHSE.600000", 250)

    state = ledger.state("SHSE.600000")
    assert state.total == 250
    assert state.available == 250
    assert state.frozen == 0


def test_sell_fill_must_be_reserved_and_snapshot_is_deterministic() -> None:
    ledger = T1PositionLedger({"SZSE.000002": 100, "SHSE.600001": 200})

    with pytest.raises(ValueError, match="exceeds reserved"):
        ledger.sell_filled("SZSE.000002", 100)

    assert list(ledger.snapshot()) == ["SHSE.600001", "SZSE.000002"]


def test_invalid_or_inconsistent_initial_position_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed total"):
        T1PositionLedger(
            {
                "SZSE.000001": {
                    "total": 100,
                    "available": 100,
                    "unsettled_buy": 100,
                }
            }
        )
