from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

from paper_activation_registry import summarize_paper_session  # noqa: E402


def _event(
    order_id: int,
    *,
    status: int,
    status_name: str,
    symbol: str,
    side: int,
    detail: str = "",
) -> dict:
    return {
        "cl_ord_id": order_id,
        "status": status,
        "status_name": status_name,
        "symbol": symbol,
        "side": side,
        "ord_rej_reason_detail": detail,
    }


def test_order_callbacks_are_deduplicated_by_order_id() -> None:
    result = summarize_paper_session(
        order_events=[
            _event(
                1,
                status=10,
                status_name="PendingNew",
                symbol="SZSE.300001",
                side=1,
            ),
            _event(
                1,
                status=1,
                status_name="New",
                symbol="SZSE.300001",
                side=1,
            ),
            _event(
                1,
                status=3,
                status_name="Filled",
                symbol="SZSE.300001",
                side=1,
            ),
        ],
        target_volumes={"SZSE.300001": 100},
        actual_volumes={"SZSE.300001": 100},
    )
    assert result["terminal_orders"] == 1
    assert result["filled_orders"] == 1
    assert result["unexplained_target_volume_abs_diff"] == 0


def test_market_restriction_sell_explains_stale_holding() -> None:
    result = summarize_paper_session(
        order_events=[
            _event(
                2,
                status=8,
                status_name="Rejected",
                symbol="SZSE.300002",
                side=2,
                detail="标的跌停，委托价格低于跌停价格",
            )
        ],
        target_volumes={"SZSE.300002": 0},
        actual_volumes={"SZSE.300002": 300},
    )
    assert result["market_restriction_rejected_orders"] == 1
    assert result["unexpected_rejected_orders"] == 0
    assert result["unresolved_rejected_sell_symbol_list"] == [
        "SZSE.300002"
    ]
    assert result["unexplained_target_volume_abs_diff"] == 0


def test_sell_first_deferred_buy_explains_underweight() -> None:
    result = summarize_paper_session(
        order_events=[],
        target_volumes={"SZSE.300003": 500},
        actual_volumes={"SZSE.300003": 100},
        deferred_buy_symbols={"SZSE.300003"},
    )
    assert result["deferred_buy_symbols"] == ["SZSE.300003"]
    assert result["target_volume_abs_diff"] == 400
    assert result["unexplained_target_volume_abs_diff"] == 0


def test_unexpected_rejection_and_unexplained_position_gap_remain_visible() -> None:
    result = summarize_paper_session(
        order_events=[
            _event(
                3,
                status=8,
                status_name="Rejected",
                symbol="SZSE.300004",
                side=1,
                detail="委托量不正确",
            )
        ],
        target_volumes={"SZSE.300004": 200},
        actual_volumes={},
    )
    assert result["unexpected_rejected_orders"] == 1
    assert result["unexplained_target_mismatch_symbols"] == 1
    assert result["unexplained_target_volume_abs_diff"] == 200
