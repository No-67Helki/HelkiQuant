from __future__ import annotations

import pandas as pd

from helki_quant.research.compare_gm_local_audit import summarize_rejections


def test_rejection_summary_separates_market_blocks_and_resolution() -> None:
    orders = pd.DataFrame(
        [
            {
                "event_date": "2026-01-05",
                "symbol": "SZSE.300001",
                "side": 2,
                "status_name": "Rejected",
                "ord_rej_reason_detail": "标的停牌",
            },
            {
                "event_date": "2026-01-06",
                "symbol": "SZSE.300001",
                "side": 2,
                "status_name": "Filled",
                "ord_rej_reason_detail": "",
            },
            {
                "event_date": "2026-01-05",
                "symbol": "SZSE.300002",
                "side": 1,
                "status_name": "Rejected",
                "ord_rej_reason_detail": "连接失败",
            },
        ]
    )
    summary = summarize_rejections(orders)
    assert summary["market_restriction_rejected_orders"] == 1
    assert summary["unexpected_rejected_orders"] == 1
    assert summary["unique_rejected_symbol_sides"] == 2
    assert summary["unresolved_rejected_sell_symbols"] == 0
