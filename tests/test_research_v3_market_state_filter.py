from __future__ import annotations

from pathlib import Path

import pandas as pd

from helki_quant.research.filter_gm_targets_market_state import filter_targets


def _write_daily(root: Path, code: str, pct_changes: list[float]) -> None:
    pd.DataFrame(
        {
            "日期": ["2025-01-02", "2025-01-03"],
            "成交量": [1000, 1000],
            "涨跌幅": pct_changes,
        }
    ).to_csv(root / f"{code}_daily_qfq.csv", index=False)


def test_runtime_retry_keeps_blocked_sell_as_lower_target(tmp_path: Path) -> None:
    daily_root = tmp_path / "daily"
    daily_root.mkdir()
    _write_daily(daily_root, "000001", [0.0, -10.0])
    _write_daily(daily_root, "000002", [0.0, 0.0])

    target_csv = tmp_path / "targets.csv"
    pd.DataFrame(
        [
            {
                "trade_date": "2025-01-02",
                "symbol": "SZSE.000001",
                "target_shares": 100,
                "target_weight": 0.1,
                "rank": 1,
            },
            {
                "trade_date": "2025-01-02",
                "symbol": "SZSE.000002",
                "target_shares": 100,
                "target_weight": 0.1,
                "rank": 2,
            },
            {
                "trade_date": "2025-01-03",
                "symbol": "SZSE.000002",
                "target_shares": 100,
                "target_weight": 0.1,
                "rank": 1,
            },
        ]
    ).to_csv(target_csv, index=False)

    hold_output = tmp_path / "hold.csv"
    hold_result = filter_targets(
        target_csv,
        hold_output,
        daily_root,
        None,
        9.5,
        runtime_retry_blocked_sells=False,
    )
    held = pd.read_csv(hold_output)
    assert (
        (held["trade_date"] == "2025-01-03")
        & (held["symbol"] == "SZSE.000001")
        & (held["target_shares"] == 100)
    ).any()
    assert hold_result["blocked_sell_actions"] == 1

    retry_output = tmp_path / "retry.csv"
    retry_result = filter_targets(
        target_csv,
        retry_output,
        daily_root,
        None,
        9.5,
        runtime_retry_blocked_sells=True,
    )
    retry = pd.read_csv(retry_output)
    assert not (
        (retry["trade_date"] == "2025-01-03")
        & (retry["symbol"] == "SZSE.000001")
    ).any()
    assert retry_result["blocked_sell_actions"] == 0
    assert retry_result["runtime_retry_sell_actions"] == 1
