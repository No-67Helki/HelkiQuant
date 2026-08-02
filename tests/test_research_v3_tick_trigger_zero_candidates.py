from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from replay_held_intraday_tick_trigger import run_trigger_replay  # noqa: E402


@pytest.mark.parametrize("headerless", [False, True])
def test_zero_candidate_setting_writes_fail_closed_replay(
    tmp_path: Path, headerless: bool
) -> None:
    trades = tmp_path / "trades.csv"
    daily = tmp_path / "daily.csv"
    account = tmp_path / "account.csv"
    if headerless:
        trades.write_text("", encoding="utf-8")
    else:
        pd.DataFrame(
            columns=[
                "trade_date",
                "instrument",
                "decision_time",
                "threshold",
                "daily_top_n",
            ]
        ).to_csv(trades, index=False)
    pd.DataFrame(
        {
            "trade_date": ["2026-04-08"],
            "threshold": [0.975],
            "daily_top_n": [1],
            "base_nav": [1_000_000.0],
            "fold": [7],
        }
    ).to_csv(daily, index=False)
    pd.DataFrame(
        {
            "trade_date": ["2026-04-08"],
            "nav": [1_000_000.0],
            "cash": [400_000.0],
        }
    ).to_csv(account, index=False)

    report = run_trigger_replay(
        trades,
        daily,
        account,
        tmp_path / "replay.json",
        tmp_path / "replay_trades.csv",
        tmp_path / "replay_daily.csv",
        threshold=0.975,
        daily_top_n=1,
        trade_direction="sell_first",
        trigger_distances=[0.0075],
        max_lots_grid=[1],
        touch_buffers=[0.0, 0.001],
        stage_dir=None,
        window_end_minute=660,
        max_inventory_fraction=0.5,
        max_daily_turnover=0.03,
        lot_size=100,
        buy_cost=0.001,
        sell_cost=0.0025,
        slippage=0.0005,
        min_cost=5.0,
    )

    assert len(report["results"]) == 2
    assert report["best"]["round_trips"] == 0
    assert report["best"]["cum_pnl"] == 0.0
    assert all(row["profit_factor"] is None for row in report["results"])
