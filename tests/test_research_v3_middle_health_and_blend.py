from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


RESEARCH_V3 = Path(__file__).resolve().parents[1] / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH_V3))

from build_middle_alpha_health_outer import build_label_calendar, health_snapshot
from build_middle_rank_blends import build as build_rank_blends
from evaluate_expanded_c_controls import load_forbidden_instruments
from export_c_baseline_production_logs import stale_sell_block_reason


def test_label_calendar_uses_only_completed_market_sessions() -> None:
    market_dates = pd.DatetimeIndex(
        pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    )
    signal_dates = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05"]))

    result = build_label_calendar(market_dates, signal_dates, horizon=2)

    assert result.to_dict(orient="records") == [
        {
            "datetime": pd.Timestamp("2026-01-02"),
            "entry_date": pd.Timestamp("2026-01-05"),
            "available_date": pd.Timestamp("2026-01-07"),
        }
    ]


def test_health_snapshot_excludes_not_yet_available_ic() -> None:
    daily_ic = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "available_date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-10"]),
            "ic": [-0.10, -0.20, 1.00],
        }
    )

    known, health, trigger = health_snapshot(
        daily_ic,
        pd.Timestamp("2026-01-06"),
        rolling_window=2,
        min_observations=2,
        health_threshold=0.0,
    )

    assert known["signal_date"].tolist() == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
    ]
    assert health == pytest.approx(-0.15)
    assert trigger is True


def test_finite_stale_sell_block_starts_on_rejection_and_expires() -> None:
    assert stale_sell_block_reason(None, "SELL", 9, [10], False, 3) is None
    assert "3 trading days" in stale_sell_block_reason(None, "SELL", 10, [10], False, 3)
    assert "3 trading days" in stale_sell_block_reason(None, "SELL", 12, [10], False, 3)
    assert stale_sell_block_reason(None, "SELL", 13, [10], False, 3) is None
    assert stale_sell_block_reason(None, "BUY", 11, [10], True, 0) is None
    assert stale_sell_block_reason(None, "SELL", 9, [10], True, 0) is None
    assert "persistent" in stale_sell_block_reason(None, "SELL", 100, [10], True, 0)
    assert stale_sell_block_reason("exchange rejection", "SELL", 1, [10], False, 0) == (
        "exchange rejection"
    )


def test_rank_blend_uses_daily_percentile_ranks(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    pd.DataFrame(
        {
            "datetime": ["2026-01-02", "2026-01-02"],
            "instrument": ["SZ000001", "SZ000002"],
            "middle": [1.0, 2.0],
        }
    ).to_csv(baseline, index=False)
    pd.DataFrame(
        {
            "datetime": ["2026-01-02", "2026-01-02"],
            "instrument": ["SZ000001", "SZ000002"],
            "middle": [2.0, 1.0],
        }
    ).to_csv(candidate, index=False)

    report = build_rank_blends(baseline, candidate, tmp_path / "out", [0.25])
    output = pd.read_csv(report["outputs"][0]["path"])

    assert output["middle"].tolist() == pytest.approx([0.625, 0.875])
    assert report["daily_rank_correlation_mean"] == pytest.approx(-1.0)


def test_forbidden_loader_normalizes_local_and_gm_symbols(tmp_path: Path) -> None:
    path = tmp_path / "forbidden.csv"
    pd.DataFrame(
        {
            "instrument": ["SZ000001", ""],
            "gm_symbol": ["", "SHSE.600000"],
        }
    ).to_csv(path, index=False)

    assert load_forbidden_instruments(path) == {"SZ000001", "SH600000"}
