from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

from audit_target_transition import audit_target_transition  # noqa: E402
from build_inner_multidecision_shadow_candidate import (  # noqa: E402
    ORDER_CALL_NAMES,
    called_function_names,
)
from gm_export_paper_account_snapshot import capture_snapshot  # noqa: E402


def _write_raw(raw: Path, code: str, price: float, date: str = "2026-06-05") -> None:
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "\u65e5\u671f": [date],
            "\u5f00\u76d8": [price],
            "\u6536\u76d8": [price],
            "\u6700\u9ad8": [price],
            "\u6700\u4f4e": [price],
            "\u6210\u4ea4\u91cf": [1_000_000],
            "\u6210\u4ea4\u989d": [price * 1_000_000],
        }
    ).to_csv(raw / f"{code}_daily_qfq.csv", index=False)


def _write_target(
    path: Path,
    rows: list[tuple[str, int]],
    *,
    trade_date: str = "2026-06-08",
    signal_date: str = "2026-06-05",
) -> None:
    pd.DataFrame(
        {
            "trade_date": trade_date,
            "signal_date": signal_date,
            "instrument": [instrument for instrument, _ in rows],
            "target_shares": [shares for _, shares in rows],
        }
    ).to_csv(path, index=False)


def test_buffered_transition_audits_sell_first_cost_and_cash(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_raw(raw, "000001", 10.0)
    _write_raw(raw, "000002", 20.0)
    _write_raw(raw, "000003", 15.0)
    previous = tmp_path / "previous.csv"
    next_target = tmp_path / "next.csv"
    _write_target(previous, [("SZ000001", 1000), ("SZ000002", 1000)])
    _write_target(next_target, [("SZ000001", 500), ("SZ000003", 1000)])

    report = audit_target_transition(
        next_target=next_target,
        previous_target=previous,
        initial_launch=False,
        raw_daily_dir=raw,
        output_path=tmp_path / "transition.json",
    )

    assert report["passed"] is True
    assert report["mode"] == "buffered_previous_target"
    assert report["counts"] == {
        "previous_names": 2,
        "previous_target_names": 2,
        "starting_position_names": 2,
        "next_names": 2,
        "retained_names": 1,
        "retained_changed_shares": 1,
        "full_exits": 1,
        "new_entries": 1,
        "sell_orders": 2,
        "buy_orders": 1,
        "actionable_orders": 3,
    }
    assert report["metrics"]["two_way_turnover"] == 0.04
    assert report["metrics"]["min_cash"] > 0
    assert report["metrics"]["estimated_cost"] > 0
    assert (tmp_path / "transition.orders.csv").exists()


def test_transition_fails_closed_when_turnover_is_excessive(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_raw(raw, "000001", 10.0)
    _write_raw(raw, "000002", 10.0)
    previous = tmp_path / "previous.csv"
    next_target = tmp_path / "next.csv"
    _write_target(previous, [("SZ000001", 50_000)])
    _write_target(next_target, [("SZ000002", 40_000)])

    report = audit_target_transition(
        next_target=next_target,
        previous_target=previous,
        initial_launch=False,
        raw_daily_dir=raw,
        output_path=tmp_path / "transition.json",
    )

    assert report["passed"] is False
    assert "two_way_turnover" in report["failed_checks"]
    assert report["metrics"]["min_cash"] > 0


def test_transition_fails_closed_on_missing_signal_date_price(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_raw(raw, "000001", 10.0)
    previous = tmp_path / "previous.csv"
    next_target = tmp_path / "next.csv"
    _write_target(previous, [("SZ000001", 1000)])
    _write_target(next_target, [("SZ000002", 1000)])

    report = audit_target_transition(
        next_target=next_target,
        previous_target=previous,
        initial_launch=False,
        raw_daily_dir=raw,
        output_path=tmp_path / "transition.json",
    )

    assert report["passed"] is False
    assert report["missing_prices"] == ["SZ000002"]
    assert "current_prices_complete" in report["failed_checks"]


def test_initial_launch_uses_separate_build_threshold(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_raw(raw, "000001", 10.0)
    target = tmp_path / "next.csv"
    _write_target(target, [("SZ000001", 30_000)])

    report = audit_target_transition(
        next_target=target,
        previous_target=None,
        initial_launch=True,
        raw_daily_dir=raw,
        output_path=tmp_path / "transition.json",
    )

    assert report["passed"] is True
    assert report["mode"] == "initial_launch"
    assert report["limits"]["max_two_way_turnover"] == 0.65
    assert report["metrics"]["two_way_turnover"] == 0.30


def test_gm_account_snapshot_entrypoints_contain_no_order_calls() -> None:
    for name in (
        "gm_export_paper_account_snapshot.py",
        "gm_capture_outer_middle_paper_account.py",
    ):
        called = called_function_names(RESEARCH / name)
        assert called.isdisjoint(ORDER_CALL_NAMES)


def test_account_snapshot_replaces_target_assumption_and_exposes_drift(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    _write_raw(raw, "000001", 10.0)
    _write_raw(raw, "000002", 20.0)
    _write_raw(raw, "000003", 15.0)
    _write_raw(raw, "000004", 25.0)
    previous = tmp_path / "previous.csv"
    next_target = tmp_path / "next.csv"
    _write_target(previous, [("SZ000001", 1000), ("SZ000002", 1000)])
    _write_target(next_target, [("SZ000001", 500), ("SZ000003", 1000)])
    account_id = "paper-account"
    captured = capture_snapshot(
        positions=[
            {
                "account_id": account_id,
                "symbol": "SZSE.000001",
                "side": 1,
                "volume": 1000,
                "available": 1000,
                "last_price": 10.0,
            },
            {
                "account_id": account_id,
                "symbol": "SZSE.000004",
                "side": 1,
                "volume": 1000,
                "available": 1000,
                "last_price": 25.0,
            },
        ],
        cash={
            "account_id": account_id,
            "nav": 1_000_000.0,
            "available": 965_000.0,
            "market_value": 35_000.0,
        },
        account_id=account_id,
        captured_at="2026-06-08 08:30:00",
        output_root=tmp_path / "snapshots",
        snapshot_id="snapshot_1",
    )

    report = audit_target_transition(
        next_target=next_target,
        previous_target=previous,
        initial_launch=False,
        raw_daily_dir=raw,
        output_path=tmp_path / "transition.json",
        account_snapshot=Path(str(captured["snapshot_file"])),
        expected_account_id=account_id,
        as_of_date="2026-06-08",
    )

    assert report["passed"] is True
    assert report["position_source"] == "account_snapshot"
    assert report["counts"]["starting_position_names"] == 2
    assert report["previous_target_account_drift"] == {
        "extra_account_positions": ["SZ000004"],
        "missing_account_positions": ["SZ000002"],
        "share_mismatch_positions": [],
        "drifted_symbols": 2,
    }
    assert report["counts"]["full_exits"] == 1
    assert report["metrics"]["initial_cash"] == 965_000.0

    with pytest.raises(ValueError, match="account snapshot is not fresh"):
        audit_target_transition(
            next_target=next_target,
            previous_target=previous,
            initial_launch=False,
            raw_daily_dir=raw,
            output_path=tmp_path / "stale_transition.json",
            account_snapshot=Path(str(captured["snapshot_file"])),
            expected_account_id=account_id,
            as_of_date="2026-06-10",
        )


def test_release_boundary_snapshot_accepts_friday_for_monday_target(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    _write_raw(raw, "000001", 10.0, date="2026-06-05")
    previous = tmp_path / "previous.csv"
    next_target = tmp_path / "next.csv"
    _write_target(previous, [("SZ000001", 1000)])
    _write_target(next_target, [("SZ000001", 1000)])
    account_id = "paper-account"
    captured = capture_snapshot(
        positions=[
            {
                "account_id": account_id,
                "symbol": "SZSE.000001",
                "side": 1,
                "volume": 1000,
                "available": 1000,
                "last_price": 10.0,
            }
        ],
        cash={
            "account_id": account_id,
            "nav": 1_000_000.0,
            "available": 990_000.0,
            "market_value": 10_000.0,
        },
        account_id=account_id,
        captured_at="2026-06-05 15:10:00",
        output_root=tmp_path / "snapshots",
        snapshot_id="friday_close",
        next_trading_date="2026-06-08",
    )

    assert captured["passed"] is True
    assert captured["trading_calendar"] == {
        "source": "gm.api.get_next_trading_date",
        "exchange": "SHSE",
        "next_trading_date": "2026-06-08",
        "calendar_day_gap": 3,
    }
    report = audit_target_transition(
        next_target=next_target,
        previous_target=previous,
        initial_launch=False,
        raw_daily_dir=raw,
        output_path=tmp_path / "weekend_transition.json",
        account_snapshot=Path(str(captured["snapshot_file"])),
        expected_account_id=account_id,
        as_of_date="2026-06-08",
        account_snapshot_allowed_dates=("2026-06-05", "2026-06-08"),
    )

    assert report["passed"] is True
    assert report["account_snapshot"]["age_days"] == 3
    assert report["account_snapshot"]["freshness_policy"] == "signal_or_trade_date"
    assert report["account_snapshot"]["allowed_capture_dates"] == [
        "2026-06-05",
        "2026-06-08",
    ]

    with pytest.raises(
        ValueError,
        match="not from an allowed release boundary",
    ):
        audit_target_transition(
            next_target=next_target,
            previous_target=previous,
            initial_launch=False,
            raw_daily_dir=raw,
            output_path=tmp_path / "wrong_boundary.json",
            account_snapshot=Path(str(captured["snapshot_file"])),
            expected_account_id=account_id,
            as_of_date="2026-06-08",
            account_snapshot_allowed_dates=("2026-06-04", "2026-06-08"),
        )
