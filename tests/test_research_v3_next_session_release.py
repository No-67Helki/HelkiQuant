from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

from gm_export_paper_account_snapshot import capture_snapshot  # noqa: E402
from build_forbidden_st_symbols import build as build_forbidden_symbols  # noqa: E402
from prepare_outer_direct_loss5_next_session_release import (  # noqa: E402
    build_release_plan,
)
from paper_activation_registry import (  # noqa: E402
    ACTIVATION_SCHEMA_VERSION,
    EVENT_FINALIZED,
    EVENT_READY,
    EVENT_STARTED,
    append_activation_event,
    sha256_file,
)


ACCOUNT_ID = "paper-account"


def _provider(root: Path, name: str, end_date: str = "2026-06-05") -> Path:
    provider = root / name
    calendar = provider / "calendars" / "day.txt"
    calendar.parent.mkdir(parents=True, exist_ok=True)
    calendar.write_text(
        f"2026-06-03\n2026-06-04\n{end_date}\n",
        encoding="utf-8",
    )
    return provider


def _file(root: Path, name: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture\n", encoding="utf-8")
    return path


def _inputs(
    root: Path,
    *,
    with_calendar: bool = True,
    with_stock_universe: bool = True,
) -> dict:
    raw = root / "raw"
    raw.mkdir()
    base_candidate = root / "candidate"
    base_candidate.mkdir()
    previous = root / "previous.csv"
    pd.DataFrame(
        {
            "trade_date": ["2026-06-04"],
            "signal_date": ["2026-06-03"],
            "instrument": ["SZ000001"],
            "target_shares": [1000],
        }
    ).to_csv(previous, index=False)
    snapshot = capture_snapshot(
        positions=[
            {
                "account_id": ACCOUNT_ID,
                "symbol": "SZSE.000001",
                "side": 1,
                "volume": 1000,
                "available": 1000,
                "last_price": 10.0,
            }
        ],
        cash={
            "account_id": ACCOUNT_ID,
            "nav": 1_000_000.0,
            "available": 990_000.0,
            "market_value": 10_000.0,
        },
        account_id=ACCOUNT_ID,
        captured_at="2026-06-05 15:10:00",
        output_root=root / "snapshots",
        snapshot_id="snapshot_1",
        next_trading_date="2026-06-08" if with_calendar else None,
        instruments=(
            [
                {
                    "symbol": "SZSE.000001",
                    "sec_name": "TEST STOCK",
                    "is_suspended": 0,
                    "delisted_date": None,
                },
                {
                    "symbol": "SZSE.000002",
                    "sec_name": "*ST TEST",
                    "is_suspended": 0,
                    "delisted_date": None,
                },
            ]
            if with_stock_universe
            else None
        ),
        minimum_stock_universe=1,
    )
    return {
        "middle_provider": _provider(root, "middle"),
        "outer_provider": _provider(root, "outer"),
        "raw_daily_dir": raw,
        "group_metadata": _file(root, "group.csv"),
        "forbidden_overrides": _file(root, "overrides.csv"),
        "historical_gate": _file(root, "historical_gate.json"),
        "previous_target": previous,
        "account_snapshot": Path(str(snapshot["snapshot_file"])),
        "stage_root": root / "releases",
        "expected_account_id": ACCOUNT_ID,
        "now": pd.Timestamp("2026-06-05 15:30:00"),
        "base_candidate": base_candidate,
        "middle_base_config": _file(root, "middle.yaml"),
        "middle_whitelist": _file(root, "whitelist.json"),
        "outer_base_config": _file(root, "outer.yaml"),
        "minimum_stock_universe": 1,
    }


def test_next_session_plan_derives_dates_and_frozen_training_command(
    tmp_path: Path,
) -> None:
    plan = build_release_plan(**_inputs(tmp_path))

    assert plan["signal_date"] == "2026-06-05"
    assert plan["trade_date"] == "2026-06-08"
    assert plan["release_window"]["mode"] == "signal_day_after_close"
    assert plan["account_snapshot"]["freshness_policy"] == "signal_or_trade_date"
    assert plan["account_snapshot"]["age_days"] == 3
    assert plan["stock_universe"]["source"] == "gm.api.get_instruments"
    assert plan["stock_universe"]["rows"] == 2
    assert plan["paper_orders_allowed"] is False
    assert plan["execution_started"] is False
    assert "--train-forward-models" in plan["command"]
    assert "--require-inner-shadow" in plan["command"]
    assert "--middle-prediction" not in plan["command"]
    assert plan["command"][plan["command"].index("--signal-date") + 1] == "2026-06-05"
    assert plan["command"][plan["command"].index("--trade-date") + 1] == "2026-06-08"


def test_next_session_plan_rejects_mismatched_provider_calendars(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    _provider(tmp_path, "outer", end_date="2026-06-04")

    with pytest.raises(ValueError, match="end on different dates"):
        build_release_plan(**inputs)


def test_next_session_plan_requires_gm_calendar_evidence(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, with_calendar=False)

    with pytest.raises(ValueError, match="lacks GmQuant trading-calendar evidence"):
        build_release_plan(**inputs)


def test_next_session_plan_requires_gm_stock_universe_evidence(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path, with_stock_universe=False)

    with pytest.raises(ValueError, match="lacks a hash-protected GmQuant stock"):
        build_release_plan(**inputs)


def test_gm_snapshot_stock_list_feeds_static_st_filter(tmp_path: Path) -> None:
    plan = build_release_plan(**_inputs(tmp_path))
    output = tmp_path / "forbidden.csv"
    report = build_forbidden_symbols(
        Path(plan["stock_universe"]["path"]),
        output,
        tmp_path / "forbidden_report.json",
        overrides_path=None,
    )

    forbidden = pd.read_csv(output)
    assert report["rows_forbidden"] == 1
    assert forbidden["instrument"].tolist() == ["SZ000002"]
    assert forbidden["reason"].tolist() == ["name_contains_ST"]


def test_next_session_plan_fails_closed_before_signal_day_close(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs["now"] = pd.Timestamp("2026-06-05 14:59:00")

    with pytest.raises(ValueError, match="outside the fail-closed execution window"):
        build_release_plan(**inputs)


def test_next_session_plan_resolves_previous_target_from_activation_registry(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    previous = Path(inputs.pop("previous_target"))
    registry = tmp_path / "PAPER_ACTIVATION_REGISTRY.jsonl"
    identity = {
        "activation_schema_version": ACTIVATION_SCHEMA_VERSION,
        "activation_id": "A" * 64,
        "run_id": "paper_run_001",
        "strategy_id": "outer-middle-paper",
        "account_id": ACCOUNT_ID,
        "run_mode": "LIVE",
        "trading_env": "PAPER",
        "target_path": str(previous),
        "target_sha256": sha256_file(previous),
        "signal_date": "2026-06-03",
        "trade_date": "2026-06-04",
    }
    for event, timestamp in (
        (EVENT_STARTED, "2026-06-04 09:20:00"),
        (EVENT_READY, "2026-06-04 09:21:00"),
        (EVENT_FINALIZED, "2026-06-04 15:05:00"),
    ):
        append_activation_event(
            registry,
            event=event,
            identity=identity,
            timestamp=timestamp,
        )
    inputs["previous_target"] = None
    inputs["activation_registry"] = registry

    plan = build_release_plan(**inputs)

    assert plan["previous_target"]["source"] == "finalized_paper_activation"
    assert plan["previous_target"]["activation"]["run_id"] == "paper_run_001"
    assert plan["previous_target"]["sha256"] == sha256_file(previous)
    assert plan["activation_chain_seed"]["path"] == str(registry.resolve())
    assert plan["activation_chain_seed"]["sha256"] == sha256_file(registry)
    assert "--activation-registry-seed" in plan["command"]
    seed_index = plan["command"].index("--activation-registry-seed")
    assert plan["command"][seed_index + 1] == str(registry.resolve())
