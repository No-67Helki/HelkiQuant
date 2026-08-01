from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

from prepare_outer_direct_loss5_daily_release import (  # noqa: E402
    PROFILE,
    derive_forward_training_segments,
    derive_middle_rebalance_schedule,
    prepare_release,
    prepare_group_metadata,
    validate_historical_gate,
    validate_prediction,
    validate_release_clock,
)


def test_inner_shadow_skip_and_require_flags_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_release(
            argparse.Namespace(
                skip_inner_shadow=True,
                require_inner_shadow=True,
            )
        )


def test_live_release_clock_rejects_historical_date_without_smoke_flag() -> None:
    with pytest.raises(ValueError, match="target must be today"):
        validate_release_clock(
            signal_date="2026-06-05",
            as_of_date="2026-06-08",
            st_risk_refreshed_at="2026-06-08",
            historical_smoke=False,
            today=pd.Timestamp("2026-07-15"),
        )

    validate_release_clock(
        signal_date="2026-06-05",
        as_of_date="2026-06-08",
        st_risk_refreshed_at="2026-06-05",
        historical_smoke=True,
        today=pd.Timestamp("2026-07-15"),
    )


def test_live_release_requires_today_st_snapshot() -> None:
    with pytest.raises(ValueError, match="rebuild the static ST"):
        validate_release_clock(
            signal_date="2026-07-14",
            as_of_date="2026-07-15",
            st_risk_refreshed_at="2026-07-14",
            historical_smoke=False,
            today=pd.Timestamp("2026-07-15"),
        )


def test_forward_training_segments_are_calendar_derived_and_purged(tmp_path: Path) -> None:
    provider = tmp_path / "provider"
    calendar_dir = provider / "calendars"
    calendar_dir.mkdir(parents=True)
    calendar = pd.bdate_range("2024-01-02", periods=320)
    (calendar_dir / "day.txt").write_text(
        "\n".join(value.strftime("%Y-%m-%d") for value in calendar),
        encoding="utf-8",
    )

    result = derive_forward_training_segments(
        provider,
        calendar[-1].strftime("%Y-%m-%d"),
    )

    assert result["valid_days"] == 120
    assert result["purge_days"] == 21
    assert result["embargo_days"] == 5
    assert result["valid_end"] == calendar[-28].strftime("%Y-%m-%d")
    assert result["valid_start"] == calendar[-147].strftime("%Y-%m-%d")
    assert result["train_end"] == calendar[-169].strftime("%Y-%m-%d")


def test_middle_rebalance_schedule_uses_provider_trading_sessions(
    tmp_path: Path,
) -> None:
    provider = tmp_path / "provider"
    calendar_dir = provider / "calendars"
    calendar_dir.mkdir(parents=True)
    calendar = pd.bdate_range("2026-01-05", periods=40)
    (calendar_dir / "day.txt").write_text(
        "\n".join(value.strftime("%Y-%m-%d") for value in calendar),
        encoding="utf-8",
    )
    previous = tmp_path / "previous.csv"
    pd.DataFrame(
        {
            "signal_date": calendar[5].strftime("%Y-%m-%d"),
            "symbol": ["SZSE.300001"],
            "target_shares": [100],
        }
    ).to_csv(previous, index=False)

    carry = derive_middle_rebalance_schedule(
        provider,
        calendar[24].strftime("%Y-%m-%d"),
        previous_target=previous,
        initial_launch=False,
        rebalance_every=20,
    )
    due = derive_middle_rebalance_schedule(
        provider,
        calendar[25].strftime("%Y-%m-%d"),
        previous_target=previous,
        initial_launch=False,
        rebalance_every=20,
    )

    assert carry["due"] is False
    assert carry["trading_sessions_since_rebalance"] == 19
    assert carry["legacy_target_bootstrap"] is True
    assert due["due"] is True
    assert due["last_rebalance_signal_date"] == calendar[25].strftime("%Y-%m-%d")
    assert due["trading_sessions_since_rebalance"] == 0


def test_group_metadata_extends_only_latest_interval_for_risk_control(tmp_path: Path) -> None:
    rows = [
        {
            "instrument": f"SZ{index:06d}",
            "industry": "industry",
            "start_date": "2025-01-02",
            "end_date": "2026-06-05",
            "source": "snapshot",
        }
        for index in range(1, 1001)
    ]
    rows.append(
        {
            "instrument": "SZ000001",
            "industry": "old-industry",
            "start_date": "2024-01-02",
            "end_date": "2024-12-31",
            "source": "snapshot",
        }
    )
    source = tmp_path / "industry.csv"
    output = tmp_path / "industry_extended.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    report = prepare_group_metadata(source, output, "2026-07-14")
    result = pd.read_csv(output, dtype=str)

    assert report["active_instruments"] == 1000
    assert report["extended_latest_intervals"] == 1000
    latest = result[
        (result["instrument"] == "SZ000001")
        & (result["start_date"] == "2025-01-02")
    ].iloc[0]
    old = result[
        (result["instrument"] == "SZ000001")
        & (result["start_date"] == "2024-01-02")
    ].iloc[0]
    assert latest["end_date"] == "2026-07-14"
    assert old["end_date"] == "2024-12-31"
    assert "risk_control_ffill" in latest["source"]


def test_prediction_requires_model_config_provider_and_exact_signal_date(
    tmp_path: Path,
) -> None:
    provider = tmp_path / "provider"
    provider.mkdir()
    prediction = tmp_path / "fold_99.csv"
    pd.DataFrame(
        {
            "datetime": "2026-07-14",
            "instrument": [f"SZ{index:06d}" for index in range(1, 21)],
            "middle": [0.1 + index * 0.001 for index in range(20)],
        }
    ).to_csv(prediction, index=False)
    config = tmp_path / "middle.yaml"
    config.write_text(
        "\n".join(
            [
                "qlib_init:",
                "  provider_uri:",
                f"    day: '{provider}'",
                "middle_model:",
                "  model:",
                "    class: CatBoostDEnsemble",
            ]
        ),
        encoding="utf-8",
    )
    model = tmp_path / "fold_99_model.pkl"
    model.write_bytes(b"model")
    prediction.with_suffix(".json").write_text(
        json.dumps(
            {
                "layer": "middle",
                "config": str(config),
                "prediction_start": "2026-07-14 00:00:00",
                "prediction_end": "2026-07-14 00:00:00",
            }
        ),
        encoding="utf-8",
    )

    report = validate_prediction(
        prediction,
        layer="middle",
        signal_date="2026-07-14",
        provider=provider,
    )

    assert report["symbols"] == 20
    assert report["model_class"] == "CatBoostDEnsemble"
    with pytest.raises(ValueError, match="frozen purged protocol"):
        validate_prediction(
            prediction,
            layer="middle",
            signal_date="2026-07-14",
            provider=provider,
            expected_segments={
                "train_end": "2025-09-18",
                "valid_start": "2025-10-28",
                "valid_end": "2026-04-24",
                "test_date": "2026-07-14",
            },
        )


def test_historical_gate_profile_is_exact(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    payload = {
        "passed": True,
        "failed_checks": [],
        "profile": PROFILE,
        "checks": [{"name": "example", "passed": True}],
        "local_audit": {"total_return": 0.1, "max_drawdown": 0.02},
        "gm": {"total_return": 0.09, "rejected_orders": 0},
    }
    gate.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_historical_gate(gate)["passed"] is True

    payload["profile"] = "wrong-profile"
    gate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="profile mismatch"):
        validate_historical_gate(gate)
