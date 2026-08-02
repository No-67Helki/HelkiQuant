from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

from paper_activation_registry import (  # noqa: E402
    EVENT_FINALIZED,
    EVENT_READY,
    EVENT_STARTED,
    append_activation_event,
)
from validate_strategy_live_readiness import validate  # noqa: E402


ACCOUNT = "paper-account"
PROFILE = "challenger"
PROFILE_ALIAS = "runtime-challenger"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _artifact(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def _build_inputs(tmp_path: Path, sessions: int = 3) -> dict[str, Path]:
    config = _write_json(
        tmp_path / "config.json",
        {
            "profile_aliases": {PROFILE: PROFILE_ALIAS},
            "paper_observation_rules": {
                "min_finalized_sessions": sessions,
                "max_error_events": 0,
                "min_rebalance_events_per_session": 1,
                "require_one_finalized_session_per_trade_date": True,
                "max_sessions_with_unexpected_rejects": 0,
                "max_sessions_with_unexplained_target_mismatch": 0,
                "max_consecutive_unresolved_sell_sessions": 3,
                "max_latest_unresolved_sell_symbols": 0,
            },
        },
    )
    canonical_dates = pd.bdate_range("2026-04-08", periods=60)
    canonical_manifest = _write_json(
        tmp_path / "canonical_manifest.json",
        {"version": "test"},
    )
    canonical_readiness = _write_json(
        tmp_path / "canonical_readiness.json",
        {
            "passed": True,
            "data_integrity_passed": True,
            "promotion_window_ready": True,
            "profile_frozen": True,
            "return_metrics_evaluated": False,
            "canonical_manifest": _artifact(canonical_manifest),
            "holdout": {
                "first_session": str(canonical_dates[0].date()),
                "last_session": str(canonical_dates[-1].date()),
                "sessions": 60,
                "required_sessions": 60,
                "remaining_sessions": 0,
                "calendar_sha256": hashlib.sha256(
                    "\n".join(
                        day.strftime("%Y-%m-%d") for day in canonical_dates
                    ).encode("ascii")
                ).hexdigest().upper(),
            },
        },
    )
    canonical_binding = {
        "schema_version": 1,
        "readiness": _artifact(canonical_readiness),
        "manifest": _artifact(canonical_manifest),
        "holdout": {
            "start": str(canonical_dates[0].date()),
            "end": str(canonical_dates[-1].date()),
            "sessions": 60,
            "calendar_sha256": hashlib.sha256(
                "\n".join(
                    day.strftime("%Y-%m-%d") for day in canonical_dates
                ).encode("ascii")
            ).hexdigest().upper(),
        },
    }
    evidence = _write_json(
        tmp_path / "evidence.json",
        {
            "schema_version": 2,
            "holdout": {
                "end": str(canonical_dates[-1].date()),
            },
            "canonical_market_data": canonical_binding,
        },
    )
    promotion = _write_json(
        tmp_path / "promotion.json",
        {
            "passed": True,
            "paper_candidate_promotion_allowed": True,
            "selected_profile_id": PROFILE,
            "evidence": str(evidence.resolve()),
            "evidence_sha256": _artifact(evidence)["sha256"],
            "canonical_market_data": canonical_binding,
        },
    )
    last_date = pd.Timestamp("2026-07-01") + pd.offsets.BDay(sessions - 1)
    preflight = _write_json(
        tmp_path / "preflight.json",
        {
            "passed": True,
            "errors": [],
            "target": {
                "invalid_lots": 0,
                "forbidden_hits": [],
                "trade_dates": [last_date.strftime("%Y-%m-%d")],
            },
        },
    )
    compare = _write_json(
        tmp_path / "compare.json",
        {
            "profile": PROFILE_ALIAS,
            "gm_date_shift_trading_days": 0,
            "fill_comparison": {
                "gm_only_keys": 0,
                "local_only_keys": 0,
                "volume_mismatch_keys": 0,
                "filled_volume_diff_total": 0,
            },
            "gm": {
                "unexpected_rejected_orders": 0,
                "unresolved_rejected_sell_symbols": 0,
            },
        },
    )
    registry = tmp_path / "registry.jsonl"
    for index, day in enumerate(pd.bdate_range("2026-07-01", periods=sessions)):
        run_id = f"run-{index}"
        identity = {
            "activation_schema_version": 1,
            "run_id": run_id,
            "activation_id": f"activation-{index}",
            "strategy_id": "paper-strategy",
            "account_id": ACCOUNT,
            "run_mode": "LIVE",
            "trading_env": "PAPER",
            "trade_date": day.strftime("%Y-%m-%d"),
            "signal_date": (day - pd.offsets.BDay(1)).strftime("%Y-%m-%d"),
        }
        append_activation_event(
            registry,
            event=EVENT_STARTED,
            identity=identity,
            timestamp=day + pd.Timedelta(hours=9),
        )
        append_activation_event(
            registry,
            event=EVENT_READY,
            identity=identity,
            timestamp=day + pd.Timedelta(hours=9, minutes=1),
            metrics={"position_sync_succeeded": True},
        )
        append_activation_event(
            registry,
            event=EVENT_FINALIZED,
            identity=identity,
            timestamp=day + pd.Timedelta(hours=15),
            metrics={
                "rebalance_events": 1,
                "pending_target_order_symbols": [],
                "pending_execution_symbols": [],
                "forbidden_clear_pending": [],
                "pending_buy_symbols": [],
                "position_sync_succeeded_at_finalize": True,
                "session_metrics_schema_version": 1,
                "unexpected_rejected_orders": 0,
                "unexplained_target_volume_abs_diff": 0,
                "unresolved_rejected_sell_symbol_list": [],
            },
        )
    return {
        "config": config,
        "canonical_readiness": canonical_readiness,
        "promotion": promotion,
        "preflight": preflight,
        "compare": compare,
        "registry": registry,
    }


def _validate(tmp_path: Path, paths: dict[str, Path]) -> dict:
    return validate(
        config_path=paths["config"],
        canonical_readiness_path=paths["canonical_readiness"],
        promotion_path=paths["promotion"],
        preflight_path=paths["preflight"],
        gm_compare_path=paths["compare"],
        activation_registry_path=paths["registry"],
        expected_account_id=ACCOUNT,
        output_path=tmp_path / "result.json",
    )


def test_complete_evidence_passes_paper_readiness(tmp_path: Path) -> None:
    result = _validate(tmp_path, _build_inputs(tmp_path))
    assert result["passed"] is True
    assert result["paper_simulation_candidate_ready"] is True
    assert result["real_money_deployment_allowed"] is False


def test_promotion_evidence_tampering_breaks_canonical_chain(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    promotion = json.loads(paths["promotion"].read_text(encoding="utf-8"))
    evidence_path = Path(promotion["evidence"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["tampered"] = True
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    result = _validate(tmp_path, paths)

    assert result["passed"] is False
    assert "promotion.canonical_evidence_chain" in {
        row["name"] for row in result["failed_checks"]
    }


def test_incomplete_canonical_window_fails_even_with_passing_promotion(
    tmp_path: Path,
) -> None:
    paths = _build_inputs(tmp_path)
    payload = json.loads(
        paths["canonical_readiness"].read_text(encoding="utf-8")
    )
    payload.update({"passed": False, "promotion_window_ready": False})
    payload["holdout"].update(
        {
            "last_session": "2026-07-31",
            "sessions": 39,
            "remaining_sessions": 21,
        }
    )
    _write_json(paths["canonical_readiness"], payload)

    result = _validate(tmp_path, paths)

    failed = {row["name"] for row in result["failed_checks"]}
    assert "canonical_readiness.passed" in failed
    assert "canonical_readiness.promotion_window" in failed
    assert "canonical_readiness.holdout_sessions" in failed
    assert "promotion.matches_canonical_holdout" in failed


def test_canonical_window_cannot_lower_required_sessions(
    tmp_path: Path,
) -> None:
    paths = _build_inputs(tmp_path)
    payload = json.loads(
        paths["canonical_readiness"].read_text(encoding="utf-8")
    )
    payload["holdout"].update(
        {
            "sessions": 20,
            "required_sessions": 20,
            "remaining_sessions": 0,
        }
    )
    _write_json(paths["canonical_readiness"], payload)

    result = _validate(tmp_path, paths)

    failed = {row["name"] for row in result["failed_checks"]}
    assert "canonical_readiness.minimum_required_sessions" in failed
    assert "canonical_readiness.holdout_sessions" in failed


def test_package_module_entrypoint_is_importable() -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "helki_quant.research.validate_strategy_live_readiness",
            "--help",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--canonical-readiness" in completed.stdout


def test_volume_mismatch_fails_closed(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    payload = json.loads(paths["compare"].read_text(encoding="utf-8"))
    payload["fill_comparison"]["volume_mismatch_keys"] = 1
    _write_json(paths["compare"], payload)
    result = _validate(tmp_path, paths)
    assert result["passed"] is False
    assert "gm_compare.volume_mismatch_keys" in {
        row["name"] for row in result["failed_checks"]
    }


def test_pending_execution_fails_closed(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    lines = paths["registry"].read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    record["metrics"]["pending_execution_symbols"] = ["SZSE.300001"]
    record.pop("record_hash")
    from paper_activation_registry import _canonical_hash

    record["record_hash"] = _canonical_hash(record)
    lines[-1] = json.dumps(record, sort_keys=True)
    paths["registry"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = _validate(tmp_path, paths)
    assert result["passed"] is False
    assert "paper.pending_execution" in {
        row["name"] for row in result["failed_checks"]
    }


def test_paper_dates_must_follow_untouched_holdout(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    evidence_path = Path(
        json.loads(paths["promotion"].read_text(encoding="utf-8"))["evidence"]
    )
    _write_json(evidence_path, {"holdout": {"end": "2026-07-31"}})
    result = _validate(tmp_path, paths)
    assert result["passed"] is False
    assert "paper.after_untouched_holdout" in {
        row["name"] for row in result["failed_checks"]
    }


def test_unexpected_rejection_fails_session_quality(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    lines = paths["registry"].read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    record["metrics"]["unexpected_rejected_orders"] = 1
    record.pop("record_hash")
    from paper_activation_registry import _canonical_hash

    record["record_hash"] = _canonical_hash(record)
    lines[-1] = json.dumps(record, sort_keys=True)
    paths["registry"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = _validate(tmp_path, paths)
    assert "paper.unexpected_reject_sessions" in {
        row["name"] for row in result["failed_checks"]
    }


def test_latest_unresolved_sell_fails_even_when_market_restricted(
    tmp_path: Path,
) -> None:
    paths = _build_inputs(tmp_path)
    lines = paths["registry"].read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    record["metrics"]["unresolved_rejected_sell_symbol_list"] = [
        "SZSE.300001"
    ]
    record.pop("record_hash")
    from paper_activation_registry import _canonical_hash

    record["record_hash"] = _canonical_hash(record)
    lines[-1] = json.dumps(record, sort_keys=True)
    paths["registry"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = _validate(tmp_path, paths)
    assert "paper.latest_unresolved_sell_symbols" in {
        row["name"] for row in result["failed_checks"]
    }


def test_duplicate_finalized_session_for_trade_date_fails_closed(
    tmp_path: Path,
) -> None:
    paths = _build_inputs(tmp_path)
    day = pd.Timestamp("2026-07-01")
    identity = {
        "activation_schema_version": 1,
        "run_id": "duplicate-run",
        "activation_id": "duplicate-activation",
        "strategy_id": "paper-strategy",
        "account_id": ACCOUNT,
        "run_mode": "LIVE",
        "trading_env": "PAPER",
        "trade_date": day.strftime("%Y-%m-%d"),
        "signal_date": (day - pd.offsets.BDay(1)).strftime("%Y-%m-%d"),
    }
    from paper_activation_registry import _canonical_hash

    records = [
        json.loads(line)
        for line in paths["registry"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    previous_hash = records[-1]["record_hash"]
    for event, timestamp, metrics in (
        (EVENT_STARTED, day + pd.Timedelta(hours=10), {}),
        (
            EVENT_READY,
            day + pd.Timedelta(hours=10, minutes=1),
            {"position_sync_succeeded": True},
        ),
        (
            EVENT_FINALIZED,
            day + pd.Timedelta(hours=15),
            {
                "rebalance_events": 1,
                "pending_target_order_symbols": [],
                "pending_execution_symbols": [],
                "forbidden_clear_pending": [],
                "pending_buy_symbols": [],
                "position_sync_succeeded_at_finalize": True,
                "session_metrics_schema_version": 1,
                "unexpected_rejected_orders": 0,
                "unexplained_target_volume_abs_diff": 0,
                "unresolved_rejected_sell_symbol_list": [],
            },
        ),
    ):
        record = {
            **identity,
            "event": event,
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
            "previous_hash": previous_hash,
        }
        record["record_hash"] = _canonical_hash(record)
        records.append(record)
        previous_hash = record["record_hash"]
    paths["registry"].write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )

    result = _validate(tmp_path, paths)

    assert "paper.one_finalized_session_per_trade_date" in {
        row["name"] for row in result["failed_checks"]
    }
