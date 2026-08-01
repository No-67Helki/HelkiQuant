from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RESEARCH = Path(__file__).resolve().parents[1] / "src" / "helki_quant" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from held_intraday_live_features import (  # noqa: E402
    add_held_cross_sectional_features,
    minimum_lot,
    normalize_gm_minute_bars,
)
from held_intraday_factor_engineering import add_realtime_reproducible_factors  # noqa: E402
from inner_t0_bidirectional_engine import (  # noqa: E402
    build_entry_intent,
    build_exit_intent,
    percentile_score,
    select_bidirectional_candidates,
    trigger_reached,
)
from inner_t0_multidecision_shadow_engine import (  # noqa: E402
    build_daily_meta_values,
    build_sell_entry_intent,
    combine_primary_secondary,
    isotonic_score,
    require_fresh_signal,
    require_fresh_target,
    ridge_gate_score,
    select_sell_first_candidates,
    trigger_reached as shadow_trigger_reached,
)
import gm_inner_t0_multidecision_shadow_main as shadow_runtime  # noqa: E402
from build_inner_multidecision_shadow_candidate import (  # noqa: E402
    ORDER_CALL_NAMES,
    called_function_names,
    target_snapshot,
)
from compare_inner_multidecision_shadow_audit import (  # noqa: E402
    compare_shadow_audits,
    sha256_file,
)
from inner_shadow_audit_contract import (  # noqa: E402
    AUDIT_SCHEMA_VERSION,
    REGISTRY_FILENAME,
    REQUIRED_AUDIT_TABLES,
    registry_record_hash,
)


def _write_registry(root: Path, records: list[dict]) -> None:
    previous_hash = ""
    lines = []
    for values in records:
        record = {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "previous_hash": previous_hash,
            "timestamp": "2026-07-15T15:00:00+08:00",
            **values,
        }
        record["record_hash"] = registry_record_hash(record)
        previous_hash = record["record_hash"]
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    (root / REGISTRY_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_complete_shadow_run(
    root: Path,
    *,
    run_id: str = "run_001",
    trade_date: str = "2026-07-15",
    primary_time: str = "09:45:15",
    entry_value_ref: float = 1010.0,
    exit_value_ref: float = 980.0,
    virtual_pnl: float = 20.0,
) -> tuple[Path, dict]:
    run = root / run_id
    run.mkdir(parents=True)
    tables = {
        "decision_scores.csv": pd.DataFrame(
            [
                {
                    "date": trade_date,
                    "time": primary_time,
                    "component": "0945_high_confidence",
                    "symbol": "SZSE.300001",
                    "local_symbol": "SZ300001",
                    "raw_score": 0.01,
                    "model_score": 0.01,
                },
                {
                    "date": trade_date,
                    "time": "10:00:15",
                    "component": "1000_daily_ridge_gate",
                    "symbol": "SZSE.300001",
                    "local_symbol": "SZ300001",
                    "raw_score": 0.8,
                    "model_score": 0.8,
                },
            ]
        ),
        "candidate_intents.csv": pd.DataFrame(
            [
                {
                    "date": trade_date,
                    "time": primary_time,
                    "symbol": "SZSE.300001",
                    "local_symbol": "SZ300001",
                    "component": "0945_high_confidence",
                    "score": 0.01,
                    "meta_gate_score": np.nan,
                    "action": "COMPONENT_CANDIDATE",
                    "dry_run": True,
                }
            ]
        ),
        "entry_intents.csv": pd.DataFrame(
            [
                {
                    "date": trade_date,
                    "time": "10:10:00",
                    "symbol": "SZSE.300001",
                    "local_symbol": "SZ300001",
                    "component": "0945_high_confidence",
                    "score": 0.01,
                    "action": "SELL_FIRST_TRIGGERED",
                    "dry_run": True,
                    "trigger_price": 10.10,
                    "entry_limit": 10.075,
                    "projected_daily_turnover": 0.002,
                    "volume": 100,
                    "held_volume": 300,
                }
            ]
        ),
        "exit_intents.csv": pd.DataFrame(
            [
                {
                    "date": trade_date,
                    "time": "14:45:10",
                    "symbol": "SZSE.300001",
                    "local_symbol": "SZ300001",
                    "component": "0945_high_confidence",
                    "action": "BUYBACK_INTENT",
                    "dry_run": True,
                    "entry_value_ref": entry_value_ref,
                    "exit_value_ref": exit_value_ref,
                    "virtual_pnl": virtual_pnl,
                }
            ]
        ),
        "runtime_events.csv": pd.DataFrame(
            [
                {
                    "date": trade_date,
                    "time": primary_time,
                    "event": "DECISION_COMPLETE",
                    "component": "0945_high_confidence",
                    "combined_candidates": 1,
                },
                {
                    "date": trade_date,
                    "time": "10:00:15",
                    "event": "DECISION_COMPLETE",
                    "component": "1000_daily_ridge_gate",
                    "combined_candidates": 1,
                },
                {
                    "date": trade_date,
                    "time": "14:51:05",
                    "event": "SESSION_FINALIZED",
                    "complete": True,
                },
            ]
        ),
    }
    audit_files = {}
    for name in REQUIRED_AUDIT_TABLES:
        tables[name].to_csv(run / name, index=False)
        audit_files[name] = {
            "rows": len(tables[name]),
            "bytes": (run / name).stat().st_size,
            "sha256": sha256_file(run / name),
        }
    model_hash = "A" * 64
    target_hash = "B" * 64
    forbidden_hash = "C" * 64
    summary = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_run_id": run_id,
        "run_mode": "LIVE",
        "dry_run": True,
        "account_id": "paper-account",
        "model_manifest_sha256": model_hash,
        "target_context_sha256": target_hash,
        "forbidden_sha256": forbidden_hash,
        "target_source_date": trade_date,
        "target_age_days": 0,
        "signal_age_days": 1,
        "max_signal_age_days": 4,
        "actual_submission_api_present": False,
        "paper_orders_allowed": False,
        "main_py_integration_allowed": False,
        "deployment_allowed": False,
        "decision_score_rows": len(tables["decision_scores.csv"]),
        "candidate_rows": len(tables["candidate_intents.csv"]),
        "entry_rows": len(tables["entry_intents.csv"]),
        "entry_triggered_rows": 1,
        "exit_rows": len(tables["exit_intents.csv"]),
        "successful_exit_intents": 1,
        "runtime_event_rows": len(tables["runtime_events.csv"]),
        "audit_files": audit_files,
        "complete_session_dates": [trade_date],
        "incomplete_session_dates": [],
        "observation_session_count": 1,
        "session_complete": True,
        "session_details": [
            {
                "date": trade_date,
                "completed_decisions": [
                    "0945_high_confidence",
                    "1000_daily_ridge_gate",
                ],
                "missing_decisions": [],
                "blocked_decisions": [],
                "exit_scan_complete": True,
                "finalized": True,
                "complete": True,
            }
        ],
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    registry_base = {
        "run_id": run_id,
        "run_mode": "LIVE",
        "dry_run": True,
        "account_id": summary["account_id"],
    }
    _write_registry(
        root,
        [
            {
                **registry_base,
                "event": "RUN_STARTED",
                "model_manifest_sha256": model_hash,
                "target_context_sha256": target_hash,
                "forbidden_sha256": forbidden_hash,
            },
            {
                **registry_base,
                "event": "SESSION_FINALIZED",
                "date": trade_date,
                "complete": True,
            },
        ],
    )
    return run, summary


def test_multidecision_isotonic_and_daily_ridge_scores() -> None:
    assert isotonic_score(0.25, np.array([0.1, 0.2, 0.4]), np.array([-0.01, 0.0, 0.02])) == pytest.approx(0.005)
    scores = pd.DataFrame(
        {
            "instrument": ["SZA", "SZB", "SZC"],
            "model_score": [0.8, 0.6, 0.1],
        }
    )
    values = build_daily_meta_values(scores)
    assert values["held_count"] == 3.0
    assert values["score_top2_mean"] == pytest.approx(0.7)
    artifact = {
        "meta_features": ["held_count", "score_top2_mean"],
        "scaler": {"mean": [2.0, 0.5], "scale": [1.0, 0.1]},
        "ridge": {"coefficient": [0.2, 0.3], "intercept": -0.1},
    }
    assert ridge_gate_score(values, artifact) == pytest.approx(0.7)


def test_multidecision_primary_wins_conflict_and_target_fails_stale() -> None:
    base = pd.DataFrame(
        [
            {"symbol": "SZSE.300001", "local_symbol": "SZ300001", "held_volume": 300, "available_volume": 300, "decision_price": 10.0, "model_score": 0.02},
            {"symbol": "SZSE.300002", "local_symbol": "SZ300002", "held_volume": 300, "available_volume": 300, "decision_price": 20.0, "model_score": 0.01},
            {"symbol": "SZSE.300003", "local_symbol": "SZ300003", "held_volume": 300, "available_volume": 300, "decision_price": 30.0, "model_score": 0.001},
        ]
    )
    primary = select_sell_first_candidates(
        base,
        component="0945_high_confidence",
        score_threshold=0.005,
    )
    secondary_scores = base.assign(model_score=[0.9, 0.7, 0.8])
    secondary = select_sell_first_candidates(
        secondary_scores,
        component="1000_daily_ridge_gate",
        score_threshold=None,
    )
    combined, conflicts = combine_primary_secondary(primary, secondary)
    assert set(conflicts["symbol"]) == {"SZSE.300001"}
    assert set(combined["symbol"]) == {"SZSE.300001", "SZSE.300002", "SZSE.300003"}
    assert combined.loc[combined["symbol"] == "SZSE.300001", "component"].item() == "0945_high_confidence"
    assert require_fresh_target("2026-07-15", "2026-07-15") == 0
    assert require_fresh_signal("2026-07-15", "2026-07-14") == 1
    with pytest.raises(RuntimeError, match="stale target context"):
        require_fresh_target("2026-07-15", "2026-06-12")
    with pytest.raises(RuntimeError, match="completed earlier session"):
        require_fresh_signal("2026-07-15", "2026-07-15")
    with pytest.raises(RuntimeError, match="stale daily signal"):
        require_fresh_signal("2026-07-15", "2026-06-05")


def test_multidecision_sell_trigger_and_turnover_budget() -> None:
    candidate = {
        "symbol": "SZSE.300001",
        "local_symbol": "SZ300001",
        "component": "0945_high_confidence",
        "volume": 100,
        "available_volume": 300,
    }
    assert shadow_trigger_reached(10.075, 10.075)
    assert not shadow_trigger_reached(10.07, 10.075)
    intent, accepted = build_sell_entry_intent(
        candidate,
        trigger_price=10.075,
        nav=1_000_000,
        turnover_used=0.0,
    )
    assert accepted
    assert intent["action"] == "SELL_FIRST_TRIGGERED"
    _, accepted = build_sell_entry_intent(
        candidate,
        trigger_price=10.075,
        nav=10_000,
        turnover_used=2_000,
    )
    assert not accepted


def test_multidecision_runtime_combines_primary_and_ridge_candidates(monkeypatch) -> None:
    today = "2026-07-15"
    context = type("Context", (), {"now": pd.Timestamp(f"{today} 10:00:05")})()
    rows = []
    for index in range(20):
        rows.append(
            {
                "symbol": f"SZSE.{300001 + index:06d}",
                "local_symbol": f"SZ{300001 + index:06d}",
                "held_volume": 300,
                "available_volume": 300,
                "decision_price": 10.0 + index,
                "mark_price": 9.9 + index,
                "target_missing": 0.0,
                "volume_scale": 0.01,
                "shares": 300.0,
            }
        )
    feature_frame = pd.DataFrame(rows)
    scores = {
        shadow_runtime.PRIMARY_COMPONENT: [0.020, 0.010, 0.001] + [0.0] * 17,
        shadow_runtime.SECONDARY_COMPONENT: [0.90, 0.70, 0.80] + [0.0] * 17,
    }

    def fake_score(frame: pd.DataFrame, component: str) -> pd.DataFrame:
        out = frame.copy()
        out["raw_score"] = scores[component]
        out["model_score"] = scores[component]
        return out

    shadow_runtime._manifest = {
        "primary_0945": {
            "feature_cols": ["shares"],
            "daily_top_n": 2,
            "score_threshold": 0.005,
            "trigger_distance": 0.0075,
        },
        "secondary_1000": {
            "feature_cols": ["shares"],
            "daily_top_n": 2,
            "trigger_distance": 0.0075,
            "daily_gate": {"threshold": 0.0},
        },
    }
    shadow_runtime._decision_keys = set()
    shadow_runtime._component_candidates = {}
    shadow_runtime._daily_candidates = {}
    shadow_runtime._daily_entries = {}
    shadow_runtime._subscribed_symbols = set()
    shadow_runtime._decision_scores = []
    shadow_runtime._candidate_intents = []
    shadow_runtime._runtime_events = []
    monkeypatch.setattr(shadow_runtime, "_load_target_context", lambda _today: None)
    monkeypatch.setattr(
        shadow_runtime,
        "_build_decision_features",
        lambda *_args, **_kwargs: (
            feature_frame.copy(),
            {
                "positions": 20,
                "allowed": 20,
                "feature_rows": 20,
                "failures": 0,
                "success_ratio": 1.0,
                "target_missing_ratio": 0.0,
            },
        ),
    )
    monkeypatch.setattr(shadow_runtime, "_score_frame", fake_score)
    monkeypatch.setattr(shadow_runtime, "ridge_gate_score", lambda *_args, **_kwargs: 0.01)
    monkeypatch.setattr(shadow_runtime, "subscribe", lambda **_kwargs: None)
    monkeypatch.setattr(shadow_runtime, "unsubscribe", lambda **_kwargs: None)
    monkeypatch.setattr(shadow_runtime, "_write_audit", lambda: None)
    monkeypatch.setattr(
        shadow_runtime,
        "_event",
        lambda _context, event, **values: shadow_runtime._runtime_events.append(
            {"event": event, **values}
        ),
    )

    shadow_runtime.on_primary_decision_scan(context)
    shadow_runtime.on_secondary_decision_scan(context)

    assert (today, shadow_runtime.PRIMARY_COMPONENT) in shadow_runtime._decision_keys
    assert (today, shadow_runtime.SECONDARY_COMPONENT) in shadow_runtime._decision_keys
    assert set(shadow_runtime._daily_candidates) == {
        "SZSE.300001",
        "SZSE.300002",
        "SZSE.300003",
    }
    assert (
        shadow_runtime._daily_candidates["SZSE.300001"]["component"]
        == shadow_runtime.PRIMARY_COMPONENT
    )
    assert any(
        row.get("action") == "DROP_SECONDARY_PRIMARY_CONFLICT"
        and row.get("symbol") == "SZSE.300001"
        for row in shadow_runtime._candidate_intents
    )
    ridge_events = [
        row for row in shadow_runtime._runtime_events if row["event"] == "DAILY_RIDGE_GATE"
    ]
    assert len(ridge_events) == 1
    assert ridge_events[0]["enabled"] is True


def test_shadow_target_freshness_cannot_be_bypassed_by_relabeling(tmp_path: Path) -> None:
    target = tmp_path / "targets.csv"
    frame = pd.DataFrame(
        {
            "instrument": ["SZ300001", "SZ300002"],
            "trade_date": ["2026-07-15", "2026-07-15"],
            "signal_date": ["2026-06-05", "2026-06-05"],
        }
    )
    frame.to_csv(target, index=False)
    relabeled = target_snapshot(target, pd.Timestamp("2026-07-15"))
    assert relabeled["target_date_passed"] is True
    assert relabeled["signal_date_passed"] is False
    assert relabeled["passed"] is False

    frame["signal_date"] = "2026-07-14"
    frame.to_csv(target, index=False)
    fresh = target_snapshot(target, pd.Timestamp("2026-07-15"))
    assert fresh["signal_date_passed"] is True
    assert fresh["passed"] is True


def test_multidecision_shadow_runtime_has_no_order_submission_calls() -> None:
    runtime_path = RESEARCH / "gm_inner_t0_multidecision_shadow_main.py"
    assert called_function_names(runtime_path).intersection(ORDER_CALL_NAMES) == set()


def test_shadow_runtime_writes_atomic_hash_manifested_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "models.json"
    target = tmp_path / "targets.csv"
    forbidden = tmp_path / "forbidden.csv"
    model.write_text("{}", encoding="utf-8")
    target.write_text("instrument\nSZ300001\n", encoding="utf-8")
    forbidden.write_text("instrument\nSZ300999\n", encoding="utf-8")
    monkeypatch.setattr(shadow_runtime, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(shadow_runtime, "AUDIT_RUN_ID", "atomic_run")
    monkeypatch.setattr(shadow_runtime, "MODEL_MANIFEST_PATH", model)
    monkeypatch.setattr(shadow_runtime, "TARGET_CONTEXT_PATH", target)
    monkeypatch.setattr(shadow_runtime, "FORBIDDEN_PATH", forbidden)
    monkeypatch.setattr(shadow_runtime, "_decision_scores", [])
    monkeypatch.setattr(shadow_runtime, "_candidate_intents", [])
    monkeypatch.setattr(shadow_runtime, "_entry_intents", [])
    monkeypatch.setattr(shadow_runtime, "_exit_intents", [])
    monkeypatch.setattr(
        shadow_runtime,
        "_runtime_events",
        [{"date": "2026-07-15", "time": "09:00:00", "event": "RUN_INITIALIZED"}],
    )
    monkeypatch.setattr(shadow_runtime, "_decision_keys", set())
    monkeypatch.setattr(shadow_runtime, "_exit_dates", set())
    monkeypatch.setattr(shadow_runtime, "_finalized_dates", set())

    shadow_runtime._append_registry_event(
        "RUN_STARTED",
        model_manifest_sha256=sha256_file(model),
        target_context_sha256=sha256_file(target),
        forbidden_sha256=sha256_file(forbidden),
    )
    shadow_runtime._write_audit()

    run = tmp_path / "audit" / "atomic_run"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["audit_schema_version"] == AUDIT_SCHEMA_VERSION
    assert set(summary["audit_files"]) == set(REQUIRED_AUDIT_TABLES)
    for name in REQUIRED_AUDIT_TABLES:
        assert (run / name).exists()
        assert summary["audit_files"][name]["sha256"] == sha256_file(run / name)
    assert not list(run.glob("*.pending"))


def test_session_finalize_does_not_extend_buyback_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = "2026-07-15"
    context = type("Context", (), {"now": pd.Timestamp(f"{today} 14:51:00")})()
    events = [
        {
            "date": today,
            "time": "09:45:10",
            "event": "DECISION_COMPLETE",
            "component": shadow_runtime.PRIMARY_COMPONENT,
        },
        {
            "date": today,
            "time": "10:00:10",
            "event": "DECISION_COMPLETE",
            "component": shadow_runtime.SECONDARY_COMPONENT,
        },
    ]
    registry = []
    monkeypatch.setattr(shadow_runtime, "_runtime_events", events)
    monkeypatch.setattr(shadow_runtime, "_exit_dates", set())
    monkeypatch.setattr(shadow_runtime, "_finalized_dates", set())
    monkeypatch.setattr(
        shadow_runtime,
        "on_exit_scan",
        lambda _context: pytest.fail("14:51 finalization must not evaluate a new exit"),
    )
    monkeypatch.setattr(shadow_runtime, "_write_audit", lambda: None)
    monkeypatch.setattr(
        shadow_runtime,
        "_append_registry_event",
        lambda event, **values: registry.append({"event": event, **values}),
    )

    shadow_runtime.on_session_finalize(context)

    assert registry[0]["event"] == "SESSION_FINALIZED"
    assert registry[0]["complete"] is False
    assert registry[0]["exit_scan_complete"] is False


def test_shadow_audit_separates_technical_and_economic_gates(tmp_path: Path) -> None:
    run, _ = _build_complete_shadow_run(tmp_path)
    forbidden = tmp_path / "forbidden.csv"
    pd.DataFrame({"instrument": ["SZ300999"]}).to_csv(forbidden, index=False)

    strict = compare_shadow_audits(
        [run],
        forbidden,
        tmp_path / "strict.json",
        expected_account_id="paper-account",
    )
    assert strict["technical_passed"] is True
    assert strict["economic_shadow_gate_passed"] is False
    assert strict["status"] == "research_only_collect_more_shadow_evidence"

    relaxed = compare_shadow_audits(
        [run],
        forbidden,
        tmp_path / "relaxed.json",
        minimum_round_trips=1,
        minimum_symbols=1,
        minimum_active_days=1,
        minimum_active_months=1,
        minimum_observation_sessions=1,
        maximum_top3_positive_pnl_share=1.0,
        expected_account_id="paper-account",
    )
    assert relaxed["technical_passed"] is True
    assert relaxed["economic_shadow_gate_passed"] is True
    assert relaxed["virtual_economics"]["cumulative_pnl"] == pytest.approx(20.0)
    assert relaxed["stress_economics"]["additional_slippage_cost"] == pytest.approx(
        1.99
    )
    assert relaxed["stress_economics"]["cumulative_pnl"] == pytest.approx(18.01)


def test_shadow_economic_gate_uses_adverse_two_sided_slippage(tmp_path: Path) -> None:
    run, _ = _build_complete_shadow_run(tmp_path, virtual_pnl=1.0)
    forbidden = tmp_path / "forbidden.csv"
    pd.DataFrame({"instrument": ["SZ300999"]}).to_csv(forbidden, index=False)

    report = compare_shadow_audits(
        [run],
        forbidden,
        tmp_path / "stress_failed.json",
        minimum_round_trips=1,
        minimum_symbols=1,
        minimum_active_days=1,
        minimum_active_months=1,
        minimum_observation_sessions=1,
        minimum_profit_factor=0.0,
        maximum_top3_positive_pnl_share=1.0,
        expected_account_id="paper-account",
    )

    assert report["technical_passed"] is True
    assert report["virtual_economics"]["cumulative_pnl"] == pytest.approx(1.0)
    assert report["stress_economics"]["cumulative_pnl"] == pytest.approx(-0.99)
    assert report["economic_checks"]["positive_stress_cumulative_pnl"] is False
    assert report["economic_shadow_gate_passed"] is False


def test_shadow_audit_rejects_invalid_stress_slippage_rate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stress_slippage_rate"):
        compare_shadow_audits(
            [],
            tmp_path / "forbidden.csv",
            tmp_path / "invalid_stress.json",
            stress_slippage_rate=-0.001,
        )


def test_shadow_audit_rejects_tampered_csv(tmp_path: Path) -> None:
    run, _ = _build_complete_shadow_run(tmp_path)
    forbidden = tmp_path / "forbidden.csv"
    pd.DataFrame({"instrument": ["SZ300999"]}).to_csv(forbidden, index=False)
    with (run / "entry_intents.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    report = compare_shadow_audits(
        [run],
        forbidden,
        tmp_path / "tampered.json",
        minimum_observation_sessions=1,
    )

    assert report["technical_passed"] is False
    assert any("audit hash mismatch" in error for error in report["errors"])


def test_shadow_audit_rejects_late_decision(tmp_path: Path) -> None:
    run, _ = _build_complete_shadow_run(tmp_path, primary_time="10:05:00")
    forbidden = tmp_path / "forbidden.csv"
    pd.DataFrame({"instrument": ["SZ300999"]}).to_csv(forbidden, index=False)

    report = compare_shadow_audits(
        [run],
        forbidden,
        tmp_path / "late.json",
        minimum_observation_sessions=1,
    )

    assert report["technical_passed"] is False
    assert any("decision latency invalid" in error for error in report["errors"])


def test_shadow_audit_rejects_missing_decision_even_with_rehashed_bundle(
    tmp_path: Path,
) -> None:
    run, summary = _build_complete_shadow_run(tmp_path)
    events_path = run / "runtime_events.csv"
    events = pd.read_csv(events_path)
    events = events[
        ~(
            events["event"].eq("DECISION_COMPLETE")
            & events["component"].eq("1000_daily_ridge_gate")
        )
    ]
    events.to_csv(events_path, index=False)
    summary["runtime_event_rows"] = len(events)
    summary["audit_files"]["runtime_events.csv"] = {
        "rows": len(events),
        "bytes": events_path.stat().st_size,
        "sha256": sha256_file(events_path),
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    forbidden = tmp_path / "forbidden.csv"
    pd.DataFrame({"instrument": ["SZ300999"]}).to_csv(forbidden, index=False)

    report = compare_shadow_audits(
        [run],
        forbidden,
        tmp_path / "missing_decision.json",
        minimum_observation_sessions=1,
    )

    assert report["technical_passed"] is False
    assert any("requires exactly one completed" in error for error in report["errors"])


def test_shadow_audit_rejects_registered_run_omission(tmp_path: Path) -> None:
    run, summary = _build_complete_shadow_run(tmp_path)
    registry_base = {
        "run_mode": "LIVE",
        "dry_run": True,
        "account_id": summary["account_id"],
    }
    _write_registry(
        tmp_path,
        [
            {
                **registry_base,
                "run_id": "run_001",
                "event": "RUN_STARTED",
                "model_manifest_sha256": summary["model_manifest_sha256"],
                "target_context_sha256": summary["target_context_sha256"],
                "forbidden_sha256": summary["forbidden_sha256"],
            },
            {
                **registry_base,
                "run_id": "run_001",
                "event": "SESSION_FINALIZED",
                "date": "2026-07-15",
                "complete": True,
            },
            {
                **registry_base,
                "run_id": "run_002",
                "event": "RUN_STARTED",
                "model_manifest_sha256": summary["model_manifest_sha256"],
                "target_context_sha256": "D" * 64,
                "forbidden_sha256": summary["forbidden_sha256"],
            },
        ],
    )
    forbidden = tmp_path / "forbidden.csv"
    pd.DataFrame({"instrument": ["SZ300999"]}).to_csv(forbidden, index=False)

    report = compare_shadow_audits(
        [run],
        forbidden,
        tmp_path / "omitted.json",
        minimum_observation_sessions=1,
    )

    assert report["technical_passed"] is False
    assert any("registered runs missing" in error for error in report["errors"])


def test_shadow_audit_rejects_registered_crash_without_summary(tmp_path: Path) -> None:
    account_id = "paper-account"
    _write_registry(
        tmp_path,
        [
            {
                "run_id": "crashed_run",
                "run_mode": "LIVE",
                "dry_run": True,
                "account_id": account_id,
                "event": "RUN_STARTED",
                "model_manifest_sha256": "A" * 64,
                "target_context_sha256": "B" * 64,
                "forbidden_sha256": "C" * 64,
            }
        ],
    )
    forbidden = tmp_path / "forbidden.csv"
    pd.DataFrame({"instrument": ["SZ300999"]}).to_csv(forbidden, index=False)

    report = compare_shadow_audits(
        [],
        forbidden,
        tmp_path / "crashed.json",
        registry_path=tmp_path / REGISTRY_FILENAME,
    )

    assert report["technical_passed"] is False
    assert report["status"] == "shadow_technical_audit_failed"
    assert any("no readable summary" in error for error in report["errors"])


def test_gm_share_volume_is_normalized_to_research_lots() -> None:
    bars = pd.DataFrame(
        {
            "eob": pd.date_range("2026-06-12 09:31", periods=3, freq="min"),
            "open": [10.0, 10.1, 10.2],
            "high": [10.1, 10.2, 10.3],
            "low": [9.9, 10.0, 10.1],
            "close": [10.0, 10.1, 10.2],
            "volume": [10_000, 20_000, 30_000],
            "amount": [100_000, 202_000, 306_000],
        }
    )
    normalized, audit = normalize_gm_minute_bars(bars)
    assert audit["valid"] is True
    assert audit["volume_scale"] == 0.01
    assert normalized["volume"].sum() == 600


def test_ambiguous_minute_units_fail_closed() -> None:
    bars = pd.DataFrame(
        {
            "eob": ["2026-06-12 09:31"],
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
            "volume": [100.0],
            "amount": [10_000.0],
        }
    )
    normalized, audit = normalize_gm_minute_bars(bars)
    assert normalized.empty
    assert audit["reason"] == "ambiguous_volume_unit"


def test_cross_sectional_features_are_held_universe_only() -> None:
    frame = pd.DataFrame(
        {
            "instrument": ["SZA", "SZB"],
            "held_unrealized_ret_approx": [-0.1, 0.1],
            "held_weight_gap_to_target": [0.01, -0.01],
            "visible_ret_from_open": [-0.02, 0.02],
            "visible_gap_vs_mark": [-0.01, 0.01],
            "visible_last_vs_mark": [-0.03, 0.03],
            "visible_range": [0.01, 0.02],
            "visible_range_pos": [0.2, 0.8],
            "visible_vwap_dev": [-0.01, 0.01],
            "visible_minute_vol": [0.1, 0.2],
            "visible_recent_5m_ret": [-0.01, 0.01],
            "visible_recent_10m_ret": [-0.01, 0.01],
            "visible_trend_slope": [-0.01, 0.01],
            "visible_drawdown_from_high": [-0.03, -0.01],
            "visible_rebound_from_low": [0.01, 0.03],
            "visible_log_volume": [10.0, 12.0],
            "visible_log_amount": [15.0, 17.0],
            "visible_volume_to_held": [5.0, 10.0],
            "visible_amount_to_position": [2.0, 4.0],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = add_held_cross_sectional_features(frame)
    assert not any(
        isinstance(item.message, pd.errors.PerformanceWarning)
        for item in caught
    )
    assert set(out["held_universe_size"]) == {2.0}
    assert np.allclose(out["held_market_ret_mean"], 0.0)
    assert np.allclose(out["held_market_positive_breadth"], 0.5)
    assert list(out["cs_visible_ret_from_open_rank"]) == [0.5, 1.0]


def test_limit_and_industry_factors_are_reproducible_from_live_snapshot() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2026-06-12"] * 3,
            "decision_time": ["1000"] * 3,
            "instrument": ["SZ300001", "SZ000001", "SH688001"],
            "group": ["software", "software", "semiconductor"],
            "mark_price": [10.0, 10.0, 20.0],
            "sell_price_decision": [11.0, 10.5, 22.0],
            "visible_ret_from_open": [0.04, 0.00, 0.03],
            "visible_gap_vs_mark": [0.01, -0.01, 0.02],
            "visible_vwap_dev": [0.02, -0.02, 0.01],
            "target_weight": [0.01, 0.02, 0.03],
            "weight": [0.008, 0.018, 0.025],
            "middle": [0.6, 0.4, 0.8],
        }
    )

    out = add_realtime_reproducible_factors(frame)

    assert out.loc[0, "board_limit_ratio"] == 0.20
    assert out.loc[1, "board_limit_ratio"] == 0.10
    assert out.loc[2, "board_limit_ratio"] == 0.20
    assert np.isclose(out.loc[0, "visible_distance_to_limit_up"], 12.0 / 11.0 - 1.0)
    assert np.isclose(out.loc[0, "industry_visible_ret_mean"], 0.02)
    assert np.isclose(out.loc[0, "industry_visible_ret_rel"], 0.02)
    assert np.isclose(out.loc[1, "industry_visible_ret_rel"], -0.02)
    assert out.loc[2, "industry_held_count"] == 1.0


def test_candidate_selection_drops_conflict_and_respects_caps() -> None:
    scores = pd.DataFrame(
        [
            {"symbol": "SZSE.300001", "local_symbol": "SZ300001", "held_volume": 300, "available_volume": 300, "decision_price": 10.0, "buy_score": 0.99, "sell_score": 0.99},
            {"symbol": "SZSE.300002", "local_symbol": "SZ300002", "held_volume": 300, "available_volume": 300, "decision_price": 20.0, "buy_score": 0.98, "sell_score": 0.10},
            {"symbol": "SZSE.300003", "local_symbol": "SZ300003", "held_volume": 300, "available_volume": 300, "decision_price": 30.0, "buy_score": 0.95, "sell_score": 0.10},
            {"symbol": "SZSE.300004", "local_symbol": "SZ300004", "held_volume": 300, "available_volume": 300, "decision_price": 40.0, "buy_score": 0.10, "sell_score": 0.98},
            {"symbol": "SZSE.300005", "local_symbol": "SZ300005", "held_volume": 100, "available_volume": 100, "decision_price": 50.0, "buy_score": 0.99, "sell_score": 0.10},
        ]
    )
    selected, conflicts = select_bidirectional_candidates(scores)
    assert set(conflicts["symbol"]) == {"SZSE.300001"}
    assert set(selected["symbol"]) == {"SZSE.300002"}
    assert len(selected) == 1
    assert all(selected["volume"] == 100)
    assert "SZSE.300005" not in set(selected["symbol"])

    no_conflict = scores.copy()
    no_conflict.loc[no_conflict["symbol"] == "SZSE.300001", "sell_score"] = 0.10
    selected, conflicts = select_bidirectional_candidates(no_conflict)
    assert conflicts.empty
    assert set(selected["symbol"]) == {"SZSE.300001", "SZSE.300002", "SZSE.300004"}
    assert len(selected) == 3


def test_star_lot_and_half_inventory_guard() -> None:
    assert minimum_lot("SHSE.688001") == 200
    scores = pd.DataFrame(
        [{"symbol": "SHSE.688001", "local_symbol": "SH688001", "held_volume": 200, "available_volume": 200, "decision_price": 10.0, "buy_score": 1.0, "sell_score": 0.0}]
    )
    selected, _ = select_bidirectional_candidates(scores)
    assert selected.empty


def test_percentile_and_bidirectional_trigger_boundaries() -> None:
    calibration = np.array([0.1, 0.2, 0.3, 0.4])
    assert percentile_score(0.3, calibration) == 0.75
    assert trigger_reached("buy_first", 9.94, 9.94)
    assert not trigger_reached("buy_first", 9.95, 9.94)
    assert trigger_reached("sell_first", 10.075, 10.075)
    assert not trigger_reached("sell_first", 10.07, 10.075)


def test_entry_cash_turnover_and_inventory_guards() -> None:
    candidate = {
        "symbol": "SZSE.300001",
        "local_symbol": "SZ300001",
        "direction": "buy_first",
        "volume": 100,
        "available_volume": 300,
    }
    intent, accepted = build_entry_intent(
        candidate,
        trigger_price=10.0,
        nav=1_000_000,
        cash_available=2_000,
        turnover_used=0.0,
        buy_cash_reserved=0.0,
    )
    assert accepted
    assert intent["action"] == "TRIGGERED"
    _, cash_blocked = build_entry_intent(
        candidate,
        trigger_price=10.0,
        nav=1_000_000,
        cash_available=1_000,
        turnover_used=0.0,
        buy_cash_reserved=0.0,
    )
    assert not cash_blocked
    _, turnover_blocked = build_entry_intent(
        candidate,
        trigger_price=10.0,
        nav=10_000,
        cash_available=10_000,
        turnover_used=2_000,
        buy_cash_reserved=0.0,
    )
    assert not turnover_blocked


def test_exit_intents_restore_original_inventory() -> None:
    buy_exit = build_exit_intent(
        {"symbol": "SZSE.300001", "local_symbol": "SZ300001", "direction": "buy_first", "volume": 100},
        exit_price=10.5,
    )
    sell_exit = build_exit_intent(
        {"symbol": "SZSE.300002", "local_symbol": "SZ300002", "direction": "sell_first", "volume": 100},
        exit_price=9.5,
    )
    assert buy_exit["action"] == "SELL_OLD_INVENTORY_INTENT"
    assert sell_exit["action"] == "BUYBACK_INTENT"
    assert buy_exit["restores_original_inventory"] is True
    assert sell_exit["restores_original_inventory"] is True
