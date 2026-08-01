from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
RUNTIME_TEMPLATE = ROOT / "src" / "helki_quant" / "deployment" / "gmquant"
sys.path.insert(0, str(RESEARCH))

from preflight_outer_middle_paper_candidate import run_preflight, sha256_file  # noqa: E402
from build_outer_middle_paper_launch_candidate import build_candidate  # noqa: E402
from paper_activation_registry import (  # noqa: E402
    EVENT_FINALIZED,
    EVENT_READY,
    EVENT_STARTED,
    append_activation_event,
)


ACCOUNT_ID = "paper-account"
HISTORICAL_PROFILE = "c_outer_loss5_top150_rb20_risk0.60_floor0.30_cap0.30_nost"


def _activation_seed(root: Path) -> Path:
    registry = root / "PAPER_ACTIVATION_REGISTRY.jsonl"
    identity = {
        "activation_schema_version": 1,
        "activation_id": "seed-activation",
        "run_id": "seed-run",
        "strategy_id": "outer-middle-paper",
        "account_id": ACCOUNT_ID,
        "run_mode": "LIVE",
        "trading_env": "PAPER",
        "signal_date": "2026-07-13",
        "trade_date": "2026-07-14",
    }
    for event, timestamp in (
        (EVENT_STARTED, "2026-07-14 09:20:00"),
        (EVENT_READY, "2026-07-14 09:21:00"),
        (EVENT_FINALIZED, "2026-07-14 15:05:00"),
    ):
        append_activation_event(
            registry,
            event=event,
            identity=identity,
            timestamp=timestamp,
        )
    return registry


def _copy_candidate(tmp_path: Path) -> Path:
    destination = tmp_path / "candidate"
    destination.mkdir()
    for name in (
        "main.py",
        "gm_outer_direct_loss5_market_filtered_paper.py",
        "paper_activation_registry.py",
    ):
        shutil.copy2(RUNTIME_TEMPLATE / name, destination / name)

    instruments = [f"SH{600000 + index:06d}" for index in range(80)]
    symbols = [f"SHSE.{600000 + index:06d}" for index in range(80)]
    target_path = destination / "gm_c_baseline_targets.csv"
    pd.DataFrame(
        {
            "trade_date": "2026-06-12",
            "symbol": symbols,
            "instrument": instruments,
            "rank": range(1, 81),
            "middle": [0.80 - index * 0.001 for index in range(80)],
            "target_weight": 0.004,
            "target_shares": 100,
            "nominal_target_weight": 0.004,
            "effective_weight_ref": 0.004,
            "target_notional_ref": 4000.0,
            "group": [f"industry_{index % 10}" for index in range(80)],
            "signal_date": "2026-06-05",
            "price_ref_close": 40.0,
            "avg_amount": 100_000_000.0,
            "listing_days": 1000,
        }
    ).to_csv(target_path, index=False)
    target_manifest_path = destination / "gm_c_baseline_targets.manifest.json"
    target_manifest_path.write_text(
        json.dumps(
            {
                "status": "synthetic_test_target",
                "output_csv": target_path.name,
                "input_rows": 80,
                "output_rows": 80,
                "output_symbols": 80,
                "target_weight_sum_min": 0.32,
                "target_weight_sum_max": 0.32,
                "blocked_actions": 0,
                "deployment_allowed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    forbidden_path = destination / "gm_c_forbidden_symbols.csv"
    pd.DataFrame(
        {
            "instrument": ["SZ300999"],
            "gm_symbol": ["SZSE.300999"],
            "reason": ["synthetic_test_forbidden"],
        }
    ).to_csv(forbidden_path, index=False)
    audit_name = "TARGET_AUDIT.json"
    audit_path = destination / audit_name
    audit_path.write_text(
        json.dumps(
            {
                "status": "gm_target_csv_audit",
                "target_csv": target_path.name,
                "rows": 80,
                "dates": 1,
                "symbols": 80,
                "date_start": "2026-06-12",
                "date_end": "2026-06-12",
                "target_weight_sum_min": 0.32,
                "target_weight_sum_max": 0.32,
                "target_weight_sum_mean": 0.32,
                "min_rows_per_date": 80,
                "max_rows_per_date": 80,
                "forbidden_path": forbidden_path.name,
                "forbidden_hits": 0,
                "forbidden_hit_symbols": [],
                "lot_violations": 0,
                "min_lot_violations": 0,
                "passed": True,
                "deployment_allowed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    source_manifest_path = destination / "SOURCE_SELECTION_INPUT.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "status": "paper_forward_gm_targets_exported",
                "target": target_path.name,
                "prediction": "MIDDLE_SOURCE.csv",
                "signal_date": "2026-06-05",
                "trade_date": "2026-06-12",
                "top_k": 150,
                "rebalance_every": 20,
                "buffer_multiple": 2,
                "risk_budget": 0.60,
                "base_risk_budget": 0.60,
                "industry_cap": 0.30,
                "allocation": {"mode": "fixed_topk"},
                "outer_overlay": {
                    "required": True,
                    "prediction": "OUTER_SOURCE.csv",
                    "probability": 0.20,
                    "threshold": 0.50,
                    "risk_floor": 0.30,
                    "triggered": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    middle_path = destination / "MIDDLE_SOURCE.csv"
    pd.DataFrame(
        {
            "datetime": "2026-06-05",
            "instrument": instruments,
            "middle": [0.80 - index * 0.001 for index in range(80)],
        }
    ).to_csv(middle_path, index=False)
    selection_path = destination / "SELECTION_PROVENANCE.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "paper_selection_provenance",
                "source_manifest": {
                    "path": source_manifest_path.name,
                    "sha256": sha256_file(source_manifest_path),
                },
                "source_target": {
                    "path": target_path.name,
                    "sha256": sha256_file(target_path),
                },
                "middle_prediction": {
                    "path": middle_path.name,
                    "sha256": sha256_file(middle_path),
                },
                "outer_prediction": None,
                "selection": {
                    "middle_model": "canonical_densemble",
                    "outer_model": "broad_adverse_loss5_20d",
                    "signal_date": "2026-06-05",
                    "trade_date": "2026-06-12",
                    "top_k": 150,
                    "rebalance_every": 20,
                    "buffer_multiple": 2,
                    "base_risk_budget": 0.60,
                    "industry_cap": 0.30,
                    "allocation_mode": "fixed_topk",
                    "outer_required": True,
                    "outer_threshold": 0.50,
                    "outer_risk_floor": 0.30,
                    "outer_probability": None,
                    "outer_triggered": False,
                },
                "production_parity_passed": False,
                "failure_reason": "synthetic fixture intentionally lacks outer evidence",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "status": "outer_middle_paper_runtime_template_v5",
        "candidate_name": "synthetic_preflight_fixture",
        "created_at": "2026-06-12",
        "runtime_template_version": 5,
        "deployment_allowed": False,
        "paper_only": True,
        "inner_t0_enabled": False,
        "activation_audit_required": True,
        "session_quality_audit_required": True,
        "session_metrics_schema_version": 1,
        "activation_registry_lock_required": True,
        "single_session_per_trade_date_required": True,
        "default_main_replaced": False,
        "paper_account_id": ACCOUNT_ID,
        "strategy_contract": {
            "middle_model": "canonical_densemble",
            "outer_model": "broad_adverse_loss5_20d",
            "outer_prediction_required": True,
            "top_k": 150,
            "rebalance_every": 20,
            "buffer_multiple": 2,
            "base_risk_budget": 0.60,
            "outer_risk_threshold": 0.50,
            "outer_risk_floor": 0.30,
            "industry_cap": 0.30,
            "allocation_mode": "fixed_topk",
            "initial_cash": 1_000_000.0,
        },
        "runtime_integrity": {
            "frozen_at": "2026-06-12",
            "sha256": {
                name: sha256_file(destination / name)
                for name in (
                    "main.py",
                    "gm_outer_direct_loss5_market_filtered_paper.py",
                    "paper_activation_registry.py",
                )
            },
            "required_runtime_guards": [],
        },
        "target_provenance": {
            "frozen_at": "2026-06-12",
            "source_data_end": "2026-06-05",
            "signal_date": "2026-06-05",
            "trade_date": "2026-06-12",
            "audit_file": audit_name,
            "selection_provenance_file": selection_path.name,
            "sha256": {},
        },
        "target": {
            "file": target_path.name,
            "current_target_date_start": "2026-06-12",
            "current_target_date_end": "2026-06-12",
            "fresh_future_target_required_before_paper": True,
            "fresh_target_generated_at": "2026-06-12",
            "fresh_target_signal_date": "2026-06-05",
            "fresh_target_weight_sum": 0.32,
            "fresh_target_rows": 80,
            "fresh_target_symbols": 80,
            "fresh_target_shares_sum": 8000,
            "st_risk_refreshed_at": "2026-06-12",
            "fresh_target_audit": audit_name,
        },
    }
    for name in (
        target_path.name,
        target_manifest_path.name,
        forbidden_path.name,
        audit_name,
        selection_path.name,
    ):
        manifest["target_provenance"]["sha256"][name] = sha256_file(
            destination / name
        )
    (destination / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return destination


def _rewrite_target_dates(
    candidate: Path,
    *,
    trade_date: str,
    signal_date: str,
    metadata_date: str = "2026-07-14",
) -> None:
    target_path = candidate / "gm_c_baseline_targets.csv"
    frame = pd.read_csv(target_path)
    frame["trade_date"] = trade_date
    frame["signal_date"] = signal_date
    frame.to_csv(target_path, index=False, encoding="utf-8-sig")
    manifest_path = candidate / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    target = manifest["target"]
    target["current_target_date_start"] = trade_date
    target["current_target_date_end"] = trade_date
    target["fresh_target_signal_date"] = signal_date
    target["st_risk_refreshed_at"] = metadata_date
    provenance = manifest["target_provenance"]
    provenance["source_data_end"] = signal_date
    provenance["signal_date"] = signal_date
    provenance["trade_date"] = trade_date
    audit_name = provenance["audit_file"]
    audit_path = candidate / audit_name
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    audit["target_csv"] = str(target_path.resolve())
    audit["forbidden_path"] = str((candidate / "gm_c_forbidden_symbols.csv").resolve())
    audit["date_start"] = trade_date
    audit["date_end"] = trade_date
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in (
        "gm_c_baseline_targets.csv",
        "gm_c_baseline_targets.manifest.json",
        "gm_c_forbidden_symbols.csv",
        audit_name,
    ):
        provenance["sha256"][name] = sha256_file(candidate / name)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_production_parity_selection(
    candidate: Path,
    *,
    trade_date: str,
    signal_date: str,
) -> Path:
    target_path = candidate / "gm_c_baseline_targets.csv"
    target = pd.read_csv(target_path)
    instruments = target["instrument"].astype(str).tolist()
    middle_path = candidate / "MIDDLE_SOURCE.csv"
    outer_path = candidate / "OUTER_SOURCE.csv"
    pd.DataFrame(
        {
            "datetime": signal_date,
            "instrument": instruments,
            "middle": [0.60 + index * 1e-5 for index in range(len(instruments))],
        }
    ).to_csv(middle_path, index=False)
    pd.DataFrame(
        {
            "datetime": signal_date,
            "instrument": instruments,
            "outer": 0.20,
        }
    ).to_csv(outer_path, index=False)
    source_manifest_path = candidate / "SOURCE_SELECTION_INPUT.json"
    source_manifest = {
        "status": "paper_forward_gm_targets_exported",
        "target": str(target_path.resolve()),
        "prediction": str(middle_path.resolve()),
        "signal_date": signal_date,
        "trade_date": trade_date,
        "top_k": 150,
        "rebalance_every": 20,
        "buffer_multiple": 2,
        "risk_budget": 0.60,
        "base_risk_budget": 0.60,
        "industry_cap": 0.30,
        "allocation": {"mode": "fixed_topk"},
        "outer_overlay": {
            "required": True,
            "prediction": str(outer_path.resolve()),
            "probability": 0.20,
            "threshold": 0.50,
            "risk_floor": 0.30,
            "triggered": False,
        },
    }
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selection_path = candidate / "SELECTION_PROVENANCE.json"
    selection = {
        "status": "paper_selection_provenance",
        "frozen_at": "2026-07-15",
        "source_manifest": {
            "path": str(source_manifest_path.resolve()),
            "sha256": sha256_file(source_manifest_path),
        },
        "source_target": {
            "path": str(target_path.resolve()),
            "sha256": sha256_file(target_path),
        },
        "middle_prediction": {
            "path": str(middle_path.resolve()),
            "sha256": sha256_file(middle_path),
        },
        "outer_prediction": {
            "path": str(outer_path.resolve()),
            "sha256": sha256_file(outer_path),
        },
        "selection": {
            "middle_model": "canonical_densemble",
            "outer_model": "broad_adverse_loss5_20d",
            "signal_date": signal_date,
            "trade_date": trade_date,
            "top_k": 150,
            "rebalance_every": 20,
            "buffer_multiple": 2,
            "base_risk_budget": 0.60,
            "industry_cap": 0.30,
            "allocation_mode": "fixed_topk",
            "outer_required": True,
            "outer_threshold": 0.50,
            "outer_risk_floor": 0.30,
            "outer_probability": 0.20,
            "outer_triggered": False,
        },
        "production_parity_passed": True,
        "failure_reason": None,
    }
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    package_manifest_path = candidate / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json"
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8-sig"))
    package_manifest["target_provenance"]["selection_provenance_file"] = selection_path.name
    package_manifest["target_provenance"]["sha256"][selection_path.name] = sha256_file(
        selection_path
    )
    package_manifest_path.write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return source_manifest_path


def _make_release_evidence(
    root: Path,
    *,
    trade_date: str,
    signal_date: str,
    include_stock_universe: bool = False,
) -> tuple[Path, Path, Path, Path]:
    contract = {
        "middle_model": "canonical_densemble",
        "outer_model": "broad_adverse_loss5_20d",
        "outer_prediction_required": True,
        "top_k": 150,
        "rebalance_every": 20,
        "buffer_multiple": 2,
        "base_risk_budget": 0.60,
        "outer_risk_threshold": 0.50,
        "outer_risk_floor": 0.30,
        "industry_cap": 0.30,
        "allocation_mode": "fixed_topk",
        "initial_cash": 1_000_000.0,
    }
    account_positions_path = root / "account_positions.csv"
    pd.DataFrame(columns=["account_id", "symbol", "side", "volume"]).to_csv(
        account_positions_path,
        index=False,
    )
    account_snapshot = {
        "status": "gm_paper_account_snapshot",
        "passed": True,
        "failed_checks": [],
        "snapshot_id": "test-snapshot",
        "captured_at": f"{trade_date}T08:30:00",
        "captured_date": trade_date,
        "account_id": ACCOUNT_ID,
        "paper_only": True,
        "no_order": True,
        "orders_submitted": 0,
        "position_rows": 0,
        "position_symbols": 0,
        "position_shares": 0,
        "cash": {"nav": 1_000_000.0, "available": 1_000_000.0},
        "positions": {
            "file": account_positions_path.name,
            "sha256": sha256_file(account_positions_path),
            "rows": 0,
        },
        "checks": [{"name": "capture", "passed": True}],
    }
    if include_stock_universe:
        stock_universe_path = root / "GM_STOCK_LIST_REFRESH.csv"
        pd.DataFrame(
            {
                "TS代码": [
                    f"{600000 + index:06d}.SH"
                    for index in range(1000)
                ],
                "股票代码": [
                    f"{600000 + index:06d}"
                    for index in range(1000)
                ],
                "股票名称": [
                    f"TEST{index:04d}"
                    for index in range(1000)
                ],
                "上市状态": "上市",
                "退市日期": "",
                "所属行业": "",
                "gm_symbol": [
                    f"SHSE.{600000 + index:06d}"
                    for index in range(1000)
                ],
                "is_suspended": 0,
            }
        ).to_csv(stock_universe_path, index=False, encoding="utf-8-sig")
        account_snapshot["stock_universe"] = {
            "source": "gm.api.get_instruments",
            "file": stock_universe_path.name,
            "sha256": sha256_file(stock_universe_path),
            "rows": 1000,
            "st_name_rows": 0,
            "delisted_rows": 0,
        }
    account_snapshot_path = root / "account_snapshot.json"
    account_snapshot_path.write_text(json.dumps(account_snapshot), encoding="utf-8")
    transition = {
        "status": "target_transition_audit",
        "passed": True,
        "failed_checks": [],
        "mode": "buffered_previous_target",
        "signal_date": signal_date,
        "trade_date": trade_date,
        "initial_nav": 1_000_000.0,
        "cost": {
            "name": "stress",
            "buy_cost": 0.001,
            "sell_cost": 0.0025,
            "slippage": 0.0005,
            "min_cost": 5.0,
        },
        "limits": {
            "max_two_way_turnover": 0.25,
            "max_estimated_cost_ratio": 0.0015,
            "min_cash_ratio": 0.20,
        },
        "previous_target": {"path": "previous.csv", "sha256": "PREVIOUS"},
        "next_target": {"path": "target.csv", "sha256": "REPLACED_BY_BUILDER_TEST"},
        "position_source": "account_snapshot",
        "account_snapshot": {
            "path": str(account_snapshot_path),
            "sha256": sha256_file(account_snapshot_path),
            "positions_path": str(account_positions_path),
            "positions_sha256": sha256_file(account_positions_path),
            "account_id": ACCOUNT_ID,
            "captured_at": account_snapshot["captured_at"],
            "age_days": 0,
            "nav": 1_000_000.0,
            "available_cash": 1_000_000.0,
            "position_rows": 0,
            "position_symbols": 0,
            "position_shares": 0,
        },
        "missing_prices": [],
        "metrics": {
            "two_way_turnover": 0.05,
            "estimated_cost_ratio": 0.0002,
            "min_cash_ratio": 0.50,
            "min_cash": 500_000.0,
            "lot_violations": 0,
        },
        "checks": [{"name": "example", "passed": True}],
    }
    transition_path = root / "transition.json"
    transition_path.write_text(json.dumps(transition), encoding="utf-8")
    artifacts = {
        name: {"sha256": name.upper()}
        for name in (
            "middle_prediction",
            "outer_prediction",
            "middle_prediction_metadata",
            "outer_prediction_metadata",
            "middle_config",
            "outer_config",
            "forbidden_symbols",
            "group_metadata",
        )
    }
    artifacts["target_transition_audit"] = {
        "sha256": sha256_file(transition_path)
    }
    artifacts["account_snapshot"] = {"sha256": sha256_file(account_snapshot_path)}
    artifacts["account_positions"] = {"sha256": sha256_file(account_positions_path)}
    release_path = root / "release.json"
    release_path.write_text(
        json.dumps(
            {
                "status": "outer_direct_loss5_daily_release_provenance",
                "profile": HISTORICAL_PROFILE,
                "strategy_contract": contract,
                "signal_date": signal_date,
                "trade_date": trade_date,
                "as_of_date": trade_date,
                "historical_smoke": False,
                "prediction_mode": "provided",
                "training_protocol": {
                    "train_end": "2026-01-01",
                    "valid_start": "2026-02-01",
                    "valid_end": "2026-06-01",
                    "test_date": signal_date,
                    "valid_days": 120,
                    "purge_days": 21,
                    "embargo_days": 5,
                },
                "prediction_protocol_passed": True,
                "provider_calendars": {
                    "middle": {"max_date": signal_date, "sha256": "MIDDLE"},
                    "outer": {"max_date": signal_date, "sha256": "OUTER"},
                },
                "artifacts": artifacts,
                "target_transition": transition,
                "historical_evidence_scope": "historical_replay_only",
                "future_holdout_proven": False,
            }
        ),
        encoding="utf-8",
    )
    gate_path = root / "historical_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "passed": True,
                "failed_checks": [],
                "profile": HISTORICAL_PROFILE,
                "checks": [{"name": "historical", "passed": True}],
            }
        ),
        encoding="utf-8",
    )
    return release_path, gate_path, transition_path, account_snapshot_path


def test_stale_candidate_is_blocked_by_dates_and_missing_outer_parity(tmp_path):
    candidate = _copy_candidate(tmp_path)
    report = run_preflight(
        candidate,
        as_of_date=pd.Timestamp("2026-07-15"),
        expected_account_id=ACCOUNT_ID,
    )

    assert report["passed"] is False
    assert len(report["errors"]) >= 4
    assert any("target date is not launch-ready" in value for value in report["errors"])
    assert any("signal date is not fresh" in value for value in report["errors"])
    assert any("signal-to-target lag is invalid" in value for value in report["errors"])
    assert any("selection production parity failed" in value for value in report["errors"])


def test_fresh_target_signal_pair_passes_full_candidate_preflight(tmp_path):
    candidate = _copy_candidate(tmp_path)
    _rewrite_target_dates(
        candidate,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    _make_production_parity_selection(
        candidate,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )

    report = run_preflight(
        candidate,
        as_of_date=pd.Timestamp("2026-07-15"),
        expected_account_id=ACCOUNT_ID,
    )

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["paper_orders_allowed"] is True


def test_trade_date_only_relabel_cannot_bypass_signal_freshness(tmp_path):
    candidate = _copy_candidate(tmp_path)
    _rewrite_target_dates(
        candidate,
        trade_date="2026-07-15",
        signal_date="2026-06-05",
    )
    _make_production_parity_selection(
        candidate,
        trade_date="2026-07-15",
        signal_date="2026-06-05",
    )

    report = run_preflight(
        candidate,
        as_of_date=pd.Timestamp("2026-07-15"),
        expected_account_id=ACCOUNT_ID,
    )

    assert report["passed"] is False
    assert not any("target date is not launch-ready" in value for value in report["errors"])
    assert any("signal date is not fresh" in value for value in report["errors"])
    assert any("signal-to-target lag is invalid" in value for value in report["errors"])


def test_runtime_code_hash_change_is_rejected(tmp_path):
    candidate = _copy_candidate(tmp_path)
    _rewrite_target_dates(
        candidate,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    _make_production_parity_selection(
        candidate,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    main_path = candidate / "main.py"
    main_path.write_text(main_path.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8")

    report = run_preflight(
        candidate,
        as_of_date=pd.Timestamp("2026-07-15"),
        expected_account_id=ACCOUNT_ID,
    )

    assert report["passed"] is False
    assert any("runtime integrity hash mismatch: main.py" in value for value in report["errors"])


def test_outer_prediction_content_must_match_declared_probability(tmp_path):
    candidate = _copy_candidate(tmp_path)
    _rewrite_target_dates(
        candidate,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    _make_production_parity_selection(
        candidate,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    outer_path = candidate / "OUTER_SOURCE.csv"
    outer = pd.read_csv(outer_path)
    outer["outer"] = 0.25
    outer.to_csv(outer_path, index=False)
    selection_path = candidate / "SELECTION_PROVENANCE.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["outer_prediction"]["sha256"] = sha256_file(outer_path)
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = candidate / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target_provenance"]["sha256"]["SELECTION_PROVENANCE.json"] = sha256_file(
        selection_path
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = run_preflight(
        candidate,
        as_of_date=pd.Timestamp("2026-07-15"),
        expected_account_id=ACCOUNT_ID,
    )

    assert report["passed"] is False
    assert any(
        "outer probability does not match" in value for value in report["errors"]
    )


def test_versioned_builder_publishes_only_fresh_preflight_passed_candidate(tmp_path):
    source = _copy_candidate(tmp_path)
    _rewrite_target_dates(
        source,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    selection_manifest = _make_production_parity_selection(
        source,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    output = tmp_path / "paper_20260715"

    report = build_candidate(
        base_candidate=source,
        target_csv=source / "gm_c_baseline_targets.csv",
        target_manifest=source / "gm_c_baseline_targets.manifest.json",
        selection_manifest=selection_manifest,
        forbidden_symbols=source / "gm_c_forbidden_symbols.csv",
        output_dir=output,
        as_of_date=pd.Timestamp("2026-07-15"),
        st_risk_refreshed_at="2026-07-14",
        expected_account_id=ACCOUNT_ID,
    )

    assert report["passed"] is True
    assert report["paper_orders_allowed"] is True
    assert (output / "PREFLIGHT_20260715.json").exists()
    assert (output / "PAPER_READY_20260715.json").exists()
    packaged = json.loads((output / "PREFLIGHT_20260715.json").read_text(encoding="utf-8"))
    assert packaged["passed"] is True


def test_versioned_builder_retains_failed_stale_package_without_marking_ready(tmp_path):
    source = _copy_candidate(tmp_path)
    selection_manifest = _make_production_parity_selection(
        source,
        trade_date="2026-06-12",
        signal_date="2026-06-05",
    )
    output = tmp_path / "stale_candidate"

    with pytest.raises(RuntimeError, match="failed preflight"):
        build_candidate(
            base_candidate=source,
            target_csv=source / "gm_c_baseline_targets.csv",
            target_manifest=source / "gm_c_baseline_targets.manifest.json",
            selection_manifest=selection_manifest,
            forbidden_symbols=source / "gm_c_forbidden_symbols.csv",
            output_dir=output,
            as_of_date=pd.Timestamp("2026-07-15"),
            st_risk_refreshed_at="2026-07-14",
            expected_account_id=ACCOUNT_ID,
        )

    preflight = json.loads((output / "PREFLIGHT_20260715.json").read_text(encoding="utf-8"))
    ready = json.loads((output / "PAPER_READY_20260612.json").read_text(encoding="utf-8"))
    assert preflight["passed"] is False
    assert ready["passed"] is False
    assert ready["paper_orders_allowed"] is False


def test_packaged_release_and_gate_evidence_are_hash_protected(tmp_path):
    source = _copy_candidate(tmp_path)
    _rewrite_target_dates(
        source,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    selection_manifest = _make_production_parity_selection(
        source,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    release, gate, transition, account_snapshot = _make_release_evidence(
        tmp_path,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    transition_payload = json.loads(transition.read_text(encoding="utf-8"))
    transition_payload["next_target"]["sha256"] = sha256_file(
        source / "gm_c_baseline_targets.csv"
    )
    transition.write_text(json.dumps(transition_payload), encoding="utf-8")
    release_payload = json.loads(release.read_text(encoding="utf-8"))
    release_payload["target_transition"] = transition_payload
    release_payload["artifacts"]["target_transition_audit"]["sha256"] = sha256_file(
        transition
    )
    release.write_text(json.dumps(release_payload), encoding="utf-8")
    output = tmp_path / "protected_candidate"
    report = build_candidate(
        base_candidate=source,
        target_csv=source / "gm_c_baseline_targets.csv",
        target_manifest=source / "gm_c_baseline_targets.manifest.json",
        selection_manifest=selection_manifest,
        forbidden_symbols=source / "gm_c_forbidden_symbols.csv",
        output_dir=output,
        as_of_date=pd.Timestamp("2026-07-15"),
        st_risk_refreshed_at="2026-07-14",
        expected_account_id=ACCOUNT_ID,
        release_provenance=release,
        historical_gate_evidence=gate,
        transition_audit=transition,
        account_snapshot=account_snapshot,
    )
    assert report["passed"] is True

    packaged_gate = output / "HISTORICAL_GATE_EVIDENCE.json"
    packaged_gate.write_text(
        packaged_gate.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    tampered = run_preflight(
        output,
        as_of_date=pd.Timestamp("2026-07-15"),
        expected_account_id=ACCOUNT_ID,
    )
    assert tampered["passed"] is False
    assert any(
        "target provenance hash mismatch: HISTORICAL_GATE_EVIDENCE.json" in error
        for error in tampered["errors"]
    )

    shutil.copy2(gate, packaged_gate)
    packaged_transition = output / "TARGET_TRANSITION_AUDIT.json"
    packaged_transition.write_text(
        packaged_transition.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    transition_tampered = run_preflight(
        output,
        as_of_date=pd.Timestamp("2026-07-15"),
        expected_account_id=ACCOUNT_ID,
    )
    assert transition_tampered["passed"] is False
    assert any(
        "target provenance hash mismatch: TARGET_TRANSITION_AUDIT.json" in error
        for error in transition_tampered["errors"]
    )

    shutil.copy2(transition, packaged_transition)
    packaged_account_snapshot = output / "ACCOUNT_SNAPSHOT.json"
    packaged_account_snapshot.write_text(
        packaged_account_snapshot.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    account_tampered = run_preflight(
        output,
        as_of_date=pd.Timestamp("2026-07-15"),
        expected_account_id=ACCOUNT_ID,
    )
    assert account_tampered["passed"] is False
    assert any(
        "target provenance hash mismatch: ACCOUNT_SNAPSHOT.json" in error
        for error in account_tampered["errors"]
    )


def test_v3_runtime_package_is_self_contained_and_stock_universe_is_hash_protected(
    tmp_path,
):
    source = _copy_candidate(tmp_path)
    _rewrite_target_dates(
        source,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    selection_manifest = _make_production_parity_selection(
        source,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
    )
    release, gate, transition, account_snapshot = _make_release_evidence(
        tmp_path,
        trade_date="2026-07-15",
        signal_date="2026-07-14",
        include_stock_universe=True,
    )
    transition_payload = json.loads(transition.read_text(encoding="utf-8"))
    transition_payload["next_target"]["sha256"] = sha256_file(
        source / "gm_c_baseline_targets.csv"
    )
    transition.write_text(json.dumps(transition_payload), encoding="utf-8")
    release_payload = json.loads(release.read_text(encoding="utf-8"))
    release_payload["target_transition"] = transition_payload
    release_payload["artifacts"]["target_transition_audit"]["sha256"] = sha256_file(
        transition
    )
    release_payload["artifacts"]["account_snapshot"]["sha256"] = sha256_file(
        account_snapshot
    )
    release.write_text(json.dumps(release_payload), encoding="utf-8")
    activation_seed = _activation_seed(tmp_path)

    output = tmp_path / "v3_candidate"
    report = build_candidate(
        base_candidate=source,
        target_csv=source / "gm_c_baseline_targets.csv",
        target_manifest=source / "gm_c_baseline_targets.manifest.json",
        selection_manifest=selection_manifest,
        forbidden_symbols=source / "gm_c_forbidden_symbols.csv",
        output_dir=output,
        as_of_date=pd.Timestamp("2026-07-15"),
        st_risk_refreshed_at="2026-07-14",
        expected_account_id=ACCOUNT_ID,
        release_provenance=release,
        historical_gate_evidence=gate,
        transition_audit=transition,
        account_snapshot=account_snapshot,
        activation_registry_seed=activation_seed,
    )

    assert report["passed"] is True
    assert report["runtime_contract"] == {
        "activation_audit_required": True,
        "session_quality_audit_required": True,
        "session_metrics_schema_version": 1,
        "activation_registry_lock_required": True,
        "single_session_per_trade_date_required": True,
    }
    manifest = json.loads(
        (output / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["activation_audit_required"] is True
    assert manifest["session_quality_audit_required"] is True
    assert manifest["session_metrics_schema_version"] == 1
    assert manifest["activation_registry_lock_required"] is True
    assert manifest["single_session_per_trade_date_required"] is True
    seed_contract = manifest["activation_chain_seed"]
    seed_copy = output / seed_contract["file"]
    assert seed_copy.is_file()
    assert seed_contract["sha256"] == sha256_file(seed_copy)
    assert report["activation_chain_seed"]["records"] == 3
    assert report["activation_chain_seed"]["finalized_sessions"] == 1
    for name in (
        "main.py",
        "gm_outer_direct_loss5_market_filtered_paper.py",
        "paper_activation_registry.py",
    ):
        assert (output / name).is_file()
        assert manifest["runtime_integrity"]["sha256"][name] == sha256_file(
            output / name
        )
    stock_name = manifest["target_provenance"]["account_stock_universe_file"]
    assert stock_name == "GM_STOCK_LIST_REFRESH.csv"
    assert manifest["target_provenance"]["sha256"][stock_name] == sha256_file(
        output / stock_name
    )

    seed_copy.write_text(
        seed_copy.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    seed_tampered = run_preflight(
        output,
        as_of_date=pd.Timestamp("2026-07-15"),
        expected_account_id=ACCOUNT_ID,
    )
    assert seed_tampered["passed"] is False
    assert "activation registry seed hash mismatch" in seed_tampered["errors"]
    shutil.copy2(activation_seed, seed_copy)

    stock_path = output / stock_name
    stock_path.write_text(
        stock_path.read_text(encoding="utf-8-sig") + "\n",
        encoding="utf-8-sig",
    )
    tampered = run_preflight(
        output,
        as_of_date=pd.Timestamp("2026-07-15"),
        expected_account_id=ACCOUNT_ID,
    )
    assert tampered["passed"] is False
    assert any(
        f"target provenance hash mismatch: {stock_name}" in error
        for error in tampered["errors"]
    )


def test_v5_session_quality_guard_tamper_is_rejected(tmp_path: Path) -> None:
    candidate = _copy_candidate(tmp_path)
    activation = candidate / "paper_activation_registry.py"
    source = activation.read_text(encoding="utf-8-sig")
    activation.write_text(
        source.replace(
            '"unexplained_target_volume_abs_diff"',
            '"unexplained_volume_abs_diff"',
        ),
        encoding="utf-8",
    )
    manifest_path = candidate / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest["runtime_integrity"]["sha256"][
        "paper_activation_registry.py"
    ] = sha256_file(activation)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_preflight(
        candidate,
        as_of_date=pd.Timestamp("2026-07-31"),
        expected_account_id=ACCOUNT_ID,
    )

    assert report["passed"] is False
    assert any(
        "required session quality guard missing in "
        "paper_activation_registry.py: unexplained_target_volume_abs_diff"
        in error
        for error in report["errors"]
    )


def test_v5_activation_lock_guard_tamper_is_rejected(tmp_path: Path) -> None:
    candidate = _copy_candidate(tmp_path)
    activation = candidate / "paper_activation_registry.py"
    source = activation.read_text(encoding="utf-8-sig")
    activation.write_text(
        source.replace(
            "def _activation_registry_lock",
            "def _disabled_activation_registry_lock",
        ),
        encoding="utf-8",
    )
    manifest_path = candidate / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest["runtime_integrity"]["sha256"][
        "paper_activation_registry.py"
    ] = sha256_file(activation)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_preflight(
        candidate,
        as_of_date=pd.Timestamp("2026-07-31"),
        expected_account_id=ACCOUNT_ID,
    )

    assert report["passed"] is False
    assert any(
        "required activation concurrency guard missing in "
        "paper_activation_registry.py: def _activation_registry_lock"
        in error
        for error in report["errors"]
    )
