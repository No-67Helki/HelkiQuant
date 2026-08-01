from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from audit_gm_target_csv import load_forbidden
from paper_activation_registry import read_activation_registry


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CANDIDATE = (
    REPO_ROOT / "outputs" / "gmquant_outer_direct_loss5_v2_market_filtered_paper_candidate"
)
WRAPPER_NAME = "gm_outer_direct_loss5_market_filtered_paper.py"
REQUIRED_FILES = {
    "main.py",
    WRAPPER_NAME,
    "gm_c_baseline_targets.csv",
    "gm_c_forbidden_symbols.csv",
    "gm_c_baseline_targets.manifest.json",
    "PAPER_SIMULATION_CANDIDATE_MANIFEST.json",
    "SELECTION_PROVENANCE.json",
}
REQUIRED_WRAPPER_ENV = {
    "GM_MODE": "LIVE",
    "GM_C_TRADING_ENV": "PAPER",
    "GM_C_REQUIRE_ACCOUNT_ID": "1",
    "GM_C_DYNAMIC_ST_CHECK": "1",
    "GM_C_DYNAMIC_ST_FAIL_CLOSED": "1",
    "GM_C_REQUIRE_SIGNAL_DATE": "1",
    "GM_C_ORDER_STYLE": "VOLUME",
    "GM_C_SYNC_EXISTING_POSITIONS": "1",
}
REQUIRED_MAIN_GUARDS = {
    "_validate_live_signal_context",
    "signal_date is stale",
    "_position_sync_succeeded",
    "unknown account state cannot be treated as an empty portfolio",
    "_refresh_dynamic_market_risk",
    "_pending_target_orders",
    "awaiting fill",
    "pending_target_order_symbols_at_finish",
    "refusing terminal-default-account fallback",
    "await_market_price",
    "side_priority",
    "on_audit_flush_schedule",
    "_atomic_write_frame",
}
HISTORICAL_GATE_PROFILE = "c_outer_loss5_top150_rb20_risk0.60_floor0.30_cap0.30_nost"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _compile_source(path: Path) -> None:
    compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")


def _subscript_key(node: ast.Subscript) -> str | None:
    value = node.value
    if not (
        isinstance(value, ast.Attribute)
        and value.attr == "environ"
        and isinstance(value.value, ast.Name)
        and value.value.id == "os"
    ):
        return None
    key = node.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def wrapper_environment(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript):
            continue
        key = _subscript_key(target)
        if key is None or not isinstance(node.value, ast.Constant):
            continue
        if isinstance(node.value.value, str):
            result[key] = node.value.value
    return result


def _read_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        errors.append(f"{label} unreadable: {exc}")
        return {}


def _artifact_path(candidate_dir: Path, raw: object) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else candidate_dir / path


def _prediction_snapshot(path: Path, signal_date: object, layer: str) -> dict[str, Any]:
    frame = pd.read_csv(path)
    date_column = "datetime" if "datetime" in frame.columns else "trade_date"
    candidates = ("middle", "pred_middle") if layer == "middle" else ("outer", "pred_outer")
    prediction_column = next((name for name in candidates if name in frame.columns), None)
    if date_column not in frame.columns or prediction_column is None:
        raise ValueError(f"{layer} prediction missing date or prediction column")
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    day = frame[dates.eq(pd.Timestamp(signal_date).normalize())].copy()
    values = pd.to_numeric(day[prediction_column], errors="coerce").dropna()
    symbols = int(day["instrument"].astype(str).nunique()) if "instrument" in day.columns else 0
    return {
        "rows": int(len(day)),
        "finite_rows": int(len(values)),
        "symbols": symbols,
        "minimum": float(values.min()) if len(values) else None,
        "maximum": float(values.max()) if len(values) else None,
        "column": prediction_column,
        "passed": bool(len(values) >= 20 and symbols >= 20),
    }


def _audit_selection_provenance(
    candidate_dir: Path,
    manifest: dict[str, Any],
    target_info: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    provenance = manifest.get("target_provenance", {})
    selection_name = str(provenance.get("selection_provenance_file", ""))
    path = candidate_dir / selection_name if selection_name else candidate_dir / "SELECTION_PROVENANCE.json"
    details: list[str] = []
    if not path.exists():
        return {"path": str(path), "passed": False}, ["selection provenance file missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"path": str(path), "passed": False}, [f"selection provenance unreadable: {exc}"]
    selection = payload.get("selection", {})
    verified_artifacts: dict[str, Any] = {}
    for label in ("source_manifest", "source_target", "middle_prediction"):
        item = payload.get(label)
        if not isinstance(item, dict):
            details.append(f"{label} missing")
            continue
        artifact = _artifact_path(candidate_dir, item.get("path"))
        expected = str(item.get("sha256", "")).upper()
        observed = sha256_file(artifact) if artifact.exists() else None
        verified_artifacts[label] = {
            "path": str(artifact),
            "exists": artifact.exists(),
            "sha256": observed,
        }
        if observed is None:
            details.append(f"{label} file missing")
        elif not expected or observed != expected:
            details.append(f"{label} hash mismatch")
        elif label == "middle_prediction":
            try:
                snapshot = _prediction_snapshot(
                    artifact,
                    selection.get("signal_date"),
                    "middle",
                )
                verified_artifacts[label]["snapshot"] = snapshot
                if not snapshot["passed"]:
                    details.append(f"middle prediction coverage failed: {snapshot}")
            except Exception as exc:
                details.append(f"middle prediction content audit failed: {exc}")
    contract = manifest.get("strategy_contract", {})
    outer_item = payload.get("outer_prediction")
    if bool(contract.get("outer_prediction_required", False)):
        if not isinstance(outer_item, dict):
            details.append("required outer prediction evidence missing")
        else:
            outer_path = _artifact_path(candidate_dir, outer_item.get("path"))
            expected = str(outer_item.get("sha256", "")).upper()
            observed = sha256_file(outer_path) if outer_path.exists() else None
            verified_artifacts["outer_prediction"] = {
                "path": str(outer_path),
                "exists": outer_path.exists(),
                "sha256": observed,
            }
            if observed is None:
                details.append("outer prediction file missing")
            elif not expected or observed != expected:
                details.append("outer prediction hash mismatch")
            else:
                try:
                    snapshot = _prediction_snapshot(
                        outer_path,
                        selection.get("signal_date"),
                        "outer",
                    )
                    verified_artifacts["outer_prediction"]["snapshot"] = snapshot
                    if not snapshot["passed"]:
                        details.append(f"outer prediction coverage failed: {snapshot}")
                    probability = selection.get("outer_probability")
                    if probability is None or snapshot["minimum"] is None:
                        details.append("outer probability missing from selection or prediction")
                    elif not (
                        abs(float(probability) - float(snapshot["minimum"])) <= 1e-12
                        and abs(float(probability) - float(snapshot["maximum"])) <= 1e-12
                    ):
                        details.append(
                            "outer probability does not match date-collapsed prediction values"
                        )
                except Exception as exc:
                    details.append(f"outer prediction content audit failed: {exc}")
    exact_fields = {
        "middle_model": contract.get("middle_model"),
        "outer_model": contract.get("outer_model"),
        "top_k": contract.get("top_k"),
        "rebalance_every": contract.get("rebalance_every"),
        "buffer_multiple": contract.get("buffer_multiple"),
        "allocation_mode": contract.get("allocation_mode"),
        "outer_required": contract.get("outer_prediction_required"),
    }
    for field, expected in exact_fields.items():
        if selection.get(field) != expected:
            details.append(
                f"contract {field} mismatch observed={selection.get(field)!r} expected={expected!r}"
            )
    float_fields = {
        "base_risk_budget": contract.get("base_risk_budget"),
        "industry_cap": contract.get("industry_cap"),
        "outer_threshold": contract.get("outer_risk_threshold"),
        "outer_risk_floor": contract.get("outer_risk_floor"),
    }
    for field, expected in float_fields.items():
        try:
            observed_value = float(selection.get(field))
            expected_value = float(expected)
        except (TypeError, ValueError):
            details.append(f"contract {field} is not numeric")
            continue
        if abs(observed_value - expected_value) > 1e-12:
            details.append(
                f"contract {field} mismatch observed={observed_value} expected={expected_value}"
            )
    trade_dates = target_info.get("trade_dates", [])
    signal_dates = target_info.get("signal_dates", [])
    if len(trade_dates) == 1 and selection.get("trade_date") != trade_dates[0]:
        details.append("selection trade_date does not match packaged target")
    if len(signal_dates) == 1 and selection.get("signal_date") != signal_dates[0]:
        details.append("selection signal_date does not match packaged target")
    if payload.get("production_parity_passed") is not True:
        details.append("selection production_parity_passed is not true")
    source_manifest_item = payload.get("source_manifest")
    source_manifest: dict[str, Any] = {}
    if isinstance(source_manifest_item, dict):
        source_path = _artifact_path(candidate_dir, source_manifest_item.get("path"))
        if source_path.exists():
            try:
                source_manifest = json.loads(source_path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                details.append(f"source selection manifest unreadable: {exc}")
    if source_manifest:
        source_outer = source_manifest.get("outer_overlay", {})
        source_checks = {
            "top_k": source_manifest.get("top_k"),
            "rebalance_every": source_manifest.get("rebalance_every"),
            "buffer_multiple": source_manifest.get("buffer_multiple"),
            "allocation_mode": source_manifest.get("allocation", {}).get("mode"),
            "outer_required": source_outer.get("required"),
        }
        for field, observed in source_checks.items():
            if selection.get(field) != observed:
                details.append(
                    f"selection snapshot differs from source manifest: {field} "
                    f"selection={selection.get(field)!r} source={observed!r}"
                )
    info = {
        "path": str(path),
        "sha256": sha256_file(path),
        "selection": selection,
        "verified_artifacts": verified_artifacts,
        "contract_mismatches": details,
        "passed": not details,
    }
    return info, (["selection production parity failed: " + "; ".join(details)] if details else [])


def _audit_release_evidence(
    candidate_dir: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    target_provenance = manifest.get("target_provenance", {})
    release_name = str(target_provenance.get("release_provenance_file") or "")
    gate_name = str(target_provenance.get("historical_gate_evidence_file") or "")
    transition_name = str(target_provenance.get("transition_audit_file") or "")
    account_snapshot_name = str(target_provenance.get("account_snapshot_file") or "")
    account_positions_name = str(target_provenance.get("account_positions_file") or "")
    result: dict[str, Any] = {
        "release_provenance_file": release_name or None,
        "historical_gate_evidence_file": gate_name or None,
        "transition_audit_file": transition_name or None,
        "account_snapshot_file": account_snapshot_name or None,
        "account_positions_file": account_positions_name or None,
    }
    if not release_name and not gate_name:
        if transition_name:
            return result, [
                "target transition audit cannot be declared without release provenance"
            ]
        return result, errors
    if not release_name or not gate_name:
        return result, [
            "release provenance and historical gate evidence must be declared together"
        ]
    if not transition_name:
        errors.append("release provenance requires a target transition audit")

    release_path = candidate_dir / release_name
    gate_path = candidate_dir / gate_name
    if not release_path.exists() or not gate_path.exists():
        return result, errors
    release = _read_json(release_path, errors, "release provenance")
    gate = _read_json(gate_path, errors, "historical gate evidence")
    transition_path = candidate_dir / transition_name if transition_name else None
    transition = (
        _read_json(transition_path, errors, "target transition audit")
        if transition_path is not None and transition_path.exists()
        else {}
    )
    result["release"] = release
    result["historical_gate"] = gate
    result["target_transition"] = transition

    if release.get("status") != "outer_direct_loss5_daily_release_provenance":
        errors.append("release provenance has invalid status")
    if release.get("strategy_contract") != manifest.get("strategy_contract"):
        errors.append("release provenance strategy contract mismatch")
    for field in ("signal_date", "trade_date"):
        if release.get(field) != target_provenance.get(field):
            errors.append(f"release provenance {field} mismatch")
    if release.get("historical_evidence_scope") != "historical_replay_only":
        errors.append("release provenance must label historical evidence scope")
    if release.get("future_holdout_proven") is not False:
        errors.append("release provenance cannot claim a future holdout")
    if release.get("prediction_mode") not in {"trained", "provided"}:
        errors.append("release provenance prediction_mode must be trained or provided")
    historical_smoke = release.get("historical_smoke") is True
    if historical_smoke:
        errors.append("historical engineering smoke cannot authorize PAPER orders")
    elif release.get("historical_smoke") is not False:
        errors.append("release provenance historical_smoke flag is missing")
    if bool(account_snapshot_name) != bool(account_positions_name):
        errors.append(
            "PAPER account snapshot and positions evidence must be declared together"
        )
    if not historical_smoke and not account_snapshot_name:
        errors.append("real PAPER release requires a fresh no-order account snapshot")
    account_snapshot_payload: dict[str, Any] = {}
    account_snapshot_path = (
        candidate_dir / account_snapshot_name if account_snapshot_name else None
    )
    account_positions_path = (
        candidate_dir / account_positions_name if account_positions_name else None
    )
    if (
        account_snapshot_path is not None
        and account_positions_path is not None
        and account_snapshot_path.exists()
        and account_positions_path.exists()
    ):
        account_snapshot_payload = _read_json(
            account_snapshot_path,
            errors,
            "PAPER account snapshot",
        )
        if account_snapshot_payload.get("status") != "gm_paper_account_snapshot":
            errors.append("PAPER account snapshot has invalid status")
        if (
            account_snapshot_payload.get("passed") is not True
            or account_snapshot_payload.get("failed_checks")
        ):
            errors.append("PAPER account snapshot capture checks are not passed")
        if (
            account_snapshot_payload.get("paper_only") is not True
            or account_snapshot_payload.get("no_order") is not True
            or int(account_snapshot_payload.get("orders_submitted", -1)) != 0
        ):
            errors.append("PAPER account snapshot must be verified no-order evidence")
        expected_account = str(manifest.get("paper_account_id") or "").strip()
        if str(account_snapshot_payload.get("account_id") or "").strip() != expected_account:
            errors.append("PAPER account snapshot account id mismatch")
        try:
            captured = pd.Timestamp(
                account_snapshot_payload.get("captured_at")
            ).normalize()
            release_as_of = pd.Timestamp(release.get("as_of_date")).normalize()
            snapshot_age = int((release_as_of - captured).days)
            result["account_snapshot_age_days"] = snapshot_age
            if snapshot_age < 0 or snapshot_age > 1:
                errors.append(
                    f"PAPER account snapshot is not fresh: age_days={snapshot_age} allowed=0..1"
                )
        except (TypeError, ValueError):
            errors.append("PAPER account snapshot captured_at is invalid")
        positions_meta = account_snapshot_payload.get("positions", {})
        expected_positions_hash = str(positions_meta.get("sha256") or "").upper()
        if (
            not expected_positions_hash
            or sha256_file(account_positions_path) != expected_positions_hash
        ):
            errors.append("PAPER account snapshot positions hash mismatch")
        try:
            account_positions = pd.read_csv(account_positions_path)
        except Exception as exc:
            errors.append(f"PAPER account positions CSV cannot be read: {exc}")
            account_positions = pd.DataFrame()
        if int(account_snapshot_payload.get("position_rows", -1)) != len(
            account_positions
        ):
            errors.append("PAPER account snapshot position row count mismatch")
        cash = account_snapshot_payload.get("cash", {})
        try:
            if float(cash.get("nav", 0.0)) <= 0 or float(
                cash.get("available", -1.0)
            ) < 0:
                errors.append("PAPER account snapshot cash/NAV is invalid")
        except (TypeError, ValueError):
            errors.append("PAPER account snapshot cash/NAV is invalid")
    protocol = release.get("training_protocol", {})
    expected_protocol = {
        "valid_days": 120,
        "purge_days": 21,
        "embargo_days": 5,
        "test_date": target_provenance.get("signal_date"),
    }
    if not historical_smoke:
        for field, expected in expected_protocol.items():
            if protocol.get(field) != expected:
                errors.append(
                    f"release provenance frozen training protocol mismatch: {field}"
                )
        if release.get("prediction_protocol_passed") is not True:
            errors.append("forward predictions do not match the frozen purged protocol")
    provider_calendars = release.get("provider_calendars", {})
    for layer in ("middle", "outer"):
        item = provider_calendars.get(layer, {})
        if item.get("max_date") != target_provenance.get("signal_date"):
            errors.append(f"release provenance {layer} provider max_date mismatch")
        if not str(item.get("sha256", "")):
            errors.append(f"release provenance {layer} provider hash missing")
    artifacts = release.get("artifacts", {})
    for name in (
        "middle_prediction",
        "outer_prediction",
        "middle_prediction_metadata",
        "outer_prediction_metadata",
        "middle_config",
        "outer_config",
        "forbidden_symbols",
        "group_metadata",
    ):
        item = artifacts.get(name, {})
        if not str(item.get("sha256", "")):
            errors.append(f"release provenance artifact hash missing: {name}")
    transition_artifact = artifacts.get("target_transition_audit", {})
    transition_artifact_hash = str(transition_artifact.get("sha256", "")).upper()
    if not transition_artifact_hash:
        errors.append("release provenance artifact hash missing: target_transition_audit")
    elif transition_path is not None and transition_path.exists():
        if sha256_file(transition_path) != transition_artifact_hash:
            errors.append("release provenance target transition artifact hash mismatch")
    if account_snapshot_name:
        for artifact_name, packaged_path in (
            ("account_snapshot", account_snapshot_path),
            ("account_positions", account_positions_path),
        ):
            expected_hash = str(
                artifacts.get(artifact_name, {}).get("sha256", "")
            ).upper()
            if not expected_hash:
                errors.append(
                    f"release provenance artifact hash missing: {artifact_name}"
                )
            elif packaged_path is not None and packaged_path.exists():
                if sha256_file(packaged_path) != expected_hash:
                    errors.append(
                        f"release provenance artifact hash mismatch: {artifact_name}"
                    )

    embedded_transition = release.get("target_transition", {})
    if not embedded_transition:
        errors.append("release provenance target transition evidence is missing")
    elif transition and embedded_transition != transition:
        errors.append("release provenance target transition evidence mismatch")
    if transition:
        if transition.get("status") != "target_transition_audit":
            errors.append("target transition audit has invalid status")
        if transition.get("passed") is not True or transition.get("failed_checks"):
            errors.append("target transition stability gate is not passed")
        if transition.get("mode") not in {
            "initial_launch",
            "buffered_previous_target",
        }:
            errors.append("target transition audit has invalid mode")
        if transition.get("signal_date") != target_provenance.get("signal_date"):
            errors.append("target transition signal_date mismatch")
        if transition.get("trade_date") != target_provenance.get("trade_date"):
            errors.append("target transition trade_date mismatch")
        if transition.get("cost", {}).get("name") != "stress":
            errors.append("target transition audit must use stress costs")
        if transition.get("position_source") == "account_snapshot":
            if not account_snapshot_payload:
                errors.append("target transition account snapshot evidence is missing")
            else:
                snapshot_evidence = transition.get("account_snapshot") or {}
                if str(snapshot_evidence.get("sha256", "")).upper() != sha256_file(
                    account_snapshot_path
                ):
                    errors.append("target transition account snapshot hash mismatch")
                if str(
                    snapshot_evidence.get("positions_sha256", "")
                ).upper() != sha256_file(account_positions_path):
                    errors.append("target transition account positions hash mismatch")
                if float(transition.get("initial_nav", 0.0)) != float(
                    account_snapshot_payload.get("cash", {}).get("nav", -1.0)
                ):
                    errors.append("target transition account NAV mismatch")
        else:
            if not historical_smoke:
                errors.append(
                    "real PAPER target transition must use actual account positions"
                )
            if transition.get("initial_nav") != manifest.get(
                "strategy_contract", {}
            ).get("initial_cash"):
                errors.append("target transition initial NAV mismatch")
        next_target_hash = str(
            transition.get("next_target", {}).get("sha256", "")
        ).upper()
        packaged_target = candidate_dir / "gm_c_baseline_targets.csv"
        if packaged_target.exists() and next_target_hash != sha256_file(packaged_target):
            errors.append("target transition next-target hash mismatch")
        if transition.get("mode") == "buffered_previous_target":
            if not str(
                (transition.get("previous_target") or {}).get("sha256", "")
            ):
                errors.append("buffered target transition previous-target hash missing")
        elif transition.get("previous_target") is not None:
            errors.append("initial target transition cannot declare a previous target")
        metrics = transition.get("metrics", {})
        limits = transition.get("limits", {})
        transition_checks = transition.get("checks", [])
        if not transition_checks or any(
            item.get("passed") is not True for item in transition_checks
        ):
            errors.append("target transition audit contains a failed or missing check")
        comparisons = (
            (
                "two_way_turnover",
                "max_two_way_turnover",
                lambda observed, limit: observed <= limit + 1e-12,
            ),
            (
                "estimated_cost_ratio",
                "max_estimated_cost_ratio",
                lambda observed, limit: observed <= limit + 1e-12,
            ),
            (
                "min_cash_ratio",
                "min_cash_ratio",
                lambda observed, limit: observed >= limit - 1e-12,
            ),
        )
        for metric_name, limit_name, predicate in comparisons:
            try:
                observed = float(metrics[metric_name])
                limit = float(limits[limit_name])
            except (KeyError, TypeError, ValueError):
                errors.append(
                    f"target transition metric or limit missing: {metric_name}"
                )
                continue
            if not predicate(observed, limit):
                errors.append(f"target transition limit breached: {metric_name}")
        if transition.get("missing_prices"):
            errors.append("target transition audit has missing current prices")
        if int(metrics.get("lot_violations", -1)) != 0:
            errors.append("target transition audit has target lot violations")
        if float(metrics.get("min_cash", -1.0)) < -1e-6:
            errors.append("target transition audit projects negative cash")

    gate_section = manifest.get("historical_gate_evidence", {})
    if gate_section.get("evidence_scope") != "historical_replay_only":
        errors.append("candidate historical gate scope must remain historical-only")
    if gate_section.get("future_holdout_proven") is not False:
        errors.append("candidate historical gate cannot claim a future holdout")
    if gate.get("passed") is not True or gate.get("failed_checks"):
        errors.append("historical gate evidence is not passed")
    if gate.get("profile") != HISTORICAL_GATE_PROFILE:
        errors.append(
            f"historical gate profile mismatch: {gate.get('profile')!r}"
        )
    checks = gate.get("checks", [])
    if not checks or any(item.get("passed") is not True for item in checks):
        errors.append("historical gate contains a failed or missing check")
    return result, errors


def _audit_target(
    target_path: Path,
    forbidden_path: Path,
    *,
    as_of_date: pd.Timestamp,
    max_target_forward_days: int,
    max_signal_age_days: int,
    max_signal_to_target_days: int,
    errors: list[str],
) -> dict[str, Any]:
    frame = pd.read_csv(target_path, dtype=str)
    required = {
        "trade_date",
        "signal_date",
        "symbol",
        "instrument",
        "target_weight",
        "target_shares",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"target missing required columns: {missing}")
        return {"rows": int(len(frame)), "columns": list(frame.columns)}
    trade_dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    signal_dates = pd.to_datetime(frame["signal_date"], errors="coerce").dt.normalize()
    if trade_dates.isna().any():
        errors.append("target contains invalid trade_date")
    if signal_dates.isna().any():
        errors.append("target contains invalid signal_date")
    frame["symbol"] = frame["symbol"].fillna("").str.strip().str.upper()
    frame["instrument"] = frame["instrument"].fillna("").str.strip().str.upper()
    frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce")
    frame["target_shares"] = pd.to_numeric(frame["target_shares"], errors="coerce")
    unique_targets = sorted(pd.Timestamp(value) for value in trade_dates.dropna().unique())
    unique_signals = sorted(pd.Timestamp(value) for value in signal_dates.dropna().unique())
    if len(unique_targets) != 1:
        errors.append(f"PAPER target must contain exactly one trade_date: {unique_targets}")
    if len(unique_signals) != 1:
        errors.append(f"PAPER target must contain exactly one signal_date: {unique_signals}")
    target_date = unique_targets[0] if len(unique_targets) == 1 else None
    signal_date = unique_signals[0] if len(unique_signals) == 1 else None
    target_forward_days = None
    signal_age_days = None
    signal_to_target_days = None
    as_of_date = pd.Timestamp(as_of_date).normalize()
    if target_date is not None:
        target_forward_days = int((target_date - as_of_date).days)
        if target_forward_days < 0 or target_forward_days > max_target_forward_days:
            errors.append(
                "target date is not launch-ready: "
                f"target={target_date.date()} as_of={as_of_date.date()} "
                f"forward_days={target_forward_days} allowed=0..{max_target_forward_days}"
            )
    if signal_date is not None:
        signal_age_days = int((as_of_date - signal_date).days)
        if signal_age_days < 0 or signal_age_days > max_signal_age_days:
            errors.append(
                "signal date is not fresh: "
                f"signal={signal_date.date()} as_of={as_of_date.date()} "
                f"age_days={signal_age_days} allowed=0..{max_signal_age_days}"
            )
    if target_date is not None and signal_date is not None:
        signal_to_target_days = int((target_date - signal_date).days)
        if signal_to_target_days <= 0 or signal_to_target_days > max_signal_to_target_days:
            errors.append(
                "signal-to-target lag is invalid: "
                f"signal={signal_date.date()} target={target_date.date()} "
                f"lag_days={signal_to_target_days} allowed=1..{max_signal_to_target_days}"
            )
    duplicate_keys = int(
        pd.DataFrame({"trade_date": trade_dates, "symbol": frame["symbol"]})
        .duplicated(["trade_date", "symbol"])
        .sum()
    )
    if duplicate_keys:
        errors.append(f"target has duplicate trade_date/symbol keys: {duplicate_keys}")
    invalid_weights = int(
        (frame["target_weight"].isna() | frame["target_weight"].le(0) | frame["target_weight"].gt(1)).sum()
    )
    if invalid_weights:
        errors.append(f"target has invalid positive weights: {invalid_weights}")
    weight_sum = float(frame["target_weight"].fillna(0).sum())
    if not 0.20 <= weight_sum <= 1.0:
        errors.append(f"target weight sum outside 20%-100%: {weight_sum:.6f}")
    shares = frame["target_shares"]
    invalid_lots = int((shares.isna() | shares.le(0) | shares.mod(100).ne(0)).sum())
    if invalid_lots:
        errors.append(f"target has invalid positive board-lot shares: {invalid_lots}")
    if len(frame) < 20 or frame["symbol"].nunique() < 20:
        errors.append(
            f"target breadth too small: rows={len(frame)} symbols={frame['symbol'].nunique()}"
        )
    forbidden_local, forbidden_gm = load_forbidden(forbidden_path)
    forbidden_mask = frame["instrument"].isin(forbidden_local) | frame["symbol"].isin(forbidden_gm)
    forbidden_hits = sorted(frame.loc[forbidden_mask, "symbol"].unique().tolist())
    if forbidden_hits:
        errors.append(f"target contains forbidden symbols: {forbidden_hits}")
    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "trade_dates": [str(value.date()) for value in unique_targets],
        "signal_dates": [str(value.date()) for value in unique_signals],
        "target_forward_days": target_forward_days,
        "signal_age_days": signal_age_days,
        "signal_to_target_days": signal_to_target_days,
        "weight_sum": weight_sum,
        "shares_sum": int(shares.fillna(0).sum()),
        "duplicate_keys": duplicate_keys,
        "invalid_weights": invalid_weights,
        "invalid_lots": invalid_lots,
        "forbidden_hits": forbidden_hits,
        "sha256": sha256_file(target_path),
    }


def run_preflight(
    candidate_dir: Path,
    *,
    as_of_date: pd.Timestamp,
    max_target_forward_days: int = 0,
    max_signal_age_days: int = 4,
    max_signal_to_target_days: int = 4,
    max_metadata_age_days: int = 7,
    expected_account_id: str = "",
) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(name for name in REQUIRED_FILES if not (candidate_dir / name).exists())
    if missing:
        errors.append(f"required files missing: {missing}")
    manifest_path = candidate_dir / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json"
    manifest = _read_json(manifest_path, errors, "candidate manifest") if manifest_path.exists() else {}
    activation_required = manifest.get("activation_audit_required") is True
    session_quality_required = (
        manifest.get("session_quality_audit_required") is True
    )
    activation_registry_lock_required = (
        manifest.get("activation_registry_lock_required") is True
    )
    single_session_per_trade_date_required = (
        manifest.get("single_session_per_trade_date_required") is True
    )
    if session_quality_required and not activation_required:
        errors.append("session quality audit requires activation audit")
    if (
        activation_registry_lock_required
        or single_session_per_trade_date_required
    ) and not activation_required:
        errors.append("activation concurrency guards require activation audit")
    if activation_required and not (
        candidate_dir / "paper_activation_registry.py"
    ).is_file():
        errors.append("required files missing: ['paper_activation_registry.py']")
    provenance_info: dict[str, Any] = {}
    activation_seed_info: dict[str, Any] = {"present": False}
    if manifest:
        if manifest.get("paper_only") is not True:
            errors.append("candidate manifest paper_only must be true")
        if manifest.get("deployment_allowed") is not False:
            errors.append("candidate manifest deployment_allowed must remain false")
        if manifest.get("inner_t0_enabled") is not False:
            errors.append("candidate manifest inner_t0_enabled must remain false")
        expected_hashes = manifest.get("runtime_integrity", {}).get("sha256", {})
        runtime_names = ["main.py", WRAPPER_NAME]
        if activation_required:
            runtime_names.append("paper_activation_registry.py")
        for name in runtime_names:
            path = candidate_dir / name
            expected = str(expected_hashes.get(name, "")).upper()
            if not expected:
                errors.append(f"runtime integrity hash missing: {name}")
            elif path.exists() and sha256_file(path) != expected:
                errors.append(f"runtime integrity hash mismatch: {name}")
        provenance = manifest.get("target_provenance", {})
        provenance_hashes = provenance.get("sha256", {})
        provenance_info = {
            "source_data_end": provenance.get("source_data_end"),
            "signal_date": provenance.get("signal_date"),
            "trade_date": provenance.get("trade_date"),
            "audit_file": provenance.get("audit_file"),
            "verified_files": {},
        }
        required_provenance_files = {
            "gm_c_baseline_targets.csv",
            "gm_c_baseline_targets.manifest.json",
            "gm_c_forbidden_symbols.csv",
            str(provenance.get("audit_file", "")),
            str(provenance.get("selection_provenance_file", "")),
        }
        for optional_key in (
            "release_provenance_file",
            "historical_gate_evidence_file",
            "transition_audit_file",
            "account_snapshot_file",
            "account_positions_file",
            "account_stock_universe_file",
        ):
            optional_name = str(provenance.get(optional_key) or "")
            if optional_name:
                required_provenance_files.add(optional_name)
        required_provenance_files.discard("")
        base_provenance_names = {
            "gm_c_baseline_targets.csv",
            "gm_c_baseline_targets.manifest.json",
            "gm_c_forbidden_symbols.csv",
            str(provenance.get("audit_file", "")),
            str(provenance.get("selection_provenance_file", "")),
        }
        base_provenance_names.discard("")
        if len(base_provenance_names) != 5:
            errors.append(
                "target provenance must declare target, market-filter manifest, forbidden, "
                "audit, and selection provenance files"
            )
        for name in sorted(required_provenance_files):
            path = candidate_dir / name
            expected = str(provenance_hashes.get(name, "")).upper()
            if not path.exists():
                errors.append(f"target provenance file missing: {name}")
                continue
            actual = sha256_file(path)
            provenance_info["verified_files"][name] = actual
            if not expected:
                errors.append(f"target provenance hash missing: {name}")
            elif actual != expected:
                errors.append(f"target provenance hash mismatch: {name}")
        activation_seed = manifest.get("activation_chain_seed")
        if activation_seed:
            seed_name = str(activation_seed.get("file") or "")
            seed_path = (candidate_dir / seed_name).resolve()
            activation_seed_info = {
                "present": True,
                "file": seed_name,
                "path": str(seed_path),
            }
            try:
                seed_path.relative_to(candidate_dir)
            except ValueError:
                errors.append("activation registry seed escapes candidate directory")
            else:
                if not seed_path.is_file():
                    errors.append(
                        f"activation registry seed missing: {seed_name}"
                    )
                else:
                    actual_seed_hash = sha256_file(seed_path)
                    expected_seed_hash = str(
                        activation_seed.get("sha256") or ""
                    ).upper()
                    activation_seed_info["sha256"] = actual_seed_hash
                    if actual_seed_hash != expected_seed_hash:
                        errors.append("activation registry seed hash mismatch")
                    seed_records, seed_errors = read_activation_registry(seed_path)
                    activation_seed_info["records"] = len(seed_records)
                    activation_seed_info["errors"] = seed_errors
                    if seed_errors:
                        errors.append(
                            "activation registry seed integrity failed: "
                            + "; ".join(seed_errors)
                        )
                    seed_accounts = sorted(
                        {
                            str(record.get("account_id") or "")
                            for record in seed_records
                            if record.get("account_id")
                        }
                    )
                    activation_seed_info["account_ids"] = seed_accounts
                    manifest_account = str(
                        manifest.get("paper_account_id") or ""
                    )
                    if seed_accounts != [manifest_account]:
                        errors.append(
                            "activation registry seed account mismatch: "
                            f"observed={seed_accounts} expected={[manifest_account]}"
                        )
                    finalized_count = sum(
                        record.get("event") == "PAPER_SESSION_FINALIZED"
                        for record in seed_records
                    )
                    activation_seed_info[
                        "finalized_sessions"
                    ] = finalized_count
                    if int(
                        activation_seed.get("records", -1)
                    ) != len(seed_records):
                        errors.append(
                            "activation registry seed record count mismatch"
                        )
                    if int(
                        activation_seed.get("finalized_sessions", -1)
                    ) != finalized_count:
                        errors.append(
                            "activation registry seed finalized count mismatch"
                        )
                    latest_hash = (
                        str(seed_records[-1].get("record_hash") or "").upper()
                        if seed_records
                        else ""
                    )
                    activation_seed_info["latest_record_hash"] = latest_hash
                    if latest_hash != str(
                        activation_seed.get("latest_record_hash") or ""
                    ).upper():
                        errors.append(
                            "activation registry seed latest hash mismatch"
                        )
    compile_results: dict[str, bool] = {}
    compile_names = ["main.py", WRAPPER_NAME]
    if activation_required:
        compile_names.append("paper_activation_registry.py")
    for name in compile_names:
        path = candidate_dir / name
        if not path.exists():
            continue
        try:
            _compile_source(path)
            compile_results[name] = True
        except Exception as exc:
            compile_results[name] = False
            errors.append(f"source compile failed: {name}: {exc}")
    wrapper_env: dict[str, str] = {}
    wrapper_path = candidate_dir / WRAPPER_NAME
    if wrapper_path.exists():
        try:
            wrapper_env = wrapper_environment(wrapper_path)
            for key, expected in REQUIRED_WRAPPER_ENV.items():
                if wrapper_env.get(key) != expected:
                    errors.append(
                        f"wrapper environment mismatch: {key}={wrapper_env.get(key)!r} expected={expected!r}"
                    )
            if activation_required:
                activation_env = {
                    "GM_C_REQUIRE_ACTIVATION_AUDIT": "1",
                }
                for key, expected in activation_env.items():
                    if wrapper_env.get(key) != expected:
                        errors.append(
                            "wrapper activation environment mismatch: "
                            f"{key}={wrapper_env.get(key)!r} expected={expected!r}"
                        )
            account_id = wrapper_env.get("GM_ACCOUNT_ID", "").strip()
            if account_id:
                if expected_account_id and account_id != expected_account_id:
                    errors.append(
                        "wrapper account mismatch: "
                        f"observed={account_id!r} expected={expected_account_id!r}"
                    )
            else:
                wrapper_source = wrapper_path.read_text(encoding="utf-8-sig")
                requires_runtime_account = (
                    'os.environ.get("GM_ACCOUNT_ID"' in wrapper_source
                    and "GM_ACCOUNT_ID must identify the selected simulation account"
                    in wrapper_source
                )
                if not requires_runtime_account:
                    errors.append(
                        "wrapper must either pin GM_ACCOUNT_ID or require it at runtime"
                    )
                manifest_account = str(manifest.get("paper_account_id") or "").strip()
                if expected_account_id and manifest_account != expected_account_id:
                    errors.append(
                        "candidate account mismatch: "
                        f"observed={manifest_account!r} expected={expected_account_id!r}"
                    )
        except Exception as exc:
            errors.append(f"wrapper AST audit failed: {exc}")
    main_path = candidate_dir / "main.py"
    if main_path.exists():
        source = main_path.read_text(encoding="utf-8-sig")
        for guard in REQUIRED_MAIN_GUARDS:
            if guard not in source:
                errors.append(f"required main.py guard missing: {guard}")
        if activation_required:
            for guard in (
                "build_activation_identity",
                "_start_activation_audit",
                "_mark_activation_ready",
                "_finalize_activation",
                "PAPER_ACTIVATION_REGISTRY.jsonl",
            ):
                if guard not in source:
                    errors.append(
                        f"required activation audit guard missing: {guard}"
                    )
        if session_quality_required:
            for guard in (
                "summarize_paper_session",
                "position_sync_succeeded_at_finalize",
                "pending_buy_symbols",
            ):
                if guard not in source:
                    errors.append(
                        f"required session quality guard missing in main.py: {guard}"
                    )
            activation_path = candidate_dir / "paper_activation_registry.py"
            if activation_path.is_file():
                activation_source = activation_path.read_text(encoding="utf-8-sig")
                for guard in (
                    "SESSION_METRICS_SCHEMA_VERSION = 1",
                    "def summarize_paper_session",
                    "unexplained_target_volume_abs_diff",
                ):
                    if guard not in activation_source:
                        errors.append(
                            "required session quality guard missing in "
                            f"paper_activation_registry.py: {guard}"
                        )
        if activation_registry_lock_required or single_session_per_trade_date_required:
            activation_path = candidate_dir / "paper_activation_registry.py"
            if activation_path.is_file():
                activation_source = activation_path.read_text(encoding="utf-8-sig")
                concurrency_guards = []
                if activation_registry_lock_required:
                    concurrency_guards.extend(
                        (
                            "def _activation_registry_lock",
                            "os.O_CREAT | os.O_EXCL | os.O_WRONLY",
                            "os.fsync",
                        )
                    )
                if single_session_per_trade_date_required:
                    concurrency_guards.extend(
                        (
                            "another PAPER run already owns",
                            "record_scope == scope",
                        )
                    )
                for guard in concurrency_guards:
                    if guard not in activation_source:
                        errors.append(
                            "required activation concurrency guard missing in "
                            f"paper_activation_registry.py: {guard}"
                        )
    target_info: dict[str, Any] = {}
    target_manifest: dict[str, Any] = {}
    target_path = candidate_dir / "gm_c_baseline_targets.csv"
    forbidden_path = candidate_dir / "gm_c_forbidden_symbols.csv"
    if target_path.exists() and forbidden_path.exists():
        try:
            target_info = _audit_target(
                target_path,
                forbidden_path,
                as_of_date=as_of_date,
                max_target_forward_days=max_target_forward_days,
                max_signal_age_days=max_signal_age_days,
                max_signal_to_target_days=max_signal_to_target_days,
                errors=errors,
            )
        except Exception as exc:
            errors.append(f"target audit failed: {exc}")
    target_manifest_path = candidate_dir / "gm_c_baseline_targets.manifest.json"
    if target_manifest_path.exists():
        target_manifest = _read_json(target_manifest_path, errors, "target manifest")
    if target_info and target_manifest:
        integer_checks = {
            "output_rows": target_info.get("rows"),
            "output_symbols": target_info.get("symbols"),
        }
        for field, observed in integer_checks.items():
            if int(target_manifest.get(field, -1)) != int(observed):
                errors.append(
                    f"target manifest mismatch: {field}={target_manifest.get(field)!r} "
                    f"csv={observed!r}"
                )
        for field in ("target_weight_sum_min", "target_weight_sum_max"):
            observed = float(target_info.get("weight_sum", 0.0))
            try:
                declared = float(target_manifest.get(field))
            except (TypeError, ValueError):
                errors.append(f"target manifest has invalid {field}: {target_manifest.get(field)!r}")
                continue
            if abs(declared - observed) > 1e-9:
                errors.append(
                    f"target manifest mismatch: {field}={declared} csv={observed}"
                )
        if int(target_manifest.get("blocked_actions", 0)) != 0:
            errors.append("target manifest contains blocked market-state actions")
        if target_manifest.get("deployment_allowed") is not False:
            errors.append("target manifest deployment_allowed must remain false")
    if target_info and manifest:
        candidate_target = manifest.get("target", {})
        trade_dates = target_info.get("trade_dates", [])
        signal_dates = target_info.get("signal_dates", [])
        declared_checks = {
            "current_target_date_start": trade_dates[0] if len(trade_dates) == 1 else None,
            "current_target_date_end": trade_dates[0] if len(trade_dates) == 1 else None,
            "fresh_target_signal_date": signal_dates[0] if len(signal_dates) == 1 else None,
            "fresh_target_rows": target_info.get("rows"),
            "fresh_target_symbols": target_info.get("symbols"),
            "fresh_target_shares_sum": target_info.get("shares_sum"),
        }
        for field, observed in declared_checks.items():
            if candidate_target.get(field) != observed:
                errors.append(
                    f"candidate manifest target mismatch: {field}={candidate_target.get(field)!r} "
                    f"csv={observed!r}"
                )
        try:
            declared_weight = float(candidate_target.get("fresh_target_weight_sum"))
        except (TypeError, ValueError):
            errors.append("candidate manifest has invalid fresh_target_weight_sum")
        else:
            observed_weight = float(target_info.get("weight_sum", 0.0))
            if abs(declared_weight - observed_weight) > 1e-9:
                errors.append(
                    "candidate manifest target mismatch: "
                    f"fresh_target_weight_sum={declared_weight} csv={observed_weight}"
                )
        provenance = manifest.get("target_provenance", {})
        provenance_checks = {
            "trade_date": trade_dates[0] if len(trade_dates) == 1 else None,
            "signal_date": signal_dates[0] if len(signal_dates) == 1 else None,
            "source_data_end": signal_dates[0] if len(signal_dates) == 1 else None,
        }
        for field, observed in provenance_checks.items():
            if provenance.get(field) != observed:
                errors.append(
                    f"target provenance mismatch: {field}={provenance.get(field)!r} csv={observed!r}"
                )
        audit_name = str(provenance.get("audit_file", ""))
        audit_path = candidate_dir / audit_name if audit_name else None
        if audit_path is not None and audit_path.exists():
            audit = _read_json(audit_path, errors, "target provenance audit")
            provenance_info["audit"] = audit
            audit_checks = {
                "rows": target_info.get("rows"),
                "symbols": target_info.get("symbols"),
                "date_start": trade_dates[0] if len(trade_dates) == 1 else None,
                "date_end": trade_dates[0] if len(trade_dates) == 1 else None,
                "forbidden_hits": 0,
            }
            for field, observed in audit_checks.items():
                if audit.get(field) != observed:
                    errors.append(
                        f"target provenance audit mismatch: {field}={audit.get(field)!r} csv={observed!r}"
                    )
            if audit.get("passed") is not True:
                errors.append("target provenance audit must be passed")
            for field in ("target_weight_sum_min", "target_weight_sum_max"):
                try:
                    declared = float(audit.get(field))
                except (TypeError, ValueError):
                    errors.append(f"target provenance audit has invalid {field}")
                    continue
                if abs(declared - float(target_info.get("weight_sum", 0.0))) > 1e-9:
                    errors.append(
                        f"target provenance audit mismatch: {field}={declared} "
                        f"csv={target_info.get('weight_sum')}"
                    )
    metadata_info: dict[str, Any] = {}
    if manifest:
        raw_refresh = manifest.get("target", {}).get("st_risk_refreshed_at")
        refresh = pd.to_datetime(raw_refresh, errors="coerce")
        metadata_info = {"st_risk_refreshed_at": raw_refresh, "max_age_days": max_metadata_age_days}
        if pd.isna(refresh):
            errors.append("candidate manifest has no valid st_risk_refreshed_at")
        else:
            age = int((pd.Timestamp(as_of_date).normalize() - pd.Timestamp(refresh).normalize()).days)
            metadata_info["age_days"] = age
            if age < 0 or age > max_metadata_age_days:
                errors.append(
                    "static ST metadata is stale: "
                    f"refreshed={pd.Timestamp(refresh).date()} as_of={pd.Timestamp(as_of_date).date()} "
                    f"age_days={age} allowed=0..{max_metadata_age_days}"
                )
    selection_info: dict[str, Any] = {}
    if manifest and target_info:
        selection_info, selection_errors = _audit_selection_provenance(
            candidate_dir,
            manifest,
            target_info,
        )
        errors.extend(selection_errors)
    release_evidence: dict[str, Any] = {}
    if manifest:
        release_evidence, release_errors = _audit_release_evidence(
            candidate_dir,
            manifest,
        )
        errors.extend(release_errors)
    passed = not errors
    return {
        "status": "outer_middle_paper_candidate_preflight",
        "candidate_dir": str(candidate_dir),
        "as_of_date": str(pd.Timestamp(as_of_date).date()),
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "target": target_info,
        "target_manifest": target_manifest,
        "target_provenance": provenance_info,
        "selection_provenance": selection_info,
        "release_evidence": release_evidence,
        "activation_chain_seed": activation_seed_info,
        "metadata": metadata_info,
        "runtime_contract": {
            "activation_audit_required": activation_required,
            "session_quality_audit_required": session_quality_required,
            "session_metrics_schema_version": manifest.get(
                "session_metrics_schema_version"
            ),
            "activation_registry_lock_required": (
                activation_registry_lock_required
            ),
            "single_session_per_trade_date_required": (
                single_session_per_trade_date_required
            ),
        },
        "wrapper_environment": wrapper_env,
        "source_compile": compile_results,
        "paper_orders_allowed": passed,
        "real_money_deployment_allowed": False,
        "inner_t0_enabled": False,
        "next_action": "run_paper_wrapper" if passed else "regenerate_fresh_target_and_metadata_then_rerun",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--as-of-date", default=str(pd.Timestamp.now().date()))
    parser.add_argument("--output", default="")
    parser.add_argument("--max-target-forward-days", type=int, default=0)
    parser.add_argument("--max-signal-age-days", type=int, default=4)
    parser.add_argument("--max-signal-to-target-days", type=int, default=4)
    parser.add_argument("--max-metadata-age-days", type=int, default=7)
    parser.add_argument("--expected-account-id", default="")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    candidate = Path(args.candidate_dir).resolve()
    report = run_preflight(
        candidate,
        as_of_date=pd.Timestamp(args.as_of_date).normalize(),
        max_target_forward_days=args.max_target_forward_days,
        max_signal_age_days=args.max_signal_age_days,
        max_signal_to_target_days=args.max_signal_to_target_days,
        max_metadata_age_days=args.max_metadata_age_days,
        expected_account_id=args.expected_account_id,
    )
    output = (
        Path(args.output).resolve()
        if args.output
        else candidate / f"PREFLIGHT_{args.as_of_date.replace('-', '')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[outer+middle PAPER preflight] passed={report['passed']} "
        f"errors={len(report['errors'])} target={report['target']} output={output}",
        flush=True,
    )
    for error in report["errors"]:
        print(f"[outer+middle PAPER preflight] ERROR {error}", flush=True)
    if not report["passed"] and not args.report_only:
        sys.exit(2)


if __name__ == "__main__":
    main()
