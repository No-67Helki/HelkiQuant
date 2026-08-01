from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from audit_gm_target_csv import audit as audit_target
from paper_activation_registry import read_activation_registry
from preflight_outer_middle_paper_candidate import (
    WRAPPER_NAME,
    run_preflight,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_BASE = (
    HERE / "runtime_templates" / "outer_middle_paper_v5"
)
ACTIVATION_REGISTRY_RELATIVE = Path(
    "outputs/gm_audit_paper_candidate/PAPER_ACTIVATION_REGISTRY.jsonl"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _single_date(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        raise ValueError(f"target missing {column}")
    values = pd.to_datetime(frame[column], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique()
    if len(values) != 1:
        raise ValueError(f"target must contain exactly one {column}: {sorted(values.tolist())}")
    return str(values[0])


def _selection_artifact(selection_manifest: Path, raw: object) -> Path:
    path = Path(str(raw or ""))
    return (path if path.is_absolute() else selection_manifest.parent / path).resolve()


def build_candidate(
    *,
    base_candidate: Path,
    target_csv: Path,
    target_manifest: Path,
    selection_manifest: Path,
    forbidden_symbols: Path,
    output_dir: Path,
    as_of_date: pd.Timestamp,
    st_risk_refreshed_at: str,
    expected_account_id: str,
    max_target_forward_days: int = 0,
    max_signal_age_days: int = 4,
    max_signal_to_target_days: int = 4,
    max_metadata_age_days: int = 7,
    release_provenance: Path | None = None,
    historical_gate_evidence: Path | None = None,
    transition_audit: Path | None = None,
    account_snapshot: Path | None = None,
    activation_registry_seed: Path | None = None,
) -> dict[str, Any]:
    base_candidate = base_candidate.resolve()
    target_csv = target_csv.resolve()
    target_manifest = target_manifest.resolve()
    selection_manifest = selection_manifest.resolve()
    forbidden_symbols = forbidden_symbols.resolve()
    release_provenance = release_provenance.resolve() if release_provenance else None
    historical_gate_evidence = (
        historical_gate_evidence.resolve() if historical_gate_evidence else None
    )
    transition_audit = transition_audit.resolve() if transition_audit else None
    account_snapshot = account_snapshot.resolve() if account_snapshot else None
    activation_registry_seed = (
        activation_registry_seed.resolve() if activation_registry_seed else None
    )
    output_dir = output_dir.resolve()
    base_manifest_path = base_candidate / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json"
    base_manifest = _read_json(base_manifest_path)
    activation_required = base_manifest.get("activation_audit_required") is True
    session_quality_required = (
        base_manifest.get("session_quality_audit_required") is True
    )
    activation_registry_lock_required = (
        base_manifest.get("activation_registry_lock_required") is True
    )
    single_session_per_trade_date_required = (
        base_manifest.get("single_session_per_trade_date_required") is True
    )
    if session_quality_required and not activation_required:
        raise ValueError("session quality audit requires activation audit")
    if (
        activation_registry_lock_required
        or single_session_per_trade_date_required
    ) and not activation_required:
        raise ValueError("activation concurrency guards require activation audit")
    account_id = expected_account_id or str(
        base_manifest.get("paper_account_id", "")
    ).strip()
    if not account_id:
        raise ValueError("expected PAPER account id is required")
    required_inputs = [
        (base_candidate, "base candidate"),
        (target_csv, "target CSV"),
        (target_manifest, "target manifest"),
        (selection_manifest, "selection manifest"),
        (forbidden_symbols, "forbidden symbols"),
    ]
    if release_provenance is not None:
        required_inputs.append((release_provenance, "release provenance"))
    if historical_gate_evidence is not None:
        required_inputs.append((historical_gate_evidence, "historical gate evidence"))
    if transition_audit is not None:
        required_inputs.append((transition_audit, "target transition audit"))
    if account_snapshot is not None:
        required_inputs.append((account_snapshot, "PAPER account snapshot"))
    if activation_registry_seed is not None:
        required_inputs.append(
            (activation_registry_seed, "PAPER activation registry seed")
        )
    for path, label in required_inputs:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    activation_seed_records: list[dict[str, Any]] = []
    if activation_registry_seed is not None:
        activation_seed_records, activation_seed_errors = read_activation_registry(
            activation_registry_seed
        )
        if activation_seed_errors:
            raise ValueError(
                "PAPER activation registry seed integrity failed: "
                + "; ".join(activation_seed_errors)
            )
        seed_accounts = {
            str(record.get("account_id") or "")
            for record in activation_seed_records
            if record.get("account_id")
        }
        if seed_accounts != {account_id}:
            raise ValueError(
                "PAPER activation registry seed account mismatch: "
                f"observed={sorted(seed_accounts)} expected={[account_id]}"
            )
    if output_dir.exists():
        raise FileExistsError(f"versioned output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    for name in ("main.py", WRAPPER_NAME):
        shutil.copy2(base_candidate / name, output_dir / name)
    activation_contract = base_candidate / "paper_activation_registry.py"
    if activation_required:
        if not activation_contract.is_file():
            raise FileNotFoundError(
                f"activation audit contract not found: {activation_contract}"
            )
        shutil.copy2(
            activation_contract,
            output_dir / activation_contract.name,
        )
    packaged_activation_seed = None
    if activation_registry_seed is not None:
        packaged_activation_seed = output_dir / ACTIVATION_REGISTRY_RELATIVE
        packaged_activation_seed.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(activation_registry_seed, packaged_activation_seed)
    shutil.copy2(target_csv, output_dir / "gm_c_baseline_targets.csv")
    shutil.copy2(target_manifest, output_dir / "gm_c_baseline_targets.manifest.json")
    shutil.copy2(forbidden_symbols, output_dir / "gm_c_forbidden_symbols.csv")
    packaged_release_provenance = None
    if release_provenance is not None:
        packaged_release_provenance = output_dir / "RELEASE_PROVENANCE.json"
        shutil.copy2(release_provenance, packaged_release_provenance)
    packaged_historical_gate = None
    if historical_gate_evidence is not None:
        packaged_historical_gate = output_dir / "HISTORICAL_GATE_EVIDENCE.json"
        shutil.copy2(historical_gate_evidence, packaged_historical_gate)
    packaged_transition_audit = None
    if transition_audit is not None:
        packaged_transition_audit = output_dir / "TARGET_TRANSITION_AUDIT.json"
        shutil.copy2(transition_audit, packaged_transition_audit)
    packaged_account_snapshot = None
    packaged_account_positions = None
    packaged_account_stock_universe = None
    if account_snapshot is not None:
        account_snapshot_payload = _read_json(account_snapshot)
        positions_value = account_snapshot_payload.get("positions", {}).get("file")
        positions_source = Path(str(positions_value or ""))
        positions_source = (
            positions_source
            if positions_source.is_absolute()
            else account_snapshot.parent / positions_source
        ).resolve()
        if not positions_source.is_file():
            raise FileNotFoundError(
                f"PAPER account positions not found: {positions_source}"
            )
        packaged_account_snapshot = output_dir / "ACCOUNT_SNAPSHOT.json"
        packaged_account_positions = output_dir / "ACCOUNT_POSITIONS.csv"
        shutil.copy2(account_snapshot, packaged_account_snapshot)
        shutil.copy2(positions_source, packaged_account_positions)
        stock_universe_value = (
            account_snapshot_payload.get("stock_universe") or {}
        ).get("file")
        if stock_universe_value:
            stock_universe_source = Path(str(stock_universe_value))
            stock_universe_source = (
                stock_universe_source
                if stock_universe_source.is_absolute()
                else account_snapshot.parent / stock_universe_source
            ).resolve()
            if not stock_universe_source.is_file():
                raise FileNotFoundError(
                    "PAPER account stock universe not found: "
                    f"{stock_universe_source}"
                )
            expected_stock_hash = str(
                (account_snapshot_payload.get("stock_universe") or {}).get(
                    "sha256",
                    "",
                )
            ).upper()
            if sha256_file(stock_universe_source) != expected_stock_hash:
                raise ValueError("PAPER account stock universe hash mismatch")
            packaged_account_stock_universe = (
                output_dir / stock_universe_source.name
            )
            shutil.copy2(
                stock_universe_source,
                packaged_account_stock_universe,
            )
    for optional in ("README_outer_direct_loss5_v2.md", "README.md"):
        source = base_candidate / optional
        if source.exists():
            shutil.copy2(source, output_dir / optional)

    packaged_target = output_dir / "gm_c_baseline_targets.csv"
    packaged_forbidden = output_dir / "gm_c_forbidden_symbols.csv"
    frame = pd.read_csv(packaged_target)
    trade_date = _single_date(frame, "trade_date")
    signal_date = _single_date(frame, "signal_date")
    target_audit_path = output_dir / "TARGET_AUDIT.json"
    target_audit = audit_target(packaged_target, target_audit_path, packaged_forbidden)

    packaged_target_manifest_path = output_dir / "gm_c_baseline_targets.manifest.json"
    packaged_target_manifest = _read_json(packaged_target_manifest_path)
    packaged_target_manifest["output_csv"] = str(packaged_target)
    packaged_target_manifest["deployment_allowed"] = False
    _write_json(packaged_target_manifest_path, packaged_target_manifest)

    strategy_contract = base_manifest.get("strategy_contract", {})
    source_selection = _read_json(selection_manifest)
    source_selection_target = _selection_artifact(selection_manifest, source_selection.get("target"))
    source_middle_prediction = _selection_artifact(selection_manifest, source_selection.get("prediction"))
    source_outer_prediction = _selection_artifact(
        selection_manifest,
        source_selection.get("outer_overlay", {}).get("prediction"),
    )
    for path, label in (
        (source_selection_target, "source selection target"),
        (source_middle_prediction, "middle prediction"),
        (source_outer_prediction, "outer prediction"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    packaged_source_manifest = output_dir / "SOURCE_SELECTION_MANIFEST.json"
    packaged_source_target = output_dir / "SOURCE_SELECTION_TARGET.csv"
    packaged_middle_prediction = output_dir / "MIDDLE_PREDICTION.csv"
    packaged_outer_prediction = output_dir / "OUTER_PREDICTION.csv"
    shutil.copy2(selection_manifest, packaged_source_manifest)
    shutil.copy2(source_selection_target, packaged_source_target)
    shutil.copy2(source_middle_prediction, packaged_middle_prediction)
    shutil.copy2(source_outer_prediction, packaged_outer_prediction)
    target_weight_sum = float(pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0).sum())
    target_shares_sum = int(pd.to_numeric(frame["target_shares"], errors="coerce").fillna(0).sum())
    symbols = int(frame["symbol"].astype(str).nunique())
    rows = int(len(frame))
    base_manifest.update(
        {
            "status": "outer_direct_loss5_market_filtered_versioned_paper_candidate",
            "candidate_name": f"outer_direct_loss5_market_filtered_{trade_date.replace('-', '')}",
            "created_at": str(pd.Timestamp(as_of_date).date()),
            "deployment_allowed": False,
            "paper_only": True,
            "inner_t0_enabled": False,
            "default_main_replaced": False,
            "paper_account_id": account_id,
        }
    )
    if packaged_activation_seed is not None:
        finalized_seed_records = [
            record
            for record in activation_seed_records
            if record.get("event") == "PAPER_SESSION_FINALIZED"
        ]
        base_manifest["activation_chain_seed"] = {
            "file": ACTIVATION_REGISTRY_RELATIVE.as_posix(),
            "sha256": sha256_file(packaged_activation_seed),
            "records": len(activation_seed_records),
            "finalized_sessions": len(finalized_seed_records),
            "latest_record_hash": (
                str(activation_seed_records[-1].get("record_hash") or "").upper()
                if activation_seed_records
                else ""
            ),
            "account_id": account_id,
            "mutable_after_launch": True,
            "note": (
                "Preflight verifies this seed before launch. The PAPER runtime "
                "then appends new hash-linked events to the packaged copy."
            ),
        }
    else:
        base_manifest.pop("activation_chain_seed", None)
    target_section = base_manifest.setdefault("target", {})
    target_section.update(
        {
            "file": "gm_c_baseline_targets.csv",
            "current_target_date_start": trade_date,
            "current_target_date_end": trade_date,
            "fresh_future_target_required_before_paper": True,
            "fresh_target_generated_at": str(pd.Timestamp(as_of_date).date()),
            "fresh_target_signal_date": signal_date,
            "fresh_target_weight_sum": target_weight_sum,
            "fresh_target_rows": rows,
            "fresh_target_symbols": symbols,
            "fresh_target_shares_sum": target_shares_sum,
            "st_risk_refreshed_at": st_risk_refreshed_at,
            "fresh_target_audit": "TARGET_AUDIT.json",
            "fresh_target_ready_manifest": f"PAPER_READY_{trade_date.replace('-', '')}.json",
            "note": "Versioned target; run only after PREFLIGHT/PAPER_READY passes for its observation date.",
        }
    )
    runtime_hashes = {
        "main.py": sha256_file(output_dir / "main.py"),
        WRAPPER_NAME: sha256_file(output_dir / WRAPPER_NAME),
    }
    if activation_required:
        runtime_hashes["paper_activation_registry.py"] = sha256_file(
            output_dir / "paper_activation_registry.py"
        )
    required_runtime_guards = [
        "trade_date freshness",
        "signal_date provenance and freshness",
        "bound-account position sync fail-closed",
        "dynamic ST/delisting/suspension fail-closed",
        "fill-confirmed holdings and one in-flight target order per symbol",
        "required account never falls back to terminal default",
        "tick execution requires observed price before force window",
        "tick subscription prioritizes reductions before buys",
        "order callbacks and 15:05 schedule atomically persist PAPER audit",
        "hash-linked PAPER activation requires STARTED, READY, and FINALIZED events",
    ]
    if session_quality_required:
        required_runtime_guards.append(
            "final account position resync and session-quality reconciliation"
        )
    if (
        activation_registry_lock_required
        or single_session_per_trade_date_required
    ):
        required_runtime_guards.append(
            "cross-process activation registry lock and one run per account "
            "strategy trade date"
        )
    base_manifest["runtime_integrity"] = {
        "frozen_at": str(pd.Timestamp(as_of_date).date()),
        "sha256": runtime_hashes,
        "required_runtime_guards": required_runtime_guards,
    }
    source_outer = source_selection.get("outer_overlay", {})
    selection_snapshot = {
        "middle_model": strategy_contract.get("middle_model"),
        "outer_model": strategy_contract.get("outer_model"),
        "signal_date": source_selection.get("signal_date"),
        "trade_date": source_selection.get("trade_date"),
        "top_k": source_selection.get("top_k"),
        "rebalance_every": source_selection.get("rebalance_every"),
        "buffer_multiple": source_selection.get("buffer_multiple"),
        "base_risk_budget": source_selection.get(
            "base_risk_budget", source_selection.get("risk_budget")
        ),
        "industry_cap": source_selection.get("industry_cap"),
        "allocation_mode": source_selection.get("allocation", {}).get("mode"),
        "outer_required": source_outer.get("required"),
        "outer_threshold": source_outer.get("threshold"),
        "outer_risk_floor": source_outer.get("risk_floor"),
        "outer_probability": source_outer.get("probability"),
        "outer_triggered": source_outer.get("triggered"),
    }
    exact_contract = {
        "middle_model": strategy_contract.get("middle_model"),
        "outer_model": strategy_contract.get("outer_model"),
        "top_k": strategy_contract.get("top_k"),
        "rebalance_every": strategy_contract.get("rebalance_every"),
        "buffer_multiple": strategy_contract.get("buffer_multiple"),
        "allocation_mode": strategy_contract.get("allocation_mode"),
        "outer_required": strategy_contract.get("outer_prediction_required"),
    }
    float_contract = {
        "base_risk_budget": strategy_contract.get("base_risk_budget"),
        "industry_cap": strategy_contract.get("industry_cap"),
        "outer_threshold": strategy_contract.get("outer_risk_threshold"),
        "outer_risk_floor": strategy_contract.get("outer_risk_floor"),
    }
    parity_mismatches = [
        field
        for field, expected in exact_contract.items()
        if selection_snapshot.get(field) != expected
    ]
    for field, expected in float_contract.items():
        try:
            matches = abs(float(selection_snapshot.get(field)) - float(expected)) <= 1e-12
        except (TypeError, ValueError):
            matches = False
        if not matches:
            parity_mismatches.append(field)
    if selection_snapshot["signal_date"] != signal_date:
        parity_mismatches.append("signal_date")
    if selection_snapshot["trade_date"] != trade_date:
        parity_mismatches.append("trade_date")
    production_parity = not parity_mismatches
    selection_provenance = {
        "status": "paper_selection_provenance",
        "frozen_at": str(pd.Timestamp(as_of_date).date()),
        "source_manifest": {
            "path": packaged_source_manifest.name,
            "sha256": sha256_file(packaged_source_manifest),
        },
        "source_target": {
            "path": packaged_source_target.name,
            "sha256": sha256_file(packaged_source_target),
        },
        "middle_prediction": {
            "path": packaged_middle_prediction.name,
            "sha256": sha256_file(packaged_middle_prediction),
        },
        "outer_prediction": {
            "path": packaged_outer_prediction.name,
            "sha256": sha256_file(packaged_outer_prediction),
        },
        "selection": selection_snapshot,
        "production_parity_passed": production_parity,
        "failure_reason": (
            None
            if production_parity
            else "selection contract mismatch: " + ", ".join(sorted(set(parity_mismatches)))
        ),
    }
    selection_provenance_path = output_dir / "SELECTION_PROVENANCE.json"
    _write_json(selection_provenance_path, selection_provenance)
    provenance_files = {
        "gm_c_baseline_targets.csv": output_dir / "gm_c_baseline_targets.csv",
        "gm_c_baseline_targets.manifest.json": packaged_target_manifest_path,
        "gm_c_forbidden_symbols.csv": output_dir / "gm_c_forbidden_symbols.csv",
        "TARGET_AUDIT.json": target_audit_path,
        "SELECTION_PROVENANCE.json": selection_provenance_path,
    }
    if packaged_release_provenance is not None:
        provenance_files[packaged_release_provenance.name] = packaged_release_provenance
    if packaged_historical_gate is not None:
        provenance_files[packaged_historical_gate.name] = packaged_historical_gate
    if packaged_transition_audit is not None:
        provenance_files[packaged_transition_audit.name] = packaged_transition_audit
    if packaged_account_snapshot is not None:
        provenance_files[packaged_account_snapshot.name] = packaged_account_snapshot
    if packaged_account_positions is not None:
        provenance_files[packaged_account_positions.name] = packaged_account_positions
    if packaged_account_stock_universe is not None:
        provenance_files[
            packaged_account_stock_universe.name
        ] = packaged_account_stock_universe
    base_manifest["target_provenance"] = {
        "frozen_at": str(pd.Timestamp(as_of_date).date()),
        "source_data_end": signal_date,
        "signal_date": signal_date,
        "trade_date": trade_date,
        "source_target_csv": str(target_csv),
        "source_target_manifest": str(target_manifest),
        "source_selection_manifest": str(selection_manifest),
        "source_forbidden_symbols": str(forbidden_symbols),
        "audit_file": "TARGET_AUDIT.json",
        "selection_provenance_file": "SELECTION_PROVENANCE.json",
        "release_provenance_file": (
            packaged_release_provenance.name if packaged_release_provenance else None
        ),
        "historical_gate_evidence_file": (
            packaged_historical_gate.name if packaged_historical_gate else None
        ),
        "transition_audit_file": (
            packaged_transition_audit.name if packaged_transition_audit else None
        ),
        "account_snapshot_file": (
            packaged_account_snapshot.name if packaged_account_snapshot else None
        ),
        "account_positions_file": (
            packaged_account_positions.name if packaged_account_positions else None
        ),
        "account_stock_universe_file": (
            packaged_account_stock_universe.name
            if packaged_account_stock_universe
            else None
        ),
        "sha256": {name: sha256_file(path) for name, path in provenance_files.items()},
    }
    if packaged_release_provenance is not None:
        base_manifest["release_provenance"] = {
            "file": packaged_release_provenance.name,
            "sha256": sha256_file(packaged_release_provenance),
        }
    if packaged_historical_gate is not None:
        historical_gate = _read_json(packaged_historical_gate)
        base_manifest["historical_gate_evidence"] = {
            "file": packaged_historical_gate.name,
            "sha256": sha256_file(packaged_historical_gate),
            "profile": historical_gate.get("profile"),
            "passed": historical_gate.get("passed") is True,
            "evidence_scope": "historical_replay_only",
            "future_holdout_proven": False,
        }
    base_manifest["next_required_before_paper"] = [
        "Run the packaged PRELIGHT report for the actual launch date.",
        "Launch only the packaged PAPER wrapper with the declared account.",
        "Export GmQuant submissions/order_status/summary and compare with local target-volume intent.",
        "Keep inner T+0 research-only until its independent economic gate passes.",
    ]
    candidate_manifest_path = output_dir / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json"
    _write_json(candidate_manifest_path, base_manifest)

    report = run_preflight(
        output_dir,
        as_of_date=pd.Timestamp(as_of_date).normalize(),
        max_target_forward_days=max_target_forward_days,
        max_signal_age_days=max_signal_age_days,
        max_signal_to_target_days=max_signal_to_target_days,
        max_metadata_age_days=max_metadata_age_days,
        expected_account_id=account_id,
    )
    preflight_path = output_dir / f"PREFLIGHT_{pd.Timestamp(as_of_date).strftime('%Y%m%d')}.json"
    _write_json(preflight_path, report)
    build_report = {
        "status": "paper_candidate_ready" if report["passed"] else "paper_candidate_blocked",
        "passed": bool(report["passed"]),
        "candidate_dir": str(output_dir),
        "paper_entrypoint": str(output_dir / WRAPPER_NAME),
        "paper_account_id": account_id,
        "trade_date": trade_date,
        "signal_date": signal_date,
        "target_audit_passed": bool(target_audit.get("passed")),
        "preflight": str(preflight_path),
        "runtime_contract": report.get("runtime_contract", {}),
        "activation_chain_seed": report.get("activation_chain_seed", {}),
        "errors": list(report["errors"]),
        "paper_orders_allowed": bool(report["passed"]),
        "real_money_deployment_allowed": False,
        "inner_t0_enabled": False,
    }
    ready_path = output_dir / f"PAPER_READY_{trade_date.replace('-', '')}.json"
    _write_json(ready_path, build_report)
    if not report["passed"]:
        raise RuntimeError(
            f"versioned candidate failed preflight; package retained for audit: {preflight_path}"
        )
    return build_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-candidate", default=str(DEFAULT_BASE))
    parser.add_argument("--target-csv", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--forbidden-symbols", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--st-risk-refreshed-at", required=True)
    parser.add_argument("--expected-account-id", default="")
    parser.add_argument("--max-target-forward-days", type=int, default=0)
    parser.add_argument("--max-signal-age-days", type=int, default=4)
    parser.add_argument("--max-signal-to-target-days", type=int, default=4)
    parser.add_argument("--max-metadata-age-days", type=int, default=7)
    parser.add_argument("--release-provenance", default="")
    parser.add_argument("--historical-gate-evidence", default="")
    parser.add_argument("--transition-audit", default="")
    parser.add_argument("--account-snapshot", default="")
    parser.add_argument("--activation-registry-seed", default="")
    args = parser.parse_args()
    report = build_candidate(
        base_candidate=Path(args.base_candidate),
        target_csv=Path(args.target_csv),
        target_manifest=Path(args.target_manifest),
        selection_manifest=Path(args.selection_manifest),
        forbidden_symbols=Path(args.forbidden_symbols),
        output_dir=Path(args.output_dir),
        as_of_date=pd.Timestamp(args.as_of_date),
        st_risk_refreshed_at=args.st_risk_refreshed_at,
        expected_account_id=args.expected_account_id,
        max_target_forward_days=args.max_target_forward_days,
        max_signal_age_days=args.max_signal_age_days,
        max_signal_to_target_days=args.max_signal_to_target_days,
        max_metadata_age_days=args.max_metadata_age_days,
        release_provenance=(
            Path(args.release_provenance) if args.release_provenance else None
        ),
        historical_gate_evidence=(
            Path(args.historical_gate_evidence)
            if args.historical_gate_evidence
            else None
        ),
        transition_audit=(Path(args.transition_audit) if args.transition_audit else None),
        account_snapshot=(Path(args.account_snapshot) if args.account_snapshot else None),
        activation_registry_seed=(
            Path(args.activation_registry_seed)
            if args.activation_registry_seed
            else None
        ),
    )
    print(
        f"[outer+middle PAPER build] passed={report['passed']} "
        f"trade_date={report['trade_date']} signal_date={report['signal_date']} "
        f"candidate={report['candidate_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
