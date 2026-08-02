from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .build_inner_multidecision_shadow_candidate import (
        ORDER_CALL_NAMES,
        called_function_names,
        sha256_file,
        target_snapshot,
    )
except ImportError:  # pragma: no cover - direct script execution compatibility
    from build_inner_multidecision_shadow_candidate import (
        ORDER_CALL_NAMES,
        called_function_names,
        sha256_file,
        target_snapshot,
    )


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CANDIDATE = REPO_ROOT / "outputs" / "gmquant_inner_t0_0945_1000_shadow_candidate"
REQUIRED_FILES = {
    "main.py",
    "held_intraday_live_features.py",
    "held_intraday_factor_engineering.py",
    "inner_t0_multidecision_shadow_engine.py",
    "inner_shadow_audit_contract.py",
    "FROZEN_SHADOW_MODELS_MANIFEST.json",
    "gm_c_baseline_targets.csv",
    "gm_c_forbidden_symbols.csv",
    "PACKAGE_MANIFEST.json",
    "README.md",
}
MUTABLE_RUNTIME_FILES = {"gm_c_baseline_targets.csv"}


def _artifact_items(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("source_profile", manifest["source_profile"]),
        ("source_strict_gate", manifest["source_strict_gate"]),
        ("primary_model", manifest["primary_0945"]["model"]),
        ("primary_calibration", manifest["primary_0945"]["calibration"]),
        ("primary_metadata", manifest["primary_0945"]["metadata"]),
        ("secondary_model", manifest["secondary_1000"]["model"]),
        ("secondary_metadata", manifest["secondary_1000"]["metadata"]),
        ("secondary_daily_gate", manifest["secondary_1000"]["daily_gate"]),
    ]


def _compile_source(path: Path) -> None:
    compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")


def run_preflight(
    candidate_dir: Path,
    *,
    as_of_date: pd.Timestamp,
    max_signal_age_days: int = 4,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(name for name in REQUIRED_FILES if not (candidate_dir / name).exists())
    if missing:
        errors.append(f"required files missing: {missing}")
    package_path = candidate_dir / "PACKAGE_MANIFEST.json"
    model_manifest_path = candidate_dir / "FROZEN_SHADOW_MODELS_MANIFEST.json"
    package: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    if package_path.exists():
        try:
            package = json.loads(package_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append(f"package manifest unreadable: {exc}")
    if model_manifest_path.exists():
        try:
            manifest = json.loads(model_manifest_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append(f"model manifest unreadable: {exc}")
    if package:
        for name, item in package.get("files", {}).items():
            if name in MUTABLE_RUNTIME_FILES:
                continue
            path = candidate_dir / name
            if not path.exists():
                errors.append(f"packaged immutable file missing: {name}")
                continue
            actual = sha256_file(path)
            if actual.upper() != str(item.get("sha256", "")).upper():
                errors.append(f"packaged immutable hash mismatch: {name}")
        for key in (
            "actual_submission_api_present",
            "paper_orders_allowed",
            "main_py_integration_allowed",
            "deployment_allowed",
        ):
            expected = False
            if package.get(key) is not expected:
                errors.append(f"package {key} must be false")
    if manifest:
        permissions = manifest.get("permissions", {})
        for key in (
            "paper_orders_allowed",
            "main_py_integration_allowed",
            "deployment_allowed",
        ):
            if permissions.get(key) is not False:
                errors.append(f"model manifest permission {key} must be false")
        if manifest.get("source_strict_gate", {}).get("passed") is not False:
            errors.append("source strict gate must remain failed/research-only")
        if manifest.get("primary_0945", {}).get("decision_time") != "09:45":
            errors.append("primary decision_time is not frozen at 09:45")
        if manifest.get("secondary_1000", {}).get("decision_time") != "10:00":
            errors.append("secondary decision_time is not frozen at 10:00")
        if manifest.get("portfolio", {}).get("max_daily_turnover") != 0.03:
            errors.append("portfolio turnover cap is not frozen at 3%")
        for label, item in _artifact_items(manifest):
            path = candidate_dir / Path(str(item.get("path", ""))).name
            if not path.exists():
                errors.append(f"manifest artifact missing: {label} path={path.name}")
                continue
            actual = sha256_file(path)
            if actual.upper() != str(item.get("sha256", "")).upper():
                errors.append(f"manifest artifact hash mismatch: {label}")
    source_files = [
        candidate_dir / "main.py",
        candidate_dir / "held_intraday_live_features.py",
        candidate_dir / "held_intraday_factor_engineering.py",
        candidate_dir / "inner_t0_multidecision_shadow_engine.py",
        candidate_dir / "inner_shadow_audit_contract.py",
    ]
    compile_results = {}
    for path in source_files:
        if not path.exists():
            continue
        try:
            _compile_source(path)
            compile_results[path.name] = True
        except Exception as exc:
            compile_results[path.name] = False
            errors.append(f"source compile failed: {path.name}: {exc}")
    order_calls: list[str] = []
    main_path = candidate_dir / "main.py"
    if main_path.exists():
        try:
            order_calls = sorted(called_function_names(main_path).intersection(ORDER_CALL_NAMES))
            if order_calls:
                errors.append(f"order submission calls present: {order_calls}")
        except Exception as exc:
            errors.append(f"main AST audit failed: {exc}")
    target_info: dict[str, Any] = {}
    target_path = candidate_dir / "gm_c_baseline_targets.csv"
    if target_path.exists():
        try:
            target_info = target_snapshot(
                target_path,
                as_of_date,
                max_signal_age_days=max_signal_age_days,
            )
            if target_info["rows"] < 20 or target_info["symbols"] < 20:
                errors.append(f"target universe too small: {target_info}")
            if not target_info["passed"]:
                errors.append(
                    "target is not fresh for observation date: "
                    f"source={target_info['source_date']} as_of={target_info['as_of_date']} "
                    f"target_age_days={target_info['age_days']} "
                    f"signal={target_info['signal_date']} "
                    f"signal_age_days={target_info['signal_age_days']}"
                )
        except Exception as exc:
            errors.append(f"target audit failed: {exc}")
    forbidden_info: dict[str, Any] = {}
    forbidden_path = candidate_dir / "gm_c_forbidden_symbols.csv"
    if forbidden_path.exists():
        try:
            forbidden = pd.read_csv(forbidden_path, dtype=str)
            forbidden_info = {
                "rows": int(len(forbidden)),
                "columns": list(forbidden.columns),
                "sha256": sha256_file(forbidden_path),
            }
            if forbidden.empty:
                errors.append("forbidden symbol file is empty")
        except Exception as exc:
            errors.append(f"forbidden symbol audit failed: {exc}")
    passed = not errors
    return {
        "status": "inner_t0_multidecision_shadow_preflight",
        "candidate_dir": str(candidate_dir.resolve()),
        "as_of_date": str(as_of_date.date()),
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "target": target_info,
        "forbidden": forbidden_info,
        "source_compile": compile_results,
        "called_order_submission_apis": order_calls,
        "actual_submission_api_present": bool(order_calls),
        "paper_orders_allowed": False,
        "main_py_integration_allowed": False,
        "deployment_allowed": False,
        "next_action": (
            "run_gmquant_no_order_shadow"
            if passed
            else "replace_only_gm_c_baseline_targets_csv_with_fresh_same_day_target_then_rerun"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--as-of-date", default=str(pd.Timestamp.now().date()))
    parser.add_argument("--output", default="")
    parser.add_argument("--max-signal-age-days", type=int, default=4)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write a failed report without returning a non-zero exit status.",
    )
    args = parser.parse_args()
    candidate = Path(args.candidate_dir).resolve()
    report = run_preflight(
        candidate,
        as_of_date=pd.Timestamp(args.as_of_date).normalize(),
        max_signal_age_days=args.max_signal_age_days,
    )
    output = (
        Path(args.output).resolve()
        if args.output
        else candidate / f"PREFLIGHT_{args.as_of_date.replace('-', '')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[inner shadow preflight] passed={report['passed']} "
        f"errors={len(report['errors'])} target={report['target']} output={output}",
        flush=True,
    )
    for error in report["errors"]:
        print(f"[inner shadow preflight] ERROR {error}", flush=True)
    if not report["passed"] and not args.report_only:
        sys.exit(2)


if __name__ == "__main__":
    main()
