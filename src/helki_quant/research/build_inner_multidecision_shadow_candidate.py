from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_MODEL_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "inner_0945_1000_shadow_models_20260715"
    / "FROZEN_SHADOW_MODELS_MANIFEST.json"
)
DEFAULT_TARGET = (
    REPO_ROOT
    / "outputs"
    / "gmquant_outer_direct_loss5_v2_market_filtered_paper_candidate"
    / "gm_c_baseline_targets.csv"
)
DEFAULT_FORBIDDEN = (
    REPO_ROOT
    / "outputs"
    / "gmquant_outer_direct_loss5_v2_market_filtered_paper_candidate"
    / "gm_c_forbidden_symbols.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "gmquant_inner_t0_0945_1000_shadow_candidate"
ORDER_CALL_NAMES = {
    "algo_order",
    "order_batch",
    "order_close_all",
    "order_percent",
    "order_target_percent",
    "order_target_value",
    "order_target_volume",
    "order_value",
    "order_volume",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_required(source: Path, destination: Path) -> None:
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(f"required source missing or empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def called_function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def portable_manifest(source: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(json.dumps(source))
    artifact_items = [
        manifest["source_profile"],
        manifest["source_strict_gate"],
        manifest["primary_0945"]["model"],
        manifest["primary_0945"]["calibration"],
        manifest["primary_0945"]["metadata"],
        manifest["secondary_1000"]["model"],
        manifest["secondary_1000"]["metadata"],
        manifest["secondary_1000"]["daily_gate"],
    ]
    for item in artifact_items:
        item["path"] = Path(item["path"]).name
    manifest["package_dir"] = str(output_dir.resolve())
    manifest["runtime_status"] = {
        "pure_shadow_engine_complete": True,
        "gmquant_wrapper_built": True,
        "orders_implemented": False,
    }
    return manifest


def target_snapshot(
    target_path: Path,
    as_of_date: pd.Timestamp,
    *,
    max_signal_age_days: int = 4,
) -> dict[str, Any]:
    frame = pd.read_csv(target_path)
    required = {"instrument", "trade_date", "signal_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"target columns missing: {missing}")
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    if dates.isna().all():
        raise ValueError("target contains no valid trade_date")
    source_date = pd.Timestamp(dates.max()).normalize()
    age_days = int((as_of_date - source_date).days)
    latest = frame[dates == source_date].copy()
    signal_dates = pd.to_datetime(latest["signal_date"], errors="coerce").dt.normalize()
    unique_signal_dates = sorted(
        str(pd.Timestamp(value).date()) for value in signal_dates.dropna().unique()
    )
    signal_date = pd.Timestamp(signal_dates.max()).normalize() if not signal_dates.isna().all() else None
    signal_age_days = int((as_of_date - signal_date).days) if signal_date is not None else None
    signal_passed = bool(
        len(unique_signal_dates) == 1
        and signal_age_days is not None
        and 1 <= signal_age_days <= int(max_signal_age_days)
    )
    return {
        "rows": int(len(frame)),
        "symbols": int(frame["instrument"].astype(str).nunique()),
        "source_date": str(source_date.date()),
        "as_of_date": str(as_of_date.date()),
        "age_days": age_days,
        "max_age_days": 0,
        "target_date_passed": bool(age_days == 0),
        "signal_date": str(signal_date.date()) if signal_date is not None else None,
        "signal_dates": unique_signal_dates,
        "signal_age_days": signal_age_days,
        "max_signal_age_days": int(max_signal_age_days),
        "signal_date_passed": signal_passed,
        "passed": bool(age_days == 0 and signal_passed),
    }


def build_candidate(
    model_manifest_path: Path,
    target_path: Path,
    forbidden_path: Path,
    output_dir: Path,
    output_manifest_path: Path,
    *,
    as_of_date: pd.Timestamp,
    max_signal_age_days: int,
) -> dict[str, Any]:
    source_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8-sig"))
    permissions = source_manifest.get("permissions", {})
    if permissions.get("paper_orders_allowed") is not False:
        raise RuntimeError("source manifest must keep paper_orders_allowed=false")
    if permissions.get("main_py_integration_allowed") is not False:
        raise RuntimeError("source manifest must keep main_py_integration_allowed=false")
    if permissions.get("deployment_allowed") is not False:
        raise RuntimeError("source manifest must keep deployment_allowed=false")
    runtime_source = HERE / "gm_inner_t0_multidecision_shadow_main.py"
    called = called_function_names(runtime_source)
    forbidden_calls = sorted(called.intersection(ORDER_CALL_NAMES))
    if forbidden_calls:
        raise RuntimeError(f"order submission calls found in shadow runtime: {forbidden_calls}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {
        "main.py": runtime_source,
        "held_intraday_live_features.py": HERE / "held_intraday_live_features.py",
        "held_intraday_factor_engineering.py": HERE / "held_intraday_factor_engineering.py",
        "inner_t0_multidecision_shadow_engine.py": HERE / "inner_t0_multidecision_shadow_engine.py",
        "inner_shadow_audit_contract.py": HERE / "inner_shadow_audit_contract.py",
        "gm_c_baseline_targets.csv": target_path,
        "gm_c_forbidden_symbols.csv": forbidden_path,
    }
    artifact_items = [
        source_manifest["source_profile"],
        source_manifest["source_strict_gate"],
        source_manifest["primary_0945"]["model"],
        source_manifest["primary_0945"]["calibration"],
        source_manifest["primary_0945"]["metadata"],
        source_manifest["secondary_1000"]["model"],
        source_manifest["secondary_1000"]["metadata"],
        source_manifest["secondary_1000"]["daily_gate"],
    ]
    for item in artifact_items:
        source = Path(item["path"]).resolve()
        if sha256_file(source).upper() != str(item["sha256"]).upper():
            raise RuntimeError(f"source artifact hash mismatch: {source}")
        sources[source.name] = source
    for name, source in sources.items():
        copy_required(source, output_dir / name)
    packaged_manifest = portable_manifest(source_manifest, output_dir)
    packaged_manifest_path = output_dir / "FROZEN_SHADOW_MODELS_MANIFEST.json"
    packaged_manifest_path.write_text(
        json.dumps(packaged_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    snapshot = target_snapshot(
        target_path,
        as_of_date,
        max_signal_age_days=max_signal_age_days,
    )
    readme_text = f"""# 09:45 + 10:00 held-only T+0 no-order observer

This package is permanently observation-only. It contains no GmQuant order
submission call and does not replace the active outer+middle `main.py`.

Frozen behavior:

- 09:45 sell-first model: isotonic edge >= 0.5%, daily Top-2.
- 10:00 sell-first model: daily Ridge predicted edge > 0, daily Top-2.
- Same date/symbol conflict: retain the 09:45 primary signal.
- Combined maximum: four held symbols; one lot, never above 50% inventory.
- Entry trigger: decision price +0.75%, observed by selected-symbol ticks with
  a 60-second snapshot fallback, ending at 11:00.
- Virtual buyback intent: 14:45-14:50; daily two-sided turnover cap 3% NAV.
- Session evidence is finalized at 14:51 and hash-linked in
  `RUN_REGISTRY.jsonl`; incomplete or selectively omitted runs fail audit.
- ST, delisting-name, suspension, static-forbidden, lookup-missing, stale
  target, and insufficient-feature cases fail closed.

Packaged target source date: `{snapshot['source_date']}`. Observation as-of
date: `{snapshot['as_of_date']}`. Signal date: `{snapshot['signal_date']}`.
Freshness passed: `{snapshot['passed']}`.

Do not run this package until `preflight_inner_multidecision_shadow_candidate.py`
returns `passed=true` for the actual observation date. Keep
`GM_INNER_SHADOW_DRY_RUN=1`; setting it to 0 raises at startup.
"""
    (output_dir / "README.md").write_text(readme_text, encoding="utf-8")
    launcher = output_dir / "run_local_gmquant_shadow.cmd"
    launcher.write_text(
        "@echo off\n"
        "if \"%GM_ACCOUNT_ID%\"==\"\" (echo GM_ACCOUNT_ID is required & exit /b 2)\n"
        "set GM_INNER_SHADOW_MODE=LIVE\n"
        "set GM_INNER_SHADOW_DRY_RUN=1\n"
        "set GM_INNER_SHADOW_MAX_TARGET_AGE_DAYS=0\n"
        f"set GM_INNER_SHADOW_MAX_SIGNAL_AGE_DAYS={int(max_signal_age_days)}\n"
        "python main.py\n"
        "pause\n",
        encoding="ascii",
    )
    package_names = set(sources)
    package_names.update(
        {
            packaged_manifest_path.name,
            "README.md",
            launcher.name,
        }
    )
    package_files = sorted(output_dir / name for name in package_names)
    report = {
        "status": "gmquant_inner_t0_0945_1000_no_order_shadow_candidate_built",
        "candidate_dir": str(output_dir.resolve()),
        "source_model_manifest": str(model_manifest_path.resolve()),
        "target_context_source": str(target_path.resolve()),
        "forbidden_source": str(forbidden_path.resolve()),
        "target_freshness": snapshot,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in package_files
        },
        "account_source": "GM_ACCOUNT_ID",
        "called_order_submission_apis": forbidden_calls,
        "actual_submission_api_present": False,
        "paper_orders_allowed": False,
        "main_py_integration_allowed": False,
        "deployment_allowed": False,
        "runnable_with_packaged_target": bool(snapshot["passed"]),
        "next_gate": "fresh_target_then_gmquant_no_order_shadow_audit",
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-manifest", default=str(DEFAULT_MODEL_MANIFEST))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--forbidden", default=str(DEFAULT_FORBIDDEN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-manifest", default="")
    parser.add_argument("--as-of-date", default=str(pd.Timestamp.now().date()))
    parser.add_argument("--max-signal-age-days", type=int, default=4)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_manifest = (
        Path(args.output_manifest).resolve()
        if args.output_manifest
        else output_dir / "PACKAGE_MANIFEST.json"
    )
    report = build_candidate(
        Path(args.model_manifest).resolve(),
        Path(args.target).resolve(),
        Path(args.forbidden).resolve(),
        output_dir,
        output_manifest,
        as_of_date=pd.Timestamp(args.as_of_date).normalize(),
        max_signal_age_days=args.max_signal_age_days,
    )
    print(
        f"[inner shadow package] files={len(report['files'])} "
        f"target_date={report['target_freshness']['source_date']} "
        f"fresh={report['target_freshness']['passed']} "
        f"runnable={report['runnable_with_packaged_target']} "
        f"candidate={report['candidate_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
