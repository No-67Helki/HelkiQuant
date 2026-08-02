from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .build_inner_multidecision_shadow_candidate import (
        DEFAULT_MODEL_MANIFEST,
        build_candidate as build_inner_candidate,
        sha256_file,
    )
    from .preflight_inner_multidecision_shadow_candidate import run_preflight
except ImportError:  # pragma: no cover - direct script execution compatibility
    from build_inner_multidecision_shadow_candidate import (
        DEFAULT_MODEL_MANIFEST,
        build_candidate as build_inner_candidate,
        sha256_file,
    )
    from preflight_inner_multidecision_shadow_candidate import run_preflight


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _single_date(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        raise ValueError(f"outer target missing {column}")
    values = pd.to_datetime(frame[column], errors="coerce").dropna().dt.strftime(
        "%Y-%m-%d"
    ).unique()
    if len(values) != 1:
        raise ValueError(f"outer target must contain one {column}: {list(values)}")
    return str(values[0])


def build_synchronized_shadow(
    *,
    outer_package: Path,
    output_dir: Path,
    as_of_date: pd.Timestamp,
    model_manifest: Path = DEFAULT_MODEL_MANIFEST,
    historical_smoke: bool = False,
    max_signal_age_days: int = 4,
) -> dict[str, Any]:
    outer_package = outer_package.resolve()
    output_dir = output_dir.resolve()
    model_manifest = model_manifest.resolve()
    if output_dir.exists():
        raise FileExistsError(f"versioned inner shadow output already exists: {output_dir}")
    required = {
        "gm_c_baseline_targets.csv": outer_package / "gm_c_baseline_targets.csv",
        "gm_c_forbidden_symbols.csv": outer_package / "gm_c_forbidden_symbols.csv",
        "PAPER_SIMULATION_CANDIDATE_MANIFEST.json": (
            outer_package / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json"
        ),
        "RELEASE_PROVENANCE.json": outer_package / "RELEASE_PROVENANCE.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"outer package evidence missing: {missing}")
    if not model_manifest.is_file():
        raise FileNotFoundError(f"frozen inner model manifest not found: {model_manifest}")

    target_path = required["gm_c_baseline_targets.csv"]
    forbidden_path = required["gm_c_forbidden_symbols.csv"]
    target = pd.read_csv(target_path)
    trade_date = _single_date(target, "trade_date")
    signal_date = _single_date(target, "signal_date")
    as_of = pd.Timestamp(as_of_date).normalize()
    if pd.Timestamp(trade_date).normalize() != as_of:
        raise ValueError(
            f"outer target date must equal shadow observation date: {trade_date} != {as_of.date()}"
        )
    signal_age = int((as_of - pd.Timestamp(signal_date).normalize()).days)
    if signal_age < 1 or signal_age > max_signal_age_days:
        raise ValueError(
            f"outer signal is not fresh for inner shadow: age_days={signal_age}"
        )

    release = _read_json(required["RELEASE_PROVENANCE.json"])
    if release.get("signal_date") != signal_date or release.get("trade_date") != trade_date:
        raise ValueError("outer release provenance date mismatch")
    if release.get("inner_t0_enabled") is not False:
        raise ValueError("outer release must keep inner order integration disabled")
    if release.get("historical_smoke") is not historical_smoke:
        raise ValueError("historical_smoke flag does not match outer release evidence")

    ready_path = outer_package / f"PAPER_READY_{trade_date.replace('-', '')}.json"
    preflight_path = outer_package / f"PREFLIGHT_{as_of.strftime('%Y%m%d')}.json"
    if not ready_path.is_file() or not preflight_path.is_file():
        raise FileNotFoundError("outer PAPER_READY/PREFLIGHT evidence is missing")
    outer_ready = _read_json(ready_path)
    outer_preflight = _read_json(preflight_path)
    if historical_smoke:
        if outer_ready.get("paper_orders_allowed") is not False:
            raise ValueError("historical outer package cannot allow PAPER orders")
        if not any(
            "historical engineering smoke" in str(error)
            for error in outer_preflight.get("errors", [])
        ):
            raise ValueError("historical outer preflight lacks the smoke-only block")
    else:
        if outer_ready.get("passed") is not True or outer_ready.get(
            "paper_orders_allowed"
        ) is not True:
            raise ValueError("real outer package is not PAPER_READY")
        if outer_preflight.get("passed") is not True:
            raise ValueError("real outer package preflight is not passed")

    package_manifest_path = output_dir / "PACKAGE_MANIFEST.json"
    inner_build = build_inner_candidate(
        model_manifest,
        target_path,
        forbidden_path,
        output_dir,
        package_manifest_path,
        as_of_date=as_of,
        max_signal_age_days=max_signal_age_days,
    )
    inner_preflight = run_preflight(
        output_dir,
        as_of_date=as_of,
        max_signal_age_days=max_signal_age_days,
    )
    inner_preflight_path = output_dir / f"PREFLIGHT_{as_of.strftime('%Y%m%d')}.json"
    _write_json(inner_preflight_path, inner_preflight)
    if not inner_preflight.get("passed"):
        raise RuntimeError(
            f"synchronized inner shadow preflight failed: {inner_preflight['errors']}"
        )

    sync = {
        "status": "outer_middle_inner_shadow_synchronized",
        "historical_smoke": bool(historical_smoke),
        "as_of_date": str(as_of.date()),
        "signal_date": signal_date,
        "trade_date": trade_date,
        "outer_package": str(outer_package),
        "inner_shadow_package": str(output_dir),
        "source_evidence": {
            "outer_target": {
                "path": str(target_path),
                "sha256": sha256_file(target_path),
            },
            "outer_forbidden": {
                "path": str(forbidden_path),
                "sha256": sha256_file(forbidden_path),
            },
            "outer_release": {
                "path": str(required["RELEASE_PROVENANCE.json"]),
                "sha256": sha256_file(required["RELEASE_PROVENANCE.json"]),
            },
            "outer_ready": {
                "path": str(ready_path),
                "sha256": sha256_file(ready_path),
            },
            "outer_preflight": {
                "path": str(preflight_path),
                "sha256": sha256_file(preflight_path),
            },
            "inner_model_manifest": {
                "path": str(model_manifest),
                "sha256": sha256_file(model_manifest),
            },
        },
        "inner_build_manifest": str(package_manifest_path),
        "inner_preflight": str(inner_preflight_path),
        "inner_preflight_passed": True,
        "actual_submission_api_present": False,
        "paper_orders_allowed": False,
        "main_py_integration_allowed": False,
        "deployment_allowed": False,
        "runnable_no_order_shadow": bool(not historical_smoke),
        "next_action": (
            "engineering_chain_verified_only"
            if historical_smoke
            else "run_versioned_inner_shadow_in_gmquant_and_export_audit"
        ),
        "inner_build": inner_build,
    }
    sync_path = output_dir / "SYNC_PROVENANCE.json"
    _write_json(sync_path, sync)
    return sync


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a versioned no-order inner observer from one frozen outer+middle package."
        )
    )
    parser.add_argument("--outer-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--max-signal-age-days", type=int, default=4)
    parser.add_argument("--historical-smoke", action="store_true")
    args = parser.parse_args()
    report = build_synchronized_shadow(
        outer_package=args.outer_package,
        output_dir=args.output_dir,
        as_of_date=pd.Timestamp(args.as_of_date),
        model_manifest=args.model_manifest,
        historical_smoke=args.historical_smoke,
        max_signal_age_days=args.max_signal_age_days,
    )
    print(
        "[synchronized inner shadow] "
        f"historical_smoke={report['historical_smoke']} "
        f"preflight={report['inner_preflight_passed']} "
        f"runnable={report['runnable_no_order_shadow']} "
        f"package={report['inner_shadow_package']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
