from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from build_inner_multidecision_shadow_candidate import sha256_file, target_snapshot
from preflight_inner_multidecision_shadow_candidate import (
    DEFAULT_CANDIDATE,
    run_preflight,
)


def refresh_target(
    source_target: Path,
    candidate_dir: Path,
    *,
    as_of_date: pd.Timestamp,
    max_signal_age_days: int = 4,
    output_path: Path | None = None,
) -> dict:
    if not source_target.exists() or source_target.stat().st_size == 0:
        raise FileNotFoundError(f"source target missing or empty: {source_target}")
    source_audit = target_snapshot(
        source_target,
        as_of_date,
        max_signal_age_days=max_signal_age_days,
    )
    if source_audit["rows"] < 20 or source_audit["symbols"] < 20:
        raise RuntimeError(f"source target universe too small: {source_audit}")
    if not source_audit["passed"]:
        raise RuntimeError(
            "source target freshness failed before copy: "
            f"source_date={source_audit['source_date']} "
            f"signal_date={source_audit['signal_date']} "
            f"as_of={source_audit['as_of_date']}"
        )
    destination = candidate_dir / "gm_c_baseline_targets.csv"
    if not candidate_dir.exists() or not (candidate_dir / "PACKAGE_MANIFEST.json").exists():
        raise FileNotFoundError(f"candidate package is incomplete: {candidate_dir}")
    before_preflight = run_preflight(
        candidate_dir,
        as_of_date=as_of_date,
        max_signal_age_days=max_signal_age_days,
    )
    non_target_errors = [
        error
        for error in before_preflight["errors"]
        if not error.startswith("target is not fresh for observation date:")
    ]
    if non_target_errors:
        raise RuntimeError(
            f"candidate has non-target preflight errors before refresh: {non_target_errors}"
        )
    previous_hash = sha256_file(destination) if destination.exists() else None
    pending = destination.with_name(f"{destination.name}.pending")
    shutil.copy2(source_target, pending)
    pending_hash = sha256_file(pending)
    source_hash = sha256_file(source_target)
    if pending_hash != source_hash:
        raise RuntimeError("staged target hash does not match source")
    os.replace(pending, destination)
    copied_hash = sha256_file(destination)
    if copied_hash != source_hash:
        raise RuntimeError("copied target hash does not match source")
    preflight = run_preflight(
        candidate_dir,
        as_of_date=as_of_date,
        max_signal_age_days=max_signal_age_days,
    )
    if not preflight["passed"]:
        raise RuntimeError(
            f"candidate preflight failed after target copy: {preflight['errors']}"
        )
    report = {
        "status": "inner_t0_multidecision_shadow_target_refreshed",
        "candidate_dir": str(candidate_dir.resolve()),
        "source_target": str(source_target.resolve()),
        "destination_target": str(destination.resolve()),
        "as_of_date": str(as_of_date.date()),
        "source_audit": source_audit,
        "before_preflight": before_preflight,
        "previous_target_sha256": previous_hash,
        "source_target_sha256": source_hash,
        "copied_target_sha256": copied_hash,
        "preflight": preflight,
        "paper_orders_allowed": False,
        "main_py_integration_allowed": False,
        "deployment_allowed": False,
        "next_action": "run_candidate_main_in_gmquant_no_order_shadow_mode",
    }
    output = output_path or (
        candidate_dir / f"TARGET_REFRESH_{as_of_date.strftime('%Y%m%d')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-target", required=True)
    parser.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--as-of-date", default=str(pd.Timestamp.now().date()))
    parser.add_argument("--max-signal-age-days", type=int, default=4)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = refresh_target(
        Path(args.source_target).resolve(),
        Path(args.candidate_dir).resolve(),
        as_of_date=pd.Timestamp(args.as_of_date).normalize(),
        max_signal_age_days=args.max_signal_age_days,
        output_path=Path(args.output).resolve() if args.output else None,
    )
    print(
        f"[inner shadow refresh] passed={report['preflight']['passed']} "
        f"target={report['source_audit']['source_date']} "
        f"signal={report['source_audit']['signal_date']} "
        f"candidate={report['candidate_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
