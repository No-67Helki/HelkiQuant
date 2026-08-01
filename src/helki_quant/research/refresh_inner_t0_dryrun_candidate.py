from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from preflight_inner_t0_dryrun_candidate import run_preflight


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CANDIDATE_DIR = REPO_ROOT / "outputs" / "gmquant_inner_t0_0935_1445_dryrun_candidate"
DEFAULT_TARGET_SOURCE = REPO_ROOT / "gm_c_baseline_targets.csv"
DEFAULT_FORBIDDEN_SOURCE = REPO_ROOT / "gm_c_forbidden_symbols.csv"
DEFAULT_OUTPUT = HERE / "outputs" / "inner_t0_dryrun_candidate_refresh_20260611.json"


def copy_file(src: Path, dst: Path) -> dict:
    if not src.exists():
        return {"source": str(src.resolve()), "dest": str(dst.resolve()), "copied": False, "reason": "source_missing"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": str(src.resolve()),
        "dest": str(dst.resolve()),
        "copied": True,
        "bytes": dst.stat().st_size,
        "mtime": dst.stat().st_mtime,
    }


def refresh(candidate_dir: Path, target_source: Path, forbidden_source: Path, output_json: Path) -> dict:
    target_copy = copy_file(target_source, candidate_dir / "gm_c_baseline_targets.csv")
    forbidden_copy = copy_file(forbidden_source, candidate_dir / "gm_c_forbidden_symbols.csv")
    preflight_output = output_json.with_name(output_json.stem + "_preflight.json")
    preflight = run_preflight(candidate_dir, preflight_output)
    errors = []
    if not target_copy["copied"]:
        errors.append("target context source missing")
    if not forbidden_copy["copied"]:
        errors.append("forbidden source missing")
    if preflight["status"] != "passed":
        errors.append("preflight failed")
    report = {
        "status": "passed" if not errors else "failed",
        "candidate_dir": str(candidate_dir.resolve()),
        "target_copy": target_copy,
        "forbidden_copy": forbidden_copy,
        "preflight_output": str(preflight_output.resolve()),
        "preflight_status": preflight["status"],
        "errors": errors,
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE_DIR))
    parser.add_argument("--target-source", default=str(DEFAULT_TARGET_SOURCE))
    parser.add_argument("--forbidden-source", default=str(DEFAULT_FORBIDDEN_SOURCE))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = refresh(
        Path(args.candidate_dir).resolve(),
        Path(args.target_source).resolve(),
        Path(args.forbidden_source).resolve(),
        Path(args.output_json).resolve(),
    )
    print(
        "[inner t0 refresh] "
        f"status={report['status']} preflight={report['preflight_status']} output={Path(args.output_json).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
