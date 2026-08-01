from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_MODEL_DIR = HERE / "outputs" / "held_intraday_frozen_bidirectional_1000_1445_20260714"
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "gmquant_inner_t0_bidirectional_1000_1445_dryrun_candidate"


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


def portable_model_manifest(source: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(json.dumps(source))
    for item in manifest["models"].values():
        for key in ("model_path", "calibration_path", "calibration_scores_path", "meta_path"):
            if item.get(key):
                item[key] = Path(item[key]).name
    manifest["package_dir"] = str(output_dir.resolve())
    manifest["runtime_intent_only"] = True
    manifest["deployment_allowed"] = False
    return manifest


def build_candidate(
    model_dir: Path,
    target_path: Path,
    forbidden_path: Path,
    output_dir: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    source_manifest_path = model_dir / "frozen_models_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("deployment_allowed") is not False:
        raise RuntimeError("frozen model manifest must keep deployment_allowed=false")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "main.py": HERE / "gm_inner_t0_bidirectional_dryrun_main.py",
        "held_intraday_live_features.py": HERE / "held_intraday_live_features.py",
        "held_intraday_factor_engineering.py": HERE / "held_intraday_factor_engineering.py",
        "inner_t0_bidirectional_engine.py": HERE / "inner_t0_bidirectional_engine.py",
        "gm_c_baseline_targets.csv": target_path,
        "gm_c_forbidden_symbols.csv": forbidden_path,
    }
    for direction, item in source_manifest["models"].items():
        for key in ("model_path", "calibration_path", "calibration_scores_path", "meta_path"):
            source = Path(item[key]).resolve()
            sources[source.name] = source
    for name, source in sources.items():
        copy_required(source, output_dir / name)
    packaged_models = portable_model_manifest(source_manifest, output_dir)
    packaged_model_path = output_dir / "frozen_models_manifest.json"
    packaged_model_path.write_text(
        json.dumps(packaged_models, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = output_dir / "README.md"
    readme.write_text(
        """# Held-only bidirectional T+0 dry-run candidate

This is an isolated intent-audit package. It does not replace the active
outer+middle PAPER strategy and contains no order submission API call.

Frozen profile:

- Decision: 10:00 using only bars ending at or before 10:00.
- Buy-first: percentile >= 0.925, daily Top-2, entry trigger -0.60%.
- Sell-first: percentile >= 0.975, daily Top-1, entry trigger +0.75%.
- Same-symbol opposite signals: drop both.
- Size: one whole lot, never above 50% of the held inventory.
- Tick subscription: selected holdings only, at most three symbols.
- Trigger window: 10:00-11:00; inventory-restoring exit: 14:45-14:50.
- Daily two-sided turnover cap: 3% of NAV.
- ST, delisting-name, suspended, static-forbidden, and lookup-missing symbols
  are blocked before feature construction.

Run `main.py` from GmQuant in simulation mode after setting `GM_ACCOUNT_ID`.
Keep `GM_INNER_T0_DRY_RUN=1`; setting it to 0
causes a hard startup failure.

Audit output is written under the sibling directory
`gm_inner_t0_bidirectional_dryrun_audit/<run_id>/`. A real platform audit and
strict paper gate are required before any integration is considered.
""",
        encoding="utf-8",
    )
    launcher = output_dir / "run_local_gmquant_dryrun.cmd"
    launcher.write_text(
        "@echo off\n"
        "if \"%GM_ACCOUNT_ID%\"==\"\" (echo GM_ACCOUNT_ID is required & exit /b 2)\n"
        "set GM_INNER_T0_MODE=LIVE\n"
        "set GM_INNER_T0_DRY_RUN=1\n"
        "python main.py\n"
        "pause\n",
        encoding="ascii",
    )
    package_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    report = {
        "status": "gmquant_inner_t0_bidirectional_dryrun_candidate_built",
        "candidate_dir": str(output_dir.resolve()),
        "source_model_manifest": str(source_manifest_path.resolve()),
        "target_context_source": str(target_path.resolve()),
        "forbidden_source": str(forbidden_path.resolve()),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in package_files
        },
        "account_source": "GM_ACCOUNT_ID",
        "actual_submission_api_present": False,
        "deployment_allowed": False,
        "next_gate": "real_gmquant_dryrun_audit_then_local_compare",
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if output_manifest.parent.resolve() == output_dir.resolve():
        report["files"][output_manifest.name] = {
            "bytes": output_manifest.stat().st_size,
            "sha256": sha256_file(output_manifest),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--forbidden", default=str(DEFAULT_FORBIDDEN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-manifest", default="")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_manifest = (
        Path(args.output_manifest).resolve()
        if args.output_manifest
        else output_dir / "PACKAGE_MANIFEST.json"
    )
    report = build_candidate(
        Path(args.model_dir).resolve(),
        Path(args.target).resolve(),
        Path(args.forbidden).resolve(),
        output_dir,
        output_manifest,
    )
    print(
        f"[inner package] files={len(report['files'])} candidate={report['candidate_dir']} "
        f"deployment_allowed={report['deployment_allowed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
