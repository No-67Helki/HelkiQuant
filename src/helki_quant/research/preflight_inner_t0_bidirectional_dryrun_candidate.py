from __future__ import annotations

import argparse
import ast
import json
import py_compile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CANDIDATE = REPO_ROOT / "outputs" / "gmquant_inner_t0_bidirectional_1000_1445_dryrun_candidate"
DEFAULT_OUTPUT = HERE / "outputs" / "inner_t0_bidirectional_dryrun_preflight_20260714.json"
FORBIDDEN_CALLS = {
    "order_volume",
    "order_value",
    "order_percent",
    "order_target_volume",
    "order_target_value",
    "order_target_percent",
    "order_close_all",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def validate_target(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    required = {"trade_date", "instrument", "target_weight", "target_shares", "rank", "middle"}
    missing = sorted(required - set(frame.columns))
    duplicates = int(frame.duplicated(["trade_date", "instrument"]).sum()) if not missing else -1
    return {
        "passed": not missing and len(frame) > 0 and duplicates == 0,
        "rows": int(len(frame)),
        "missing": missing,
        "duplicates": duplicates,
        "trade_dates": sorted(pd.to_datetime(frame["trade_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique()),
    }


def validate_models(candidate: Path) -> dict[str, Any]:
    manifest_path = candidate / "frozen_models_manifest.json"
    manifest = load_json(manifest_path)
    checks = {
        "deployment_disabled": manifest.get("deployment_allowed") is False,
        "runtime_intent_only": manifest.get("runtime_intent_only") is True,
        "directions": set(manifest.get("models", {})) == {"buy_first", "sell_first"},
    }
    details = {}
    for direction, expected in {
        "buy_first": {"threshold": 0.925, "top_n": 2, "trigger": 0.006},
        "sell_first": {"threshold": 0.975, "top_n": 1, "trigger": 0.0075},
    }.items():
        item = manifest.get("models", {}).get(direction, {})
        model_path = candidate / Path(item.get("model_path", "missing")).name
        calibration_path = candidate / Path(item.get("calibration_path", "missing")).name
        meta_path = candidate / Path(item.get("meta_path", "missing")).name
        meta = load_json(meta_path) if meta_path.exists() else {}
        calibration = np.load(calibration_path) if calibration_path.exists() else np.array([])
        direction_checks = {
            "model_exists": model_path.exists() and model_path.stat().st_size > 0,
            "calibration_rows": len(calibration) >= 100,
            "calibration_sorted": bool(len(calibration) and np.all(np.diff(calibration) >= 0)),
            "meta_exists": meta_path.exists(),
            "deployment_disabled": meta.get("deployment_allowed") is False,
            "feature_mode_live": meta.get("feature_mode") == "live",
            "feature_count_115": len(meta.get("feature_cols") or []) == 115,
            "no_unstable_features": not ({"held_age_days", "held_prev_day_ret"} & set(meta.get("feature_cols") or [])),
            "threshold": abs(float(meta.get("score_threshold", -1)) - expected["threshold"]) < 1e-12,
            "top_n": int(meta.get("daily_top_n", -1)) == expected["top_n"],
            "trigger": abs(float(meta.get("trigger_distance", -1)) - expected["trigger"]) < 1e-12,
        }
        details[direction] = {
            "passed": all(direction_checks.values()),
            "checks": direction_checks,
            "validation_metrics": meta.get("validation_metrics"),
            "calibration_metrics": meta.get("calibration_metrics"),
        }
        checks[f"{direction}_passed"] = details[direction]["passed"]
    return {"passed": all(checks.values()), "checks": checks, "details": details}


def preflight(candidate: Path, output_json: Path) -> dict[str, Any]:
    required = [
        "main.py",
        "held_intraday_live_features.py",
        "inner_t0_bidirectional_engine.py",
        "frozen_models_manifest.json",
        "gm_c_baseline_targets.csv",
        "gm_c_forbidden_symbols.csv",
        "README.md",
        "PACKAGE_MANIFEST.json",
    ]
    file_checks = {
        name: {
            "exists": (candidate / name).exists(),
            "bytes": (candidate / name).stat().st_size if (candidate / name).exists() else 0,
        }
        for name in required
    }
    compile_errors = {}
    forbidden_hits = {}
    for path in sorted(candidate.glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors[path.name] = str(exc)
        hits = sorted(called_names(path) & FORBIDDEN_CALLS)
        if hits:
            forbidden_hits[path.name] = hits
    model_check = validate_models(candidate) if (candidate / "frozen_models_manifest.json").exists() else {"passed": False}
    target_check = validate_target(candidate / "gm_c_baseline_targets.csv") if (candidate / "gm_c_baseline_targets.csv").exists() else {"passed": False}
    main_text = (candidate / "main.py").read_text(encoding="utf-8") if (candidate / "main.py").exists() else ""
    source_checks = {
        "dry_run_hard_lock": "permanently intent-only" in main_text,
        "selected_tick_subscription": "SELECTED_TICK_SUBSCRIBE" in main_text,
        "dynamic_risk_filter": "get_instruments" in main_text and "_risk_reason" in main_text,
        "outer_middle_not_imported": (
            "import gm_c_baseline" not in main_text
            and "from gm_c_baseline" not in main_text
        ),
    }
    errors = []
    if not all(row["exists"] and row["bytes"] > 0 for row in file_checks.values()):
        errors.append("required package files missing or empty")
    if compile_errors:
        errors.append("python compilation failed")
    if forbidden_hits:
        errors.append("submission API calls found")
    if not model_check.get("passed"):
        errors.append("frozen model checks failed")
    if not target_check.get("passed"):
        errors.append("target context checks failed")
    if not all(source_checks.values()):
        errors.append("runtime source safety checks failed")
    report = {
        "status": "passed" if not errors else "failed",
        "candidate_dir": str(candidate.resolve()),
        "files": file_checks,
        "compile_errors": compile_errors,
        "submission_api_scan": {"passed": not forbidden_hits, "hits": forbidden_hits},
        "model_check": model_check,
        "target_check": target_check,
        "source_checks": source_checks,
        "errors": errors,
        "deployment_allowed": False,
        "next_gate": "real_gmquant_dryrun_audit",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = preflight(Path(args.candidate_dir).resolve(), Path(args.output_json).resolve())
    print(
        f"[inner bidirectional preflight] status={report['status']} "
        f"errors={len(report['errors'])} output={Path(args.output_json).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
