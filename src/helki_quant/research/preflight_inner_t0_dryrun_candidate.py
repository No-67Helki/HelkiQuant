from __future__ import annotations

import argparse
import json
import py_compile
import re
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CANDIDATE_DIR = REPO_ROOT / "outputs" / "gmquant_inner_t0_0935_1445_dryrun_candidate"
DEFAULT_OUTPUT = HERE / "outputs" / "inner_t0_dryrun_candidate_preflight_20260611.json"
FORBIDDEN_ORDER_PATTERNS = [
    r"\border_volume\b",
    r"\border_value\b",
    r"\border_target\b",
    r"\border_target_volume\b",
    r"\border_target_value\b",
    r"\border_percent\b",
    r"\border_close_all\b",
]
REQUIRED_TARGET_COLUMNS = {
    "trade_date",
    "symbol",
    "instrument",
    "rank",
    "middle",
    "target_weight",
    "target_shares",
}
LIVE_UNSTABLE_FEATURES = {"held_age_days", "held_prev_day_ret"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_file(path: Path, min_bytes: int = 1) -> dict[str, Any]:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "bytes": size,
        "passed": exists and size >= min_bytes,
    }


def find_candidate_file(candidate_dir: Path, preferred_name: str, pattern: str) -> Path:
    preferred = candidate_dir / preferred_name
    if preferred.exists():
        return preferred
    matches = sorted(candidate_dir.glob(pattern))
    if not matches:
        return preferred
    return matches[0]


def compile_ok(path: Path) -> tuple[bool, str | None]:
    try:
        py_compile.compile(str(path), doraise=True)
        return True, None
    except Exception as exc:
        return False, str(exc)


def scan_order_calls(main_py: Path) -> list[str]:
    text = main_py.read_text(encoding="utf-8")
    hits = []
    for pattern in FORBIDDEN_ORDER_PATTERNS:
        if re.search(pattern, text):
            hits.append(pattern)
    return hits


def validate_target(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"passed": False, "reason": "missing"}
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_TARGET_COLUMNS - set(frame.columns))
    trade_dates = sorted(pd.to_datetime(frame["trade_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())
    duplicate_rows = int(frame.duplicated(["trade_date", "instrument"]).sum()) if {"trade_date", "instrument"} <= set(frame.columns) else 0
    return {
        "passed": not missing and len(frame) > 0 and duplicate_rows == 0,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_columns": missing,
        "trade_dates": trade_dates,
        "duplicate_trade_date_instrument_rows": duplicate_rows,
    }


def validate_forbidden(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"passed": False, "reason": "missing"}
    frame = pd.read_csv(path, dtype=str).fillna("")
    symbol_cols = [col for col in frame.columns if "symbol" in col.lower() or "instrument" in col.lower() or "code" in col.lower()]
    return {
        "passed": len(frame) > 0 and bool(symbol_cols),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "symbol_like_columns": symbol_cols,
    }


def validate_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"passed": False, "reason": "missing"}
    meta = load_json(path)
    feature_cols = list(meta.get("feature_cols") or [])
    unstable = sorted(set(feature_cols) & LIVE_UNSTABLE_FEATURES)
    checks = {
        "feature_mode_live": meta.get("feature_mode") == "live",
        "deployment_disabled": meta.get("deployment_allowed") is False,
        "paper_candidate_only": meta.get("paper_candidate_only") is True,
        "threshold_080": abs(float(meta.get("threshold", -1)) - 0.80) < 1e-9,
        "trade_fraction_030": abs(float(meta.get("trade_fraction", -1)) - 0.30) < 1e-9,
        "no_live_unstable_features": not unstable,
        "feature_cols_present": bool(feature_cols),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "feature_cols": feature_cols,
        "live_unstable_features_found": unstable,
        "train_metrics": meta.get("train_metrics") or {},
    }


def run_preflight(candidate_dir: Path, output_json: Path) -> dict[str, Any]:
    paths = {
        "main_py": candidate_dir / "main.py",
        "model": find_candidate_file(candidate_dir, "inner_t0_0935_1445_catboost.cbm", "inner_t0_*_catboost.cbm"),
        "model_meta": find_candidate_file(candidate_dir, "inner_t0_0935_1445_model_meta.json", "inner_t0_*_model_meta.json"),
        "target_context": candidate_dir / "gm_c_baseline_targets.csv",
        "forbidden": candidate_dir / "gm_c_forbidden_symbols.csv",
        "readme": candidate_dir / "README.md",
    }
    file_checks = {name: check_file(path) for name, path in paths.items()}
    compile_passed, compile_error = compile_ok(paths["main_py"]) if paths["main_py"].exists() else (False, "main.py missing")
    order_hits = scan_order_calls(paths["main_py"]) if paths["main_py"].exists() else []
    meta_check = validate_meta(paths["model_meta"])
    target_check = validate_target(paths["target_context"])
    forbidden_check = validate_forbidden(paths["forbidden"])
    errors = []
    if not all(row["passed"] for row in file_checks.values()):
        errors.append("required package files missing or empty")
    if not compile_passed:
        errors.append("main.py py_compile failed")
    if order_hits:
        errors.append("main.py contains order submission API calls")
    if not meta_check["passed"]:
        errors.append("model metadata checks failed")
    if not target_check["passed"]:
        errors.append("target context checks failed")
    if not forbidden_check["passed"]:
        errors.append("forbidden symbol checks failed")
    report = {
        "status": "passed" if not errors else "failed",
        "candidate_dir": str(candidate_dir.resolve()),
        "file_checks": file_checks,
        "main_py_compile": {"passed": compile_passed, "error": compile_error},
        "order_call_scan": {"passed": not order_hits, "hits": order_hits},
        "model_meta": meta_check,
        "target_context": target_check,
        "forbidden": forbidden_check,
        "errors": errors,
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE_DIR))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = run_preflight(Path(args.candidate_dir).resolve(), Path(args.output_json).resolve())
    print(
        "[inner t0 preflight] "
        f"status={report['status']} errors={len(report['errors'])} output={Path(args.output_json).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
