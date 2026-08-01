from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checked_artifact(path_value: str | None, expected_hash: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected_hash and actual.upper() != str(expected_hash).upper():
        raise RuntimeError(
            f"artifact hash mismatch path={path} expected={expected_hash} actual={actual}"
        )
    return {"path": str(path), "sha256": actual}


def build_manifest(
    primary_meta_path: Path,
    secondary_meta_path: Path,
    daily_gate_path: Path,
    frozen_profile_path: Path,
    strict_gate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    primary = read_json(primary_meta_path)
    secondary = read_json(secondary_meta_path)
    daily_gate = read_json(daily_gate_path)
    frozen_profile = read_json(frozen_profile_path)
    strict_gate = read_json(strict_gate_path)
    if int(primary.get("fold", -1)) != 1 or primary.get("test_start") != "2026-04-08":
        raise RuntimeError("primary model is not the exact frozen 09:45 forward fold")
    if primary.get("score_calibration") != "isotonic":
        raise RuntimeError("primary model must use frozen isotonic edge calibration")
    if int(secondary.get("fold", -1)) != 19 or secondary.get("test_start") != "2026-04-08":
        raise RuntimeError("secondary model is not the exact frozen 10:00 fold 19")
    if secondary.get("score_calibration") != "none":
        raise RuntimeError("secondary stock model must expose its raw CatBoost score")
    if daily_gate.get("expected_forward_parity", {}).get("passed") is not True:
        raise RuntimeError("daily Ridge gate did not reproduce the consumed forward scores")
    if frozen_profile.get("strict_gate", {}).get("passed") is not False:
        raise RuntimeError("source inner profile must remain strict-gate failed")
    for document in (primary, secondary, daily_gate):
        if document.get("deployment_allowed") is not False:
            raise RuntimeError("all frozen shadow artifacts must keep deployment_allowed=false")
        if document.get("paper_orders_allowed") is not False:
            raise RuntimeError("all frozen shadow artifacts must keep paper_orders_allowed=false")
    primary_model = checked_artifact(primary["model_path"], primary["model_sha256"])
    primary_calibration = checked_artifact(
        primary["calibration_path"], primary["calibration_sha256"]
    )
    secondary_model = checked_artifact(secondary["model_path"], secondary["model_sha256"])
    feature_cols_match = primary.get("feature_cols") == secondary.get("feature_cols")
    if not feature_cols_match:
        raise RuntimeError("09:45 and 10:00 model feature contracts differ")
    report = {
        "status": "inner_t0_multidecision_shadow_models_frozen",
        "profile_name": frozen_profile.get("name"),
        "source_profile": {
            "path": str(frozen_profile_path.resolve()),
            "sha256": sha256_file(frozen_profile_path),
        },
        "source_strict_gate": {
            "path": str(strict_gate_path.resolve()),
            "sha256": sha256_file(strict_gate_path),
            "passed": bool(strict_gate.get("passed", False)),
            "decision": strict_gate.get("decision"),
        },
        "primary_0945": {
            "component": "0945_high_confidence",
            "decision_time": "09:45",
            "direction": "sell_first",
            "trigger_distance": 0.0075,
            "score_threshold": 0.005,
            "daily_top_n": 2,
            "model": primary_model,
            "calibration": primary_calibration,
            "metadata": {
                "path": str(primary_meta_path.resolve()),
                "sha256": sha256_file(primary_meta_path),
            },
            "feature_cols": primary["feature_cols"],
        },
        "secondary_1000": {
            "component": "1000_daily_ridge_gate",
            "decision_time": "10:00",
            "direction": "sell_first",
            "trigger_distance": 0.0075,
            "stock_score_threshold": None,
            "daily_top_n": 2,
            "model": secondary_model,
            "metadata": {
                "path": str(secondary_meta_path.resolve()),
                "sha256": sha256_file(secondary_meta_path),
            },
            "daily_gate": {
                "path": str(daily_gate_path.resolve()),
                "sha256": sha256_file(daily_gate_path),
                "threshold": 0.0,
                "expected_forward_parity": daily_gate["expected_forward_parity"],
            },
            "feature_cols": secondary["feature_cols"],
        },
        "portfolio": {
            "same_date_symbol_conflict": "keep_0945_primary",
            "max_symbols_per_day": 4,
            "max_daily_turnover": 0.03,
            "sizing": "one_lot_max50",
            "trigger_window_end": "11:00:00",
            "buyback_window": "14:45-14:50",
        },
        "target_context_policy": {
            "default_max_calendar_age_days": 0,
            "signal_date_required": True,
            "default_max_signal_calendar_age_days": 4,
            "signal_must_precede_observation_date": True,
            "behavior": "fail_closed_before_scoring_when_target_source_date_is_not_today",
        },
        "evidence": {
            "primary_prediction_parity": "3132/3132 keys; raw_score and score max_abs_diff=0",
            "secondary_prediction_parity": "3132/3132 keys; raw_score and score max_abs_diff=0",
            "daily_gate_prediction_parity": daily_gate["expected_forward_parity"],
        },
        "permissions": {
            "offline_research_allowed": True,
            "no_order_shadow_allowed_after_fresh_target": True,
            "paper_orders_allowed": False,
            "main_py_integration_allowed": False,
            "deployment_allowed": False,
        },
        "runtime_status": {
            "pure_shadow_engine_complete": True,
            "gmquant_wrapper_built": False,
            "blocked_by": "fresh middle+outer target/context after 2026-06-05",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-meta", required=True)
    parser.add_argument("--secondary-meta", required=True)
    parser.add_argument("--daily-gate", required=True)
    parser.add_argument("--frozen-profile", required=True)
    parser.add_argument("--strict-gate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_manifest(
        Path(args.primary_meta).resolve(),
        Path(args.secondary_meta).resolve(),
        Path(args.daily_gate).resolve(),
        Path(args.frozen_profile).resolve(),
        Path(args.strict_gate).resolve(),
        Path(args.output).resolve(),
    )
    print(
        f"[inner shadow manifest] status={report['status']} "
        f"runtime={report['runtime_status']} output={Path(args.output).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
