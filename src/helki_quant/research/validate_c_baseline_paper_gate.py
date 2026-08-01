from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "c_baseline_paper_gate.json"
DEFAULT_LOCAL_LOG = (
    HERE
    / "outputs"
    / "production_logs_c_baseline_gm_reject_stress_top150_80"
    / "c_top150_rb45_risk0.80_cap0.30_stress"
)
DEFAULT_GM_COMPARE = HERE / "outputs" / "gm_local_audit_compare_top150_80_reject_stress_20260605.json"
DEFAULT_HOLDOUT_SUMMARY = HERE / "outputs" / "extended_daily_holdout_c_baseline_20260605_summary.csv"
DEFAULT_SELLBLOCK_SUMMARY = HERE / "outputs" / "canonical_20260605_v2_sellblock_stress_summary.csv"
DEFAULT_STALEEXIT_SUMMARY = HERE / "outputs" / "canonical_20260605_v2_staleexit_stress_summary.csv"
DEFAULT_OUTPUT = HERE / "outputs" / "c_baseline_paper_gate_validation.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_check(checks: list[dict], name: str, passed: bool, value, threshold, detail: str = "") -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "value": value,
            "threshold": threshold,
            "detail": detail,
        }
    )


def validate_local(local_log_dir: Path, thresholds: dict) -> tuple[list[dict], dict]:
    audit = load_json(local_log_dir / "audit.json")
    checks: list[dict] = []
    add_check(
        checks,
        "local.max_single_day_turnover",
        float(audit.get("max_daily_turnover", 0.0)) <= thresholds["max_single_day_turnover"],
        float(audit.get("max_daily_turnover", 0.0)),
        thresholds["max_single_day_turnover"],
    )
    add_check(
        checks,
        "local.max_gross_exposure",
        float(audit.get("max_gross_exposure", 0.0)) <= thresholds["max_gross_exposure"],
        float(audit.get("max_gross_exposure", 0.0)),
        thresholds["max_gross_exposure"],
    )
    add_check(
        checks,
        "local.min_cash",
        float(audit.get("min_cash", 0.0)) >= thresholds["min_cash"],
        float(audit.get("min_cash", 0.0)),
        thresholds["min_cash"],
    )
    add_check(
        checks,
        "local.max_drawdown",
        float(audit.get("max_drawdown", 0.0)) <= thresholds["max_drawdown"],
        float(audit.get("max_drawdown", 0.0)),
        thresholds["max_drawdown"],
    )
    add_check(
        checks,
        "local.min_total_return",
        float(audit.get("total_return", 0.0)) >= thresholds["min_total_return"],
        float(audit.get("total_return", 0.0)),
        thresholds["min_total_return"],
    )
    add_check(
        checks,
        "local.max_lot_violations",
        int(audit.get("lot_violations", 0)) <= thresholds["max_lot_violations"],
        int(audit.get("lot_violations", 0)),
        thresholds["max_lot_violations"],
    )
    add_check(
        checks,
        "local.max_negative_cash_events",
        int(audit.get("negative_cash_events", 0)) <= thresholds["max_negative_cash_events"],
        int(audit.get("negative_cash_events", 0)),
        thresholds["max_negative_cash_events"],
    )
    add_check(
        checks,
        "local.max_nav_mismatch",
        float(audit.get("nav_mismatch_max", 0.0)) <= thresholds["max_nav_mismatch"],
        float(audit.get("nav_mismatch_max", 0.0)),
        thresholds["max_nav_mismatch"],
    )
    optional_checks = (
        (
            "min_avg_rebalance_budget_utilization",
            "avg_rebalance_budget_utilization",
            ">=",
        ),
        (
            "min_rebalance_budget_utilization",
            "min_rebalance_budget_utilization",
            ">=",
        ),
        ("min_avg_rebalance_effective_weight", "avg_rebalance_effective_weight", ">="),
        ("max_allocated_name_weight", "max_allocated_name_weight", "<="),
        ("min_rebalance_count", "rebalance_count", ">="),
        ("min_trades", "trades", ">="),
    )
    for threshold_key, audit_key, operator in optional_checks:
        if threshold_key not in thresholds:
            continue
        observed = float(audit.get(audit_key, 0.0))
        threshold = float(thresholds[threshold_key])
        passed = observed >= threshold if operator == ">=" else observed <= threshold
        add_check(checks, f"local.{threshold_key}", passed, observed, threshold)
    expected_mode = thresholds.get("expected_allocation_mode")
    if expected_mode:
        observed_mode = audit.get("allocation_mode")
        add_check(
            checks,
            "local.expected_allocation_mode",
            observed_mode == expected_mode,
            observed_mode,
            expected_mode,
        )
    expected_buffer = thresholds.get("expected_buffer_multiple")
    if expected_buffer is not None:
        observed_buffer = audit.get("config", {}).get("buffer_multiple")
        add_check(
            checks,
            "local.expected_buffer_multiple",
            observed_buffer == expected_buffer,
            observed_buffer,
            expected_buffer,
        )
    for threshold_key, audit_key in (
        ("expected_outer_risk_threshold", "outer_risk_threshold"),
        ("expected_outer_risk_floor", "outer_risk_floor"),
    ):
        if threshold_key not in thresholds:
            continue
        observed = audit.get(audit_key)
        expected = thresholds[threshold_key]
        add_check(
            checks,
            f"local.{threshold_key}",
            observed is not None and abs(float(observed) - float(expected)) <= 1e-12,
            observed,
            expected,
        )
    return checks, audit


def validate_gm(compare_path: Path, thresholds: dict) -> tuple[list[dict], dict]:
    checks: list[dict] = []
    if not compare_path.exists():
        add_check(checks, "gm.exists", False, False, True)
        return checks, {}
    report = load_json(compare_path)
    gm = report.get("gm", {})
    fill = report.get("fill_comparison", {})
    diffs = report.get("differences", {})
    submitted = max(int(gm.get("submitted_orders", 0)), 1)
    rejected = int(gm.get("rejected_orders", 0))
    reject_ratio = rejected / submitted
    add_check(checks, "gm.max_rejected_orders", rejected <= thresholds["max_rejected_orders"], rejected, thresholds["max_rejected_orders"])
    add_check(
        checks,
        "gm.max_rejected_order_ratio",
        reject_ratio <= thresholds["max_rejected_order_ratio"],
        reject_ratio,
        thresholds["max_rejected_order_ratio"],
    )
    if "max_unique_rejected_symbol_sides" in thresholds:
        value = int(gm.get("unique_rejected_symbol_sides", rejected))
        add_check(
            checks,
            "gm.max_unique_rejected_symbol_sides",
            value <= thresholds["max_unique_rejected_symbol_sides"],
            value,
            thresholds["max_unique_rejected_symbol_sides"],
        )
    if "max_unexpected_rejected_orders" in thresholds:
        value = int(gm.get("unexpected_rejected_orders", rejected))
        add_check(
            checks,
            "gm.max_unexpected_rejected_orders",
            value <= thresholds["max_unexpected_rejected_orders"],
            value,
            thresholds["max_unexpected_rejected_orders"],
        )
    if "max_unresolved_rejected_sell_symbols" in thresholds:
        value = int(gm.get("unresolved_rejected_sell_symbols", rejected))
        add_check(
            checks,
            "gm.max_unresolved_rejected_sell_symbols",
            value <= thresholds["max_unresolved_rejected_sell_symbols"],
            value,
            thresholds["max_unresolved_rejected_sell_symbols"],
        )
    add_check(checks, "gm.max_gm_only_fills", int(fill.get("gm_only_keys", 0)) <= thresholds["max_gm_only_fills"], int(fill.get("gm_only_keys", 0)), thresholds["max_gm_only_fills"])
    add_check(checks, "gm.max_local_only_fills", int(fill.get("local_only_keys", 0)) <= thresholds["max_local_only_fills"], int(fill.get("local_only_keys", 0)), thresholds["max_local_only_fills"])
    add_check(checks, "gm.max_volume_mismatch_keys", int(fill.get("volume_mismatch_keys", 0)) <= thresholds["max_volume_mismatch_keys"], int(fill.get("volume_mismatch_keys", 0)), thresholds["max_volume_mismatch_keys"])
    return_gap = abs(float(diffs.get("total_return_gm_minus_local") or 0.0))
    drawdown_gap = abs(float(diffs.get("max_drawdown_gm_minus_local") or 0.0))
    add_check(checks, "gm.max_return_gap_abs", return_gap <= thresholds["max_return_gap_abs"], return_gap, thresholds["max_return_gap_abs"])
    add_check(checks, "gm.max_drawdown_gap_abs", drawdown_gap <= thresholds["max_drawdown_gap_abs"], drawdown_gap, thresholds["max_drawdown_gap_abs"])
    return checks, report


def validate_holdout(summary_path: Path, thresholds: dict, profile_name: str) -> tuple[list[dict], dict]:
    checks: list[dict] = []
    if not summary_path.exists():
        add_check(checks, "holdout.exists", not thresholds["require_extended_daily_holdout"], False, True)
        return checks, {}
    frame = pd.read_csv(summary_path)
    stress = frame[(frame["profile"] == profile_name) & (frame["cost"] == "stress")].copy()
    if stress.empty:
        add_check(
            checks,
            "holdout.profile_stress_row",
            not thresholds["require_extended_daily_holdout"],
            None,
            profile_name,
            "canonical profile row is absent; non-blocking only when future holdout is not required",
        )
        return checks, {}
    row = stress.iloc[0].to_dict()
    add_check(checks, "holdout.min_stress_return", float(row["total_return"]) >= thresholds["min_stress_return"], float(row["total_return"]), thresholds["min_stress_return"])
    add_check(checks, "holdout.min_stress_sharpe", float(row["sharpe"]) >= thresholds["min_stress_sharpe"], float(row["sharpe"]), thresholds["min_stress_sharpe"])
    add_check(checks, "holdout.max_stress_drawdown", float(row["max_drawdown"]) <= thresholds["max_stress_drawdown"], float(row["max_drawdown"]), thresholds["max_stress_drawdown"])
    add_check(
        checks,
        "holdout.min_first_target_weight_sum",
        float(row["first_target_weight_sum"]) >= thresholds["min_first_target_weight_sum"],
        float(row["first_target_weight_sum"]),
        thresholds["min_first_target_weight_sum"],
    )
    return checks, row


def validate_target_manifest(
    manifest_path: Path,
    thresholds: dict,
    as_of_date: date,
) -> tuple[list[dict], dict]:
    checks: list[dict] = []
    required = bool(thresholds.get("required", False))
    if not manifest_path.exists():
        add_check(checks, "target_manifest.exists", not required, False, True)
        return checks, {}

    manifest = load_json(manifest_path)
    allocation = manifest.get("allocation", {})
    allocation_gate = manifest.get("allocation_gate", {})
    outer = manifest.get("outer_overlay", {})
    target_path = Path(str(manifest.get("target", "")))

    add_check(checks, "target_manifest.target_exists", target_path.exists(), target_path.exists(), True)
    add_check(
        checks,
        "target_manifest.min_symbols",
        int(manifest.get("symbols", 0)) >= int(thresholds.get("min_symbols", 1)),
        int(manifest.get("symbols", 0)),
        int(thresholds.get("min_symbols", 1)),
    )
    observed_ratio = float(
        manifest.get("effective_exposure_ratio", allocation.get("budget_utilization", 0.0))
    )
    min_ratio = float(thresholds.get("min_effective_exposure_ratio", 0.0))
    add_check(
        checks,
        "target_manifest.min_effective_exposure_ratio",
        observed_ratio >= min_ratio,
        observed_ratio,
        min_ratio,
    )
    observed_name_weight = float(allocation.get("max_name_weight", 0.0))
    max_name_weight = float(thresholds.get("max_name_weight", 1.0))
    add_check(
        checks,
        "target_manifest.max_name_weight",
        observed_name_weight <= max_name_weight,
        observed_name_weight,
        max_name_weight,
    )
    if bool(thresholds.get("require_allocation_gate", False)):
        add_check(
            checks,
            "target_manifest.allocation_gate",
            bool(allocation_gate.get("passed", False)),
            bool(allocation_gate.get("passed", False)),
            True,
        )
    expected_mode = thresholds.get("expected_allocation_mode")
    if expected_mode:
        observed_mode = allocation.get("mode")
        add_check(
            checks,
            "target_manifest.expected_allocation_mode",
            observed_mode == expected_mode,
            observed_mode,
            expected_mode,
        )
    for threshold_key, manifest_key in (
        ("expected_rebalance_every", "rebalance_every"),
        ("expected_buffer_multiple", "buffer_multiple"),
    ):
        if threshold_key not in thresholds:
            continue
        observed = manifest.get(manifest_key)
        expected = thresholds[threshold_key]
        add_check(
            checks,
            f"target_manifest.{threshold_key}",
            observed == expected,
            observed,
            expected,
        )
    if bool(thresholds.get("require_pause_buys_on_sell_reject", False)):
        observed = bool(
            manifest.get("execution_risk_controls", {}).get(
                "pause_buys_on_sell_reject", False
            )
        )
        add_check(
            checks,
            "target_manifest.pause_buys_on_sell_reject",
            observed,
            observed,
            True,
        )
    max_forbidden_hits = int(thresholds.get("max_forbidden_order_hits", 0))
    observed_forbidden_hits = int(manifest.get("forbidden_order_hits", 0))
    add_check(
        checks,
        "target_manifest.max_forbidden_order_hits",
        observed_forbidden_hits <= max_forbidden_hits,
        observed_forbidden_hits,
        max_forbidden_hits,
    )
    if bool(thresholds.get("require_outer_prediction", False)):
        outer_path = Path(str(outer.get("prediction", "")))
        outer_ok = bool(outer.get("required", False)) and outer_path.exists()
        add_check(
            checks,
            "target_manifest.outer_prediction",
            outer_ok,
            str(outer_path) if str(outer_path) else None,
            "existing required outer prediction",
        )

    for field, max_age_key, max_future_key in (
        ("signal_date", "max_signal_age_calendar_days", "max_future_signal_days"),
        ("trade_date", "max_target_age_calendar_days", "max_future_target_days"),
    ):
        raw = manifest.get(field)
        try:
            parsed = pd.Timestamp(raw).date()
        except Exception:
            add_check(checks, f"target_manifest.{field}_valid", False, raw, "YYYY-MM-DD")
            continue
        age = (as_of_date - parsed).days
        max_age = int(thresholds.get(max_age_key, 10**9))
        max_future = int(thresholds.get(max_future_key, 10**9))
        add_check(
            checks,
            f"target_manifest.{field}_freshness",
            -max_future <= age <= max_age,
            age,
            {"max_age_days": max_age, "max_future_days": max_future},
            f"as_of={as_of_date.isoformat()} value={parsed.isoformat()}",
        )
    return checks, manifest


def normalize_profile_name(name: str) -> str:
    text = str(name)
    return text[:-7] if text.endswith("_stress") else text


def validate_stress_summary(
    summary_path: Path,
    thresholds: dict,
    profile_name: str,
    prefix: str,
) -> tuple[list[dict], dict]:
    checks: list[dict] = []
    required = bool(thresholds.get("required", False))
    if not summary_path.exists():
        add_check(checks, f"{prefix}.exists", not required, False, True)
        return checks, {}
    frame = pd.read_csv(summary_path)
    if "profile" not in frame.columns:
        add_check(checks, f"{prefix}.profile_column", False, None, "profile")
        return checks, {}
    normalized = frame["profile"].map(normalize_profile_name)
    stress = frame[normalized.eq(normalize_profile_name(profile_name))].copy()
    if stress.empty:
        add_check(checks, f"{prefix}.profile_row", False, None, profile_name)
        return checks, {}
    row = stress.iloc[0].to_dict()
    return_col = f"{prefix}_return"
    mdd_col = f"{prefix}_mdd"
    cash_col = f"{prefix}_min_cash"
    add_check(
        checks,
        f"{prefix}.min_total_return",
        float(row[return_col]) >= thresholds["min_total_return"],
        float(row[return_col]),
        thresholds["min_total_return"],
    )
    add_check(
        checks,
        f"{prefix}.max_drawdown",
        float(row[mdd_col]) <= thresholds["max_drawdown"],
        float(row[mdd_col]),
        thresholds["max_drawdown"],
    )
    add_check(
        checks,
        f"{prefix}.max_return_drop",
        -float(row["return_delta"]) <= thresholds["max_return_drop"],
        -float(row["return_delta"]),
        thresholds["max_return_drop"],
    )
    add_check(
        checks,
        f"{prefix}.min_cash",
        float(row[cash_col]) >= thresholds["min_cash"],
        float(row[cash_col]),
        thresholds["min_cash"],
    )
    add_check(
        checks,
        f"{prefix}.max_final_holdings_delta",
        int(row["final_holdings_delta"]) <= thresholds["max_final_holdings_delta"],
        int(row["final_holdings_delta"]),
        thresholds["max_final_holdings_delta"],
    )
    return checks, row


def validate(
    config_path: Path,
    local_log_dir: Path,
    gm_compare_path: Path,
    holdout_summary_path: Path,
    sellblock_summary_path: Path,
    staleexit_summary_path: Path,
    target_manifest_path: Path,
    output_path: Path,
    as_of_date: date | None = None,
) -> dict:
    as_of_date = as_of_date or date.today()
    config = load_json(config_path)
    local_checks, local_audit = validate_local(local_log_dir, config["local_log_thresholds"])
    gm_checks, gm_report = validate_gm(gm_compare_path, config["gm_audit_thresholds"])
    profile_name = local_audit.get("profile", {}).get("name", "")
    holdout_checks, holdout_row = validate_holdout(holdout_summary_path, config["holdout_thresholds"], profile_name)
    sellblock_checks, sellblock_row = validate_stress_summary(
        sellblock_summary_path,
        config["sellblock_stress_thresholds"],
        profile_name,
        "sellblock",
    )
    staleexit_checks, staleexit_row = validate_stress_summary(
        staleexit_summary_path,
        config["staleexit_stress_thresholds"],
        profile_name,
        "staleexit",
    )
    target_checks, target_manifest = validate_target_manifest(
        target_manifest_path,
        config.get("target_manifest_thresholds", {}),
        as_of_date,
    )
    checks = (
        local_checks
        + gm_checks
        + holdout_checks
        + sellblock_checks
        + staleexit_checks
        + target_checks
    )
    failed = [check for check in checks if not check["passed"]]
    result = {
        "status": "paper_gate_validation",
        "passed": len(failed) == 0,
        "deployment_allowed": False,
        "config": str(config_path),
        "local_log_dir": str(local_log_dir),
        "gm_compare": str(gm_compare_path),
        "holdout_summary": str(holdout_summary_path),
        "sellblock_summary": str(sellblock_summary_path),
        "staleexit_summary": str(staleexit_summary_path),
        "target_manifest": str(target_manifest_path),
        "as_of_date": as_of_date.isoformat(),
        "profile": profile_name,
        "checks": checks,
        "failed_checks": failed,
        "local_audit": {
            "total_return": local_audit.get("total_return"),
            "max_drawdown": local_audit.get("max_drawdown"),
            "turnover": local_audit.get("turnover"),
            "max_daily_turnover": local_audit.get("max_daily_turnover"),
            "max_gross_exposure": local_audit.get("max_gross_exposure"),
            "min_cash": local_audit.get("min_cash"),
        },
        "gm": {
            "total_return": gm_report.get("gm", {}).get("total_return"),
            "rejected_orders": gm_report.get("gm", {}).get("rejected_orders"),
            "return_gap": gm_report.get("differences", {}).get("total_return_gm_minus_local"),
        },
        "holdout": holdout_row,
        "sellblock_stress": sellblock_row,
        "staleexit_stress": staleexit_row,
        "target": {
            "signal_date": target_manifest.get("signal_date"),
            "trade_date": target_manifest.get("trade_date"),
            "symbols": target_manifest.get("symbols"),
            "effective_exposure_ratio": target_manifest.get("effective_exposure_ratio"),
            "allocation_mode": target_manifest.get("allocation", {}).get("mode"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--local-log-dir", default=str(DEFAULT_LOCAL_LOG))
    parser.add_argument("--gm-compare", default=str(DEFAULT_GM_COMPARE))
    parser.add_argument("--holdout-summary", default=str(DEFAULT_HOLDOUT_SUMMARY))
    parser.add_argument("--sellblock-summary", default=str(DEFAULT_SELLBLOCK_SUMMARY))
    parser.add_argument("--staleexit-summary", default=str(DEFAULT_STALEEXIT_SUMMARY))
    parser.add_argument("--target-manifest", default="")
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = validate(
        Path(args.config).resolve(),
        Path(args.local_log_dir).resolve(),
        Path(args.gm_compare).resolve(),
        Path(args.holdout_summary).resolve(),
        Path(args.sellblock_summary).resolve(),
        Path(args.staleexit_summary).resolve(),
        Path(args.target_manifest).resolve() if args.target_manifest else Path("__missing_target_manifest__"),
        Path(args.output).resolve(),
        pd.Timestamp(args.as_of_date).date() if args.as_of_date else None,
    )
    print(
        f"[paper gate] passed={result['passed']} "
        f"profile={result['profile']} failed={len(result['failed_checks'])} "
        f"output={args.output}",
        flush=True,
    )
    for check in result["failed_checks"]:
        print(
            f"  FAIL {check['name']}: value={check['value']} threshold={check['threshold']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
