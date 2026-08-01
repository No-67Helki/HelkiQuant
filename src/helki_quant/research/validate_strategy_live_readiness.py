from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from paper_activation_registry import (
    EVENT_ERROR,
    EVENT_FINALIZED,
    EVENT_READY,
    SESSION_METRICS_SCHEMA_VERSION,
    read_activation_registry,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    value: object,
    threshold: object,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "value": value,
            "threshold": threshold,
        }
    )


def _read_optional(
    path: Path,
    label: str,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        payload = load_json(path, label)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        add_check(checks, f"{label}.readable", False, str(exc), "valid JSON")
        return {}
    add_check(checks, f"{label}.readable", True, str(path.resolve()), "valid JSON")
    return payload


def _promotion_holdout_end(promotion: dict[str, Any]) -> pd.Timestamp | None:
    evidence_path = promotion.get("evidence")
    if not evidence_path:
        return None
    path = Path(str(evidence_path)).resolve()
    if not path.is_file():
        return None
    evidence = load_json(path, "promotion evidence")
    value = pd.to_datetime(
        (evidence.get("holdout") or {}).get("end"),
        errors="coerce",
    )
    if pd.isna(value):
        return None
    return pd.Timestamp(value).normalize()


def validate(
    *,
    config_path: Path,
    promotion_path: Path,
    preflight_path: Path,
    gm_compare_path: Path,
    activation_registry_path: Path,
    expected_account_id: str,
    output_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    promotion_path = promotion_path.resolve()
    preflight_path = preflight_path.resolve()
    gm_compare_path = gm_compare_path.resolve()
    activation_registry_path = activation_registry_path.resolve()
    expected_account_id = str(expected_account_id).strip()
    if not expected_account_id:
        raise ValueError("expected_account_id is required")

    config = load_json(config_path, "live-readiness config")
    checks: list[dict[str, Any]] = []
    promotion = _read_optional(promotion_path, "promotion", checks)
    preflight = _read_optional(preflight_path, "preflight", checks)
    gm_compare = _read_optional(gm_compare_path, "gm_compare", checks)

    selected_profile = str(promotion.get("selected_profile_id") or "")
    add_check(
        checks,
        "promotion.passed",
        promotion.get("passed") is True,
        promotion.get("passed"),
        True,
    )
    add_check(
        checks,
        "promotion.paper_candidate_allowed",
        promotion.get("paper_candidate_promotion_allowed") is True,
        promotion.get("paper_candidate_promotion_allowed"),
        True,
    )
    add_check(
        checks,
        "promotion.selected_profile",
        bool(selected_profile),
        selected_profile or None,
        "non-empty",
    )
    holdout_end = _promotion_holdout_end(promotion)
    add_check(
        checks,
        "promotion.untouched_holdout_end",
        holdout_end is not None,
        str(holdout_end.date()) if holdout_end is not None else None,
        "valid evidence holdout end",
    )

    target = preflight.get("target") or {}
    add_check(
        checks,
        "preflight.passed",
        preflight.get("passed") is True and not preflight.get("errors"),
        {
            "passed": preflight.get("passed"),
            "errors": preflight.get("errors") or [],
        },
        {"passed": True, "errors": []},
    )
    add_check(
        checks,
        "preflight.invalid_lots",
        int(target.get("invalid_lots") or 0) == 0,
        int(target.get("invalid_lots") or 0),
        0,
    )
    add_check(
        checks,
        "preflight.forbidden_hits",
        not (target.get("forbidden_hits") or []),
        target.get("forbidden_hits") or [],
        [],
    )
    target_dates = sorted(str(value) for value in target.get("trade_dates") or [])
    add_check(
        checks,
        "preflight.single_target_date",
        len(target_dates) == 1,
        target_dates,
        "exactly one",
    )

    comparison = gm_compare.get("fill_comparison") or {}
    gm = gm_compare.get("gm") or {}
    aliases = config.get("profile_aliases") or {}
    expected_compare_profile = str(aliases.get(selected_profile) or selected_profile)
    add_check(
        checks,
        "gm_compare.profile",
        bool(selected_profile)
        and str(gm_compare.get("profile") or "") == expected_compare_profile,
        gm_compare.get("profile"),
        expected_compare_profile or "selected profile alias",
    )
    for field in (
        "gm_only_keys",
        "local_only_keys",
        "volume_mismatch_keys",
        "filled_volume_diff_total",
    ):
        value = int(comparison.get(field) or 0)
        add_check(checks, f"gm_compare.{field}", value == 0, value, 0)
    add_check(
        checks,
        "gm_compare.unexpected_rejected_orders",
        int(gm.get("unexpected_rejected_orders") or 0) == 0,
        int(gm.get("unexpected_rejected_orders") or 0),
        0,
    )
    add_check(
        checks,
        "gm_compare.unresolved_rejected_sell_symbols",
        int(gm.get("unresolved_rejected_sell_symbols") or 0) == 0,
        int(gm.get("unresolved_rejected_sell_symbols") or 0),
        0,
    )
    add_check(
        checks,
        "gm_compare.date_shift",
        int(gm_compare.get("gm_date_shift_trading_days") or 0) == 0,
        int(gm_compare.get("gm_date_shift_trading_days") or 0),
        0,
    )

    records, registry_errors = read_activation_registry(activation_registry_path)
    add_check(
        checks,
        "activation_registry.integrity",
        not registry_errors,
        registry_errors,
        [],
    )
    account_records = [
        row
        for row in records
        if str(row.get("account_id") or "") == expected_account_id
    ]
    ready_by_run = {
        str(row.get("run_id") or ""): row
        for row in account_records
        if row.get("event") == EVENT_READY
    }
    finalized = [
        row for row in account_records if row.get("event") == EVENT_FINALIZED
    ]
    error_records = [
        row for row in account_records if row.get("event") == EVENT_ERROR
    ]
    trade_dates = sorted(
        {
            str(row.get("trade_date") or "")
            for row in finalized
            if row.get("trade_date")
        }
    )
    finalized_sessions_by_date: dict[str, int] = {}
    for row in finalized:
        trade_date = str(row.get("trade_date") or "")
        if trade_date:
            finalized_sessions_by_date[trade_date] = (
                finalized_sessions_by_date.get(trade_date, 0) + 1
            )
    duplicate_session_dates = sorted(
        trade_date
        for trade_date, count in finalized_sessions_by_date.items()
        if count != 1
    )
    rules = config["paper_observation_rules"]
    min_sessions = int(rules["min_finalized_sessions"])
    add_check(
        checks,
        "paper.finalized_sessions",
        len(finalized) >= min_sessions,
        len(finalized),
        {">=": min_sessions},
    )
    add_check(
        checks,
        "paper.unique_trade_dates",
        len(trade_dates) >= min_sessions,
        len(trade_dates),
        {">=": min_sessions},
    )
    require_one_session_per_date = bool(
        rules.get("require_one_finalized_session_per_trade_date", True)
    )
    add_check(
        checks,
        "paper.one_finalized_session_per_trade_date",
        not require_one_session_per_date or not duplicate_session_dates,
        {
            trade_date: finalized_sessions_by_date[trade_date]
            for trade_date in duplicate_session_dates
        },
        {"count_per_trade_date": 1},
    )
    add_check(
        checks,
        "paper.error_events",
        len(error_records) <= int(rules["max_error_events"]),
        len(error_records),
        {"<=": int(rules["max_error_events"])},
    )

    position_sync_failures: list[str] = []
    finalize_sync_failures: list[str] = []
    pending_runs: list[str] = []
    no_rebalance_runs: list[str] = []
    quality_schema_failures: list[str] = []
    unexpected_reject_runs: list[str] = []
    unexplained_mismatch_runs: list[str] = []
    deferred_buy_runs: list[str] = []
    unresolved_by_date: dict[str, set[str]] = {}
    for row in finalized:
        run_id = str(row.get("run_id") or "")
        ready = ready_by_run.get(run_id) or {}
        if (ready.get("metrics") or {}).get("position_sync_succeeded") is not True:
            position_sync_failures.append(run_id)
        metrics = row.get("metrics") or {}
        pending = (
            list(metrics.get("pending_target_order_symbols") or [])
            + list(metrics.get("pending_execution_symbols") or [])
            + list(metrics.get("forbidden_clear_pending") or [])
        )
        if pending:
            pending_runs.append(run_id)
        if int(metrics.get("rebalance_events") or 0) < int(
            rules["min_rebalance_events_per_session"]
        ):
            no_rebalance_runs.append(run_id)
        if (
            int(metrics.get("session_metrics_schema_version") or 0)
            != SESSION_METRICS_SCHEMA_VERSION
        ):
            quality_schema_failures.append(run_id)
        if metrics.get("position_sync_succeeded_at_finalize") is not True:
            finalize_sync_failures.append(run_id)
        if int(metrics.get("unexpected_rejected_orders") or 0) > 0:
            unexpected_reject_runs.append(run_id)
        if int(metrics.get("unexplained_target_volume_abs_diff") or 0) > 0:
            unexplained_mismatch_runs.append(run_id)
        if metrics.get("pending_buy_symbols") or []:
            deferred_buy_runs.append(run_id)
        unresolved_by_date[str(row.get("trade_date") or "")] = {
            str(symbol)
            for symbol in metrics.get(
                "unresolved_rejected_sell_symbol_list"
            )
            or []
        }
    add_check(
        checks,
        "paper.position_sync",
        not position_sync_failures,
        position_sync_failures,
        [],
    )
    add_check(
        checks,
        "paper.final_position_sync",
        not finalize_sync_failures,
        finalize_sync_failures,
        [],
    )
    add_check(
        checks,
        "paper.pending_execution",
        not pending_runs,
        pending_runs,
        [],
    )
    add_check(
        checks,
        "paper.rebalance_presence",
        not no_rebalance_runs,
        no_rebalance_runs,
        [],
    )
    add_check(
        checks,
        "paper.session_quality_schema",
        not quality_schema_failures,
        quality_schema_failures,
        [],
    )
    max_unexpected_sessions = int(
        rules["max_sessions_with_unexpected_rejects"]
    )
    add_check(
        checks,
        "paper.unexpected_reject_sessions",
        len(unexpected_reject_runs) <= max_unexpected_sessions,
        unexpected_reject_runs,
        {"max_sessions": max_unexpected_sessions},
    )
    max_unexplained_sessions = int(
        rules["max_sessions_with_unexplained_target_mismatch"]
    )
    add_check(
        checks,
        "paper.unexplained_target_mismatch_sessions",
        len(unexplained_mismatch_runs) <= max_unexplained_sessions,
        unexplained_mismatch_runs,
        {"max_sessions": max_unexplained_sessions},
    )
    max_stale_run = 0
    stale_run_by_symbol: dict[str, int] = {}
    previous_symbols: set[str] = set()
    for trade_date in trade_dates:
        current_symbols = unresolved_by_date.get(trade_date, set())
        next_runs: dict[str, int] = {}
        for symbol in current_symbols:
            next_runs[symbol] = (
                stale_run_by_symbol.get(symbol, 0) + 1
                if symbol in previous_symbols
                else 1
            )
            max_stale_run = max(max_stale_run, next_runs[symbol])
        stale_run_by_symbol = next_runs
        previous_symbols = current_symbols
    max_allowed_stale_run = int(
        rules["max_consecutive_unresolved_sell_sessions"]
    )
    add_check(
        checks,
        "paper.consecutive_unresolved_sell_sessions",
        max_stale_run <= max_allowed_stale_run,
        max_stale_run,
        {"<=": max_allowed_stale_run},
    )
    latest_unresolved = (
        sorted(unresolved_by_date.get(trade_dates[-1], set()))
        if trade_dates
        else []
    )
    max_latest_unresolved = int(
        rules["max_latest_unresolved_sell_symbols"]
    )
    add_check(
        checks,
        "paper.latest_unresolved_sell_symbols",
        len(latest_unresolved) <= max_latest_unresolved,
        latest_unresolved,
        {"max_symbols": max_latest_unresolved},
    )
    strategy_ids = sorted(
        {
            str(row.get("strategy_id") or "")
            for row in finalized
            if row.get("strategy_id")
        }
    )
    add_check(
        checks,
        "paper.single_strategy_id",
        len(strategy_ids) == 1,
        strategy_ids,
        "exactly one",
    )
    if holdout_end is None:
        paper_after_holdout = False
    else:
        parsed_trade_dates = pd.to_datetime(trade_dates, errors="coerce")
        paper_after_holdout = bool(
            len(parsed_trade_dates)
            and not parsed_trade_dates.isna().any()
            and parsed_trade_dates.min().normalize() > holdout_end
        )
    add_check(
        checks,
        "paper.after_untouched_holdout",
        paper_after_holdout,
        {
            "paper_start": trade_dates[0] if trade_dates else None,
            "holdout_end": (
                str(holdout_end.date()) if holdout_end is not None else None
            ),
        },
        "paper_start > untouched_holdout_end",
    )
    latest_target_date = target_dates[0] if len(target_dates) == 1 else None
    latest_paper_date = trade_dates[-1] if trade_dates else None
    add_check(
        checks,
        "paper.latest_preflight_finalized",
        latest_target_date is not None and latest_target_date == latest_paper_date,
        {
            "preflight": latest_target_date,
            "latest_finalized": latest_paper_date,
        },
        "equal",
    )

    failed = [item for item in checks if not item["passed"]]
    passed = not failed
    inputs: dict[str, Any] = {"config": artifact(config_path)}
    for name, path in (
        ("promotion", promotion_path),
        ("preflight", preflight_path),
        ("gm_compare", gm_compare_path),
        ("activation_registry", activation_registry_path),
    ):
        inputs[name] = artifact(path) if path.is_file() else {"path": str(path)}
    result = {
        "schema_version": 1,
        "status": "strategy_live_readiness_gate",
        "passed": passed,
        "paper_simulation_candidate_ready": passed,
        "real_money_deployment_allowed": False,
        "selected_profile_id": selected_profile or None,
        "expected_compare_profile": expected_compare_profile or None,
        "expected_account_id": expected_account_id,
        "inputs": inputs,
        "paper_observation": {
            "finalized_sessions": len(finalized),
            "trade_date_start": trade_dates[0] if trade_dates else None,
            "trade_date_end": trade_dates[-1] if trade_dates else None,
            "strategy_ids": strategy_ids,
            "sessions_with_deferred_buys": len(deferred_buy_runs),
            "max_consecutive_unresolved_sell_sessions": max_stale_run,
            "latest_unresolved_sell_symbols": latest_unresolved,
        },
        "checks": checks,
        "failed_checks": failed,
        "next_action": (
            "retain PAPER observation and prepare a separate human-approved live review"
            if passed
            else "complete the failed evidence and PAPER checks without changing the frozen profile"
        ),
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--config", type=Path, required=True)
    root.add_argument("--promotion", type=Path, required=True)
    root.add_argument("--preflight", type=Path, required=True)
    root.add_argument("--gm-compare", type=Path, required=True)
    root.add_argument("--activation-registry", type=Path, required=True)
    root.add_argument("--expected-account-id", required=True)
    root.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    result = validate(
        config_path=args.config,
        promotion_path=args.promotion,
        preflight_path=args.preflight,
        gm_compare_path=args.gm_compare,
        activation_registry_path=args.activation_registry,
        expected_account_id=args.expected_account_id,
        output_path=args.output,
    )
    print(
        "[live readiness] "
        f"passed={result['passed']} selected={result['selected_profile_id']} "
        f"paper_sessions={result['paper_observation']['finalized_sessions']} "
        f"failed={len(result['failed_checks'])} output={args.output}",
        flush=True,
    )
    for check in result["failed_checks"]:
        print(
            f"[live readiness] FAIL {check['name']}: "
            f"value={check['value']} threshold={check['threshold']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
