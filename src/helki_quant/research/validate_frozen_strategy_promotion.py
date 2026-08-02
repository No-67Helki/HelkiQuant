from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_repo_path(raw: object) -> Path:
    path = Path(str(raw or ""))
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def parse_mapping(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} entries must be PROFILE_ID=PATH")
        profile_id, raw_path = value.split("=", 1)
        profile_id = profile_id.strip()
        if not profile_id or profile_id in result:
            raise ValueError(f"invalid or duplicate {label} profile id: {profile_id!r}")
        result[profile_id] = Path(raw_path.strip()).resolve()
    return result


def profile_map(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = contract.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("promotion contract profiles must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    champion_count = 0
    for profile in profiles:
        profile_id = str(profile.get("id") or "")
        if not profile_id or profile_id in result:
            raise ValueError(f"invalid or duplicate contract profile id: {profile_id!r}")
        role = str(profile.get("role") or "")
        if role == "champion":
            champion_count += 1
        elif role != "challenger":
            raise ValueError(f"unsupported role for {profile_id}: {role!r}")
        result[profile_id] = profile
    if champion_count != 1:
        raise ValueError("promotion contract must contain exactly one champion")
    return result


def verify_artifact(record: dict[str, Any], label: str) -> dict[str, Any]:
    path = resolve_repo_path(record.get("path"))
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    expected = str(record.get("sha256") or "").upper()
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected={expected or None} actual={actual}"
        )
    return {"path": str(path), "sha256": actual, "bytes": int(path.stat().st_size)}


def verify_frozen_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if int(contract.get("schema_version", 0)) != 1:
        raise ValueError("unsupported promotion contract schema")
    profiles = profile_map(contract)
    verified: dict[str, Any] = {}
    for profile_id, profile in profiles.items():
        verified[profile_id] = verify_artifact(
            profile.get("frozen_artifact") or {},
            f"frozen artifact for {profile_id}",
        )
        overlay = profile.get("overlay") or {}
        generator = overlay.get("generator")
        if generator:
            verified[f"{profile_id}.generator"] = verify_artifact(
                generator,
                f"frozen generator for {profile_id}",
            )
    return verified


def calendar_sha256(dates: pd.DatetimeIndex) -> str:
    return hashlib.sha256(
        "\n".join(day.strftime("%Y-%m-%d") for day in dates).encode("ascii")
    ).hexdigest().upper()


def build_canonical_binding(
    readiness_path: Path,
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    readiness_path = readiness_path.resolve()
    readiness = load_json(readiness_path)
    required_true = (
        "passed",
        "data_integrity_passed",
        "promotion_window_ready",
        "profile_frozen",
    )
    failed = [name for name in required_true if readiness.get(name) is not True]
    if failed:
        raise ValueError(f"canonical readiness has not passed: {failed}")
    if readiness.get("return_metrics_evaluated") is not False:
        raise ValueError("canonical readiness must precede return evaluation")
    holdout = readiness.get("holdout") or {}
    expected = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "sessions": int(len(dates)),
        "calendar_sha256": calendar_sha256(dates),
    }
    observed = {
        "start": holdout.get("first_session"),
        "end": holdout.get("last_session"),
        "sessions": int(holdout.get("sessions") or 0),
        "calendar_sha256": str(holdout.get("calendar_sha256") or "").upper(),
    }
    if observed != expected:
        raise ValueError(
            f"replay calendar does not match canonical readiness: "
            f"observed={observed} expected={expected}"
        )
    manifest = verify_artifact(
        readiness.get("canonical_manifest") or {},
        "canonical manifest",
    )
    return {
        "schema_version": 1,
        "readiness": {
            "path": str(readiness_path),
            "sha256": sha256_file(readiness_path),
            "bytes": int(readiness_path.stat().st_size),
        },
        "manifest": manifest,
        "holdout": expected,
    }


def verify_canonical_binding(
    binding: object,
    evidence_holdout: dict[str, Any],
) -> dict[str, Any]:
    try:
        schema_version = int(binding.get("schema_version", 0))  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError):
        schema_version = 0
    if not isinstance(binding, dict) or schema_version != 1:
        raise ValueError("evidence has no supported canonical binding")
    readiness_record = binding.get("readiness") or {}
    readiness_artifact = verify_artifact(
        readiness_record,
        "canonical readiness",
    )
    readiness = load_json(Path(readiness_artifact["path"]))
    for name in (
        "passed",
        "data_integrity_passed",
        "promotion_window_ready",
        "profile_frozen",
    ):
        if readiness.get(name) is not True:
            raise ValueError(f"canonical readiness no longer passes: {name}")
    if readiness.get("return_metrics_evaluated") is not False:
        raise ValueError("canonical readiness was evaluated before promotion")
    manifest_artifact = verify_artifact(
        binding.get("manifest") or {},
        "canonical manifest",
    )
    readiness_manifest = readiness.get("canonical_manifest") or {}
    if (
        str(readiness_manifest.get("sha256") or "").upper()
        != manifest_artifact["sha256"]
    ):
        raise ValueError("canonical manifest is not the readiness-audited artifact")
    holdout = binding.get("holdout") or {}
    readiness_holdout = readiness.get("holdout") or {}
    expected = {
        "start": evidence_holdout.get("start"),
        "end": evidence_holdout.get("end"),
        "sessions": int(evidence_holdout.get("sessions") or 0),
        "calendar_sha256": str(
            evidence_holdout.get("calendar_sha256") or ""
        ).upper(),
    }
    observed = {
        "start": holdout.get("start"),
        "end": holdout.get("end"),
        "sessions": int(holdout.get("sessions") or 0),
        "calendar_sha256": str(holdout.get("calendar_sha256") or "").upper(),
    }
    audited = {
        "start": readiness_holdout.get("first_session"),
        "end": readiness_holdout.get("last_session"),
        "sessions": int(readiness_holdout.get("sessions") or 0),
        "calendar_sha256": str(
            readiness_holdout.get("calendar_sha256") or ""
        ).upper(),
    }
    if observed != expected or audited != expected:
        raise ValueError(
            "canonical binding, evidence, and readiness calendars do not match"
        )
    return {
        "readiness": readiness_artifact,
        "manifest": manifest_artifact,
        "holdout": observed,
    }


def verify_source_provenance(
    provenance: object,
    *,
    profile_id: str,
    required_names: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise ValueError(f"{profile_id} audit has no source_provenance")
    required = required_names or {
        "middle_prediction",
        "outer_prediction",
        "minute_windows",
        "group_metadata",
        "forbidden_symbols",
    }
    missing = sorted(
        name for name in required if not isinstance(provenance.get(name), dict)
    )
    if missing:
        raise ValueError(f"{profile_id} source provenance missing: {missing}")
    verified: dict[str, Any] = {}
    for name, record in provenance.items():
        if record is None:
            continue
        if not isinstance(record, dict):
            raise ValueError(f"{profile_id} invalid provenance record: {name}")
        path = Path(str(record.get("path") or "")).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{profile_id} provenance file not found: {name}={path}"
            )
        expected = str(record.get("sha256") or "").upper()
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise ValueError(
                f"{profile_id} provenance hash mismatch: {name}"
            )
        verified[name] = {
            "path": str(path),
            "sha256": actual,
            "bytes": int(path.stat().st_size),
        }
    return verified


def observed_parameters(audit: dict[str, Any]) -> dict[str, Any]:
    profile = audit.get("profile") or {}
    config = audit.get("config") or {}
    cost = audit.get("cost") or {}
    return {
        "top_k": profile.get("top_k"),
        "rebalance_every": profile.get("rebalance_every"),
        "risk_budget": profile.get("risk_budget"),
        "industry_cap": profile.get("industry_cap"),
        "buffer_multiple": config.get("buffer_multiple"),
        "allocation_mode": audit.get("allocation_mode"),
        "outer_risk_threshold": audit.get("outer_risk_threshold"),
        "outer_risk_floor": audit.get("outer_risk_floor"),
        "pause_buys_on_sell_reject": audit.get("pause_buys_on_sell_reject"),
        "sell_first": audit.get("sell_first"),
        "cost_name": cost.get("name"),
    }


def values_equal(observed: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        try:
            return abs(float(observed) - float(expected)) <= 1e-12
        except (TypeError, ValueError):
            return False
    return observed == expected


def validate_profile_parameters(
    profile_id: str,
    audit: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    observed = observed_parameters(audit)
    mismatches = {
        name: {"observed": observed.get(name), "expected": value}
        for name, value in expected.items()
        if not values_equal(observed.get(name), value)
    }
    if mismatches:
        raise ValueError(f"{profile_id} parameter mismatch: {mismatches}")
    return observed


def validate_alpha_health_manifest(
    profile_id: str,
    manifest_path: Path,
    profile: dict[str, Any],
    audit_provenance: dict[str, Any],
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if int(manifest.get("schema_version", 0)) < 2:
        raise ValueError(f"{profile_id} alpha-health manifest schema must be >=2")
    expected_policy = (profile.get("overlay") or {}).get("policy") or {}
    observed_policy = manifest.get("policy") or {}
    mismatches = {
        name: {"observed": observed_policy.get(name), "expected": value}
        for name, value in expected_policy.items()
        if not values_equal(observed_policy.get(name), value)
    }
    if mismatches:
        raise ValueError(f"{profile_id} alpha-health policy mismatch: {mismatches}")
    verified = verify_source_provenance(
        manifest.get("source_provenance"),
        profile_id=f"{profile_id}.alpha_health",
        required_names={
            "generator",
            "middle_prediction",
            "broad_outer_prediction",
            "forbidden_symbols",
            "combined_outer",
            "daily_health",
        },
    )
    outer_hash = audit_provenance["outer_prediction"]["sha256"]
    if verified["combined_outer"]["sha256"] != outer_hash:
        raise ValueError(
            f"{profile_id} replay outer prediction is not the alpha-health output"
        )
    expected_generator = str(
        ((profile.get("overlay") or {}).get("generator") or {}).get("sha256") or ""
    ).upper()
    if verified["generator"]["sha256"] != expected_generator:
        raise ValueError(f"{profile_id} alpha-health generator hash mismatch")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "policy": observed_policy,
        "source_provenance": verified,
    }


def max_drawdown(nav: pd.Series) -> float:
    running = nav.cummax()
    return float((1.0 - nav / running).max()) if len(nav) else 0.0


def calculate_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    nav = pd.to_numeric(frame["nav"], errors="coerce")
    if nav.isna().any() or (nav <= 0).any():
        raise ValueError("daily_account contains invalid NAV")
    returns = nav.pct_change().dropna()
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (
        float(math.sqrt(252.0) * returns.mean() / volatility)
        if volatility > 0
        else 0.0
    )
    indexed = frame.assign(nav=nav).set_index("trade_date")
    month_end = indexed["nav"].resample("M").last().dropna()
    previous = pd.concat(
        [
            pd.Series(
                [float(nav.iloc[0])],
                index=[month_end.index[0] - pd.offsets.MonthEnd(1)],
            ),
            month_end,
        ]
    )
    monthly_returns = previous.pct_change().dropna()
    cash = pd.to_numeric(frame["cash"], errors="coerce")
    cash_ratio = cash / nav
    return {
        "sessions": int(len(frame)),
        "initial_nav": float(nav.iloc[0]),
        "final_nav": float(nav.iloc[-1]),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(nav),
        "turnover": float(
            pd.to_numeric(frame["day_turnover"], errors="coerce").fillna(0).sum()
        ),
        "max_daily_turnover": float(
            pd.to_numeric(frame["day_turnover"], errors="coerce").fillna(0).max()
        ),
        "min_cash_ratio": float(cash_ratio.min()),
        "max_gross_exposure": float(
            pd.to_numeric(frame["gross_exposure"], errors="coerce").fillna(0).max()
        ),
        "trades": int(
            pd.to_numeric(frame["day_trades"], errors="coerce").fillna(0).sum()
        ),
        "rebalance_count": int(
            pd.to_numeric(frame["is_rebalance"], errors="coerce").fillna(0).sum()
        ),
        "worst_month_return": (
            float(monthly_returns.min()) if len(monthly_returns) else 0.0
        ),
        "positive_month_ratio": (
            float((monthly_returns > 0).mean()) if len(monthly_returns) else 0.0
        ),
        "months": int(len(monthly_returns)),
    }


def build_evidence(
    contract_path: Path,
    profile_logs: dict[str, Path],
    profile_manifests: dict[str, Path],
    output_path: Path,
    canonical_readiness_path: Path,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"promotion evidence is immutable: {output_path}")
    contract = load_json(contract_path)
    profiles = profile_map(contract)
    verified_frozen = verify_frozen_contract(contract)
    expected_ids = set(profiles)
    if set(profile_logs) != expected_ids:
        raise ValueError(
            "profile-log ids must exactly match contract profiles: "
            f"observed={sorted(profile_logs)} expected={sorted(expected_ids)}"
        )

    training_end = pd.Timestamp(contract["training_data_end"]).normalize()
    common_dates: pd.DatetimeIndex | None = None
    rows: dict[str, Any] = {}
    for profile_id, profile in profiles.items():
        log_dir = profile_logs[profile_id].resolve()
        audit_path = log_dir / "audit.json"
        daily_path = log_dir / "daily_account.csv"
        if not audit_path.is_file() or not daily_path.is_file():
            raise FileNotFoundError(
                f"{profile_id} requires audit.json and daily_account.csv: {log_dir}"
            )
        audit = load_json(audit_path)
        if audit.get("status") != "production_log_export_research_only":
            raise ValueError(f"{profile_id} is not a production-style replay audit")
        observed = validate_profile_parameters(
            profile_id,
            audit,
            profile.get("parameters") or {},
        )
        provenance = verify_source_provenance(
            audit.get("source_provenance"),
            profile_id=profile_id,
        )
        frame = pd.read_csv(daily_path, parse_dates=["trade_date"])
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"], errors="coerce"
        ).dt.normalize()
        if frame["trade_date"].isna().any() or frame["trade_date"].duplicated().any():
            raise ValueError(f"{profile_id} daily dates are invalid or duplicated")
        frame = frame.sort_values("trade_date").reset_index(drop=True)
        if int(audit.get("days", -1)) != len(frame):
            raise ValueError(f"{profile_id} audit days do not match daily_account")
        dates = pd.DatetimeIndex(frame["trade_date"])
        if dates[0] <= training_end:
            raise ValueError(
                f"{profile_id} holdout overlaps frozen data boundary: "
                f"start={dates[0].date()} training_end={training_end.date()}"
            )
        if common_dates is None:
            common_dates = dates
        elif not common_dates.equals(dates):
            raise ValueError(f"{profile_id} does not use the common holdout calendar")
        metrics = calculate_metrics(frame)
        metrics.update(
            {
                "negative_cash_events": int(audit.get("negative_cash_events", -1)),
                "lot_violations": int(audit.get("lot_violations", -1)),
                "nav_mismatch_max": float(audit.get("nav_mismatch_max", np.inf)),
            }
        )
        overlay = profile.get("overlay") or {}
        alpha_manifest = None
        if overlay.get("kind") == "broad_loss5_or_alpha_health":
            if profile_id not in profile_manifests:
                raise ValueError(f"{profile_id} requires an alpha-health manifest")
            alpha_manifest = validate_alpha_health_manifest(
                profile_id,
                profile_manifests[profile_id].resolve(),
                profile,
                provenance,
            )
        rows[profile_id] = {
            "role": profile["role"],
            "log_dir": str(log_dir),
            "parameters": observed,
            "metrics": metrics,
            "audit": {
                "path": str(audit_path),
                "sha256": sha256_file(audit_path),
            },
            "daily_account": {
                "path": str(daily_path),
                "sha256": sha256_file(daily_path),
            },
            "source_provenance": provenance,
            "alpha_health_manifest": alpha_manifest,
        }

    assert common_dates is not None
    canonical_binding = build_canonical_binding(
        canonical_readiness_path,
        common_dates,
    )
    evidence = {
        "schema_version": 2,
        "status": "frozen_strategy_untouched_evaluation",
        "generated_at": datetime.now().astimezone().isoformat(),
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "training_data_end": str(training_end.date()),
        "holdout": {
            "status": "untouched_by_contract",
            "start": str(common_dates[0].date()),
            "end": str(common_dates[-1].date()),
            "sessions": int(len(common_dates)),
            "calendar_sha256": calendar_sha256(common_dates),
        },
        "canonical_market_data": canonical_binding,
        "verified_frozen_artifacts": verified_frozen,
        "profiles": rows,
        "deployment_allowed": False,
    }
    write_json(output_path, evidence)
    return evidence


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


def utility(metrics: dict[str, Any], weights: dict[str, Any]) -> float:
    return float(
        float(weights["return"]) * float(metrics["total_return"])
        + float(weights["sharpe"]) * float(metrics["sharpe"])
        - float(weights["drawdown"]) * float(metrics["max_drawdown"])
        - float(weights["turnover"]) * float(metrics["turnover"])
    )


def validate(
    contract_path: Path,
    evidence_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = load_json(contract_path)
    profiles = profile_map(contract)
    checks: list[dict[str, Any]] = []
    try:
        verified_frozen = verify_frozen_contract(contract)
        add_check(checks, "contract.frozen_artifacts", True, verified_frozen, "valid")
    except Exception as exc:
        verified_frozen = {}
        add_check(checks, "contract.frozen_artifacts", False, str(exc), "valid")

    evidence: dict[str, Any] = {}
    if evidence_path is None or not evidence_path.is_file():
        add_check(checks, "evidence.exists", False, False, True)
    else:
        evidence = load_json(evidence_path)
        try:
            evidence_schema = int(evidence.get("schema_version", 0))
        except (TypeError, ValueError):
            evidence_schema = 0
        add_check(
            checks,
            "evidence.schema_version",
            evidence_schema == 2,
            evidence_schema,
            2,
        )
        expected_contract_hash = sha256_file(contract_path)
        observed_contract_hash = str(
            (evidence.get("contract") or {}).get("sha256") or ""
        ).upper()
        add_check(
            checks,
            "evidence.contract_hash",
            observed_contract_hash == expected_contract_hash,
            observed_contract_hash,
            expected_contract_hash,
        )
        holdout = evidence.get("holdout") or {}
        try:
            canonical_binding = verify_canonical_binding(
                evidence.get("canonical_market_data"),
                holdout,
            )
            add_check(
                checks,
                "evidence.canonical_binding",
                True,
                canonical_binding,
                "valid and unchanged",
            )
        except Exception as exc:
            add_check(
                checks,
                "evidence.canonical_binding",
                False,
                str(exc),
                "valid and unchanged",
            )
        training_end = pd.Timestamp(contract["training_data_end"]).normalize()
        start = pd.to_datetime(holdout.get("start"), errors="coerce")
        end = pd.to_datetime(holdout.get("end"), errors="coerce")
        rules = contract["holdout_rules"]
        add_check(
            checks,
            "holdout.status",
            holdout.get("status") == "untouched_by_contract",
            holdout.get("status"),
            "untouched_by_contract",
        )
        add_check(
            checks,
            "holdout.after_training_boundary",
            pd.notna(start) and pd.Timestamp(start).normalize() > training_end,
            str(pd.Timestamp(start).date()) if pd.notna(start) else None,
            f"> {training_end.date()}",
        )
        add_check(
            checks,
            "holdout.valid_end",
            pd.notna(start) and pd.notna(end) and end >= start,
            str(pd.Timestamp(end).date()) if pd.notna(end) else None,
            ">= start",
        )
        add_check(
            checks,
            "holdout.min_sessions",
            int(holdout.get("sessions", 0)) >= int(rules["min_sessions"]),
            int(holdout.get("sessions", 0)),
            int(rules["min_sessions"]),
        )
        evidence_profiles = evidence.get("profiles") or {}
        add_check(
            checks,
            "evidence.exact_profiles",
            set(evidence_profiles) == set(profiles),
            sorted(evidence_profiles),
            sorted(profiles),
        )

    rules = contract["holdout_rules"]
    profile_results: dict[str, Any] = {}
    for profile_id, profile in profiles.items():
        row = (evidence.get("profiles") or {}).get(profile_id) or {}
        metrics = row.get("metrics") or {}
        profile_checks: list[dict[str, Any]] = []
        metric_rules = (
            ("sessions", ">=", rules["min_sessions"]),
            ("total_return", ">=", rules["min_total_return"]),
            ("sharpe", ">=", rules["min_sharpe"]),
            ("max_drawdown", "<=", rules["max_drawdown"]),
            ("max_daily_turnover", "<=", rules["max_daily_turnover"]),
            ("min_cash_ratio", ">=", rules["min_cash_ratio"]),
            ("max_gross_exposure", "<=", rules["max_gross_exposure"]),
            ("rebalance_count", ">=", rules["min_rebalance_count"]),
            ("trades", ">=", rules["min_trades"]),
            ("worst_month_return", ">=", rules["min_worst_month_return"]),
            ("positive_month_ratio", ">=", rules["min_positive_month_ratio"]),
            ("negative_cash_events", "<=", rules["max_negative_cash_events"]),
            ("lot_violations", "<=", rules["max_lot_violations"]),
            ("nav_mismatch_max", "<=", rules["max_nav_mismatch"]),
        )
        for name, operator, threshold in metric_rules:
            try:
                observed = float(metrics[name])
                finite = math.isfinite(observed)
            except (KeyError, TypeError, ValueError):
                observed = None
                finite = False
            passed = (
                finite
                and (
                    observed >= float(threshold)
                    if operator == ">="
                    else observed <= float(threshold)
                )
            )
            add_check(
                profile_checks,
                f"{profile_id}.{name}",
                passed,
                observed,
                {operator: threshold},
            )
        eligible = all(item["passed"] for item in profile_checks)
        profile_results[profile_id] = {
            "role": profile["role"],
            "eligible": eligible,
            "metrics": metrics,
            "checks": profile_checks,
            "utility": (
                utility(metrics, contract["promotion_rules"]["utility_weights"])
                if eligible
                else None
            ),
        }
        checks.extend(profile_checks)

    champion_id = next(
        profile_id
        for profile_id, profile in profiles.items()
        if profile["role"] == "champion"
    )
    champion = profile_results[champion_id]
    promotion_rules = contract["promotion_rules"]
    challenger_comparisons: dict[str, Any] = {}
    promotable: list[str] = []
    for profile_id, profile in profiles.items():
        if profile["role"] != "challenger":
            continue
        candidate = profile_results[profile_id]
        comparison_checks: list[dict[str, Any]] = []
        if champion["eligible"] and candidate["eligible"]:
            champion_metrics = champion["metrics"]
            candidate_metrics = candidate["metrics"]
            comparisons = (
                (
                    "return_delta",
                    float(candidate_metrics["total_return"])
                    - float(champion_metrics["total_return"]),
                    ">=",
                    promotion_rules["min_return_delta"],
                ),
                (
                    "drawdown_increase",
                    float(candidate_metrics["max_drawdown"])
                    - float(champion_metrics["max_drawdown"]),
                    "<=",
                    promotion_rules["max_drawdown_increase"],
                ),
                (
                    "turnover_ratio",
                    float(candidate_metrics["turnover"])
                    / max(float(champion_metrics["turnover"]), 1e-12),
                    "<=",
                    promotion_rules["max_turnover_ratio"],
                ),
                (
                    "worst_month_delta",
                    float(candidate_metrics["worst_month_return"])
                    - float(champion_metrics["worst_month_return"]),
                    ">=",
                    promotion_rules["min_worst_month_delta"],
                ),
                (
                    "utility_delta",
                    float(candidate["utility"]) - float(champion["utility"]),
                    ">=",
                    promotion_rules["min_utility_delta"],
                ),
            )
            for name, value, operator, threshold in comparisons:
                passed = (
                    value >= float(threshold)
                    if operator == ">="
                    else value <= float(threshold)
                )
                add_check(
                    comparison_checks,
                    f"{profile_id}.vs_champion.{name}",
                    passed,
                    value,
                    {operator: threshold},
                )
        else:
            add_check(
                comparison_checks,
                f"{profile_id}.vs_champion.eligible_profiles",
                False,
                {
                    "champion": champion["eligible"],
                    "challenger": candidate["eligible"],
                },
                {"champion": True, "challenger": True},
            )
        passes = all(item["passed"] for item in comparison_checks)
        if passes:
            promotable.append(profile_id)
        challenger_comparisons[profile_id] = {
            "passed": passes,
            "checks": comparison_checks,
        }
        checks.extend(comparison_checks)

    selected = champion_id if champion["eligible"] else None
    if promotable:
        selected = max(
            promotable,
            key=lambda profile_id: float(profile_results[profile_id]["utility"]),
        )
    failed = [item for item in checks if not item["passed"]]
    evidence_complete = bool(evidence) and not any(
        item["name"].startswith(("contract.", "evidence.", "holdout."))
        and not item["passed"]
        for item in checks
    )
    result = {
        "schema_version": 1,
        "status": "frozen_strategy_promotion_gate",
        "passed": evidence_complete and champion["eligible"],
        "paper_candidate_promotion_allowed": (
            evidence_complete and champion["eligible"] and selected is not None
        ),
        "real_money_deployment_allowed": False,
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "evidence": str(evidence_path.resolve()) if evidence_path else None,
        "evidence_sha256": (
            sha256_file(evidence_path)
            if evidence_path is not None and evidence_path.is_file()
            else None
        ),
        "canonical_market_data": evidence.get("canonical_market_data"),
        "training_data_end": contract.get("training_data_end"),
        "champion_profile_id": champion_id,
        "selected_profile_id": selected,
        "challenger_promotions": promotable,
        "profiles": profile_results,
        "challenger_comparisons": challenger_comparisons,
        "checks": checks,
        "failed_checks": failed,
        "next_action": (
            "build all frozen profiles on one new untouched interval"
            if not evidence_complete
            else "retain champion or package selected profile for PAPER audit"
        ),
    }
    write_json(output_path.resolve(), result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--contract", type=Path, required=True)
    build.add_argument("--profile-log", action="append", default=[])
    build.add_argument("--profile-manifest", action="append", default=[])
    build.add_argument("--canonical-readiness", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("validate")
    check.add_argument("--contract", type=Path, required=True)
    check.add_argument("--evidence", type=Path)
    check.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "build":
        evidence = build_evidence(
            args.contract,
            parse_mapping(args.profile_log, "profile-log"),
            parse_mapping(args.profile_manifest, "profile-manifest"),
            args.output,
            args.canonical_readiness,
        )
        print(
            "[strategy promotion] evidence "
            f"holdout={evidence['holdout']['start']}..{evidence['holdout']['end']} "
            f"sessions={evidence['holdout']['sessions']} output={args.output}",
            flush=True,
        )
        return
    result = validate(args.contract, args.evidence, args.output)
    print(
        "[strategy promotion] "
        f"passed={result['passed']} selected={result['selected_profile_id']} "
        f"failed={len(result['failed_checks'])} output={args.output}",
        flush=True,
    )
    for check in result["failed_checks"]:
        print(
            f"  FAIL {check['name']}: "
            f"value={check['value']} threshold={check['threshold']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
