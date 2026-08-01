from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inner_shadow_audit_contract import (
    AUDIT_SCHEMA_VERSION,
    AUDIT_TABLE_COLUMNS,
    EXPECTED_DECISION_TIMES,
    MAX_DECISION_LATENCY_SECONDS,
    MAX_FINALIZE_LATENCY_SECONDS,
    REGISTRY_FILENAME,
    REQUIRED_AUDIT_TABLES,
    SESSION_FINALIZE_TIME,
    clock_seconds,
    registry_record_hash,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_AUDIT_ROOT = REPO_ROOT / "outputs" / "gm_inner_t0_0945_1000_shadow_audit"
DEFAULT_FORBIDDEN = (
    REPO_ROOT
    / "outputs"
    / "gmquant_inner_t0_0945_1000_shadow_candidate"
    / "gm_c_forbidden_symbols.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "inner_t0_0945_1000_shadow_audit_compare_latest.json"
)
DEFAULT_ACCOUNT_ID = os.environ.get("GM_ACCOUNT_ID", "").strip()
DEFAULT_STRESS_SLIPPAGE_RATE = 0.001


def local_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if "." in text:
        exchange, code = text.split(".", 1)
        return ("SH" if exchange in {"SHSE", "SH"} else "SZ") + code
    if text.startswith(("SH", "SZ")):
        return text
    code = text[-6:]
    return ("SH" if code.startswith(("6", "9")) else "SZ") + code


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_read_error": str(exc)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_registry(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], [f"audit registry missing: {path}"]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    previous_hash = ""
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"registry line {line_number} is invalid JSON: {exc}")
            continue
        if record.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
            errors.append(f"registry line {line_number} schema version mismatch")
        if record.get("previous_hash", "") != previous_hash:
            errors.append(f"registry chain break at line {line_number}")
        expected = registry_record_hash(record)
        observed = str(record.get("record_hash", "")).upper()
        if observed != expected:
            errors.append(f"registry record hash mismatch at line {line_number}")
        previous_hash = observed
        records.append(record)
    if not records and not errors:
        errors.append("audit registry is empty")
    return records, errors


def discover_runs(audit_root: Path, explicit_runs: list[Path]) -> list[Path]:
    if explicit_runs:
        return [path.resolve() for path in explicit_runs if path.exists()]
    elif audit_root.exists():
        runs = sorted(
            path
            for path in audit_root.iterdir()
            if path.is_dir()
            and not path.name.lower().startswith("mock_")
            and (path / "summary.json").exists()
        )
    else:
        runs = []
    return [path for path in runs if path.exists() and (path / "summary.json").exists()]


def concatenate_runs(runs: list[Path], filename: str) -> pd.DataFrame:
    parts = []
    for run in runs:
        frame = read_csv_or_empty(run / filename)
        if frame.empty:
            continue
        frame["audit_run_id"] = run.name
        parts.append(frame)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def load_forbidden(path: Path) -> set[str]:
    frame = read_csv_or_empty(path)
    if frame.empty:
        return set()
    values: set[str] = set()
    for column in frame.columns:
        lowered = column.lower()
        if any(token in lowered for token in ("instrument", "symbol", "code")):
            values.update(frame[column].dropna().map(local_symbol))
    return {value for value in values if value}


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    source = out["local_symbol"] if "local_symbol" in out else out["symbol"]
    out["instrument"] = source.map(local_symbol)
    return out


def profit_factor(pnl: pd.Series) -> float | None:
    values = pd.to_numeric(pnl, errors="coerce").dropna()
    gain = float(values[values > 0].sum())
    loss = float(-values[values < 0].sum())
    if loss <= 0:
        return None if gain <= 0 else float("inf")
    return gain / loss


def concentration_metrics(
    exits: pd.DataFrame,
    pnl_column: str = "virtual_pnl",
) -> dict[str, Any]:
    if exits.empty:
        return {
            "symbols": 0,
            "active_days": 0,
            "active_months": 0,
            "losing_months": 0,
            "top_symbol_positive_pnl_share": None,
            "top3_positive_pnl_share": None,
    }
    frame = exits.copy()
    frame[pnl_column] = pd.to_numeric(frame[pnl_column], errors="coerce").fillna(0.0)
    frame["month"] = pd.to_datetime(frame["trade_date"]).dt.to_period("M").astype(str)
    symbol_pnl = frame.groupby("instrument")[pnl_column].sum().sort_values(ascending=False)
    positive = symbol_pnl[symbol_pnl > 0]
    positive_total = float(positive.sum())
    monthly = frame.groupby("month")[pnl_column].sum()
    return {
        "symbols": int(frame["instrument"].nunique()),
        "active_days": int(frame["trade_date"].nunique()),
        "active_months": int(frame["month"].nunique()),
        "losing_months": int((monthly < 0).sum()),
        "top_symbol_positive_pnl_share": (
            float(positive.iloc[0] / positive_total) if positive_total > 0 else None
        ),
        "top3_positive_pnl_share": (
            float(positive.head(3).sum() / positive_total) if positive_total > 0 else None
        ),
    }


def _truthy(value: object) -> bool:
    return str(value).strip().upper() in {"TRUE", "1", "YES"}


def _int_value(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _validate_run_bundle(
    run: Path,
    summary: dict[str, Any],
    errors: list[str],
    *,
    expected_account_id: str,
    required_run_mode: str,
    max_decision_latency_seconds: int,
    max_finalize_latency_seconds: int,
) -> tuple[dict[str, pd.DataFrame], list[str], list[dict[str, Any]]]:
    label = run.name
    frames: dict[str, pd.DataFrame] = {}
    if summary.get("_read_error"):
        errors.append(f"{label}: summary.json is unreadable: {summary['_read_error']}")
    if summary.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
        errors.append(f"{label}: audit schema version mismatch")
    if summary.get("audit_run_id") != label:
        errors.append(f"{label}: summary audit_run_id mismatch")
    if summary.get("run_mode") != required_run_mode:
        errors.append(
            f"{label}: run_mode={summary.get('run_mode')!r} "
            f"required={required_run_mode!r}"
        )
    if summary.get("account_id") != expected_account_id:
        errors.append(f"{label}: PAPER account id mismatch")
    for key in (
        "model_manifest_sha256",
        "target_context_sha256",
        "forbidden_sha256",
    ):
        value = str(summary.get(key, ""))
        if len(value) != 64:
            errors.append(f"{label}: missing or invalid {key}")

    audit_files = summary.get("audit_files")
    if not isinstance(audit_files, dict):
        errors.append(f"{label}: summary.audit_files is missing")
        audit_files = {}
    for name in REQUIRED_AUDIT_TABLES:
        path = run / name
        if not path.is_file():
            errors.append(f"{label}: required audit table missing: {name}")
            frames[name] = pd.DataFrame()
            continue
        frame = read_csv_or_empty(path)
        missing_columns = sorted(
            set(AUDIT_TABLE_COLUMNS[name]).difference(frame.columns)
        )
        if missing_columns:
            errors.append(
                f"{label}: audit table {name} missing columns: {missing_columns}"
            )
            for column in missing_columns:
                frame[column] = pd.NA
        frames[name] = frame
        declared = audit_files.get(name)
        if not isinstance(declared, dict):
            errors.append(f"{label}: audit manifest missing {name}")
            continue
        if _int_value(declared.get("rows")) != len(frame):
            errors.append(f"{label}: audit row count mismatch: {name}")
        if str(declared.get("sha256", "")).upper() != sha256_file(path):
            errors.append(f"{label}: audit hash mismatch: {name}")

    count_contract = {
        "decision_scores.csv": "decision_score_rows",
        "candidate_intents.csv": "candidate_rows",
        "entry_intents.csv": "entry_rows",
        "exit_intents.csv": "exit_rows",
        "runtime_events.csv": "runtime_event_rows",
    }
    for filename, field in count_contract.items():
        if _int_value(summary.get(field)) != len(frames.get(filename, pd.DataFrame())):
            errors.append(f"{label}: summary count mismatch: {field}")
    entries = frames.get("entry_intents.csv", pd.DataFrame())
    exits = frames.get("exit_intents.csv", pd.DataFrame())
    triggered = (
        int(entries["action"].astype(str).eq("SELL_FIRST_TRIGGERED").sum())
        if "action" in entries
        else 0
    )
    successful = (
        int(exits["action"].astype(str).eq("BUYBACK_INTENT").sum())
        if "action" in exits
        else 0
    )
    if _int_value(summary.get("entry_triggered_rows")) != triggered:
        errors.append(f"{label}: entry_triggered_rows mismatch")
    if _int_value(summary.get("successful_exit_intents")) != successful:
        errors.append(f"{label}: successful_exit_intents mismatch")

    complete_dates = [str(value) for value in summary.get("complete_session_dates", [])]
    incomplete_dates = [str(value) for value in summary.get("incomplete_session_dates", [])]
    if summary.get("session_complete") is not True:
        errors.append(f"{label}: session_complete is not true")
    if incomplete_dates:
        errors.append(f"{label}: incomplete sessions present: {incomplete_dates}")
    if len(complete_dates) != 1:
        errors.append(
            f"{label}: each live audit run must contain exactly one complete session"
        )
    if _int_value(summary.get("observation_session_count")) != len(complete_dates):
        errors.append(f"{label}: observation_session_count mismatch")
    if complete_dates and summary.get("target_source_date") != complete_dates[0]:
        errors.append(f"{label}: target source date does not match observed session")

    details = summary.get("session_details")
    if not isinstance(details, list):
        errors.append(f"{label}: session_details is missing")
        details = []
    detail_dates = [str(row.get("date")) for row in details if isinstance(row, dict)]
    if sorted(detail_dates) != sorted(complete_dates + incomplete_dates):
        errors.append(f"{label}: session_details date set mismatch")
    for row in details:
        if not isinstance(row, dict) or row.get("complete") is not True:
            errors.append(f"{label}: session detail is not complete: {row}")

    events = frames.get("runtime_events.csv", pd.DataFrame()).copy()
    latency_rows: list[dict[str, Any]] = []
    if events.empty or not {"date", "time", "event"}.issubset(events.columns):
        errors.append(f"{label}: runtime events lack the completeness contract")
        return frames, complete_dates, latency_rows
    events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    blocked = events[events["event"].astype(str).eq("DECISION_BLOCKED")]
    if not blocked.empty:
        errors.append(f"{label}: decision pipeline contains blocked decisions")
    for trade_date in complete_dates:
        date_events = events[events["date"].eq(trade_date)]
        for component, expected_time in EXPECTED_DECISION_TIMES.items():
            rows = date_events[
                date_events["event"].astype(str).eq("DECISION_COMPLETE")
                & date_events.get("component", pd.Series(index=date_events.index, dtype=str))
                .astype(str)
                .eq(component)
            ]
            if len(rows) != 1:
                errors.append(
                    f"{label}: {trade_date} requires exactly one completed {component} decision"
                )
                continue
            observed_time = rows.iloc[0]["time"]
            observed_seconds = clock_seconds(observed_time)
            expected_seconds = clock_seconds(expected_time)
            latency = (
                observed_seconds - expected_seconds
                if observed_seconds is not None and expected_seconds is not None
                else None
            )
            latency_rows.append(
                {
                    "run_id": label,
                    "date": trade_date,
                    "component": component,
                    "expected_time": expected_time,
                    "observed_time": str(observed_time),
                    "latency_seconds": latency,
                }
            )
            if latency is None or not (0 <= latency <= max_decision_latency_seconds):
                errors.append(
                    f"{label}: {component} decision latency invalid: {latency} seconds"
                )
        final_rows = date_events[
            date_events["event"].astype(str).eq("SESSION_FINALIZED")
        ]
        if len(final_rows) != 1:
            errors.append(f"{label}: {trade_date} requires one SESSION_FINALIZED event")
        else:
            final = final_rows.iloc[0]
            observed_seconds = clock_seconds(final["time"])
            expected_seconds = clock_seconds(SESSION_FINALIZE_TIME)
            latency = (
                observed_seconds - expected_seconds
                if observed_seconds is not None and expected_seconds is not None
                else None
            )
            latency_rows.append(
                {
                    "run_id": label,
                    "date": trade_date,
                    "component": "session_finalize",
                    "expected_time": SESSION_FINALIZE_TIME,
                    "observed_time": str(final["time"]),
                    "latency_seconds": latency,
                }
            )
            if not _truthy(final.get("complete")):
                errors.append(f"{label}: SESSION_FINALIZED is not complete")
            if latency is None or not (0 <= latency <= max_finalize_latency_seconds):
                errors.append(f"{label}: session finalization latency invalid: {latency}")
    return frames, complete_dates, latency_rows


def compare_shadow_audits(
    runs: list[Path],
    forbidden_path: Path,
    output_path: Path,
    *,
    minimum_round_trips: int = 80,
    minimum_symbols: int = 20,
    minimum_active_days: int = 20,
    minimum_active_months: int = 3,
    minimum_observation_sessions: int = 20,
    minimum_profit_factor: float = 1.2,
    maximum_losing_months: int = 1,
    maximum_top3_positive_pnl_share: float = 0.60,
    expected_account_id: str = DEFAULT_ACCOUNT_ID,
    required_run_mode: str = "LIVE",
    max_decision_latency_seconds: int = MAX_DECISION_LATENCY_SECONDS,
    max_finalize_latency_seconds: int = MAX_FINALIZE_LATENCY_SECONDS,
    stress_slippage_rate: float = DEFAULT_STRESS_SLIPPAGE_RATE,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    if not np.isfinite(stress_slippage_rate) or stress_slippage_rate < 0.0:
        raise ValueError("stress_slippage_rate must be finite and non-negative")
    runs = sorted({path.resolve() for path in runs})
    summaries = [read_json(run / "summary.json") for run in runs]
    errors: list[str] = []
    warnings: list[str] = []
    complete_sessions: list[dict[str, str]] = []
    latency_rows: list[dict[str, Any]] = []
    run_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for run, summary in zip(runs, summaries):
        frames, complete_dates, run_latency = _validate_run_bundle(
            run,
            summary,
            errors,
            expected_account_id=expected_account_id,
            required_run_mode=required_run_mode,
            max_decision_latency_seconds=max_decision_latency_seconds,
            max_finalize_latency_seconds=max_finalize_latency_seconds,
        )
        run_frames[run.name] = frames
        complete_sessions.extend(
            {"run_id": run.name, "date": date} for date in complete_dates
        )
        latency_rows.extend(run_latency)

    registry_records: list[dict[str, Any]] = []
    registry_file: Path | None = None
    if runs:
        parents = {run.parent.resolve() for run in runs}
        if len(parents) != 1:
            errors.append("audit runs must belong to one immutable campaign root")
        else:
            audit_root = next(iter(parents))
            all_runs = set(discover_runs(audit_root, []))
            selected_runs = set(runs)
            if all_runs != selected_runs:
                errors.append(
                    "selective audit subset is forbidden; include every run in the campaign root"
                )
            registry_file = (
                registry_path.resolve()
                if registry_path is not None
                else audit_root / REGISTRY_FILENAME
            )
            registry_records, registry_errors = read_registry(registry_file)
            errors.extend(registry_errors)
            starts = [
                row for row in registry_records if row.get("event") == "RUN_STARTED"
            ]
            started_ids = [str(row.get("run_id", "")) for row in starts]
            if len(started_ids) != len(set(started_ids)):
                errors.append("audit registry contains duplicate RUN_STARTED ids")
            run_ids = {run.name for run in runs}
            missing_runs = sorted(set(started_ids).difference(run_ids))
            unregistered_runs = sorted(run_ids.difference(started_ids))
            if missing_runs:
                errors.append(
                    f"registered runs missing from audit comparison: {missing_runs}"
                )
            if unregistered_runs:
                errors.append(f"audit runs are not registered: {unregistered_runs}")
            start_by_id = {str(row.get("run_id")): row for row in starts}
            final_records = [
                row
                for row in registry_records
                if row.get("event") == "SESSION_FINALIZED"
            ]
            for run, summary in zip(runs, summaries):
                start = start_by_id.get(run.name)
                if start is None:
                    continue
                for field in (
                    "account_id",
                    "run_mode",
                    "model_manifest_sha256",
                    "target_context_sha256",
                    "forbidden_sha256",
                ):
                    if start.get(field) != summary.get(field):
                        errors.append(
                            f"{run.name}: registry/summary mismatch for {field}"
                        )
                for date in summary.get("complete_session_dates", []):
                    matching = [
                        row
                        for row in final_records
                        if row.get("run_id") == run.name and row.get("date") == date
                    ]
                    if len(matching) != 1 or matching[0].get("complete") is not True:
                        errors.append(
                            f"{run.name}: complete session lacks one valid registry finalization"
                        )
    elif registry_path is not None and registry_path.resolve().is_file():
        registry_file = registry_path.resolve()
        registry_records, registry_errors = read_registry(registry_file)
        errors.extend(registry_errors)
        started_ids = sorted(
            {
                str(row.get("run_id", ""))
                for row in registry_records
                if row.get("event") == "RUN_STARTED"
            }
        )
        if started_ids:
            errors.append(
                f"registered runs have no readable summary bundle: {started_ids}"
            )

    model_hashes = {
        str(summary.get("model_manifest_sha256"))
        for summary in summaries
        if summary.get("model_manifest_sha256")
    }
    if len(model_hashes) > 1:
        errors.append("audit campaign mixes different frozen inner model manifests")
    scores = _normalize_keys(concatenate_runs(runs, "decision_scores.csv"))
    candidates = _normalize_keys(concatenate_runs(runs, "candidate_intents.csv"))
    entries = _normalize_keys(concatenate_runs(runs, "entry_intents.csv"))
    exits = _normalize_keys(concatenate_runs(runs, "exit_intents.csv"))
    events = concatenate_runs(runs, "runtime_events.csv")
    forbidden = load_forbidden(forbidden_path)
    for index, summary in enumerate(summaries):
        label = runs[index].name
        if summary.get("dry_run") is not True:
            errors.append(f"{label}: summary.dry_run is not true")
        for key in (
            "actual_submission_api_present",
            "paper_orders_allowed",
            "main_py_integration_allowed",
            "deployment_allowed",
        ):
            if summary.get(key) is not False:
                errors.append(f"{label}: summary.{key} must be false")
        if summary.get("target_age_days") != 0:
            errors.append(f"{label}: target_age_days is not zero")
        signal_age = summary.get("signal_age_days")
        max_signal_age = summary.get("max_signal_age_days", 4)
        if signal_age is None or not (
            1 <= _int_value(signal_age) <= _int_value(max_signal_age)
        ):
            errors.append(f"{label}: signal age is not valid")
    non_dry_entries = pd.DataFrame()
    if not entries.empty and "dry_run" in entries:
        non_dry_entries = entries[
            entries["dry_run"].astype(str).str.upper().isin({"FALSE", "0", "NO"})
        ]
        if not non_dry_entries.empty:
            errors.append("entry_intents contains non-dry-run rows")
    selected_candidates = (
        candidates[candidates.get("action", "").astype(str).eq("COMPONENT_CANDIDATE")].copy()
        if not candidates.empty and "action" in candidates
        else pd.DataFrame()
    )
    primary = (
        selected_candidates[
            selected_candidates.get("component", "").astype(str).eq("0945_high_confidence")
        ]
        if not selected_candidates.empty
        else pd.DataFrame()
    )
    secondary = (
        selected_candidates[
            selected_candidates.get("component", "").astype(str).eq("1000_daily_ridge_gate")
        ]
        if not selected_candidates.empty
        else pd.DataFrame()
    )
    if not primary.empty:
        below = pd.to_numeric(primary["score"], errors="coerce") < 0.005
        if below.any():
            errors.append("09:45 candidates contain score below frozen 0.5% edge gate")
    if not secondary.empty:
        gate_score = pd.to_numeric(secondary["meta_gate_score"], errors="coerce")
        if (gate_score <= 0.0).any() or gate_score.isna().any():
            errors.append("10:00 candidates contain disabled/non-positive Ridge gate")
    candidate_forbidden = (
        sorted(set(selected_candidates["instrument"]) & forbidden)
        if not selected_candidates.empty
        else []
    )
    if candidate_forbidden:
        errors.append("component candidates contain forbidden symbols")
    accepted = (
        entries[entries.get("action", "").astype(str).eq("SELL_FIRST_TRIGGERED")].copy()
        if not entries.empty and "action" in entries
        else pd.DataFrame()
    )
    if not accepted.empty and accepted.duplicated(
        ["trade_date", "instrument"], keep=False
    ).any():
        errors.append("accepted entries contain duplicate date/symbol keys")
    if not accepted.empty:
        trigger_price = pd.to_numeric(accepted["trigger_price"], errors="coerce")
        entry_limit = pd.to_numeric(accepted["entry_limit"], errors="coerce")
        if (trigger_price + 1e-12 < entry_limit).any():
            errors.append("accepted entry triggered below sell-first limit")
        projected_turnover = pd.to_numeric(
            accepted["projected_daily_turnover"], errors="coerce"
        )
        if (projected_turnover > 0.03 + 1e-12).any():
            errors.append("accepted entry exceeds 3% daily turnover cap")
        volume = pd.to_numeric(accepted["volume"], errors="coerce")
        held = pd.to_numeric(accepted["held_volume"], errors="coerce")
        if (volume > held * 0.5 + 1e-12).any():
            errors.append("accepted entry exceeds half held inventory")
    entry_forbidden = (
        sorted(set(accepted["instrument"]) & forbidden) if not accepted.empty else []
    )
    if entry_forbidden:
        errors.append("accepted entries contain forbidden symbols")
    successful_exits = (
        exits[exits.get("action", "").astype(str).eq("BUYBACK_INTENT")].copy()
        if not exits.empty and "action" in exits
        else pd.DataFrame()
    )
    if not successful_exits.empty and successful_exits.duplicated(
        ["trade_date", "instrument"], keep=False
    ).any():
        errors.append("successful buybacks contain duplicate date/symbol keys")
    blocked_exits = (
        exits[~exits.index.isin(successful_exits.index)].copy()
        if not exits.empty
        else pd.DataFrame()
    )
    entry_keys = (
        accepted[["trade_date", "instrument"]].drop_duplicates()
        if not accepted.empty
        else pd.DataFrame(columns=["trade_date", "instrument"])
    )
    exit_keys = (
        successful_exits[["trade_date", "instrument"]].drop_duplicates()
        if not successful_exits.empty
        else pd.DataFrame(columns=["trade_date", "instrument"])
    )
    matched = entry_keys.merge(
        exit_keys,
        on=["trade_date", "instrument"],
        how="left",
        indicator=True,
    )
    unmatched = matched[matched["_merge"] == "left_only"].copy()
    if not unmatched.empty:
        errors.append("accepted entries lack same-day successful buyback intent")
    if not events.empty and "combined_candidates" in events:
        combined = pd.to_numeric(events["combined_candidates"], errors="coerce").dropna()
        if not combined.empty and int(combined.max()) > 4:
            errors.append("combined candidate count exceeds four")
    economics_frame = successful_exits.copy()
    required_economic_columns = (
        "entry_value_ref",
        "exit_value_ref",
        "virtual_pnl",
    )
    if not economics_frame.empty:
        missing_economic_columns = [
            column
            for column in required_economic_columns
            if column not in economics_frame
        ]
        if missing_economic_columns:
            errors.append(
                "successful exits lack economic fields: "
                f"{missing_economic_columns}"
            )
            economics_frame = economics_frame.iloc[0:0].copy()
        else:
            numeric_economics = economics_frame[
                list(required_economic_columns)
            ].apply(pd.to_numeric, errors="coerce")
            invalid_economics = numeric_economics.isna().any(axis=1)
            if invalid_economics.any():
                errors.append(
                    "successful exits contain non-numeric economic fields: "
                    f"{int(invalid_economics.sum())} rows"
                )
            economics_frame = economics_frame.loc[~invalid_economics].copy()
            economics_frame[list(required_economic_columns)] = (
                numeric_economics.loc[~invalid_economics]
            )
    if not economics_frame.empty:
        economics_frame["stress_slippage_cost"] = stress_slippage_rate * (
            economics_frame["entry_value_ref"]
            + economics_frame["exit_value_ref"]
        )
        economics_frame["stress_pnl"] = (
            economics_frame["virtual_pnl"]
            - economics_frame["stress_slippage_cost"]
        )
    else:
        economics_frame["stress_slippage_cost"] = pd.Series(dtype=float)
        economics_frame["stress_pnl"] = pd.Series(dtype=float)

    pnl = economics_frame.get("virtual_pnl", pd.Series(dtype=float))
    stress_pnl = economics_frame["stress_pnl"]
    cumulative_pnl = float(pnl.sum())
    stress_cumulative_pnl = float(stress_pnl.sum())
    stress_slippage_cost = float(economics_frame["stress_slippage_cost"].sum())
    pf = profit_factor(pnl)
    stress_pf = profit_factor(stress_pnl)
    positive_ratio = float((pnl > 0).mean()) if len(pnl) else None
    stress_positive_ratio = (
        float((stress_pnl > 0).mean()) if len(stress_pnl) else None
    )
    concentration = concentration_metrics(economics_frame)
    stress_concentration = concentration_metrics(
        economics_frame,
        pnl_column="stress_pnl",
    )
    economic_checks = {
        "minimum_observation_sessions": (
            len(complete_sessions) >= minimum_observation_sessions
        ),
        "minimum_round_trips": int(len(successful_exits)) >= minimum_round_trips,
        "positive_stress_cumulative_pnl": stress_cumulative_pnl > 0.0,
        "minimum_stress_profit_factor": (
            stress_pf is not None and stress_pf >= minimum_profit_factor
        ),
        "minimum_symbols": stress_concentration["symbols"] >= minimum_symbols,
        "minimum_active_days": (
            stress_concentration["active_days"] >= minimum_active_days
        ),
        "minimum_active_months": (
            stress_concentration["active_months"] >= minimum_active_months
        ),
        "maximum_stress_losing_months": (
            stress_concentration["losing_months"] <= maximum_losing_months
        ),
        "maximum_top3_positive_pnl_share": (
            stress_concentration["top3_positive_pnl_share"] is not None
            and stress_concentration["top3_positive_pnl_share"]
            <= maximum_top3_positive_pnl_share
        ),
        "no_unmatched_buybacks": unmatched.empty,
        "no_blocked_buybacks": blocked_exits.empty,
    }
    technical_passed = bool(runs) and not errors
    economic_passed = technical_passed and all(economic_checks.values())
    if not runs and not errors:
        status = "waiting_for_real_shadow_audit"
    elif not technical_passed:
        status = "shadow_technical_audit_failed"
    elif not economic_passed:
        status = "research_only_collect_more_shadow_evidence"
    else:
        status = "shadow_gate_passed_pending_untouched_replay_and_paper_decision"
    if accepted.empty:
        warnings.append("no accepted virtual entries observed")
    report = {
        "status": status,
        "audit_runs": [str(path.resolve()) for path in runs],
        "technical_passed": technical_passed,
        "economic_shadow_gate_passed": economic_passed,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "runs": len(runs),
            "registry_records": len(registry_records),
            "complete_observation_sessions": len(complete_sessions),
            "decision_scores": int(len(scores)),
            "component_candidates": int(len(selected_candidates)),
            "accepted_entries": int(len(accepted)),
            "successful_buybacks": int(len(successful_exits)),
            "blocked_buybacks": int(len(blocked_exits)),
            "unmatched_buybacks": int(len(unmatched)),
        },
        "virtual_economics": {
            "cumulative_pnl": cumulative_pnl,
            "profit_factor": pf,
            "positive_trade_ratio": positive_ratio,
            **concentration,
        },
        "stress_economics": {
            "additional_slippage_cost": stress_slippage_cost,
            "cumulative_pnl": stress_cumulative_pnl,
            "profit_factor": stress_pf,
            "positive_trade_ratio": stress_positive_ratio,
            **stress_concentration,
        },
        "economic_thresholds": {
            "minimum_observation_sessions": minimum_observation_sessions,
            "minimum_round_trips": minimum_round_trips,
            "minimum_symbols": minimum_symbols,
            "minimum_active_days": minimum_active_days,
            "minimum_active_months": minimum_active_months,
            "minimum_profit_factor": minimum_profit_factor,
            "maximum_losing_months": maximum_losing_months,
            "maximum_top3_positive_pnl_share": maximum_top3_positive_pnl_share,
            "stress_slippage_rate_per_side": stress_slippage_rate,
        },
        "economic_checks": economic_checks,
        "audit_integrity": {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "registry": str(registry_file) if registry_file is not None else None,
            "registered_run_ids": sorted(
                {
                    str(row.get("run_id"))
                    for row in registry_records
                    if row.get("event") == "RUN_STARTED"
                }
            ),
            "complete_sessions": complete_sessions,
            "model_manifest_sha256": (
                next(iter(model_hashes)) if len(model_hashes) == 1 else None
            ),
            "required_run_mode": required_run_mode,
            "expected_account_id": expected_account_id,
            "max_decision_latency_seconds": max_decision_latency_seconds,
            "max_finalize_latency_seconds": max_finalize_latency_seconds,
            "latency_observations": latency_rows,
            "selective_subset_allowed": False,
        },
        "candidate_forbidden_hits": candidate_forbidden,
        "entry_forbidden_hits": entry_forbidden,
        "unmatched_entries": unmatched.head(100).to_dict("records"),
        "blocked_exit_rows": blocked_exits.head(100).to_dict("records"),
        "paper_orders_allowed": False,
        "main_py_integration_allowed": False,
        "deployment_allowed": False,
        "next_action": (
            "keep_collecting_no_order_shadow"
            if not economic_passed
            else "run_new_untouched_portfolio_replay_before_any_paper_order_integration"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    parser.add_argument("--audit-run", action="append", default=[])
    parser.add_argument("--forbidden", default=str(DEFAULT_FORBIDDEN))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--minimum-round-trips", type=int, default=80)
    parser.add_argument("--minimum-symbols", type=int, default=20)
    parser.add_argument("--minimum-active-days", type=int, default=20)
    parser.add_argument("--minimum-active-months", type=int, default=3)
    parser.add_argument("--minimum-observation-sessions", type=int, default=20)
    parser.add_argument("--minimum-profit-factor", type=float, default=1.2)
    parser.add_argument("--maximum-losing-months", type=int, default=1)
    parser.add_argument("--maximum-top3-positive-pnl-share", type=float, default=0.60)
    parser.add_argument(
        "--stress-slippage-rate",
        type=float,
        default=DEFAULT_STRESS_SLIPPAGE_RATE,
        help="additional adverse slippage per side, applied after recorded fees",
    )
    parser.add_argument("--expected-account-id", default=DEFAULT_ACCOUNT_ID)
    parser.add_argument("--required-run-mode", default="LIVE")
    parser.add_argument(
        "--max-decision-latency-seconds",
        type=int,
        default=MAX_DECISION_LATENCY_SECONDS,
    )
    parser.add_argument(
        "--max-finalize-latency-seconds",
        type=int,
        default=MAX_FINALIZE_LATENCY_SECONDS,
    )
    parser.add_argument("--registry", default="")
    args = parser.parse_args()
    audit_root = Path(args.audit_root).resolve()
    runs = discover_runs(
        audit_root,
        [Path(value).resolve() for value in args.audit_run],
    )
    report = compare_shadow_audits(
        runs,
        Path(args.forbidden).resolve(),
        Path(args.output).resolve(),
        minimum_round_trips=args.minimum_round_trips,
        minimum_symbols=args.minimum_symbols,
        minimum_active_days=args.minimum_active_days,
        minimum_active_months=args.minimum_active_months,
        minimum_observation_sessions=args.minimum_observation_sessions,
        minimum_profit_factor=args.minimum_profit_factor,
        maximum_losing_months=args.maximum_losing_months,
        maximum_top3_positive_pnl_share=args.maximum_top3_positive_pnl_share,
        expected_account_id=args.expected_account_id,
        required_run_mode=args.required_run_mode,
        max_decision_latency_seconds=args.max_decision_latency_seconds,
        max_finalize_latency_seconds=args.max_finalize_latency_seconds,
        stress_slippage_rate=args.stress_slippage_rate,
        registry_path=(
            Path(args.registry).resolve()
            if args.registry
            else audit_root / REGISTRY_FILENAME
        ),
    )
    print(
        f"[inner shadow compare] status={report['status']} "
        f"technical={report['technical_passed']} "
        f"economic={report['economic_shadow_gate_passed']} "
        f"round_trips={report['counts']['successful_buybacks']} "
        f"raw_pnl={report['virtual_economics']['cumulative_pnl']:.2f} "
        f"stress_pnl={report['stress_economics']['cumulative_pnl']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
