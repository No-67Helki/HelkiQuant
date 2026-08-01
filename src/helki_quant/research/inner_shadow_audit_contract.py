from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


AUDIT_SCHEMA_VERSION = 2
REGISTRY_FILENAME = "RUN_REGISTRY.jsonl"
REQUIRED_AUDIT_TABLES = (
    "decision_scores.csv",
    "candidate_intents.csv",
    "entry_intents.csv",
    "exit_intents.csv",
    "runtime_events.csv",
)
AUDIT_TABLE_COLUMNS = {
    "decision_scores.csv": (
        "date",
        "time",
        "component",
        "symbol",
        "local_symbol",
        "raw_score",
        "model_score",
    ),
    "candidate_intents.csv": (
        "date",
        "time",
        "component",
        "symbol",
        "local_symbol",
        "score",
        "meta_gate_score",
        "action",
        "dry_run",
    ),
    "entry_intents.csv": (
        "date",
        "time",
        "component",
        "symbol",
        "local_symbol",
        "trigger_price",
        "entry_limit",
        "projected_daily_turnover",
        "volume",
        "held_volume",
        "action",
        "dry_run",
    ),
    "exit_intents.csv": (
        "date",
        "time",
        "component",
        "symbol",
        "local_symbol",
        "entry_value_ref",
        "exit_value_ref",
        "virtual_pnl",
        "action",
        "dry_run",
    ),
    "runtime_events.csv": (
        "date",
        "time",
        "event",
    ),
}
EXPECTED_DECISION_TIMES = {
    "0945_high_confidence": "09:45:05",
    "1000_daily_ridge_gate": "10:00:05",
}
SESSION_FINALIZE_TIME = "14:51:00"
MAX_DECISION_LATENCY_SECONDS = 120
MAX_FINALIZE_LATENCY_SECONDS = 240


def clock_seconds(value: object) -> int | None:
    text = str(value or "").strip()
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hour, minute, second = (int(part) for part in parts)
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    return hour * 3600 + minute * 60 + second


def registry_record_hash(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_hash", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()
