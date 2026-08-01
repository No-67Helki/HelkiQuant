# coding=utf-8
"""GmQuant held-only 09:45 + 10:00 T+0 no-order observer.

This research-only observer never submits orders. It applies the exact frozen
09:45 high-confidence model and 10:00 stock model plus daily Ridge gate to
actual non-ST holdings, records virtual sell-first triggers through 11:00, and
records inventory-restoring buyback intents at 14:45. The active outer+middle
PAPER strategy remains separate.
"""
from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gm.api import *  # noqa: F401,F403
from catboost import CatBoostClassifier

from held_intraday_live_features import (
    add_held_cross_sectional_features,
    build_position_features,
    current_session_bars,
    finite_float,
    normalize_gm_minute_bars,
    visible_features,
)
from inner_t0_multidecision_shadow_engine import (
    build_buyback_intent,
    build_daily_meta_values,
    build_sell_entry_intent,
    combine_primary_secondary,
    isotonic_score,
    require_fresh_signal,
    require_fresh_target,
    ridge_gate_score,
    select_sell_first_candidates,
    trigger_reached,
)
from inner_shadow_audit_contract import (
    AUDIT_SCHEMA_VERSION,
    AUDIT_TABLE_COLUMNS,
    EXPECTED_DECISION_TIMES,
    REGISTRY_FILENAME,
    REQUIRED_AUDIT_TABLES,
    SESSION_FINALIZE_TIME,
    registry_record_hash,
)


ROOT = Path(__file__).resolve().parent
GM_TOKEN = os.environ.get("GM_TOKEN", "").strip()
GM_STRATEGY_ID = os.environ.get(
    "GM_STRATEGY_ID", "inner-t0-0945-1000-no-order-shadow"
).strip()
GM_ACCOUNT_ID = os.environ.get("GM_ACCOUNT_ID", "").strip()
RUN_MODE_NAME = os.environ.get("GM_INNER_SHADOW_MODE", "LIVE").strip().upper()
DRY_RUN = os.environ.get("GM_INNER_SHADOW_DRY_RUN", "1").upper() not in {"0", "FALSE", "NO"}
VIRTUAL_POSITIONS = os.environ.get("GM_INNER_SHADOW_VIRTUAL_POSITIONS", "0").upper() in {
    "1",
    "TRUE",
    "YES",
}
BACKTEST_START = os.environ.get("GM_INNER_SHADOW_BACKTEST_START", "2026-04-08 09:00:00")
BACKTEST_END = os.environ.get("GM_INNER_SHADOW_BACKTEST_END", "2026-06-05 15:30:00")
BACKTEST_INITIAL_CASH = float(os.environ.get("GM_INNER_SHADOW_BACKTEST_INITIAL_CASH", "1000000"))
AUDIT_DIR = Path(
    os.environ.get(
        "GM_INNER_SHADOW_AUDIT_DIR",
        str(ROOT.parent / "gm_inner_t0_0945_1000_shadow_audit"),
    )
).resolve()
AUDIT_RUN_ID = os.environ.get(
    "GM_INNER_SHADOW_AUDIT_RUN_ID", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
)
TARGET_CONTEXT_PATH = Path(
    os.environ.get("GM_INNER_SHADOW_TARGET_CONTEXT", str(ROOT / "gm_c_baseline_targets.csv"))
).resolve()
FORBIDDEN_PATH = Path(
    os.environ.get("GM_INNER_SHADOW_FORBIDDEN_SYMBOLS", str(ROOT / "gm_c_forbidden_symbols.csv"))
).resolve()
MODEL_MANIFEST_PATH = Path(
    os.environ.get(
        "GM_INNER_SHADOW_MODEL_MANIFEST",
        str(ROOT / "FROZEN_SHADOW_MODELS_MANIFEST.json"),
    )
).resolve()
MAX_TARGET_AGE_DAYS = int(os.environ.get("GM_INNER_SHADOW_MAX_TARGET_AGE_DAYS", "0"))
MAX_SIGNAL_AGE_DAYS = int(os.environ.get("GM_INNER_SHADOW_MAX_SIGNAL_AGE_DAYS", "4"))

PRIMARY_COMPONENT = "0945_high_confidence"
SECONDARY_COMPONENT = "1000_daily_ridge_gate"
PRIMARY_DECISION_TIME = "09:45:05"
SECONDARY_DECISION_TIME = "10:00:05"
PRIMARY_BAR_END = "09:45:00"
SECONDARY_BAR_END = "10:00:00"
TRIGGER_START = "09:45:00"
TRIGGER_END = "11:00:00"
EXIT_TIME = "14:45:00"
EXIT_END = "14:50:00"
MAX_SYMBOLS = 4
MAX_DAILY_TURNOVER = 0.03
MIN_FEATURE_UNIVERSE = 20
MIN_FEATURE_SUCCESS_RATIO = 0.70
MAX_TARGET_MISSING_RATIO = 0.20
BUY_COST = 0.001
SELL_COST = 0.0025
MIN_COST = 5.0
LIMIT_BUFFER = 0.002

_manifest: dict[str, Any] = {}
_models: dict[str, CatBoostClassifier] = {}
_primary_calibration: tuple[np.ndarray, np.ndarray] | None = None
_daily_gate: dict[str, Any] = {}
_target_context: dict[str, dict[str, float]] = {}
_target_source_date: str | None = None
_target_age_days: int | None = None
_target_signal_date: str | None = None
_signal_age_days: int | None = None
_forbidden_local: set[str] = set()
_forbidden_gm: set[str] = set()
_decision_scores: list[dict[str, Any]] = []
_candidate_intents: list[dict[str, Any]] = []
_entry_intents: list[dict[str, Any]] = []
_exit_intents: list[dict[str, Any]] = []
_runtime_events: list[dict[str, Any]] = []
_component_candidates: dict[str, pd.DataFrame] = {}
_daily_candidates: dict[str, dict[str, Any]] = {}
_daily_entries: dict[str, dict[str, Any]] = {}
_decision_keys: set[tuple[str, str]] = set()
_exit_dates: set[str] = set()
_finalized_dates: set[str] = set()
_daily_turnover_reserved = 0.0
_daily_buy_cash_reserved = 0.0
_daily_nav = 0.0
_daily_cash = 0.0
_subscribed_symbols: set[str] = set()
_successful_exit_keys: set[tuple[str, str]] = set()


def _audit_path(name: str) -> Path:
    return AUDIT_DIR / AUDIT_RUN_ID / name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_if_file(path: Path) -> str | None:
    return _sha256_file(path).upper() if path.is_file() else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    pending = path.with_name(f"{path.name}.pending")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(pending, path)


def _registry_path() -> Path:
    return AUDIT_DIR / REGISTRY_FILENAME


def _read_registry() -> list[dict[str, Any]]:
    path = _registry_path()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    previous_hash = ""
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"audit registry JSON error at line {line_number}: {exc}")
        if record.get("previous_hash", "") != previous_hash:
            raise RuntimeError(f"audit registry chain break at line {line_number}")
        expected = registry_record_hash(record)
        if str(record.get("record_hash", "")).upper() != expected:
            raise RuntimeError(f"audit registry hash mismatch at line {line_number}")
        previous_hash = expected
        records.append(record)
    return records


def _append_registry_event(event: str, **values: Any) -> dict[str, Any]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    records = _read_registry()
    if event == "RUN_STARTED" and any(
        row.get("event") == "RUN_STARTED" and row.get("run_id") == AUDIT_RUN_ID
        for row in records
    ):
        raise RuntimeError(f"audit run id already registered: {AUDIT_RUN_ID}")
    record = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "previous_hash": records[-1]["record_hash"] if records else "",
        "timestamp": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "run_id": AUDIT_RUN_ID,
        "event": event,
        "run_mode": RUN_MODE_NAME,
        "dry_run": DRY_RUN,
        "account_id": GM_ACCOUNT_ID,
        **values,
    }
    record["record_hash"] = registry_record_hash(record)
    path = _registry_path()
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def _register_run_start() -> None:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if not AUDIT_RUN_ID or any(char not in allowed for char in AUDIT_RUN_ID):
        raise RuntimeError(f"unsafe GM_INNER_SHADOW_AUDIT_RUN_ID: {AUDIT_RUN_ID!r}")
    _append_registry_event(
        "RUN_STARTED",
        model_manifest_sha256=_sha256_if_file(MODEL_MANIFEST_PATH),
        target_context_sha256=_sha256_if_file(TARGET_CONTEXT_PATH),
        forbidden_sha256=_sha256_if_file(FORBIDDEN_PATH),
    )


def _packaged_path(path_value: object) -> Path:
    return ROOT / Path(str(path_value)).name


def _verify_packaged_artifact(item: dict[str, Any], label: str) -> Path:
    path = _packaged_path(item.get("path"))
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    expected = str(item.get("sha256", "")).upper()
    actual = _sha256_file(path).upper()
    if not expected or actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch path={path} expected={expected} actual={actual}"
        )
    return path


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _today(context) -> str:
    try:
        return pd.Timestamp(context.now).strftime("%Y-%m-%d")
    except Exception:
        return pd.Timestamp.now().strftime("%Y-%m-%d")


def _now_time(context) -> str:
    try:
        return pd.Timestamp(context.now).strftime("%H:%M:%S")
    except Exception:
        return pd.Timestamp.now().strftime("%H:%M:%S")


def _gm_to_local_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if "." not in text:
        return text
    exchange, code = text.split(".", 1)
    return ("SH" if exchange in {"SHSE", "SH"} else "SZ") + code


def _local_to_gm_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    code = text[-6:]
    return f"SHSE.{code}" if text.startswith("SH") or code.startswith(("6", "9")) else f"SZSE.{code}"


def _risk_reason(sec_name: object, is_suspended: object, delisted_date: object, today: str) -> str:
    compact = "".join(str(sec_name or "").upper().replace("＊", "*").split())
    reasons: list[str] = []
    if compact.startswith(("*ST", "ST", "S*ST", "SST", "PT")):
        reasons.append(f"risk_name:{compact}")
    if "退" in compact:
        reasons.append(f"delisting_name:{compact}")
    try:
        suspended = int(float(is_suspended or 0)) != 0
    except (TypeError, ValueError):
        suspended = str(is_suspended).strip().upper() in {"TRUE", "YES"}
    if suspended:
        reasons.append("suspended")
    delisted = pd.to_datetime(delisted_date, errors="coerce")
    if pd.notna(delisted) and delisted.date() <= pd.Timestamp(today).date():
        reasons.append(f"delisted:{delisted.date()}")
    return "|".join(reasons)


def _event(context, event: str, **values: Any) -> None:
    row = {"date": _today(context), "time": _now_time(context), "event": event, **values}
    _runtime_events.append(row)
    print(f"[INNER-SHADOW] {event} {values}", flush=True)


def _load_forbidden() -> None:
    global _forbidden_local, _forbidden_gm
    if not FORBIDDEN_PATH.exists():
        raise FileNotFoundError(f"forbidden symbol file missing: {FORBIDDEN_PATH}")
    frame = pd.read_csv(FORBIDDEN_PATH, dtype=str).fillna("")
    local: set[str] = set()
    gm: set[str] = set()
    for col in frame.columns:
        lowered = col.lower()
        if "instrument" in lowered or "symbol" in lowered or "code" in lowered:
            values = frame[col].astype(str).str.strip().str.upper()
            for value in values:
                if not value:
                    continue
                if "." in value:
                    gm.add(value)
                    local.add(_gm_to_local_symbol(value))
                else:
                    local.add(value)
                    gm.add(_local_to_gm_symbol(value))
    _forbidden_local = local
    _forbidden_gm = gm
    print(f"[INNER-SHADOW] forbidden loaded local={len(local)} gm={len(gm)}", flush=True)


def _load_target_context(today: str) -> None:
    global _target_context, _target_source_date, _target_age_days
    global _target_signal_date, _signal_age_days
    if not TARGET_CONTEXT_PATH.exists():
        raise FileNotFoundError(f"target context missing: {TARGET_CONTEXT_PATH}")
    frame = pd.read_csv(TARGET_CONTEXT_PATH)
    required = {"instrument", "trade_date", "signal_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"target context columns missing: {missing}")
    frame["instrument"] = frame["instrument"].map(_gm_to_local_symbol)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    eligible = frame[frame["trade_date"] <= today].copy()
    if eligible.empty:
        raise RuntimeError(f"no target snapshot on or before {today}")
    _target_source_date = str(eligible["trade_date"].max())
    _target_age_days = require_fresh_target(
        today,
        _target_source_date,
        max_calendar_age_days=MAX_TARGET_AGE_DAYS,
    )
    source = eligible[eligible["trade_date"] == _target_source_date].copy()
    signal_dates = pd.to_datetime(source["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    unique_signal_dates = sorted(set(signal_dates.dropna()))
    if len(unique_signal_dates) != 1:
        raise RuntimeError(
            f"target snapshot must have exactly one signal_date: {unique_signal_dates}"
        )
    _target_signal_date = unique_signal_dates[0]
    _signal_age_days = require_fresh_signal(
        today,
        _target_signal_date,
        max_calendar_age_days=MAX_SIGNAL_AGE_DAYS,
    )
    _target_context = {}
    for row in source.itertuples(index=False):
        local = str(row.instrument).upper()
        _target_context[local] = {
            "target_weight": finite_float(getattr(row, "target_weight", 0.0), 0.0),
            "target_shares": finite_float(getattr(row, "target_shares", 0.0), 0.0),
            "rank": finite_float(getattr(row, "rank", np.nan)),
            "middle": finite_float(getattr(row, "middle", np.nan)),
            "group": str(getattr(row, "group", "") or ""),
        }
    print(
        f"[INNER-SHADOW] target context rows={len(_target_context)} "
        f"source_date={_target_source_date} target_age_days={_target_age_days} "
        f"signal_date={_target_signal_date} signal_age_days={_signal_age_days}",
        flush=True,
    )


def _load_models() -> None:
    global _manifest, _primary_calibration, _daily_gate
    if not MODEL_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"model manifest missing: {MODEL_MANIFEST_PATH}")
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    permissions = manifest.get("permissions", {})
    if permissions.get("paper_orders_allowed") is not False:
        raise RuntimeError("shadow manifest must keep paper_orders_allowed=false")
    if permissions.get("main_py_integration_allowed") is not False:
        raise RuntimeError("shadow manifest must keep main_py_integration_allowed=false")
    if permissions.get("deployment_allowed") is not False:
        raise RuntimeError("shadow manifest must keep deployment_allowed=false")
    _manifest = manifest
    for component, key in (
        (PRIMARY_COMPONENT, "primary_0945"),
        (SECONDARY_COMPONENT, "secondary_1000"),
    ):
        item = manifest[key]
        model_path = _verify_packaged_artifact(item["model"], f"{component} model")
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        feature_cols = list(item.get("feature_cols", []))
        if not feature_cols:
            raise RuntimeError(f"{component} feature contract is empty")
        _models[component] = model
        print(
            f"[INNER-SHADOW] model loaded component={component} "
            f"features={len(feature_cols)} sha256={item['model']['sha256']}",
            flush=True,
        )
    primary_item = manifest["primary_0945"]
    calibration_path = _verify_packaged_artifact(
        primary_item["calibration"],
        "0945 isotonic calibration",
    )
    with np.load(calibration_path) as calibration:
        x = np.asarray(calibration["x_thresholds"], dtype=float)
        y = np.asarray(calibration["y_thresholds"], dtype=float)
    if len(x) < 2 or len(x) != len(y):
        raise RuntimeError("invalid 09:45 isotonic calibration")
    _primary_calibration = (x, y)
    secondary_item = manifest["secondary_1000"]
    daily_gate_path = _verify_packaged_artifact(
        secondary_item["daily_gate"],
        "1000 daily Ridge gate",
    )
    _daily_gate = json.loads(daily_gate_path.read_text(encoding="utf-8-sig"))
    if _daily_gate.get("deployment_allowed") is not False:
        raise RuntimeError("daily Ridge gate must keep deployment_allowed=false")
    if _daily_gate.get("paper_orders_allowed") is not False:
        raise RuntimeError("daily Ridge gate must keep paper_orders_allowed=false")
    if _daily_gate.get("expected_forward_parity", {}).get("passed") is not True:
        raise RuntimeError("daily Ridge gate forward parity is not proven")


def _fetch_positions(context) -> list[dict[str, Any]]:
    if VIRTUAL_POSITIONS:
        rows = []
        for local, target in sorted(_target_context.items(), key=lambda item: (item[1].get("rank", 1e12), item[0])):
            shares = int(finite_float(target.get("target_shares"), 0.0))
            if shares <= 0:
                continue
            rows.append(
                {
                    "symbol": _local_to_gm_symbol(local),
                    "local": local,
                    "volume": shares,
                    "available": shares,
                    "cost_price": np.nan,
                }
            )
        return rows
    raw = get_position(account_id=GM_ACCOUNT_ID) if GM_ACCOUNT_ID else get_position()  # noqa: F405
    rows = []
    for position in raw or []:
        symbol = str(_field(position, "symbol", "")).upper()
        local = _gm_to_local_symbol(symbol)
        volume = int(finite_float(_field(position, "volume", 0), 0.0))
        available = int(
            finite_float(
                _field(
                    position,
                    "available_now",
                    _field(position, "available", volume),
                ),
                0.0,
            )
        )
        cost_price = finite_float(
            _field(position, "vwap", _field(position, "cost_price", np.nan)),
            np.nan,
        )
        if symbol and volume > 0:
            rows.append(
                {
                    "symbol": symbol,
                    "local": local,
                    "volume": volume,
                    "available": available,
                    "cost_price": cost_price,
                }
            )
    return rows


def _fetch_cash(positions: list[dict[str, Any]]) -> tuple[float, float]:
    if VIRTUAL_POSITIONS:
        return BACKTEST_INITIAL_CASH, BACKTEST_INITIAL_CASH
    cash = get_cash(account_id=GM_ACCOUNT_ID) if GM_ACCOUNT_ID else get_cash()  # noqa: F405
    nav = finite_float(_field(cash, "nav", np.nan), np.nan)
    available = finite_float(_field(cash, "available", np.nan), np.nan)
    if not np.isfinite(nav) or nav <= 0:
        raise RuntimeError(f"invalid account nav: {cash}")
    if not np.isfinite(available) or available < 0:
        raise RuntimeError(f"invalid available cash: {cash}")
    return nav, available


def _dynamic_allowed(context, positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    static_allowed = [
        row
        for row in positions
        if row["symbol"] not in _forbidden_gm and row["local"] not in _forbidden_local
    ]
    symbols = sorted({str(row["symbol"]) for row in static_allowed})
    if not symbols:
        return []
    rows = get_instruments(  # noqa: F405
        symbols=symbols,
        skip_suspended=False,
        skip_st=False,
        fields=["symbol", "sec_name", "sec_abbr", "is_suspended", "delisted_date", "trade_date"],
        df=False,
    )
    by_symbol = {str(_field(row, "symbol", "")).upper(): row for row in (rows or [])}
    allowed = []
    for position in static_allowed:
        symbol = str(position["symbol"])
        instrument = by_symbol.get(symbol)
        if instrument is None:
            _event(context, "RISK_BLOCK", symbol=symbol, reason="instrument_lookup_missing")
            continue
        name = _field(instrument, "sec_name", "") or _field(instrument, "sec_abbr", "")
        reason = _risk_reason(
            name,
            _field(instrument, "is_suspended", 0),
            _field(instrument, "delisted_date", None),
            _today(context),
        )
        if reason:
            _event(context, "RISK_BLOCK", symbol=symbol, sec_name=str(name), reason=reason)
            continue
        allowed.append(position)
    return allowed


def _history_at_decision(
    symbol: str,
    today: str,
    *,
    decision_minute: int,
    bar_end: str,
) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    end_time = f"{today} {bar_end}"
    raw = history_n(  # noqa: F405
        symbol=symbol,
        frequency="60s",
        count=55,
        end_time=end_time,
        fields="eob,open,high,low,close,volume,amount",
        skip_suspended=False,
        adjust=ADJUST_PREV,  # noqa: F405
        adjust_end_time=end_time,
        df=True,
    )
    normalized, unit_audit = normalize_gm_minute_bars(pd.DataFrame(raw))
    if normalized.empty:
        return pd.DataFrame(), np.nan, unit_audit
    session, previous_close = current_session_bars(normalized, today, decision_minute)
    if session.empty or not np.isfinite(previous_close) or previous_close <= 0:
        unit_audit = {**unit_audit, "valid": False, "reason": "missing_session_or_previous_close"}
        return pd.DataFrame(), np.nan, unit_audit
    return session, previous_close, unit_audit


def _score_frame(features: pd.DataFrame, component: str) -> pd.DataFrame:
    out = features.copy()
    key = "primary_0945" if component == PRIMARY_COMPONENT else "secondary_1000"
    columns = list(_manifest[key]["feature_cols"])
    matrix = out.reindex(columns=columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw = _models[component].predict_proba(matrix)[:, 1]
    out["raw_score"] = raw
    if component == PRIMARY_COMPONENT:
        if _primary_calibration is None:
            raise RuntimeError("09:45 isotonic calibration is not loaded")
        x, y = _primary_calibration
        out["model_score"] = [isotonic_score(value, x, y) for value in raw]
    else:
        out["model_score"] = raw
    return out


def _session_details() -> list[dict[str, Any]]:
    required = set(EXPECTED_DECISION_TIMES)
    event_dates = {
        str(row.get("date"))
        for row in _runtime_events
        if row.get("date")
    }
    observed_dates = sorted(
        event_dates
        | {date for date, _ in _decision_keys}
        | set(_exit_dates)
        | set(_finalized_dates)
    )
    details = []
    for date in observed_dates:
        completed = {
            str(row.get("component"))
            for row in _runtime_events
            if row.get("date") == date and row.get("event") == "DECISION_COMPLETE"
        }
        blocked = sorted(
            {
                str(row.get("component"))
                for row in _runtime_events
                if row.get("date") == date and row.get("event") == "DECISION_BLOCKED"
            }
        )
        missing = sorted(required.difference(completed))
        exit_complete = date in _exit_dates
        finalized = date in _finalized_dates
        details.append(
            {
                "date": date,
                "completed_decisions": sorted(completed),
                "missing_decisions": missing,
                "blocked_decisions": blocked,
                "exit_scan_complete": exit_complete,
                "finalized": finalized,
                "complete": bool(
                    not missing and not blocked and exit_complete and finalized
                ),
            }
        )
    return details


def _write_audit() -> None:
    out = AUDIT_DIR / AUDIT_RUN_ID
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "decision_scores.csv": _decision_scores,
        "candidate_intents.csv": _candidate_intents,
        "entry_intents.csv": _entry_intents,
        "exit_intents.csv": _exit_intents,
        "runtime_events.csv": _runtime_events,
    }
    audit_files: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_AUDIT_TABLES:
        rows = tables[name]
        frame = pd.DataFrame(rows)
        required_columns = list(AUDIT_TABLE_COLUMNS[name])
        for column in required_columns:
            if column not in frame:
                frame[column] = pd.Series(dtype="object")
        extra_columns = sorted(set(frame.columns).difference(required_columns))
        frame = frame[required_columns + extra_columns]
        path = out / name
        pending = path.with_name(f"{path.name}.pending")
        frame.to_csv(pending, index=False, encoding="utf-8-sig")
        os.replace(pending, path)
        audit_files[name] = {
            "rows": int(len(frame)),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path).upper(),
        }
    triggered_keys = {
        (str(row.get("date")), str(row.get("symbol")))
        for row in _entry_intents
        if row.get("action") == "SELL_FIRST_TRIGGERED"
    }
    exit_keys = {
        (str(row.get("date")), str(row.get("symbol")))
        for row in _exit_intents
        if row.get("action") == "BUYBACK_INTENT"
    }
    unmatched = [
        {"date": date, "symbol": symbol}
        for date, symbol in sorted(triggered_keys - exit_keys)
    ]
    session_details = _session_details()
    complete_dates = [row["date"] for row in session_details if row["complete"]]
    incomplete_dates = [row["date"] for row in session_details if not row["complete"]]
    summary = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "status": "inner_t0_0945_1000_no_order_shadow_audit",
        "audit_run_id": AUDIT_RUN_ID,
        "audit_written_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "run_mode": RUN_MODE_NAME,
        "dry_run": DRY_RUN,
        "account_id": GM_ACCOUNT_ID,
        "model_manifest": str(MODEL_MANIFEST_PATH),
        "model_manifest_sha256": _sha256_if_file(MODEL_MANIFEST_PATH),
        "target_context": str(TARGET_CONTEXT_PATH),
        "target_context_sha256": _sha256_if_file(TARGET_CONTEXT_PATH),
        "forbidden_sha256": _sha256_if_file(FORBIDDEN_PATH),
        "target_source_date": _target_source_date,
        "target_age_days": _target_age_days,
        "max_target_age_days": MAX_TARGET_AGE_DAYS,
        "target_signal_date": _target_signal_date,
        "signal_age_days": _signal_age_days,
        "max_signal_age_days": MAX_SIGNAL_AGE_DAYS,
        "decision_keys": [f"{date}:{component}" for date, component in sorted(_decision_keys)],
        "exit_dates": sorted(_exit_dates),
        "finalized_dates": sorted(_finalized_dates),
        "session_details": session_details,
        "complete_session_dates": complete_dates,
        "incomplete_session_dates": incomplete_dates,
        "observation_session_count": len(complete_dates),
        "session_complete": bool(session_details and not incomplete_dates),
        "expected_decision_times": EXPECTED_DECISION_TIMES,
        "session_finalize_time": SESSION_FINALIZE_TIME,
        "decision_score_rows": len(_decision_scores),
        "candidate_rows": len(_candidate_intents),
        "entry_rows": len(_entry_intents),
        "entry_triggered_rows": sum(
            row.get("action") == "SELL_FIRST_TRIGGERED" for row in _entry_intents
        ),
        "exit_rows": len(_exit_intents),
        "successful_exit_intents": sum(
            row.get("action") == "BUYBACK_INTENT" for row in _exit_intents
        ),
        "runtime_event_rows": len(_runtime_events),
        "unmatched_entry_symbols": unmatched,
        "audit_files": audit_files,
        "max_symbols_per_day": MAX_SYMBOLS,
        "max_daily_turnover": MAX_DAILY_TURNOVER,
        "selected_tick_subscriptions_only": True,
        "actual_submission_api_present": False,
        "paper_orders_allowed": False,
        "main_py_integration_allowed": False,
        "deployment_allowed": False,
    }
    _atomic_write_json(out / "summary.json", summary)


def _reset_daily_state(context, today: str) -> None:
    global _daily_candidates, _daily_entries, _daily_turnover_reserved
    global _daily_buy_cash_reserved, _component_candidates, _subscribed_symbols
    if any(date == today for date, _ in _decision_keys):
        return
    if _subscribed_symbols:
        try:
            unsubscribe(symbols=sorted(_subscribed_symbols), frequency="tick")  # noqa: F405
        except Exception as exc:
            _event(context, "STALE_TICK_UNSUBSCRIBE_FAILED", reason=str(exc))
        _subscribed_symbols.clear()
    _daily_candidates = {}
    _daily_entries = {}
    _component_candidates = {}
    _daily_turnover_reserved = 0.0
    _daily_buy_cash_reserved = 0.0
    _load_target_context(today)


def _build_decision_features(
    context,
    *,
    component: str,
    decision_minute: int,
    bar_end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    global _daily_nav, _daily_cash
    today = _today(context)
    positions = _fetch_positions(context)
    _daily_nav, _daily_cash = _fetch_cash(positions)
    allowed = _dynamic_allowed(context, positions)
    rows: list[dict[str, Any]] = []
    failures = 0
    for position in allowed:
        symbol = str(position["symbol"])
        try:
            day, previous_close, unit_audit = _history_at_decision(
                symbol,
                today,
                decision_minute=decision_minute,
                bar_end=bar_end,
            )
            visible = visible_features(day, decision_minute, previous_close)
            if not visible:
                raise ValueError("empty visible features")
            target = _target_context.get(str(position["local"]), {})
            features = build_position_features(
                position,
                target,
                visible,
                previous_close=previous_close,
                nav=_daily_nav,
                buy_cost=BUY_COST,
                sell_cost=SELL_COST,
                min_cost=MIN_COST,
            )
            rows.append(
                {
                    **features,
                    "date": today,
                    "component": component,
                    "symbol": symbol,
                    "local_symbol": position["local"],
                    "held_volume": int(position["volume"]),
                    "available_volume": int(position["available"]),
                    "decision_price": float(visible["sell_price_decision"]),
                    "volume_scale": unit_audit.get("volume_scale"),
                    "volume_unit_source": unit_audit.get("unit_source"),
                }
            )
        except Exception as exc:
            failures += 1
            _event(
                context,
                "FEATURE_SKIP",
                component=component,
                symbol=symbol,
                reason=str(exc),
            )
    success_ratio = len(rows) / max(len(allowed), 1)
    if len(rows) < MIN_FEATURE_UNIVERSE or success_ratio < MIN_FEATURE_SUCCESS_RATIO:
        raise RuntimeError(
            f"feature universe gate failed component={component} rows={len(rows)} "
            f"allowed={len(allowed)} ratio={success_ratio:.3f}"
        )
    feature_frame = add_held_cross_sectional_features(pd.DataFrame(rows))
    missing_ratio = float(
        pd.to_numeric(feature_frame["target_missing"], errors="coerce").mean()
    )
    if missing_ratio > MAX_TARGET_MISSING_RATIO:
        raise RuntimeError(
            f"target context mismatch component={component} ratio={missing_ratio:.3f}"
        )
    audit = {
        "positions": len(positions),
        "allowed": len(allowed),
        "feature_rows": len(rows),
        "failures": failures,
        "success_ratio": success_ratio,
        "target_missing_ratio": missing_ratio,
    }
    return feature_frame, audit


def _refresh_combined_candidates(context, today: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    global _daily_candidates, _subscribed_symbols
    empty = pd.DataFrame(columns=["symbol", "local_symbol", "score"])
    primary = _component_candidates.get(PRIMARY_COMPONENT, empty)
    secondary = _component_candidates.get(SECONDARY_COMPONENT, empty)
    combined, conflicts = combine_primary_secondary(
        primary,
        secondary,
        max_symbols=MAX_SYMBOLS,
    )
    _daily_candidates = {
        str(row["symbol"]): {
            "date": today,
            "time": _now_time(context),
            **row,
            "dry_run": True,
            "action": "ACTIVE_COMBINED_CANDIDATE",
        }
        for row in combined.to_dict("records")
    }
    for row in conflicts.to_dict("records"):
        _candidate_intents.append(
            {
                "date": today,
                "time": _now_time(context),
                **row,
                "dry_run": True,
                "action": "DROP_SECONDARY_PRIMARY_CONFLICT",
            }
        )
    new_symbols = sorted(set(_daily_candidates).difference(_subscribed_symbols))
    if new_symbols:
        try:
            subscribe(  # noqa: F405
                symbols=new_symbols,
                frequency="tick",
                count=1,
                unsubscribe_previous=False,
            )
            _subscribed_symbols.update(new_symbols)
            _event(
                context,
                "SELECTED_TICK_SUBSCRIBE",
                symbols=",".join(new_symbols),
                count=len(new_symbols),
            )
        except Exception as exc:
            _event(
                context,
                "TICK_SUBSCRIBE_FAILED_POLL_FALLBACK",
                symbols=",".join(new_symbols),
                reason=str(exc),
            )
    return combined, conflicts


def _run_component_decision(
    context,
    *,
    component: str,
    decision_minute: int,
    bar_end: str,
) -> None:
    today = _today(context)
    key = (today, component)
    if key in _decision_keys:
        return
    _reset_daily_state(context, today)
    _decision_keys.add(key)
    try:
        feature_frame, feature_audit = _build_decision_features(
            context,
            component=component,
            decision_minute=decision_minute,
            bar_end=bar_end,
        )
        scored = _score_frame(feature_frame, component)
        manifest_key = "primary_0945" if component == PRIMARY_COMPONENT else "secondary_1000"
        feature_cols = list(_manifest[manifest_key]["feature_cols"])
        score_rows = []
        for row in scored.itertuples(index=False):
            payload = {
                "date": today,
                "time": _now_time(context),
                "component": component,
                "symbol": row.symbol,
                "local_symbol": row.local_symbol,
                "held_volume": int(row.held_volume),
                "available_volume": int(row.available_volume),
                "decision_price": float(row.decision_price),
                "previous_close": float(row.mark_price),
                "target_missing": float(row.target_missing),
                "raw_score": float(row.raw_score),
                "model_score": float(row.model_score),
                "volume_scale": row.volume_scale,
                "features": json.dumps(
                    {
                        col: finite_float(getattr(row, col, 0.0), 0.0)
                        for col in feature_cols
                    },
                    ensure_ascii=False,
                ),
            }
            score_rows.append(payload)
        score_frame = pd.DataFrame(score_rows)
        if component == PRIMARY_COMPONENT:
            gate_score = None
            gate_enabled = True
            selected = select_sell_first_candidates(
                score_frame,
                component=component,
                daily_top_n=int(_manifest[manifest_key]["daily_top_n"]),
                score_threshold=float(_manifest[manifest_key]["score_threshold"]),
                trigger_distance=float(_manifest[manifest_key]["trigger_distance"]),
            )
        else:
            meta_values = build_daily_meta_values(score_frame, score_col="model_score")
            gate_score = ridge_gate_score(meta_values, _daily_gate)
            gate_threshold = float(_manifest[manifest_key]["daily_gate"]["threshold"])
            gate_enabled = bool(gate_score > gate_threshold)
            for payload in score_rows:
                payload["meta_gate_score"] = gate_score
                payload["meta_gate_enabled"] = gate_enabled
            if gate_enabled:
                selected = select_sell_first_candidates(
                    score_frame,
                    component=component,
                    daily_top_n=int(_manifest[manifest_key]["daily_top_n"]),
                    score_threshold=None,
                    trigger_distance=float(_manifest[manifest_key]["trigger_distance"]),
                )
            else:
                selected = score_frame.iloc[0:0].copy()
            _event(
                context,
                "DAILY_RIDGE_GATE",
                component=component,
                score=gate_score,
                threshold=gate_threshold,
                enabled=gate_enabled,
                held_count=meta_values["held_count"],
            )
        _decision_scores.extend(score_rows)
        _component_candidates[component] = selected
        for row in selected.to_dict("records"):
            _candidate_intents.append(
                {
                    "date": today,
                    "time": _now_time(context),
                    **row,
                    "dry_run": True,
                    "action": "COMPONENT_CANDIDATE",
                    "meta_gate_score": gate_score,
                    "meta_gate_enabled": gate_enabled,
                }
            )
        combined, conflicts = _refresh_combined_candidates(context, today)
        _event(
            context,
            "DECISION_COMPLETE",
            component=component,
            component_candidates=len(selected),
            combined_candidates=len(combined),
            conflicts=len(conflicts),
            **feature_audit,
        )
        _write_audit()
    except Exception as exc:
        _component_candidates[component] = pd.DataFrame()
        _event(context, "DECISION_BLOCKED", component=component, reason=str(exc))
        traceback.print_exc()
        _write_audit()


def on_primary_decision_scan(context):
    _run_component_decision(
        context,
        component=PRIMARY_COMPONENT,
        decision_minute=9 * 60 + 45,
        bar_end=PRIMARY_BAR_END,
    )


def on_secondary_decision_scan(context):
    today = _today(context)
    if (today, PRIMARY_COMPONENT) not in _decision_keys:
        on_primary_decision_scan(context)
    _run_component_decision(
        context,
        component=SECONDARY_COMPONENT,
        decision_minute=10 * 60,
        bar_end=SECONDARY_BAR_END,
    )


def _tick_price(tick: Any) -> tuple[str, float]:
    symbol = str(_field(tick, "symbol", "")).upper()
    price = finite_float(_field(tick, "price", _field(tick, "last_price", np.nan)), np.nan)
    return symbol, price


def _process_trigger(context, symbol: str, price: float, source: str) -> None:
    global _daily_turnover_reserved
    if symbol not in _daily_candidates or symbol in _daily_entries:
        return
    candidate = _daily_candidates[symbol]
    if not trigger_reached(price, float(candidate["entry_limit"])):
        return
    intent, accepted = build_sell_entry_intent(
        candidate,
        trigger_price=price,
        nav=_daily_nav,
        turnover_used=_daily_turnover_reserved,
        max_daily_turnover=MAX_DAILY_TURNOVER,
        sell_cost=SELL_COST,
        min_cost=MIN_COST,
    )
    event = {
        "date": _today(context),
        "time": _now_time(context),
        **intent,
        "nav": _daily_nav,
        "turnover_reserved_before": _daily_turnover_reserved,
        "projected_daily_turnover": (
            _daily_turnover_reserved + float(intent["reserved_roundtrip_turnover"])
        )
        / max(_daily_nav, 1e-12),
        "trigger_source": source,
        "dry_run": True,
    }
    _entry_intents.append(event)
    if accepted:
        _daily_entries[symbol] = event
        _daily_turnover_reserved += float(event["reserved_roundtrip_turnover"])
    _event(context, "ENTRY_EVALUATED", symbol=symbol, action=event["action"], price=price, source=source)
    _write_audit()


def on_tick(context, tick):
    now = _now_time(context)
    if not (TRIGGER_START <= now <= TRIGGER_END):
        return
    symbol, price = _tick_price(tick)
    if symbol and np.isfinite(price):
        _process_trigger(context, symbol, price, "tick")


def on_trigger_poll(context):
    now = _now_time(context)
    if EXIT_TIME <= now <= EXIT_END:
        on_exit_scan(context)
        return
    if not (TRIGGER_START <= now <= TRIGGER_END) or not _daily_candidates:
        return
    pending = sorted(set(_daily_candidates) - set(_daily_entries))
    if not pending:
        return
    try:
        snapshots = current(symbols=pending, fields="symbol,price")  # noqa: F405
        for tick in snapshots or []:
            symbol, price = _tick_price(tick)
            if symbol and np.isfinite(price):
                _process_trigger(context, symbol, price, "snapshot_poll")
    except Exception as exc:
        _event(context, "TRIGGER_POLL_FAILED", reason=str(exc))


def _exit_market_allowed(
    symbol: str,
    direction: str,
    price: float,
    previous_close: float,
) -> tuple[bool, str]:
    if not np.isfinite(price) or price <= 0 or not np.isfinite(previous_close) or previous_close <= 0:
        return False, "SKIP_EXIT_NO_PRICE"
    code = str(symbol).upper()[-6:]
    pct = 0.20 if code.startswith(("30", "68")) else 0.10
    if direction == "buy_first" and price <= previous_close * (1.0 - pct) * (1.0 + LIMIT_BUFFER):
        return False, "SKIP_EXIT_SELL_LIMIT_DOWN"
    if direction == "sell_first" and price >= previous_close * (1.0 + pct) * (1.0 - LIMIT_BUFFER):
        return False, "SKIP_EXIT_BUY_LIMIT_UP"
    return True, "OK"


def on_exit_scan(context):
    global _subscribed_symbols
    today = _today(context)
    if today in _exit_dates:
        return
    try:
        symbols = sorted(
            symbol
            for symbol in _daily_entries
            if (today, symbol) not in _successful_exit_keys
        )
        if not symbols:
            _exit_dates.add(today)
            return
        snapshots = current(symbols=symbols, fields="symbol,price") if symbols else []  # noqa: F405
        by_symbol = {str(_field(row, "symbol", "")).upper(): finite_float(_field(row, "price", np.nan)) for row in (snapshots or [])}
        for symbol, entry in sorted(_daily_entries.items()):
            if (today, symbol) in _successful_exit_keys:
                continue
            price = by_symbol.get(symbol, np.nan)
            previous_close = finite_float(entry.get("previous_close"), np.nan)
            allowed, reason = _exit_market_allowed(
                symbol,
                str(entry["direction"]),
                price,
                previous_close,
            )
            if allowed:
                exit_intent = build_buyback_intent(
                    entry,
                    exit_price=price,
                    buy_cost=BUY_COST,
                    min_cost=MIN_COST,
                )
            else:
                exit_intent = {
                    "symbol": symbol,
                    "local_symbol": entry["local_symbol"],
                    "direction": entry["direction"],
                    "volume": entry["volume"],
                    "exit_price_ref": price,
                    "action": reason,
                    "restores_original_inventory": False,
                }
            _exit_intents.append(
                {"date": today, "time": _now_time(context), **exit_intent, "dry_run": True}
            )
            if allowed:
                _successful_exit_keys.add((today, symbol))
            _event(context, "EXIT_EVALUATED", symbol=symbol, action=exit_intent["action"], price=price)
        if all((today, symbol) in _successful_exit_keys for symbol in _daily_entries):
            _exit_dates.add(today)
            if _subscribed_symbols:
                try:
                    unsubscribe(symbols=sorted(_subscribed_symbols), frequency="tick")  # noqa: F405
                    _event(
                        context,
                        "SELECTED_TICK_UNSUBSCRIBE",
                        symbols=",".join(sorted(_subscribed_symbols)),
                    )
                    _subscribed_symbols.clear()
                except Exception as exc:
                    _event(context, "TICK_UNSUBSCRIBE_FAILED", reason=str(exc))
        _write_audit()
    except Exception as exc:
        _event(context, "EXIT_SCAN_ERROR", reason=str(exc))
        traceback.print_exc()
        _write_audit()


def on_session_finalize(context):
    today = _today(context)
    if today in _finalized_dates:
        return
    _finalized_dates.add(today)
    completed = {
        str(row.get("component"))
        for row in _runtime_events
        if row.get("date") == today and row.get("event") == "DECISION_COMPLETE"
    }
    missing = sorted(set(EXPECTED_DECISION_TIMES).difference(completed))
    blocked = sorted(
        {
            str(row.get("component"))
            for row in _runtime_events
            if row.get("date") == today and row.get("event") == "DECISION_BLOCKED"
        }
    )
    complete = bool(not missing and not blocked and today in _exit_dates)
    _event(
        context,
        "SESSION_FINALIZED",
        complete=complete,
        missing_decisions="|".join(missing),
        blocked_decisions="|".join(blocked),
        exit_scan_complete=today in _exit_dates,
    )
    _write_audit()
    _append_registry_event(
        "SESSION_FINALIZED",
        date=today,
        complete=complete,
        missing_decisions=missing,
        blocked_decisions=blocked,
        exit_scan_complete=today in _exit_dates,
    )


def init(context):
    print("[INNER-SHADOW] 09:45 + 10:00 no-order observer init", flush=True)
    if not DRY_RUN:
        raise RuntimeError(
            "This package is permanently no-order; GM_INNER_SHADOW_DRY_RUN must remain 1"
        )
    if GM_TOKEN:
        set_token(GM_TOKEN)  # noqa: F405
    if GM_ACCOUNT_ID:
        set_account_id(GM_ACCOUNT_ID)  # noqa: F405
    registered = False
    try:
        _register_run_start()
        registered = True
        _load_forbidden()
        _load_models()
        _load_target_context(_today(context))
        schedule(  # noqa: F405
            schedule_func=on_primary_decision_scan,
            date_rule="1d",
            time_rule=PRIMARY_DECISION_TIME,
        )
        schedule(  # noqa: F405
            schedule_func=on_secondary_decision_scan,
            date_rule="1d",
            time_rule=SECONDARY_DECISION_TIME,
        )
        schedule(schedule_func=on_trigger_poll, date_rule="1d", time_rule="60s")  # noqa: F405
        schedule(schedule_func=on_exit_scan, date_rule="1d", time_rule=EXIT_TIME)  # noqa: F405
        schedule(  # noqa: F405
            schedule_func=on_session_finalize,
            date_rule="1d",
            time_rule=SESSION_FINALIZE_TIME,
        )
        _event(context, "RUN_INITIALIZED")
        now = _now_time(context)
        if PRIMARY_BAR_END <= now < SECONDARY_BAR_END:
            print(f"[INNER-SHADOW] catch-up 09:45 decision at {now}", flush=True)
            on_primary_decision_scan(context)
            on_trigger_poll(context)
        elif SECONDARY_BAR_END <= now <= TRIGGER_END:
            print(f"[INNER-SHADOW] catch-up 09:45 + 10:00 decisions at {now}", flush=True)
            on_secondary_decision_scan(context)
            on_trigger_poll(context)
        elif TRIGGER_END < now < EXIT_TIME:
            _event(context, "STARTED_AFTER_DECISION_WINDOW", observed_time=now)
        elif EXIT_TIME <= now <= EXIT_END:
            _event(context, "STARTED_DURING_EXIT_WINDOW", observed_time=now)
            on_exit_scan(context)
        elif now > EXIT_END:
            _event(context, "STARTED_AFTER_OBSERVATION_WINDOW", observed_time=now)
            on_session_finalize(context)
        _write_audit()
        print(
            f"[INNER-SHADOW] DRY_RUN={DRY_RUN} account={GM_ACCOUNT_ID} "
            f"decisions={PRIMARY_DECISION_TIME},{SECONDARY_DECISION_TIME} "
            f"trigger={TRIGGER_START}-{TRIGGER_END} exit={EXIT_TIME} "
            f"finalize={SESSION_FINALIZE_TIME}",
            flush=True,
        )
    except Exception as exc:
        if registered:
            _event(context, "INIT_FAILED", reason=str(exc))
            _write_audit()
        raise


def on_backtest_finished(context, indicator):
    _write_audit()


def on_error(context, code, info):
    _event(context, "GM_ERROR", code=code, info=str(info))
    _write_audit()
    print(f"[INNER-SHADOW-ERROR] code={code} info={info}", flush=True)


if __name__ == "__main__":
    print("[INNER-SHADOW] __main__ entered", flush=True)
    mode = MODE_BACKTEST if RUN_MODE_NAME == "BACKTEST" else MODE_LIVE  # noqa: F405
    kwargs = {
        "strategy_id": GM_STRATEGY_ID,
        "filename": "main.py",
        "token": GM_TOKEN,
        "mode": mode,
    }
    if mode == MODE_BACKTEST:  # noqa: F405
        kwargs.update(
            {
                "backtest_start_time": BACKTEST_START,
                "backtest_end_time": BACKTEST_END,
                "backtest_adjust": 1,
                "backtest_initial_cash": BACKTEST_INITIAL_CASH,
                "backtest_commission_ratio": 0.0,
                "backtest_slippage_ratio": 0.0,
                "backtest_match_mode": 0,
            }
        )
    run(**kwargs)  # noqa: F405
