# coding=utf-8
"""GmQuant held-only bidirectional T+0 intent audit.

This candidate never submits orders. It scores actual non-ST holdings at 10:00,
subscribes ticks only for at most three selected symbols, records virtual entry
intents during 10:00-11:00, and records the inventory-restoring exit intents at
14:45. The active outer+middle PAPER strategy remains separate.
"""
from __future__ import annotations

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
from inner_t0_bidirectional_engine import (
    build_entry_intent,
    build_exit_intent,
    percentile_score,
    select_bidirectional_candidates,
    trigger_reached,
)


ROOT = Path(__file__).resolve().parent
GM_TOKEN = os.environ.get("GM_TOKEN", "").strip()
GM_STRATEGY_ID = os.environ.get(
    "GM_STRATEGY_ID", "inner-t0-bidirectional-1000-1445-dryrun"
).strip()
GM_ACCOUNT_ID = os.environ.get("GM_ACCOUNT_ID", "").strip()
RUN_MODE_NAME = os.environ.get("GM_INNER_T0_MODE", "LIVE").strip().upper()
DRY_RUN = os.environ.get("GM_INNER_T0_DRY_RUN", "1").upper() not in {"0", "FALSE", "NO"}
VIRTUAL_POSITIONS = os.environ.get("GM_INNER_T0_VIRTUAL_POSITIONS", "0").upper() in {
    "1",
    "TRUE",
    "YES",
}
BACKTEST_START = os.environ.get("GM_INNER_T0_BACKTEST_START", "2026-02-05 09:00:00")
BACKTEST_END = os.environ.get("GM_INNER_T0_BACKTEST_END", "2026-04-07 15:30:00")
BACKTEST_INITIAL_CASH = float(os.environ.get("GM_INNER_T0_BACKTEST_INITIAL_CASH", "1000000"))
AUDIT_DIR = Path(
    os.environ.get("GM_INNER_T0_AUDIT_DIR", str(ROOT.parent / "gm_inner_t0_bidirectional_dryrun_audit"))
).resolve()
AUDIT_RUN_ID = os.environ.get(
    "GM_INNER_T0_AUDIT_RUN_ID", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
)
TARGET_CONTEXT_PATH = Path(
    os.environ.get("GM_INNER_T0_TARGET_CONTEXT", str(ROOT / "gm_c_baseline_targets.csv"))
).resolve()
FORBIDDEN_PATH = Path(
    os.environ.get("GM_INNER_T0_FORBIDDEN_SYMBOLS", str(ROOT / "gm_c_forbidden_symbols.csv"))
).resolve()
MODEL_MANIFEST_PATH = Path(
    os.environ.get("GM_INNER_T0_MODEL_MANIFEST", str(ROOT / "frozen_models_manifest.json"))
).resolve()

DECISION_TIME = "10:00:05"
DECISION_BAR_END = "10:00:00"
TRIGGER_START = "10:00:00"
TRIGGER_END = "11:00:00"
EXIT_TIME = "14:45:00"
EXIT_END = "14:50:00"
MAX_SYMBOLS = 3
MAX_DAILY_TURNOVER = 0.03
MIN_FEATURE_UNIVERSE = 20
MIN_FEATURE_SUCCESS_RATIO = 0.70
MAX_TARGET_MISSING_RATIO = 0.20
BUY_COST = 0.001
SELL_COST = 0.0025
MIN_COST = 5.0
LIMIT_BUFFER = 0.002

_models: dict[str, CatBoostClassifier] = {}
_model_meta: dict[str, dict[str, Any]] = {}
_calibration: dict[str, np.ndarray] = {}
_target_context: dict[str, dict[str, float]] = {}
_target_source_date: str | None = None
_forbidden_local: set[str] = set()
_forbidden_gm: set[str] = set()
_decision_scores: list[dict[str, Any]] = []
_candidate_intents: list[dict[str, Any]] = []
_entry_intents: list[dict[str, Any]] = []
_exit_intents: list[dict[str, Any]] = []
_runtime_events: list[dict[str, Any]] = []
_daily_candidates: dict[str, dict[str, Any]] = {}
_daily_entries: dict[str, dict[str, Any]] = {}
_decision_dates: set[str] = set()
_exit_dates: set[str] = set()
_daily_turnover_reserved = 0.0
_daily_buy_cash_reserved = 0.0
_daily_nav = 0.0
_daily_cash = 0.0
_subscribed_symbols: set[str] = set()
_successful_exit_keys: set[tuple[str, str]] = set()


def _audit_path(name: str) -> Path:
    return AUDIT_DIR / AUDIT_RUN_ID / name


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
    print(f"[INNER-T0] {event} {values}", flush=True)


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
    print(f"[INNER-T0] forbidden loaded local={len(local)} gm={len(gm)}", flush=True)


def _load_target_context(today: str) -> None:
    global _target_context, _target_source_date
    if not TARGET_CONTEXT_PATH.exists():
        raise FileNotFoundError(f"target context missing: {TARGET_CONTEXT_PATH}")
    frame = pd.read_csv(TARGET_CONTEXT_PATH)
    if "instrument" not in frame.columns or "trade_date" not in frame.columns:
        raise ValueError("target context requires instrument and trade_date")
    frame["instrument"] = frame["instrument"].map(_gm_to_local_symbol)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    eligible = frame[frame["trade_date"] <= today].copy()
    if eligible.empty:
        raise RuntimeError(f"no target snapshot on or before {today}")
    _target_source_date = str(eligible["trade_date"].max())
    source = eligible[eligible["trade_date"] == _target_source_date].copy()
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
        f"[INNER-T0] target context rows={len(_target_context)} source_date={_target_source_date}",
        flush=True,
    )


def _load_models() -> None:
    if not MODEL_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"model manifest missing: {MODEL_MANIFEST_PATH}")
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    for direction in ("buy_first", "sell_first"):
        item = manifest["models"][direction]
        model_path = ROOT / Path(item["model_path"]).name
        calibration_path = ROOT / Path(item["calibration_path"]).name
        meta_path = ROOT / Path(item["meta_path"]).name
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("deployment_allowed") is not False:
            raise RuntimeError(f"model metadata is not deployment-disabled: {meta_path}")
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        values = np.asarray(np.load(calibration_path), dtype=float)
        values = np.sort(values[np.isfinite(values)])
        if len(values) < 100:
            raise RuntimeError(f"insufficient calibration values: {calibration_path}")
        _models[direction] = model
        _model_meta[direction] = meta
        _calibration[direction] = values
        print(
            f"[INNER-T0] model loaded direction={direction} features={len(meta['feature_cols'])} "
            f"calibration={len(values)}",
            flush=True,
        )


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


def _history_at_decision(symbol: str, today: str) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    end_time = f"{today} {DECISION_BAR_END}"
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
    session, previous_close = current_session_bars(normalized, today, 600)
    if session.empty or not np.isfinite(previous_close) or previous_close <= 0:
        unit_audit = {**unit_audit, "valid": False, "reason": "missing_session_or_previous_close"}
        return pd.DataFrame(), np.nan, unit_audit
    return session, previous_close, unit_audit


def _score_frame(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for direction in ("buy_first", "sell_first"):
        columns = list(_model_meta[direction]["feature_cols"])
        matrix = out.reindex(columns=columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = _models[direction].predict_proba(matrix)[:, 1]
        out[f"{direction}_raw_score"] = raw
        out[f"{direction}_score"] = [
            percentile_score(value, _calibration[direction]) for value in raw
        ]
    return out


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
    for name, rows in tables.items():
        if rows:
            pd.DataFrame(rows).to_csv(out / name, index=False, encoding="utf-8-sig")
    triggered_keys = {
        (str(row.get("date")), str(row.get("symbol")))
        for row in _entry_intents
        if row.get("action") == "TRIGGERED"
    }
    exit_keys = {
        (str(row.get("date")), str(row.get("symbol")))
        for row in _exit_intents
        if row.get("action") in {"BUYBACK_INTENT", "SELL_OLD_INVENTORY_INTENT"}
    }
    unmatched = [
        {"date": date, "symbol": symbol}
        for date, symbol in sorted(triggered_keys - exit_keys)
    ]
    summary = {
        "status": "inner_t0_bidirectional_dryrun_audit",
        "run_mode": RUN_MODE_NAME,
        "dry_run": DRY_RUN,
        "account_id": GM_ACCOUNT_ID,
        "model_manifest": str(MODEL_MANIFEST_PATH),
        "target_context": str(TARGET_CONTEXT_PATH),
        "target_source_date": _target_source_date,
        "decision_dates": sorted(_decision_dates),
        "exit_dates": sorted(_exit_dates),
        "decision_score_rows": len(_decision_scores),
        "candidate_rows": len(_candidate_intents),
        "entry_rows": len(_entry_intents),
        "entry_triggered_rows": sum(row.get("action") == "TRIGGERED" for row in _entry_intents),
        "exit_rows": len(_exit_intents),
        "successful_exit_intents": sum(row.get("action") in {"BUYBACK_INTENT", "SELL_OLD_INVENTORY_INTENT"} for row in _exit_intents),
        "unmatched_entry_symbols": unmatched,
        "max_symbols_per_day": MAX_SYMBOLS,
        "max_daily_turnover": MAX_DAILY_TURNOVER,
        "selected_tick_subscriptions_only": True,
        "actual_submission_api_present": False,
        "deployment_allowed": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def on_decision_scan(context):
    global _daily_candidates, _daily_entries, _daily_turnover_reserved
    global _daily_buy_cash_reserved, _daily_nav, _daily_cash, _subscribed_symbols
    today = _today(context)
    if today in _decision_dates:
        return
    _decision_dates.add(today)
    _daily_candidates = {}
    _daily_entries = {}
    _daily_turnover_reserved = 0.0
    _daily_buy_cash_reserved = 0.0
    try:
        if _subscribed_symbols:
            try:
                unsubscribe(symbols=sorted(_subscribed_symbols), frequency="tick")  # noqa: F405
                _subscribed_symbols.clear()
            except Exception as exc:
                _event(context, "STALE_TICK_UNSUBSCRIBE_FAILED", reason=str(exc))
        _load_target_context(today)
        positions = _fetch_positions(context)
        _daily_nav, _daily_cash = _fetch_cash(positions)
        allowed = _dynamic_allowed(context, positions)
        rows = []
        failures = 0
        for position in allowed:
            symbol = str(position["symbol"])
            try:
                day, previous_close, unit_audit = _history_at_decision(symbol, today)
                visible = visible_features(day, 600, previous_close)
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
                _event(context, "FEATURE_SKIP", symbol=symbol, reason=str(exc))
        success_ratio = len(rows) / max(len(allowed), 1)
        if len(rows) < MIN_FEATURE_UNIVERSE or success_ratio < MIN_FEATURE_SUCCESS_RATIO:
            raise RuntimeError(
                f"feature universe gate failed rows={len(rows)} allowed={len(allowed)} ratio={success_ratio:.3f}"
            )
        feature_frame = add_held_cross_sectional_features(pd.DataFrame(rows))
        missing_ratio = float(pd.to_numeric(feature_frame["target_missing"], errors="coerce").mean())
        if missing_ratio > MAX_TARGET_MISSING_RATIO:
            raise RuntimeError(f"target context mismatch ratio={missing_ratio:.3f}")
        scored = _score_frame(feature_frame)
        score_rows = []
        for row in scored.itertuples(index=False):
            payload = {
                "date": today,
                "time": _now_time(context),
                "symbol": row.symbol,
                "local_symbol": row.local_symbol,
                "held_volume": int(row.held_volume),
                "available_volume": int(row.available_volume),
                "decision_price": float(row.decision_price),
                "previous_close": float(row.mark_price),
                "target_missing": float(row.target_missing),
                "buy_raw_score": float(row.buy_first_raw_score),
                "buy_score": float(row.buy_first_score),
                "sell_raw_score": float(row.sell_first_raw_score),
                "sell_score": float(row.sell_first_score),
                "volume_scale": row.volume_scale,
                "features": json.dumps(
                    {col: finite_float(getattr(row, col, 0.0), 0.0) for col in _model_meta["buy_first"]["feature_cols"]},
                    ensure_ascii=False,
                ),
            }
            score_rows.append(payload)
            _decision_scores.append(payload)
        selected, conflicts = select_bidirectional_candidates(pd.DataFrame(score_rows))
        for row in selected.to_dict("records"):
            candidate = {
                "date": today,
                "time": _now_time(context),
                **row,
                "dry_run": True,
                "action": "CANDIDATE",
            }
            _candidate_intents.append(candidate)
            _daily_candidates[str(row["symbol"])] = candidate
        for row in conflicts.to_dict("records"):
            _candidate_intents.append(
                {
                    "date": today,
                    "time": _now_time(context),
                    **row,
                    "dry_run": True,
                    "action": "DROP_BIDIRECTIONAL_CONFLICT",
                }
            )
        symbols = sorted(_daily_candidates)
        if symbols:
            try:
                subscribe(symbols=symbols, frequency="tick", count=1, unsubscribe_previous=False)  # noqa: F405
                _subscribed_symbols.update(symbols)
                _event(context, "SELECTED_TICK_SUBSCRIBE", symbols=",".join(symbols), count=len(symbols))
            except Exception as exc:
                _event(context, "TICK_SUBSCRIBE_FAILED_POLL_FALLBACK", symbols=",".join(symbols), reason=str(exc))
        _event(
            context,
            "DECISION_COMPLETE",
            positions=len(positions),
            allowed=len(allowed),
            feature_rows=len(rows),
            failures=failures,
            success_ratio=success_ratio,
            target_missing_ratio=missing_ratio,
            candidates=len(symbols),
        )
        _write_audit()
    except Exception as exc:
        _event(context, "DECISION_BLOCKED", reason=str(exc))
        traceback.print_exc()
        _write_audit()


def _tick_price(tick: Any) -> tuple[str, float]:
    symbol = str(_field(tick, "symbol", "")).upper()
    price = finite_float(_field(tick, "price", _field(tick, "last_price", np.nan)), np.nan)
    return symbol, price


def _process_trigger(context, symbol: str, price: float, source: str) -> None:
    global _daily_turnover_reserved, _daily_buy_cash_reserved
    if symbol not in _daily_candidates or symbol in _daily_entries:
        return
    candidate = _daily_candidates[symbol]
    if not trigger_reached(str(candidate["direction"]), price, float(candidate["entry_limit"])):
        return
    intent, accepted = build_entry_intent(
        candidate,
        trigger_price=price,
        nav=_daily_nav,
        cash_available=_daily_cash,
        turnover_used=_daily_turnover_reserved,
        buy_cash_reserved=_daily_buy_cash_reserved,
        max_daily_turnover=MAX_DAILY_TURNOVER,
        buy_cost=BUY_COST,
        sell_cost=SELL_COST,
        min_cost=MIN_COST,
    )
    event = {
        "date": _today(context),
        "time": _now_time(context),
        **intent,
        "trigger_source": source,
        "dry_run": True,
    }
    _entry_intents.append(event)
    if accepted:
        _daily_entries[symbol] = event
        _daily_turnover_reserved += float(event["reserved_roundtrip_turnover"])
        if event["direction"] == "buy_first":
            _daily_buy_cash_reserved += float(event["entry_value"]) + float(event["entry_fee_est"])
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
                exit_intent = build_exit_intent(
                    entry,
                    exit_price=price,
                    buy_cost=BUY_COST,
                    sell_cost=SELL_COST,
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


def init(context):
    print("[INNER-T0] bidirectional dry-run init", flush=True)
    if not DRY_RUN:
        raise RuntimeError("This package is permanently intent-only; GM_INNER_T0_DRY_RUN must remain 1")
    if GM_TOKEN:
        set_token(GM_TOKEN)  # noqa: F405
    if GM_ACCOUNT_ID:
        set_account_id(GM_ACCOUNT_ID)  # noqa: F405
    _load_forbidden()
    _load_target_context(_today(context))
    _load_models()
    schedule(schedule_func=on_decision_scan, date_rule="1d", time_rule=DECISION_TIME)  # noqa: F405
    schedule(schedule_func=on_trigger_poll, date_rule="1d", time_rule="60s")  # noqa: F405
    schedule(schedule_func=on_exit_scan, date_rule="1d", time_rule=EXIT_TIME)  # noqa: F405
    now = _now_time(context)
    if TRIGGER_START <= now <= TRIGGER_END:
        print(f"[INNER-T0] catch-up decision scan at {now}", flush=True)
        on_decision_scan(context)
        on_trigger_poll(context)
    elif EXIT_TIME <= now <= EXIT_END:
        print(f"[INNER-T0] catch-up exit scan at {now}", flush=True)
        on_exit_scan(context)
    print(
        f"[INNER-T0] DRY_RUN={DRY_RUN} account={GM_ACCOUNT_ID} decision={DECISION_TIME} "
        f"trigger={TRIGGER_START}-{TRIGGER_END} exit={EXIT_TIME}",
        flush=True,
    )


def on_backtest_finished(context, indicator):
    _write_audit()


def on_error(context, code, info):
    print(f"[INNER-T0-ERROR] code={code} info={info}", flush=True)


if __name__ == "__main__":
    print("[INNER-T0] __main__ entered", flush=True)
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
