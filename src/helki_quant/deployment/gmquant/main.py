# coding=utf-8
"""
HelkiQuant GmQuant 模拟交易入口

当前策略默认用于掘金模拟实盘。若需要历史回测，请显式设置 GM_MODE=BACKTEST。

用法：
1. 先在本地生成目标权重：
   python -m helki_quant.research.export_paper_forward_gm_targets
2. 在掘金终端运行本文件，或本地设置 GM_TOKEN 后运行：
   python main.py

可选环境变量：
- GM_C_TARGETS: 目标权重 CSV，默认使用 paper-forward 2026-06-05 no-ST Top150 版本
- GM_MODE: BACKTEST 或 LIVE，默认 LIVE
- GM_C_ALLOW_LIVE: 默认允许 PAPER 模拟；若要额外加锁，可设为其他值阻断 LIVE
- GM_C_TRADING_ENV: LIVE 模式环境声明，默认 PAPER
- GM_ACCOUNT_ID: 可选；设置后订单显式发往该模拟账户，否则使用掘金终端当前选中账户
- GM_C_REQUIRE_ACCOUNT_ID: 若设为 1，则 LIVE/PAPER 必须设置 GM_ACCOUNT_ID
- GM_C_BACKTEST_START / GM_C_BACKTEST_END: 回测区间
- GM_C_REBALANCE_PHASE: AT_ONCE 或 SELL_FIRST，默认 AT_ONCE，SELL_FIRST 为风控实验模式
- GM_C_FORBIDDEN_SYMBOLS: 禁入标的 CSV，默认读取 gm_c_forbidden_symbols.csv
- GM_C_DYNAMIC_ST_CHECK: LIVE/PAPER 每日查询最新简称和停牌状态，默认 1
- GM_C_DYNAMIC_ST_FAIL_CLOSED: 动态风险查询失败时阻断调仓，默认 1
- GM_C_REQUIRE_SIGNAL_DATE: LIVE/PAPER 要求 target 带唯一 signal_date，默认 1
- GM_C_MAX_LIVE_SIGNAL_AGE_DAYS: signal 相对启动日最大日历天数，默认 4
- GM_C_MAX_LIVE_SIGNAL_TO_TARGET_DAYS: signal 到 target 最大日历天数，默认 4
- GM_C_SYNC_EXISTING_POSITIONS: LIVE/PAPER 启动时同步账户已有持仓，默认 1
- GM_C_SUBSCRIBE_TARGETS: 是否订阅所有 target 股票行情，默认 1
- GM_C_REBALANCE_ON_INIT: 启动时若已过 09:31 且当天是 target date，立即补调仓，默认 LIVE=1
- GM_C_EXECUTION_MODE: AT_ONCE 或 TICK_EXEC，默认 LIVE=TICK_EXEC / BACKTEST=AT_ONCE
- GM_C_MAX_TICK_SUBSCRIPTIONS: LIVE tick 订阅上限，默认 45；超出部分按目标量兜底执行
- GM_C_EXEC_START / GM_C_EXEC_FORCE_TIME / GM_C_EXEC_END: tick 执行窗口
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import pandas as pd
from gm.api import *  # noqa: F401,F403
from paper_activation_registry import (
    EVENT_ERROR,
    EVENT_FINALIZED,
    EVENT_READY,
    EVENT_STARTED,
    append_activation_event,
    build_activation_identity,
    summarize_paper_session,
)


ROOT = Path(__file__).resolve().parent
ROOT_TARGETS = ROOT / "gm_c_baseline_targets.csv"
DEFAULT_TARGETS = ROOT_TARGETS
ROOT_FORBIDDEN = ROOT / "gm_c_forbidden_symbols.csv"

GM_TOKEN = os.environ.get("GM_TOKEN", "").strip()
GM_STRATEGY_ID = os.environ.get("GM_STRATEGY_ID", "c-baseline-paper").strip()
GM_ACCOUNT_ID = os.environ.get("GM_ACCOUNT_ID", "").strip()
TARGETS_ENV = os.environ.get("GM_C_TARGETS", "").strip()
FORBIDDEN_ENV = os.environ.get("GM_C_FORBIDDEN_SYMBOLS", "").strip()
LIVE_UNLOCK = "C_BASELINE_APPROVED_FOR_PAPER_TRADING"
LIVE_ALLOWED = os.environ.get("GM_C_ALLOW_LIVE", LIVE_UNLOCK) == LIVE_UNLOCK
TRADING_ENV = os.environ.get("GM_C_TRADING_ENV", "PAPER").strip().upper()
RUN_MODE_ENV = os.environ.get("GM_MODE", "").strip().upper()
RUN_MODE = RUN_MODE_ENV or "LIVE"
INITIAL_CASH = float(os.environ.get("GM_C_INITIAL_CASH", "1000000"))
BACKTEST_START = os.environ.get("GM_C_BACKTEST_START", "2025-01-03 09:00:00")
BACKTEST_END = os.environ.get("GM_C_BACKTEST_END", "2026-04-03 15:30:00")
BACKTEST_COMMISSION = float(os.environ.get("GM_C_COMMISSION", "0.0025"))
BACKTEST_SLIPPAGE = float(os.environ.get("GM_C_SLIPPAGE", "0.0005"))
AUDIT_DIR = Path(os.environ.get("GM_C_AUDIT_DIR", str(ROOT / "outputs" / "gm_c_baseline_audit"))).resolve()
AUDIT_RUN_ID = os.environ.get("GM_C_AUDIT_RUN_ID", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
ACTIVATION_REGISTRY = Path(
    os.environ.get(
        "GM_C_ACTIVATION_REGISTRY",
        str(AUDIT_DIR / "PAPER_ACTIVATION_REGISTRY.jsonl"),
    )
).resolve()
REQUIRE_ACTIVATION_AUDIT = os.environ.get(
    "GM_C_REQUIRE_ACTIVATION_AUDIT",
    "1",
).upper() not in {"0", "FALSE", "NO"}
ORDER_STYLE = os.environ.get("GM_C_ORDER_STYLE", "VOLUME").upper()
VERBOSE_ORDERS = os.environ.get("GM_C_VERBOSE_ORDERS", "0").upper() not in {"0", "FALSE", "NO"}
REBALANCE_PHASE = os.environ.get("GM_C_REBALANCE_PHASE", "AT_ONCE").upper()
PAUSE_BUYS_ON_SELL_REJECT = os.environ.get("GM_C_PAUSE_BUYS_ON_SELL_REJECT", "1").upper() not in {
    "0",
    "FALSE",
    "NO",
}
SELL_REJECT_PAUSE_DAYS = int(os.environ.get("GM_C_SELL_REJECT_PAUSE_DAYS", "1"))
MAX_LIVE_TARGET_FORWARD_DAYS = int(os.environ.get("GM_C_MAX_LIVE_TARGET_FORWARD_DAYS", "7"))
REQUIRE_SIGNAL_DATE = os.environ.get("GM_C_REQUIRE_SIGNAL_DATE", "1").upper() not in {
    "0",
    "FALSE",
    "NO",
}
MAX_LIVE_SIGNAL_AGE_DAYS = int(os.environ.get("GM_C_MAX_LIVE_SIGNAL_AGE_DAYS", "4"))
MAX_LIVE_SIGNAL_TO_TARGET_DAYS = int(
    os.environ.get("GM_C_MAX_LIVE_SIGNAL_TO_TARGET_DAYS", "4")
)
SYNC_EXISTING_POSITIONS = os.environ.get("GM_C_SYNC_EXISTING_POSITIONS", "1").upper() not in {
    "0",
    "FALSE",
    "NO",
}
REQUIRE_ACCOUNT_ID = os.environ.get("GM_C_REQUIRE_ACCOUNT_ID", "0").upper() in {
    "1",
    "TRUE",
    "YES",
}
SUBSCRIBE_TARGETS_ENV = os.environ.get("GM_C_SUBSCRIBE_TARGETS", "").strip().upper()
SUBSCRIBE_TARGETS = (
    SUBSCRIBE_TARGETS_ENV not in {"0", "FALSE", "NO"}
    if SUBSCRIBE_TARGETS_ENV
    else True
)
REBALANCE_ON_INIT_ENV = os.environ.get("GM_C_REBALANCE_ON_INIT", "").strip().upper()
REBALANCE_ON_INIT = (
    REBALANCE_ON_INIT_ENV not in {"0", "FALSE", "NO"}
    if REBALANCE_ON_INIT_ENV
    else RUN_MODE != "BACKTEST"
)
INIT_REBALANCE_AFTER = os.environ.get("GM_C_INIT_REBALANCE_AFTER", "09:31:00")
INIT_REBALANCE_BEFORE = os.environ.get("GM_C_INIT_REBALANCE_BEFORE", "14:55:00")
EXECUTION_MODE = os.environ.get(
    "GM_C_EXECUTION_MODE",
    "AT_ONCE" if RUN_MODE == "BACKTEST" else "TICK_EXEC",
).strip().upper()
EXEC_START = os.environ.get("GM_C_EXEC_START", "09:31:00")
EXEC_FORCE_TIME = os.environ.get("GM_C_EXEC_FORCE_TIME", "14:45:00")
EXEC_END = os.environ.get("GM_C_EXEC_END", "14:55:00")
EXEC_SCAN_INTERVAL_SECONDS = int(os.environ.get("GM_C_EXEC_SCAN_INTERVAL_SECONDS", "10"))
EXEC_MAX_CHILD_FRACTION = float(os.environ.get("GM_C_EXEC_MAX_CHILD_FRACTION", "0.25"))
EXEC_BUY_MAX_PREMIUM_BPS = float(os.environ.get("GM_C_BUY_MAX_PREMIUM_BPS", "30"))
EXEC_SELL_MAX_DISCOUNT_BPS = float(os.environ.get("GM_C_SELL_MAX_DISCOUNT_BPS", "30"))
MAX_TICK_SUBSCRIPTIONS = int(os.environ.get("GM_C_MAX_TICK_SUBSCRIPTIONS", "45"))
DYNAMIC_ST_CHECK = os.environ.get("GM_C_DYNAMIC_ST_CHECK", "1").upper() not in {
    "0",
    "FALSE",
    "NO",
}
DYNAMIC_ST_FAIL_CLOSED = os.environ.get("GM_C_DYNAMIC_ST_FAIL_CLOSED", "1").upper() not in {
    "0",
    "FALSE",
    "NO",
}

_targets_by_date: dict[str, dict[str, dict[str, float]]] = {}
_all_target_dates: set[str] = set()
_signal_dates_by_target_date: dict[str, tuple[str, ...]] = {}
_previous_symbols: set[str] = set()
_previous_volumes: dict[str, int] = {}
_position_sync_succeeded = False
_position_sync_source: str | None = None
_pending_target_orders: dict[str, dict[str, object]] = {}
_last_rebalance_date: str | None = None
_pending_buy_targets: dict[str, dict[str, float]] = {}
_pending_buy_source_date: str | None = None
_pause_buys_until: str | None = None
_forbidden_local_symbols: set[str] = set()
_forbidden_gm_symbols: set[str] = set()
_dynamic_risk_events: list[dict[str, object]] = []
_dynamic_risk_check_date: str | None = None
_forbidden_clear_pending: set[str] = set()
_submission_events: list[dict[str, object]] = []
_order_events: list[dict[str, object]] = []
_rebalance_events: list[dict[str, object]] = []
_execution_targets: dict[str, dict[str, object]] = {}
_execution_date: str | None = None
_last_exec_scan_ts: dict[str, pd.Timestamp] = {}
_execution_events: list[dict[str, object]] = []
_tick_execution_available = True
_tick_subscribed_symbols: set[str] = set()
_activation_identity: dict[str, object] | None = None
_activation_ready_recorded = False
_activation_finalized = False

STATUS_NAMES = {
    0: "Unknown",
    1: "New",
    2: "PartiallyFilled",
    3: "Filled",
    4: "DoneForDay",
    5: "Canceled",
    6: "PendingCancel",
    7: "Stopped",
    8: "Rejected",
    9: "Suspended",
    10: "PendingNew",
    11: "Calculated",
    12: "Expired",
    13: "AcceptedForBidding",
    14: "PendingReplace",
}
TERMINAL_STATUS = {3, 4, 5, 7, 8, 12}


def _resolve_targets_path() -> Path:
    if TARGETS_ENV:
        return Path(TARGETS_ENV).resolve()
    for candidate in (ROOT_TARGETS, DEFAULT_TARGETS):
        if candidate.exists():
            return candidate.resolve()
    return ROOT_TARGETS.resolve()


TARGETS_PATH = _resolve_targets_path()
FORBIDDEN_PATH = Path(FORBIDDEN_ENV).resolve() if FORBIDDEN_ENV else ROOT_FORBIDDEN.resolve()


def _activation_time(context=None) -> object:
    return getattr(context, "now", None) or pd.Timestamp.now(
        tz="Asia/Shanghai"
    )


def _start_activation_audit(context) -> None:
    global _activation_identity
    if RUN_MODE == "BACKTEST" or not REQUIRE_ACTIVATION_AUDIT:
        return
    _activation_identity = build_activation_identity(
        package_dir=ROOT,
        target_path=TARGETS_PATH,
        forbidden_path=FORBIDDEN_PATH,
        account_id=GM_ACCOUNT_ID,
        run_id=AUDIT_RUN_ID,
        strategy_id=GM_STRATEGY_ID,
        run_mode=RUN_MODE,
        trading_env=TRADING_ENV,
    )
    append_activation_event(
        ACTIVATION_REGISTRY,
        event=EVENT_STARTED,
        identity=_activation_identity,
        timestamp=_activation_time(context),
    )
    print(
        "[GM-C-ACTIVATION] STARTED "
        f"run={AUDIT_RUN_ID} target_sha256="
        f"{_activation_identity['target_sha256']} registry={ACTIVATION_REGISTRY}",
        flush=True,
    )


def _mark_activation_ready(context) -> None:
    global _activation_ready_recorded
    if _activation_identity is None or _activation_ready_recorded:
        return
    append_activation_event(
        ACTIVATION_REGISTRY,
        event=EVENT_READY,
        identity=_activation_identity,
        timestamp=_activation_time(context),
        metrics={
            "position_sync_succeeded": bool(_position_sync_succeeded),
            "position_sync_source": _position_sync_source,
            "dynamic_risk_check_date": _dynamic_risk_check_date,
            "tick_execution_available": bool(_tick_execution_available),
            "tick_subscribed_symbols": int(len(_tick_subscribed_symbols)),
        },
    )
    _activation_ready_recorded = True
    print(
        f"[GM-C-ACTIVATION] READY run={AUDIT_RUN_ID}",
        flush=True,
    )


def _finalize_activation(context) -> None:
    global _activation_finalized
    if (
        _activation_identity is None
        or not _activation_ready_recorded
        or _activation_finalized
    ):
        return
    today = _today(context)
    target_date = str(_activation_identity["trade_date"])
    if today != target_date or _last_rebalance_date != target_date:
        append_activation_event(
            ACTIVATION_REGISTRY,
            event=EVENT_ERROR,
            identity=_activation_identity,
            timestamp=_activation_time(context),
            metrics={
                "reason": "session_not_completed_on_target_date",
                "today": today,
                "target_date": target_date,
                "last_rebalance_date": _last_rebalance_date,
            },
        )
        print(
            "[GM-C-ACTIVATION] FINALIZE BLOCKED "
            f"today={today} target={target_date} "
            f"last_rebalance={_last_rebalance_date}",
            flush=True,
        )
        return
    try:
        _sync_existing_positions(context)
    except Exception as exc:
        append_activation_event(
            ACTIVATION_REGISTRY,
            event=EVENT_ERROR,
            identity=_activation_identity,
            timestamp=_activation_time(context),
            metrics={
                "reason": "final_position_sync_failed",
                "error": str(exc),
            },
        )
        print(
            f"[GM-C-ACTIVATION] FINALIZE BLOCKED position sync failed: {exc}",
            flush=True,
        )
        return
    if not _position_sync_succeeded:
        append_activation_event(
            ACTIVATION_REGISTRY,
            event=EVENT_ERROR,
            identity=_activation_identity,
            timestamp=_activation_time(context),
            metrics={"reason": "final_position_sync_unavailable"},
        )
        print(
            "[GM-C-ACTIVATION] FINALIZE BLOCKED final position sync unavailable",
            flush=True,
        )
        return
    target_volumes = {
        symbol: max(0, int(float(row.get("target_shares", 0))))
        for symbol, row in _targets_by_date[target_date].items()
    }
    session_quality = summarize_paper_session(
        order_events=_order_events,
        target_volumes=target_volumes,
        actual_volumes=_previous_volumes,
        deferred_buy_symbols=set(_pending_buy_targets),
    )
    append_activation_event(
        ACTIVATION_REGISTRY,
        event=EVENT_FINALIZED,
        identity=_activation_identity,
        timestamp=_activation_time(context),
        metrics={
            "submitted_orders": int(len(_submission_events)),
            "order_status_events": int(len(_order_events)),
            "rebalance_events": int(len(_rebalance_events)),
            "execution_events": int(len(_execution_events)),
            "pending_target_order_symbols": sorted(_pending_target_orders),
            "pending_execution_symbols": sorted(_execution_targets),
            "forbidden_clear_pending": sorted(_forbidden_clear_pending),
            "pending_buy_symbols": sorted(_pending_buy_targets),
            "position_sync_succeeded_at_finalize": bool(
                _position_sync_succeeded
            ),
            "position_sync_source_at_finalize": _position_sync_source,
            **session_quality,
        },
    )
    _activation_finalized = True
    print(
        f"[GM-C-ACTIVATION] FINALIZED run={AUDIT_RUN_ID}",
        flush=True,
    )


def _record_activation_error(context, code, info) -> None:
    if _activation_identity is None:
        return
    append_activation_event(
        ACTIVATION_REGISTRY,
        event=EVENT_ERROR,
        identity=_activation_identity,
        timestamp=_activation_time(context),
        metrics={"code": str(code), "info": str(info)},
    )


def _order_field(order, name: str, default=None):
    if isinstance(order, dict):
        return order.get(name, default)
    return getattr(order, name, default)


def _is_invalid_account_error(exc: Exception) -> bool:
    text = str(exc)
    return "1020" in text or "ACCOUNT_ID" in text or "无效的ACCOUNT_ID" in text


def _position_field(position, name: str, default=None):
    if isinstance(position, dict):
        return position.get(name, default)
    return getattr(position, name, default)


def _min_buy_volume(symbol: str) -> int:
    code = str(symbol).upper()[-6:]
    if code.startswith(("688", "689")):
        return 200
    return 100


def _gm_to_local_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if "." not in text:
        return text
    exchange, code = text.split(".", 1)
    prefix = "SH" if exchange in {"SHSE", "SH"} else "SZ"
    return f"{prefix}{code}"


def _local_to_gm_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    code = text[-6:]
    if text.startswith("SH") or code.startswith(("6", "9")):
        return f"SHSE.{code}"
    return f"SZSE.{code}"


def _load_forbidden_symbols(path: Path) -> tuple[set[str], set[str]]:
    if not path.exists():
        print(f"[GM-C-RISK] forbidden symbols file not found: {path}", flush=True)
        return set(), set()
    frame = pd.read_csv(path, dtype=str).fillna("")
    local: set[str] = set()
    gm: set[str] = set()
    if "instrument" in frame.columns:
        local.update(frame["instrument"].astype(str).str.upper())
        gm.update(frame["instrument"].map(_local_to_gm_symbol).astype(str).str.upper())
    if "local_instrument" in frame.columns:
        local.update(frame["local_instrument"].astype(str).str.upper())
        gm.update(frame["local_instrument"].map(_local_to_gm_symbol).astype(str).str.upper())
    if "gm_symbol" in frame.columns:
        gm.update(frame["gm_symbol"].astype(str).str.upper())
        local.update(frame["gm_symbol"].map(_gm_to_local_symbol).astype(str).str.upper())
    if "symbol" in frame.columns:
        gm.update(frame["symbol"].astype(str).str.upper())
        local.update(frame["symbol"].map(_gm_to_local_symbol).astype(str).str.upper())
    print(
        f"[GM-C-RISK] forbidden symbols loaded local={len(local)} gm={len(gm)} file={path}",
        flush=True,
    )
    return local, gm


def _instrument_field(row, name: str, default=None):
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _dynamic_risk_reason(sec_name: object, is_suspended: object, delisted_date: object, today: str) -> str:
    compact_name = "".join(str(sec_name or "").upper().replace("＊", "*").split())
    reasons: list[str] = []
    if compact_name.startswith(("*ST", "ST", "S*ST", "SST", "PT")):
        reasons.append(f"risk_name:{compact_name}")
    if "退" in compact_name:
        reasons.append(f"delisting_name:{compact_name}")
    try:
        suspended = int(float(is_suspended or 0)) != 0
    except (TypeError, ValueError):
        suspended = str(is_suspended).strip().upper() in {"TRUE", "YES"}
    if suspended:
        reasons.append("suspended")
    delisted_ts = pd.to_datetime(delisted_date, errors="coerce")
    if pd.notna(delisted_ts) and delisted_ts.date() <= pd.Timestamp(today).date():
        reasons.append(f"delisted:{delisted_ts.date()}")
    return "|".join(reasons)


def _refresh_dynamic_market_risk(context, source: str, force: bool = False) -> None:
    global _dynamic_risk_check_date, _forbidden_local_symbols, _forbidden_gm_symbols
    if RUN_MODE == "BACKTEST" or not DYNAMIC_ST_CHECK:
        return
    today = _today(context)
    if not force and _dynamic_risk_check_date == today:
        return
    if _dynamic_risk_check_date != today:
        _forbidden_clear_pending.clear()
    requested = set(_previous_symbols)
    for targets in _targets_by_date.values():
        requested.update(targets)
    symbols = sorted(symbol for symbol in requested if symbol)
    if not symbols:
        _dynamic_risk_check_date = today
        return
    try:
        rows = get_instruments(  # noqa: F405
            symbols=symbols,
            skip_suspended=False,
            skip_st=False,
            fields=[
                "symbol",
                "sec_name",
                "sec_abbr",
                "is_suspended",
                "listed_date",
                "delisted_date",
                "trade_date",
            ],
            df=False,
        )
    except Exception as exc:
        event = {
            "check_date": today,
            "source": source,
            "status": "query_failed",
            "symbol": "",
            "sec_name": "",
            "reason": str(exc),
        }
        _dynamic_risk_events.append(event)
        print(f"[GM-C-RISK] dynamic ST/market-state query failed: {exc}", flush=True)
        if DYNAMIC_ST_FAIL_CLOSED:
            raise RuntimeError("LIVE rebalance blocked: dynamic ST/market-state query failed") from exc
        return

    by_symbol = {
        str(_instrument_field(row, "symbol", "")).strip().upper(): row
        for row in (rows or [])
        if str(_instrument_field(row, "symbol", "")).strip()
    }
    dynamic_forbidden: dict[str, tuple[str, str]] = {}
    for symbol in symbols:
        row = by_symbol.get(symbol)
        if row is None:
            dynamic_forbidden[symbol] = ("", "instrument_lookup_missing")
            continue
        sec_name = _instrument_field(row, "sec_name", "") or _instrument_field(row, "sec_abbr", "")
        reason = _dynamic_risk_reason(
            sec_name,
            _instrument_field(row, "is_suspended", 0),
            _instrument_field(row, "delisted_date", None),
            today,
        )
        if reason:
            dynamic_forbidden[symbol] = (str(sec_name), reason)

    removed_targets = 0
    for symbol, (sec_name, reason) in dynamic_forbidden.items():
        _forbidden_gm_symbols.add(symbol)
        _forbidden_local_symbols.add(_gm_to_local_symbol(symbol))
        removed = 0
        for targets in _targets_by_date.values():
            if symbol in targets:
                del targets[symbol]
                removed += 1
        removed_targets += removed
        _dynamic_risk_events.append(
            {
                "check_date": today,
                "source": source,
                "status": "forbidden",
                "symbol": symbol,
                "sec_name": sec_name,
                "reason": reason,
                "held_shares": int(_previous_volumes.get(symbol, 0)),
                "removed_target_dates": removed,
            }
        )

    _dynamic_risk_check_date = today
    out_dir = AUDIT_DIR / AUDIT_RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_dynamic_risk_events).to_csv(
        out_dir / "dynamic_market_risk.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        "[GM-C-RISK] dynamic ST/market-state check "
        f"date={today} requested={len(symbols)} returned={len(by_symbol)} "
        f"forbidden={len(dynamic_forbidden)} removed_target_rows={removed_targets}",
        flush=True,
    )
    for symbol, (sec_name, reason) in sorted(dynamic_forbidden.items()):
        print(
            f"[GM-C-RISK] DYNAMIC_FORBIDDEN {symbol} name={sec_name} "
            f"reason={reason} held_shares={int(_previous_volumes.get(symbol, 0))}",
            flush=True,
        )


def _audit_path(name: str) -> Path:
    return AUDIT_DIR / AUDIT_RUN_ID / name


def _load_targets(path: Path) -> None:
    global _targets_by_date, _all_target_dates, _signal_dates_by_target_date
    global _forbidden_local_symbols, _forbidden_gm_symbols
    if not path.exists():
        raise FileNotFoundError(
            "目标权重文件不存在。掘金云端运行时，请把 "
            "`gm_c_baseline_targets.csv` 放在 `gm_c_baseline.py` 同级目录，"
            "或设置 GM_C_TARGETS 指向云端可读路径。当前查找路径: "
            f"{path}"
        )
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    required = {"trade_date", "symbol", "target_weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    if "target_shares" not in frame.columns:
        frame["target_shares"] = -1
    if "signal_date" not in frame.columns:
        if RUN_MODE != "BACKTEST" and REQUIRE_SIGNAL_DATE:
            raise ValueError(
                f"{path} 缺少 signal_date；LIVE/PAPER 禁止无法追溯来源日期的 target"
            )
        frame["signal_date"] = pd.NaT
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    if RUN_MODE != "BACKTEST" and REQUIRE_SIGNAL_DATE and frame["signal_date"].isna().any():
        raise ValueError(f"{path} 含无效 signal_date；LIVE/PAPER 必须全部可解析")
    frame["trade_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")
    frame["signal_date"] = frame["signal_date"].dt.strftime("%Y-%m-%d")
    frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0)
    frame["target_shares"] = pd.to_numeric(frame["target_shares"], errors="coerce").fillna(-1).astype(int)
    forbidden_local, forbidden_gm = _load_forbidden_symbols(FORBIDDEN_PATH)
    _forbidden_local_symbols = set(forbidden_local)
    _forbidden_gm_symbols = set(forbidden_gm)
    if forbidden_local or forbidden_gm:
        local_symbol = (
            frame["instrument"].astype(str).str.upper()
            if "instrument" in frame.columns
            else frame["symbol"].map(_gm_to_local_symbol).astype(str).str.upper()
        )
        before = len(frame)
        frame = frame[
            ~frame["symbol"].astype(str).str.upper().isin(forbidden_gm)
            & ~local_symbol.isin(forbidden_local)
        ].copy()
        removed = before - len(frame)
        if removed:
            print(f"[GM-C-RISK] removed forbidden target rows={removed}", flush=True)
    frame = frame[frame["target_weight"] > 0].copy()
    if frame.empty:
        raise ValueError(f"{path} 没有正 target_weight")
    _targets_by_date = {}
    _signal_dates_by_target_date = {}
    for date, part in frame.groupby("trade_date", sort=True):
        signal_dates = tuple(
            sorted(
                {
                    str(value)
                    for value in part["signal_date"].dropna().astype(str)
                    if str(value) and str(value).upper() != "NAT"
                }
            )
        )
        _signal_dates_by_target_date[date] = signal_dates
        _targets_by_date[date] = {
            str(row.symbol): {
                "target_weight": float(row.target_weight),
                "target_shares": int(row.target_shares),
                "price_ref_close": float(getattr(row, "price_ref_close", 0.0) or 0.0),
            }
            for row in part.itertuples(index=False)
        }
    _all_target_dates = set(_targets_by_date)


def _reference_target_date_for_subscription() -> str | None:
    if not _all_target_dates:
        return None
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    future_dates = sorted(date for date in _all_target_dates if date >= today)
    if future_dates:
        return future_dates[0]
    return sorted(_all_target_dates)[-1]


def _ordered_subscription_symbols() -> list[str]:
    target_date = _reference_target_date_for_subscription()
    if target_date is None:
        return []
    targets = _targets_by_date[target_date]
    universe = set(_previous_symbols) | set(targets)

    def priority(symbol: str) -> tuple[int, float, str]:
        current = int(_previous_volumes.get(symbol, 0))
        row = targets.get(symbol, {})
        target = int(row.get("target_shares", 0)) if row else 0
        delta = target - current
        side_priority = 0 if delta < 0 else 1 if delta > 0 else 2
        ref_price = float(row.get("price_ref_close", 0.0) or 0.0) if row else 0.0
        notional = abs(delta) * max(ref_price, 0.0)
        return side_priority, -notional, symbol

    return sorted(universe, key=priority)


def _today(context) -> str:
    try:
        return context.now.strftime("%Y-%m-%d")
    except Exception:
        return pd.Timestamp.now().strftime("%Y-%m-%d")


def _now_time(context) -> str:
    try:
        return context.now.strftime("%H:%M:%S")
    except Exception:
        return pd.Timestamp.now().strftime("%H:%M:%S")


def _validate_live_launch() -> None:
    if RUN_MODE == "BACKTEST":
        return
    if not LIVE_ALLOWED:
        raise RuntimeError(
            "[GM-C] LIVE blocked. Set GM_C_ALLOW_LIVE="
            f"{LIVE_UNLOCK} only after explicit paper-trading approval."
        )
    if TRADING_ENV not in {"PAPER", "SIM", "SIMULATION"}:
        raise RuntimeError(
            "[GM-C] LIVE blocked. Set GM_C_TRADING_ENV=PAPER to confirm this "
            "is simulated/paper trading, not real-money trading."
        )
    if REQUIRE_ACCOUNT_ID and not GM_ACCOUNT_ID:
        raise RuntimeError(
            "[GM-C] LIVE blocked. GM_ACCOUNT_ID is required in paper mode to "
            "avoid submitting orders to a platform default account."
        )
    if not GM_ACCOUNT_ID:
        print(
            "[GM-C-RISK] GM_ACCOUNT_ID not set; using the account selected by "
            "the GmQuant terminal/platform simulation.",
            flush=True,
        )
    target_dates = pd.DatetimeIndex(pd.to_datetime(sorted(_all_target_dates)))
    if target_dates.empty:
        raise RuntimeError("[GM-C] LIVE blocked. No target dates were loaded.")
    now = pd.Timestamp.now()
    today = now.normalize()
    future_dates = target_dates[target_dates >= today]
    if future_dates.empty:
        raise RuntimeError(
            "[GM-C] LIVE blocked. Target file is stale: latest target date is "
            f"{target_dates.max().date()}, today is {today.date()}."
        )
    next_target = pd.Timestamp(future_dates.min()).normalize()
    forward_days = int((next_target - today).days)
    if forward_days > MAX_LIVE_TARGET_FORWARD_DAYS:
        raise RuntimeError(
            "[GM-C] LIVE blocked. Next target date is too far ahead: "
            f"{next_target.date()} > {MAX_LIVE_TARGET_FORWARD_DAYS} days from today."
        )
    signal_date = _validate_live_signal_context(now, next_target)
    print(
        "[GM-C-RISK] paper LIVE launch validated "
        f"account={GM_ACCOUNT_ID} trading_env={TRADING_ENV} "
        f"next_target={next_target.date()} signal_date="
        f"{signal_date.date() if signal_date is not None else 'not_required'}",
        flush=True,
    )


def _validate_live_signal_context(
    now: pd.Timestamp,
    next_target: pd.Timestamp,
) -> pd.Timestamp | None:
    if not REQUIRE_SIGNAL_DATE:
        return None
    target_key = pd.Timestamp(next_target).strftime("%Y-%m-%d")
    signal_dates = _signal_dates_by_target_date.get(target_key, ())
    if len(signal_dates) != 1:
        raise RuntimeError(
            "[GM-C] LIVE blocked. Target must have exactly one signal_date for "
            f"{target_key}; observed={list(signal_dates)}."
        )
    signal_date = pd.Timestamp(signal_dates[0]).normalize()
    target_date = pd.Timestamp(next_target).normalize()
    now = pd.Timestamp(now)
    today = now.normalize()
    if signal_date >= target_date:
        raise RuntimeError(
            "[GM-C] LIVE blocked. signal_date must be earlier than target date: "
            f"signal={signal_date.date()} target={target_date.date()}."
        )
    if signal_date > today:
        raise RuntimeError(
            "[GM-C] LIVE blocked. signal_date is in the future relative to launch: "
            f"signal={signal_date.date()} today={today.date()}."
        )
    if signal_date == today and now.strftime("%H:%M:%S") < "15:05:00":
        raise RuntimeError(
            "[GM-C] LIVE blocked. Today's signal session is not complete before 15:05: "
            f"signal={signal_date.date()} now={now.strftime('%H:%M:%S')}."
        )
    signal_age_days = int((today - signal_date).days)
    if signal_age_days > MAX_LIVE_SIGNAL_AGE_DAYS:
        raise RuntimeError(
            "[GM-C] LIVE blocked. signal_date is stale: "
            f"signal={signal_date.date()} today={today.date()} age_days={signal_age_days} "
            f"max={MAX_LIVE_SIGNAL_AGE_DAYS}."
        )
    signal_to_target_days = int((target_date - signal_date).days)
    if signal_to_target_days > MAX_LIVE_SIGNAL_TO_TARGET_DAYS:
        raise RuntimeError(
            "[GM-C] LIVE blocked. signal-to-target lag is too large: "
            f"signal={signal_date.date()} target={target_date.date()} "
            f"lag_days={signal_to_target_days} max={MAX_LIVE_SIGNAL_TO_TARGET_DAYS}."
        )
    return signal_date


def _position_symbol(position) -> str:
    symbol = _position_field(position, "symbol", "")
    return str(symbol or "").strip().upper()


def _position_volume(position) -> int:
    for key in ("volume", "amount", "current_volume", "position_volume"):
        value = _position_field(position, key, None)
        if value is not None:
            try:
                return max(0, int(float(value)))
            except (TypeError, ValueError):
                continue
    return 0


def _fetch_account_positions(context) -> list:
    global _position_sync_succeeded, _position_sync_source
    _position_sync_succeeded = False
    _position_sync_source = None
    attempts = []
    if GM_ACCOUNT_ID:
        attempts.append(("get_position(account_id)", lambda: get_position(account_id=GM_ACCOUNT_ID)))  # noqa: F405
    attempts.append(("get_position()", lambda: get_position()))  # noqa: F405
    if hasattr(context, "account"):
        def _context_positions():
            account_obj = context.account()
            if hasattr(account_obj, "positions"):
                try:
                    return account_obj.positions(side=PositionSide_Long)  # noqa: F405
                except TypeError:
                    return account_obj.positions()
            return []

        attempts.append(("context.account().positions()", _context_positions))

    last_error = None
    for name, getter in attempts:
        try:
            positions = getter()
        except Exception as exc:
            last_error = exc
            continue
        if positions is None:
            continue
        rows = list(positions) if not isinstance(positions, list) else positions
        _position_sync_succeeded = True
        _position_sync_source = name
        print(f"[GM-C-RISK] position sync source={name} rows={len(rows)}", flush=True)
        return rows
    if last_error is not None:
        print(f"[GM-C-RISK] position sync unavailable: {last_error}", flush=True)
    else:
        print("[GM-C-RISK] position sync unavailable: no position API returned rows", flush=True)
    return []


def _sync_existing_positions(context) -> None:
    global _previous_symbols, _previous_volumes
    if not SYNC_EXISTING_POSITIONS:
        return
    positions = _fetch_account_positions(context)
    if RUN_MODE != "BACKTEST" and REQUIRE_ACCOUNT_ID and not _position_sync_succeeded:
        raise RuntimeError(
            "[GM-C] LIVE blocked. Bound PAPER account positions could not be read; "
            "an unknown account state cannot be treated as an empty portfolio."
        )
    volumes: dict[str, int] = {}
    for position in positions:
        symbol = _position_symbol(position)
        volume = _position_volume(position)
        if not symbol or volume <= 0:
            continue
        volumes[symbol] = volumes.get(symbol, 0) + volume
    _previous_volumes = volumes
    _previous_symbols = set(volumes)
    print(
        f"[GM-C-RISK] synced existing positions symbols={len(_previous_symbols)} "
        f"shares={sum(_previous_volumes.values())}",
        flush=True,
    )


def _is_forbidden_position(symbol: str) -> bool:
    return symbol in _forbidden_gm_symbols or _gm_to_local_symbol(symbol) in _forbidden_local_symbols


def _clear_forbidden_positions(context, source: str) -> None:
    if RUN_MODE == "BACKTEST" or not LIVE_ALLOWED:
        return
    today = _today(context)
    forbidden_positions = sorted(symbol for symbol in _previous_symbols if _is_forbidden_position(symbol))
    if not forbidden_positions:
        print("[GM-C-RISK] forbidden held positions=0", flush=True)
        return
    print(
        f"[GM-C-RISK] forbidden held positions detected={len(forbidden_positions)}; "
        "submitting target_volume=0 clears before normal rebalance",
        flush=True,
    )
    for symbol in forbidden_positions:
        prev_shares = int(_previous_volumes.get(symbol, 0))
        if prev_shares <= 0:
            continue
        if symbol in _forbidden_clear_pending:
            print(f"[GM-C-RISK] forbidden clear already pending symbol={symbol}", flush=True)
            continue
        _record_submission(today, source, "clear_forbidden", symbol, 0.0, 0)
        _forbidden_clear_pending.add(symbol)
        try:
            _order_target_volume_safe(symbol, 0)
        except Exception:
            _forbidden_clear_pending.discard(symbol)
            raise
        if VERBOSE_ORDERS:
            print(f"[GM-C-RISK] CLEAR_FORBIDDEN submitted {symbol} {prev_shares}->0", flush=True)


def _subscribe_targets(symbols: list[str]) -> None:
    global _tick_execution_available, _tick_subscribed_symbols
    _tick_subscribed_symbols = set()
    if not SUBSCRIBE_TARGETS:
        print(
            "[GM-C] target subscription skipped "
            f"mode={RUN_MODE} symbols={len(symbols)}; schedule rebalance remains active",
            flush=True,
        )
        if EXECUTION_MODE == "TICK_EXEC" and RUN_MODE != "BACKTEST":
            _tick_execution_available = False
            print("[GM-C-RISK] tick execution disabled because target subscription is off", flush=True)
        return
    if not symbols:
        return
    try:
        frequency = "tick" if EXECUTION_MODE == "TICK_EXEC" and RUN_MODE != "BACKTEST" else "1d"
        request_symbols = list(symbols)
        if frequency == "tick" and MAX_TICK_SUBSCRIPTIONS > 0 and len(request_symbols) > MAX_TICK_SUBSCRIPTIONS:
            request_symbols = request_symbols[:MAX_TICK_SUBSCRIPTIONS]
            print(
                "[GM-C-RISK] tick subscription capped "
                f"requested={len(symbols)} subscribed={len(request_symbols)} "
                f"cap={MAX_TICK_SUBSCRIPTIONS}; remaining symbols use target-volume fallback",
                flush=True,
            )
        subscribe(symbols=",".join(request_symbols), frequency=frequency, count=1)  # noqa: F405
        if frequency == "tick":
            _tick_subscribed_symbols = set(request_symbols)
        print(
            f"[GM-C] subscribed target symbols={len(request_symbols)} "
            f"frequency={frequency} requested={len(symbols)}",
            flush=True,
        )
    except Exception as exc:
        print(
            "[GM-C-RISK] target subscription failed but strategy will continue "
            f"with schedule rebalance only: {exc}",
            flush=True,
        )
        if EXECUTION_MODE == "TICK_EXEC" and RUN_MODE != "BACKTEST":
            _tick_execution_available = False
            _tick_subscribed_symbols = set()
            print("[GM-C-RISK] tick execution unavailable; fallback to AT_ONCE rebalance", flush=True)


def _rebalance_on_init_if_needed(context) -> None:
    if not REBALANCE_ON_INIT:
        return
    today = _today(context)
    now_time = _now_time(context)
    if today not in _all_target_dates:
        print(f"[GM-C] init catch-up skipped: {today} is not a target date", flush=True)
        return
    if now_time < INIT_REBALANCE_AFTER:
        print(
            f"[GM-C] init catch-up skipped: now={now_time} before {INIT_REBALANCE_AFTER}",
            flush=True,
        )
        return
    if now_time > INIT_REBALANCE_BEFORE:
        print(
            f"[GM-C-RISK] init catch-up skipped: now={now_time} after {INIT_REBALANCE_BEFORE}",
            flush=True,
        )
        return
    print(
        f"[GM-C] init catch-up rebalance triggered today={today} now={now_time}",
        flush=True,
    )
    _rebalance(context, "init_catchup")


def _tick_field(tick, name: str, default=None):
    if isinstance(tick, dict):
        return tick.get(name, default)
    return getattr(tick, name, default)


def _tick_symbol(tick) -> str | None:
    if isinstance(tick, (list, tuple)) and tick:
        return _tick_symbol(tick[0])
    value = _tick_field(tick, "symbol", None)
    return str(value).strip().upper() if value else None


def _tick_price(tick) -> float | None:
    if isinstance(tick, (list, tuple)) and tick:
        return _tick_price(tick[0])
    for name in ("price", "last_price", "lastPrice", "last", "close"):
        value = _tick_field(tick, name, None)
        if value is None:
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if price > 0:
            return price
    return None


def _time_in_execution_window(now_time: str) -> bool:
    return EXEC_START <= now_time <= EXEC_END


def _in_force_window(now_time: str) -> bool:
    return EXEC_FORCE_TIME <= now_time <= EXEC_END


def _round_child_volume(symbol: str, remaining: int, force: bool) -> int:
    remaining = abs(int(remaining))
    if remaining <= 0:
        return 0
    if force:
        raw = remaining
    else:
        raw = max(_min_buy_volume(symbol), int(remaining * max(0.01, EXEC_MAX_CHILD_FRACTION)))
    lot = _min_buy_volume(symbol)
    return min(remaining, max(lot, int(raw // lot) * lot))


def _price_gate(symbol: str, side: str, price: float | None, ref_price: float, force: bool) -> tuple[bool, str]:
    if force:
        return True, "force_window"
    if price is None:
        return False, "await_market_price"
    if ref_price <= 0:
        return True, "no_price_gate"
    if side == "BUY":
        limit = ref_price * (1.0 + EXEC_BUY_MAX_PREMIUM_BPS / 10000.0)
        return price <= limit, f"buy_price={price:.4f} limit={limit:.4f}"
    limit = ref_price * (1.0 - EXEC_SELL_MAX_DISCOUNT_BPS / 10000.0)
    return price >= limit, f"sell_price={price:.4f} limit={limit:.4f}"


def _start_tick_execution(today: str, targets: dict[str, dict[str, object]], source: str) -> None:
    global _execution_targets, _execution_date, _last_exec_scan_ts
    planned: dict[str, dict[str, object]] = {}
    all_symbols = sorted(set(_previous_volumes) | set(targets))
    fallback_symbols = 0
    for symbol in all_symbols:
        current_shares = int(_previous_volumes.get(symbol, 0))
        if symbol in _forbidden_clear_pending:
            continue
        row = targets.get(symbol)
        target_shares = int(row.get("target_shares", 0)) if row else 0
        if target_shares < 0:
            continue
        if target_shares == current_shares:
            continue
        target_weight = float(row.get("target_weight", 0.0)) if row else 0.0
        ref_close = float(row.get("price_ref_close", 0.0)) if row else 0.0
        if _tick_subscribed_symbols and symbol not in _tick_subscribed_symbols:
            side = "BUY" if target_shares > current_shares else "SELL"
            _record_execution_event(
                today,
                f"{source}_unsubscribed_fallback",
                symbol,
                side,
                None,
                ref_close,
                current_shares,
                target_shares,
                target_shares,
                "not_tick_subscribed_target_volume",
            )
            _submit_target_volume(
                today,
                f"{source}_unsubscribed_fallback",
                symbol,
                target_weight,
                target_shares,
                current_shares,
            )
            fallback_symbols += 1
            continue
        planned[symbol] = {
            "target_shares": target_shares,
            "target_weight": target_weight,
            "price_ref_close": ref_close,
            "source": source,
        }
    _execution_targets = planned
    _execution_date = today if planned else None
    _last_exec_scan_ts = {}
    print(
        f"[GM-C-EXEC] tick execution plan date={today} symbols={len(planned)} "
        f"fallback_symbols={fallback_symbols} mode={EXECUTION_MODE} "
        f"window={EXEC_START}-{EXEC_END} force={EXEC_FORCE_TIME}",
        flush=True,
    )


def _record_execution_event(
    today: str,
    source: str,
    symbol: str,
    side: str,
    price: float | None,
    ref_price: float,
    from_shares: int,
    to_shares: int,
    target_shares: int,
    reason: str,
) -> None:
    _execution_events.append(
        {
            "trade_date": today,
            "source": source,
            "symbol": symbol,
            "side": side,
            "price": "" if price is None else float(price),
            "ref_price": float(ref_price),
            "from_shares": int(from_shares),
            "to_shares": int(to_shares),
            "target_shares": int(target_shares),
            "reason": reason,
        }
    )


def _process_tick_execution(context, source: str, tick=None) -> None:
    global _execution_targets, _execution_date
    if isinstance(tick, (list, tuple)) and tick:
        for item in tick:
            _process_tick_execution(context, source, item)
        return
    if not _execution_targets:
        return
    today = _today(context)
    now_time = _now_time(context)
    if _execution_date != today:
        print(
            f"[GM-C-EXEC] clear stale execution plan date={_execution_date} today={today}",
            flush=True,
        )
        _execution_targets = {}
        _execution_date = None
        return
    if not _time_in_execution_window(now_time):
        return
    tick_symbol = _tick_symbol(tick) if tick is not None else None
    tick_price = _tick_price(tick) if tick is not None else None
    symbols = [tick_symbol] if tick_symbol in _execution_targets else sorted(_execution_targets)
    now_ts = pd.Timestamp.now()
    force = _in_force_window(now_time)
    completed: list[str] = []
    for symbol in symbols:
        if symbol not in _execution_targets:
            continue
        if symbol in _pending_target_orders:
            continue
        last_ts = _last_exec_scan_ts.get(symbol)
        if last_ts is not None and (now_ts - last_ts).total_seconds() < EXEC_SCAN_INTERVAL_SECONDS:
            continue
        plan = _execution_targets[symbol]
        target_shares = int(plan["target_shares"])
        current_shares = int(_previous_volumes.get(symbol, 0))
        remaining = target_shares - current_shares
        if remaining == 0:
            completed.append(symbol)
            continue
        side = "BUY" if remaining > 0 else "SELL"
        price = tick_price if tick_symbol == symbol else None
        ref_price = float(plan.get("price_ref_close", 0.0) or 0.0)
        allowed, reason = _price_gate(symbol, side, price, ref_price, force)
        _last_exec_scan_ts[symbol] = now_ts
        if not allowed:
            if VERBOSE_ORDERS:
                print(f"[GM-C-EXEC] WAIT {symbol} {side} {reason}", flush=True)
            continue
        child = _round_child_volume(symbol, remaining, force)
        if child <= 0:
            continue
        next_shares = current_shares + child if remaining > 0 else current_shares - child
        if remaining > 0:
            next_shares = min(next_shares, target_shares)
        else:
            next_shares = max(next_shares, target_shares)
        _record_execution_event(
            today,
            source,
            symbol,
            side,
            price,
            ref_price,
            current_shares,
            next_shares,
            target_shares,
            reason,
        )
        _submit_target_volume(
            today,
            f"{source}_tick_exec",
            symbol,
            float(plan.get("target_weight", 0.0)),
            next_shares,
            current_shares,
        )
    for symbol in completed:
        _execution_targets.pop(symbol, None)
    if not _execution_targets:
        print(f"[GM-C-EXEC] tick execution completed date={today}", flush=True)
        _execution_date = None


def _order_target_percent_safe(symbol: str, percent: float) -> None:
    kwargs = {
        "symbol": symbol,
        "percent": float(max(0.0, percent)),
        "order_type": OrderType_Market,  # noqa: F405
        "position_side": PositionSide_Long,  # noqa: F405
    }
    if GM_ACCOUNT_ID:
        kwargs["account"] = GM_ACCOUNT_ID
    try:
        order_target_percent(**kwargs)  # noqa: F405
        return
    except TypeError:
        kwargs.pop("position_side", None)
    except Exception as exc:
        if "account" in kwargs and _is_invalid_account_error(exc):
            if REQUIRE_ACCOUNT_ID:
                raise
            print(
                f"[GM-C-RISK] account parameter rejected for {symbol}; retrying with selected terminal account",
                flush=True,
            )
            kwargs.pop("account", None)
        else:
            raise
    try:
        order_target_percent(**kwargs)  # noqa: F405
        return
    except TypeError as exc:
        if "account" in kwargs and REQUIRE_ACCOUNT_ID:
            raise RuntimeError(
                f"bound PAPER account parameter was rejected for {symbol}; "
                "refusing terminal-default-account fallback"
            ) from exc
        kwargs.pop("account", None)
    order_target_percent(**kwargs)  # noqa: F405


def _order_target_volume_safe(symbol: str, volume: int) -> None:
    kwargs = {
        "symbol": symbol,
        "volume": int(max(0, volume)),
        "order_type": OrderType_Market,  # noqa: F405
        "position_side": PositionSide_Long,  # noqa: F405
    }
    if GM_ACCOUNT_ID:
        kwargs["account"] = GM_ACCOUNT_ID
    try:
        order_target_volume(**kwargs)  # noqa: F405
        return
    except TypeError:
        kwargs.pop("position_side", None)
    except Exception as exc:
        if "account" in kwargs and _is_invalid_account_error(exc):
            if REQUIRE_ACCOUNT_ID:
                raise
            print(
                f"[GM-C-RISK] account parameter rejected for {symbol}; retrying with selected terminal account",
                flush=True,
            )
            kwargs.pop("account", None)
        else:
            raise
    try:
        order_target_volume(**kwargs)  # noqa: F405
        return
    except TypeError as exc:
        if "account" in kwargs and REQUIRE_ACCOUNT_ID:
            raise RuntimeError(
                f"bound PAPER account parameter was rejected for {symbol}; "
                "refusing terminal-default-account fallback"
            ) from exc
        kwargs.pop("account", None)
    order_target_volume(**kwargs)  # noqa: F405


def _record_submission(
    today: str,
    source: str,
    action: str,
    symbol: str,
    percent: float,
    volume: int | None,
) -> None:
    _submission_events.append(
        {
            "trade_date": today,
            "source": source,
            "action": action,
            "symbol": symbol,
            "target_percent": float(percent),
            "target_volume": "" if volume is None else int(volume),
            "order_style": ORDER_STYLE,
        }
    )


def _buy_paused(today: str) -> bool:
    if not _pause_buys_until:
        return False
    return pd.Timestamp(today) <= pd.Timestamp(_pause_buys_until)


def _set_buy_pause(today: str, reason: str) -> None:
    global _pause_buys_until
    until = pd.Timestamp(today) + pd.Timedelta(days=max(0, SELL_REJECT_PAUSE_DAYS))
    _pause_buys_until = until.strftime("%Y-%m-%d")
    print(
        f"[GM-C-RISK] pause buy-increase orders until {_pause_buys_until}: {reason}",
        flush=True,
    )


def _submit_target_volume(
    today: str,
    source: str,
    symbol: str,
    weight: float,
    shares: int,
    prev_shares: int,
) -> bool:
    buy_delta = shares - prev_shares
    if shares == prev_shares:
        if VERBOSE_ORDERS:
            print(f"[GM-C] SKIP {symbol} unchanged at {shares} shares", flush=True)
        return False
    if buy_delta > 0 and buy_delta < _min_buy_volume(symbol):
        if VERBOSE_ORDERS:
            print(
                f"[GM-C] SKIP {symbol} buy_delta={buy_delta} below min "
                f"{_min_buy_volume(symbol)}; keep {prev_shares} shares",
                flush=True,
            )
        return False
    pending = _pending_target_orders.get(symbol) if RUN_MODE != "BACKTEST" else None
    if pending is not None:
        if VERBOSE_ORDERS:
            print(
                f"[GM-C-RISK] SKIP {symbol} target={shares}; "
                f"pending_target={pending.get('target_shares')}",
                flush=True,
            )
        return False
    _record_submission(today, source, "target", symbol, weight, shares)
    if RUN_MODE != "BACKTEST":
        _pending_target_orders[symbol] = {
            "target_shares": int(shares),
            "previous_shares": int(prev_shares),
            "source": source,
            "submitted_at": pd.Timestamp.now().isoformat(),
        }
    try:
        _order_target_volume_safe(symbol, shares)
    except Exception:
        _pending_target_orders.pop(symbol, None)
        raise
    if RUN_MODE == "BACKTEST":
        _previous_volumes[symbol] = shares
    if VERBOSE_ORDERS:
        print(
            f"[GM-C] TARGET {symbol} -> {shares} shares ({weight:.4%}); awaiting fill",
            flush=True,
        )
    return True


def _submit_pending_buys(today: str, source: str) -> bool:
    global _pending_buy_targets, _pending_buy_source_date
    if not _pending_buy_targets:
        return False
    if _buy_paused(today):
        print(
            f"[GM-C-RISK] skip pending buy-increase orders on {today}; paused until {_pause_buys_until}",
            flush=True,
        )
        return True
    submitted = 0
    for symbol in sorted(_pending_buy_targets):
        row = _pending_buy_targets[symbol]
        weight = float(row["target_weight"])
        shares = int(row.get("target_shares", -1))
        if ORDER_STYLE == "VOLUME" and shares >= 0:
            prev_shares = int(_previous_volumes.get(symbol, 0))
            if shares <= prev_shares:
                continue
            submitted += int(
                _submit_target_volume(today, source, symbol, weight, shares, prev_shares)
            )
        else:
            _record_submission(today, source, "pending_target", symbol, weight, None)
            _order_target_percent_safe(symbol, weight)
            submitted += 1
    print(
        f"[GM-C-RISK] submitted pending buy-increase orders={submitted} "
        f"from target_date={_pending_buy_source_date}",
        flush=True,
    )
    _pending_buy_targets = {}
    _pending_buy_source_date = None
    return True


def _rebalance(context, source: str) -> None:
    global _previous_symbols, _previous_volumes, _last_rebalance_date
    global _pending_buy_targets, _pending_buy_source_date
    today = _today(context)
    _refresh_dynamic_market_risk(context, f"{source}_daily")
    _clear_forbidden_positions(context, f"{source}_forbidden_clear")
    if today not in _all_target_dates:
        if _submit_pending_buys(today, f"{source}_pending_buy"):
            _last_rebalance_date = today
        return
    if _last_rebalance_date == today:
        return
    if _pending_buy_targets:
        print(f"[GM-C-RISK] supersede stale pending buys with target date {today}", flush=True)
        _pending_buy_targets = {}
        _pending_buy_source_date = None
    targets = _targets_by_date[today]
    new_symbols = set(targets)
    to_clear = sorted(_previous_symbols - new_symbols)
    to_set = sorted(new_symbols)
    has_sell_delta = any(
        ORDER_STYLE == "VOLUME"
        and int(row.get("target_shares", -1)) >= 0
        and int(row.get("target_shares", -1)) < int(_previous_volumes.get(symbol, 0))
        for symbol, row in targets.items()
    )

    print(
        f"[GM-C] {source} rebalance {today}: "
        f"targets={len(to_set)} clear={len(to_clear)} "
        f"weight_sum={sum(row['target_weight'] for row in targets.values()):.2%}",
        flush=True,
    )
    _rebalance_events.append(
        {
            "trade_date": today,
            "source": source,
            "target_count": len(to_set),
            "clear_count": len(to_clear),
            "weight_sum": float(sum(row["target_weight"] for row in targets.values())),
            "order_style": ORDER_STYLE,
            "rebalance_phase": REBALANCE_PHASE,
            "buy_paused": _buy_paused(today),
        }
    )
    if RUN_MODE != "BACKTEST" and not LIVE_ALLOWED:
        print(
            "[GM-C] LIVE blocked. Set GM_C_ALLOW_LIVE="
            f"{LIVE_UNLOCK} only after approval.",
            flush=True,
        )
        _previous_symbols = new_symbols
        _last_rebalance_date = today
        return

    if EXECUTION_MODE == "TICK_EXEC" and RUN_MODE != "BACKTEST" and _tick_execution_available:
        _start_tick_execution(today, targets, source)
        _last_rebalance_date = today
        _process_tick_execution(context, source)
        return

    for symbol in to_clear:
        if ORDER_STYLE == "VOLUME" and _previous_volumes.get(symbol, 0) <= 0:
            continue
        if symbol in _forbidden_clear_pending:
            continue
        if ORDER_STYLE == "VOLUME":
            _submit_target_volume(
                today,
                source,
                symbol,
                0.0,
                0,
                int(_previous_volumes.get(symbol, 0)),
            )
        else:
            _record_submission(today, source, "clear", symbol, 0.0, 0)
            _order_target_percent_safe(symbol, 0.0)
        if VERBOSE_ORDERS:
            print(f"[GM-C] CLEAR {symbol} -> 0.00%", flush=True)
    defer_buy_increases = (
        REBALANCE_PHASE == "SELL_FIRST"
        and ORDER_STYLE == "VOLUME"
        and (bool(to_clear) or has_sell_delta)
    )
    if defer_buy_increases:
        print(
            f"[GM-C-RISK] SELL_FIRST active: deferring buy-increase orders "
            f"after clears={len(to_clear)} has_sell_delta={has_sell_delta}",
            flush=True,
        )
    for symbol in to_set:
        row = targets[symbol]
        weight = float(row["target_weight"])
        shares = int(row.get("target_shares", -1))
        if ORDER_STYLE == "VOLUME" and shares >= 0:
            prev_shares = int(_previous_volumes.get(symbol, 0))
            buy_delta = shares - prev_shares
            if buy_delta > 0 and (_buy_paused(today) or defer_buy_increases):
                _pending_buy_targets[symbol] = row
                _pending_buy_source_date = today
                continue
            _submit_target_volume(today, source, symbol, weight, shares, prev_shares)
        else:
            if _buy_paused(today):
                _pending_buy_targets[symbol] = row
                _pending_buy_source_date = today
                continue
            _record_submission(today, source, "target", symbol, weight, None)
            _order_target_percent_safe(symbol, weight)
            if VERBOSE_ORDERS:
                print(f"[GM-C] TARGET {symbol} -> {weight:.4%}", flush=True)
    if _pending_buy_targets:
        print(
            f"[GM-C-RISK] pending buy-increase symbols={len(_pending_buy_targets)} "
            f"target_date={_pending_buy_source_date}",
            flush=True,
        )

    if RUN_MODE == "BACKTEST":
        _previous_symbols = new_symbols
        if ORDER_STYLE == "VOLUME":
            _previous_volumes = {
                symbol: volume
                for symbol, volume in _previous_volumes.items()
                if volume > 0 and symbol in new_symbols
            }
    _last_rebalance_date = today


def init(context):
    print("[GM-C] init entered", flush=True)
    if GM_TOKEN:
        set_token(GM_TOKEN)  # noqa: F405
    else:
        print("[GM-C] GM_TOKEN not set; assuming GmQuant platform-managed auth.", flush=True)
    if GM_ACCOUNT_ID:
        try:
            set_account_id(GM_ACCOUNT_ID)  # noqa: F405
            print(f"[GM-C-RISK] GM_ACCOUNT_ID set account={GM_ACCOUNT_ID}", flush=True)
        except Exception as exc:
            print(f"[GM-C-RISK] set_account_id failed account={GM_ACCOUNT_ID}: {exc}", flush=True)
            if REQUIRE_ACCOUNT_ID and RUN_MODE != "BACKTEST":
                raise RuntimeError(
                    "[GM-C] LIVE blocked. The required PAPER account could not be selected."
                ) from exc
    _load_targets(TARGETS_PATH)
    _validate_live_launch()
    _start_activation_audit(context)
    _sync_existing_positions(context)
    _refresh_dynamic_market_risk(context, "init_dynamic_risk", force=True)
    _clear_forbidden_positions(context, "init_forbidden_clear")
    symbols = _ordered_subscription_symbols()
    _subscribe_targets(symbols)
    schedule(schedule_func=on_rebalance_schedule, date_rule="1d", time_rule="09:31:00")  # noqa: F405
    if RUN_MODE != "BACKTEST":
        schedule(schedule_func=on_audit_flush_schedule, date_rule="1d", time_rule="15:05:00")  # noqa: F405
        _mark_activation_ready(context)
    print(
        f"[GM-C] targets={TARGETS_PATH} dates={len(_targets_by_date)} "
        f"symbols={len(symbols)} mode={RUN_MODE} order_style={ORDER_STYLE} "
        f"execution_mode={EXECUTION_MODE} tick_subscribed={len(_tick_subscribed_symbols)} "
        f"max_tick_subscriptions={MAX_TICK_SUBSCRIPTIONS}",
        flush=True,
    )
    print(f"[GM-C] target dates={sorted(_all_target_dates)}", flush=True)
    print(
        "[GM-C] frequency=daily targets; rebalance schedule=09:31; "
        f"tick_exec_window={EXEC_START}-{EXEC_END} force={EXEC_FORCE_TIME}",
        flush=True,
    )
    _rebalance_on_init_if_needed(context)
    if RUN_MODE != "BACKTEST":
        _write_audit()


def on_rebalance_schedule(context):
    try:
        _rebalance(context, "schedule")
    except Exception as e:
        print(f"[GM-C] schedule error: {e}", flush=True)
        traceback.print_exc()
    finally:
        if RUN_MODE != "BACKTEST":
            _write_audit()


def on_audit_flush_schedule(context):
    _finalize_activation(context)
    _write_audit()


def on_bar(context, bars):
    # 兜底：若 schedule 未触发，日 bar 回调也会尝试调仓；若已有执行计划则推进计划。
    try:
        _rebalance(context, "bar")
        _process_tick_execution(context, "bar")
    except Exception as e:
        print(f"[GM-C] bar error: {e}", flush=True)
        traceback.print_exc()


def on_tick(context, tick):
    try:
        _process_tick_execution(context, "tick", tick)
    except Exception as e:
        print(f"[GM-C] tick execution error: {e}", flush=True)
        traceback.print_exc()


def on_order_status(context, order):
    try:
        global _previous_symbols, _previous_volumes, _execution_date
        status = _order_field(order, "status")
        symbol = _order_field(order, "symbol")
        side = _order_field(order, "side")
        filled = _order_field(order, "filled_volume")
        avg_px = _order_field(order, "filled_vwap")
        event = {
            "event_date": _today(context),
            "symbol": symbol,
            "side": side,
            "status": status,
            "status_name": STATUS_NAMES.get(status, str(status)),
            "volume": _order_field(order, "volume"),
            "value": _order_field(order, "value"),
            "percent": _order_field(order, "percent"),
            "target_volume": _order_field(order, "target_volume"),
            "target_value": _order_field(order, "target_value"),
            "target_percent": _order_field(order, "target_percent"),
            "filled_volume": filled,
            "filled_vwap": avg_px,
            "filled_amount": _order_field(order, "filled_amount"),
            "filled_commission": _order_field(order, "filled_commission"),
            "ord_rej_reason": _order_field(order, "ord_rej_reason"),
            "ord_rej_reason_detail": _order_field(order, "ord_rej_reason_detail"),
            "order_id": _order_field(order, "order_id"),
            "cl_ord_id": _order_field(order, "cl_ord_id"),
            "created_at": _order_field(order, "created_at"),
            "updated_at": _order_field(order, "updated_at"),
        }
        _order_events.append(event)
        if symbol and status in TERMINAL_STATUS:
            _forbidden_clear_pending.discard(str(symbol))
            _pending_target_orders.pop(str(symbol), None)
        if status == 3 and symbol:
            target_volume = event["target_volume"]
            try:
                target_shares = max(0, int(float(target_volume)))
            except (TypeError, ValueError):
                target_shares = None
            if target_shares is not None:
                if target_shares > 0:
                    _previous_volumes[str(symbol)] = target_shares
                    _previous_symbols.add(str(symbol))
                else:
                    _previous_volumes.pop(str(symbol), None)
                    _previous_symbols.discard(str(symbol))
                plan = _execution_targets.get(str(symbol))
                if plan is not None and target_shares == int(plan.get("target_shares", -1)):
                    _execution_targets.pop(str(symbol), None)
                    if not _execution_targets:
                        _execution_date = None
            else:
                _sync_existing_positions(context)
        if status in TERMINAL_STATUS and status != 3:
            _sync_existing_positions(context)
            _execution_targets.pop(str(symbol), None)
            if not _execution_targets:
                _execution_date = None
        if (
            PAUSE_BUYS_ON_SELL_REJECT
            and status == 8
            and str(side) in {"2", "SELL", "OrderSide_Sell"}
        ):
            _set_buy_pause(
                _today(context),
                f"sell order rejected symbol={symbol} reason={event['ord_rej_reason_detail']}",
            )
        if VERBOSE_ORDERS:
            print(
                f"[GM-C-ORDER] {symbol} side={side} status={status}"
                f"({STATUS_NAMES.get(status, status)}) filled={filled} avg_px={avg_px}",
                flush=True,
            )
        if RUN_MODE != "BACKTEST":
            _write_audit()
    except Exception as exc:
        print(f"[GM-C-ERROR] order status handling failed: {exc}", flush=True)
        traceback.print_exc()


def _atomic_write_frame(frame: pd.DataFrame, path: Path) -> None:
    pending = path.with_name(path.name + ".pending")
    frame.to_csv(pending, index=False, encoding="utf-8-sig")
    os.replace(pending, path)


def _write_audit(indicator=None) -> None:
    try:
        out_dir = AUDIT_DIR / AUDIT_RUN_ID
        out_dir.mkdir(parents=True, exist_ok=True)
        submissions = pd.DataFrame(_submission_events)
        orders = pd.DataFrame(_order_events)
        rebalances = pd.DataFrame(_rebalance_events)
        executions = pd.DataFrame(_execution_events)
        if not submissions.empty:
            _atomic_write_frame(submissions, _audit_path("submissions.csv"))
        if not orders.empty:
            _atomic_write_frame(orders, _audit_path("order_status.csv"))
        if not rebalances.empty:
            _atomic_write_frame(rebalances, _audit_path("rebalances.csv"))
        if not executions.empty:
            _atomic_write_frame(executions, _audit_path("execution_events.csv"))

        summary = {
            "target_file": str(TARGETS_PATH),
            "target_sha256": (
                _activation_identity.get("target_sha256")
                if _activation_identity is not None
                else None
            ),
            "forbidden_symbols_file": str(FORBIDDEN_PATH),
            "account_id": GM_ACCOUNT_ID,
            "run_mode": RUN_MODE,
            "trading_env": TRADING_ENV,
            "strategy_id": GM_STRATEGY_ID,
            "audit_run_id": AUDIT_RUN_ID,
            "activation_registry": str(ACTIVATION_REGISTRY),
            "activation_id": (
                _activation_identity.get("activation_id")
                if _activation_identity is not None
                else None
            ),
            "activation_ready_recorded": _activation_ready_recorded,
            "activation_finalized": _activation_finalized,
            "dynamic_st_check": DYNAMIC_ST_CHECK,
            "dynamic_st_fail_closed": DYNAMIC_ST_FAIL_CLOSED,
            "dynamic_risk_check_date": _dynamic_risk_check_date,
            "dynamic_forbidden_events": int(
                sum(event.get("status") == "forbidden" for event in _dynamic_risk_events)
            ),
            "forbidden_clear_pending": sorted(_forbidden_clear_pending),
            "target_dates": sorted(_all_target_dates),
            "signal_dates_by_target_date": {
                date: list(values) for date, values in sorted(_signal_dates_by_target_date.items())
            },
            "require_signal_date": REQUIRE_SIGNAL_DATE,
            "max_live_signal_age_days": MAX_LIVE_SIGNAL_AGE_DAYS,
            "max_live_signal_to_target_days": MAX_LIVE_SIGNAL_TO_TARGET_DAYS,
            "frequency": "daily_target_rebalance_at_09:31_with_optional_tick_execution",
            "execution_mode": EXECUTION_MODE,
            "exec_window": f"{EXEC_START}-{EXEC_END}",
            "exec_force_time": EXEC_FORCE_TIME,
            "tick_execution_available": _tick_execution_available,
            "max_tick_subscriptions": MAX_TICK_SUBSCRIPTIONS,
            "tick_subscribed_symbols": sorted(_tick_subscribed_symbols),
            "tick_subscribed_symbol_count": int(len(_tick_subscribed_symbols)),
            "exec_scan_interval_seconds": EXEC_SCAN_INTERVAL_SECONDS,
            "exec_max_child_fraction": EXEC_MAX_CHILD_FRACTION,
            "exec_buy_max_premium_bps": EXEC_BUY_MAX_PREMIUM_BPS,
            "exec_sell_max_discount_bps": EXEC_SELL_MAX_DISCOUNT_BPS,
            "rebalance_phase": REBALANCE_PHASE,
            "pause_buys_on_sell_reject": PAUSE_BUYS_ON_SELL_REJECT,
            "sync_existing_positions": SYNC_EXISTING_POSITIONS,
            "position_sync_succeeded": _position_sync_succeeded,
            "position_sync_source": _position_sync_source,
            "sell_reject_pause_days": SELL_REJECT_PAUSE_DAYS,
            "pending_buy_symbols_at_finish": int(len(_pending_buy_targets)),
            "pending_target_order_symbols_at_finish": sorted(_pending_target_orders),
            "pending_execution_symbols_at_finish": int(len(_execution_targets)),
            "pause_buys_until": _pause_buys_until,
            "submitted_orders": int(len(submissions)),
            "order_status_events": int(len(orders)),
            "rebalance_events": int(len(rebalances)),
            "execution_events": int(len(executions)),
            "indicator": dict(indicator or {}),
        }
        if not orders.empty:
            terminal = orders[orders["status"].isin(TERMINAL_STATUS)].copy()
            zero_fills = terminal[pd.to_numeric(terminal["filled_volume"], errors="coerce").fillna(0) <= 0]
            rejected = orders[orders["status"] == 8].copy()
            summary.update(
                {
                    "status_counts": orders["status_name"].value_counts(dropna=False).to_dict(),
                    "terminal_zero_fill_events": int(len(zero_fills)),
                    "rejected_events": int(len(rejected)),
                    "rejected_by_symbol_prefix": rejected["symbol"].astype(str).str.slice(0, 9).value_counts().to_dict(),
                    "zero_fill_by_exchange": zero_fills["symbol"].astype(str).str.slice(0, 4).value_counts().to_dict(),
                    "zero_fill_sample": zero_fills.head(20).to_dict(orient="records"),
                    "reject_detail_counts": rejected["ord_rej_reason_detail"].fillna("").value_counts().head(20).to_dict(),
                }
            )
        summary_path = _audit_path("summary.json")
        pending_summary = summary_path.with_name(summary_path.name + ".pending")
        pd.Series(summary, dtype="object").to_json(
            pending_summary,
            force_ascii=False,
            indent=2,
        )
        os.replace(pending_summary, summary_path)
        print(f"[GM-C-AUDIT] wrote audit files to {out_dir}", flush=True)
    except Exception as e:
        print(f"[GM-C-AUDIT] write failed: {e}", flush=True)
        traceback.print_exc()


def on_backtest_finished(context, indicator):
    print("[GM-C] backtest finished", flush=True)
    try:
        for key in ("pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown"):
            print(f"  {key}={indicator.get(key)}", flush=True)
    except Exception as e:
        print(f"  indicator dump failed: {e}", flush=True)
    _write_audit(indicator)


def on_error(context, code, info):
    print(f"[GM-C-ERROR] code={code} info={info}", flush=True)
    if RUN_MODE != "BACKTEST":
        try:
            _record_activation_error(context, code, info)
        except Exception as exc:
            print(
                f"[GM-C-ACTIVATION] error record failed: {exc}",
                flush=True,
            )
        _write_audit()


if __name__ == "__main__":
    print("[GM-C] __main__ entered", flush=True)
    if not GM_TOKEN:
        raise RuntimeError("GM_TOKEN environment variable is required")
    set_token(GM_TOKEN)  # noqa: F405
    is_backtest = RUN_MODE == "BACKTEST"
    kwargs = {
        "strategy_id": GM_STRATEGY_ID,
        "filename": Path(__file__).name,
        "token": GM_TOKEN,
        "mode": MODE_BACKTEST if is_backtest else MODE_LIVE,  # noqa: F405
    }
    if is_backtest:
        kwargs.update(
            {
                "backtest_start_time": BACKTEST_START,
                "backtest_end_time": BACKTEST_END,
                "backtest_adjust": ADJUST_PREV,  # noqa: F405
                "backtest_initial_cash": INITIAL_CASH,
                "backtest_commission_ratio": BACKTEST_COMMISSION,
                "backtest_slippage_ratio": BACKTEST_SLIPPAGE,
                "backtest_match_mode": 0,
            }
        )
    log_kwargs = dict(kwargs)
    if log_kwargs.get("token"):
        log_kwargs["token"] = "***MASKED***"
    print(f"[GM-C] run kwargs={log_kwargs}", flush=True)
    run(**kwargs)  # noqa: F405
