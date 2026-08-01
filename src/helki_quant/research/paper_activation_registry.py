from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ACTIVATION_SCHEMA_VERSION = 1
ACTIVATION_REGISTRY_FILENAME = "PAPER_ACTIVATION_REGISTRY.jsonl"
EVENT_STARTED = "PAPER_RUN_STARTED"
EVENT_READY = "PAPER_RUN_READY"
EVENT_FINALIZED = "PAPER_SESSION_FINALIZED"
EVENT_ERROR = "PAPER_RUN_ERROR"
SESSION_METRICS_SCHEMA_VERSION = 1
MARKET_RESTRICTION_TERMS = ("停牌", "涨停", "跌停", "价格超过", "价格低于")
ACTIVATION_LOCK_TIMEOUT_SECONDS = 5.0
ACTIVATION_LOCK_STALE_SECONDS = 30.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    values = dict(payload)
    values.pop("record_hash", None)
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _order_side_name(value: object) -> str:
    side = _integer(value, -1)
    if side == 1:
        return "BUY"
    if side == 2:
        return "SELL"
    if side == 0:
        return "PENDING"
    return f"UNKNOWN_{side}"


def _order_event_key(event: Mapping[str, Any], index: int) -> tuple[object, ...]:
    for field in ("cl_ord_id", "order_id"):
        value = event.get(field)
        if value not in (None, "", 0, 0.0):
            return field, str(value)
    created_at = event.get("created_at")
    if created_at not in (None, ""):
        return (
            "fallback",
            str(event.get("symbol") or ""),
            str(event.get("side") or ""),
            str(event.get("target_volume") or ""),
            str(created_at),
        )
    return "event", index


def summarize_paper_session(
    *,
    order_events: list[Mapping[str, Any]],
    target_volumes: Mapping[str, int],
    actual_volumes: Mapping[str, int],
    deferred_buy_symbols: set[str] | list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Summarize one PAPER session using final order and account state.

    Market-restriction sell rejections and SELL_FIRST deferred buys explain
    temporary target differences. Every other difference remains unexplained
    and must fail the live-readiness gate.
    """

    latest: dict[tuple[object, ...], dict[str, Any]] = {}
    for index, raw in enumerate(order_events):
        event = dict(raw)
        latest[_order_event_key(event, index)] = event
    terminal = [
        event
        for event in latest.values()
        if _integer(event.get("status"), -1) in {3, 4, 5, 7, 8, 12}
    ]
    status_counts: dict[str, int] = {}
    rejected: list[dict[str, Any]] = []
    filled: list[dict[str, Any]] = []
    for event in terminal:
        status = _integer(event.get("status"), -1)
        name = str(event.get("status_name") or status)
        status_counts[name] = status_counts.get(name, 0) + 1
        if status == 8:
            rejected.append(event)
        elif status == 3:
            filled.append(event)

    market_restriction_rejected: list[dict[str, Any]] = []
    unexpected_rejected: list[dict[str, Any]] = []
    expected_rejected_sell_symbols: set[str] = set()
    for event in rejected:
        detail = str(event.get("ord_rej_reason_detail") or "")
        if any(term in detail for term in MARKET_RESTRICTION_TERMS):
            market_restriction_rejected.append(event)
            if _order_side_name(event.get("side")) == "SELL":
                expected_rejected_sell_symbols.add(
                    str(event.get("symbol") or "").upper()
                )
        else:
            unexpected_rejected.append(event)

    targets = {
        str(symbol).upper(): max(0, _integer(volume))
        for symbol, volume in target_volumes.items()
    }
    actual = {
        str(symbol).upper(): max(0, _integer(volume))
        for symbol, volume in actual_volumes.items()
    }
    deferred = {str(symbol).upper() for symbol in deferred_buy_symbols}
    all_symbols = sorted(set(targets) | set(actual))
    mismatches: list[dict[str, Any]] = []
    unexplained: list[dict[str, Any]] = []
    unresolved_rejected_sells: list[str] = []
    for symbol in all_symbols:
        target = targets.get(symbol, 0)
        observed = actual.get(symbol, 0)
        difference = observed - target
        if difference == 0:
            continue
        explanation = ""
        if difference > 0 and symbol in expected_rejected_sell_symbols:
            explanation = "market_restriction_sell"
            unresolved_rejected_sells.append(symbol)
        elif difference < 0 and symbol in deferred:
            explanation = "sell_first_deferred_buy"
        row = {
            "symbol": symbol,
            "target_volume": target,
            "actual_volume": observed,
            "difference": difference,
            "absolute_difference": abs(difference),
            "explanation": explanation or "unexplained",
        }
        mismatches.append(row)
        if not explanation:
            unexplained.append(row)

    return {
        "session_metrics_schema_version": SESSION_METRICS_SCHEMA_VERSION,
        "terminal_orders": len(terminal),
        "filled_orders": len(filled),
        "rejected_orders": len(rejected),
        "market_restriction_rejected_orders": len(market_restriction_rejected),
        "unexpected_rejected_orders": len(unexpected_rejected),
        "unexpected_rejected_samples": [
            {
                "symbol": event.get("symbol"),
                "side": _order_side_name(event.get("side")),
                "detail": event.get("ord_rej_reason_detail"),
            }
            for event in unexpected_rejected[:10]
        ],
        "terminal_order_status_counts": status_counts,
        "target_symbols": len(targets),
        "actual_symbols": len(actual),
        "target_volume_total": int(sum(targets.values())),
        "actual_volume_total": int(sum(actual.values())),
        "target_mismatch_symbols": len(mismatches),
        "target_volume_abs_diff": int(
            sum(row["absolute_difference"] for row in mismatches)
        ),
        "unexplained_target_mismatch_symbols": len(unexplained),
        "unexplained_target_volume_abs_diff": int(
            sum(row["absolute_difference"] for row in unexplained)
        ),
        "target_mismatch_sample": mismatches[:20],
        "deferred_buy_symbols": sorted(deferred),
        "unresolved_rejected_sell_symbols": len(unresolved_rejected_sells),
        "unresolved_rejected_sell_symbol_list": sorted(
            unresolved_rejected_sells
        ),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _single_date(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        raise ValueError(f"target missing {column}")
    values = (
        pd.to_datetime(frame[column], errors="coerce")
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .unique()
    )
    if len(values) != 1:
        raise ValueError(
            f"target must contain one {column}: {sorted(values.tolist())}"
        )
    return str(values[0])


def _manifest_hash(
    manifest: dict[str, Any],
    name: str,
) -> str:
    return str(
        manifest.get("target_provenance", {})
        .get("sha256", {})
        .get(name, "")
    ).upper()


def build_activation_identity(
    *,
    package_dir: Path,
    target_path: Path,
    forbidden_path: Path,
    account_id: str,
    run_id: str,
    strategy_id: str,
    run_mode: str,
    trading_env: str,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    target_path = target_path.resolve()
    forbidden_path = forbidden_path.resolve()
    account_id = str(account_id).strip()
    if not account_id:
        raise ValueError("PAPER activation requires an explicit account id")
    if run_mode != "LIVE" or trading_env != "PAPER":
        raise ValueError(
            "activation registry is PAPER LIVE only: "
            f"run_mode={run_mode!r} trading_env={trading_env!r}"
        )
    manifest_path = package_dir / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json"
    manifest = _read_json(manifest_path, "candidate manifest")
    if manifest.get("paper_only") is not True:
        raise ValueError("candidate manifest is not PAPER-only")
    if str(manifest.get("paper_account_id") or "").strip() != account_id:
        raise ValueError("candidate manifest account does not match runtime account")

    frame = pd.read_csv(target_path, dtype={"symbol": str, "instrument": str})
    trade_date = _single_date(frame, "trade_date")
    signal_date = _single_date(frame, "signal_date")
    target_hash = sha256_file(target_path)
    forbidden_hash = sha256_file(forbidden_path)
    expected_target_hash = _manifest_hash(manifest, target_path.name)
    expected_forbidden_hash = _manifest_hash(manifest, forbidden_path.name)
    if target_hash != expected_target_hash:
        raise ValueError("runtime target hash does not match candidate manifest")
    if forbidden_hash != expected_forbidden_hash:
        raise ValueError("runtime forbidden hash does not match candidate manifest")

    provenance = manifest.get("target_provenance", {})
    if provenance.get("trade_date") != trade_date:
        raise ValueError("candidate manifest trade date does not match target")
    if provenance.get("signal_date") != signal_date:
        raise ValueError("candidate manifest signal date does not match target")
    ready_path = package_dir / f"PAPER_READY_{trade_date.replace('-', '')}.json"
    ready = _read_json(ready_path, "PAPER_READY evidence")
    if (
        ready.get("passed") is not True
        or ready.get("paper_orders_allowed") is not True
    ):
        raise ValueError("PAPER_READY evidence does not authorize PAPER orders")
    if str(ready.get("paper_account_id") or "").strip() != account_id:
        raise ValueError("PAPER_READY account does not match runtime account")
    if ready.get("trade_date") != trade_date or ready.get("signal_date") != signal_date:
        raise ValueError("PAPER_READY dates do not match runtime target")

    numeric_shares = pd.to_numeric(
        frame.get("target_shares"),
        errors="coerce",
    )
    if numeric_shares.isna().any() or numeric_shares.lt(0).any():
        raise ValueError("runtime target contains invalid target_shares")
    symbol_source = frame["symbol"] if "symbol" in frame else frame["instrument"]
    identity = {
        "activation_schema_version": ACTIVATION_SCHEMA_VERSION,
        "run_id": str(run_id),
        "strategy_id": str(strategy_id),
        "account_id": account_id,
        "run_mode": run_mode,
        "trading_env": trading_env,
        "package_dir": str(package_dir),
        "target_path": str(target_path),
        "target_sha256": target_hash,
        "forbidden_path": str(forbidden_path),
        "forbidden_sha256": forbidden_hash,
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "paper_ready_path": str(ready_path),
        "paper_ready_sha256": sha256_file(ready_path),
        "signal_date": signal_date,
        "trade_date": trade_date,
        "target_rows": int(len(frame)),
        "target_symbols": int(symbol_source.astype(str).nunique()),
        "target_shares": int(numeric_shares.sum()),
    }
    identity["activation_id"] = _canonical_hash(identity)
    return identity


def read_activation_registry(
    path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    path = path.resolve()
    if not path.is_file():
        return [], []
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    previous_hash = ""
    starts: set[str] = set()
    ready: set[str] = set()
    finalized: set[str] = set()
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number} invalid JSON: {exc}")
            continue
        if record.get("activation_schema_version") != ACTIVATION_SCHEMA_VERSION:
            errors.append(f"line {line_number} schema mismatch")
        if record.get("previous_hash", "") != previous_hash:
            errors.append(f"line {line_number} hash-chain break")
        observed_hash = str(record.get("record_hash") or "").upper()
        if observed_hash != _canonical_hash(record):
            errors.append(f"line {line_number} record hash mismatch")
        previous_hash = observed_hash
        run_id = str(record.get("run_id") or "")
        event = record.get("event")
        if event == EVENT_STARTED:
            if run_id in starts:
                errors.append(f"line {line_number} duplicate start for {run_id}")
            starts.add(run_id)
        elif event == EVENT_READY:
            if run_id not in starts:
                errors.append(f"line {line_number} ready without start for {run_id}")
            if run_id in ready:
                errors.append(f"line {line_number} duplicate ready for {run_id}")
            ready.add(run_id)
        elif event == EVENT_FINALIZED:
            if run_id not in ready:
                errors.append(
                    f"line {line_number} finalization without ready for {run_id}"
                )
            if run_id in finalized:
                errors.append(
                    f"line {line_number} duplicate finalization for {run_id}"
                )
            finalized.add(run_id)
        elif event != EVENT_ERROR:
            errors.append(f"line {line_number} unsupported event: {event!r}")
        records.append(record)
    return records, errors


@contextmanager
def _activation_registry_lock(registry_path: Path):
    registry_path = registry_path.resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    deadline = time.monotonic() + ACTIVATION_LOCK_TIMEOUT_SECONDS
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            try:
                age_seconds = max(
                    0.0,
                    time.time() - lock_path.stat().st_mtime,
                )
            except FileNotFoundError:
                continue
            if age_seconds > ACTIVATION_LOCK_STALE_SECONDS:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for PAPER activation registry lock: "
                    f"{lock_path}"
                )
            time.sleep(0.05)
    try:
        lock_payload = json.dumps(
            {
                "pid": os.getpid(),
                "created_at": pd.Timestamp.now().isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
        os.write(descriptor, lock_payload)
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def append_activation_event(
    registry_path: Path,
    *,
    event: str,
    identity: Mapping[str, Any],
    timestamp: object,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with _activation_registry_lock(registry_path):
        return _append_activation_event_unlocked(
            registry_path,
            event=event,
            identity=identity,
            timestamp=timestamp,
            metrics=metrics,
        )


def _append_activation_event_unlocked(
    registry_path: Path,
    *,
    event: str,
    identity: Mapping[str, Any],
    timestamp: object,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if event not in {EVENT_STARTED, EVENT_READY, EVENT_FINALIZED, EVENT_ERROR}:
        raise ValueError(f"unsupported activation event: {event}")
    registry_path = registry_path.resolve()
    records, errors = read_activation_registry(registry_path)
    if errors:
        raise ValueError("activation registry integrity failed: " + "; ".join(errors))
    run_id = str(identity.get("run_id") or "")
    existing_events = {
        str(record.get("event"))
        for record in records
        if str(record.get("run_id") or "") == run_id
    }
    if event == EVENT_STARTED:
        scope = (
            str(identity.get("account_id") or ""),
            str(identity.get("strategy_id") or ""),
            str(identity.get("trade_date") or ""),
        )
        for record in records:
            if record.get("event") != EVENT_STARTED:
                continue
            record_scope = (
                str(record.get("account_id") or ""),
                str(record.get("strategy_id") or ""),
                str(record.get("trade_date") or ""),
            )
            if record_scope == scope and str(record.get("run_id") or "") != run_id:
                raise ValueError(
                    "another PAPER run already owns account/strategy/trade_date: "
                    f"account={scope[0]} strategy={scope[1]} trade_date={scope[2]}"
                )
    if event in existing_events and event != EVENT_ERROR:
        raise ValueError(f"activation event already recorded: {run_id} {event}")
    if event == EVENT_READY and EVENT_STARTED not in existing_events:
        raise ValueError("PAPER_RUN_READY requires PAPER_RUN_STARTED")
    if event == EVENT_FINALIZED and EVENT_READY not in existing_events:
        raise ValueError("PAPER_SESSION_FINALIZED requires PAPER_RUN_READY")
    previous_hash = (
        str(records[-1].get("record_hash") or "").upper() if records else ""
    )
    record = {
        **dict(identity),
        "event": event,
        "timestamp": pd.Timestamp(timestamp).isoformat(),
        "metrics": dict(metrics or {}),
        "previous_hash": previous_hash,
    }
    record["record_hash"] = _canonical_hash(record)
    lines = [
        json.dumps(item, ensure_ascii=False, sort_keys=True)
        for item in [*records, record]
    ]
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    pending = registry_path.with_suffix(registry_path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, registry_path)
    return record


def resolve_latest_finalized_target(
    registry_path: Path,
    *,
    expected_account_id: str,
    before_trade_date: str,
) -> dict[str, Any]:
    records, errors = read_activation_registry(registry_path)
    if errors:
        raise ValueError("activation registry integrity failed: " + "; ".join(errors))
    limit = pd.Timestamp(before_trade_date).normalize()
    candidates = [
        record
        for record in records
        if record.get("event") == EVENT_FINALIZED
        and str(record.get("account_id") or "") == str(expected_account_id)
        and pd.Timestamp(record.get("trade_date")).normalize() < limit
    ]
    if not candidates:
        raise ValueError(
            "no finalized prior PAPER activation exists for the requested account"
        )
    selected = max(
        candidates,
        key=lambda row: (
            pd.Timestamp(row["trade_date"]),
            pd.Timestamp(row["timestamp"]),
        ),
    )
    target_path = Path(str(selected.get("target_path") or "")).resolve()
    if not target_path.is_file():
        raise FileNotFoundError(
            f"finalized activation target not found: {target_path}"
        )
    if sha256_file(target_path) != str(selected.get("target_sha256") or "").upper():
        raise ValueError("finalized activation target hash mismatch")
    return {
        "registry": str(registry_path.resolve()),
        "run_id": selected["run_id"],
        "activation_id": selected["activation_id"],
        "account_id": selected["account_id"],
        "signal_date": selected["signal_date"],
        "trade_date": selected["trade_date"],
        "target_path": str(target_path),
        "target_sha256": selected["target_sha256"],
        "finalized_at": selected["timestamp"],
    }
