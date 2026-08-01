from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd


SOURCE = "rqdata"
DEFAULT_HOST = "rqdatad-pro.ricequant.com"
DEFAULT_PORT = 16011


def log(message: str) -> None:
    print(f"[RQData bridge] {message}", flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pending.replace(path)


def configure_license_environment() -> None:
    raw = os.environ.pop("RQSDK_LICENSE_INPUT", "").strip()
    if not raw:
        return
    from rqsdk.license_helper import format_rqdatac_uri

    os.environ["RQDATAC_CONF"] = format_rqdatac_uri(raw)


def initialize_rqdata():
    configure_license_environment()
    import rqdatac

    rqdatac.init()
    return rqdatac


def package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    result: dict[str, str | None] = {}
    for package in ("rqsdk", "rqdatac"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def tcp_probe(host: str, port: int, timeout: float = 5.0) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return {
            "passed": True,
            "host": host,
            "port": int(port),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "error": None,
        }
    except Exception as exc:
        return {
            "passed": False,
            "host": host,
            "port": int(port),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def retry_call(
    label: str,
    call: Callable[[], Any],
    *,
    retries: int,
    backoff_seconds: float,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return call()
        except Exception as exc:
            last_error = exc
            log(f"{label} attempt={attempt}/{retries} failed: {type(exc).__name__}: {exc}")
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
    assert last_error is not None
    raise last_error


def rq_to_local(order_book_id: object) -> str:
    text = str(order_book_id).strip().upper()
    if text.endswith(".XSHG"):
        return "sh" + text.split(".", 1)[0]
    if text.endswith(".XSHE"):
        return "sz" + text.split(".", 1)[0]
    raise ValueError(f"unsupported RQData stock id: {order_book_id!r}")


def local_to_rq(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if "." in text:
        if text.endswith((".XSHG", ".XSHE")):
            return text
        if text.startswith("SHSE."):
            return text.split(".", 1)[1] + ".XSHG"
        if text.startswith("SZSE."):
            return text.split(".", 1)[1] + ".XSHE"
    code = text[-6:]
    if not code.isdigit():
        raise ValueError(f"unsupported local stock id: {symbol!r}")
    if text.startswith("SH") or code.startswith(("6", "9")):
        return code + ".XSHG"
    return code + ".XSHE"


def normalize_price_frame(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.reset_index()
    if "order_book_id" not in out.columns:
        raise ValueError("RQData price frame is missing order_book_id")
    time_column = "datetime" if "datetime" in out.columns else "date"
    if time_column not in out.columns:
        raise ValueError("RQData price frame is missing date/datetime")
    out = out.rename(
        columns={
            time_column: "date",
            "total_turnover": "amount",
        }
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["instrument"] = out["order_book_id"].map(rq_to_local)
    out = out.dropna(subset=["date"]).copy()
    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "limit_up",
        "limit_down",
        "prev_close",
    ]
    for column in numeric:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        else:
            out[column] = float("nan")
    out["source"] = SOURCE
    out["frequency"] = frequency
    out["adjust_type"] = "pre"
    columns = [
        "date",
        "instrument",
        *numeric,
        "source",
        "frequency",
        "adjust_type",
    ]
    return (
        out[columns]
        .drop_duplicates(["instrument", "date"], keep="last")
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )


def normalize_boolean_panel(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "instrument", field])
    panel = frame.copy()
    panel.index.name = "date"
    panel.columns.name = "order_book_id"
    out = panel.stack(dropna=False).rename(field).reset_index()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["instrument"] = out["order_book_id"].map(rq_to_local)
    out[field] = out[field].astype("boolean")
    return out[["date", "instrument", field]].dropna(subset=["date"])


def atomic_merge_csv(path: Path, incoming: pd.DataFrame) -> dict[str, Any]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    before_rows = 0
    if path.exists():
        existing = pd.read_csv(path, parse_dates=["date"])
        before_rows = int(len(existing))
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming.copy()
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = (
        combined.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    output = combined.copy()
    if output["date"].dt.normalize().equals(output["date"]):
        output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    else:
        output["date"] = output["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    pending = path.with_suffix(path.suffix + ".pending")
    output.to_csv(pending, index=False, encoding="utf-8-sig")
    pending.replace(path)
    return {
        "path": str(path),
        "before_rows": before_rows,
        "incoming_rows": int(len(incoming)),
        "after_rows": int(len(combined)),
        "first": combined["date"].min().isoformat() if len(combined) else None,
        "last": combined["date"].max().isoformat() if len(combined) else None,
    }


def read_symbols(path: Path) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        raw = line.strip().split()[0] if line.strip() else ""
        if not raw:
            continue
        value = local_to_rq(raw)
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError(f"symbol file is empty: {path}")
    return values


def chunks(values: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [values[pos : pos + size] for pos in range(0, len(values), size)]


def command_doctor(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "status": "rqdata_doctor",
        "versions": package_versions(),
        "transport": tcp_probe(args.host, args.port, args.timeout),
        "license_supplied": bool(os.environ.get("RQSDK_LICENSE_INPUT")),
        "init_passed": False,
        "api_checks": {},
        "passed": False,
        "error": None,
    }
    try:
        rqdatac = initialize_rqdata()
        report["init_passed"] = True
        dates = rqdatac.get_trading_dates(args.start_date, args.end_date)
        has_dates = len(dates) > 0
        report["api_checks"]["trading_dates"] = {
            "passed": has_dates,
            "rows": len(dates),
            "first": str(dates[0]) if has_dates else None,
            "last": str(dates[-1]) if has_dates else None,
        }
        daily = normalize_price_frame(
            rqdatac.get_price(
                args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                frequency="1d",
                fields=["open", "high", "low", "close", "volume", "total_turnover"],
                adjust_type="pre",
                skip_suspended=True,
                expect_df=True,
            ),
            "1d",
        )
        report["api_checks"]["daily_price"] = {
            "passed": not daily.empty,
            "rows": int(len(daily)),
        }
        minute_date = str(dates[-1]) if has_dates else args.end_date
        minute = normalize_price_frame(
            rqdatac.get_price(
                args.symbol,
                start_date=minute_date,
                end_date=minute_date,
                frequency="1m",
                fields=["open", "high", "low", "close", "volume", "total_turnover"],
                adjust_type="pre",
                skip_suspended=True,
                expect_df=True,
            ),
            "1m",
        )
        report["api_checks"]["minute_price"] = {
            "passed": not minute.empty,
            "rows": int(len(minute)),
            "date": minute_date,
        }
        report["passed"] = bool(
            report["transport"]["passed"]
            and report["init_passed"]
            and all(item["passed"] for item in report["api_checks"].values())
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    write_json(args.output, report)
    log(
        f"doctor passed={report['passed']} transport={report['transport']['passed']} "
        f"init={report['init_passed']} output={args.output.resolve()}"
    )
    return 0 if report["passed"] else 2


def command_instruments(args: argparse.Namespace) -> int:
    rqdatac = initialize_rqdata()
    frame = retry_call(
        "all_instruments",
        lambda: rqdatac.all_instruments(type="CS", date=args.date, market="cn"),
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    if frame is None or frame.empty:
        raise RuntimeError("RQData returned no common-stock instruments")
    frame = frame.copy()
    frame["instrument"] = frame["order_book_id"].map(rq_to_local)
    frame["source"] = SOURCE
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    log(f"instruments rows={len(frame)} output={args.output.resolve()}")
    return 0


def command_fetch(args: argparse.Namespace, frequency: str) -> int:
    rqdatac = initialize_rqdata()
    symbols = read_symbols(args.symbols_file)
    fields = [item.strip() for item in args.fields.split(",") if item.strip()]
    batches = chunks(symbols, args.batch_size)
    files: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for pos, batch in enumerate(batches, start=1):
        label = f"get_price {frequency} batch={pos}/{len(batches)} symbols={len(batch)}"
        try:
            raw = retry_call(
                label,
                lambda batch=batch: rqdatac.get_price(
                    batch,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    frequency=frequency,
                    fields=fields,
                    adjust_type="pre",
                    skip_suspended=True,
                    expect_df=True,
                    market="cn",
                ),
                retries=args.retries,
                backoff_seconds=args.backoff,
            )
            normalized = normalize_price_frame(raw, frequency)
            for instrument, part in normalized.groupby("instrument", sort=True):
                code = str(instrument)[2:]
                if frequency == "1d":
                    path = args.output_dir / f"{code}_daily_qfq.csv"
                else:
                    suffix = (
                        f"{pd.Timestamp(args.start_date):%Y%m%d}_"
                        f"{pd.Timestamp(args.end_date):%Y%m%d}"
                    )
                    path = args.output_dir / f"{instrument}_{SOURCE}_{suffix}.csv"
                files.append(atomic_merge_csv(path, part.drop(columns=["instrument"])))
            missing = sorted(set(batch) - set(normalized.get("instrument", pd.Series(dtype=str)).map(local_to_rq)))
            for symbol in missing:
                errors.append({"symbol": symbol, "error": "empty_response"})
            log(f"{label} rows={len(normalized)} files={len(files)} errors={len(errors)}")
        except Exception as exc:
            for symbol in batch:
                errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            log(f"{label} failed for batch; continuing with manifest evidence")
    manifest = {
        "status": "rqdata_fetch_complete",
        "passed": not errors,
        "frequency": frequency,
        "source": SOURCE,
        "adjust_type": "pre",
        "skip_suspended": True,
        "start_date": pd.Timestamp(args.start_date).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(args.end_date).strftime("%Y-%m-%d"),
        "requested_symbols": len(symbols),
        "written_files": len(files),
        "errors": errors,
        "files": files,
    }
    write_json(args.manifest, manifest)
    log(
        f"fetch complete frequency={frequency} files={len(files)} "
        f"errors={len(errors)} manifest={args.manifest.resolve()}"
    )
    return 0 if not errors else 3


def command_market_state(args: argparse.Namespace) -> int:
    rqdatac = initialize_rqdata()
    symbols = read_symbols(args.symbols_file)
    batches = chunks(symbols, args.batch_size)
    parts: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    for pos, batch in enumerate(batches, start=1):
        label = f"market-state batch={pos}/{len(batches)} symbols={len(batch)}"
        try:
            st = retry_call(
                f"{label} ST",
                lambda batch=batch: rqdatac.is_st_stock(
                    batch, args.start_date, args.end_date, market="cn"
                ),
                retries=args.retries,
                backoff_seconds=args.backoff,
            )
            suspended = retry_call(
                f"{label} suspension",
                lambda batch=batch: rqdatac.is_suspended(
                    batch, args.start_date, args.end_date, market="cn"
                ),
                retries=args.retries,
                backoff_seconds=args.backoff,
            )
            normalized = normalize_boolean_panel(st, "is_st").merge(
                normalize_boolean_panel(suspended, "is_suspended"),
                on=["date", "instrument"],
                how="outer",
            )
            parts.append(normalized)
            log(f"{label} rows={len(normalized)}")
        except Exception as exc:
            for symbol in batch:
                errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            log(f"{label} failed; continuing with manifest evidence")
    incoming = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if args.output.is_file():
        existing = pd.read_csv(args.output, parse_dates=["date"])
        incoming = pd.concat([existing, incoming], ignore_index=True)
    if not incoming.empty:
        incoming = (
            incoming.drop_duplicates(["date", "instrument"], keep="last")
            .sort_values(["date", "instrument"])
            .reset_index(drop=True)
        )
        output = incoming.copy()
        output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pending = args.output.with_suffix(args.output.suffix + ".pending")
        output.to_csv(pending, index=False, encoding="utf-8-sig")
        pending.replace(args.output)
    manifest = {
        "status": "rqdata_market_state_complete",
        "passed": not errors and not incoming.empty,
        "start_date": pd.Timestamp(args.start_date).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(args.end_date).strftime("%Y-%m-%d"),
        "requested_symbols": len(symbols),
        "rows": len(incoming),
        "output": str(args.output.resolve()),
        "errors": errors,
    }
    write_json(args.manifest, manifest)
    log(f"market-state complete rows={len(incoming)} errors={len(errors)}")
    return 0 if manifest["passed"] else 3


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Isolated RQData API bridge")
    sub = root.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--output", type=Path, required=True)
    doctor.add_argument("--host", default=DEFAULT_HOST)
    doctor.add_argument("--port", type=int, default=DEFAULT_PORT)
    doctor.add_argument("--timeout", type=float, default=5.0)
    doctor.add_argument("--symbol", default="000001.XSHE")
    doctor.add_argument("--start-date", default="2026-06-01")
    doctor.add_argument("--end-date", default="2026-06-05")

    instruments = sub.add_parser("instruments")
    instruments.add_argument("--date", required=True)
    instruments.add_argument("--output", type=Path, required=True)
    instruments.add_argument("--retries", type=int, default=3)
    instruments.add_argument("--backoff", type=float, default=2.0)

    for name in ("fetch-daily", "fetch-minute"):
        command = sub.add_parser(name)
        command.add_argument("--symbols-file", type=Path, required=True)
        command.add_argument("--start-date", required=True)
        command.add_argument("--end-date", required=True)
        command.add_argument("--fields", required=True)
        command.add_argument("--batch-size", type=int, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--retries", type=int, default=3)
        command.add_argument("--backoff", type=float, default=2.0)
    state = sub.add_parser("fetch-market-state")
    state.add_argument("--symbols-file", type=Path, required=True)
    state.add_argument("--start-date", required=True)
    state.add_argument("--end-date", required=True)
    state.add_argument("--batch-size", type=int, required=True)
    state.add_argument("--output", type=Path, required=True)
    state.add_argument("--manifest", type=Path, required=True)
    state.add_argument("--retries", type=int, default=3)
    state.add_argument("--backoff", type=float, default=2.0)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "doctor":
        return command_doctor(args)
    if args.command == "instruments":
        return command_instruments(args)
    if args.command == "fetch-daily":
        return command_fetch(args, "1d")
    if args.command == "fetch-minute":
        return command_fetch(args, "1m")
    if args.command == "fetch-market-state":
        return command_market_state(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
