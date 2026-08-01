from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "gm_paper_account_snapshots"

POSITION_FIELDS = (
    "account_id",
    "account_name",
    "symbol",
    "side",
    "volume",
    "volume_today",
    "available",
    "available_today",
    "available_now",
    "order_frozen",
    "price",
    "last_price",
    "vwap",
    "market_value",
    "amount",
    "fpnl",
    "cost",
    "updated_at",
)
CASH_FIELDS = (
    "account_id",
    "account_name",
    "nav",
    "available",
    "balance",
    "market_value",
    "market_value_long",
    "frozen",
    "order_frozen",
    "pnl",
    "fpnl",
    "cum_commission",
    "updated_at",
)


def _field(row: object, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if pd.notna(result) else default


def _integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalize_positions(positions: list[object]) -> pd.DataFrame:
    rows = []
    for position in positions:
        row = {name: _json_value(_field(position, name)) for name in POSITION_FIELDS}
        row["symbol"] = str(row.get("symbol") or "").strip().upper()
        row["side"] = _integer(row.get("side"))
        for name in (
            "volume",
            "volume_today",
            "available",
            "available_today",
            "available_now",
            "order_frozen",
        ):
            row[name] = _integer(row.get(name))
        for name in (
            "price",
            "last_price",
            "vwap",
            "market_value",
            "amount",
            "fpnl",
            "cost",
        ):
            row[name] = _number(row.get(name))
        if row["symbol"] and row["volume"] > 0:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=POSITION_FIELDS)
    return pd.DataFrame(rows, columns=POSITION_FIELDS).sort_values(
        ["symbol", "side"]
    ).reset_index(drop=True)


def normalize_cash(cash: object) -> dict[str, object]:
    result = {name: _json_value(_field(cash, name)) for name in CASH_FIELDS}
    for name in (
        "nav",
        "available",
        "balance",
        "market_value",
        "market_value_long",
        "frozen",
        "order_frozen",
        "pnl",
        "fpnl",
        "cum_commission",
    ):
        result[name] = _number(result.get(name))
    return result


def normalize_stock_universe(
    instruments: list[object],
    *,
    captured_at: object,
) -> pd.DataFrame:
    captured_date = pd.Timestamp(captured_at).normalize()
    rows: list[dict[str, object]] = []
    for instrument in instruments:
        symbol = str(_field(instrument, "symbol", "") or "").strip().upper()
        if "." not in symbol:
            continue
        exchange, code = symbol.split(".", 1)
        if exchange not in {"SHSE", "SZSE"} or len(code) != 6 or not code.isdigit():
            continue
        sec_name = str(
            _field(instrument, "sec_name", "")
            or _field(instrument, "sec_abbr", "")
            or ""
        ).strip()
        delisted = pd.to_datetime(
            _field(instrument, "delisted_date", None),
            errors="coerce",
        )
        is_delisted = (
            pd.notna(delisted)
            and pd.Timestamp(delisted).date() <= captured_date.date()
        )
        suffix = "SH" if exchange == "SHSE" else "SZ"
        rows.append(
            {
                "TS代码": f"{code}.{suffix}",
                "股票代码": code,
                "股票名称": sec_name,
                "上市状态": "退市" if is_delisted else "上市",
                "退市日期": (
                    pd.Timestamp(delisted).strftime("%Y-%m-%d")
                    if is_delisted
                    else ""
                ),
                "所属行业": "",
                "gm_symbol": symbol,
                "is_suspended": _integer(
                    _field(instrument, "is_suspended", 0)
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "TS代码",
                "股票代码",
                "股票名称",
                "上市状态",
                "退市日期",
                "所属行业",
                "gm_symbol",
                "is_suspended",
            ]
        )
    frame = pd.DataFrame(rows)
    return (
        frame.drop_duplicates(subset=["TS代码"], keep="last")
        .sort_values("TS代码")
        .reset_index(drop=True)
    )


def capture_snapshot(
    *,
    positions: list[object],
    cash: object,
    account_id: str,
    captured_at: object,
    output_root: Path,
    snapshot_id: str,
    next_trading_date: object | None = None,
    instruments: list[object] | None = None,
    minimum_stock_universe: int = 1000,
) -> dict[str, object]:
    account_id = str(account_id).strip()
    if not account_id:
        raise ValueError("an explicit PAPER account id is required")
    if not snapshot_id or any(value in snapshot_id for value in ("/", "\\", "..")):
        raise ValueError("snapshot_id must be a simple version name")
    captured_ts = pd.Timestamp(captured_at)
    if pd.isna(captured_ts):
        raise ValueError("captured_at is invalid")
    next_trade_text = ""
    next_trade_gap = None
    if next_trading_date is not None:
        next_trade_ts = pd.Timestamp(next_trading_date)
        if pd.isna(next_trade_ts):
            raise ValueError("next_trading_date is invalid")
        next_trade_ts = next_trade_ts.normalize()
        next_trade_text = next_trade_ts.strftime("%Y-%m-%d")
        next_trade_gap = int((next_trade_ts - captured_ts.normalize()).days)

    frame = normalize_positions(positions)
    cash_row = normalize_cash(cash)
    observed_accounts = {
        str(value).strip()
        for value in frame.get("account_id", pd.Series(dtype=str)).dropna()
        if str(value).strip()
    }
    cash_account = str(cash_row.get("account_id") or "").strip()
    if cash_account:
        observed_accounts.add(cash_account)
    account_mismatches = sorted(value for value in observed_accounts if value != account_id)
    unsupported = frame[(frame["side"] != 1) & (frame["volume"] > 0)].copy()

    run_dir = output_root.resolve() / snapshot_id
    if run_dir.exists():
        raise FileExistsError(f"versioned account snapshot already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    positions_path = run_dir / "ACCOUNT_POSITIONS.csv"
    frame.to_csv(positions_path, index=False, encoding="utf-8-sig")
    stock_universe = None
    stock_universe_path = None
    if instruments is not None:
        stock_universe = normalize_stock_universe(
            instruments,
            captured_at=captured_ts,
        )
        stock_universe_path = run_dir / "GM_STOCK_LIST_REFRESH.csv"
        stock_universe.to_csv(
            stock_universe_path,
            index=False,
            encoding="utf-8-sig",
        )

    nav = _number(cash_row.get("nav"))
    available = _number(cash_row.get("available"))
    checks = [
        {
            "name": "account_id_match",
            "passed": not account_mismatches,
            "observed": sorted(observed_accounts),
            "expected": account_id,
        },
        {
            "name": "long_only_positions",
            "passed": unsupported.empty,
            "observed": int(len(unsupported)),
            "expected": 0,
        },
        {
            "name": "positive_nav",
            "passed": nav > 0,
            "observed": nav,
            "expected": "> 0",
        },
        {
            "name": "nonnegative_available_cash",
            "passed": available >= 0,
            "observed": available,
            "expected": ">= 0",
        },
    ]
    if next_trading_date is not None:
        checks.append(
            {
                "name": "next_trading_date_gap",
                "passed": next_trade_gap is not None and 1 <= next_trade_gap <= 4,
                "observed": {
                    "next_trading_date": next_trade_text,
                    "calendar_day_gap": next_trade_gap,
                },
                "expected": "1..4 calendar days after captured_date",
            }
        )
    if stock_universe is not None:
        missing_name_ratio = float(
            stock_universe["股票名称"].astype(str).str.strip().eq("").mean()
        ) if len(stock_universe) else 1.0
        checks.extend(
            [
                {
                    "name": "stock_universe_coverage",
                    "passed": len(stock_universe) >= minimum_stock_universe,
                    "observed": int(len(stock_universe)),
                    "expected": f">={minimum_stock_universe}",
                },
                {
                    "name": "stock_universe_names",
                    "passed": missing_name_ratio <= 0.01,
                    "observed": missing_name_ratio,
                    "expected": "<=1% missing names",
                },
            ]
        )
    failed_checks = [item["name"] for item in checks if not item["passed"]]
    payload = {
        "status": "gm_paper_account_snapshot",
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "snapshot_id": snapshot_id,
        "captured_at": captured_ts.isoformat(),
        "captured_date": captured_ts.strftime("%Y-%m-%d"),
        "account_id": account_id,
        "paper_only": True,
        "no_order": True,
        "orders_submitted": 0,
        "position_rows": int(len(frame)),
        "position_symbols": int(frame["symbol"].nunique()),
        "position_shares": int(frame["volume"].sum()),
        "unsupported_position_count": int(len(unsupported)),
        "account_mismatches": account_mismatches,
        "cash": cash_row,
        "trading_calendar": {
            "source": (
                "gm.api.get_next_trading_date"
                if next_trading_date is not None
                else None
            ),
            "exchange": "SHSE",
            "next_trading_date": next_trade_text or None,
            "calendar_day_gap": next_trade_gap,
        },
        "stock_universe": (
            {
                "source": "gm.api.get_instruments",
                "file": stock_universe_path.name,
                "sha256": sha256_file(stock_universe_path),
                "rows": int(len(stock_universe)),
                "st_name_rows": int(
                    stock_universe["股票名称"]
                    .astype(str)
                    .str.upper()
                    .str.contains("ST", regex=False)
                    .sum()
                ),
                "delisted_rows": int(
                    stock_universe["上市状态"].astype(str).ne("上市").sum()
                ),
            }
            if stock_universe is not None and stock_universe_path is not None
            else None
        ),
        "positions": {
            "file": positions_path.name,
            "sha256": sha256_file(positions_path),
            "rows": int(len(frame)),
        },
        "checks": checks,
        "paper_orders_allowed": False,
        "real_money_deployment_allowed": False,
    }
    snapshot_path = run_dir / "ACCOUNT_SNAPSHOT.json"
    pending = snapshot_path.with_suffix(".json.pending")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(pending, snapshot_path)
    payload["snapshot_file"] = str(snapshot_path)
    payload["positions_file"] = str(positions_path)
    return payload


def init(context) -> None:
    from gm.api import (
        get_cash,
        get_instruments,
        get_next_trading_date,
        get_position,
        set_account_id,
    )

    account_id = os.environ.get("GM_ACCOUNT_ID", "").strip()
    if not account_id:
        raise RuntimeError("GM_ACCOUNT_ID is required for a PAPER account snapshot")
    set_account_id(account_id)
    positions = get_position(account_id=account_id)
    cash = get_cash(account_id=account_id)
    captured_at = getattr(context, "now", None) or pd.Timestamp.now()
    next_trading_date = get_next_trading_date(
        "SHSE",
        pd.Timestamp(captured_at).strftime("%Y-%m-%d"),
    )
    instruments = get_instruments(
        exchanges=["SHSE", "SZSE"],
        sec_types=[1],
        skip_suspended=False,
        skip_st=False,
        fields=[
            "symbol",
            "sec_name",
            "sec_abbr",
            "is_suspended",
            "listed_date",
            "delisted_date",
        ],
        df=False,
    )
    snapshot_id = os.environ.get(
        "GM_ACCOUNT_SNAPSHOT_ID",
        f"paper_{pd.Timestamp(captured_at).strftime('%Y%m%d_%H%M%S')}",
    ).strip()
    output_root = Path(
        os.environ.get("GM_ACCOUNT_SNAPSHOT_DIR", str(DEFAULT_OUTPUT_ROOT))
    )
    payload = capture_snapshot(
        positions=list(positions or []),
        cash=cash or {},
        account_id=account_id,
        captured_at=captured_at,
        output_root=output_root,
        snapshot_id=snapshot_id,
        next_trading_date=next_trading_date,
        instruments=list(instruments or []),
    )
    print(
        "[GM-ACCOUNT-SNAPSHOT] NO ORDERS "
        f"passed={payload['passed']} account={account_id} "
        f"positions={payload['position_symbols']} shares={payload['position_shares']} "
        f"next_trade={payload['trading_calendar']['next_trading_date']} "
        f"stock_universe={payload['stock_universe']['rows']}",
        flush=True,
    )
    print(
        f"[GM-ACCOUNT-SNAPSHOT] snapshot={payload['snapshot_file']}",
        flush=True,
    )
    print(
        "[GM-ACCOUNT-SNAPSHOT] capture complete; this read-only strategy may now be stopped.",
        flush=True,
    )
    if not payload["passed"]:
        raise RuntimeError(
            "PAPER account snapshot failed: " + ", ".join(payload["failed_checks"])
        )


def on_error(context, code, info) -> None:
    print(f"[GM-ACCOUNT-SNAPSHOT] ERROR code={code} info={info}", flush=True)


if __name__ == "__main__":
    from gm.api import MODE_LIVE, run, set_token

    token = os.environ.get("GM_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GM_TOKEN environment variable is required")
    set_token(token)
    kwargs = {
        "strategy_id": os.environ.get(
            "GM_ACCOUNT_SNAPSHOT_STRATEGY_ID",
            "outer-middle-paper-account-snapshot",
        ),
        "filename": Path(__file__).name,
        "token": token,
        "mode": MODE_LIVE,
    }
    print(
        "[GM-ACCOUNT-SNAPSHOT] run kwargs="
        + str({**kwargs, "token": "***MASKED***"}),
        flush=True,
    )
    run(**kwargs)
