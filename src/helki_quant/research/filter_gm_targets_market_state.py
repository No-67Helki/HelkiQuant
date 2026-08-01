from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CHINESE_COLUMNS = {
    "日期": "trade_date",
    "股票代码": "code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
}


def gm_to_local_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if "." not in text:
        return text
    exchange, code = text.split(".", 1)
    prefix = "SH" if exchange in {"SHSE", "SH"} else "SZ"
    return f"{prefix}{code}"


def local_code(symbol: object) -> str:
    return gm_to_local_symbol(symbol)[-6:]


def side_from_gm(value: object) -> str:
    try:
        side = int(value)
    except Exception:
        return str(value).upper()
    if side == 1:
        return "BUY"
    if side == 2:
        return "SELL"
    return str(side)


def load_daily_state(
    daily_root: Path,
    symbols: set[str],
    dates: set[str],
    limit_pct_threshold: float,
    allow_future_missing_daily: bool = False,
) -> tuple[dict[tuple[str, str], set[str]], list[dict[str, object]]]:
    blocks: dict[tuple[str, str], set[str]] = {}
    rows: list[dict[str, object]] = []
    for symbol in sorted(symbols):
        code = local_code(symbol)
        path = daily_root / f"{code}_daily_qfq.csv"
        if not path.exists():
            for date in dates:
                blocks.setdefault((date, symbol), set()).update({"BUY", "SELL"})
                rows.append(
                    {
                        "trade_date": date,
                        "symbol": symbol,
                        "source": "daily",
                        "block_sides": "BUY,SELL",
                        "reason": "daily_file_missing",
                    }
                )
            continue
        frame = pd.read_csv(path)
        frame = frame.rename(columns={k: v for k, v in CHINESE_COLUMNS.items() if k in frame.columns})
        if "trade_date" not in frame.columns:
            continue
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        max_available_date = frame["trade_date"].dropna().max()
        frame = frame[frame["trade_date"].isin(dates)].copy()
        by_date = {str(row.trade_date): row for row in frame.itertuples(index=False)}
        for date in sorted(dates):
            row = by_date.get(date)
            reason = ""
            if row is None:
                if allow_future_missing_daily and max_available_date is not None and date > str(max_available_date):
                    continue
                reason = "daily_row_missing_or_suspended"
            else:
                volume = pd.to_numeric(getattr(row, "volume", None), errors="coerce")
                pct_chg = pd.to_numeric(getattr(row, "pct_chg", None), errors="coerce")
                if pd.isna(volume) or float(volume) <= 0:
                    reason = "daily_volume_zero_or_nan"
                elif not pd.isna(pct_chg) and abs(float(pct_chg)) >= limit_pct_threshold:
                    reason = f"daily_abs_pct_chg_ge_{limit_pct_threshold:g}"
            if reason:
                blocks.setdefault((date, symbol), set()).update({"BUY", "SELL"})
                rows.append(
                    {
                        "trade_date": date,
                        "symbol": symbol,
                        "source": "daily",
                        "block_sides": "BUY,SELL",
                        "reason": reason,
                    }
                )
    return blocks, rows


def load_gm_rejection_blocks(audit_dir: Path | None) -> tuple[dict[tuple[str, str], set[str]], list[dict[str, object]]]:
    blocks: dict[tuple[str, str], set[str]] = {}
    rows: list[dict[str, object]] = []
    if audit_dir is None:
        return blocks, rows
    path = audit_dir / "order_status.csv"
    if not path.exists():
        return blocks, rows
    frame = pd.read_csv(path)
    if frame.empty or "status_name" not in frame.columns:
        return blocks, rows
    rejected = frame[frame["status_name"].astype(str).eq("Rejected")].copy()
    for row in rejected.itertuples(index=False):
        date = str(getattr(row, "event_date"))
        symbol = str(getattr(row, "symbol")).strip().upper()
        side = side_from_gm(getattr(row, "side", ""))
        detail = str(getattr(row, "ord_rej_reason_detail", ""))
        sides = {"BUY", "SELL"} if "停牌" in detail else {side}
        blocks.setdefault((date, symbol), set()).update(sides)
        rows.append(
            {
                "trade_date": date,
                "symbol": symbol,
                "source": "gm_audit",
                "block_sides": ",".join(sorted(sides)),
                "reason": detail,
            }
        )
    return blocks, rows


def merge_blocks(*items: tuple[dict[tuple[str, str], set[str]], list[dict[str, object]]]) -> tuple[dict[tuple[str, str], set[str]], list[dict[str, object]]]:
    merged: dict[tuple[str, str], set[str]] = {}
    rows: list[dict[str, object]] = []
    for blocks, block_rows in items:
        rows.extend(block_rows)
        for key, sides in blocks.items():
            merged.setdefault(key, set()).update(sides)
    return merged, rows


def row_for_output(
    row: pd.Series,
    trade_date: str,
    target_shares: int,
    signal_date: str | None = None,
) -> dict[str, object]:
    out = row.to_dict()
    out["trade_date"] = trade_date
    if signal_date is not None:
        out["signal_date"] = signal_date
    out["target_shares"] = int(target_shares)
    return out


def load_initial_target_holdings(
    path: Path | None,
) -> tuple[dict[str, int], dict[str, pd.Series]]:
    if path is None:
        return {}, {}
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"initial target not found: {path}")
    frame = pd.read_csv(path, dtype={"symbol": str, "instrument": str})
    required = {"symbol", "target_shares"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"initial target missing columns: {sorted(missing)}")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["target_shares"] = (
        pd.to_numeric(frame["target_shares"], errors="coerce").fillna(0).astype(int)
    )
    frame = frame[frame["target_shares"].gt(0)].copy()
    if frame["symbol"].duplicated().any():
        raise ValueError("initial target contains duplicate symbols")
    held = dict(zip(frame["symbol"], frame["target_shares"]))
    rows = {
        str(row["symbol"]): row.copy()
        for _, row in frame.iterrows()
    }
    return held, rows


def filter_targets(
    target_csv: Path,
    output_csv: Path,
    daily_root: Path,
    gm_audit_dir: Path | None,
    limit_pct_threshold: float,
    allow_future_missing_daily: bool = False,
    runtime_retry_blocked_sells: bool = False,
    initial_target_csv: Path | None = None,
) -> dict[str, object]:
    target = pd.read_csv(target_csv, dtype={"symbol": str, "instrument": str})
    required = {"trade_date", "symbol", "target_shares"}
    missing = required - set(target.columns)
    if missing:
        raise ValueError(f"missing columns in {target_csv}: {sorted(missing)}")
    target["trade_date"] = pd.to_datetime(target["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    target["symbol"] = target["symbol"].astype(str).str.upper()
    target["target_shares"] = pd.to_numeric(target["target_shares"], errors="coerce").fillna(0).astype(int)
    dates = set(target["trade_date"].dropna().astype(str))
    symbols = set(target["symbol"].dropna().astype(str))

    daily_blocks = load_daily_state(
        daily_root,
        symbols,
        dates,
        limit_pct_threshold,
        allow_future_missing_daily,
    )
    gm_blocks = load_gm_rejection_blocks(gm_audit_dir)
    blocks, block_rows = merge_blocks(daily_blocks, gm_blocks)

    output_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    held, last_row = load_initial_target_holdings(initial_target_csv)
    initial_holding_count = len(held)
    for date, day in target.groupby("trade_date", sort=True):
        signal_dates = (
            day["signal_date"].dropna().astype(str).unique()
            if "signal_date" in day.columns
            else []
        )
        if len(signal_dates) > 1:
            raise ValueError(f"target date {date} contains multiple signal dates")
        current_signal_date = str(signal_dates[0]) if len(signal_dates) == 1 else None
        desired = {str(row.symbol): row for row in day.itertuples(index=False)}
        day_symbols = sorted(set(held) | set(desired))
        next_held: dict[str, int] = {}
        for symbol in day_symbols:
            current = int(held.get(symbol, 0))
            row_tuple = desired.get(symbol)
            desired_shares = int(getattr(row_tuple, "target_shares")) if row_tuple is not None else 0
            side = "BUY" if desired_shares > current else "SELL" if desired_shares < current else "HOLD"
            final_shares = desired_shares
            reason = "target"
            if side in blocks.get((date, symbol), set()):
                if side == "SELL" and runtime_retry_blocked_sells:
                    # Keep the lower target so SELL_FIRST can retry the order on
                    # later sessions while holding all buy increases.
                    reason = "runtime_retry_blocked_sell"
                else:
                    final_shares = current
                    reason = f"blocked_{side.lower()}"
            if final_shares <= 0:
                if current > 0 and final_shares == 0:
                    action_rows.append(
                        {
                            "trade_date": date,
                            "symbol": symbol,
                            "side": side,
                            "current_shares": current,
                            "desired_shares": desired_shares,
                            "final_shares": final_shares,
                            "reason": reason,
                        }
                    )
                continue
            if row_tuple is not None:
                row_series = pd.Series(row_tuple._asdict())
            else:
                row_series = last_row[symbol].copy()
            output_rows.append(
                row_for_output(
                    row_series,
                    str(date),
                    final_shares,
                    current_signal_date,
                )
            )
            next_held[symbol] = final_shares
            last_row[symbol] = row_series.copy()
            action_rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "side": side,
                    "current_shares": current,
                    "desired_shares": desired_shares,
                    "final_shares": final_shares,
                    "reason": reason,
                }
            )
        held = next_held

    out = pd.DataFrame(output_rows)
    if out.empty:
        out = target.iloc[0:0].copy()
    ordered_cols = [column for column in target.columns if column in out.columns] + [
        column for column in out.columns if column not in target.columns
    ]
    out = out[ordered_cols].sort_values(["trade_date", "rank", "symbol"], na_position="last").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    actions = pd.DataFrame(action_rows)
    blocks_df = pd.DataFrame(block_rows)
    actions_path = output_csv.with_suffix(".market_state_actions.csv")
    blocks_path = output_csv.with_suffix(".market_state_blocks.csv")
    actions.to_csv(actions_path, index=False, encoding="utf-8-sig")
    blocks_df.to_csv(blocks_path, index=False, encoding="utf-8-sig")

    blocked_actions = actions[actions["reason"].astype(str).str.startswith("blocked_")] if not actions.empty else actions
    runtime_retry_sells = (
        actions[actions["reason"].astype(str).eq("runtime_retry_blocked_sell")]
        if not actions.empty
        else actions
    )
    by_date = (
        out.groupby("trade_date")
        .agg(rows=("symbol", "size"), symbols=("symbol", "nunique"), weight_sum=("target_weight", "sum"))
        .reset_index()
        if not out.empty and "target_weight" in out.columns
        else pd.DataFrame()
    )
    by_date_path = output_csv.with_suffix(".by_date.csv")
    by_date.to_csv(by_date_path, index=False, encoding="utf-8-sig")
    result = {
        "status": "gm_targets_market_state_filtered",
        "input_csv": str(target_csv),
        "output_csv": str(output_csv),
        "daily_root": str(daily_root),
        "gm_audit_dir": str(gm_audit_dir) if gm_audit_dir else None,
        "limit_pct_threshold": limit_pct_threshold,
        "allow_future_missing_daily": allow_future_missing_daily,
        "runtime_retry_blocked_sells": runtime_retry_blocked_sells,
        "initial_target": (
            str(initial_target_csv.resolve()) if initial_target_csv is not None else None
        ),
        "initial_holding_count": initial_holding_count,
        "input_rows": int(len(target)),
        "output_rows": int(len(out)),
        "input_dates": int(target["trade_date"].nunique()),
        "output_dates": int(out["trade_date"].nunique()) if len(out) else 0,
        "input_symbols": int(target["symbol"].nunique()),
        "output_symbols": int(out["symbol"].nunique()) if len(out) else 0,
        "blocked_actions": int(len(blocked_actions)),
        "blocked_buy_actions": int((blocked_actions["reason"] == "blocked_buy").sum()) if not blocked_actions.empty else 0,
        "blocked_sell_actions": int((blocked_actions["reason"] == "blocked_sell").sum()) if not blocked_actions.empty else 0,
        "runtime_retry_sell_actions": int(len(runtime_retry_sells)),
        "block_rows": int(len(blocks_df)),
        "actions_csv": str(actions_path),
        "blocks_csv": str(blocks_path),
        "by_date_csv": str(by_date_path),
        "target_weight_sum_min": float(by_date["weight_sum"].min()) if not by_date.empty else 0.0,
        "target_weight_sum_max": float(by_date["weight_sum"].max()) if not by_date.empty else 0.0,
        "deployment_allowed": False,
    }
    output_csv.with_suffix(".manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--daily-root", required=True, type=Path)
    parser.add_argument("--gm-audit-dir", type=Path)
    parser.add_argument("--limit-pct-threshold", type=float, default=19.5)
    parser.add_argument(
        "--allow-future-missing-daily",
        action="store_true",
        help="Do not block targets whose trade_date is after the latest available daily bar.",
    )
    parser.add_argument(
        "--runtime-retry-blocked-sells",
        action="store_true",
        help=(
            "Preserve lower/zero sell targets when the execution-day market state "
            "is blocked, so SELL_FIRST can retry them on later sessions."
        ),
    )
    args = parser.parse_args()
    result = filter_targets(
        args.target_csv.resolve(),
        args.output_csv.resolve(),
        args.daily_root.resolve(),
        args.gm_audit_dir.resolve() if args.gm_audit_dir else None,
        args.limit_pct_threshold,
        args.allow_future_missing_daily,
        args.runtime_retry_blocked_sells,
    )
    print(
        "[gm target market-state filter] "
        f"rows={result['input_rows']}->{result['output_rows']} "
        f"blocked={result['blocked_actions']} "
        f"retry_sells={result['runtime_retry_sell_actions']} "
        f"weight={result['target_weight_sum_min']:.2%}-{result['target_weight_sum_max']:.2%} "
        f"output={args.output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
