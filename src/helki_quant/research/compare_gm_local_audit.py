from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_GM_AUDIT = REPO_ROOT / "outputs" / "gm_c_baseline_audit" / "audit_top150_80_volume_deltafixed_20260605"
DEFAULT_LOCAL_LOG = (
    HERE
    / "outputs"
    / "production_logs_c_baseline"
    / "c_top150_rb45_risk0.80_cap0.30_stress"
)
DEFAULT_OUTPUT = HERE / "outputs" / "gm_local_audit_compare_top150_80_20260605.json"
MARKET_RESTRICTION_TERMS = ("停牌", "涨停", "跌停", "价格超过", "价格低于")


def gm_to_local_symbol(symbol: object) -> str:
    text = str(symbol).upper()
    if "." not in text:
        return text
    exchange, code = text.split(".", 1)
    prefix = "SH" if exchange in {"SHSE", "SH"} else "SZ"
    return f"{prefix}{code}"


def local_to_gm_symbol(symbol: object) -> str:
    text = str(symbol).upper()
    if text.startswith("SH"):
        return f"SHSE.{text[-6:]}"
    if text.startswith("SZ"):
        return f"SZSE.{text[-6:]}"
    return text


def side_name_from_gm(value: object) -> str:
    try:
        side = int(value)
    except Exception:
        return str(value).upper()
    if side == 1:
        return "BUY"
    if side == 2:
        return "SELL"
    if side == 0:
        return "PENDING"
    return f"UNKNOWN_{side}"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _shift_dates_by_local_calendar(
    dates: pd.Series,
    local_log_dir: Path,
    shift_trading_days: int,
) -> pd.Series:
    if shift_trading_days == 0:
        return dates
    calendar_source = local_log_dir / "daily_account.csv"
    if not calendar_source.exists():
        raise FileNotFoundError(f"local calendar source not found: {calendar_source}")
    local_daily = pd.read_csv(calendar_source)
    if "trade_date" not in local_daily.columns:
        raise ValueError(f"trade_date column not found in {calendar_source}")
    calendar = (
        pd.to_datetime(local_daily["trade_date"])
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not calendar:
        raise ValueError(f"empty local calendar: {calendar_source}")
    shifted = []
    for value in dates:
        date = str(value)
        insertion = 0
        while insertion < len(calendar) and calendar[insertion] <= date:
            insertion += 1
        index = insertion + shift_trading_days - 1
        if index < 0:
            index = 0
        if index >= len(calendar):
            shifted.append(date)
        else:
            shifted.append(calendar[index])
    return pd.Series(shifted, index=dates.index)


def aggregate_gm_fills(path: Path, local_log_dir: Path, gm_date_shift_trading_days: int) -> pd.DataFrame:
    orders = pd.read_csv(path / "order_status.csv")
    if orders.empty:
        return pd.DataFrame(columns=["trade_date", "instrument", "side", "gm_filled_volume", "gm_filled_amount"])
    filled = orders[(orders["status_name"] == "Filled") & (pd.to_numeric(orders["filled_volume"], errors="coerce") > 0)]
    if filled.empty:
        return pd.DataFrame(columns=["trade_date", "instrument", "side", "gm_filled_volume", "gm_filled_amount"])
    filled = filled.copy()
    filled["trade_date"] = pd.to_datetime(filled["event_date"]).dt.strftime("%Y-%m-%d")
    filled["trade_date"] = _shift_dates_by_local_calendar(
        filled["trade_date"],
        local_log_dir,
        gm_date_shift_trading_days,
    )
    filled["instrument"] = filled["symbol"].map(gm_to_local_symbol)
    filled["side"] = filled["side"].map(side_name_from_gm)
    filled["filled_volume"] = pd.to_numeric(filled["filled_volume"], errors="coerce").fillna(0).astype(int)
    filled["filled_amount"] = pd.to_numeric(filled["filled_amount"], errors="coerce").fillna(0.0)
    return (
        filled.groupby(["trade_date", "instrument", "side"], as_index=False)
        .agg(gm_filled_volume=("filled_volume", "sum"), gm_filled_amount=("filled_amount", "sum"))
        .sort_values(["trade_date", "instrument", "side"])
    )


def aggregate_local_fills(path: Path) -> pd.DataFrame:
    fills = pd.read_csv(path / "fills.csv")
    if fills.empty:
        return pd.DataFrame(columns=["trade_date", "instrument", "side", "local_filled_volume", "local_filled_amount"])
    fills = fills.copy()
    fills["trade_date"] = pd.to_datetime(fills["trade_date"]).dt.strftime("%Y-%m-%d")
    fills["instrument"] = fills["instrument"].astype(str).str.upper()
    fills["side"] = fills["side"].astype(str).str.upper()
    fills["shares"] = pd.to_numeric(fills["shares"], errors="coerce").fillna(0).astype(int)
    fills["gross_value"] = pd.to_numeric(fills["gross_value"], errors="coerce").fillna(0.0)
    return (
        fills.groupby(["trade_date", "instrument", "side"], as_index=False)
        .agg(local_filled_volume=("shares", "sum"), local_filled_amount=("gross_value", "sum"))
        .sort_values(["trade_date", "instrument", "side"])
    )


def summarize_diff(gm: pd.DataFrame, local: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    merged = gm.merge(local, on=["trade_date", "instrument", "side"], how="outer", indicator=True)
    for col in ["gm_filled_volume", "gm_filled_amount", "local_filled_volume", "local_filled_amount"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    merged["volume_diff"] = merged["gm_filled_volume"] - merged["local_filled_volume"]
    merged["amount_diff"] = merged["gm_filled_amount"] - merged["local_filled_amount"]
    mismatches = merged[merged["volume_diff"].abs() > 1e-9].copy()
    summary = {
        "gm_filled_keys": int(len(gm)),
        "local_filled_keys": int(len(local)),
        "matched_keys": int((merged["_merge"] == "both").sum()),
        "gm_only_keys": int((merged["_merge"] == "left_only").sum()),
        "local_only_keys": int((merged["_merge"] == "right_only").sum()),
        "volume_mismatch_keys": int(len(mismatches)),
        "gm_filled_volume_total": int(merged["gm_filled_volume"].sum()),
        "local_filled_volume_total": int(merged["local_filled_volume"].sum()),
        "filled_volume_diff_total": int(merged["volume_diff"].sum()),
        "gm_filled_amount_total": float(merged["gm_filled_amount"].sum()),
        "local_filled_amount_total": float(merged["local_filled_amount"].sum()),
        "filled_amount_diff_total": float(merged["amount_diff"].sum()),
    }
    return merged, summary


def summarize_rejections(order_status: pd.DataFrame) -> dict[str, object]:
    rejected = order_status[order_status["status_name"] == "Rejected"].copy()
    if rejected.empty:
        return {
            "market_restriction_rejected_orders": 0,
            "unexpected_rejected_orders": 0,
            "unique_rejected_symbols": 0,
            "unique_rejected_symbol_sides": 0,
            "rejected_buy_orders": 0,
            "rejected_sell_orders": 0,
            "unresolved_rejected_sell_symbols": 0,
            "unresolved_rejected_sell_symbol_list": [],
        }
    rejected["side_name"] = rejected["side"].map(side_name_from_gm)
    rejected["event_ts"] = pd.to_datetime(rejected["event_date"], errors="coerce")
    details = rejected["ord_rej_reason_detail"].fillna("").astype(str)
    expected_mask = details.map(
        lambda detail: any(term in detail for term in MARKET_RESTRICTION_TERMS)
    )
    filled = order_status[order_status["status_name"] == "Filled"].copy()
    filled["side_name"] = filled["side"].map(side_name_from_gm)
    filled["event_ts"] = pd.to_datetime(filled["event_date"], errors="coerce")
    unresolved: list[str] = []
    sell_rejected = rejected[rejected["side_name"] == "SELL"]
    for symbol, part in sell_rejected.groupby("symbol"):
        last_rejection = part["event_ts"].max()
        resolution = filled[
            (filled["symbol"].astype(str) == str(symbol))
            & (filled["side_name"] == "SELL")
            & (filled["event_ts"] >= last_rejection)
        ]
        if resolution.empty:
            unresolved.append(str(symbol))
    return {
        "market_restriction_rejected_orders": int(expected_mask.sum()),
        "unexpected_rejected_orders": int((~expected_mask).sum()),
        "unique_rejected_symbols": int(rejected["symbol"].astype(str).nunique()),
        "unique_rejected_symbol_sides": int(
            rejected[["symbol", "side_name"]].drop_duplicates().shape[0]
        ),
        "rejected_buy_orders": int((rejected["side_name"] == "BUY").sum()),
        "rejected_sell_orders": int((rejected["side_name"] == "SELL").sum()),
        "unresolved_rejected_sell_symbols": len(unresolved),
        "unresolved_rejected_sell_symbol_list": sorted(unresolved),
    }


def compare(
    gm_audit_dir: Path,
    local_log_dir: Path,
    output_path: Path,
    gm_date_shift_trading_days: int = 0,
) -> dict:
    gm_summary = load_json(gm_audit_dir / "summary.json")
    local_audit = load_json(local_log_dir / "audit.json")
    gm_fills = aggregate_gm_fills(gm_audit_dir, local_log_dir, gm_date_shift_trading_days)
    local_fills = aggregate_local_fills(local_log_dir)
    merged, fill_summary = summarize_diff(gm_fills, local_fills)

    submissions = pd.read_csv(gm_audit_dir / "submissions.csv")
    order_status = pd.read_csv(gm_audit_dir / "order_status.csv")
    rejected = order_status[order_status["status_name"] == "Rejected"].copy()
    rejection_summary = summarize_rejections(order_status)
    local_daily = pd.read_csv(local_log_dir / "daily_account.csv")

    gm_indicator = gm_summary.get("indicator", {})
    result = {
        "status": "gm_local_audit_compare_research_only",
        "gm_audit_dir": str(gm_audit_dir),
        "local_log_dir": str(local_log_dir),
        "gm_date_shift_trading_days": gm_date_shift_trading_days,
        "profile": local_audit.get("profile", {}).get("name"),
        "cost": local_audit.get("cost", {}).get("name"),
        "fill_comparison": fill_summary,
        "gm": {
            "submitted_orders": int(gm_summary.get("submitted_orders", len(submissions))),
            "filled_orders": int((order_status["status_name"] == "Filled").sum()),
            "rejected_orders": int(len(rejected)),
            **rejection_summary,
            "status_counts": gm_summary.get("status_counts", {}),
            "total_return": gm_indicator.get("pnl_ratio"),
            "annualized_return": gm_indicator.get("pnl_ratio_annual"),
            "sharpe": gm_indicator.get("sharp_ratio"),
            "max_drawdown": gm_indicator.get("max_drawdown"),
        },
        "local": {
            "trades": int(local_audit.get("trades", 0)),
            "turnover": float(local_audit.get("turnover", 0.0)),
            "max_daily_turnover": float(local_audit.get("max_daily_turnover", 0.0)),
            "max_gross_exposure": float(local_audit.get("max_gross_exposure", 0.0)),
            "min_cash": float(local_audit.get("min_cash", 0.0)),
            "total_return": float(local_audit.get("total_return", 0.0)),
            "max_drawdown": float(local_audit.get("max_drawdown", 0.0)),
            "final_nav": float(local_audit.get("final_nav", 0.0)),
            "final_cash": float(local_daily["cash"].iloc[-1]) if not local_daily.empty else None,
            "final_exposure": float(local_daily["gross_exposure"].iloc[-1]) if not local_daily.empty else None,
        },
        "differences": {
            "total_return_gm_minus_local": (
                None
                if gm_indicator.get("pnl_ratio") is None
                else float(gm_indicator.get("pnl_ratio")) - float(local_audit.get("total_return", 0.0))
            ),
            "max_drawdown_gm_minus_local": (
                None
                if gm_indicator.get("max_drawdown") is None
                else float(gm_indicator.get("max_drawdown")) - float(local_audit.get("max_drawdown", 0.0))
            ),
            "filled_order_count_gm_minus_local": int((order_status["status_name"] == "Filled").sum())
            - int(local_audit.get("trades", 0)),
        },
        "rejected_sample": rejected.head(20).to_dict(orient="records"),
        "volume_mismatch_sample": merged[merged["volume_diff"].abs() > 1e-9]
        .head(50)
        .to_dict(orient="records"),
        "notes": [
            "Fill amount and return differences can be caused by platform price, fee, and mark-to-market rules.",
            "Volume comparison is the primary production-parity check because both paths use target-volume orders.",
        ],
        "deployment_allowed": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    mismatch_path = output_path.with_suffix(".volume_mismatches.csv")
    merged[merged["volume_diff"].abs() > 1e-9].to_csv(mismatch_path, index=False, encoding="utf-8-sig")
    result["volume_mismatch_csv"] = str(mismatch_path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gm-audit-dir", default=str(DEFAULT_GM_AUDIT))
    parser.add_argument("--local-log-dir", default=str(DEFAULT_LOCAL_LOG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--gm-date-shift-trading-days",
        type=int,
        default=0,
        help="Shift GM filled event dates forward by N local trading sessions before volume comparison.",
    )
    args = parser.parse_args()
    result = compare(
        Path(args.gm_audit_dir).resolve(),
        Path(args.local_log_dir).resolve(),
        Path(args.output).resolve(),
        args.gm_date_shift_trading_days,
    )
    fill = result["fill_comparison"]
    print(
        "[gm-local compare] "
        f"profile={result['profile']} cost={result['cost']} "
        f"gm_filled={result['gm']['filled_orders']} local_trades={result['local']['trades']} "
        f"gm_only={fill['gm_only_keys']} local_only={fill['local_only_keys']} "
        f"volume_mismatch={fill['volume_mismatch_keys']} "
        f"ret_diff={result['differences']['total_return_gm_minus_local']:+.4%}",
        flush=True,
    )
    print(f"[gm-local compare] output={args.output}", flush=True)


if __name__ == "__main__":
    main()
