from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def local_to_gm_symbol(symbol: object) -> str:
    text = str(symbol).upper()
    if text.startswith("SH"):
        return f"SHSE.{text[-6:]}"
    if text.startswith("SZ"):
        return f"SZSE.{text[-6:]}"
    return text


def select_blocked_sells(
    orders: pd.DataFrame,
    mode: str,
    sell_notional_fraction: float,
    max_blocked_per_date: int,
) -> pd.DataFrame:
    sells = orders[orders["side"].astype(str).str.upper().eq("SELL")].copy()
    if sells.empty:
        return sells
    sells["order_notional"] = pd.to_numeric(sells["order_notional"], errors="coerce").fillna(0.0)
    sells["target_shares"] = pd.to_numeric(sells["target_shares"], errors="coerce").fillna(0).astype(int)
    if mode == "exit_only":
        sells = sells[sells["target_shares"].eq(0)].copy()
    elif mode != "largest_fraction":
        raise ValueError(f"unknown mode: {mode}")
    if sells.empty:
        return sells

    selected = []
    for _, day in sells.groupby("trade_date", sort=True):
        day = day.sort_values("order_notional", ascending=False).copy()
        if max_blocked_per_date > 0:
            day = day.head(max_blocked_per_date)
        if sell_notional_fraction >= 1.0:
            selected.append(day)
            continue
        threshold = float(day["order_notional"].sum()) * sell_notional_fraction
        running = 0.0
        picked_rows = []
        for row in day.itertuples(index=False):
            if running >= threshold and picked_rows:
                break
            picked_rows.append(row._asdict())
            running += float(getattr(row, "order_notional"))
        if picked_rows:
            selected.append(pd.DataFrame(picked_rows))
    if not selected:
        return sells.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True)


def build_audit(
    local_log_dir: Path,
    output_dir: Path,
    mode: str,
    sell_notional_fraction: float,
    max_blocked_per_date: int,
    reason: str,
) -> dict:
    orders_path = local_log_dir / "orders.csv"
    if not orders_path.exists():
        raise FileNotFoundError(f"missing local orders file: {orders_path}")
    orders = pd.read_csv(orders_path)
    required = {"trade_date", "instrument", "side", "order_notional", "target_shares"}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"orders.csv missing columns: {sorted(missing)}")

    blocked = select_blocked_sells(orders, mode, sell_notional_fraction, max_blocked_per_date)
    output_dir.mkdir(parents=True, exist_ok=True)

    status_fields = [
        "event_date",
        "symbol",
        "side",
        "status_name",
        "ord_rej_reason_detail",
        "filled_volume",
        "filled_amount",
        "local_instrument",
        "local_order_notional",
        "stress_mode",
    ]
    status_path = output_dir / "order_status.csv"
    with status_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=status_fields)
        writer.writeheader()
        for row in blocked.itertuples(index=False):
            writer.writerow(
                {
                    "event_date": str(pd.Timestamp(getattr(row, "trade_date")).date()),
                    "symbol": local_to_gm_symbol(getattr(row, "instrument")),
                    "side": 2,
                    "status_name": "Rejected",
                    "ord_rej_reason_detail": reason,
                    "filled_volume": 0,
                    "filled_amount": 0.0,
                    "local_instrument": getattr(row, "instrument"),
                    "local_order_notional": float(getattr(row, "order_notional")),
                    "stress_mode": mode,
                }
            )

    blocked_by_date = (
        blocked.groupby("trade_date", as_index=False)
        .agg(blocked_orders=("instrument", "count"), blocked_notional=("order_notional", "sum"))
        .sort_values("trade_date")
    )
    blocked_by_date.to_csv(output_dir / "blocked_by_date.csv", index=False, encoding="utf-8-sig")

    sells = orders[orders["side"].astype(str).str.upper().eq("SELL")].copy()
    total_sell_notional = float(pd.to_numeric(sells["order_notional"], errors="coerce").fillna(0.0).sum())
    report = {
        "status": "synthetic_sell_block_stress_audit_research_only",
        "local_log_dir": str(local_log_dir),
        "output_dir": str(output_dir),
        "orders_path": str(orders_path),
        "mode": mode,
        "sell_notional_fraction": sell_notional_fraction,
        "max_blocked_per_date": max_blocked_per_date,
        "reason": reason,
        "sell_orders": int(len(sells)),
        "blocked_orders": int(len(blocked)),
        "sell_notional": total_sell_notional,
        "blocked_notional": float(blocked["order_notional"].sum()) if len(blocked) else 0.0,
        "blocked_notional_ratio": (
            float(blocked["order_notional"].sum()) / total_sell_notional if total_sell_notional > 0 else 0.0
        ),
        "blocked_dates": int(blocked["trade_date"].nunique()) if len(blocked) else 0,
        "deployment_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[sell-block stress] "
        f"mode={mode} blocked_orders={report['blocked_orders']} "
        f"blocked_ratio={report['blocked_notional_ratio']:.2%} "
        f"dates={report['blocked_dates']} output={output_dir}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-log-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["largest_fraction", "exit_only"], default="largest_fraction")
    parser.add_argument("--sell-notional-fraction", type=float, default=0.30)
    parser.add_argument("--max-blocked-per-date", type=int, default=8)
    parser.add_argument(
        "--reason",
        default="synthetic sell blocked stress: unavailable sell liquidity / limit-down style rejection",
    )
    args = parser.parse_args()
    build_audit(
        Path(args.local_log_dir).resolve(),
        Path(args.output_dir).resolve(),
        args.mode,
        args.sell_notional_fraction,
        args.max_blocked_per_date,
        args.reason,
    )


if __name__ == "__main__":
    main()
