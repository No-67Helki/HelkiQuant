from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--run must use label=profile_log_dir")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise ValueError("--run must use non-empty label=profile_log_dir")
    return label.strip(), Path(path.strip()).resolve()


def max_drawdown(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return float("nan")
    peak = np.maximum.accumulate(array)
    return float(np.max(1.0 - array / np.maximum(peak, 1e-12)))


def evaluate(label: str, path: Path, start: str, end: str) -> dict:
    account_path = path / "daily_account.csv"
    fills_path = path / "fills.csv"
    if not account_path.exists() or not fills_path.exists():
        raise FileNotFoundError(f"missing daily_account.csv or fills.csv under {path}")

    account = pd.read_csv(account_path, parse_dates=["trade_date"])
    fills = pd.read_csv(fills_path, parse_dates=["trade_date"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    before = account[account["trade_date"] < start_ts]
    if before.empty:
        raise ValueError(f"{path} has no account row before {start}")
    initial_row = before.iloc[-1]
    window = account[account["trade_date"].between(start_ts, end_ts)].copy()
    if window.empty:
        raise ValueError(f"{path} has no account rows in {start}..{end}")
    window_fills = fills[fills["trade_date"].between(start_ts, end_ts)].copy()
    nav_path = pd.concat(
        [pd.Series([float(initial_row["nav"])]), window["nav"].reset_index(drop=True)],
        ignore_index=True,
    )
    initial_nav = float(initial_row["nav"])
    final_nav = float(window.iloc[-1]["nav"])
    return {
        "label": label,
        "profile_log_dir": str(path),
        "initial_date": str(pd.Timestamp(initial_row["trade_date"]).date()),
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "days": int(len(window)),
        "initial_nav": initial_nav,
        "final_nav": final_nav,
        "return": final_nav / initial_nav - 1.0,
        "max_drawdown": max_drawdown(nav_path),
        "turnover": float(window["day_turnover"].sum()),
        "trades": int(window["day_trades"].sum()),
        "fills": int(len(window_fills)),
        "fees": float(window_fills["fee"].sum()) if len(window_fills) else 0.0,
        "min_cash": float(window["cash"].min()),
        "max_gross_exposure": float(window["gross_exposure"].max()),
        "final_holdings_count": int(window.iloc[-1]["holdings_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    rows = [evaluate(*parse_run(value), args.start, args.end) for value in args.run]
    report = {
        "status": "production_forward_slice_comparison_diagnostic_only",
        "window": {"start": args.start, "end": args.end},
        "holdout_status": "consumed_not_untouched",
        "rows": rows,
        "deployment_allowed": False,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_path = Path(args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"[forward slice] {row['label']} ret={row['return']:+.2%} "
            f"mdd={row['max_drawdown']:.2%} turn={row['turnover']:.2f} "
            f"trades={row['trades']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
