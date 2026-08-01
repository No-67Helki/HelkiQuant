from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from export_gm_c_baseline_targets import gm_to_local_symbol, min_buy_shares


def load_forbidden(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None or not path.exists():
        return set(), set()
    frame = pd.read_csv(path, dtype=str).fillna("")
    local: set[str] = set()
    gm: set[str] = set()
    if "instrument" in frame.columns:
        local.update(frame["instrument"].astype(str).str.strip().str.upper())
    if "local_instrument" in frame.columns:
        local.update(frame["local_instrument"].astype(str).str.strip().str.upper())
    if "gm_symbol" in frame.columns:
        gm.update(frame["gm_symbol"].astype(str).str.strip().str.upper())
        local.update(frame["gm_symbol"].map(gm_to_local_symbol).astype(str).str.upper())
    if "symbol" in frame.columns:
        gm.update(frame["symbol"].astype(str).str.strip().str.upper())
        local.update(frame["symbol"].map(gm_to_local_symbol).astype(str).str.upper())
    return {v for v in local if v and v != "NAN"}, {v for v in gm if v and v != "NAN"}


def audit(target_csv: Path, output_path: Path, forbidden_path: Path | None) -> dict:
    frame = pd.read_csv(target_csv, parse_dates=["trade_date"])
    required = {"trade_date", "symbol", "instrument", "target_weight", "target_shares"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"target csv missing columns: {missing}")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0)
    frame["target_shares"] = pd.to_numeric(frame["target_shares"], errors="coerce").fillna(0).astype(int)
    forbidden_local, forbidden_gm = load_forbidden(forbidden_path)
    forbidden_mask = frame["instrument"].isin(forbidden_local) | frame["symbol"].isin(forbidden_gm)
    lot_bad = []
    min_lot_bad = []
    for row in frame.itertuples(index=False):
        shares = int(getattr(row, "target_shares"))
        symbol = str(getattr(row, "symbol"))
        if shares % 100 != 0:
            lot_bad.append({"trade_date": str(getattr(row, "trade_date").date()), "symbol": symbol, "shares": shares})
        if shares > 0 and shares < min_buy_shares(symbol):
            min_lot_bad.append({"trade_date": str(getattr(row, "trade_date").date()), "symbol": symbol, "shares": shares})
    by_date = (
        frame.groupby("trade_date")
        .agg(
            rows=("symbol", "count"),
            symbols=("symbol", "nunique"),
            target_weight_sum=("target_weight", "sum"),
            target_shares_sum=("target_shares", "sum"),
        )
        .reset_index()
        .sort_values("trade_date")
    )
    low_weight = by_date[by_date["target_weight_sum"] < 0.20].copy()
    report = {
        "status": "gm_target_csv_audit",
        "target_csv": str(target_csv.resolve()),
        "rows": int(len(frame)),
        "dates": int(frame["trade_date"].nunique()),
        "symbols": int(frame["symbol"].nunique()),
        "date_start": str(frame["trade_date"].min().date()) if len(frame) else None,
        "date_end": str(frame["trade_date"].max().date()) if len(frame) else None,
        "target_weight_sum_min": float(by_date["target_weight_sum"].min()) if len(by_date) else 0.0,
        "target_weight_sum_max": float(by_date["target_weight_sum"].max()) if len(by_date) else 0.0,
        "target_weight_sum_mean": float(by_date["target_weight_sum"].mean()) if len(by_date) else 0.0,
        "min_rows_per_date": int(by_date["rows"].min()) if len(by_date) else 0,
        "max_rows_per_date": int(by_date["rows"].max()) if len(by_date) else 0,
        "forbidden_path": str(forbidden_path.resolve()) if forbidden_path else None,
        "forbidden_hits": int(forbidden_mask.sum()),
        "forbidden_hit_symbols": sorted(set(frame.loc[forbidden_mask, "symbol"])),
        "lot_violations": len(lot_bad),
        "min_lot_violations": len(min_lot_bad),
        "low_weight_dates_lt_20pct": int(len(low_weight)),
        "low_weight_date_sample": [
            {
                "trade_date": str(row.trade_date.date()),
                "rows": int(row.rows),
                "target_weight_sum": float(row.target_weight_sum),
            }
            for row in low_weight.head(20).itertuples(index=False)
        ],
        "passed": bool(
            int(forbidden_mask.sum()) == 0
            and not lot_bad
            and not min_lot_bad
            and len(frame) > 0
        ),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    by_date_path = output_path.with_suffix(".by_date.csv")
    by_date.to_csv(by_date_path, index=False, encoding="utf-8-sig")
    report["by_date_csv"] = str(by_date_path.resolve())
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--forbidden-symbols", default="")
    args = parser.parse_args()
    report = audit(
        Path(args.target_csv).resolve(),
        Path(args.output).resolve(),
        Path(args.forbidden_symbols).resolve() if args.forbidden_symbols else None,
    )
    print(
        "[gm target audit] "
        f"passed={report['passed']} rows={report['rows']} dates={report['dates']} "
        f"symbols={report['symbols']} forbidden={report['forbidden_hits']} "
        f"weight={report['target_weight_sum_min']:.2%}-{report['target_weight_sum_max']:.2%} "
        f"low_weight_dates={report['low_weight_dates_lt_20pct']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
