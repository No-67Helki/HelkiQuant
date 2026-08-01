from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from augment_inner_exec_labels import read_instruments, window_vwap


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"


WINDOWS = {
    "open_0931_0935": (9 * 60 + 31, 9 * 60 + 35),
    "am_sell_1000_1005": (10 * 60, 10 * 60 + 5),
    "am_buy_1030_1035": (10 * 60 + 30, 10 * 60 + 35),
    "pm_buy_1400_1405": (14 * 60, 14 * 60 + 5),
    "close_1445_1450": (14 * 60 + 45, 14 * 60 + 50),
}


def build_cache(stage_dir: Path, pool_file: Path, output_csv: Path, report_path: Path) -> dict:
    instruments = read_instruments(pool_file)
    rows = []
    details = []
    for pos, inst in enumerate(instruments, start=1):
        path = stage_dir / f"{inst}.csv"
        if not path.exists():
            details.append({"instrument": inst.upper(), "status": "missing_csv"})
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["trade_date"] = frame["date"].dt.normalize()
        frame["minute_of_day"] = frame["date"].dt.hour * 60 + frame["date"].dt.minute
        inst_rows = []
        for trade_date, day in frame.groupby("trade_date", sort=True):
            row = {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "instrument": inst.upper(),
            }
            for name, (start, end) in WINDOWS.items():
                row[name] = window_vwap(day, start, end)
            inst_rows.append(row)
        rows.extend(inst_rows)
        details.append(
            {
                "instrument": inst.upper(),
                "status": "ok",
                "rows": len(inst_rows),
                "first": inst_rows[0]["trade_date"] if inst_rows else None,
                "last": inst_rows[-1]["trade_date"] if inst_rows else None,
            }
        )
        print(f"[t0 windows] {pos}/{len(instruments)} {inst} rows={len(inst_rows)}", flush=True)
    out = pd.DataFrame(rows)
    if not out.empty:
        price_cols = list(WINDOWS)
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=price_cols, how="all")
        out = out.sort_values(["trade_date", "instrument"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    report = {
        "status": "inner_t0_window_cache_built",
        "stage_dir": str(stage_dir.resolve()),
        "pool_file": str(pool_file.resolve()),
        "output_csv": str(output_csv.resolve()),
        "windows": {name: f"{start//60:02d}:{start%60:02d}-{end//60:02d}:{end%60:02d}" for name, (start, end) in WINDOWS.items()},
        "instruments": len(instruments),
        "rows": int(len(out)),
        "details": details,
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", default=str(DATA / "_top150_stable_1min_pool_csv_20260605"))
    parser.add_argument("--pool-file", default=str(DATA / "inner_pool_top150_stable_20260605.txt"))
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = build_cache(
        Path(args.stage_dir).resolve(),
        Path(args.pool_file).resolve(),
        Path(args.output_csv).resolve(),
        Path(args.report).resolve(),
    )
    print(f"[t0 windows] rows={report['rows']} -> {report['output_csv']}")


if __name__ == "__main__":
    main()
