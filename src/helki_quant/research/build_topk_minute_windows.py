from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MULTI_LAYER = HERE.parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
RAW_MINUTE = DATA / "A_Stock_1min"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_minute_staging import MinuteSource, read_one, source_label  # noqa: E402


def normalize_symbol(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith(("sh", "sz")):
        return text[:2] + text[-6:]
    code = text.replace(".", "")[-6:]
    return ("sh" if code.startswith("68") else "sz") + code


def symbol_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem.lower()
    if len(stem) < 8:
        return None
    prefix = stem[:2]
    code = stem[2:8]
    if prefix not in {"sh", "sz"} or not code.isdigit():
        return None
    return prefix + code


def index_paths(symbols: set[str], start_year: int, end_year: int) -> dict[str, list[MinuteSource]]:
    out = {symbol: [] for symbol in symbols}
    allowed = {f"{year}_1min" for year in range(start_year, end_year + 1)}
    scanned_dirs = 0
    for dirpath, _, filenames in os.walk(RAW_MINUTE):
        parts = set(Path(dirpath).parts)
        if not parts.intersection(allowed):
            continue
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue
            symbol = symbol_from_filename(filename)
            if symbol in out:
                out[symbol].append(Path(dirpath) / filename)
        for filename in filenames:
            if not filename.lower().endswith(".zip"):
                continue
            zip_path = Path(dirpath) / filename
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    for member in archive.namelist():
                        symbol = symbol_from_filename(Path(member).name)
                        if symbol in out:
                            out[symbol].append((zip_path, member))
            except zipfile.BadZipFile:
                print(f"[minute windows build] bad zip skipped: {zip_path}", flush=True)
        scanned_dirs += 1
        if scanned_dirs % 100 == 0:
            found = sum(1 for paths in out.values() if paths)
            print(
                f"[minute windows build] scanned_dirs={scanned_dirs} "
                f"symbols_with_files={found}/{len(symbols)}",
                flush=True,
            )
    return {symbol: sorted(set(paths), key=source_label) for symbol, paths in out.items()}


def window_vwap(day: pd.DataFrame, start_minute: int, end_minute: int) -> float:
    mask = (day["minute_of_day"] >= start_minute) & (day["minute_of_day"] <= end_minute)
    if not mask.any():
        return np.nan
    sub = day.loc[mask]
    volume = sub["volume"].to_numpy(dtype=float)
    amount = sub["amount"].to_numpy(dtype=float)
    vol_sum = float(np.nansum(volume))
    if vol_sum <= 0:
        return np.nan
    return float(np.nansum(amount) / (vol_sum + 1e-12))


def build_one(symbol: str, paths: list[MinuteSource], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for path in paths:
        frame = read_one(path)
        frame = frame[frame["date"].dt.normalize().between(start, end)].copy()
        if not frame.empty:
            parts.append(frame)
    if not parts:
        return pd.DataFrame()
    frame = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    frame["trade_date"] = frame["date"].dt.normalize()
    frame["minute_of_day"] = frame["date"].dt.hour * 60 + frame["date"].dt.minute
    rows = []
    for trade_date, day in frame.groupby("trade_date", sort=True):
        open_vwap = window_vwap(day, 9 * 60 + 31, 9 * 60 + 35)
        close_vwap = window_vwap(day, 14 * 60 + 45, 14 * 60 + 50)
        last_close = float(day.sort_values("minute_of_day")["close"].iloc[-1])
        if np.isfinite(last_close) and last_close > 0:
            if np.isfinite(open_vwap) and open_vwap > last_close * 10:
                open_vwap /= 100.0
            if np.isfinite(close_vwap) and close_vwap > last_close * 10:
                close_vwap /= 100.0
        rows.append(
            {
                "trade_date": trade_date,
                "instrument": symbol.upper(),
                "open_exec": open_vwap,
                "close_exec": close_vwap,
                "mark_close": last_close,
            }
        )
    return pd.DataFrame(rows)


def build(pool_file: Path, output_path: Path, report_path: Path, start: str, end: str) -> dict:
    symbols = {
        normalize_symbol(line)
        for line in pool_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    paths_by_symbol = index_paths(symbols, start_ts.year, end_ts.year)
    rows = []
    details = []
    for pos, symbol in enumerate(sorted(symbols), start=1):
        frame = build_one(symbol, paths_by_symbol.get(symbol, []), start_ts, end_ts)
        if not frame.empty:
            rows.append(frame)
        details.append(
            {
                "instrument": symbol,
                "raw_files": len(paths_by_symbol.get(symbol, [])),
                "window_days": int(len(frame)),
                "first": str(frame["trade_date"].min().date()) if not frame.empty else None,
                "last": str(frame["trade_date"].max().date()) if not frame.empty else None,
            }
        )
        if pos % 25 == 0:
            print(
                f"[minute windows build] {pos}/{len(symbols)} {symbol} days={len(frame)}",
                flush=True,
            )
    if not rows:
        raise ValueError("No minute windows were built")
    out = pd.concat(rows, ignore_index=True).sort_values(["instrument", "trade_date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    report = {
        "status": "topk_minute_execution_windows_built",
        "pool_file": str(pool_file),
        "output": str(output_path),
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "instruments_requested": len(symbols),
        "instruments_with_windows": int(out["instrument"].nunique()),
        "rows": int(len(out)),
        "details": details,
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-file", default=str(HERE / "outputs" / "topk_minute_pool.txt"))
    parser.add_argument("--output", default=str(DATA / "_research_topk_minute_windows_2025_2026.csv"))
    parser.add_argument("--report", default=str(HERE / "outputs" / "topk_minute_windows_build_report.json"))
    parser.add_argument("--start", default="2025-01-03")
    parser.add_argument("--end", default="2026-04-03")
    args = parser.parse_args()
    report = build(
        Path(args.pool_file).resolve(),
        Path(args.output).resolve(),
        Path(args.report).resolve(),
        args.start,
        args.end,
    )
    print(
        f"[minute windows build] instruments={report['instruments_with_windows']}/"
        f"{report['instruments_requested']} rows={report['rows']} output={report['output']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
