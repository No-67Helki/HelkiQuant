from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_calendar(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "trade_date" not in frame.columns:
        raise ValueError(f"trade_date column not found in {path}")
    return (
        pd.to_datetime(frame["trade_date"])
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


def next_trade_date(date: str, calendar: list[str], shift: int) -> str:
    if shift == 0:
        return date
    index = 0
    while index < len(calendar) and calendar[index] <= date:
        index += 1
    shifted = index + shift - 1
    if shifted >= len(calendar):
        raise ValueError(f"no shifted trading date for {date} with shift={shift}")
    return calendar[shifted]


def shift_target_dates(input_csv: Path, output_csv: Path, calendar_csv: Path, shift: int) -> dict:
    target = pd.read_csv(input_csv, dtype={"symbol": str, "instrument": str})
    if "trade_date" not in target.columns:
        raise ValueError(f"trade_date column not found in {input_csv}")
    calendar = load_calendar(calendar_csv)
    original_dates = pd.to_datetime(target["trade_date"]).dt.strftime("%Y-%m-%d")
    mapping = {date: next_trade_date(date, calendar, shift) for date in sorted(original_dates.unique())}
    target["signal_date"] = original_dates
    target["trade_date"] = original_dates.map(mapping)
    columns = ["trade_date", "signal_date"] + [
        column for column in target.columns if column not in {"trade_date", "signal_date"}
    ]
    target = target[columns].sort_values(["trade_date", "rank", "symbol"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    target.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return {
        "status": "gm_target_dates_shifted",
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "calendar_csv": str(calendar_csv),
        "shift_trading_days": shift,
        "rows": int(len(target)),
        "dates": int(target["trade_date"].nunique()),
        "date_start": str(target["trade_date"].min()) if len(target) else None,
        "date_end": str(target["trade_date"].max()) if len(target) else None,
        "date_mapping": mapping,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calendar", required=True, type=Path)
    parser.add_argument("--shift-trading-days", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    result = shift_target_dates(
        args.input.resolve(),
        args.output.resolve(),
        args.calendar.resolve(),
        args.shift_trading_days,
    )
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[shift gm target dates] "
        f"rows={result['rows']} dates={result['dates']} "
        f"range={result['date_start']}..{result['date_end']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
