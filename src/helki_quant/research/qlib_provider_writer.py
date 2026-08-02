from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _symbol_from_path(path: Path) -> str:
    symbol = path.stem.strip().lower()
    if len(symbol) != 8 or not symbol.startswith(("sh", "sz")):
        raise ValueError(f"unsupported provider symbol filename: {path.name}")
    if not symbol[2:].isdigit():
        raise ValueError(f"unsupported provider symbol filename: {path.name}")
    return symbol


def _read_dates(path: Path, date_field: str) -> tuple[str, pd.DatetimeIndex]:
    frame = pd.read_csv(path, usecols=[date_field])
    dates = pd.DatetimeIndex(
        pd.to_datetime(frame[date_field], errors="coerce")
    ).dropna().drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError(f"source contains no valid dates: {path}")
    return _symbol_from_path(path), dates


def _write_symbol_features(
    path: Path,
    *,
    output_root: Path,
    calendar: pd.DatetimeIndex,
    calendar_index: dict[pd.Timestamp, int],
    frequency: str,
    date_field: str,
    symbol_field: str,
) -> dict[str, Any]:
    symbol = _symbol_from_path(path)
    frame = pd.read_csv(path)
    frame[date_field] = pd.to_datetime(frame[date_field], errors="coerce")
    frame = (
        frame.dropna(subset=[date_field])
        .drop_duplicates(date_field, keep="last")
        .sort_values(date_field)
        .set_index(date_field)
    )
    if frame.empty:
        raise ValueError(f"source contains no valid rows: {path}")
    start = pd.Timestamp(frame.index.min())
    end = pd.Timestamp(frame.index.max())
    start_index = calendar_index[start]
    end_index = calendar_index[end]
    aligned = frame.reindex(calendar[start_index : end_index + 1])
    fields = [
        column
        for column in aligned.columns
        if column != symbol_field
        and pd.api.types.is_numeric_dtype(aligned[column])
    ]
    if not fields:
        raise ValueError(f"source contains no numeric features: {path}")
    feature_dir = output_root / "features" / symbol
    feature_dir.mkdir(parents=True, exist_ok=True)
    for field in fields:
        values = pd.to_numeric(aligned[field], errors="coerce").to_numpy(
            dtype=np.float32
        )
        payload = np.concatenate(
            [np.asarray([start_index], dtype=np.float32), values]
        ).astype("<f4", copy=False)
        destination = feature_dir / f"{field.lower()}.{frequency}.bin"
        pending = destination.with_suffix(destination.suffix + ".pending")
        payload.tofile(pending)
        os.replace(pending, destination)
    return {
        "instrument": symbol,
        "start": start,
        "end": end,
        "rows": int(len(frame)),
        "fields": sorted(fields),
    }


def build_qlib_provider(
    source_dir: Path,
    output_dir: Path,
    *,
    frequency: str = "day",
    date_field: str = "date",
    symbol_field: str = "symbol",
    max_workers: int = 4,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    files = sorted(source_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"no CSV files found: {source_dir}")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    pending_root = output_dir.with_name(output_dir.name + ".pending")
    if output_dir.exists() or pending_root.exists():
        raise FileExistsError(
            "refusing to overwrite provider or pending build: "
            f"output={output_dir.exists()} pending={pending_root.exists()}"
        )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            inspections = list(
                executor.map(
                    lambda path: _read_dates(path, date_field),
                    files,
                )
            )
        calendar = pd.DatetimeIndex(
            sorted({date for _, dates in inspections for date in dates})
        )
        if calendar.empty:
            raise ValueError("provider calendar is empty")
        calendar_index = {date: index for index, date in enumerate(calendar)}
        calendar_dir = pending_root / "calendars"
        instrument_dir = pending_root / "instruments"
        calendar_dir.mkdir(parents=True)
        instrument_dir.mkdir(parents=True)
        calendar_format = "%Y-%m-%d" if frequency == "day" else "%Y-%m-%d %H:%M:%S"
        (calendar_dir / f"{frequency}.txt").write_text(
            "\n".join(date.strftime(calendar_format) for date in calendar) + "\n",
            encoding="utf-8",
        )

        arguments = [
            {
                "path": path,
                "output_root": pending_root,
                "calendar": calendar,
                "calendar_index": calendar_index,
                "frequency": frequency,
                "date_field": date_field,
                "symbol_field": symbol_field,
            }
            for path in files
        ]

        def write_one(values: dict[str, Any]) -> dict[str, Any]:
            path = values.pop("path")
            return _write_symbol_features(path, **values)

        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for position, row in enumerate(executor.map(write_one, arguments), start=1):
                rows.append(row)
                if position % 250 == 0 or position == len(files):
                    print(
                        f"[provider build] {position}/{len(files)}",
                        flush=True,
                    )
        instrument_lines = [
            "\t".join(
                (
                    row["instrument"].upper(),
                    row["start"].strftime(calendar_format),
                    row["end"].strftime(calendar_format),
                )
            )
            for row in sorted(rows, key=lambda item: item["instrument"])
        ]
        (instrument_dir / "all.txt").write_text(
            "\n".join(instrument_lines) + "\n",
            encoding="utf-8",
        )
        os.replace(pending_root, output_dir)
    except Exception:
        if pending_root.exists():
            shutil.rmtree(pending_root)
        raise

    return {
        "status": "qlib_compatible_provider_built",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "frequency": frequency,
        "calendar_rows": int(len(calendar)),
        "calendar_start": calendar.min().strftime(calendar_format),
        "calendar_end": calendar.max().strftime(calendar_format),
        "instruments": int(len(rows)),
        "fields": sorted({field for row in rows for field in row["fields"]}),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Build a minimal Qlib-compatible provider from project CSVs"
    )
    root.add_argument("--source-dir", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--frequency", choices=("day", "1min"), default="day")
    root.add_argument("--date-field", default="date")
    root.add_argument("--symbol-field", default="symbol")
    root.add_argument("--max-workers", type=int, default=4)
    return root


def main() -> None:
    args = parser().parse_args()
    report = build_qlib_provider(
        args.source_dir,
        args.output_dir,
        frequency=args.frequency,
        date_field=args.date_field,
        symbol_field=args.symbol_field,
        max_workers=args.max_workers,
    )
    print(
        f"[provider build] complete instruments={report['instruments']} "
        f"calendar={report['calendar_rows']} output={report['output_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
