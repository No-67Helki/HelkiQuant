from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_topk_minute_windows import normalize_symbol


KEY_COLUMNS = ["trade_date", "instrument"]
VALUE_COLUMNS = ["open_exec", "close_exec", "mark_close"]
REQUIRED_COLUMNS = set(KEY_COLUMNS + VALUE_COLUMNS)


def load_windows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = frame[KEY_COLUMNS + VALUE_COLUMNS].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["instrument"] = out["instrument"].map(normalize_symbol).str.upper()
    for column in VALUE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{path} contains invalid minute-window keys")
    out["source"] = str(path.resolve())
    return out


def _resolve_group(group: pd.DataFrame, *, atol: float, rtol: float) -> pd.Series:
    resolved: dict[str, object] = {
        "trade_date": group["trade_date"].iloc[0],
        "instrument": group["instrument"].iloc[0],
    }
    for column in VALUE_COLUMNS:
        values = group[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            resolved[column] = np.nan
            continue
        scale = max(1.0, float(np.max(np.abs(finite))))
        if float(np.max(finite) - np.min(finite)) > atol + rtol * scale:
            sources = sorted(set(group["source"].astype(str)))
            raise ValueError(
                "Conflicting minute windows for "
                f"{resolved['trade_date']:%Y-%m-%d}/{resolved['instrument']} "
                f"column={column} values={finite.tolist()} sources={sources}"
            )
        resolved[column] = float(finite[-1])
    return pd.Series(resolved)


def assemble(
    input_paths: list[Path],
    output_path: Path,
    report_path: Path,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> dict[str, object]:
    if not input_paths:
        raise ValueError("At least one input window file is required")
    frames = [load_windows(path) for path in input_paths]
    source_rows = [
        {
            "path": str(path.resolve()),
            "rows": int(len(frame)),
            "instruments": int(frame["instrument"].nunique()),
            "first": str(frame["trade_date"].min().date()) if len(frame) else None,
            "last": str(frame["trade_date"].max().date()) if len(frame) else None,
        }
        for path, frame in zip(input_paths, frames)
    ]
    combined = pd.concat(frames, ignore_index=True)
    duplicate_mask = combined.duplicated(KEY_COLUMNS, keep=False)
    overlap_rows = int(duplicate_mask.sum())
    overlap_keys = int(combined.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates().shape[0])

    unique = combined.loc[~duplicate_mask, KEY_COLUMNS + VALUE_COLUMNS].copy()
    if overlap_rows:
        resolved = (
            combined.loc[duplicate_mask]
            .groupby(KEY_COLUMNS, sort=False, as_index=False)
            .apply(_resolve_group, atol=atol, rtol=rtol)
            .reset_index(drop=True)
        )
        unique = pd.concat([unique, resolved], ignore_index=True)
    unique = unique.sort_values(KEY_COLUMNS).reset_index(drop=True)
    if unique.duplicated(KEY_COLUMNS).any():
        raise AssertionError("Minute-window assembly left duplicate keys")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique.to_csv(output_path, index=False)
    report: dict[str, object] = {
        "status": "minute_execution_windows_assembled_conflict_checked",
        "sources": source_rows,
        "input_rows": int(len(combined)),
        "overlap_rows": overlap_rows,
        "overlap_keys": overlap_keys,
        "conflicting_keys": 0,
        "output": str(output_path.resolve()),
        "output_rows": int(len(unique)),
        "output_instruments": int(unique["instrument"].nunique()),
        "first": str(unique["trade_date"].min().date()) if len(unique) else None,
        "last": str(unique["trade_date"].max().date()) if len(unique) else None,
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-8)
    args = parser.parse_args()
    report = assemble(
        [Path(value).resolve() for value in args.input],
        Path(args.output).resolve(),
        Path(args.report).resolve(),
        atol=args.atol,
        rtol=args.rtol,
    )
    print(
        "[minute windows assemble] "
        f"inputs={len(report['sources'])} rows={report['output_rows']} "
        f"instruments={report['output_instruments']} overlaps={report['overlap_keys']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
