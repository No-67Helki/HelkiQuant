from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_topk_minute_windows import build as build_windows
from build_topk_minute_windows import normalize_symbol


def _load_windows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    required = {"trade_date", "instrument", "open_exec", "close_exec", "mark_close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame[list(required)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].map(normalize_symbol).str.upper()
    for column in ("open_exec", "close_exec", "mark_close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["trade_date", "instrument"]].isna().any().any():
        raise ValueError(f"{path} contains invalid window keys")
    return frame.sort_values(["instrument", "trade_date"]).reset_index(drop=True)


def _target_symbols(path: Path, required_start: str | None) -> tuple[set[str], set[str]]:
    frame = pd.read_csv(path)
    if "instrument" not in frame.columns:
        raise ValueError(f"{path} missing instrument")
    frame["instrument"] = frame["instrument"].map(normalize_symbol).str.upper()
    all_symbols = {
        normalize_symbol(value).upper()
        for value in frame["instrument"].dropna()
        if str(value).strip()
    }
    if required_start is None:
        return all_symbols, all_symbols
    if "trade_date" not in frame.columns:
        raise ValueError(f"{path} missing trade_date required by --required-target-start")
    trade_date = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    required = set(frame.loc[trade_date >= pd.Timestamp(required_start).normalize(), "instrument"])
    return all_symbols, required


def expand(
    targets_path: Path,
    existing_path: Path,
    output_path: Path,
    report_path: Path,
    work_dir: Path,
    start: str,
    end: str,
    required_target_start: str | None,
) -> dict[str, object]:
    existing = _load_windows(existing_path)
    targets, required_targets = _target_symbols(targets_path, required_target_start)
    existing_symbols = set(existing["instrument"])
    missing_symbols = sorted(targets - existing_symbols)

    work_dir.mkdir(parents=True, exist_ok=True)
    pool_path = work_dir / "missing_target_minute_pool.txt"
    supplement_path = work_dir / "missing_target_minute_windows.csv"
    supplement_report_path = work_dir / "missing_target_minute_windows.json"
    pool_path.write_text("\n".join(missing_symbols) + ("\n" if missing_symbols else ""), encoding="utf-8")

    supplement = existing.iloc[0:0].copy()
    supplement_report: dict[str, object] | None = None
    if missing_symbols:
        supplement_report = build_windows(
            pool_path,
            supplement_path,
            supplement_report_path,
            start,
            end,
        )
        supplement = _load_windows(supplement_path)

    combined = pd.concat([existing, supplement], ignore_index=True)
    duplicate_rows = int(combined.duplicated(["trade_date", "instrument"], keep=False).sum())
    combined = (
        combined.drop_duplicates(["trade_date", "instrument"], keep="last")
        .sort_values(["instrument", "trade_date"])
        .reset_index(drop=True)
    )
    combined_symbols = set(combined["instrument"])
    unresolved = sorted(targets - combined_symbols)
    unresolved_required = sorted(required_targets - combined_symbols)
    if unresolved_required:
        raise ValueError(
            "raw minute windows unavailable for required forward target symbols: "
            f"{unresolved_required}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8")
    report: dict[str, object] = {
        "status": "minute_execution_windows_expanded_for_all_selected_targets",
        "targets": str(targets_path.resolve()),
        "existing_windows": str(existing_path.resolve()),
        "output": str(output_path.resolve()),
        "window": {"start": start, "end": end},
        "target_symbols": len(targets),
        "required_target_start": required_target_start,
        "required_target_symbols": len(required_targets),
        "existing_symbols": len(existing_symbols),
        "missing_symbols_requested": len(missing_symbols),
        "missing_symbols": missing_symbols,
        "supplement_symbols_built": int(supplement["instrument"].nunique()),
        "combined_symbols": int(combined["instrument"].nunique()),
        "combined_rows": int(len(combined)),
        "overlap_rows_deduplicated": duplicate_rows,
        "unresolved_target_symbols": unresolved,
        "unresolved_required_target_symbols": unresolved_required,
        "supplement_report": supplement_report,
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--existing-windows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--start", default="2025-01-03")
    parser.add_argument("--end", default="2026-06-05")
    parser.add_argument(
        "--required-target-start",
        default=None,
        help="Fail only when a target on/after this date lacks minute windows.",
    )
    args = parser.parse_args()
    report = expand(
        Path(args.targets).resolve(),
        Path(args.existing_windows).resolve(),
        Path(args.output).resolve(),
        Path(args.report).resolve(),
        Path(args.work_dir).resolve(),
        args.start,
        args.end,
        args.required_target_start,
    )
    print(
        "[minute windows expand] "
        f"targets={report['target_symbols']} missing={report['missing_symbols_requested']} "
        f"supplement={report['supplement_symbols_built']} rows={report['combined_rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
