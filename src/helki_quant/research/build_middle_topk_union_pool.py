from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_topk_minute_windows import normalize_symbol
from evaluate_daily_topk_grid import load_middle_predictions
from export_c_baseline_production_logs import load_forbidden_instruments
from minute_mapped_topk_replay import MappedProfile, MappedReplayConfig, prepare_daily_frame
from universe import load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_FORBIDDEN = DATA / "forbidden_st_symbols_20260605.csv"


def _write_pool(path: Path, instruments: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [normalize_symbol(instrument) for instrument in sorted(instruments)]
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def _load_existing_symbols(path: Path | None) -> set[str]:
    if path is None:
        return set()
    frame = pd.read_csv(path, usecols=["instrument"])
    return {
        normalize_symbol(value).upper()
        for value in frame["instrument"].dropna()
        if str(value).strip()
    }


def build_pool(
    middle_path: Path,
    output_path: Path,
    missing_output_path: Path,
    report_path: Path,
    start: str,
    end: str,
    top_k: int,
    forbidden_path: Path | None,
    existing_windows_path: Path | None,
    raw_daily_dir: Path | None = None,
    price_start: str | None = None,
    price_end: str | None = None,
    min_avg_amount: float = 100_000_000.0,
    min_listing_days: int = 250,
    liquidity_window: int = 20,
) -> dict[str, object]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError("start must be on or before end")

    frame = load_middle_predictions(middle_path)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].map(normalize_symbol).str.upper()
    frame["middle"] = pd.to_numeric(frame["middle"], errors="coerce")
    frame = frame[
        frame["datetime"].between(start_ts, end_ts)
        & frame["middle"].map(np.isfinite)
    ].copy()
    if frame.empty:
        raise ValueError("No finite middle predictions in the requested date range")

    key_columns = ["datetime", "instrument"]
    duplicate_rows = int(frame.duplicated(key_columns, keep=False).sum())
    conflicting = (
        frame.groupby(key_columns, sort=False)["middle"].nunique(dropna=False).gt(1)
    )
    if conflicting.any():
        samples = [
            {"datetime": str(key[0].date()), "instrument": key[1]}
            for key in conflicting[conflicting].index[:10]
        ]
        raise ValueError(f"Conflicting duplicate middle predictions: {samples}")
    frame = frame.drop_duplicates(key_columns, keep="last")

    forbidden = load_forbidden_instruments(forbidden_path)
    forbidden_mask = frame["instrument"].isin(forbidden)
    forbidden_rows_removed = int(forbidden_mask.sum())
    forbidden_instruments_removed = sorted(set(frame.loc[forbidden_mask, "instrument"]))
    frame = frame.loc[~forbidden_mask].copy()
    if frame.empty:
        raise ValueError("No predictions remain after forbidden-symbol filtering")

    selection_date_column = "datetime"
    eligibility_mode = "finite_middle_only"
    execution_start: str | None = None
    execution_end: str | None = None
    if raw_daily_dir is not None:
        if not price_start or not price_end:
            raise ValueError("price_start and price_end are required with raw_daily_dir")
        prices = load_price_panel(
            raw_daily_dir,
            frame["instrument"].drop_duplicates().tolist(),
            start=price_start,
            end=price_end,
        )
        profile = MappedProfile(
            "minute_pool_eligibility",
            top_k,
            1,
            1.0,
            min_avg_amount,
            1.0,
        )
        cfg = MappedReplayConfig(
            min_listing_days=min_listing_days,
            liquidity_window=liquidity_window,
            buffer_multiple=1,
        )
        prepared = prepare_daily_frame(frame, prices, profile, cfg)
        prepared = prepared[prepared["eligible"].astype("boolean").fillna(False)].copy()
        if prepared.empty:
            raise ValueError("No predictions remain after production eligibility filtering")
        frame = prepared
        selection_date_column = "trade_date"
        eligibility_mode = "production_pit_eligibility_next_trade_date"
        execution_start = str(pd.Timestamp(frame["trade_date"].min()).date())
        execution_end = str(pd.Timestamp(frame["trade_date"].max()).date())

    daily_rows: list[dict[str, object]] = []
    selected_parts: list[pd.DataFrame] = []
    for trade_date, day in frame.groupby(selection_date_column, sort=True):
        ranked = day.sort_values(
            ["middle", "instrument"],
            ascending=[False, True],
            kind="mergesort",
        ).head(top_k)
        selected_part = ranked[["instrument", "middle"]].copy()
        selected_part["pool_date"] = pd.Timestamp(trade_date).normalize()
        selected_parts.append(selected_part[["pool_date", "instrument", "middle"]])
        daily_rows.append(
            {
                "trade_date": str(pd.Timestamp(trade_date).date()),
                "eligible_count": int(len(day)),
                "selected_count": int(len(ranked)),
                "cutoff_middle": float(ranked["middle"].iloc[-1]) if len(ranked) else None,
            }
        )

    selected = pd.concat(selected_parts, ignore_index=True)
    union = set(selected["instrument"])
    existing = _load_existing_symbols(existing_windows_path)
    missing = union - existing
    _write_pool(output_path, union)
    _write_pool(missing_output_path, missing)

    report: dict[str, object] = {
        "status": "middle_daily_topk_union_minute_pool_research_only",
        "middle_prediction": str(middle_path.resolve()),
        "window": {"start": str(start_ts.date()), "end": str(end_ts.date())},
        "selection_date_semantics": selection_date_column,
        "eligibility_mode": eligibility_mode,
        "execution_window": {"start": execution_start, "end": execution_end},
        "top_k": int(top_k),
        "prediction_dates": int(selected["pool_date"].nunique()),
        "prediction_rows_after_filter": int(len(frame)),
        "duplicate_prediction_rows_deduplicated": duplicate_rows,
        "forbidden_symbols": str(forbidden_path.resolve()) if forbidden_path else None,
        "forbidden_instruments_loaded": int(len(forbidden)),
        "forbidden_prediction_rows_removed": forbidden_rows_removed,
        "forbidden_instruments_removed": forbidden_instruments_removed,
        "raw_daily_dir": str(raw_daily_dir.resolve()) if raw_daily_dir else None,
        "price_window": {"start": price_start, "end": price_end},
        "eligibility": {
            "min_avg_amount": float(min_avg_amount),
            "min_listing_days": int(min_listing_days),
            "liquidity_window": int(liquidity_window),
        },
        "selected_rows": int(len(selected)),
        "union_instruments": int(len(union)),
        "existing_windows": str(existing_windows_path.resolve()) if existing_windows_path else None,
        "existing_window_instruments": int(len(existing)),
        "union_already_covered": int(len(union & existing)),
        "union_missing_windows": int(len(missing)),
        "output": str(output_path.resolve()),
        "missing_output": str(missing_output_path.resolve()),
        "daily": daily_rows,
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--middle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--missing-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--top-k", type=int, default=600)
    parser.add_argument("--forbidden-symbols", default=str(DEFAULT_FORBIDDEN))
    parser.add_argument("--existing-windows", default=None)
    parser.add_argument("--raw-daily-dir", default=None)
    parser.add_argument("--price-start", default=None)
    parser.add_argument("--price-end", default=None)
    parser.add_argument("--min-avg-amount", type=float, default=100_000_000.0)
    parser.add_argument("--min-listing-days", type=int, default=250)
    parser.add_argument("--liquidity-window", type=int, default=20)
    args = parser.parse_args()
    report = build_pool(
        Path(args.middle).resolve(),
        Path(args.output).resolve(),
        Path(args.missing_output).resolve(),
        Path(args.report).resolve(),
        args.start,
        args.end,
        args.top_k,
        Path(args.forbidden_symbols).resolve() if args.forbidden_symbols else None,
        Path(args.existing_windows).resolve() if args.existing_windows else None,
        Path(args.raw_daily_dir).resolve() if args.raw_daily_dir else None,
        args.price_start,
        args.price_end,
        args.min_avg_amount,
        args.min_listing_days,
        args.liquidity_window,
    )
    print(
        "[middle top-k union] "
        f"dates={report['prediction_dates']} union={report['union_instruments']} "
        f"covered={report['union_already_covered']} missing={report['union_missing_windows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
