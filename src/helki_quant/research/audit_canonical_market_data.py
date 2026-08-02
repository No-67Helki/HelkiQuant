from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .data_sources.rqdata_source import local_symbol
except ImportError:
    from data_sources.rqdata_source import local_symbol


def read_symbol_file(path: Path) -> set[str]:
    return {
        local_symbol(line.strip().split()[0])
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }


def build_report(
    *,
    manifest_path: Path,
    pit_state_path: Path,
    active_instruments_path: Path,
    pit_instruments_path: Path | None = None,
    target_symbols_path: Path,
    cutoff: pd.Timestamp,
    min_sessions: int,
    expected_bars_per_session: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    daily = manifest["frequencies"]["daily"]
    minute = manifest["frequencies"]["minute"]

    pit = pd.read_csv(pit_state_path, parse_dates=["date"])
    pit = pit[pit["date"] > cutoff].copy()
    sessions = pd.DatetimeIndex(sorted(pit["date"].dropna().unique()))
    active_frame = pd.read_csv(active_instruments_path, dtype=str)
    if (
        len(sessions)
        and {"listed_date", "de_listed_date"}.issubset(active_frame.columns)
    ):
        active_listed = pd.to_datetime(active_frame["listed_date"], errors="coerce")
        active_delisted = pd.to_datetime(
            active_frame["de_listed_date"].replace("0000-00-00", pd.NA),
            errors="coerce",
        )
        active_frame = active_frame.loc[
            active_listed.le(sessions.max())
            & (active_delisted.isna() | active_delisted.ge(sessions.max()))
        ]
    active_symbols = {
        local_symbol(value) for value in active_frame["instrument"].dropna()
    }
    pit_instruments = pd.read_csv(
        pit_instruments_path or active_instruments_path,
        dtype=str,
    )
    pit_instruments["instrument"] = pit_instruments["instrument"].map(
        local_symbol
    )
    if {"listed_date", "de_listed_date"}.issubset(pit_instruments.columns):
        pit_instruments["listed_date"] = pd.to_datetime(
            pit_instruments["listed_date"], errors="coerce"
        )
        pit_instruments["de_listed_date"] = pd.to_datetime(
            pit_instruments["de_listed_date"].replace("0000-00-00", pd.NA),
            errors="coerce",
        )
    else:
        pit_instruments["listed_date"] = pd.Timestamp.min
        pit_instruments["de_listed_date"] = pd.NaT
    pit_instruments = pit_instruments.drop_duplicates("instrument", keep="last")
    pit_by_date = {
        pd.Timestamp(date): set(part["instrument"].map(local_symbol))
        for date, part in pit.groupby("date", sort=True)
    }
    missing_pit_by_date: dict[str, list[str]] = {}
    expected_pit_counts: list[int] = []
    observed_pit_counts: list[int] = []
    pit_universe_symbols: set[str] = set()
    for session in sessions:
        expected = set(
            pit_instruments.loc[
                pit_instruments["listed_date"].le(session)
                & (
                    pit_instruments["de_listed_date"].isna()
                    | pit_instruments["de_listed_date"].ge(session)
                ),
                "instrument",
            ]
        )
        observed = pit_by_date.get(pd.Timestamp(session), set())
        missing = sorted(expected - observed)
        pit_universe_symbols.update(expected)
        expected_pit_counts.append(len(expected))
        observed_pit_counts.append(len(observed & expected))
        if missing:
            missing_pit_by_date[pd.Timestamp(session).strftime("%Y-%m-%d")] = missing
    target_symbols = read_symbol_file(target_symbols_path)
    minute_files = minute.get("files", [])
    minute_symbols = {local_symbol(row["instrument"]) for row in minute_files}
    missing_targets = sorted(target_symbols - minute_symbols)
    missing_active_targets = sorted(set(missing_targets) & active_symbols)
    inactive_targets = sorted(set(missing_targets) - active_symbols)

    expected_rows = len(sessions) * expected_bars_per_session
    bad_minute_rows = sorted(
        {
            local_symbol(row["instrument"]): int(row["rows"])
            for row in minute_files
            if int(row["rows"]) != expected_rows
        }.items()
    )
    daily_files = daily.get("files", [])
    daily_primary_symbols = {
        local_symbol(row["instrument"])
        for row in daily_files
        if int(row.get("rqdata_rows") or 0) > 0
    }
    missing_pit_primary_symbols = sorted(
        pit_universe_symbols - daily_primary_symbols
    )
    scale_alignment_failures = sorted(
        {
            local_symbol(row["instrument"]): str(
                row.get("scale_alignment_status") or "missing_evidence"
            )
            for row in daily_files
            if local_symbol(row["instrument"]) in pit_universe_symbols
            and int(row.get("rqdata_rows") or 0) > 0
            and int(row.get("fallback_rows") or 0) > 0
            and row.get("scale_alignment_status") != "aligned"
        }.items()
    )
    daily_complete = bool(
        int(daily["written_symbols"]) == int(daily["requested_symbols"])
        and not missing_pit_primary_symbols
        and not scale_alignment_failures
    )
    pit_complete = bool(
        len(sessions)
        and not missing_pit_by_date
    )
    minute_complete = not missing_active_targets and not bad_minute_rows
    data_integrity_passed = daily_complete and pit_complete and minute_complete
    remaining_sessions = max(0, min_sessions - len(sessions))

    return {
        "status": "canonical_market_data_readiness",
        "passed": data_integrity_passed and remaining_sessions == 0,
        "data_integrity_passed": data_integrity_passed,
        "promotion_window_ready": remaining_sessions == 0,
        "profile_frozen": True,
        "return_metrics_evaluated": False,
        "cutoff_excluded": cutoff.strftime("%Y-%m-%d"),
        "holdout": {
            "first_session": sessions.min().strftime("%Y-%m-%d") if len(sessions) else None,
            "last_session": sessions.max().strftime("%Y-%m-%d") if len(sessions) else None,
            "sessions": len(sessions),
            "required_sessions": min_sessions,
            "remaining_sessions": remaining_sessions,
        },
        "daily": {
            "passed": daily_complete,
            "requested_symbols": int(daily["requested_symbols"]),
            "written_symbols": int(daily["written_symbols"]),
            "rqdata_symbols": int(daily["rqdata_symbols"]),
            "active_symbols": len(active_symbols),
            "pit_universe_symbols": len(pit_universe_symbols),
            "missing_pit_primary_symbols": missing_pit_primary_symbols,
            "scale_alignment_failures": [
                {"instrument": symbol, "status": status}
                for symbol, status in scale_alignment_failures
            ],
        },
        "pit_market_state": {
            "passed": pit_complete,
            "rows": int(len(pit)),
            "expected_min_symbols_per_session": (
                min(expected_pit_counts) if expected_pit_counts else 0
            ),
            "expected_max_symbols_per_session": (
                max(expected_pit_counts) if expected_pit_counts else 0
            ),
            "observed_min_symbols_per_session": (
                min(observed_pit_counts) if observed_pit_counts else 0
            ),
            "observed_max_symbols_per_session": (
                max(observed_pit_counts) if observed_pit_counts else 0
            ),
            "missing_symbol_dates": int(
                sum(len(values) for values in missing_pit_by_date.values())
            ),
            "missing_examples": [
                {"date": date, "instruments": values[:20]}
                for date, values in list(missing_pit_by_date.items())[:10]
            ],
        },
        "minute": {
            "passed": minute_complete,
            "source_precedence": minute.get("source_precedence", []),
            "target_symbols": len(target_symbols),
            "written_symbols": len(minute_symbols),
            "expected_rows_per_symbol": expected_rows,
            "missing_active_targets": missing_active_targets,
            "inactive_targets_without_rows": inactive_targets,
            "bad_row_counts": [
                {"instrument": symbol, "rows": rows}
                for symbol, rows in bad_minute_rows
            ],
        },
        "next_gate": (
            "frozen_profile_untouched_replay"
            if data_integrity_passed and remaining_sessions == 0
            else "continue_data_sync_without_profile_retuning"
        ),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Audit a versioned canonical market-data build")
    root.add_argument("--manifest", type=Path, required=True)
    root.add_argument("--pit-state", type=Path, required=True)
    root.add_argument("--active-instruments", type=Path, required=True)
    root.add_argument("--pit-instruments", type=Path)
    root.add_argument("--target-symbols", type=Path, required=True)
    root.add_argument("--cutoff", default="2026-06-05")
    root.add_argument("--min-sessions", type=int, default=60)
    root.add_argument("--expected-bars-per-session", type=int, default=240)
    root.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    report = build_report(
        manifest_path=args.manifest.resolve(),
        pit_state_path=args.pit_state.resolve(),
        active_instruments_path=args.active_instruments.resolve(),
        pit_instruments_path=(
            args.pit_instruments.resolve() if args.pit_instruments else None
        ),
        target_symbols_path=args.target_symbols.resolve(),
        cutoff=pd.Timestamp(args.cutoff),
        min_sessions=args.min_sessions,
        expected_bars_per_session=args.expected_bars_per_session,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[canonical audit] integrity={report['data_integrity_passed']} "
        f"sessions={report['holdout']['sessions']}/{report['holdout']['required_sessions']} "
        f"promotion_ready={report['passed']} output={args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
