from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .data_sources.rqdata_source import (
        DEFAULT_CONFIG,
        REPO_ROOT,
        load_config,
        resolve_repo_path,
    )
except ImportError:
    from data_sources.rqdata_source import (
        DEFAULT_CONFIG,
        REPO_ROOT,
        load_config,
        resolve_repo_path,
    )


def log(message: str) -> None:
    print(f"[canonical refresh] {message}", flush=True)


def run_step(name: str, command: list[str]) -> None:
    log(f"START {name}: {subprocess.list2cmdline(command)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    log(f"DONE {name}")


def build_commands(
    *,
    config_path: Path,
    config: dict[str, Any],
    end_date: str,
    target_symbols: Path,
    cutoff: str,
    overlap_start: str,
    history_start: str,
    version_root: Path,
    readiness_output: Path,
    min_sessions: int,
    expected_bars_per_session: int,
) -> list[tuple[str, list[str]]]:
    python = sys.executable
    sync_module = "helki_quant.research.sync_rqdata_market_data"
    materialize_module = "helki_quant.research.materialize_rqdata_canonical"
    audit_module = "helki_quant.research.audit_canonical_market_data"
    post_cutoff_start = (
        pd.Timestamp(cutoff).normalize() + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    metadata_root = resolve_repo_path(config["primary"]["metadata_root"])
    historical_instruments = metadata_root / "instruments_all_history.csv"
    pit_symbols = REPO_ROOT / "outputs" / "rqdata_sync" / (
        f"pit_universe_{pd.Timestamp(post_cutoff_start):%Y%m%d}_"
        f"{pd.Timestamp(end_date):%Y%m%d}.txt"
    )
    pit_state = metadata_root / "pit_market_state.csv"
    manifest = version_root / "MARKET_DATA_MANIFEST.json"
    common_sync = [
        python,
        "-m",
        sync_module,
        "--config",
        str(config_path),
    ]
    common_materialize = [
        python,
        "-m",
        materialize_module,
    ]
    return [
        (
            "historical_universe_sync",
            [
                *common_sync,
                "universe",
                "--output",
                str(historical_instruments),
                "--pit-start",
                post_cutoff_start,
                "--pit-end",
                end_date,
                "--symbols-output",
                str(pit_symbols),
            ],
        ),
        (
            "daily_overlap_sync",
            [
                *common_sync,
                "daily",
                "--start-date",
                overlap_start,
                "--end-date",
                end_date,
                "--symbols-file",
                str(pit_symbols),
            ],
        ),
        (
            "pit_state_sync",
            [
                *common_sync,
                "state",
                "--start-date",
                post_cutoff_start,
                "--end-date",
                end_date,
                "--symbols-file",
                str(pit_symbols),
            ],
        ),
        (
            "minute_holdout_sync",
            [
                *common_sync,
                "minute",
                "--start-date",
                post_cutoff_start,
                "--end-date",
                end_date,
                "--symbols-file",
                str(target_symbols),
            ],
        ),
        (
            "daily_materialization",
            [
                *common_materialize,
                "daily",
                "--config",
                str(config_path),
                "--start-date",
                history_start,
                "--end-date",
                end_date,
                "--output-root",
                str(version_root / "daily_qfq"),
                "--manifest",
                str(manifest),
            ],
        ),
        (
            "minute_materialization",
            [
                *common_materialize,
                "minute",
                "--config",
                str(config_path),
                "--start-date",
                post_cutoff_start,
                "--end-date",
                end_date,
                "--symbols-file",
                str(target_symbols),
                "--output-root",
                str(version_root / "minute_1m"),
                "--manifest",
                str(manifest),
                "--primary-only",
            ],
        ),
        (
            "canonical_readiness_audit",
            [
                python,
                "-m",
                audit_module,
                "--manifest",
                str(manifest),
                "--pit-state",
                str(pit_state),
                "--active-instruments",
                str(historical_instruments),
                "--pit-instruments",
                str(historical_instruments),
                "--target-symbols",
                str(target_symbols),
                "--cutoff",
                cutoff,
                "--min-sessions",
                str(min_sessions),
                "--expected-bars-per-session",
                str(expected_bars_per_session),
                "--output",
                str(readiness_output),
            ],
        ),
    ]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Refresh one versioned canonical dataset without training, "
            "prediction, replay, or profile selection"
        )
    )
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    root.add_argument("--end-date", required=True)
    root.add_argument("--target-symbols", type=Path, required=True)
    root.add_argument("--cutoff", default="2026-06-05")
    root.add_argument(
        "--overlap-start",
        help=(
            "daily scale-calibration start; defaults to 90 calendar days "
            "before the untouched cutoff"
        ),
    )
    root.add_argument("--history-start", default="1990-01-01")
    root.add_argument("--version")
    root.add_argument("--version-root", type=Path)
    root.add_argument("--readiness-output", type=Path)
    root.add_argument("--min-sessions", type=int, default=60)
    root.add_argument("--expected-bars-per-session", type=int, default=240)
    return root


def main() -> None:
    args = parser().parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    end = pd.Timestamp(args.end_date).normalize()
    cutoff = pd.Timestamp(args.cutoff).normalize()
    overlap_start = (
        pd.Timestamp(args.overlap_start).normalize()
        if args.overlap_start
        else cutoff - pd.Timedelta(days=90)
    )
    if not overlap_start < cutoff < end:
        raise ValueError("required date order: overlap_start < cutoff < end_date")
    if args.min_sessions < 60:
        raise ValueError("min_sessions cannot be lower than 60")
    version = args.version or f"v{end:%Y%m%d}_rqaligned"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", version):
        raise ValueError("version may contain only letters, digits, underscores, and hyphens")
    version_root = (
        args.version_root.resolve()
        if args.version_root
        else (REPO_ROOT / "data" / "market_data" / "canonical" / version).resolve()
    )
    readiness_output = (
        args.readiness_output.resolve()
        if args.readiness_output
        else (REPO_ROOT / "outputs" / f"canonical_{version}_readiness.json").resolve()
    )
    target_symbols = args.target_symbols.resolve()
    if not target_symbols.is_file():
        raise FileNotFoundError(f"target symbol file not found: {target_symbols}")
    if version_root.exists():
        raise FileExistsError(
            f"refusing to overwrite canonical version: {version_root}"
        )

    commands = build_commands(
        config_path=config_path,
        config=config,
        end_date=end.strftime("%Y-%m-%d"),
        target_symbols=target_symbols,
        cutoff=cutoff.strftime("%Y-%m-%d"),
        overlap_start=overlap_start.strftime("%Y-%m-%d"),
        history_start=pd.Timestamp(args.history_start).strftime("%Y-%m-%d"),
        version_root=version_root,
        readiness_output=readiness_output,
        min_sessions=args.min_sessions,
        expected_bars_per_session=args.expected_bars_per_session,
    )
    for name, command in commands:
        run_step(name, command)
    report = json.loads(readiness_output.read_text(encoding="utf-8-sig"))
    if report.get("data_integrity_passed") is not True:
        raise RuntimeError(
            f"canonical refresh failed data integrity: {readiness_output}"
        )
    log(
        f"COMPLETE version={version} integrity=true "
        f"sessions={report['holdout']['sessions']}/"
        f"{report['holdout']['required_sessions']} "
        f"promotion_ready={report['promotion_window_ready']}"
    )


if __name__ == "__main__":
    main()
