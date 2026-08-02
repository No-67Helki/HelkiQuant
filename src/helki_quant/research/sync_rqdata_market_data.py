from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from .data_sources.rqdata_source import (
        DEFAULT_CONFIG,
        REPO_ROOT,
        load_config,
        local_symbol,
        read_license,
        resolve_repo_path,
        run_bridge,
        write_symbol_file,
    )
except ImportError:
    from data_sources.rqdata_source import (
        DEFAULT_CONFIG,
        REPO_ROOT,
        load_config,
        local_symbol,
        read_license,
        resolve_repo_path,
        run_bridge,
        write_symbol_file,
    )


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


def log(message: str) -> None:
    print(f"[RQData sync] {message}", flush=True)


def symbols_from_file(path: Path) -> list[str]:
    values = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            values.append(line.strip().split()[0])
    return values


def prepare_symbols(
    config: dict,
    *,
    symbols: str | None,
    symbols_file: Path | None,
    date: str,
    purpose: str,
    generated_root: Path | None = None,
) -> Path:
    if symbols:
        values = symbols.split(",")
    elif symbols_file is not None:
        values = symbols_from_file(symbols_file.resolve())
    else:
        metadata_root = resolve_repo_path(config["primary"]["metadata_root"])
        instruments = metadata_root / f"instruments_{date.replace('-', '')}.csv"
        result = run_bridge(
            config,
            ["instruments", "--date", date, "--output", str(instruments)],
            require_license=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"RQData instrument request failed with code {result.returncode}"
            )
        frame = pd.read_csv(instruments, dtype=str)
        values = frame["instrument"].dropna().tolist()
    normalized = sorted({local_symbol(value) for value in values})
    digest = hashlib.sha256("\n".join(normalized).encode("ascii")).hexdigest()[:12]
    root = generated_root or OUTPUTS / "rqdata_sync"
    generated = root / (
        f"symbols_{purpose}_{date.replace('-', '')}_{digest}.txt"
    )
    return write_symbol_file(generated, normalized)


def doctor(config: dict, output: Path) -> int:
    result = run_bridge(
        config,
        ["doctor", "--output", str(output.resolve())],
        require_license=False,
    )
    payload = json.loads(output.read_text(encoding="utf-8-sig"))
    log(
        f"doctor passed={payload['passed']} transport={payload['transport']['passed']} "
        f"license={payload['license_supplied']} init={payload['init_passed']}"
    )
    if not payload["license_supplied"]:
        license_path = resolve_repo_path(config["credentials"]["license_file"])
        log(f"license missing; fill exactly one line in {license_path}")
    log(f"doctor report={output.resolve()}")
    return result.returncode


def fetch(config: dict, args: argparse.Namespace, frequency: str) -> int:
    read_license(config, required=True)
    symbol_file = prepare_symbols(
        config,
        symbols=args.symbols,
        symbols_file=args.symbols_file,
        date=args.end_date,
        purpose=frequency,
    )
    api = config["api"]
    if frequency == "1d":
        command = "fetch-daily"
        fields = api["daily_fields"]
        batch_size = int(api["daily_batch_size"])
        output_dir = resolve_repo_path(config["primary"]["daily_root"])
    else:
        command = "fetch-minute"
        fields = api["minute_fields"]
        batch_size = int(api["minute_batch_size"])
        output_dir = resolve_repo_path(config["primary"]["minute_root"])
    manifest = (
        OUTPUTS
        / "rqdata_sync"
        / (
            f"{command}_{args.start_date.replace('-', '')}_"
            f"{args.end_date.replace('-', '')}_{symbol_file.stem}.json"
        )
    )
    bridge_args = [
        command,
        "--symbols-file",
        str(symbol_file),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--fields",
        ",".join(fields),
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(manifest),
        "--retries",
        str(api["retries"]),
        "--backoff",
        str(api["retry_backoff_seconds"]),
    ]
    log(
        f"start {frequency} symbols={len(symbols_from_file(symbol_file))} "
        f"range={args.start_date}..{args.end_date} output={output_dir}"
    )
    result = run_bridge(config, bridge_args, require_license=True)
    log(f"complete returncode={result.returncode} manifest={manifest.resolve()}")
    return result.returncode


def fetch_market_state(config: dict, args: argparse.Namespace) -> int:
    read_license(config, required=True)
    symbol_file = prepare_symbols(
        config,
        symbols=args.symbols,
        symbols_file=args.symbols_file,
        date=args.end_date,
        purpose="state",
    )
    metadata_root = resolve_repo_path(config["primary"]["metadata_root"])
    output = metadata_root / "pit_market_state.csv"
    manifest = OUTPUTS / "rqdata_sync" / (
        f"market_state_{args.start_date.replace('-', '')}_"
        f"{args.end_date.replace('-', '')}_{symbol_file.stem}.json"
    )
    api = config["api"]
    bridge_args = [
        "fetch-market-state",
        "--symbols-file",
        str(symbol_file),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--batch-size",
        str(api["market_state_batch_size"]),
        "--output",
        str(output),
        "--manifest",
        str(manifest),
        "--retries",
        str(api["retries"]),
        "--backoff",
        str(api["retry_backoff_seconds"]),
    ]
    log(
        f"start market-state symbols={len(symbols_from_file(symbol_file))} "
        f"range={args.start_date}..{args.end_date}"
    )
    result = run_bridge(config, bridge_args, require_license=True)
    log(f"complete returncode={result.returncode} output={output} manifest={manifest}")
    return result.returncode


def fetch_universe(config: dict, output: Path) -> int:
    read_license(config, required=True)
    result = run_bridge(
        config,
        [
            "instruments",
            "--all-history",
            "--output",
            str(output.resolve()),
        ],
        require_license=True,
    )
    log(
        f"complete universe returncode={result.returncode} "
        f"output={output.resolve()}"
    )
    return result.returncode


def build_pit_universe_symbols(
    instruments_path: Path,
    *,
    start_date: str,
    end_date: str,
    output: Path,
) -> Path:
    frame = pd.read_csv(instruments_path, dtype=str)
    required = {"instrument", "listed_date", "de_listed_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"historical instruments missing columns: {sorted(missing)}"
        )
    listed = pd.to_datetime(frame["listed_date"], errors="coerce")
    delisted = pd.to_datetime(
        frame["de_listed_date"].replace("0000-00-00", pd.NA),
        errors="coerce",
    )
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    eligible = listed.le(end) & (delisted.isna() | delisted.ge(start))
    return write_symbol_file(
        output,
        frame.loc[eligible, "instrument"].dropna().tolist(),
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="RQData-primary market data sync with local CSV fallback"
    )
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = root.add_subparsers(dest="command", required=True)

    doctor_cmd = sub.add_parser("doctor")
    doctor_cmd.add_argument(
        "--output", type=Path, default=OUTPUTS / "rqdata_doctor.json"
    )

    universe_cmd = sub.add_parser("universe")
    universe_cmd.add_argument("--output", type=Path)
    universe_cmd.add_argument("--pit-start")
    universe_cmd.add_argument("--pit-end")
    universe_cmd.add_argument("--symbols-output", type=Path)

    for name in ("daily", "minute", "state"):
        command = sub.add_parser(name)
        command.add_argument("--start-date", required=True)
        command.add_argument("--end-date", required=True)
        command.add_argument("--symbols")
        command.add_argument("--symbols-file", type=Path)
    return root


def main() -> int:
    args = parser().parse_args()
    config = load_config(args.config)
    if args.command == "doctor":
        return doctor(config, args.output)
    if args.command == "universe":
        metadata_root = resolve_repo_path(config["primary"]["metadata_root"])
        output = args.output or metadata_root / "instruments_all_history.csv"
        returncode = fetch_universe(config, output)
        pit_options = (args.pit_start, args.pit_end, args.symbols_output)
        if any(pit_options):
            if not all(pit_options):
                raise ValueError(
                    "--pit-start, --pit-end, and --symbols-output are required together"
                )
            symbols_output = build_pit_universe_symbols(
                output,
                start_date=args.pit_start,
                end_date=args.pit_end,
                output=args.symbols_output.resolve(),
            )
            log(
                f"PIT universe symbols={len(symbols_from_file(symbols_output))} "
                f"output={symbols_output}"
            )
        return returncode
    if pd.Timestamp(args.start_date) > pd.Timestamp(args.end_date):
        raise ValueError("start_date must not be after end_date")
    if args.command == "daily":
        return fetch(config, args, "1d")
    if args.command == "minute":
        if not args.symbols and args.symbols_file is None:
            raise ValueError("minute sync requires --symbols or --symbols-file")
        return fetch(config, args, "1m")
    if args.command == "state":
        return fetch_market_state(config, args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
