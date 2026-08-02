from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .build_minute_staging import (
        build_minute_source_index,
        files_for_instrument,
        read_one,
    )
    from .data_sources.rqdata_source import (
        DEFAULT_CONFIG,
        MarketDataGateway,
        load_config,
        local_symbol,
        merge_price_sources,
        read_price_csv,
        resolve_repo_path,
        write_price_csv_atomic,
    )
except ImportError:
    from build_minute_staging import (
        build_minute_source_index,
        files_for_instrument,
        read_one,
    )
    from data_sources.rqdata_source import (
        DEFAULT_CONFIG,
        MarketDataGateway,
        load_config,
        local_symbol,
        merge_price_sources,
        read_price_csv,
        resolve_repo_path,
        write_price_csv_atomic,
    )


HERE = Path(__file__).resolve().parent
DAILY_SCALE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "limit_up",
    "limit_down",
    "prev_close",
)
MIN_SCALE_OVERLAP_ROWS = 5
MAX_SCALE_REL_ERROR_P95 = 0.01


def align_daily_fallback_scale(
    primary: pd.DataFrame | None,
    fallback: pd.DataFrame | None,
    *,
    min_overlap_rows: int = MIN_SCALE_OVERLAP_ROWS,
    max_rel_error_p95: float = MAX_SCALE_REL_ERROR_P95,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if primary is None or primary.empty:
        return fallback, {
            "scale_alignment_status": "fallback_only",
            "scale_factor": None,
            "scale_overlap_rows": 0,
            "scale_calibration_rows": 0,
            "scale_rel_error_p95": None,
        }
    if fallback is None or fallback.empty:
        return fallback, {
            "scale_alignment_status": "primary_only",
            "scale_factor": None,
            "scale_overlap_rows": 0,
            "scale_calibration_rows": 0,
            "scale_rel_error_p95": None,
        }

    overlap = primary[["date", "close"]].merge(
        fallback[["date", "close"]],
        on="date",
        how="inner",
        suffixes=("_primary", "_fallback"),
    )
    primary_close = pd.to_numeric(overlap["close_primary"], errors="coerce")
    fallback_close = pd.to_numeric(overlap["close_fallback"], errors="coerce")
    valid = (
        primary_close.notna()
        & fallback_close.notna()
        & (primary_close > 0)
        & (fallback_close > 0)
    )
    ratios = (primary_close[valid] / fallback_close[valid]).dropna()
    if len(ratios) < min_overlap_rows:
        return fallback, {
            "scale_alignment_status": "missing_overlap",
            "scale_factor": None,
            "scale_overlap_rows": int(len(ratios)),
            "scale_calibration_rows": 0,
            "scale_rel_error_p95": None,
        }

    # Only the leading overlap calibrates the fallback-to-primary boundary.
    # Later corporate actions may legitimately change vendor adjustment ratios,
    # but those later rows are already replaced by the primary source.
    calibration = ratios.iloc[:min_overlap_rows]
    scale = float(calibration.median())
    relative_error = (calibration / scale - 1.0).abs()
    rel_error_p95 = float(relative_error.quantile(0.95))
    if not pd.notna(scale) or scale <= 0 or rel_error_p95 > max_rel_error_p95:
        return fallback, {
            "scale_alignment_status": "unstable_overlap",
            "scale_factor": scale if pd.notna(scale) else None,
            "scale_overlap_rows": int(len(ratios)),
            "scale_calibration_rows": int(len(calibration)),
            "scale_rel_error_p95": rel_error_p95,
        }

    aligned = fallback.copy()
    for column in DAILY_SCALE_COLUMNS:
        if column in aligned.columns:
            aligned[column] = pd.to_numeric(
                aligned[column], errors="coerce"
            ) * scale
    return aligned, {
        "scale_alignment_status": "aligned",
        "scale_factor": scale,
        "scale_overlap_rows": int(len(ratios)),
        "scale_calibration_rows": int(len(calibration)),
        "scale_rel_error_p95": rel_error_p95,
    }


def symbols_from_args(args: argparse.Namespace, gateway: MarketDataGateway) -> list[str]:
    values: list[str] = []
    if args.symbols:
        values.extend(args.symbols.split(","))
    if args.symbols_file:
        values.extend(
            line.strip().split()[0]
            for line in args.symbols_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    if not values and args.frequency == "daily":
        paths = list(gateway.primary_daily.glob("*_daily_qfq.csv"))
        paths.extend(gateway.fallback_daily.glob("*_daily_qfq.csv"))
        values.extend(path.name[:6] for path in paths if path.name[:6].isdigit())
    if not values:
        raise ValueError("minute materialization requires --symbols or --symbols-file")
    return sorted({local_symbol(value) for value in values})


def restrict(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, frequency: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    upper = end + pd.Timedelta(days=1) if frequency == "minute" else end
    return frame[frame["date"].between(start, upper, inclusive="left" if frequency == "minute" else "both")].copy()


def materialize_daily(
    gateway: MarketDataGateway,
    symbols: list[str],
    output_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos, symbol in enumerate(symbols, start=1):
        primary_path, fallback_path = gateway.daily_paths(symbol)
        primary = read_price_csv(primary_path, frequency="1d") if primary_path.is_file() else None
        fallback = read_price_csv(fallback_path, frequency="1d") if fallback_path.is_file() else None
        primary = restrict(primary, start, end, "daily") if primary is not None else None
        fallback = restrict(fallback, start, end, "daily") if fallback is not None else None
        fallback, scale_evidence = align_daily_fallback_scale(primary, fallback)
        merged = merge_price_sources(primary, fallback)
        if merged.empty:
            continue
        output = output_root / f"{symbol[2:]}_daily_qfq.csv"
        if output.is_file():
            existing = read_price_csv(output, frequency="1d")
            merged = merge_price_sources(merged, existing)
        write_price_csv_atomic(output, merged)
        rows.append(
            {
                "instrument": symbol,
                "rows": len(merged),
                "first": merged["date"].min().isoformat(),
                "last": merged["date"].max().isoformat(),
                "rqdata_rows": 0 if primary is None else len(primary),
                "fallback_rows": 0 if fallback is None else len(fallback),
                "output": str(output.resolve()),
                **scale_evidence,
            }
        )
        if pos % 250 == 0 or pos == len(symbols):
            print(f"[RQData canonical] daily {pos}/{len(symbols)} written={len(rows)}", flush=True)
    return rows


def materialize_minute(
    gateway: MarketDataGateway,
    symbols: list[str],
    output_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    include_fallback: bool = True,
) -> list[dict[str, Any]]:
    source_index = (
        build_minute_source_index(gateway.fallback_minute)
        if include_fallback
        else {}
    )
    rows: list[dict[str, Any]] = []
    for pos, symbol in enumerate(symbols, start=1):
        primary_parts = [read_price_csv(path, frequency="1m") for path in gateway.minute_primary_files(symbol)]
        primary = pd.concat(primary_parts, ignore_index=True) if primary_parts else None
        fallback_parts = (
            [
                read_one(source)
                for source in files_for_instrument(
                    symbol,
                    source_index=source_index,
                    start=start,
                    end=end,
                )
            ]
            if include_fallback
            else []
        )
        fallback = pd.concat(fallback_parts, ignore_index=True) if fallback_parts else None
        primary = restrict(primary, start, end, "minute") if primary is not None else None
        fallback = restrict(fallback, start, end, "minute") if fallback is not None else None
        merged = merge_price_sources(primary, fallback)
        if merged.empty:
            continue
        output = output_root / f"{symbol}_1m.csv"
        if output.is_file():
            existing = read_price_csv(output, frequency="1m")
            merged = merge_price_sources(merged, existing)
        write_price_csv_atomic(output, merged)
        rows.append(
            {
                "instrument": symbol,
                "rows": len(merged),
                "first": merged["date"].min().isoformat(),
                "last": merged["date"].max().isoformat(),
                "rqdata_rows": 0 if primary is None else len(primary),
                "fallback_rows": 0 if fallback is None else len(fallback),
                "output": str(output.resolve()),
            }
        )
        print(f"[RQData canonical] minute {pos}/{len(symbols)} written={len(rows)}", flush=True)
    return rows


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Materialize RQData-primary canonical CSVs")
    root.add_argument("frequency", choices=["daily", "minute"])
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    root.add_argument("--start-date", required=True)
    root.add_argument("--end-date", required=True)
    root.add_argument("--symbols")
    root.add_argument("--symbols-file", type=Path)
    root.add_argument("--output-root", type=Path)
    root.add_argument("--manifest", type=Path)
    root.add_argument(
        "--primary-only",
        action="store_true",
        help="Materialize only RQData rows; intended for minute data with incompatible legacy adjustment scales.",
    )
    return root


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    gateway = MarketDataGateway(config)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    symbols = symbols_from_args(args, gateway)
    key = "daily_root" if args.frequency == "daily" else "minute_root"
    output_root = args.output_root.resolve() if args.output_root else resolve_repo_path(config["canonical"][key])
    if args.frequency == "daily":
        rows = materialize_daily(gateway, symbols, output_root, start, end)
    else:
        rows = materialize_minute(
            gateway,
            symbols,
            output_root,
            start,
            end,
            include_fallback=not args.primary_only,
        )
    manifest_path = args.manifest.resolve() if args.manifest else resolve_repo_path(config["canonical"]["manifest"])
    source_precedence = (
        ["rqdata_primary"]
        if args.primary_only
        else ["rqdata_primary", "local_fallback"]
    )
    report = {
        "status": "rqdata_primary_materialized" if args.primary_only else "rqdata_primary_local_fallback_materialized",
        "frequency": args.frequency,
        "source_precedence": source_precedence,
        "range": {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")},
        "requested_symbols": len(symbols),
        "written_symbols": len(rows),
        "rqdata_symbols": sum(row["rqdata_rows"] > 0 for row in rows),
        "fallback_symbols": sum(row["fallback_rows"] > 0 for row in rows),
        "output_root": str(output_root.resolve()),
        "files": rows,
    }
    aggregate: dict[str, Any] = {
        "status": "rqdata_primary_local_fallback_canonical",
        "frequencies": {},
    }
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if isinstance(previous.get("frequencies"), dict):
                aggregate = previous
        except (json.JSONDecodeError, OSError):
            pass
    aggregate["frequencies"][args.frequency] = report
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pending = manifest_path.with_suffix(manifest_path.suffix + ".pending")
    pending.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(manifest_path)
    print(f"[RQData canonical] complete written={len(rows)} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
