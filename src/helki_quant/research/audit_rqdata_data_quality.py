from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .build_minute_staging import build_minute_source_index, read_one
    from .data_sources.rqdata_source import (
        DEFAULT_CONFIG,
        MarketDataGateway,
        load_config,
        local_symbol,
        read_price_csv,
    )
except ImportError:
    from build_minute_staging import build_minute_source_index, read_one
    from data_sources.rqdata_source import (
        DEFAULT_CONFIG,
        MarketDataGateway,
        load_config,
        local_symbol,
        read_price_csv,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs" / "rqdata_quality_audit.json"


def finite_quantile(values: pd.Series, quantile: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.quantile(quantile)) if len(clean) else None


def log_correlation(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.DataFrame(
        {
            "left": np.log1p(pd.to_numeric(left, errors="coerce").clip(lower=0)),
            "right": np.log1p(pd.to_numeric(right, errors="coerce").clip(lower=0)),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    return float(frame["left"].corr(frame["right"]))


def invalid_ohlc_rows(frame: pd.DataFrame) -> int:
    values = frame[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid = (
        values.isna().any(axis=1)
        | values.le(0).any(axis=1)
        | values["high"].lt(values[["open", "close", "low"]].max(axis=1))
        | values["low"].gt(values[["open", "close", "high"]].min(axis=1))
    )
    return int(invalid.sum())


def compare_price_frames(
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
    *,
    frequency: str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    primary = primary.copy().drop_duplicates("date", keep="last").sort_values("date")
    fallback = fallback.copy().drop_duplicates("date", keep="last").sort_values("date")
    primary["return"] = pd.to_numeric(primary["close"], errors="coerce").pct_change(
        fill_method=None
    )
    fallback["return"] = pd.to_numeric(fallback["close"], errors="coerce").pct_change(
        fill_method=None
    )
    common = primary.merge(fallback, on="date", suffixes=("_rq", "_local"), how="inner")
    union_rows = len(set(primary["date"]) | set(fallback["date"]))
    common_ratio = len(common) / union_rows if union_rows else 0.0
    if common.empty:
        return {
            "passed": False,
            "failed_checks": ["no_common_observations"],
            "primary_rows": int(len(primary)),
            "fallback_rows": int(len(fallback)),
            "common_rows": 0,
            "common_ratio": 0.0,
        }

    return_diff = (common["return_rq"] - common["return_local"]).abs()
    normalized_errors: list[pd.Series] = []
    rq_anchor = float(pd.to_numeric(common["close_rq"], errors="coerce").dropna().iloc[-1])
    local_anchor = float(
        pd.to_numeric(common["close_local"], errors="coerce").dropna().iloc[-1]
    )
    if rq_anchor > 0 and local_anchor > 0:
        for field in ("open", "high", "low", "close"):
            rq_values = pd.to_numeric(common[f"{field}_rq"], errors="coerce") / rq_anchor
            local_values = (
                pd.to_numeric(common[f"{field}_local"], errors="coerce") / local_anchor
            )
            normalized_errors.append(
                ((rq_values - local_values).abs() / local_values.abs().replace(0, np.nan))
            )
    normalized_error = (
        pd.concat(normalized_errors, ignore_index=True)
        if normalized_errors
        else pd.Series(dtype=float)
    )
    metrics = {
        "primary_rows": int(len(primary)),
        "fallback_rows": int(len(fallback)),
        "common_rows": int(len(common)),
        "common_ratio": float(common_ratio),
        "primary_first": primary["date"].min().isoformat(),
        "primary_last": primary["date"].max().isoformat(),
        "fallback_first": fallback["date"].min().isoformat(),
        "fallback_last": fallback["date"].max().isoformat(),
        "return_abs_diff_p50": finite_quantile(return_diff, 0.50),
        "return_abs_diff_p95": finite_quantile(return_diff, 0.95),
        "return_abs_diff_max": finite_quantile(return_diff, 1.00),
        "normalized_ohlc_rel_diff_p95": finite_quantile(normalized_error, 0.95),
        "volume_log_correlation": log_correlation(
            common["volume_rq"], common["volume_local"]
        ),
        "amount_log_correlation": log_correlation(
            common["amount_rq"], common["amount_local"]
        ),
        "primary_invalid_ohlc_rows": invalid_ohlc_rows(primary),
        "fallback_invalid_ohlc_rows": invalid_ohlc_rows(fallback),
        "primary_duplicate_timestamps": int(primary["date"].duplicated().sum()),
        "fallback_duplicate_timestamps": int(fallback["date"].duplicated().sum()),
    }
    prefix = "daily" if frequency == "1d" else "minute"
    checks = {
        "common_ratio": (
            metrics["common_ratio"]
            >= float(thresholds[f"min_{prefix}_common_{'date' if frequency == '1d' else 'timestamp'}_ratio"])
        ),
        "return_abs_diff_p95": (
            metrics["return_abs_diff_p95"] is not None
            and metrics["return_abs_diff_p95"]
            <= float(thresholds[f"max_{prefix}_return_abs_diff_p95"])
        ),
        "normalized_ohlc_rel_diff_p95": (
            metrics["normalized_ohlc_rel_diff_p95"] is not None
            and metrics["normalized_ohlc_rel_diff_p95"]
            <= float(thresholds[f"max_{prefix}_normalized_ohlc_rel_diff_p95"])
        ),
        "primary_ohlc_valid": metrics["primary_invalid_ohlc_rows"] == 0,
    }
    if frequency == "1d":
        for field in ("volume", "amount"):
            value = metrics[f"{field}_log_correlation"]
            checks[f"{field}_log_correlation"] = bool(
                value is not None
                and value >= float(thresholds[f"min_daily_{field}_log_correlation"])
            )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        **metrics,
    }


def load_fallback_minute(
    symbol: str,
    source_index: dict[str, list],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    parts = []
    for source in source_index.get(symbol, []):
        frame = read_one(source)
        frame = frame[frame["date"].between(start, end + pd.Timedelta(days=1))].copy()
        if len(frame):
            parts.append(frame)
    if not parts:
        raise FileNotFoundError(f"no local fallback minute rows for {symbol}")
    return (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def requested_symbols(args: argparse.Namespace, gateway: MarketDataGateway) -> list[str]:
    values: list[str] = []
    if args.symbols:
        values.extend(args.symbols.split(","))
    if args.symbols_file:
        values.extend(
            line.strip().split()[0]
            for line in args.symbols_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    if not values:
        values = [
            ("sh" if path.name.startswith(("6", "9")) else "sz") + path.name[:6]
            for path in gateway.primary_daily.glob("*_daily_qfq.csv")
        ]
    unique = sorted({local_symbol(value) for value in values})
    return unique[: args.sample_size] if args.sample_size else unique


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    gateway = MarketDataGateway(config)
    thresholds = config["quality_gate"]
    symbols = requested_symbols(args, gateway)
    if not symbols:
        raise ValueError("no RQData primary symbols available for quality audit")
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    minute_index = build_minute_source_index(gateway.fallback_minute)

    daily_rows: list[dict[str, Any]] = []
    minute_rows: list[dict[str, Any]] = []
    for pos, symbol in enumerate(symbols, start=1):
        primary_path, fallback_path = gateway.daily_paths(symbol)
        if primary_path.is_file() and fallback_path.is_file():
            primary = read_price_csv(primary_path, frequency="1d")
            fallback = read_price_csv(fallback_path, frequency="1d")
            primary = primary[primary["date"].between(start, end)]
            fallback = fallback[fallback["date"].between(start, end)]
            result = compare_price_frames(
                primary,
                fallback,
                frequency="1d",
                thresholds=thresholds,
            )
            daily_rows.append({"instrument": symbol, **result})

        primary_minute_files = gateway.minute_primary_files(symbol)
        if primary_minute_files:
            primary_parts = [
                read_price_csv(path, frequency="1m") for path in primary_minute_files
            ]
            primary_minute = (
                pd.concat(primary_parts, ignore_index=True)
                .drop_duplicates("date", keep="last")
                .sort_values("date")
            )
            primary_minute = primary_minute[
                primary_minute["date"].between(start, end + pd.Timedelta(days=1))
            ]
            try:
                fallback_minute = load_fallback_minute(
                    symbol, minute_index, start, end
                )
                result = compare_price_frames(
                    primary_minute,
                    fallback_minute,
                    frequency="1m",
                    thresholds=thresholds,
                )
                minute_rows.append({"instrument": symbol, **result})
            except FileNotFoundError as exc:
                minute_rows.append(
                    {
                        "instrument": symbol,
                        "passed": False,
                        "failed_checks": ["local_minute_overlap_missing"],
                        "error": str(exc),
                    }
                )
        if pos % 25 == 0 or pos == len(symbols):
            print(
                f"[RQData quality] {pos}/{len(symbols)} daily={len(daily_rows)} "
                f"minute={len(minute_rows)}",
                flush=True,
            )

    daily_passed = sum(bool(row.get("passed")) for row in daily_rows)
    minute_passed = sum(bool(row.get("passed")) for row in minute_rows)
    min_ratio = float(thresholds["min_valid_symbol_ratio"])
    daily_ratio = daily_passed / len(daily_rows) if daily_rows else 0.0
    minute_ratio = minute_passed / len(minute_rows) if minute_rows else 0.0
    if not daily_rows:
        daily_decision = "insufficient_sdk_overlap"
        historical_retrain = "not_evaluated"
    elif daily_ratio < min_ratio:
        daily_decision = "provider_discrepancy_investigation_required"
        historical_retrain = "blocked_until_data_semantics_are_reconciled"
    else:
        daily_decision = "canonical_incremental_promotion_allowed"
        historical_retrain = "not_required_for_source_migration"
    if not minute_rows:
        minute_decision = "minute_quality_not_evaluated"
    elif minute_ratio < min_ratio:
        minute_decision = "minute_provider_discrepancy_investigation_required"
    else:
        minute_decision = "canonical_minute_incremental_promotion_allowed"

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    daily_csv = output.with_name(output.stem + "_daily.csv")
    minute_csv = output.with_name(output.stem + "_minute.csv")
    pd.DataFrame(daily_rows).to_csv(daily_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(minute_rows).to_csv(minute_csv, index=False, encoding="utf-8-sig")
    report = {
        "status": "rqdata_vs_local_quality_audit",
        "passed": bool(daily_rows and daily_ratio >= min_ratio),
        "source_policy": "rqdata_primary_local_fallback",
        "comparison_range": {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        },
        "symbols_requested": len(symbols),
        "daily": {
            "evaluated": len(daily_rows),
            "passed": daily_passed,
            "passed_ratio": daily_ratio,
            "decision": daily_decision,
            "details_csv": str(daily_csv),
        },
        "minute": {
            "evaluated": len(minute_rows),
            "passed": minute_passed,
            "passed_ratio": minute_ratio,
            "decision": minute_decision,
            "details_csv": str(minute_csv),
        },
        "historical_data_replacement_allowed": bool(
            daily_rows and daily_ratio >= min_ratio
        ),
        "historical_retrain_decision": historical_retrain,
        "new_untouched_evaluation_required": True,
        "frozen_profile_retuning_allowed": False,
        "thresholds": thresholds,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Compare RQData with legacy local market data")
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    root.add_argument("--start-date", required=True)
    root.add_argument("--end-date", required=True)
    root.add_argument("--symbols")
    root.add_argument("--symbols-file", type=Path)
    root.add_argument("--sample-size", type=int, default=100)
    root.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return root


def main() -> None:
    args = parser().parse_args()
    report = audit(args)
    print(
        f"[RQData quality] passed={report['passed']} "
        f"daily={report['daily']['decision']} minute={report['minute']['decision']} "
        f"output={args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
