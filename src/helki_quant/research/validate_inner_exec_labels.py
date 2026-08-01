from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import qlib
from qlib.data import D


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_INNER_DAY_PROVIDER = DATA / "cn_data_pool_inner_research_2026"
DEFAULT_MINUTE_PROVIDER = DATA / "cn_data_1min_pool_research_2026"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data_readiness import bin_coverage, read_calendar  # noqa: E402


FIELDS = [
    "$INTRADAY_T_RET",
    "$INTRADAY_T_RET_STABLE",
    "$INTRADAY_T_EXEC_RET",
    "$INTRADAY_T_EXEC_NET_RET",
    "$INTRADAY_T_REVERSE_NET_RET",
    "$INTRADAY_T0_SELL_OPEN_BUY_AM_NET_RET",
    "$INTRADAY_T0_SELL_OPEN_BUY_PM_NET_RET",
    "$INTRADAY_T0_SELL_AM_BUY_PM_NET_RET",
    "$INTRADAY_T0_SELL_AM_BUY_CLOSE_NET_RET",
    "$INTRADAY_T0_BEST_BUCKET_NET_RET",
    "$INTRADAY_T0_BEST2_MEAN_NET_RET",
    "$INTRADAY_T0_BUCKET_HIT_RATIO",
]


def read_instruments(path: Path) -> list[str]:
    return [
        line.split()[0].upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def series_stats(series: pd.Series) -> dict:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "positive_ratio": float((clean > 0).mean()),
        "q05": float(clean.quantile(0.05)),
        "q25": float(clean.quantile(0.25)),
        "q75": float(clean.quantile(0.75)),
        "q95": float(clean.quantile(0.95)),
    }


def validate(
    inner_day_provider: Path,
    minute_provider: Path,
    output_path: Path,
    start_time: str,
    end_time: str,
    min_instruments: int,
) -> dict:
    instruments = read_instruments(minute_provider / "instruments" / "all.txt")
    calendar = read_calendar(inner_day_provider / "calendars" / "day.txt")
    coverage = []
    for inst in instruments:
        cov = bin_coverage(
            inner_day_provider / "features" / inst.lower() / "intraday_t_exec_net_ret.day.bin",
            calendar,
        )
        coverage.append({"instrument": inst, **cov})

    qlib.init(
        provider_uri={"day": str(inner_day_provider), "1min": str(minute_provider)},
        region="cn",
    )
    data = D.features(
        instruments,
        FIELDS,
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    data.columns = [field.removeprefix("$") for field in FIELDS]
    stats = {col: series_stats(data[col]) for col in data.columns}
    corr = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .corr(method="spearman")
        .round(6)
        .to_dict()
    )
    net = data["INTRADAY_T_EXEC_NET_RET"]
    reverse = data["INTRADAY_T_REVERSE_NET_RET"]
    both_positive = (net > 0) & (reverse > 0)
    both_negative = (net <= 0) & (reverse <= 0)
    expected_last = str(pd.Timestamp(end_time).date())
    last_counts = pd.Series([row.get("last") for row in coverage]).value_counts(dropna=False).to_dict()
    stale_coverage = [
        row for row in coverage if row.get("last") is not None and row.get("last") < expected_last
    ]
    report = {
        "status": "validated"
        if len(instruments) >= min_instruments
        and len(data) > 0
        and any(row.get("last") == expected_last for row in coverage)
        else "failed",
        "inner_day_provider": str(inner_day_provider),
        "minute_provider": str(minute_provider),
        "start_time": start_time,
        "end_time": end_time,
        "instrument_count": len(instruments),
        "min_instruments": min_instruments,
        "rows": int(len(data)),
        "field_stats": stats,
        "spearman_corr": corr,
        "direction_consistency": {
            "both_positive_ratio": float(both_positive.mean()),
            "both_nonpositive_ratio": float(both_negative.mean()),
            "net_positive_reverse_nonpositive_ratio": float(((net > 0) & (reverse <= 0)).mean()),
            "net_nonpositive_reverse_positive_ratio": float(((net <= 0) & (reverse > 0)).mean()),
        },
        "t0_best_bucket_above_cost_ratio": float(
            (data["INTRADAY_T0_BEST_BUCKET_NET_RET"] > 0).mean()
        )
        if "INTRADAY_T0_BEST_BUCKET_NET_RET" in data
        else None,
        "t0_best2_above_cost_ratio": float((data["INTRADAY_T0_BEST2_MEAN_NET_RET"] > 0).mean())
        if "INTRADAY_T0_BEST2_MEAN_NET_RET" in data
        else None,
        "t0_bucket_hit_ratio_mean": float(data["INTRADAY_T0_BUCKET_HIT_RATIO"].mean())
        if "INTRADAY_T0_BUCKET_HIT_RATIO" in data
        else None,
        "exec_net_label_coverage_first_dates": sorted({row.get("first") for row in coverage}),
        "exec_net_label_coverage_last_dates": sorted({row.get("last") for row in coverage}),
        "exec_net_label_coverage_last_counts": last_counts,
        "stale_coverage_count": int(len(stale_coverage)),
        "stale_coverage_sample": stale_coverage[:20],
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-day-provider", default=str(DEFAULT_INNER_DAY_PROVIDER))
    parser.add_argument("--minute-provider", default=str(DEFAULT_MINUTE_PROVIDER))
    parser.add_argument("--start-time", default="2022-01-04")
    parser.add_argument("--end-time", default="2026-04-28")
    parser.add_argument("--min-instruments", type=int, default=80)
    parser.add_argument(
        "--output", default=str(HERE / "outputs" / "inner_exec_label_validation.json")
    )
    args = parser.parse_args()
    report = validate(
        Path(args.inner_day_provider).resolve(),
        Path(args.minute_provider).resolve(),
        Path(args.output).resolve(),
        args.start_time,
        args.end_time,
        args.min_instruments,
    )
    net_stats = report["field_stats"]["INTRADAY_T_EXEC_NET_RET"]
    print(
        f"[inner label validation] status={report['status']} rows={report['rows']} "
        f"net_mean={net_stats.get('mean')} net_pos={net_stats.get('positive_ratio')}"
    )


if __name__ == "__main__":
    main()
