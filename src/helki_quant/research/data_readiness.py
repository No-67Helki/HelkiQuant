from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_research import build_folds


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DAILY = DATA / "cn_data_pool"
MINUTE = DATA / "cn_data_1min_pool"
RAW_MINUTE = DATA / "A_Stock_1min"
TARGET = "sz301536"
TRUE_MINUTE_FIELDS = [
    "first30m_ret",
    "last30m_ret",
    "intra_mom_diff",
    "min_vol_std",
    "vwap_dev",
    "max_min_vol_ratio",
    "price_vol_corr_min",
    "vwap_revert",
    "t_open30_ret",
    "t_mid_ret",
    "t_vwap_slope",
    "t_range_pos",
    "t_vol_conc",
    "t_late_mom",
    "intraday_t_ret_stable",
]


def read_calendar(path: Path) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_csv(path, header=None).iloc[:, 0].pipe(pd.to_datetime))


def read_instruments(path: Path) -> list[str]:
    return [
        line.split()[0].lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bin_coverage(path: Path, calendar: pd.DatetimeIndex) -> dict:
    if not path.exists():
        return {"exists": False, "nonnull": 0, "first": None, "last": None}
    raw = np.fromfile(path, dtype="<f4")
    if len(raw) < 2:
        return {"exists": True, "nonnull": 0, "first": None, "last": None}
    start_index = int(raw[0])
    values = raw[1:]
    valid = np.flatnonzero(np.isfinite(values))
    valid = valid[(start_index + valid) < len(calendar)]
    if not len(valid):
        return {"exists": True, "nonnull": 0, "first": None, "last": None}
    first_index = start_index + int(valid[0])
    last_index = start_index + int(valid[-1])
    return {
        "exists": True,
        "nonnull": int(len(valid)),
        "first": str(calendar[first_index].date()),
        "last": str(calendar[last_index].date()),
    }


def bin_nonnull_between(
    path: Path,
    calendar: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    if not path.exists():
        return 0
    raw = np.fromfile(path, dtype="<f4")
    if len(raw) < 2:
        return 0
    start_index = int(raw[0])
    values = raw[1:]
    end_index = min(start_index + len(values), len(calendar))
    dates = calendar[start_index:end_index]
    values = values[: len(dates)]
    mask = (dates >= start) & (dates <= end) & np.isfinite(values)
    return int(mask.sum())


def target_true_minute_coverage(calendar: pd.DatetimeIndex) -> dict:
    feature_dir = DAILY / "features" / TARGET
    return {
        field: bin_coverage(feature_dir / f"{field}.day.bin", calendar)
        for field in TRUE_MINUTE_FIELDS
    }


def pool_label_coverage(calendar: pd.DatetimeIndex, instruments: list[str]) -> dict:
    rows = []
    for instrument in instruments:
        coverage = bin_coverage(
            DAILY / "features" / instrument / "intraday_t_ret_stable.day.bin",
            calendar,
        )
        rows.append({"instrument": instrument, **coverage})
    last_dates = pd.to_datetime([row["last"] for row in rows if row["last"]])
    first_dates = pd.to_datetime([row["first"] for row in rows if row["first"]])
    return {
        "instruments": len(instruments),
        "with_label": sum(row["nonnull"] > 0 for row in rows),
        "earliest_first_date": str(first_dates.min().date()) if len(first_dates) else None,
        "latest_first_date": str(first_dates.max().date()) if len(first_dates) else None,
        "earliest_last_date": str(last_dates.min().date()) if len(last_dates) else None,
        "latest_last_date": str(last_dates.max().date()) if len(last_dates) else None,
        "details": rows,
    }


def fold_readiness(
    folds: list[dict],
    daily_calendar: pd.DatetimeIndex,
    minute_calendar: pd.DatetimeIndex,
    target_close: dict,
    pool_label: dict,
    minute_instruments: list[str],
) -> list[dict]:
    daily_end = daily_calendar.max()
    minute_end = minute_calendar.max()
    label_end = pd.Timestamp(pool_label["earliest_last_date"])
    listing_start = pd.Timestamp(target_close["first"])
    rows = []
    for fold in folds:
        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        valid_start = pd.Timestamp(fold["valid_start"])
        valid_end = pd.Timestamp(fold["valid_end"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"])
        target_train_days = int(((daily_calendar >= listing_start) & (daily_calendar <= train_end)).sum())
        label_paths = [
            DAILY / "features" / instrument / "intraday_t_ret_stable.day.bin"
            for instrument in minute_instruments
        ]
        inner_train_rows = sum(
            bin_nonnull_between(path, daily_calendar, train_start, train_end)
            for path in label_paths
        )
        inner_valid_rows = sum(
            bin_nonnull_between(path, daily_calendar, valid_start, valid_end)
            for path in label_paths
        )
        inner_test_rows = sum(
            bin_nonnull_between(path, daily_calendar, test_start, test_end)
            for path in label_paths
        )
        test_instruments = sum(
            bin_nonnull_between(path, daily_calendar, test_start, test_end) > 0
            for path in label_paths
        )
        target_label_path = (
            DAILY / "features" / TARGET / "intraday_t_ret_stable.day.bin"
        )
        inner_coverage_ready = minute_end >= test_end and label_end >= test_end
        inner_training_sufficient = inner_train_rows >= 10000 and inner_valid_rows >= 2000
        inner_ready = inner_coverage_ready and inner_training_sufficient
        blockers = []
        if not inner_coverage_ready:
            blockers.append(
                "inner minute calendar/true-minute labels do not cover the full fold"
            )
        if not inner_training_sufficient:
            blockers.append("inner train/valid true-minute label sample is insufficient")
        rows.append(
            {
                "fold": fold["fold"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "outer_daily_ready": bool(daily_end >= test_end),
                "middle_daily_ready": bool(daily_end >= test_end),
                "inner_true_minute_ready": bool(inner_ready),
                "inner_coverage_ready": bool(inner_coverage_ready),
                "inner_training_sufficient": bool(inner_training_sufficient),
                "inner_train_label_rows": inner_train_rows,
                "inner_valid_label_rows": inner_valid_rows,
                "inner_test_label_rows": inner_test_rows,
                "inner_test_instruments": test_instruments,
                "target_inner_train_rows": bin_nonnull_between(
                    target_label_path, daily_calendar, train_start, train_end
                ),
                "target_inner_test_rows": bin_nonnull_between(
                    target_label_path, daily_calendar, test_start, test_end
                ),
                "target_train_days": target_train_days,
                "target_has_250d_history": target_train_days >= 250,
                "blockers": blockers,
            }
        )
    return rows


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Research v3 Data Readiness",
        "",
        f"- Daily calendar: {report['daily']['start']} to {report['daily']['end']}",
        f"- Fixed daily source pool: {report['daily']['source_pool_size']} instruments",
        f"- Minute calendar: {report['minute']['start']} to {report['minute']['end']}",
        f"- Minute pool: {report['minute']['pool_size']} instruments",
        f"- Raw minute directories: {', '.join(report['minute']['raw_year_dirs'])}",
        f"- Target listing/data start: {report['target']['close']['first']}",
        "",
        "## Fold Readiness",
        "",
        "| Fold | Test | Daily | Inner true minute | Inner train/valid rows | Target train days | 250d history |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for row in report["fold_readiness"]:
        lines.append(
            f"| {row['fold']} | {row['test_start']} to {row['test_end']} | "
            f"{'ready' if row['outer_daily_ready'] else 'blocked'} | "
            f"{'ready' if row['inner_true_minute_ready'] else 'blocked'} | "
            f"{row['inner_train_label_rows']}/{row['inner_valid_label_rows']} | "
            f"{row['target_train_days']} | "
            f"{'yes' if row['target_has_250d_history'] else 'no'} |"
        )
    lines.extend(["", "## Mandatory Warnings", ""])
    lines.extend(f"- {warning}" for warning in report["mandatory_warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(folds_path: Path, output_dir: Path) -> dict:
    if not folds_path.exists():
        build_folds(folds_path.parent)
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    daily_calendar = read_calendar(DAILY / "calendars" / "day.txt")
    minute_calendar = read_calendar(MINUTE / "calendars" / "1min.txt")
    daily_instruments = read_instruments(DAILY / "instruments" / "all.txt")
    minute_instruments = read_instruments(MINUTE / "instruments" / "all.txt")
    target_close = bin_coverage(
        DAILY / "features" / TARGET / "close.day.bin", daily_calendar
    )
    target_fields = target_true_minute_coverage(daily_calendar)
    pool_label = pool_label_coverage(daily_calendar, minute_instruments)
    raw_year_dirs = sorted(path.name for path in RAW_MINUTE.glob("*_1min") if path.is_dir())
    raw_year_file_counts = {
        path.name: len(list(path.glob("*.csv")))
        for path in sorted(RAW_MINUTE.glob("*_1min"))
        if path.is_dir()
    }
    raw_target_years = sorted(
        path.parent.name
        for path in RAW_MINUTE.glob(f"*_1min/{TARGET}_*.csv")
    )
    readiness = fold_readiness(
        folds,
        daily_calendar,
        minute_calendar,
        target_close,
        pool_label,
        minute_instruments,
    )
    report = {
        "status": "research_data_gate",
        "daily": {
            "start": str(daily_calendar.min().date()),
            "end": str(daily_calendar.max().date()),
            "source_pool_size": len(daily_instruments),
        },
        "minute": {
            "start": str(minute_calendar.min()),
            "end": str(minute_calendar.max()),
            "pool_size": len(minute_instruments),
            "raw_year_dirs": raw_year_dirs,
            "raw_year_file_counts": raw_year_file_counts,
            "raw_target_years": raw_target_years,
            "pool_label_coverage": pool_label,
        },
        "target": {
            "instrument": TARGET.upper(),
            "close": target_close,
            "true_minute_fields": target_fields,
        },
        "fold_readiness": readiness,
        "mandatory_warnings": [
            (
                "The daily 1666-stock source pool was selected with later information. "
                "Point-in-time eligibility reduces, but does not eliminate, survivor bias."
            ),
            (
                "The raw 2026 minute directory exists but contains no CSV files. "
                "cn_data_1min_pool and true-minute day fields therefore cannot yet "
                "be extended through the daily model test end."
            ),
            (
                "Fold-specific factor selection is mandatory. Never reuse robust_v2's "
                "global whitelist as OOF evidence."
            ),
            (
                "The observed robust_v2 test period is research data now, not an untouched "
                "final holdout."
            ),
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_readiness.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "DATA_READINESS.md")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default=str(HERE / "outputs" / "purged_folds.json"))
    parser.add_argument("--output-dir", default=str(HERE / "outputs"))
    args = parser.parse_args()
    report = build_report(Path(args.folds).resolve(), Path(args.output_dir).resolve())
    print(
        f"[readiness] daily_end={report['daily']['end']} "
        f"minute_end={report['minute']['end']}"
    )
    for row in report["fold_readiness"]:
        print(
            f"[fold {row['fold']}] daily={row['outer_daily_ready']} "
            f"inner={row['inner_true_minute_ready']} "
            f"target_days={row['target_train_days']}"
        )


if __name__ == "__main__":
    main()
