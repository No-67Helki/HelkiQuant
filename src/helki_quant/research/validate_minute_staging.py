from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import qlib
from qlib.data import D

from data_readiness import bin_coverage, read_calendar


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_MINUTE = DATA / "cn_data_1min_pool_research_2026"
DEFAULT_INNER_DAY = DATA / "cn_data_pool_inner_research_2026"


def read_instruments(path: Path) -> list[str]:
    return [
        line.split()[0].upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate(
    minute_provider: Path,
    inner_day_provider: Path,
    output_path: Path,
    min_instruments: int,
    max_stale_instruments: int,
) -> dict:
    minute_calendar = (minute_provider / "calendars" / "1min.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    day_calendar = read_calendar(inner_day_provider / "calendars" / "day.txt")
    instruments = read_instruments(minute_provider / "instruments" / "all.txt")
    sample_instruments = instruments[:2]
    qlib.init(provider_uri={"day": str(inner_day_provider), "1min": str(minute_provider)}, region="cn")
    minute_sample = D.features(
        sample_instruments,
        ["$close", "$volume"],
        start_time="2026-06-05 09:30:00",
        end_time="2026-06-05 15:00:00",
        freq="1min",
    )
    day_sample = D.features(
        sample_instruments,
        ["$INTRADAY_T_RET_STABLE", "$FIRST30M_RET", "$OPEN15_RET", "$CLOSE_PULLUP"],
        start_time="2026-06-01",
        end_time="2026-06-05",
        freq="day",
    )
    target_inst = instruments[0].lower() if instruments else ""
    target_cov = bin_coverage(
        inner_day_provider
        / "features"
        / target_inst
        / "intraday_t_ret_stable.day.bin",
        day_calendar,
    )
    coverage_rows = []
    for inst in instruments:
        cov = bin_coverage(
            inner_day_provider
            / "features"
            / inst.lower()
            / "intraday_t_ret_stable.day.bin",
            day_calendar,
        )
        coverage_rows.append({"instrument": inst, **cov})
    last_dates = [row["last"] for row in coverage_rows if row["last"] is not None]
    expected_last = max(last_dates) if last_dates else None
    complete_to_expected_last = bool(expected_last) and all(
        row["last"] == expected_last for row in coverage_rows
    )
    stale_rows = [row for row in coverage_rows if row["last"] != expected_last]
    report = {
        "status": "validated"
        if len(instruments) >= min_instruments
        and len(minute_sample)
        and len(day_sample)
        and (complete_to_expected_last or len(stale_rows) <= max_stale_instruments)
        else "failed",
        "minute_provider": str(minute_provider),
        "inner_day_provider": str(inner_day_provider),
        "minute_calendar_start": minute_calendar[0],
        "minute_calendar_end": minute_calendar[-1],
        "minute_calendar_count": len(minute_calendar),
        "instrument_count": len(instruments),
        "min_instruments": min_instruments,
        "max_stale_instruments": max_stale_instruments,
        "sample_instruments": sample_instruments,
        "minute_sample_rows": int(len(minute_sample)),
        "minute_sample_instruments": sorted(
            minute_sample.index.get_level_values("instrument").unique().tolist()
        ),
        "day_sample_rows": int(len(day_sample)),
        "day_sample_nonnull": int(np.isfinite(day_sample.values).sum()),
        "target_intraday_label_coverage": target_cov,
        "expected_label_last_date": expected_last,
        "all_label_last_dates": sorted({row["last"] for row in coverage_rows}),
        "stale_instrument_count": len(stale_rows),
        "stale_instruments": stale_rows[:20],
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-provider", default=str(DEFAULT_MINUTE))
    parser.add_argument("--inner-day-provider", default=str(DEFAULT_INNER_DAY))
    parser.add_argument("--min-instruments", type=int, default=80)
    parser.add_argument("--max-stale-instruments", type=int, default=5)
    parser.add_argument(
        "--output", default=str(HERE / "outputs" / "minute_staging_validation.json")
    )
    args = parser.parse_args()
    report = validate(
        Path(args.minute_provider).resolve(),
        Path(args.inner_day_provider).resolve(),
        Path(args.output).resolve(),
        args.min_instruments,
        args.max_stale_instruments,
    )
    print(
        f"[minute validation] status={report['status']} "
        f"instruments={report['instrument_count']} "
        f"minute_rows={report['minute_sample_rows']} "
        f"label_last={report['all_label_last_dates']}"
    )


if __name__ == "__main__":
    main()
