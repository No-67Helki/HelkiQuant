from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_RAW = DATA / "_research_industry_raw"
DEFAULT_OUTPUT = DATA / "industry_theme_pit.csv"


def normalize_instrument(code: object) -> str:
    text = str(code).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit()).zfill(6)[-6:]
    if digits.startswith(("60", "68", "90")):
        return f"SH{digits}"
    return f"SZ{digits}"


def read_one(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
    required = {
        "date",
        "first_industry_name",
        "second_industry_name",
        "third_industry_name",
        "code",
        "symbol",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"], errors="coerce"),
            "instrument": frame["code"].map(normalize_instrument),
            "industry": frame["first_industry_name"].astype(str).str.strip(),
            "sector": frame["second_industry_name"].astype(str).str.strip(),
            "theme": frame["third_industry_name"].astype(str).str.strip(),
            "symbol": frame["symbol"].astype(str).str.strip(),
            "source_file": str(path),
        }
    )
    return out.dropna(subset=["date"])


def compress_intervals(
    daily: pd.DataFrame,
    forward_fill_to: str | None = None,
) -> pd.DataFrame:
    daily = daily.sort_values(["instrument", "date"]).drop_duplicates(
        ["instrument", "date"],
        keep="last",
    )
    rows = []
    for inst, part in daily.groupby("instrument", sort=True):
        part = part.sort_values("date").reset_index(drop=True)
        keys = part[["industry", "sector", "theme", "symbol"]].astype(str)
        change = keys.ne(keys.shift()).any(axis=1)
        group_id = change.cumsum()
        blocks = list(part.groupby(group_id, sort=True))
        for idx, (_, block) in enumerate(blocks):
            end_date = block["date"].iloc[-1].date().isoformat()
            source = "A_Stock_industry_daily_snapshot"
            if forward_fill_to and idx == len(blocks) - 1:
                fill_to = pd.Timestamp(forward_fill_to).date().isoformat()
                if fill_to > end_date:
                    end_date = fill_to
                    source = "A_Stock_industry_daily_snapshot_forward_filled_from_last_snapshot"
            rows.append(
                {
                    "instrument": inst,
                    "code": inst[-6:],
                    "symbol": block["symbol"].iloc[-1],
                    "industry": block["industry"].iloc[-1],
                    "sector": block["sector"].iloc[-1],
                    "theme": block["theme"].iloc[-1],
                    "start_date": block["date"].iloc[0].date().isoformat(),
                    "end_date": end_date,
                    "source": source,
                }
            )
    return pd.DataFrame(rows).sort_values(["instrument", "start_date"])


def build(
    raw_dir: Path,
    output_path: Path,
    report_path: Path,
    forward_fill_to: str | None = None,
) -> dict:
    files = sorted(raw_dir.glob("**/*.csv"))
    if not files:
        raise FileNotFoundError(f"no csv files under {raw_dir}")
    parts = []
    for pos, path in enumerate(files, start=1):
        parts.append(read_one(path))
        if pos % 50 == 0:
            print(f"[industry pit] read {pos}/{len(files)}", flush=True)
    daily = pd.concat(parts, ignore_index=True)
    before = len(daily)
    daily = daily.dropna(subset=["instrument", "industry"])
    daily = daily.drop_duplicates(
        ["date", "instrument", "industry", "sector", "theme"],
        keep="last",
    )
    intervals = compress_intervals(daily, forward_fill_to=forward_fill_to)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(output_path, index=False, encoding="utf-8-sig")
    report = {
        "status": "industry_pit_metadata_built",
        "raw_dir": str(raw_dir),
        "output": str(output_path),
        "source_files": len(files),
        "daily_rows_before_dedup": int(before),
        "daily_rows_after_dedup": int(len(daily)),
        "interval_rows": int(len(intervals)),
        "instrument_count": int(intervals["instrument"].nunique()),
        "date_start": str(daily["date"].min().date()),
        "date_end": str(daily["date"].max().date()),
        "forward_fill_to": forward_fill_to,
        "forward_fill_policy": (
            "last interval per instrument extended to forward_fill_to for "
            "concentration/risk controls only; do not use as alpha feature"
            if forward_fill_to
            else None
        ),
        "forward_filled_interval_count": int(
            intervals["source"].astype(str).str.contains("forward_filled").sum()
        ),
        "industry_count": int(intervals["industry"].nunique()),
        "industries": sorted(intervals["industry"].dropna().unique().tolist()),
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--report", default=str(HERE / "outputs" / "industry_pit_build_report.json")
    )
    parser.add_argument(
        "--forward-fill-to",
        default=None,
        help="Extend each instrument's final interval to this date for risk controls only.",
    )
    args = parser.parse_args()
    report = build(
        Path(args.raw_dir).resolve(),
        Path(args.output).resolve(),
        Path(args.report).resolve(),
        args.forward_fill_to,
    )
    print(
        f"[industry pit] status={report['status']} instruments={report['instrument_count']} "
        f"intervals={report['interval_rows']} date_end={report['date_end']}"
    )


if __name__ == "__main__":
    main()
