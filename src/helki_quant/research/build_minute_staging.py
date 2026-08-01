from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MULTI_LAYER = HERE.parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
RAW_MINUTE = DATA / "A_Stock_1min"
SOURCE_DAY_PROVIDER = DATA / "cn_data_pool"
POOL_INST = DATA / "cn_data_1min_pool" / "instruments" / "all.txt"
DUMP_SCRIPT = REPO_ROOT / "scripts" / "dump_bin.py"
PRECOMPUTE_SCRIPT_DIR = REPO_ROOT / "scripts"
if str(PRECOMPUTE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PRECOMPUTE_SCRIPT_DIR))
if str(MULTI_LAYER) not in sys.path:
    sys.path.insert(0, str(MULTI_LAYER))

from precompute_intra_real import FACTOR_NAMES, compute_daily_factors  # noqa: E402
from realtime_output import run_streaming, setup_realtime_output  # noqa: E402


DEFAULT_STAGE_CSV = DATA / "_research_1min_pool_csv_2026"
DEFAULT_MINUTE_PROVIDER = DATA / "cn_data_1min_pool_research_2026"
DEFAULT_INNER_DAY_PROVIDER = DATA / "cn_data_pool_inner_research_2026"

COL_MAP = {
    "时间": "date",
    "datetime": "date",
    "date": "date",
    "代码": "instrument",
    "instrument": "instrument",
    "开盘价": "open",
    "open": "open",
    "收盘价": "close",
    "close": "close",
    "最高价": "high",
    "high": "high",
    "最低价": "low",
    "low": "low",
    "成交量": "volume",
    "volume": "volume",
    "成交额": "amount",
    "amount": "amount",
    "total_turnover": "amount",
}
REQUIRED = ["date", "open", "high", "low", "close", "volume", "amount"]


def read_pool(pool_file: Path = POOL_INST) -> list[str]:
    return [
        line.split()[0].lower()
        for line in pool_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


MinuteSource = Path | tuple[Path, str]
MinuteSourceIndex = dict[str, list[MinuteSource]]


def source_label(source: MinuteSource) -> str:
    if isinstance(source, tuple):
        return f"{source[0]}::{source[1]}"
    return str(source)


def symbol_key_from_name(name: str) -> str | None:
    stem = Path(name).stem.lower()
    if len(stem) >= 8 and stem[:2] in {"sh", "sz"} and stem[2:8].isdigit():
        return stem[:8]
    return None


def build_minute_source_index(raw_minute_dir: Path = RAW_MINUTE) -> MinuteSourceIndex:
    index: MinuteSourceIndex = {}
    for csv_path in raw_minute_dir.rglob("*.csv"):
        key = symbol_key_from_name(csv_path.name)
        if key:
            index.setdefault(key, []).append(csv_path)
    for zip_path in raw_minute_dir.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.namelist():
                    key = symbol_key_from_name(Path(member).name)
                    if key:
                        index.setdefault(key, []).append((zip_path, member))
        except zipfile.BadZipFile:
            print(f"[minute staging] bad zip skipped: {zip_path}", flush=True)
    for key in index:
        index[key] = sorted(set(index[key]), key=source_label)
    return index


def source_overlaps_date_range(source: MinuteSource, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    label = source_label(source)
    day_matches = re.findall(r"(?<!\d)(20\d{6})(?!\d)", label)
    if day_matches:
        dates = pd.to_datetime(day_matches, format="%Y%m%d", errors="coerce")
        return any(pd.notna(date) and start.normalize() <= date.normalize() <= end.normalize() for date in dates)
    year_matches = re.findall(r"(?<!\d)(20\d{2})(?!\d)", label)
    if year_matches:
        start_year = int(start.year)
        end_year = int(end.year)
        return any(start_year <= int(year) <= end_year for year in year_matches)
    return True


def files_for_instrument(
    inst: str,
    raw_minute_dir: Path = RAW_MINUTE,
    source_index: MinuteSourceIndex | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> list[MinuteSource]:
    if source_index is not None:
        files = list(source_index.get(inst.lower(), []))
        if start is not None and end is not None:
            files = [source for source in files if source_overlaps_date_range(source, start, end)]
        return files
    files = []
    for year_dir in sorted(path for path in raw_minute_dir.iterdir() if path.is_dir()):
        if not year_dir.is_dir():
            continue
        files.extend(year_dir.glob(f"{inst}_*.csv"))
        files.extend(year_dir.glob(f"{inst.upper()}_*.csv"))
        files.extend(year_dir.glob(f"**/{inst}.csv"))
        files.extend(year_dir.glob(f"**/{inst.upper()}.csv"))
        wanted = {f"{inst}.csv", f"{inst.upper()}.csv"}
        for zip_path in sorted(year_dir.glob("**/*.zip")):
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    for member in archive.namelist():
                        if Path(member).name in wanted:
                            files.append((zip_path, member))
            except zipfile.BadZipFile:
                print(f"[minute staging] bad zip skipped: {zip_path}", flush=True)
    files = sorted(set(files), key=source_label)
    if start is not None and end is not None:
        files = [source for source in files if source_overlaps_date_range(source, start, end)]
    return files


def read_one(path: MinuteSource) -> pd.DataFrame:
    close_after_read = None
    try:
        if isinstance(path, tuple):
            archive = zipfile.ZipFile(path[0])
            close_after_read = archive
            handle = archive.open(path[1])
            frame = pd.read_csv(handle, dtype={"代码": str}, encoding="utf-8")
        else:
            frame = pd.read_csv(path, dtype={"代码": str}, encoding="utf-8")
    except UnicodeDecodeError:
        if isinstance(path, tuple):
            if close_after_read is not None:
                close_after_read.close()
            archive = zipfile.ZipFile(path[0])
            close_after_read = archive
            handle = archive.open(path[1])
            frame = pd.read_csv(handle, dtype={"代码": str}, encoding="gbk")
        else:
            frame = pd.read_csv(path, dtype={"代码": str}, encoding="gbk")
    finally:
        if close_after_read is not None:
            close_after_read.close()
    keep = [col for col in COL_MAP if col in frame.columns]
    frame = frame[keep].rename(columns=COL_MAP)
    if not all(col in frame for col in REQUIRED):
        raise ValueError(f"missing required columns in {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    for col in REQUIRED[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame[REQUIRED]


def read_minute_concat(inst: str, raw_minute_dir: Path = RAW_MINUTE) -> pd.DataFrame:
    parts = [read_one(path) for path in files_for_instrument(inst, raw_minute_dir)]
    if not parts:
        raise FileNotFoundError(f"no raw minute files found for {inst}")
    frame = pd.concat(parts, ignore_index=True)
    frame = (
        frame.drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    frame["factor"] = 1.0
    return frame


def read_stage_csv(inst: str, stage_dir: Path) -> pd.DataFrame:
    path = stage_dir / f"{inst}.csv"
    if not path.exists():
        raise FileNotFoundError(f"staged minute csv not found for {inst}: {path}")
    frame = pd.read_csv(path, parse_dates=["date"])
    missing = [col for col in REQUIRED if col not in frame.columns]
    if missing:
        raise ValueError(f"missing staged columns in {path}: {missing}")
    for col in REQUIRED[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "factor" not in frame.columns:
        frame["factor"] = 1.0
    return frame


def write_stage_csv(inst: str, frame: pd.DataFrame, stage_dir: Path) -> None:
    out = frame.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(stage_dir / f"{inst}.csv", index=False)


def read_day_calendar(day_provider: Path) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, int]]:
    calendar = pd.read_csv(
        day_provider / "calendars" / "day.txt",
        header=None,
        names=["date"],
        parse_dates=["date"],
    )["date"].tolist()
    return calendar, {date: idx for idx, date in enumerate(calendar)}


def write_qlib_bin(values: np.ndarray, start_idx: int, bin_path: Path) -> None:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    np.hstack([np.array([start_idx], dtype=np.float32), values.astype(np.float32)]).tofile(
        str(bin_path.resolve())
    )


def write_true_minute_fields(
    inst: str,
    minute_frame: pd.DataFrame,
    day_provider: Path,
    calendar: list[pd.Timestamp],
    cal_index: dict[pd.Timestamp, int],
) -> dict:
    source = minute_frame.rename(columns={"date": "datetime"})
    daily = compute_daily_factors(source)
    if daily.empty:
        raise ValueError(f"empty true-minute aggregation for {inst}")
    daily.index = pd.to_datetime(daily.index).normalize()
    valid_dates = [date for date in daily.index if date in cal_index]
    daily = daily.loc[valid_dates]
    start_idx = cal_index[daily.index.min()]
    end_idx = cal_index[daily.index.max()]
    aligned = daily.reindex(calendar[start_idx : end_idx + 1])
    feature_dir = day_provider / "features" / inst
    for factor in FACTOR_NAMES:
        write_qlib_bin(
            aligned[factor].values.astype(np.float32),
            start_idx,
            feature_dir / f"{factor.lower()}.day.bin",
        )
    return {
        "first": str(daily.index.min().date()),
        "last": str(daily.index.max().date()),
        "days": int(len(daily)),
    }


def run_dump(stage_dir: Path, minute_provider: Path, max_workers: int) -> None:
    setup_realtime_output()
    command = [
        sys.executable,
        str(DUMP_SCRIPT),
        "dump_all",
        "--data_path",
        str(stage_dir),
        "--qlib_dir",
        str(minute_provider),
        "--freq",
        "1min",
        "--date_field_name",
        "date",
        "--symbol_field_name",
        "symbol",
        "--max_workers",
        str(max_workers),
    ]
    rc = run_streaming(command, cwd=REPO_ROOT, prefix="[dump_bin] ")
    if rc != 0:
        raise subprocess.CalledProcessError(rc, command)


def refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing staging/output paths: "
            + "; ".join(existing)
        )


def has_all_factor_bins(provider: Path, inst: str) -> bool:
    feature_dir = provider / "features" / inst
    return all((feature_dir / f"{factor.lower()}.day.bin").exists() for factor in FACTOR_NAMES)


def write_pool_aliases(provider: Path, instruments: list[str], alias: str) -> None:
    inst_dir = provider / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    existing_ranges: dict[str, str] = {}
    for candidate in (inst_dir / "all.txt", inst_dir / f"{alias}.txt"):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 3:
                existing_ranges[parts[0].upper()] = f"{parts[1]}\t{parts[2]}"
    lines = []
    for inst in instruments:
        symbol = inst.upper()
        date_range = existing_ranges.get(symbol, "2021-01-01\t2026-06-05")
        lines.append(f"{symbol}\t{date_range}")
    text = "\n".join(lines) + "\n"
    (inst_dir / f"{alias}.txt").write_text(text, encoding="utf-8")
    (inst_dir / "all.txt").write_text(text, encoding="utf-8")


def build(
    stage_dir: Path,
    minute_provider: Path,
    inner_day_provider: Path,
    source_day_provider: Path,
    pool_file: Path,
    raw_minute_dir: Path,
    pool_alias: str,
    report_path: Path,
    max_workers: int,
    skip_dump: bool,
    reuse_stage: bool,
    resume_inner_day: bool,
    skip_completed: bool,
) -> dict:
    if reuse_stage:
        if not stage_dir.exists():
            raise FileNotFoundError(f"--reuse-stage requires existing stage-dir: {stage_dir}")
        refuse_existing(([] if resume_inner_day else [inner_day_provider]) + ([] if skip_dump else [minute_provider]))
    else:
        refuse_existing([stage_dir, minute_provider] + ([] if resume_inner_day else [inner_day_provider]))
    instruments = read_pool(pool_file)
    if not reuse_stage:
        stage_dir.mkdir(parents=True)
    if inner_day_provider.exists():
        if not resume_inner_day:
            raise FileExistsError(f"inner day provider exists: {inner_day_provider}")
    else:
        shutil.copytree(source_day_provider, inner_day_provider)
    write_pool_aliases(inner_day_provider, instruments, pool_alias)
    calendar, cal_index = read_day_calendar(inner_day_provider)
    rows = []
    for pos, inst in enumerate(instruments, start=1):
        if resume_inner_day and skip_completed and has_all_factor_bins(inner_day_provider, inst):
            rows.append({"instrument": inst, "skipped_completed": True})
            print(f"[minute staging] {pos}/{len(instruments)} {inst} skipped completed", flush=True)
            continue
        frame = read_stage_csv(inst, stage_dir) if reuse_stage else read_minute_concat(inst, raw_minute_dir)
        if not reuse_stage:
            write_stage_csv(inst, frame, stage_dir)
        coverage = write_true_minute_fields(
            inst, frame, inner_day_provider, calendar, cal_index
        )
        rows.append(
            {
                "instrument": inst,
                "minute_rows": int(len(frame)),
                "minute_start": str(frame["date"].min()),
                "minute_end": str(frame["date"].max()),
                "true_minute": coverage,
            }
        )
        print(f"[minute staging] {pos}/{len(instruments)} {inst} {coverage['last']}", flush=True)
    if not skip_dump:
        run_dump(stage_dir, minute_provider, max_workers)
        write_pool_aliases(minute_provider, instruments, pool_alias)
    report = {
        "status": "minute_and_inner_day_staging_built",
        "stage_csv": str(stage_dir),
        "minute_provider": str(minute_provider),
        "inner_day_provider": str(inner_day_provider),
        "source_day_provider": str(source_day_provider),
        "pool_file": str(pool_file),
        "raw_minute_dir": str(raw_minute_dir),
        "pool_alias": pool_alias,
        "minute_dump_skipped": skip_dump,
        "stage_reused": reuse_stage,
        "inner_day_resumed": resume_inner_day,
        "completed_skipped": skip_completed,
        "instruments": len(instruments),
        "details": rows,
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    setup_realtime_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE_CSV))
    parser.add_argument("--minute-provider", default=str(DEFAULT_MINUTE_PROVIDER))
    parser.add_argument("--inner-day-provider", default=str(DEFAULT_INNER_DAY_PROVIDER))
    parser.add_argument("--source-day-provider", default=str(SOURCE_DAY_PROVIDER))
    parser.add_argument("--pool-file", default=str(POOL_INST))
    parser.add_argument("--raw-minute-dir", default=str(RAW_MINUTE))
    parser.add_argument("--pool-alias", default="pool80")
    parser.add_argument(
        "--report", default=str(HERE / "outputs" / "minute_staging_build_report.json")
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--skip-dump", action="store_true")
    parser.add_argument("--reuse-stage", action="store_true")
    parser.add_argument("--resume-inner-day", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    report = build(
        Path(args.stage_dir).resolve(),
        Path(args.minute_provider).resolve(),
        Path(args.inner_day_provider).resolve(),
        Path(args.source_day_provider).resolve(),
        Path(args.pool_file).resolve(),
        Path(args.raw_minute_dir).resolve(),
        args.pool_alias,
        Path(args.report).resolve(),
        args.max_workers,
        args.skip_dump,
        args.reuse_stage,
        args.resume_inner_day,
        args.skip_completed,
    )
    print(
        f"[minute staging] instruments={report['instruments']} "
        f"inner_day={report['inner_day_provider']} minute={report['minute_provider']}"
    )


if __name__ == "__main__":
    main()
