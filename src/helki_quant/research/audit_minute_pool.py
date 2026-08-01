from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
RAW_MINUTE = DATA / "A_Stock_1min"
POOL_INST = DATA / "cn_data_1min_pool" / "instruments" / "all.txt"


def read_pool(path: Path) -> list[str]:
    return [
        line.split()[0].lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


MinuteSource = Path | tuple[Path, str]


def source_label(source: MinuteSource) -> str:
    if isinstance(source, tuple):
        return f"{source[0]}::{source[1]}"
    return str(source)


def files_for_instrument(inst: str, raw_dir: Path) -> list[MinuteSource]:
    files = []
    for year_dir in sorted(raw_dir.glob("*_1min")):
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
                print(f"[minute audit] bad zip skipped: {zip_path}", flush=True)
    return sorted(set(files), key=source_label)


def read_file_dates(path: MinuteSource) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    try:
        if isinstance(path, tuple):
            with zipfile.ZipFile(path[0]) as archive:
                with archive.open(path[1]) as handle:
                    frame = pd.read_csv(handle, usecols=["时间"], encoding="utf-8")
        else:
            frame = pd.read_csv(path, usecols=["时间"], encoding="utf-8")
    except UnicodeDecodeError:
        if isinstance(path, tuple):
            with zipfile.ZipFile(path[0]) as archive:
                with archive.open(path[1]) as handle:
                    frame = pd.read_csv(handle, usecols=["时间"], encoding="gbk")
        else:
            frame = pd.read_csv(path, usecols=["时间"], encoding="gbk")
    except Exception:
        return None, None, 0
    if frame.empty:
        return None, None, 0
    dates = pd.to_datetime(frame["时间"], errors="coerce").dropna()
    if dates.empty:
        return None, None, 0
    return dates.min(), dates.max(), len(dates)


def audit(output_path: Path) -> dict:
    instruments = read_pool(POOL_INST)
    rows = []
    for pos, inst in enumerate(instruments, start=1):
        files = files_for_instrument(inst, RAW_MINUTE)
        first = None
        last = None
        rows_count = 0
        year_counts: dict[str, int] = {}
        for path in files:
            f_first, f_last, count = read_file_dates(path)
            rows_count += count
            source_path = path[0] if isinstance(path, tuple) else path
            year = next((part.name for part in source_path.parents if part.name.endswith("_1min")), "unknown")
            year_counts[year] = year_counts.get(year, 0) + 1
            if f_first is not None:
                first = f_first if first is None else min(first, f_first)
            if f_last is not None:
                last = f_last if last is None else max(last, f_last)
        rows.append(
            {
                "instrument": inst,
                "file_count": len(files),
                "row_count": rows_count,
                "first": str(first) if first is not None else None,
                "last": str(last) if last is not None else None,
                "has_2026": any("2026_1min" in source_label(path) for path in files),
                "year_file_counts": year_counts,
            }
        )
        if pos % 10 == 0:
            print(f"[minute audit] {pos}/{len(instruments)}", flush=True)
    last_dates = pd.to_datetime([row["last"] for row in rows if row["last"]])
    report = {
        "status": "minute_pool_raw_audit",
        "pool_size": len(instruments),
        "with_files": sum(row["file_count"] > 0 for row in rows),
        "with_2026_files": sum(row["has_2026"] for row in rows),
        "earliest_last": str(last_dates.min()) if len(last_dates) else None,
        "latest_last": str(last_dates.max()) if len(last_dates) else None,
        "details": rows,
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default=str(HERE / "outputs" / "minute_pool_raw_audit.json")
    )
    args = parser.parse_args()
    report = audit(Path(args.output).resolve())
    print(
        f"[minute audit] pool={report['pool_size']} with_files={report['with_files']} "
        f"with_2026={report['with_2026_files']} last={report['earliest_last']}..{report['latest_last']}"
    )


if __name__ == "__main__":
    main()
