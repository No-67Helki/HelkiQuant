from __future__ import annotations

import argparse
import json
from pathlib import Path

import qlib
from qlib.data import D


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_NEW = REPO_ROOT / "data" / "cn_data_research_pit"
DEFAULT_OLD = REPO_ROOT / "data" / "cn_data_pool"


def read_instruments(provider: Path) -> dict[str, tuple[str, str]]:
    rows = {}
    for line in (provider / "instruments" / "all.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        symbol, start, end = line.split("\t")
        rows[symbol.upper()] = (start, end)
    return rows


def validate(new_provider: Path, old_provider: Path, output_path: Path) -> dict:
    calendar = (new_provider / "calendars" / "day.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    new = read_instruments(new_provider)
    old = read_instruments(old_provider)
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))

    qlib.init(provider_uri=str(new_provider), region="cn")
    sample = D.features(
        ["SZ301536", "SZ300001", "SZ301611"],
        ["$close", "$volume"],
        start_time="2024-04-01",
        end_time="2024-04-10",
        freq="day",
    )
    missing_feature_dirs = [
        symbol
        for symbol in new
        if not (new_provider / "features" / symbol.lower()).is_dir()
    ]
    report = {
        "status": "validated" if not missing_feature_dirs and len(sample) else "failed",
        "new_provider": str(new_provider),
        "old_provider": str(old_provider),
        "calendar_count": len(calendar),
        "calendar_start": calendar[0],
        "calendar_end": calendar[-1],
        "new_instrument_count": len(new),
        "old_instrument_count": len(old),
        "added_instrument_count": len(added),
        "removed_instrument_count": len(removed),
        "added_instruments": added,
        "removed_instruments": removed,
        "target_present": "SZ301536" in new,
        "sample_query_rows": len(sample),
        "sample_query_instruments": sorted(
            sample.index.get_level_values("instrument").unique().tolist()
        ),
        "missing_feature_directories": missing_feature_dirs,
        "residual_warning": (
            "The broad pool removes latest-date selection from the local raw library, "
            "but the raw library may still omit delisted or inactive historical names."
        ),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-provider", default=str(DEFAULT_NEW))
    parser.add_argument("--old-provider", default=str(DEFAULT_OLD))
    parser.add_argument(
        "--output", default=str(HERE / "outputs" / "pit_daily_validation.json")
    )
    args = parser.parse_args()
    report = validate(
        Path(args.new_provider).resolve(),
        Path(args.old_provider).resolve(),
        Path(args.output).resolve(),
    )
    print(
        f"[PIT validation] status={report['status']} "
        f"instruments={report['new_instrument_count']} "
        f"added={report['added_instrument_count']} "
        f"sample_rows={report['sample_query_rows']}"
    )


if __name__ == "__main__":
    main()
