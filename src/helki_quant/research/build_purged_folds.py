from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from splitters import PurgedWalkForwardSplitter


def build_purged_folds(
    calendar_provider: Path,
    output_path: Path,
    *,
    start: str,
    end: str,
    min_train_days: int,
    valid_days: int,
    test_days: int,
    step_days: int,
    purge_days: int,
    embargo_days: int,
) -> dict:
    calendar_path = calendar_provider / "calendars" / "day.txt"
    if not calendar_path.exists():
        raise FileNotFoundError(f"missing calendar: {calendar_path}")
    dates = pd.read_csv(calendar_path, header=None, names=["date"])["date"]
    dates = pd.to_datetime(dates, errors="coerce").dropna()
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    dates = dates.drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError(f"calendar is empty between {start} and {end}")

    splitter = PurgedWalkForwardSplitter(
        min_train_days=min_train_days,
        valid_days=valid_days,
        test_days=test_days,
        step_days=step_days,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )
    folds = [fold.as_dict() for fold in splitter.split(dates)]
    if not folds:
        raise ValueError(
            "no folds fit the requested calendar and windows: "
            f"dates={len(dates)} min_train={min_train_days} valid={valid_days} "
            f"test={test_days} purge={purge_days} embargo={embargo_days}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(folds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "purged_walk_forward_folds_built",
        "calendar_provider": str(calendar_provider.resolve()),
        "calendar_path": str(calendar_path.resolve()),
        "requested_start": start,
        "requested_end": end,
        "calendar_rows": int(len(dates)),
        "calendar_start": str(dates.iloc[0].date()),
        "calendar_end": str(dates.iloc[-1].date()),
        "windows": {
            "min_train_days": min_train_days,
            "valid_days": valid_days,
            "test_days": test_days,
            "step_days": step_days,
            "purge_days": purge_days,
            "embargo_days": embargo_days,
        },
        "fold_count": len(folds),
        "first_fold": folds[0],
        "last_fold": folds[-1],
        "folds_path": str(output_path.resolve()),
        "deployment_allowed": False,
    }
    manifest_path = output_path.with_name(f"{output_path.stem}.manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calendar-provider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--min-train-days", type=int, default=500)
    parser.add_argument("--valid-days", type=int, default=120)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--purge-days", type=int, default=21)
    parser.add_argument("--embargo-days", type=int, default=5)
    args = parser.parse_args()

    manifest = build_purged_folds(
        args.calendar_provider.resolve(),
        args.output.resolve(),
        start=args.start,
        end=args.end,
        min_train_days=args.min_train_days,
        valid_days=args.valid_days,
        test_days=args.test_days,
        step_days=args.step_days,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
    )
    print(
        "[purged folds] "
        f"calendar={manifest['calendar_start']}..{manifest['calendar_end']} "
        f"rows={manifest['calendar_rows']} folds={manifest['fold_count']} "
        f"output={manifest['folds_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
