from __future__ import annotations

import argparse
import json
from pathlib import Path

from concentration_constraints import load_group_metadata


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_POOL = DATA / "cn_data_1min_pool" / "instruments" / "all.txt"
DEFAULT_METADATA = DATA / "industry_theme_pit.csv"


def read_pool(path: Path) -> list[str]:
    return [
        line.split()[0].upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(metadata_path: Path, pool_path: Path, output_path: Path, group_col: str) -> dict:
    pool = read_pool(pool_path)
    if not metadata_path.exists():
        report = {
            "status": "missing_metadata",
            "metadata_path": str(metadata_path),
            "pool_path": str(pool_path),
            "pool_size": len(pool),
            "required_columns": {
                "instrument": ["instrument", "symbol", "ts_code", "code"],
                "group": [group_col, "industry", "sector", "theme", "concept"],
                "optional_pit": ["start_date/effective_start", "end_date/effective_end"],
            },
            "gate_passed": False,
            "deployment_allowed": False,
        }
    else:
        metadata = load_group_metadata(metadata_path, group_col)
        covered = set(metadata["instrument"]) & set(pool)
        groups = metadata[metadata["instrument"].isin(pool)]["group"].dropna().unique().tolist()
        report = {
            "status": "validated" if len(covered) == len(pool) else "incomplete_coverage",
            "metadata_path": str(metadata_path),
            "pool_path": str(pool_path),
            "pool_size": len(pool),
            "covered": len(covered),
            "missing": sorted(set(pool) - covered),
            "group_count": len(groups),
            "groups": sorted(map(str, groups)),
            "is_pit": bool(metadata["is_pit"].any()),
            "gate_passed": len(covered) == len(pool) and bool(metadata["is_pit"].any()),
            "deployment_allowed": False,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--pool", default=str(DEFAULT_POOL))
    parser.add_argument("--group-col", default="industry")
    parser.add_argument(
        "--output", default=str(HERE / "outputs" / "concentration_metadata_audit.json")
    )
    args = parser.parse_args()
    report = audit(
        Path(args.metadata).resolve(),
        Path(args.pool).resolve(),
        Path(args.output).resolve(),
        args.group_col,
    )
    print(
        f"[concentration metadata] status={report['status']} "
        f"gate_passed={report['gate_passed']}"
    )


if __name__ == "__main__":
    main()
