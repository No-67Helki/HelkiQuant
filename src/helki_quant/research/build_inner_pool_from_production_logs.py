from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


def normalize_symbol(symbol: object) -> str:
    return str(symbol).strip().upper()


def read_ranked_candidates(log_dir: Path) -> list[str]:
    holdings_path = log_dir / "holdings.csv"
    targets_path = log_dir / "targets.csv"
    if not holdings_path.exists():
        raise FileNotFoundError(holdings_path)
    if not targets_path.exists():
        raise FileNotFoundError(targets_path)

    holdings = pd.read_csv(holdings_path)
    targets = pd.read_csv(targets_path)
    holdings["instrument"] = holdings["instrument"].map(normalize_symbol)
    targets["instrument"] = targets["instrument"].map(normalize_symbol)

    held = holdings[holdings["shares"] > 0]["instrument"].value_counts()
    # The pool must include every selected target, especially names that were
    # unmapped by the old minute cache. Filtering to mapped/allocated rows here
    # makes the missing-data problem self-perpetuating.
    selected_targets = targets.dropna(subset=["instrument"]).copy()
    mapped = pd.to_numeric(
        selected_targets.get("mapped", pd.Series(0, index=selected_targets.index)),
        errors="coerce",
    ).fillna(0)
    target_shares = pd.to_numeric(
        selected_targets.get("target_shares", pd.Series(0, index=selected_targets.index)),
        errors="coerce",
    ).fillna(0)
    target_days = selected_targets["instrument"].value_counts()
    unmapped_days = selected_targets.loc[mapped.ne(1), "instrument"].value_counts()
    allocated_days = selected_targets.loc[target_shares.gt(0), "instrument"].value_counts()
    avg_rank = selected_targets.groupby("instrument")["rank"].mean()
    frame = pd.DataFrame(
        {
            "held_days": held,
            "target_days": target_days,
            "unmapped_days": unmapped_days,
            "allocated_days": allocated_days,
            "avg_rank": avg_rank,
        }
    ).fillna(
        {
            "held_days": 0,
            "target_days": 0,
            "unmapped_days": 0,
            "allocated_days": 0,
        }
    )
    frame["avg_rank"] = frame["avg_rank"].fillna(999999)
    frame["score"] = (
        frame["held_days"] * 10
        + frame["target_days"] * 2
        + frame["unmapped_days"]
        + frame["allocated_days"]
        - frame["avg_rank"] / 10000.0
    )
    return frame.sort_values(
        ["score", "unmapped_days", "held_days", "target_days"],
        ascending=False,
    ).index.tolist()


def symbol_key_from_name(name: str) -> str | None:
    stem = Path(name).stem.lower()
    if len(stem) >= 8 and stem[:2] in {"sh", "sz"} and stem[2:8].isdigit():
        return stem[:8]
    return None


def minute_availability_index(raw_minute_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for csv_path in raw_minute_dir.rglob("*.csv"):
        key = symbol_key_from_name(csv_path.name)
        if key:
            counts[key] = counts.get(key, 0) + 1
    for zip_path in raw_minute_dir.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.namelist():
                    key = symbol_key_from_name(Path(member).name)
                    if key:
                        counts[key] = counts.get(key, 0) + 1
        except zipfile.BadZipFile:
            print(f"[inner pool] bad zip skipped: {zip_path}", flush=True)
    return counts


def build_pool(
    log_dir: Path,
    raw_minute_dir: Path,
    output_pool: Path,
    audit_output: Path,
    max_symbols: int | None,
) -> dict:
    candidates = read_ranked_candidates(log_dir)
    availability = minute_availability_index(raw_minute_dir)
    rows = []
    selected = []
    for pos, symbol in enumerate(candidates, start=1):
        inst = symbol.lower()
        source_count = int(availability.get(inst, 0))
        has_minute = source_count > 0
        if has_minute and (max_symbols is None or len(selected) < max_symbols):
            selected.append(symbol)
        rows.append(
            {
                "rank": pos,
                "instrument": symbol,
                "has_minute": has_minute,
                "source_count": source_count,
                "selected": has_minute and symbol in selected,
            }
        )
        if pos % 50 == 0:
            print(
                f"[inner pool] audited={pos}/{len(candidates)} selected={len(selected)}",
                flush=True,
            )
    output_pool.parent.mkdir(parents=True, exist_ok=True)
    output_pool.write_text(
        "\n".join(f"{symbol}\t2021-01-01\t2026-06-05" for symbol in selected) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "inner_pool_from_production_logs_built",
        "log_dir": str(log_dir.resolve()),
        "raw_minute_dir": str(raw_minute_dir.resolve()),
        "output_pool": str(output_pool.resolve()),
        "candidates": len(candidates),
        "with_minute": int(sum(row["has_minute"] for row in rows)),
        "selected": len(selected),
        "max_symbols": max_symbols,
        "details": rows,
        "deployment_allowed": False,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--raw-minute-dir", default="data/A_Stock_1min")
    parser.add_argument("--output-pool", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args()
    report = build_pool(
        Path(args.log_dir).resolve(),
        Path(args.raw_minute_dir).resolve(),
        Path(args.output_pool).resolve(),
        Path(args.audit_output).resolve(),
        args.max_symbols,
    )
    print(
        f"[inner pool] candidates={report['candidates']} "
        f"with_minute={report['with_minute']} selected={report['selected']} "
        f"-> {report['output_pool']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
