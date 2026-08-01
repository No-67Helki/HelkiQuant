from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from concentration_constraints import ConcentrationRules, groups_on_date, load_group_metadata, select_with_group_cap
from evaluate_daily_topk_grid import load_middle_predictions
from minute_mapped_topk_replay import MappedProfile, MappedReplayConfig, default_profiles, prepare_daily_frame
from universe import load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
RAW_MINUTE = DATA / "A_Stock_1min"
DEFAULT_MIDDLE = HERE / "outputs" / "oof_combined" / "middle_oof_pit_de2_srfs_es.csv"
DEFAULT_GROUP_METADATA = DATA / "industry_theme_pit.csv"


def inst_to_raw_symbol(instrument: str) -> str:
    text = str(instrument).strip().lower()
    if text.startswith(("sh", "sz")):
        return text[:2] + text[-6:]
    code = text.replace(".", "")[-6:]
    return ("sh" if code.startswith("68") else "sz") + code


def symbol_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem.lower()
    if len(stem) < 8:
        return None
    prefix = stem[:2]
    code = stem[2:8]
    if prefix not in {"sh", "sz"} or not code.isdigit():
        return None
    return prefix + code


def index_raw_availability(target_symbols: set[str]) -> dict[str, dict]:
    availability = {
        symbol: {
            "instrument": symbol,
            "file_count": 0,
            "has_raw_minute": False,
            "has_2026_raw_minute": False,
            "first_file": None,
            "last_file": None,
        }
        for symbol in target_symbols
    }
    scanned = 0
    for dirpath, _, filenames in os.walk(RAW_MINUTE):
        is_2026 = "2026_1min" in dirpath
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue
            symbol = symbol_from_filename(filename)
            if symbol not in availability:
                continue
            path = str(Path(dirpath) / filename)
            row = availability[symbol]
            row["file_count"] += 1
            row["has_raw_minute"] = True
            row["has_2026_raw_minute"] = bool(row["has_2026_raw_minute"] or is_2026)
            if row["first_file"] is None or path < row["first_file"]:
                row["first_file"] = path
            if row["last_file"] is None or path > row["last_file"]:
                row["last_file"] = path
        scanned += 1
        if scanned % 100 == 0:
            found = sum(1 for row in availability.values() if row["has_2026_raw_minute"])
            print(
                f"[topk minute audit] scanned_dirs={scanned} "
                f"2026_available={found}/{len(target_symbols)}",
                flush=True,
            )
    return availability


def select_profile_targets(
    frame: pd.DataFrame,
    profile: MappedProfile,
    group_metadata: pd.DataFrame,
    cfg: MappedReplayConfig,
) -> list[dict]:
    rows: list[dict] = []
    previous: set[str] = set()
    buffer_k = profile.top_k * cfg.buffer_multiple
    by_date = {date: part.copy() for date, part in frame.groupby("trade_date", sort=True)}
    for day_no, (trade_date, day) in enumerate(by_date.items()):
        if day_no % profile.rebalance_every != 0:
            continue
        eligible = day[day["eligible"].fillna(False)].sort_values("middle", ascending=False)
        ranked = eligible["instrument"].tolist()
        groups = groups_on_date(group_metadata, trade_date)
        selected = select_with_group_cap(
            ranked,
            previous,
            top_k=profile.top_k,
            buffer_k=buffer_k,
            groups=groups,
            rules=ConcentrationRules(max_group_fraction=profile.industry_cap),
        )
        previous = set(selected)
        rows.append(
            {
                "profile": profile.name,
                "trade_date": str(pd.Timestamp(trade_date).date()),
                "selected": selected,
            }
        )
    return rows


def audit(
    middle_path: Path,
    group_metadata_path: Path,
    output_path: Path,
    pool_output: Path,
    start_signal: str,
    end_signal: str,
) -> dict:
    predictions = load_middle_predictions(middle_path)
    predictions = predictions[predictions["datetime"].between(start_signal, end_signal)].copy()
    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(DATA / "A_Stock_daily_qfq", instruments, start="2022-01-04", end="2026-04-28")
    group_metadata = load_group_metadata(group_metadata_path, "industry")
    cfg = MappedReplayConfig()
    frames = {
        profile.name: prepare_daily_frame(predictions, prices, profile, cfg)
        for profile in default_profiles()
    }
    selections: list[dict] = []
    for profile in default_profiles():
        selections.extend(select_profile_targets(frames[profile.name], profile, group_metadata, cfg))
    target_symbols = sorted({inst_to_raw_symbol(inst) for row in selections for inst in row["selected"]})
    availability = list(index_raw_availability(set(target_symbols)).values())
    available = {row["instrument"] for row in availability if row["has_raw_minute"]}
    available_2026 = {row["instrument"] for row in availability if row["has_2026_raw_minute"]}
    for row in selections:
        selected = [inst_to_raw_symbol(inst) for inst in row["selected"]]
        row["available_count"] = sum(inst in available for inst in selected)
        row["available_2026_count"] = sum(inst in available_2026 for inst in selected)
        row["available_fraction"] = row["available_count"] / len(selected) if selected else 0.0
        row["available_2026_fraction"] = row["available_2026_count"] / len(selected) if selected else 0.0
    pool_output.parent.mkdir(parents=True, exist_ok=True)
    pool_output.write_text("\n".join(sorted(available_2026)) + "\n", encoding="utf-8")
    report = {
        "status": "topk_minute_pool_audit_research_only",
        "middle_prediction": str(middle_path),
        "group_metadata": str(group_metadata_path),
        "window": {"start_signal": start_signal, "end_signal": end_signal},
        "profiles": [asdict(profile) for profile in default_profiles()],
        "target_union_size": len(target_symbols),
        "raw_available_size": len(available),
        "raw_2026_available_size": len(available_2026),
        "pool_output": str(pool_output),
        "selection_coverage": selections,
        "availability": availability,
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--middle", default=str(DEFAULT_MIDDLE))
    parser.add_argument("--group-metadata", default=str(DEFAULT_GROUP_METADATA))
    parser.add_argument("--output", default=str(HERE / "outputs" / "topk_minute_pool_audit.json"))
    parser.add_argument("--pool-output", default=str(HERE / "outputs" / "topk_minute_pool.txt"))
    parser.add_argument("--start-signal", default="2025-01-03")
    parser.add_argument("--end-signal", default="2026-04-02")
    args = parser.parse_args()
    report = audit(
        Path(args.middle).resolve(),
        Path(args.group_metadata).resolve(),
        Path(args.output).resolve(),
        Path(args.pool_output).resolve(),
        args.start_signal,
        args.end_signal,
    )
    print(
        f"[topk minute audit] target_union={report['target_union_size']} "
        f"raw_2026_available={report['raw_2026_available_size']} "
        f"pool={report['pool_output']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
