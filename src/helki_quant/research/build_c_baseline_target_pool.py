from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from concentration_constraints import ConcentrationRules, groups_on_date, load_group_metadata, select_with_group_cap
from evaluate_daily_topk_grid import load_middle_predictions
from export_c_baseline_production_logs import selected_profiles
from minute_mapped_topk_replay import MappedReplayConfig, prepare_daily_frame
from universe import load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_GROUP_METADATA = DATA / "industry_theme_pit.csv"


def raw_symbol(instrument: str) -> str:
    text = str(instrument).strip().lower()
    if text.startswith(("sh", "sz")):
        return text[:2] + text[-6:]
    code = text.replace(".", "")[-6:]
    return ("sh" if code.startswith("68") else "sz") + code


def collect(
    middle_path: Path,
    group_metadata_path: Path,
    output_path: Path,
    report_path: Path,
    start_signal: str,
    end_signal: str,
) -> dict:
    predictions = load_middle_predictions(middle_path)
    predictions = predictions[predictions["datetime"].between(start_signal, end_signal)].copy()
    prices = load_price_panel(
        DATA / "A_Stock_daily_qfq",
        predictions["instrument"].drop_duplicates().tolist(),
        start="2022-01-04",
        end="2026-04-28",
    )
    group_metadata = load_group_metadata(group_metadata_path, "industry")
    cfg = MappedReplayConfig()
    selected_rows = []
    symbols: set[str] = set()
    for profile in selected_profiles():
        frame = prepare_daily_frame(predictions, prices, profile, cfg)
        previous: set[str] = set()
        buffer_k = profile.top_k * cfg.buffer_multiple
        for day_no, (trade_date, day) in enumerate(frame.groupby("trade_date", sort=True)):
            if day_no % profile.rebalance_every != 0:
                continue
            eligible = day[day["eligible"].fillna(False)].sort_values("middle", ascending=False)
            groups = groups_on_date(group_metadata, trade_date)
            selected = select_with_group_cap(
                eligible["instrument"].tolist(),
                previous,
                top_k=profile.top_k,
                buffer_k=buffer_k,
                groups=groups,
                rules=ConcentrationRules(max_group_fraction=profile.industry_cap),
            )
            previous = set(selected)
            symbols.update(raw_symbol(inst) for inst in selected)
            selected_rows.append(
                {
                    "profile": profile.name,
                    "trade_date": str(pd.Timestamp(trade_date).date()),
                    "selected_count": len(selected),
                    "selected": selected,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sorted(symbols)) + "\n", encoding="utf-8")
    report = {
        "status": "c_baseline_target_pool_research_only",
        "middle_prediction": str(middle_path),
        "group_metadata": str(group_metadata_path),
        "window": {"start_signal": start_signal, "end_signal": end_signal},
        "profiles": [asdict(profile) for profile in selected_profiles()],
        "target_union_size": len(symbols),
        "pool_output": str(output_path),
        "selections": selected_rows,
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--middle",
        default=str(HERE / "outputs" / "oof" / "pit_holdout_de2_srfs_es" / "middle" / "fold_99.csv"),
    )
    parser.add_argument("--group-metadata", default=str(DEFAULT_GROUP_METADATA))
    parser.add_argument("--output", default=str(HERE / "outputs" / "c_baseline_holdout_target_pool.txt"))
    parser.add_argument("--report", default=str(HERE / "outputs" / "c_baseline_holdout_target_pool.json"))
    parser.add_argument("--start-signal", default="2026-04-03")
    parser.add_argument("--end-signal", default="2026-04-20")
    args = parser.parse_args()
    report = collect(
        Path(args.middle).resolve(),
        Path(args.group_metadata).resolve(),
        Path(args.output).resolve(),
        Path(args.report).resolve(),
        args.start_signal,
        args.end_signal,
    )
    print(
        f"[c baseline pool] target_union={report['target_union_size']} "
        f"pool={report['pool_output']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
