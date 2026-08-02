from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .build_held_intraday_decision_dataset import (
        TRIGGER_BUYBACK_WINDOWS,
        TRIGGER_DISTANCES,
        TRIGGER_TOUCH_BUFFERS,
        add_trigger_aligned_labels,
        trigger_label_prefix,
    )
    from .evaluate_held_intraday_decision_model import FEATURE_COLS
except ImportError:  # pragma: no cover - direct script compatibility
    from build_held_intraday_decision_dataset import (
        TRIGGER_BUYBACK_WINDOWS,
        TRIGGER_DISTANCES,
        TRIGGER_TOUCH_BUFFERS,
        add_trigger_aligned_labels,
        trigger_label_prefix,
    )
    from evaluate_held_intraday_decision_model import FEATURE_COLS


def augment(
    input_csv: Path,
    output_csv: Path,
    output_json: Path,
    *,
    buyback_windows: tuple[str, ...],
    trigger_distance_grid: dict[str, tuple[float, ...]],
    sell_cost: float,
    buy_cost: float,
    slippage: float,
    min_cost: float,
) -> dict:
    header = pd.read_csv(input_csv, nrows=0).columns.tolist()
    required = {
        "datetime",
        "trade_date",
        "instrument",
        "decision_time",
        "shares",
        "sell_price_decision",
        "label_trigger_window_low",
        "label_trigger_window_high",
        "label_trigger_window_minutes",
        "t0_exec_volume_one_lot_max50",
    }
    required.update(f"buyback_{window}_price" for window in buyback_windows)
    missing = sorted(required - set(header))
    if missing:
        raise KeyError(f"source dataset lacks trigger relabel columns: {missing}")
    keep = list(
        dict.fromkeys(
            [col for col in header if col in required]
            + [col for col in FEATURE_COLS if col in header]
        )
    )
    frame = pd.read_csv(input_csv, usecols=keep)
    for window in buyback_windows:
        for direction, distances in trigger_distance_grid.items():
            for distance in distances:
                frame = add_trigger_aligned_labels(
                    frame,
                    sell_cost=sell_cost,
                    buy_cost=buy_cost,
                    slippage=slippage,
                    min_cost=min_cost,
                    trigger_distances={direction: distance},
                    buyback_window=window,
                )
    frame = frame.sort_values(["trade_date", "instrument", "decision_time"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")

    normalized_decision = (
        frame["decision_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    )
    summary: dict[str, dict] = {}
    for decision_time in sorted(normalized_decision.unique()):
        decision = frame[normalized_decision == decision_time]
        for window in buyback_windows:
            for direction, distances in trigger_distance_grid.items():
                for distance in distances:
                    for touch_buffer in TRIGGER_TOUCH_BUFFERS:
                        prefix = trigger_label_prefix(direction, distance, touch_buffer, window)
                        valid = decision[f"{prefix}_touched"].notna()
                        touched = decision.loc[valid, f"{prefix}_touched"] > 0.5
                        key = f"{decision_time}:{prefix}"
                        summary[key] = {
                            "valid_rows": int(valid.sum()),
                            "touch_ratio": float(touched.mean()) if valid.any() else None,
                            "realized_hit_ratio": float(
                                decision.loc[valid, f"{prefix}_realized_hit"].mean()
                            )
                            if valid.any()
                            else None,
                            "realized_edge_mean": float(
                                decision.loc[valid, f"{prefix}_realized_edge"].mean()
                            )
                            if valid.any()
                            else None,
                        }
    report = {
        "status": "held_intraday_trigger_windows_augmented",
        "input_csv": str(input_csv.resolve()),
        "output_csv": str(output_csv.resolve()),
        "source_bytes": int(input_csv.stat().st_size),
        "rows": int(len(frame)),
        "dates": int(pd.to_datetime(frame["trade_date"]).nunique()),
        "instruments": int(frame["instrument"].nunique()),
        "buyback_windows": list(buyback_windows),
        "trigger_distance_grid": {
            direction: list(distances)
            for direction, distances in trigger_distance_grid.items()
        },
        "touch_buffers": list(TRIGGER_TOUCH_BUFFERS),
        "costs": {
            "sell_cost": sell_cost,
            "buy_cost": buy_cost,
            "slippage": slippage,
            "min_cost": min_cost,
        },
        "summary": summary,
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--buyback-windows",
        default=",".join(TRIGGER_BUYBACK_WINDOWS),
    )
    parser.add_argument(
        "--buy-trigger-distances",
        default=str(TRIGGER_DISTANCES["buy_first"]),
    )
    parser.add_argument(
        "--sell-trigger-distances",
        default=str(TRIGGER_DISTANCES["sell_first"]),
    )
    parser.add_argument("--sell-cost", type=float, default=0.0025)
    parser.add_argument("--buy-cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--min-cost", type=float, default=5.0)
    args = parser.parse_args()
    windows = tuple(item.strip() for item in args.buyback_windows.split(",") if item.strip())
    trigger_distance_grid = {
        "buy_first": tuple(
            float(item.strip())
            for item in args.buy_trigger_distances.split(",")
            if item.strip()
        ),
        "sell_first": tuple(
            float(item.strip())
            for item in args.sell_trigger_distances.split(",")
            if item.strip()
        ),
    }
    report = augment(
        Path(args.input_csv).resolve(),
        Path(args.output_csv).resolve(),
        Path(args.output_json).resolve(),
        buyback_windows=windows,
        trigger_distance_grid=trigger_distance_grid,
        sell_cost=args.sell_cost,
        buy_cost=args.buy_cost,
        slippage=args.slippage,
        min_cost=args.min_cost,
    )
    print(
        f"[held trigger windows] rows={report['rows']} dates={report['dates']} "
        f"windows={report['buyback_windows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
