from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_held_intraday_decision_dataset import (
    BUYBACK_WINDOWS,
    add_cross_sectional_features,
    add_execution_aligned_labels,
)


def augment(
    input_csv: Path,
    output_csv: Path,
    output_json: Path,
    *,
    sell_cost: float,
    buy_cost: float,
    slippage: float,
    min_cost: float,
) -> dict:
    frame = pd.read_csv(input_csv, parse_dates=["datetime", "trade_date"])
    frame = add_execution_aligned_labels(
        frame.replace([np.inf, -np.inf], np.nan),
        sell_cost=sell_cost,
        buy_cost=buy_cost,
        slippage=slippage,
        min_cost=min_cost,
    )
    frame = add_cross_sectional_features(frame).sort_values(
        ["trade_date", "instrument", "decision_minute"]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")

    sizing = ["10pct", "20pct", "30pct", "one_lot_max50"]
    coverage = {
        name: float(frame[f"t0_exec_tradeable_{name}"].mean())
        for name in sizing
    }
    report = {
        "status": "held_intraday_execution_labels_augmented",
        "input_csv": str(input_csv.resolve()),
        "output_csv": str(output_csv.resolve()),
        "rows": int(len(frame)),
        "dates": int(frame["trade_date"].nunique()),
        "instruments": int(frame["instrument"].nunique()),
        "buyback_windows": list(BUYBACK_WINDOWS),
        "directions": ["sell_first", "buy_first"],
        "sizing": sizing,
        "tradeable_coverage": coverage,
        "execution_assumptions": {
            "sell_cost": sell_cost,
            "buy_cost": buy_cost,
            "slippage": slippage,
            "min_cost": min_cost,
            "one_lot_max_inventory_fraction": 0.5,
        },
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
    parser.add_argument("--sell-cost", type=float, default=0.0025)
    parser.add_argument("--buy-cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--min-cost", type=float, default=5.0)
    args = parser.parse_args()
    report = augment(
        Path(args.input_csv).resolve(),
        Path(args.output_csv).resolve(),
        Path(args.output_json).resolve(),
        sell_cost=args.sell_cost,
        buy_cost=args.buy_cost,
        slippage=args.slippage,
        min_cost=args.min_cost,
    )
    print(
        "[held intraday execution labels] "
        f"rows={report['rows']} coverage={report['tradeable_coverage']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
