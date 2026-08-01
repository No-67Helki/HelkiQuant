from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_oof import fold_summary
from portfolio_experiments import (
    BASE_COST,
    STRESS_COST,
    ExperimentConfig,
    add_risk_and_timing_thresholds,
    load_predictions,
    prepare_research_frame,
)
from universe import UniverseRules, load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def score(row: dict) -> float:
    folds = row["fold_metrics"]
    return (
        (folds["median_fold_return"] or 0.0)
        + (folds["worst_fold_return"] or 0.0)
        + 0.02 * (folds["median_fold_sharpe"] or 0.0)
        - (folds["median_fold_max_drawdown"] or 0.0)
    )


GRIDS = {
    "broad": {
        "listing_days": (120, 250),
        "min_amount": (50_000_000.0, 100_000_000.0, 200_000_000.0),
        "board_cap": (0.55, 0.65, 0.75, 1.00),
    },
    "focused": {
        "listing_days": (180, 250, 350),
        "min_amount": (75_000_000.0, 100_000_000.0, 125_000_000.0, 150_000_000.0),
        "board_cap": (0.65, 0.75, 1.00),
    },
}


def evaluate(
    artifacts_dir: Path,
    folds_path: Path,
    output_path: Path,
    grid_name: str,
) -> dict:
    predictions = load_predictions(artifacts_dir)
    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(
        REPO_ROOT / "data" / "A_Stock_daily_qfq",
        instruments,
        start="2022-01-04",
        end="2026-04-28",
    )
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    rows = []
    grid = GRIDS[grid_name]
    for listing_days in grid["listing_days"]:
        for min_amount in grid["min_amount"]:
            rules = UniverseRules(
                min_listing_days=listing_days,
                min_avg_amount=min_amount,
            )
            base_frame = prepare_research_frame(predictions, prices, rules)
            for board_cap in grid["board_cap"]:
                for experiment, risk_base in (("B", 1.0), ("C", 0.4)):
                    cfg = ExperimentConfig(
                        top_k=100,
                        buffer_k=200,
                        rebalance_every=30,
                        risk_base=risk_base,
                        risk_slope=0.15,
                        risk_min=0.4 if experiment == "C" else 1.0,
                        risk_max=1.0,
                        max_board_fraction=board_cap,
                    )
                    frame = add_risk_and_timing_thresholds(base_frame, cfg)
                    for cost in (BASE_COST, STRESS_COST):
                        fold_metrics = fold_summary(frame, folds, experiment, cfg, cost)
                        row = {
                            "experiment": experiment,
                            "cost": cost.name,
                            "min_listing_days": listing_days,
                            "min_avg_amount": min_amount,
                            "max_board_fraction": board_cap,
                            "config": {
                                "top_k": cfg.top_k,
                                "buffer_k": cfg.buffer_k,
                                "rebalance_every": cfg.rebalance_every,
                                "risk_base": cfg.risk_base,
                                "risk_slope": cfg.risk_slope,
                                "risk_min": cfg.risk_min,
                            },
                            "fold_metrics": fold_metrics,
                        }
                        row["diagnostic_score"] = score(row)
                        rows.append(row)
    report = {
        "status": "strict_oof_constraint_search_research_only",
        "artifacts_dir": str(artifacts_dir),
        "grid": grid_name,
        "grid_values": grid,
        "search_policy": (
            "Small structural grid over point-in-time eligibility and 30/68 board "
            "caps. Per-fold metrics are primary; no deployment is allowed."
        ),
        "deployment_allowed": False,
        "candidates": sorted(
            rows, key=lambda row: row["diagnostic_score"], reverse=True
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--folds", default=str(HERE / "outputs" / "purged_folds.json"))
    parser.add_argument(
        "--output", default=str(HERE / "outputs" / "constraint_evaluation.json")
    )
    parser.add_argument("--grid", choices=list(GRIDS), default="broad")
    args = parser.parse_args()
    report = evaluate(
        Path(args.artifacts_dir).resolve(),
        Path(args.folds).resolve(),
        Path(args.output).resolve(),
        args.grid,
    )
    for row in report["candidates"][:12]:
        folds = row["fold_metrics"]
        print(
            f"[{row['experiment']}/{row['cost']}] listing={row['min_listing_days']} "
            f"amount={row['min_avg_amount'] / 1e6:.0f}m "
            f"board_cap={row['max_board_fraction']:.0%} "
            f"median={folds['median_fold_return']:+.2%} "
            f"worst={folds['worst_fold_return']:+.2%} "
            f"positive={folds['positive_fold_ratio']:.0%} "
            f"mdd={folds['median_fold_max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
