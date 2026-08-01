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


def diagnostic_score(folds: dict) -> float:
    return (
        (folds["median_fold_return"] or 0.0)
        + (folds["worst_fold_return"] or 0.0)
        + 0.02 * (folds["median_fold_sharpe"] or 0.0)
        - (folds["median_fold_max_drawdown"] or 0.0)
    )


def evaluate(artifacts_dir: Path, folds_path: Path, output_path: Path) -> dict:
    predictions = load_predictions(artifacts_dir)
    prices = load_price_panel(
        REPO_ROOT / "data" / "A_Stock_daily_qfq",
        predictions["instrument"].drop_duplicates().tolist(),
        start="2022-01-04",
        end="2026-04-28",
    )
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    rules = UniverseRules(min_listing_days=250, min_avg_amount=100_000_000.0)
    base_frame = prepare_research_frame(predictions, prices, rules)
    rows = []
    for lookback in (20, 40, 60):
        for risk_base in (0.20, 0.40, 0.60):
            for risk_slope in (-0.25, -0.15, 0.0, 0.15, 0.25):
                for risk_min in (0.20, 0.40):
                    if risk_min > risk_base:
                        continue
                    cfg = ExperimentConfig(
                        top_k=100,
                        buffer_k=200,
                        rebalance_every=30,
                        outer_lookback=lookback,
                        risk_base=risk_base,
                        risk_slope=risk_slope,
                        risk_min=risk_min,
                        risk_max=1.0,
                        max_board_fraction=1.0,
                    )
                    frame = add_risk_and_timing_thresholds(base_frame, cfg)
                    for cost in (BASE_COST, STRESS_COST):
                        fold_metrics = fold_summary(frame, folds, "C", cfg, cost)
                        rows.append(
                            {
                                "cost": cost.name,
                                "outer_lookback": lookback,
                                "risk_base": risk_base,
                                "risk_slope": risk_slope,
                                "risk_min": risk_min,
                                "fold_metrics": fold_metrics,
                                "diagnostic_score": diagnostic_score(fold_metrics),
                            }
                        )
    report = {
        "status": "strict_oof_risk_budget_ablation_research_only",
        "artifacts_dir": str(artifacts_dir),
        "universe_rules": {
            "min_listing_days": rules.min_listing_days,
            "min_avg_amount": rules.min_avg_amount,
        },
        "interpretation_rule": (
            "If zero slope is more robust than non-zero slopes, the outer model "
            "does not add reliable dynamic risk-budget value."
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
        "--output", default=str(HERE / "outputs" / "risk_budget_evaluation.json")
    )
    args = parser.parse_args()
    report = evaluate(
        Path(args.artifacts_dir).resolve(),
        Path(args.folds).resolve(),
        Path(args.output).resolve(),
    )
    for row in report["candidates"][:15]:
        folds = row["fold_metrics"]
        print(
            f"[{row['cost']}] lookback={row['outer_lookback']} "
            f"risk={row['risk_base']:.2f}+{row['risk_slope']:+.2f}z "
            f"floor={row['risk_min']:.2f} "
            f"median={folds['median_fold_return']:+.2%} "
            f"worst={folds['worst_fold_return']:+.2%} "
            f"positive={folds['positive_fold_ratio']:.0%} "
            f"mdd={folds['median_fold_max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
