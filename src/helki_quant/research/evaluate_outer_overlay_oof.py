from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from evaluate_oof import fold_summary
from portfolio_experiments import (
    STRESS_COST,
    ExperimentConfig,
    add_risk_and_timing_thresholds,
    load_predictions,
    prepare_research_frame,
)
from universe import UniverseRules, load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def score(folds: dict) -> float:
    return float(
        (folds["median_fold_return"] or 0.0)
        + (folds["worst_fold_return"] or 0.0)
        + 0.02 * (folds["median_fold_sharpe"] or 0.0)
        - (folds["median_fold_max_drawdown"] or 0.0)
    )


def strict_pass(candidate: dict, baseline: dict) -> bool:
    c = candidate["fold_metrics"]
    b = baseline["fold_metrics"]
    return bool(
        c["evaluated_folds"] == b["evaluated_folds"]
        and (c["positive_fold_ratio"] or 0.0) >= (b["positive_fold_ratio"] or 0.0)
        and (c["worst_fold_return"] or -1.0) >= (b["worst_fold_return"] or -1.0)
        and (c["median_fold_max_drawdown"] or 1.0) <= (b["median_fold_max_drawdown"] or 1.0)
        and score(c) > score(b)
    )


def candidate_configs(
    top_k: int,
    rebalance_every: int,
    risk_base: float,
    group_cap: float,
) -> list[tuple[str, ExperimentConfig]]:
    base = ExperimentConfig(
        top_k=top_k,
        buffer_k=top_k * 2,
        rebalance_every=rebalance_every,
        risk_base=risk_base,
        risk_slope=0.0,
        risk_min=risk_base,
        risk_max=risk_base,
        max_board_fraction=1.0,
        max_group_fraction=group_cap,
    )
    rows = [(f"fixed_{risk_base:.0%}", base)]
    for lookback in (20, 40, 60):
        for slope in (-0.20, -0.10, 0.10, 0.20):
            for floor in (0.40, 0.60):
                rows.append(
                    (
                        f"outer_z_lb{lookback}_slope{slope:+.2f}_floor{floor:.2f}",
                        replace(
                            base,
                            outer_lookback=lookback,
                            risk_slope=slope,
                            risk_min=floor,
                            risk_max=1.0,
                        ),
                    )
                )
    return rows


def evaluate(
    artifacts_dir: Path,
    folds_path: Path,
    output_path: Path,
    raw_daily_dir: Path,
    top_k: int,
    rebalance_every: int,
    risk_base: float,
    group_cap: float,
) -> dict:
    predictions = load_predictions(artifacts_dir)
    if predictions["outer"].notna().sum() == 0:
        raise ValueError("No outer-layer OOF predictions were assembled")
    prices = load_price_panel(
        raw_daily_dir,
        predictions["instrument"].drop_duplicates().tolist(),
        start="2022-01-04",
        end="2026-04-28",
    )
    frame_base = prepare_research_frame(
        predictions,
        prices,
        UniverseRules(min_listing_days=250, min_avg_amount=100_000_000.0),
    )
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    candidates = []
    for name, cfg in candidate_configs(top_k, rebalance_every, risk_base, group_cap):
        frame = add_risk_and_timing_thresholds(frame_base, cfg)
        fold_metrics = fold_summary(frame, folds, "C", cfg, STRESS_COST)
        candidates.append(
            {
                "name": name,
                "cost": STRESS_COST.name,
                "config": asdict(cfg),
                "fold_metrics": fold_metrics,
                "diagnostic_score": score(fold_metrics),
            }
        )

    baseline = candidates[0]
    ranked = sorted(candidates, key=lambda row: row["diagnostic_score"], reverse=True)
    passing = [row for row in ranked if row["name"] != baseline["name"] and strict_pass(row, baseline)]
    result = {
        "status": "strict_oof_outer_overlay_small_grid_research_only",
        "artifacts_dir": str(artifacts_dir.resolve()),
        "raw_daily_dir": str(raw_daily_dir.resolve()),
        "baseline_name": baseline["name"],
        "baseline": baseline,
        "best_candidate": ranked[0],
        "passing_overlay_candidates": passing,
        "decision": (
            "enable_outer_overlay_research_candidate"
            if passing
            else "keep_outer_disabled_for_production"
        ),
        "decision_rule": (
            "Outer overlay must improve diagnostic score while not worsening positive "
            "fold ratio, worst fold return, or median fold drawdown versus fixed "
            f"{risk_base:.0%} "
            "risk under stress costs."
        ),
        "deployment_allowed": False,
        "candidates": ranked,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        default=str(HERE / "outputs" / "oof_artifacts" / "pit_hybrid_de2"),
    )
    parser.add_argument("--folds", default=str(HERE / "outputs" / "purged_folds.json"))
    parser.add_argument(
        "--raw-daily-dir",
        default=str(REPO_ROOT / "data" / "A_Stock_daily_qfq" / "daily_qfq_6.5"),
    )
    parser.add_argument(
        "--output",
        default=str(HERE / "outputs" / "outer_overlay_oof_20260606.json"),
    )
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--rebalance-every", type=int, default=45)
    parser.add_argument("--risk-base", type=float, default=0.80)
    parser.add_argument("--group-cap", type=float, default=0.30)
    args = parser.parse_args()
    result = evaluate(
        Path(args.artifacts_dir).resolve(),
        Path(args.folds).resolve(),
        Path(args.output).resolve(),
        Path(args.raw_daily_dir).resolve(),
        args.top_k,
        args.rebalance_every,
        args.risk_base,
        args.group_cap,
    )
    base = result["baseline"]["fold_metrics"]
    best = result["best_candidate"]
    best_folds = best["fold_metrics"]
    print(
        "[outer overlay] "
        f"decision={result['decision']} "
        f"baseline_worst={base['worst_fold_return']:+.2%} "
        f"baseline_mdd={base['median_fold_max_drawdown']:.2%}"
    )
    print(
        "  best="
        f"{best['name']} score={best['diagnostic_score']:+.4f} "
        f"worst={best_folds['worst_fold_return']:+.2%} "
        f"median={best_folds['median_fold_return']:+.2%} "
        f"mdd={best_folds['median_fold_max_drawdown']:.2%}"
    )


if __name__ == "__main__":
    main()
