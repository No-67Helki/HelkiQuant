from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

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


def apply_direct_risk(frame_base: pd.DataFrame, threshold: float, floor: float, base: float) -> pd.DataFrame:
    out = frame_base.copy()
    daily_prob = (
        out[out["eligible"].fillna(False)]
        .groupby("signal_date")["outer"]
        .median()
        .rename("outer_risk_probability")
    )
    risk = daily_prob.map(lambda value: floor if value >= threshold else base).rename("risk_budget")
    out = out.merge(risk, left_on="signal_date", right_index=True, how="left")
    out["risk_budget"] = out["risk_budget"].fillna(base)
    out["inner_low"] = pd.NA
    out["inner_high"] = pd.NA
    return out


def evaluate(
    artifacts_dir: Path,
    folds_path: Path,
    raw_daily_dir: Path,
    output_path: Path,
    top_k: int,
    rebalance_every: int,
    risk_base: float,
    group_cap: float,
    initial_cash: float,
) -> dict:
    predictions = load_predictions(artifacts_dir)
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
    base_cfg = ExperimentConfig(
        top_k=top_k,
        buffer_k=top_k * 2,
        rebalance_every=rebalance_every,
        initial_cash=initial_cash,
        risk_base=risk_base,
        risk_slope=0.0,
        risk_min=risk_base,
        risk_max=risk_base,
        max_board_fraction=1.0,
        max_group_fraction=group_cap,
    )
    baseline_frame = add_risk_and_timing_thresholds(frame_base, base_cfg)
    baseline = {
        "name": f"fixed_{risk_base:.0%}",
        "cost": STRESS_COST.name,
        "config": asdict(base_cfg),
        "fold_metrics": fold_summary(baseline_frame, folds, "C", base_cfg, STRESS_COST),
    }
    baseline["diagnostic_score"] = score(baseline["fold_metrics"])
    candidates = [baseline]
    for threshold in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        for floor in (0.20, 0.30, 0.40, 0.50):
            cfg = ExperimentConfig(
                top_k=top_k,
                buffer_k=top_k * 2,
                rebalance_every=rebalance_every,
                initial_cash=initial_cash,
                risk_base=risk_base,
                risk_slope=0.0,
                risk_min=floor,
                risk_max=risk_base,
                max_board_fraction=1.0,
                max_group_fraction=group_cap,
            )
            frame = apply_direct_risk(frame_base, threshold, floor, risk_base)
            metrics = fold_summary(frame, folds, "C", cfg, STRESS_COST)
            candidates.append(
                {
                    "name": f"prob_ge_{threshold:.2f}_floor{floor:.0%}",
                    "cost": STRESS_COST.name,
                    "config": {**asdict(cfg), "outer_probability_threshold": threshold},
                    "fold_metrics": metrics,
                    "diagnostic_score": score(metrics),
                }
            )
    ranked = sorted(candidates, key=lambda row: row["diagnostic_score"], reverse=True)
    passing = [row for row in ranked if row["name"] != baseline["name"] and strict_pass(row, baseline)]
    result = {
        "status": "outer_regime_direct_probability_overlay_research_only",
        "artifacts_dir": str(artifacts_dir.resolve()),
        "raw_daily_dir": str(raw_daily_dir.resolve()),
        "initial_cash": float(initial_cash),
        "baseline": baseline,
        "best_candidate": ranked[0],
        "passing_overlay_candidates": passing,
        "decision": (
            "enable_outer_overlay_research_candidate"
            if passing
            else "keep_outer_disabled_for_production"
        ),
        "deployment_allowed": False,
        "candidates": ranked,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--folds", default=str(HERE / "outputs" / "purged_folds.json"))
    parser.add_argument(
        "--raw-daily-dir",
        default=str(REPO_ROOT / "data" / "A_Stock_daily_qfq" / "daily_qfq_6.5"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--rebalance-every", type=int, default=20)
    parser.add_argument("--risk-base", type=float, default=0.60)
    parser.add_argument("--group-cap", type=float, default=0.30)
    parser.add_argument("--initial-cash", type=float, default=500_000.0)
    args = parser.parse_args()
    result = evaluate(
        Path(args.artifacts_dir).resolve(),
        Path(args.folds).resolve(),
        Path(args.raw_daily_dir).resolve(),
        Path(args.output).resolve(),
        args.top_k,
        args.rebalance_every,
        args.risk_base,
        args.group_cap,
        args.initial_cash,
    )
    base = result["baseline"]["fold_metrics"]
    best = result["best_candidate"]
    best_folds = best["fold_metrics"]
    print(
        "[outer direct overlay] "
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
