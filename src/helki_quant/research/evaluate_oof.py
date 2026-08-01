from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_experiments import (
    BASE_COST,
    STRESS_COST,
    ExperimentConfig,
    add_risk_and_timing_thresholds,
    load_predictions,
    prepare_research_frame,
    replay_topk,
)
from universe import UniverseRules, load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def candidate_configs() -> list[tuple[str, ExperimentConfig]]:
    rows = [("A", ExperimentConfig())]
    for top_k in (50, 100):
        for rebalance in (20, 30):
            rows.append(
                (
                    "B",
                    ExperimentConfig(
                        top_k=top_k,
                        buffer_k=top_k * 2,
                        rebalance_every=rebalance,
                    ),
                )
            )
            for risk_base in (0.40, 0.60, 0.80):
                for risk_slope in (-0.25, -0.15, 0.0, 0.15, 0.25):
                    for risk_min in (0.0, 0.20, 0.40):
                        rows.append(
                            (
                                "C",
                                ExperimentConfig(
                                    top_k=top_k,
                                    buffer_k=top_k * 2,
                                    rebalance_every=rebalance,
                                    risk_base=risk_base,
                                    risk_slope=risk_slope,
                                    risk_min=risk_min,
                                    risk_max=1.0,
                                ),
                            )
                        )
    return rows


def safe_replay(frame: pd.DataFrame, experiment: str, cfg: ExperimentConfig, cost):
    if frame.empty:
        return None
    try:
        return replay_topk(frame, experiment=experiment, cfg=cfg, cost=cost)
    except (IndexError, KeyError, ValueError):
        return None


def fold_summary(
    frame: pd.DataFrame,
    folds: list[dict],
    experiment: str,
    cfg: ExperimentConfig,
    cost,
) -> dict:
    rows = []
    for fold in folds:
        part = frame[
            frame["signal_date"].between(fold["test_start"], fold["test_end"])
        ]
        result = safe_replay(part, experiment, cfg, cost)
        if result is None:
            continue
        metrics = result["metrics"]
        rows.append(
            {
                "fold": fold["fold"],
                "total_return": metrics["final_nav"] / cfg.initial_cash - 1,
                **metrics,
            }
        )
    returns = np.array([row["total_return"] for row in rows], dtype=float)
    sharpes = np.array([row["sharpe"] for row in rows], dtype=float)
    drawdowns = np.array([row["max_drawdown"] for row in rows], dtype=float)
    turnovers = np.array([row["turnover"] for row in rows], dtype=float)
    return {
        "folds": rows,
        "evaluated_folds": len(rows),
        "median_fold_return": float(np.median(returns)) if len(returns) else None,
        "worst_fold_return": float(np.min(returns)) if len(returns) else None,
        "positive_fold_ratio": float((returns > 0).mean()) if len(returns) else None,
        "median_fold_sharpe": float(np.median(sharpes)) if len(sharpes) else None,
        "median_fold_max_drawdown": (
            float(np.median(drawdowns)) if len(drawdowns) else None
        ),
        "worst_fold_max_drawdown": (
            float(np.max(drawdowns)) if len(drawdowns) else None
        ),
        "median_fold_turnover": float(np.median(turnovers)) if len(turnovers) else None,
    }


def evaluate(
    artifacts_dir: Path,
    folds_path: Path,
    output_path: Path,
    raw_daily_dir: Path,
    price_start: str,
    price_end: str,
) -> dict:
    predictions = load_predictions(artifacts_dir)
    if predictions["middle"].notna().sum() == 0:
        raise ValueError("No middle-layer OOF predictions were assembled")
    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(
        raw_daily_dir,
        instruments,
        # Eligibility uses listing age and rolling liquidity/suspension history.
        # Loading only from the first OOF test day incorrectly resets every
        # instrument's history at the test boundary.
        start=price_start,
        end=price_end,
    )
    base_frame = prepare_research_frame(predictions, prices, UniverseRules())
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    rows = []
    for experiment, cfg in candidate_configs():
        frame = add_risk_and_timing_thresholds(base_frame, cfg)
        for cost in (BASE_COST, STRESS_COST):
            result = safe_replay(frame, experiment, cfg, cost)
            if result is None:
                continue
            fold_metrics = fold_summary(frame, folds, experiment, cfg, cost)
            metrics = result["metrics"]
            # Fold-level behavior is primary. Aggregate NAV can bridge the
            # intentional gaps between test folds and is diagnostic only.
            score = (
                (fold_metrics["median_fold_return"] or 0.0)
                + (fold_metrics["worst_fold_return"] or 0.0)
                + 0.02 * (fold_metrics["median_fold_sharpe"] or 0.0)
                - (fold_metrics["median_fold_max_drawdown"] or 0.0)
            )
            rows.append(
                {
                    "experiment": experiment,
                    "cost": cost.name,
                    "top_k": cfg.top_k,
                    "buffer_k": cfg.buffer_k,
                    "rebalance_every": cfg.rebalance_every,
                    "risk_base": cfg.risk_base,
                    "risk_slope": cfg.risk_slope,
                    "risk_min": cfg.risk_min,
                    "risk_max": cfg.risk_max,
                    "diagnostic_score": float(score),
                    "metrics": metrics,
                    "fold_metrics": fold_metrics,
                }
            )
    report = {
        "status": "strict_oof_candidate_research_only",
        "artifacts_dir": str(artifacts_dir),
        "raw_daily_dir": str(raw_daily_dir),
        "price_start": price_start,
        "price_end": price_end,
        "industry_concentration_status": (
            "blocked: point-in-time industry classification metadata is not present"
        ),
        "ranking_policy": (
            "Diagnostic score uses per-fold median/worst return, median Sharpe, and "
            "median drawdown. Aggregate NAV metrics are diagnostic only because fold "
            "gaps can bridge positions."
        ),
        "deployment_allowed": False,
        "candidates": sorted(rows, key=lambda row: row["diagnostic_score"], reverse=True),
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
    parser.add_argument("--output", default=str(HERE / "outputs" / "oof_evaluation.json"))
    parser.add_argument(
        "--raw-daily-dir",
        default=str(REPO_ROOT / "data" / "A_Stock_daily_qfq"),
    )
    parser.add_argument("--price-start", default="2022-01-04")
    parser.add_argument("--price-end", default="2026-04-28")
    args = parser.parse_args()
    report = evaluate(
        Path(args.artifacts_dir).resolve(),
        Path(args.folds).resolve(),
        Path(args.output).resolve(),
        Path(args.raw_daily_dir).resolve(),
        args.price_start,
        args.price_end,
    )
    for row in report["candidates"][:10]:
        metrics = row["metrics"]
        folds = row["fold_metrics"]
        print(
            f"[{row['experiment']}/{row['cost']}] top={row['top_k']} "
            f"rebalance={row['rebalance_every']} sharpe={metrics['sharpe']:+.2f} "
            f"mdd={metrics['max_drawdown']:.2%} "
            f"median_fold={folds['median_fold_return']:+.2%} "
            f"worst_fold={folds['worst_fold_return']:+.2%} "
            f"risk={row['risk_base']:.2f}+{row['risk_slope']:.2f}z "
            f"floor={row['risk_min']:.2f}"
        )


if __name__ == "__main__":
    main()
