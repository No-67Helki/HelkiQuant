from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def candidate_key(row: dict, kind: str) -> tuple:
    if kind == "constraints":
        return (
            row["experiment"],
            row["min_listing_days"],
            row["min_avg_amount"],
            row["max_board_fraction"],
            json.dumps(row["config"], sort_keys=True),
        )
    return (
        row["outer_lookback"],
        row["risk_base"],
        row["risk_slope"],
        row["risk_min"],
    )


def fold_score(folds: list[dict]) -> float:
    returns = np.array([fold["total_return"] for fold in folds], dtype=float)
    sharpes = np.array([fold["sharpe"] for fold in folds], dtype=float)
    drawdowns = np.array([fold["max_drawdown"] for fold in folds], dtype=float)
    return float(
        np.median(returns)
        + np.min(returns)
        + 0.02 * np.median(sharpes)
        - np.median(drawdowns)
    )


def summarize(rows: list[dict], metric_key: str) -> dict:
    values = np.array([row[metric_key]["total_return"] for row in rows], dtype=float)
    return {
        "median_fold_return": float(np.median(values)),
        "worst_fold_return": float(np.min(values)),
        "positive_fold_ratio": float((values > 0).mean()),
        "median_fold_max_drawdown": float(
            np.median([row[metric_key]["max_drawdown"] for row in rows])
        ),
    }


def evaluate(
    report_path: Path,
    output_path: Path,
    kind: str,
    experiment: str | None,
    risk_slope: float | None,
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = report["candidates"]
    if experiment is not None:
        candidates = [
            row for row in candidates if row.get("experiment") == experiment
        ]
    if risk_slope is not None:
        candidates = [
            row for row in candidates if row.get("risk_slope") == risk_slope
        ]
    base = [row for row in candidates if row["cost"] == "base"]
    stress = {
        candidate_key(row, kind): row for row in candidates if row["cost"] == "stress"
    }
    heldout_rows = []
    folds = sorted(
        {
            fold["fold"]
            for row in base
            for fold in row["fold_metrics"]["folds"]
        }
    )
    for heldout in folds:
        ranked = []
        for row in base:
            development = [
                fold
                for fold in row["fold_metrics"]["folds"]
                if fold["fold"] != heldout
            ]
            ranked.append((fold_score(development), row))
        _, selected = max(ranked, key=lambda item: item[0])
        key = candidate_key(selected, kind)
        stress_selected = stress[key]
        base_fold = next(
            fold
            for fold in selected["fold_metrics"]["folds"]
            if fold["fold"] == heldout
        )
        stress_fold = next(
            fold
            for fold in stress_selected["fold_metrics"]["folds"]
            if fold["fold"] == heldout
        )
        heldout_rows.append(
            {
                "heldout_fold": heldout,
                "selected_candidate": {
                    key_name: selected[key_name]
                    for key_name in selected
                    if key_name not in {"cost", "fold_metrics", "diagnostic_score"}
                },
                "heldout_base": base_fold,
                "heldout_stress": stress_fold,
            }
        )
    result = {
        "status": "leave_one_fold_out_meta_selection_research_only",
        "kind": kind,
        "source_report": str(report_path),
        "experiment_filter": experiment,
        "risk_slope_filter": risk_slope,
        "base_summary": summarize(heldout_rows, "heldout_base"),
        "stress_summary": summarize(heldout_rows, "heldout_stress"),
        "heldout_folds": heldout_rows,
        "warning": (
            "This reduces fold-selection bias but is not a chronological untouched "
            "future holdout."
        ),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--kind", choices=["constraints", "risk"], required=True)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--risk-slope", type=float, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(
        Path(args.report).resolve(),
        Path(args.output).resolve(),
        args.kind,
        args.experiment,
        args.risk_slope,
    )
    for cost in ("base", "stress"):
        summary = result[f"{cost}_summary"]
        print(
            f"[meta {args.kind}/{cost}] "
            f"median={summary['median_fold_return']:+.2%} "
            f"worst={summary['worst_fold_return']:+.2%} "
            f"positive={summary['positive_fold_ratio']:.0%} "
            f"mdd={summary['median_fold_max_drawdown']:.2%}"
        )
    for row in result["heldout_folds"]:
        print(
            f"  fold={row['heldout_fold']} "
            f"base={row['heldout_base']['total_return']:+.2%} "
            f"stress={row['heldout_stress']['total_return']:+.2%}"
        )


if __name__ == "__main__":
    main()
