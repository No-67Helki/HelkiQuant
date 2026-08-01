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
    prepare_research_frame,
    replay_topk,
)
from universe import UniverseRules, load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


PROFILES = {
    "growth": {
        "experiment": "B",
        "risk_budget": 1.0,
    },
    "balanced": {
        "experiment": "C",
        "risk_budget": 0.6,
    },
    "defensive": {
        "experiment": "C",
        "risk_budget": 0.4,
    },
}


def load_middle_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.rename(columns={"middle": "middle"})
    frame["outer"] = 0.0
    frame["inner"] = np.nan
    return frame[["datetime", "instrument", "outer", "middle", "inner"]]


def evaluate(prediction_path: Path, output_path: Path) -> dict:
    predictions = load_middle_predictions(prediction_path)
    prices = load_price_panel(
        REPO_ROOT / "data" / "A_Stock_daily_qfq",
        predictions["instrument"].drop_duplicates().tolist(),
        start="2022-01-04",
        end="2026-04-28",
    )
    rules = UniverseRules(min_listing_days=250, min_avg_amount=100_000_000.0)
    base_frame = prepare_research_frame(predictions, prices, rules)
    rows = []
    for profile, profile_cfg in PROFILES.items():
        risk_budget = profile_cfg["risk_budget"]
        cfg = ExperimentConfig(
            top_k=100,
            buffer_k=200,
            rebalance_every=30,
            risk_base=risk_budget,
            risk_slope=0.0,
            risk_min=risk_budget,
            risk_max=risk_budget,
            max_board_fraction=1.0,
        )
        frame = add_risk_and_timing_thresholds(base_frame, cfg)
        for cost in (BASE_COST, STRESS_COST):
            result = replay_topk(
                frame,
                experiment=profile_cfg["experiment"],
                cfg=cfg,
                cost=cost,
            )
            metrics = result["metrics"]
            rows.append(
                {
                    "profile": profile,
                    "experiment": profile_cfg["experiment"],
                    "cost": cost.name,
                    "risk_budget": risk_budget,
                    "metrics": {
                        **metrics,
                        "total_return": metrics["final_nav"] / cfg.initial_cash - 1,
                    },
                }
            )
    report = {
        "status": "micro_untouched_daily_holdout_research_only",
        "prediction_path": str(prediction_path),
        "window": {
            "signal_start": str(predictions["datetime"].min().date()),
            "signal_end": str(predictions["datetime"].max().date()),
            "price_end": "2026-04-28",
        },
        "universe_rules": {
            "min_listing_days": rules.min_listing_days,
            "min_avg_amount": rules.min_avg_amount,
        },
        "warning": (
            "This holdout is chronological and unused by the OOF tuning cycle, "
            "but it is very short. It can reject bad profiles, not approve live use."
        ),
        "profiles": rows,
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction",
        default=str(
            HERE
            / "outputs"
            / "oof"
            / "pit_holdout_de2_srfs_es"
            / "middle"
            / "fold_99.csv"
        ),
    )
    parser.add_argument(
        "--output", default=str(HERE / "outputs" / "pit_micro_holdout_profiles.json")
    )
    args = parser.parse_args()
    report = evaluate(Path(args.prediction).resolve(), Path(args.output).resolve())
    for row in report["profiles"]:
        metrics = row["metrics"]
        print(
            f"[{row['profile']}/{row['cost']}] "
            f"return={metrics['total_return']:+.2%} "
            f"sharpe={metrics['sharpe']:+.2f} "
            f"mdd={metrics['max_drawdown']:.2%} "
            f"turnover={metrics['turnover']:.2f}"
        )


if __name__ == "__main__":
    main()
