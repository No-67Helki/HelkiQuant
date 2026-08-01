from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from portfolio_experiments import load_predictions
from universe import UniverseRules, add_point_in_time_eligibility, load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def quantile_summary(frame: pd.DataFrame, signal: str, outcomes: list[str]) -> list[dict]:
    ranked = frame.dropna(subset=[signal, *outcomes]).copy()
    ranked["quintile"] = pd.qcut(
        ranked[signal].rank(method="first"), 5, labels=False
    )
    rows = []
    for quintile, part in ranked.groupby("quintile", sort=True):
        rows.append(
            {
                "quintile": int(quintile),
                "days": len(part),
                "signal_mean": float(part[signal].mean()),
                **{f"{outcome}_mean": float(part[outcome].mean()) for outcome in outcomes},
            }
        )
    return rows


def evaluate(artifacts_dir: Path, output_path: Path) -> dict:
    predictions = load_predictions(artifacts_dir)
    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(
        REPO_ROOT / "data" / "A_Stock_daily_qfq",
        instruments,
        start="2022-01-04",
        end="2026-04-28",
    ).sort_values(["instrument", "datetime"])
    grouped = prices.groupby("instrument", sort=False)
    prices["forward_5d"] = grouped["close"].shift(-6) / grouped["close"].shift(-1) - 1
    prices["forward_21d"] = grouped["close"].shift(-22) / grouped["close"].shift(-1) - 1
    eligible = add_point_in_time_eligibility(
        prices,
        UniverseRules(min_listing_days=250, min_avg_amount=100_000_000.0),
    )
    sample = predictions.merge(
        eligible[
            ["datetime", "instrument", "eligible", "forward_5d", "forward_21d"]
        ],
        on=["datetime", "instrument"],
        how="inner",
    )
    sample = sample[sample["eligible"]].copy()
    daily = sample.groupby("datetime").agg(
        outer_median=("outer", "median"),
        market_forward_5d=("forward_5d", "mean"),
        market_forward_21d=("forward_21d", "mean"),
    )
    top100 = (
        sample.sort_values(["datetime", "middle"], ascending=[True, False])
        .groupby("datetime")
        .head(100)
        .groupby("datetime")
        .agg(
            top100_forward_5d=("forward_5d", "mean"),
            top100_forward_21d=("forward_21d", "mean"),
        )
    )
    daily = daily.join(top100, how="inner").dropna()
    rolling_mean = daily["outer_median"].rolling(40, min_periods=20).mean()
    rolling_std = daily["outer_median"].rolling(40, min_periods=20).std()
    daily["outer_z_40"] = (
        (daily["outer_median"] - rolling_mean) / rolling_std.replace(0.0, pd.NA)
    ).astype(float)
    outcomes = [
        "market_forward_5d",
        "market_forward_21d",
        "top100_forward_5d",
        "top100_forward_21d",
    ]
    report = {
        "status": "strict_oof_outer_signal_diagnostic_research_only",
        "artifacts_dir": str(artifacts_dir),
        "days": len(daily),
        "spearman_correlation": {
            signal: {
                outcome: float(daily[signal].corr(daily[outcome], method="spearman"))
                for outcome in outcomes
            }
            for signal in ("outer_median", "outer_z_40")
        },
        "outer_quintiles": quantile_summary(daily, "outer_median", outcomes),
        "outer_z_40_quintiles": quantile_summary(daily, "outer_z_40", outcomes),
        "interpretation_warning": (
            "This is diagnostic only. Overlapping forward returns and OOF reuse mean "
            "it must not be treated as an untouched promotion test."
        ),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument(
        "--output", default=str(HERE / "outputs" / "outer_signal_diagnostic.json")
    )
    args = parser.parse_args()
    report = evaluate(Path(args.artifacts_dir).resolve(), Path(args.output).resolve())
    print(f"[outer diagnostic] days={report['days']}")
    for signal, correlations in report["spearman_correlation"].items():
        for outcome, corr in correlations.items():
            print(f"  {signal} -> {outcome}: spearman={corr:+.4f}")
    for row in report["outer_quintiles"]:
        print(
            f"  q{row['quintile']} outer={row['signal_mean']:+.4f} "
            f"market21={row['market_forward_21d_mean']:+.2%} "
            f"top100_21={row['top100_forward_21d_mean']:+.2%}"
        )


if __name__ == "__main__":
    main()
