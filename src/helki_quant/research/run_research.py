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
from splitters import PurgedWalkForwardSplitter
from universe import UniverseRules, load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MULTI_LAYER = HERE.parent


def build_folds(
    output_dir: Path,
    calendar_provider: Path | None = None,
    start: str = "2022-01-04",
    end: str = "2026-04-28",
) -> list[dict]:
    provider = calendar_provider or REPO_ROOT / "data" / "cn_data_pool"
    calendar_path = provider / "calendars" / "day.txt"
    dates = pd.read_csv(calendar_path, header=None).iloc[:, 0]
    dates = dates[(pd.to_datetime(dates) >= start) & (pd.to_datetime(dates) <= end)]
    splitter = PurgedWalkForwardSplitter(
        min_train_days=500,
        valid_days=120,
        test_days=60,
        step_days=60,
        purge_days=21,
        embargo_days=5,
    )
    folds = [fold.as_dict() for fold in splitter.split(dates)]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "purged_folds.json").write_text(
        json.dumps(folds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return folds


def middle_alpha_diagnostic(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict:
    labels = prices.copy().sort_values(["instrument", "datetime"])
    grouped = labels.groupby("instrument", sort=False)
    labels["forward_5d"] = grouped["close"].shift(-6) / grouped["close"].shift(-1) - 1
    eligible = frame[
        ["signal_date", "instrument", "eligible"]
    ].drop_duplicates(["signal_date", "instrument"])
    sample = (
        predictions[["datetime", "instrument", "middle"]]
        .rename(columns={"datetime": "signal_date"})
        .merge(eligible, on=["signal_date", "instrument"], how="inner")
        .merge(
            labels[["datetime", "instrument", "forward_5d"]].rename(
                columns={"datetime": "signal_date"}
            ),
            on=["signal_date", "instrument"],
            how="inner",
        )
        .dropna()
    )
    sample = sample[sample["eligible"]].copy()
    pct_rank = sample.groupby("signal_date")["middle"].rank(pct=True, method="first")
    sample["decile"] = ((pct_rank * 10).apply(np.ceil) - 1).clip(0, 9).astype(int)
    deciles = (
        sample.groupby("decile")["forward_5d"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    top5 = sample.groupby("signal_date", group_keys=False).apply(
        lambda part: part.nlargest(5, "middle")["forward_5d"].mean()
    )
    return {
        "deciles": deciles,
        "top_minus_bottom_mean": float(
            sample[sample["decile"] == 9]["forward_5d"].mean()
            - sample[sample["decile"] == 0]["forward_5d"].mean()
        ),
        "extreme_top5_mean": float(top5.mean()),
        "interpretation": (
            "Positive broad-decile spread with a weak/negative extreme Top-5 "
            "means the signal is suitable for diversified ranking, not concentrated picking."
        ),
    }


def structure_sweep(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for top_k in (5, 10, 20, 50, 100, 200):
        for rebalance_every in (5, 10, 20):
            cfg = ExperimentConfig(
                top_k=top_k,
                buffer_k=top_k * 2,
                rebalance_every=rebalance_every,
            )
            for experiment in ("B", "C"):
                for cost in (BASE_COST, STRESS_COST):
                    result = replay_topk(
                        frame,
                        experiment=experiment,
                        cfg=cfg,
                        cost=cost,
                    )
                    metrics = result["metrics"]
                    diagnostic_score = (
                        metrics["sharpe"]
                        + metrics["annualized_return"]
                        - 2.0 * metrics["max_drawdown"]
                        - 0.01 * max(0.0, metrics["turnover"] - 10.0)
                    )
                    rows.append(
                        {
                            "experiment": experiment,
                            "cost": cost.name,
                            "top_k": top_k,
                            "rebalance_every": rebalance_every,
                            "diagnostic_score": float(diagnostic_score),
                            "metrics": metrics,
                        }
                    )
    return sorted(rows, key=lambda row: row["diagnostic_score"], reverse=True)


def audit_and_experiments(output_dir: Path, artifacts_dir: Path) -> dict:
    predictions = load_predictions(artifacts_dir)
    instruments = predictions["instrument"].drop_duplicates().tolist()
    start = str(predictions["datetime"].min().date())
    end = "2026-04-28"
    prices = load_price_panel(
        REPO_ROOT / "data" / "A_Stock_daily_qfq",
        instruments,
        start="2022-01-04",
        end=end,
    )
    rules = UniverseRules()
    frame = prepare_research_frame(predictions, prices, rules)
    frame = add_risk_and_timing_thresholds(frame, ExperimentConfig())
    alpha_diagnostic = middle_alpha_diagnostic(predictions, prices, frame)

    monthly = (
        frame.groupby(frame["signal_date"].dt.to_period("M"))
        .agg(
            rows=("instrument", "size"),
            eligible=("eligible", "sum"),
            inner_available=("inner", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    monthly["signal_date"] = monthly["signal_date"].astype(str)

    results = []
    nav_dir = output_dir / "nav"
    nav_dir.mkdir(parents=True, exist_ok=True)
    for experiment in ("A", "B", "C", "D"):
        for cost in (BASE_COST, STRESS_COST):
            result = replay_topk(
                frame,
                experiment=experiment,
                cfg=ExperimentConfig(),
                cost=cost,
            )
            result["nav"].rename("nav").to_csv(nav_dir / f"{experiment}_{cost.name}.csv")
            results.append({k: v for k, v in result.items() if k != "nav"})
    sweep = structure_sweep(frame)

    report = {
        "status": "directional_baseline_only_not_oof",
        "artifacts_dir": str(artifacts_dir),
        "prediction_start": start,
        "prediction_end": str(predictions["datetime"].max().date()),
        "source_universe_instruments": len(instruments),
        "dynamic_rules": rules.__dict__,
        "residual_bias_warning": (
            "Dynamic eligibility is point-in-time, but the source 1666-stock pool was "
            "selected using later information. Rebuild the source universe point-in-time "
            "before treating results as deployable evidence."
        ),
        "monthly_coverage": monthly.to_dict(orient="records"),
        "middle_alpha_diagnostic": alpha_diagnostic,
        "experiments": results,
        "structure_sweep": sweep,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "directional_baseline.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["folds", "baseline", "all"], default="all")
    parser.add_argument("--artifacts_dir", default=str(MULTI_LAYER / "artifacts" / "robust_v2"))
    parser.add_argument("--output_dir", default=str(HERE / "outputs"))
    parser.add_argument(
        "--calendar-provider",
        default=str(REPO_ROOT / "data" / "cn_data_pool"),
        help="Provider whose day calendar is used to build purged walk-forward folds.",
    )
    parser.add_argument("--fold-start", default="2022-01-04")
    parser.add_argument("--fold-end", default="2026-04-28")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if args.mode in {"folds", "all"}:
        folds = build_folds(
            output_dir,
            Path(args.calendar_provider).resolve(),
            args.fold_start,
            args.fold_end,
        )
        print(f"[folds] {len(folds)} -> {output_dir / 'purged_folds.json'}")
    if args.mode in {"baseline", "all"}:
        report = audit_and_experiments(output_dir, Path(args.artifacts_dir).resolve())
        for row in report["experiments"]:
            metrics = row["metrics"]
            print(
                f"[{row['experiment']}/{row['cost']['name']}] "
                f"ann={metrics['annualized_return']:+.2%} sharpe={metrics['sharpe']:+.2f} "
                f"mdd={metrics['max_drawdown']:.2%} trades={metrics['trades']} "
                f"turnover={metrics['turnover']:.2f}"
            )
        print("[structure sweep top 10]")
        for row in report["structure_sweep"][:10]:
            metrics = row["metrics"]
            print(
                f"[{row['experiment']}/{row['cost']}] topk={row['top_k']} "
                f"rebalance={row['rebalance_every']} ann={metrics['annualized_return']:+.2%} "
                f"sharpe={metrics['sharpe']:+.2f} mdd={metrics['max_drawdown']:.2%} "
                f"turnover={metrics['turnover']:.2f}"
            )


if __name__ == "__main__":
    main()
