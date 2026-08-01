from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_expanded_c_controls import DEFAULT_FORBIDDEN, load_forbidden_instruments
from portfolio_experiments import (
    BASE_COST,
    STRESS_COST,
    ExperimentConfig,
    add_risk_and_timing_thresholds,
    prepare_research_frame,
    replay_topk,
)
from universe import UniverseRules, add_point_in_time_eligibility, load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def read_predictions(prediction_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(prediction_dir.glob("fold_*.csv")):
        frame = pd.read_csv(path)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame["fold"] = int(path.stem.split("_")[-1])
        rows.append(frame[["datetime", "instrument", "middle", "fold"]])
    if not rows:
        raise ValueError(f"No fold predictions found under {prediction_dir}")
    return pd.concat(rows, ignore_index=True)


def score_sample(sample: pd.DataFrame, forward_column: str = "forward_5d") -> dict:
    daily_ic = sample.groupby("datetime").apply(
        lambda part: part["middle"].corr(part[forward_column], method="spearman")
    )
    ranked = sample.copy()
    pct_rank = ranked.groupby("datetime")["middle"].rank(pct=True, method="first")
    ranked["decile"] = ((pct_rank * 10).apply(np.ceil) - 1).clip(0, 9).astype(int)
    decile = ranked.groupby("decile")[forward_column].mean()
    topk = {}
    for k in (5, 20, 50, 100):
        daily = ranked.groupby("datetime", group_keys=False).apply(
            lambda part: part.nlargest(k, "middle")[forward_column].mean()
        )
        topk[str(k)] = {
            "mean_forward_return": float(daily.mean()),
            "median_forward_return": float(daily.median()),
            "positive_ratio": float((daily > 0).mean()),
        }
    return {
        "rows": int(len(sample)),
        "days": int(sample["datetime"].nunique()),
        "daily_ic_mean": float(daily_ic.mean()),
        "daily_icir": float(daily_ic.mean() / daily_ic.std()),
        "daily_ic_positive_ratio": float((daily_ic > 0).mean()),
        "bottom_decile_mean": float(decile.get(0, np.nan)),
        "top_decile_mean": float(decile.get(9, np.nan)),
        "top_minus_bottom_mean": float(decile.get(9, np.nan) - decile.get(0, np.nan)),
        "topk": topk,
    }


def factor_overlap(factor_dir: Path) -> dict:
    rows = {}
    for path in sorted(factor_dir.glob("fold_*/feature_whitelist_middle_v2.json")):
        fold = int(path.parent.name.split("_")[-1])
        rows[fold] = set(json.loads(path.read_text(encoding="utf-8"))["kept"])
    folds = sorted(rows)
    pairs = []
    for left, right in zip(folds[:-1], folds[1:]):
        union = rows[left] | rows[right]
        pairs.append(
            {
                "left_fold": left,
                "right_fold": right,
                "intersection": len(rows[left] & rows[right]),
                "jaccard": len(rows[left] & rows[right]) / len(union) if union else None,
            }
        )
    common = set.intersection(*(rows[fold] for fold in folds)) if folds else set()
    return {
        "available_folds": folds,
        "common_all_folds": sorted(common),
        "common_all_count": len(common),
        "adjacent_pairs": pairs,
    }


def portfolio_oof(predictions: pd.DataFrame, prices: pd.DataFrame) -> list[dict]:
    signals = predictions.copy()
    signals["outer"] = 0.0
    signals["inner"] = np.nan
    frame = prepare_research_frame(signals, prices, UniverseRules())
    frame = add_risk_and_timing_thresholds(frame, ExperimentConfig())
    rows = []
    candidates = [("A", 5, 20)]
    candidates.extend(
        ("B", top_k, rebalance)
        for top_k in (20, 50, 100)
        for rebalance in (10, 20, 30)
    )
    for experiment, top_k, rebalance in candidates:
        cfg = ExperimentConfig(
            top_k=top_k,
            buffer_k=top_k * 2,
            rebalance_every=rebalance,
        )
        for cost in (BASE_COST, STRESS_COST):
            fold_rows = []
            for fold, part in frame.groupby("fold", sort=True):
                result = replay_topk(part, experiment=experiment, cfg=cfg, cost=cost)
                metrics = result["metrics"]
                fold_rows.append(
                    {
                        "fold": int(fold),
                        "total_return": metrics["final_nav"] / cfg.initial_cash - 1,
                        **metrics,
                    }
                )
            returns = np.array([row["total_return"] for row in fold_rows])
            rows.append(
                {
                    "experiment": experiment,
                    "cost": cost.name,
                    "top_k": top_k,
                    "rebalance_every": rebalance,
                    "median_fold_return": float(np.median(returns)),
                    "worst_fold_return": float(np.min(returns)),
                    "positive_fold_ratio": float((returns > 0).mean()),
                    "median_fold_sharpe": float(
                        np.median([row["sharpe"] for row in fold_rows])
                    ),
                    "median_fold_max_drawdown": float(
                        np.median([row["max_drawdown"] for row in fold_rows])
                    ),
                    "median_fold_turnover": float(
                        np.median([row["turnover"] for row in fold_rows])
                    ),
                    "folds": fold_rows,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["median_fold_return"]
            + row["worst_fold_return"]
            + 0.01 * row["median_fold_sharpe"]
            - 0.5 * row["median_fold_max_drawdown"]
        ),
        reverse=True,
    )


def evaluate(
    prediction_dir: Path,
    factor_dir: Path,
    output: Path,
    raw_daily_dir: Path,
    price_start: str,
    price_end: str,
    forbidden_path: Path | None = DEFAULT_FORBIDDEN,
    label_horizon: int = 5,
) -> dict:
    if label_horizon < 1:
        raise ValueError("label_horizon must be >= 1")
    predictions = read_predictions(prediction_dir)
    forbidden = load_forbidden_instruments(forbidden_path)
    forbidden_prediction_rows = 0
    if forbidden:
        before = len(predictions)
        predictions = predictions[
            ~predictions["instrument"].astype(str).str.upper().isin(forbidden)
        ].copy()
        forbidden_prediction_rows = before - len(predictions)
        print(
            f"[middle OOF] forbidden instruments={len(forbidden)} "
            f"removed_prediction_rows={forbidden_prediction_rows}",
            flush=True,
        )
    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(
        raw_daily_dir,
        instruments,
        start=price_start,
        end=price_end,
    )
    prices = prices.sort_values(["instrument", "datetime"])
    grouped = prices.groupby("instrument", sort=False)
    forward_column = f"forward_{label_horizon}d"
    prices[forward_column] = (
        grouped["close"].shift(-(label_horizon + 1))
        / grouped["close"].shift(-1)
        - 1
    )
    eligible = add_point_in_time_eligibility(prices, UniverseRules())
    sample = (
        predictions.merge(
            eligible[["datetime", "instrument", "eligible"]],
            on=["datetime", "instrument"],
            how="inner",
        )
        .merge(
            prices[["datetime", "instrument", forward_column]],
            on=["datetime", "instrument"],
            how="inner",
        )
        .dropna(subset=["middle", forward_column])
    )
    sample = sample[sample["eligible"]].copy()
    report = {
        "status": "strict_middle_oof_research_only",
        "available_folds": sorted(sample["fold"].unique().tolist()),
        "raw_daily_dir": str(raw_daily_dir),
        "price_start": price_start,
        "price_end": price_end,
        "label_horizon_trading_days": int(label_horizon),
        "forward_column": forward_column,
        "forbidden_path": str(forbidden_path) if forbidden_path else None,
        "forbidden_instruments": len(forbidden),
        "forbidden_prediction_rows": forbidden_prediction_rows,
        "aggregate": score_sample(sample, forward_column),
        "per_fold": {
            str(fold): score_sample(part, forward_column)
            for fold, part in sample.groupby("fold", sort=True)
        },
        "factor_overlap": factor_overlap(factor_dir),
        "portfolio_oof": portfolio_oof(predictions, prices),
        "deployment_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir", default=str(HERE / "outputs" / "oof" / "simple" / "middle")
    )
    parser.add_argument(
        "--factor-dir",
        default=str(HERE / "outputs" / "factor_reports" / "simple"),
    )
    parser.add_argument(
        "--output",
        default=str(HERE / "outputs" / "oof" / "simple" / "middle_evaluation.json"),
    )
    parser.add_argument(
        "--raw-daily-dir",
        default=str(REPO_ROOT / "data" / "A_Stock_daily_qfq"),
    )
    parser.add_argument("--price-start", default="2022-01-04")
    parser.add_argument("--price-end", default="2026-04-28")
    parser.add_argument("--forbidden-symbols", default=str(DEFAULT_FORBIDDEN))
    parser.add_argument("--label-horizon", type=int, default=5)
    args = parser.parse_args()
    report = evaluate(
        Path(args.prediction_dir).resolve(),
        Path(args.factor_dir).resolve(),
        Path(args.output).resolve(),
        Path(args.raw_daily_dir).resolve(),
        args.price_start,
        args.price_end,
        Path(args.forbidden_symbols).resolve() if args.forbidden_symbols else None,
        args.label_horizon,
    )
    aggregate = report["aggregate"]
    print(
        f"[middle OOF] folds={len(report['available_folds'])} "
        f"IC={aggregate['daily_ic_mean']:+.4f} "
        f"ICIR={aggregate['daily_icir']:+.3f} "
        f"spread={aggregate['top_minus_bottom_mean']:+.3%}"
    )
    for row in report["portfolio_oof"][:10]:
        print(
            f"[{row['experiment']}/{row['cost']}] top={row['top_k']} "
            f"rebalance={row['rebalance_every']} "
            f"median={row['median_fold_return']:+.2%} "
            f"worst={row['worst_fold_return']:+.2%} "
            f"positive={row['positive_fold_ratio']:.0%} "
            f"mdd={row['median_fold_max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
