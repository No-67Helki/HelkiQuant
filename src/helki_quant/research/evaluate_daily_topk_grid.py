from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from concentration_constraints import load_group_metadata
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
DATA = REPO_ROOT / "data"
DEFAULT_MIDDLE = HERE / "outputs" / "oof_combined" / "middle_oof_pit_de2_srfs_es.csv"
DEFAULT_GROUP_METADATA = DATA / "industry_theme_pit.csv"


def load_middle_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime"])
    if "middle" not in frame.columns and "pred_middle" in frame.columns:
        frame = frame.rename(columns={"pred_middle": "middle"})
    if "middle" not in frame.columns:
        raise ValueError(f"{path} must contain a middle column")
    frame["outer"] = 0.0
    frame["inner"] = np.nan
    return frame[["datetime", "instrument", "outer", "middle", "inner"]]


def safe_replay(
    frame: pd.DataFrame,
    experiment: str,
    cfg: ExperimentConfig,
    cost,
    group_metadata: pd.DataFrame,
):
    if frame.empty:
        return None
    try:
        return replay_topk(
            frame,
            experiment=experiment,
            cfg=cfg,
            cost=cost,
            group_metadata=group_metadata,
        )
    except (IndexError, KeyError, ValueError):
        return None


def fold_summary(
    frame: pd.DataFrame,
    folds: list[dict],
    experiment: str,
    cfg: ExperimentConfig,
    cost,
    group_metadata: pd.DataFrame,
    start_signal: str,
    end_signal: str | None,
) -> dict:
    rows = []
    start_ts = pd.Timestamp(start_signal)
    end_ts = pd.Timestamp(end_signal) if end_signal else None
    for fold in folds:
        lo = max(pd.Timestamp(fold["test_start"]), start_ts)
        hi = pd.Timestamp(fold["test_end"])
        if end_ts is not None:
            hi = min(hi, end_ts)
        if lo > hi:
            continue
        part = frame[frame["signal_date"].between(lo, hi)]
        result = safe_replay(part, experiment, cfg, cost, group_metadata)
        if result is None:
            continue
        metrics = result["metrics"]
        rows.append(
            {
                "fold": fold["fold"],
                "signal_start": str(lo.date()),
                "signal_end": str(hi.date()),
                "total_return": metrics["final_nav"] / cfg.initial_cash - 1,
                **metrics,
            }
        )
    returns = np.array([row["total_return"] for row in rows], dtype=float)
    sharpes = np.array([row["sharpe"] for row in rows], dtype=float)
    drawdowns = np.array([row["max_drawdown"] for row in rows], dtype=float)
    turnovers = np.array([row["turnover"] for row in rows], dtype=float)
    trades = np.array([row["trades"] for row in rows], dtype=float)
    return {
        "folds": rows,
        "evaluated_folds": len(rows),
        "median_fold_return": float(np.median(returns)) if len(returns) else None,
        "worst_fold_return": float(np.min(returns)) if len(returns) else None,
        "positive_fold_ratio": float((returns > 0).mean()) if len(returns) else None,
        "median_fold_sharpe": float(np.median(sharpes)) if len(sharpes) else None,
        "median_fold_max_drawdown": float(np.median(drawdowns)) if len(drawdowns) else None,
        "worst_fold_max_drawdown": float(np.max(drawdowns)) if len(drawdowns) else None,
        "median_fold_turnover": float(np.median(turnovers)) if len(turnovers) else None,
        "median_fold_trades": float(np.median(trades)) if len(trades) else None,
    }


def diagnostic_score(folds: dict) -> float:
    """Prefer robust return, penalize drawdown and excessive turnover."""
    return (
        (folds["median_fold_return"] or 0.0)
        + (folds["worst_fold_return"] or 0.0)
        + 0.02 * (folds["median_fold_sharpe"] or 0.0)
        - (folds["median_fold_max_drawdown"] or 0.0)
        - 0.003 * (folds["median_fold_turnover"] or 0.0)
    )


def profile_configs() -> list[tuple[str, str, float]]:
    return [
        ("growth_full", "B", 1.0),
        ("balanced_60", "C", 0.60),
        ("defensive_40", "C", 0.40),
    ]


def evaluate(
    middle_path: Path,
    folds_path: Path,
    group_metadata_path: Path,
    output_path: Path,
    start_signal: str,
    end_signal: str | None,
    top_ks: list[int],
    rebalances: list[int],
    min_amounts: list[float],
    industry_caps: list[float],
    raw_daily_dir: Path,
    price_start: str,
    price_end: str,
) -> dict:
    predictions = load_middle_predictions(middle_path)
    if start_signal:
        predictions = predictions[predictions["datetime"] >= pd.Timestamp(start_signal)].copy()
    if end_signal:
        predictions = predictions[predictions["datetime"] <= pd.Timestamp(end_signal)].copy()
    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(
        raw_daily_dir,
        instruments,
        start=price_start,
        end=price_end,
    )
    group_metadata = load_group_metadata(group_metadata_path, "industry")
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    rows = []
    for min_amount in min_amounts:
        rules = UniverseRules(min_listing_days=250, min_avg_amount=min_amount)
        base_frame = prepare_research_frame(predictions, prices, rules)
        for top_k in top_ks:
            for rebalance in rebalances:
                for industry_cap in industry_caps:
                    for profile, experiment, risk_base in profile_configs():
                        cfg = ExperimentConfig(
                            top_k=top_k,
                            buffer_k=top_k * 2,
                            rebalance_every=rebalance,
                            risk_base=risk_base,
                            risk_slope=0.0,
                            risk_min=risk_base,
                            risk_max=risk_base,
                            max_group_fraction=industry_cap,
                        )
                        frame = add_risk_and_timing_thresholds(base_frame, cfg)
                        for cost in (BASE_COST, STRESS_COST):
                            fold_metrics = fold_summary(
                                frame,
                                folds,
                                experiment,
                                cfg,
                                cost,
                                group_metadata,
                                start_signal,
                                end_signal,
                            )
                            row = {
                                "profile": profile,
                                "experiment": experiment,
                                "cost": cost.name,
                                "min_listing_days": rules.min_listing_days,
                                "min_avg_amount": min_amount,
                                "top_k": top_k,
                                "buffer_k": cfg.buffer_k,
                                "rebalance_every": rebalance,
                                "industry_cap": industry_cap,
                                "risk_base": risk_base,
                                "config": asdict(cfg),
                                "fold_metrics": fold_metrics,
                                "diagnostic_score": diagnostic_score(fold_metrics),
                            }
                            rows.append(row)
                            print(
                                f"[topk] profile={profile} cost={cost.name} "
                                f"top={top_k} rebalance={rebalance} "
                                f"liq={min_amount/1e6:.0f}m cap={industry_cap:.0%} "
                                f"median={fold_metrics['median_fold_return']:+.2%} "
                                f"worst={fold_metrics['worst_fold_return']:+.2%} "
                                f"turn={fold_metrics['median_fold_turnover']:.2f}",
                                flush=True,
                            )
    report = {
        "status": "daily_topk_grid_with_pit_industry_cap_research_only",
        "middle_prediction": str(middle_path),
        "folds_path": str(folds_path),
        "group_metadata": str(group_metadata_path),
        "raw_daily_dir": str(raw_daily_dir),
        "price_start": price_start,
        "price_end": price_end,
        "start_signal": start_signal,
        "end_signal": end_signal,
        "grid": {
            "top_k": top_ks,
            "rebalance_every": rebalances,
            "min_avg_amount": min_amounts,
            "industry_cap": industry_caps,
            "profiles": [name for name, _, _ in profile_configs()],
            "costs": [BASE_COST.name, STRESS_COST.name],
        },
        "ranking_policy": (
            "Diagnostic score = median fold return + worst fold return + "
            "0.02*median Sharpe - median drawdown - 0.003*median turnover. "
            "Fold metrics are primary; no deployment is allowed."
        ),
        "deployment_allowed": False,
        "candidates": sorted(rows, key=lambda row: row["diagnostic_score"], reverse=True),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_csv_numbers(text: str, cast=float) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--middle", default=str(DEFAULT_MIDDLE))
    parser.add_argument("--folds", default=str(HERE / "outputs" / "purged_folds.json"))
    parser.add_argument("--group-metadata", default=str(DEFAULT_GROUP_METADATA))
    parser.add_argument(
        "--output", default=str(HERE / "outputs" / "daily_topk_grid_industry_cap.json")
    )
    parser.add_argument("--start-signal", default="2025-01-03")
    parser.add_argument("--end-signal", default="2026-04-02")
    parser.add_argument("--top-k", default="20,50,100,150")
    parser.add_argument("--rebalance", default="20,30")
    parser.add_argument("--min-amount", default="100000000,125000000,150000000")
    parser.add_argument("--industry-cap", default="0.2,0.3,0.4")
    parser.add_argument(
        "--raw-daily-dir",
        default=str(DATA / "A_Stock_daily_qfq"),
    )
    parser.add_argument("--price-start", default="2022-01-04")
    parser.add_argument("--price-end", default="2026-04-28")
    args = parser.parse_args()
    report = evaluate(
        Path(args.middle).resolve(),
        Path(args.folds).resolve(),
        Path(args.group_metadata).resolve(),
        Path(args.output).resolve(),
        args.start_signal,
        args.end_signal,
        parse_csv_numbers(args.top_k, int),
        parse_csv_numbers(args.rebalance, int),
        parse_csv_numbers(args.min_amount, float),
        parse_csv_numbers(args.industry_cap, float),
        Path(args.raw_daily_dir).resolve(),
        args.price_start,
        args.price_end,
    )
    print("[topk] best candidates:")
    for row in report["candidates"][:20]:
        folds = row["fold_metrics"]
        print(
            f"  {row['profile']}/{row['cost']} top={row['top_k']} "
            f"rebalance={row['rebalance_every']} liq={row['min_avg_amount']/1e6:.0f}m "
            f"cap={row['industry_cap']:.0%} median={folds['median_fold_return']:+.2%} "
            f"worst={folds['worst_fold_return']:+.2%} "
            f"positive={folds['positive_fold_ratio']:.0%} "
            f"mdd={folds['median_fold_max_drawdown']:.2%} "
            f"turn={folds['median_fold_turnover']:.2f}"
        )


if __name__ == "__main__":
    main()
