"""Robust outer-layer-only regime allocation experiment.

This is a structural benchmark for the current hierarchy: outer signals set a
target position, while middle and inner signals are ignored. Signals are
lagged by one trading day through optimize_thresholds.load_data.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_strategy_robust import (
    COST_SCENARIOS,
    CostScenario,
    _aggregate_window,
    _benchmark_ann_return,
    _deployment_gate,
    _scenario_score,
)
from optimize_thresholds import DEFAULT_INITIAL_CASH, _metrics, load_data, serializable_metrics


@dataclass(frozen=True)
class OuterCandidate:
    outer_upper: float
    outer_lower: float
    bull_position: float
    range_position: float
    bear_position: float
    rebalance_band: float


def _round_lot(volume: float) -> int:
    return max(0, int(volume // 100) * 100)


def _replay(
    df: pd.DataFrame,
    candidate: OuterCandidate,
    *,
    initial_held: int,
    cost: CostScenario,
) -> dict:
    cash = float(DEFAULT_INITIAL_CASH)
    held = _round_lot(initial_held)
    navs: list[float] = []
    trades = 0
    turnover = 0.0
    bull_days = range_days = bear_days = 0

    for _, row in df.iterrows():
        open_p = float(row["open"])
        close_p = float(row["close"])
        signal = float(row["outer"])
        nav_open = cash + held * open_p
        current_position = held * open_p / max(nav_open, 1e-12)

        if signal > candidate.outer_upper:
            target_position = candidate.bull_position
            bull_days += 1
        elif signal < candidate.outer_lower:
            target_position = candidate.bear_position
            bear_days += 1
        else:
            target_position = candidate.range_position
            range_days += 1

        if abs(current_position - target_position) >= candidate.rebalance_band:
            target_held = _round_lot(nav_open * target_position / open_p)
            delta = target_held - held
            if delta > 0:
                buy_px = open_p * (1 + cost.slippage)
                affordable = _round_lot(
                    max(0.0, cash - cost.min_cost)
                    / (buy_px * (1 + cost.buy_cost_rate))
                )
                volume = min(delta, affordable)
                if volume > 0:
                    value = volume * buy_px
                    fee = max(value * cost.buy_cost_rate, cost.min_cost)
                    cash -= value + fee
                    held += volume
                    trades += 1
                    turnover += value / max(nav_open, 1e-12)
            elif delta < 0:
                volume = min(-delta, held)
                if volume > 0:
                    sell_px = open_p * (1 - cost.slippage)
                    value = volume * sell_px
                    fee = max(value * cost.sell_cost_rate, cost.min_cost)
                    cash += value - fee
                    held -= volume
                    trades += 1
                    turnover += value / max(nav_open, 1e-12)
        navs.append(cash + held * close_p)

    return _metrics(
        np.asarray(navs, dtype=float),
        trades,
        turnover,
        bull_days,
        range_days,
        bear_days,
        0,
        trades,
    )


def _evaluate_scenario(
    df: pd.DataFrame,
    candidate: OuterCandidate,
    initial_held: int,
    cost: CostScenario,
) -> tuple[dict, float]:
    metrics = _replay(df, candidate, initial_held=initial_held, cost=cost)
    benchmark = _benchmark_ann_return(df, DEFAULT_INITIAL_CASH, initial_held)
    score = _scenario_score(metrics, benchmark, len(df))
    return metrics, score


def _selection_score(dev: pd.DataFrame, candidate: OuterCandidate) -> tuple[float, dict]:
    scores = []
    rows = []
    for fold_no, idx in enumerate(np.array_split(np.arange(len(dev)), 3), start=1):
        fold = dev.iloc[idx].reset_index(drop=True)
        for initial_held, cost in itertools.product((0, 2000), COST_SCENARIOS):
            metrics, score = _evaluate_scenario(fold, candidate, initial_held, cost)
            scores.append(score)
            rows.append({
                "fold": fold_no,
                "initial_held": initial_held,
                "cost": cost.name,
                "score": score,
                **serializable_metrics(metrics),
            })
    arr = np.asarray(scores, dtype=float)
    robust = float(arr.mean() - 0.35 * arr.std() + 0.25 * np.quantile(arr, 0.20))
    return robust, {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "q20": float(np.quantile(arr, 0.20)),
        "folds": rows,
    }


def _evaluate_window(df: pd.DataFrame, candidate: OuterCandidate) -> dict:
    result = {}
    for initial_held, cost in itertools.product((0, 2000), COST_SCENARIOS):
        metrics, score = _evaluate_scenario(df, candidate, initial_held, cost)
        result[f"held_{initial_held}_{cost.name}"] = {
            "score": score,
            "benchmark_ann_return": _benchmark_ann_return(
                df, DEFAULT_INITIAL_CASH, initial_held
            ),
            **serializable_metrics(metrics),
        }
    return result


def _candidates(dev: pd.DataFrame) -> list[OuterCandidate]:
    quantiles = {
        p: float(dev["outer"].quantile(p))
        for p in (0.20, 0.30, 0.40, 0.60, 0.70, 0.80)
    }
    out = []
    for lower_q, upper_q, bear, middle, bull, band in itertools.product(
        (0.20, 0.30, 0.40),
        (0.60, 0.70, 0.80),
        (0.00, 0.10),
        (0.20, 0.35, 0.50),
        (0.50, 0.65, 0.80),
        (0.05, 0.10, 0.15),
    ):
        if bear <= middle <= bull:
            out.append(OuterCandidate(
                outer_upper=quantiles[upper_q],
                outer_lower=quantiles[lower_q],
                bull_position=bull,
                range_position=middle,
                bear_position=bear,
                rebalance_band=band,
            ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", default="artifacts/robust_v2")
    parser.add_argument("--target", default="SZ301536")
    parser.add_argument("--start", default="2025-10-01")
    parser.add_argument("--end", default="2026-04-28")
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    df = load_data(artifacts_dir, args.target, args.start, args.end)
    test_n = max(20, int(round(len(df) * args.test_ratio)))
    dev = df.iloc[:-test_n].reset_index(drop=True)
    test = df.iloc[-test_n:].reset_index(drop=True)
    candidates = _candidates(dev)
    print(f"[INFO] aligned={len(df)} dev={len(dev)} test={len(test)} candidates={len(candidates)}")

    ranked = []
    for i, candidate in enumerate(candidates, start=1):
        score, detail = _selection_score(dev, candidate)
        ranked.append({"score": score, "candidate": candidate, "detail": detail})
        if i % 250 == 0 or i == len(candidates):
            print(f"[SEARCH] {i}/{len(candidates)} best={max(x['score'] for x in ranked):+.4f}", flush=True)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    best = ranked[0]
    windows = {
        "dev": _evaluate_window(dev, best["candidate"]),
        "test": _evaluate_window(test, best["candidate"]),
        "all": _evaluate_window(df, best["candidate"]),
    }
    report = {
        "artifacts_dir": str(artifacts_dir),
        "target": args.target,
        "selection_protocol": {
            "signal_lag_days": 1,
            "dev_days": len(dev),
            "test_days": len(test),
            "test_start": str(test["datetime"].min().date()),
            "test_end": str(test["datetime"].max().date()),
            "initial_holdings": [0, 2000],
            "cost_scenarios": [asdict(x) for x in COST_SCENARIOS],
        },
        "best": {
            "selection_score": best["score"],
            "candidate": asdict(best["candidate"]),
            "selection_detail": best["detail"],
            **windows,
            "dev_aggregate": _aggregate_window(windows["dev"]),
            "test_aggregate": _aggregate_window(windows["test"]),
            "all_aggregate": _aggregate_window(windows["all"]),
            "deployment_gate": _deployment_gate(windows["test"]),
        },
        "top_candidates": [
            {
                "selection_score": x["score"],
                "candidate": asdict(x["candidate"]),
                "selection_summary": {
                    k: v for k, v in x["detail"].items() if k != "folds"
                },
            }
            for x in ranked[:20]
        ],
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else artifacts_dir / "outer_regime_optimization.json"
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[BEST]", json.dumps(asdict(best["candidate"]), ensure_ascii=False))
    print("[TEST]", json.dumps(report["best"]["test_aggregate"], ensure_ascii=False))
    print("[GATE]", json.dumps(report["best"]["deployment_gate"], ensure_ascii=False))
    print(f"[OK] {output}")


if __name__ == "__main__":
    main()
