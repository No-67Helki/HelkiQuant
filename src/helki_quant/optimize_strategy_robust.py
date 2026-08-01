"""Robust strategy search for the three-layer target-stock strategy.

Selection uses only the development window. The final chronological test
window is evaluated once after candidate selection. Replays use one decision
per day because the legacy scan60 mode interpolates prices and repeats the
same daily signal, which is not a valid minute-level backtest.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from decision_core import StrategyParams
from optimize_thresholds import (
    DEFAULT_INITIAL_CASH,
    ReplayLimits,
    ThresholdParams,
    load_data,
    replay,
    serializable_metrics,
)


@dataclass(frozen=True)
class SizeProfile:
    name: str
    wave_buy_pct: float
    wave_sell_pct: float
    range_swing_pct: float
    intraday_t_pct: float
    max_position_pct: float


@dataclass(frozen=True)
class CostScenario:
    name: str
    buy_cost_rate: float
    sell_cost_rate: float
    slippage: float
    min_cost: float = 5.0


@dataclass(frozen=True)
class Candidate:
    thresholds: ThresholdParams
    size: SizeProfile
    t_mode: str

    def strategy_params(self) -> StrategyParams:
        t_pct = 0.0 if self.t_mode == "off" else self.size.intraday_t_pct
        return StrategyParams(
            outer_upper=self.thresholds.outer_upper,
            outer_lower=self.thresholds.outer_lower,
            middle_buy_thresh=self.thresholds.middle_buy,
            middle_sell_thresh=self.thresholds.middle_sell,
            inner_buy_thresh=self.thresholds.inner_buy,
            inner_sell_thresh=self.thresholds.inner_sell,
            trade_unit=100,
            wave_buy_pct=self.size.wave_buy_pct,
            wave_sell_pct=self.size.wave_sell_pct,
            range_swing_pct=self.size.range_swing_pct,
            intraday_t_pct=t_pct,
            max_position_pct=self.size.max_position_pct,
            min_cash_reserve=1000.0,
            min_buy_lots=1,
            exit_on_bull_end=False,
        )


SIZE_PROFILES = (
    SizeProfile("defensive", 0.10, 0.20, 0.05, 0.06, 0.55),
    SizeProfile("balanced", 0.15, 0.25, 0.08, 0.10, 0.65),
    SizeProfile("active", 0.20, 0.30, 0.10, 0.15, 0.70),
)

COST_SCENARIOS = (
    CostScenario("base", 0.0005, 0.0015, 0.0002),
    CostScenario("stress", 0.0010, 0.0025, 0.0005),
)


def _threshold_candidates(dev: pd.DataFrame) -> list[tuple[ThresholdParams, str]]:
    q = {
        col: {p: float(dev[col].quantile(p)) for p in (0.15, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.85)}
        for col in ("outer", "middle", "inner")
    }
    result: list[tuple[ThresholdParams, str]] = []
    for ol, ou, ml, mu in itertools.product((0.20, 0.30, 0.40), (0.60, 0.70, 0.80), repeat=2):
        base = {
            "outer_upper": q["outer"][ou],
            "outer_lower": q["outer"][ol],
            "middle_buy": q["middle"][mu],
            "middle_sell": q["middle"][ml],
        }
        result.append((
            ThresholdParams(**base, inner_buy=1.10, inner_sell=-0.10),
            "off",
        ))
        result.append((
            ThresholdParams(
                **base,
                inner_buy=q["inner"][0.85],
                inner_sell=q["inner"][0.15],
            ),
            "high_confidence",
        ))
    return result


def _benchmark_ann_return(df: pd.DataFrame, initial_cash: float, initial_held: int) -> float:
    if len(df) < 2:
        return 0.0
    nav0 = initial_cash + initial_held * float(df.iloc[0]["close"])
    nav1 = initial_cash + initial_held * float(df.iloc[-1]["close"])
    if nav0 <= 0 or nav1 <= 0:
        return 0.0
    return float((nav1 / nav0) ** (252 / len(df)) - 1)


def _scenario_score(metrics: dict, benchmark_ann: float, n_days: int) -> float:
    sharpe = float(np.clip(metrics["sharpe"], -4.0, 4.0))
    excess_ann = float(np.clip(metrics["ann_return"] - benchmark_ann, -1.5, 1.5))
    score = 0.50 * sharpe + 1.25 * excess_ann - 2.0 * metrics["mdd"]
    score -= 0.015 * max(0.0, metrics["turnover"] - n_days / 20)
    min_trades = max(3, int(np.ceil(n_days * 0.10)))
    if metrics["n_trades"] < min_trades:
        score -= 0.35 * (min_trades - metrics["n_trades"])
    return float(score)


def _replay_candidate(
    df: pd.DataFrame,
    candidate: Candidate,
    *,
    initial_held: int,
    cost: CostScenario,
) -> tuple[dict, float]:
    metrics = replay(
        df,
        candidate.thresholds,
        initial_cash=DEFAULT_INITIAL_CASH,
        initial_held=initial_held,
        limits=ReplayLimits(
            max_swing_per_day=1,
            max_t0_per_day=1,
            swing_unlimited_in_range=False,
        ),
        scan_mode="daily",
        strategy_params=candidate.strategy_params(),
        buy_cost_rate=cost.buy_cost_rate,
        sell_cost_rate=cost.sell_cost_rate,
        slippage=cost.slippage,
        min_cost=cost.min_cost,
    )
    benchmark_ann = _benchmark_ann_return(df, DEFAULT_INITIAL_CASH, initial_held)
    return metrics, _scenario_score(metrics, benchmark_ann, len(df))


def _robust_selection_score(dev: pd.DataFrame, candidate: Candidate) -> tuple[float, dict]:
    fold_indices = np.array_split(np.arange(len(dev)), 3)
    scores: list[float] = []
    fold_summary: list[dict] = []
    for fold_no, idx in enumerate(fold_indices, start=1):
        fold = dev.iloc[idx].reset_index(drop=True)
        for initial_held, cost in itertools.product((0, 2000), COST_SCENARIOS):
            metrics, score = _replay_candidate(
                fold, candidate, initial_held=initial_held, cost=cost
            )
            scores.append(score)
            fold_summary.append({
                "fold": fold_no,
                "initial_held": initial_held,
                "cost": cost.name,
                "score": score,
                "sharpe": metrics["sharpe"],
                "ann_return": metrics["ann_return"],
                "mdd": metrics["mdd"],
                "n_trades": metrics["n_trades"],
                "turnover": metrics["turnover"],
            })
    arr = np.asarray(scores, dtype=float)
    robust = float(arr.mean() - 0.35 * arr.std() + 0.25 * np.quantile(arr, 0.20))
    return robust, {"mean": float(arr.mean()), "std": float(arr.std()), "q20": float(np.quantile(arr, 0.20)), "folds": fold_summary}


def _candidate_dict(candidate: Candidate) -> dict:
    return {
        "thresholds": asdict(candidate.thresholds),
        "size": asdict(candidate.size),
        "t_mode": candidate.t_mode,
    }


def _evaluate_window(df: pd.DataFrame, candidate: Candidate) -> dict:
    result = {}
    for initial_held, cost in itertools.product((0, 2000), COST_SCENARIOS):
        metrics, score = _replay_candidate(
            df, candidate, initial_held=initial_held, cost=cost
        )
        key = f"held_{initial_held}_{cost.name}"
        result[key] = {
            "score": score,
            "benchmark_ann_return": _benchmark_ann_return(
                df, DEFAULT_INITIAL_CASH, initial_held
            ),
            **serializable_metrics(metrics),
        }
    return result


def _aggregate_window(result: dict) -> dict:
    keys = ("score", "sharpe", "ann_return", "mdd", "n_trades", "turnover")
    out = {}
    for key in keys:
        values = [v[key] for v in result.values()]
        worst = np.max(values) if key in {"mdd", "turnover"} else np.min(values)
        out[key] = {"mean": float(np.mean(values)), "worst": float(worst)}
    return out


def _deployment_gate(result: dict) -> dict:
    scores = [v["score"] for v in result.values()]
    returns = [v["ann_return"] for v in result.values()]
    passed = min(scores) >= 0.0 and min(returns) >= 0.0
    return {
        "passed": bool(passed),
        "rule": "all untouched-test cost/holding scenarios require score >= 0 and annualized return >= 0",
        "worst_score": float(min(scores)),
        "worst_ann_return": float(min(returns)),
    }


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
    if len(df) < 90:
        raise ValueError(f"Need at least 90 aligned trading days, got {len(df)}")
    test_n = max(20, int(round(len(df) * args.test_ratio)))
    dev = df.iloc[:-test_n].reset_index(drop=True)
    test = df.iloc[-test_n:].reset_index(drop=True)

    threshold_modes = _threshold_candidates(dev)
    candidates = [
        Candidate(thresholds=thresholds, size=size, t_mode=t_mode)
        for thresholds, t_mode in threshold_modes
        for size in SIZE_PROFILES
    ]
    print(
        f"[INFO] aligned={len(df)} dev={len(dev)} test={len(test)} "
        f"candidates={len(candidates)}"
    )
    print(
        f"[INFO] untouched test: {test['datetime'].min().date()} -> "
        f"{test['datetime'].max().date()}"
    )

    ranked = []
    for i, candidate in enumerate(candidates, start=1):
        score, detail = _robust_selection_score(dev, candidate)
        ranked.append({"score": score, "candidate": candidate, "detail": detail})
        if i % 50 == 0 or i == len(candidates):
            print(f"[SEARCH] {i}/{len(candidates)} best={max(x['score'] for x in ranked):+.4f}", flush=True)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    best = ranked[0]["candidate"]

    # Explicitly retain the best no-T candidate for an apples-to-apples check.
    best_no_t_entry = next(x for x in ranked if x["candidate"].t_mode == "off")
    best_no_t = best_no_t_entry["candidate"]
    best_with_t_entry = next(
        x for x in ranked if x["candidate"].t_mode == "high_confidence"
    )
    best_with_t = best_with_t_entry["candidate"]

    report = {
        "artifacts_dir": str(artifacts_dir),
        "target": args.target,
        "selection_protocol": {
            "aligned_days": len(df),
            "dev_days": len(dev),
            "test_days": len(test),
            "dev_start": str(dev["datetime"].min().date()),
            "dev_end": str(dev["datetime"].max().date()),
            "test_start": str(test["datetime"].min().date()),
            "test_end": str(test["datetime"].max().date()),
            "replay_mode": "daily",
            "signal_lag_days": 1,
            "selection_folds": 3,
            "initial_holdings": [0, 2000],
            "cost_scenarios": [asdict(x) for x in COST_SCENARIOS],
        },
        "best": {
            "selection_score": ranked[0]["score"],
            "candidate": _candidate_dict(best),
            "selection_detail": ranked[0]["detail"],
            "dev": _evaluate_window(dev, best),
            "test": _evaluate_window(test, best),
            "all": _evaluate_window(df, best),
        },
        "best_no_t": {
            "selection_score": best_no_t_entry["score"],
            "candidate": _candidate_dict(best_no_t),
            "selection_detail": best_no_t_entry["detail"],
            "dev": _evaluate_window(dev, best_no_t),
            "test": _evaluate_window(test, best_no_t),
            "all": _evaluate_window(df, best_no_t),
        },
        "best_with_t": {
            "selection_score": best_with_t_entry["score"],
            "candidate": _candidate_dict(best_with_t),
            "selection_detail": best_with_t_entry["detail"],
            "dev": _evaluate_window(dev, best_with_t),
            "test": _evaluate_window(test, best_with_t),
            "all": _evaluate_window(df, best_with_t),
        },
        "top_candidates": [
            {
                "selection_score": x["score"],
                "candidate": _candidate_dict(x["candidate"]),
                "selection_summary": {
                    k: v for k, v in x["detail"].items() if k != "folds"
                },
            }
            for x in ranked[:20]
        ],
    }
    for key in ("best", "best_no_t", "best_with_t"):
        for window in ("dev", "test", "all"):
            report[key][f"{window}_aggregate"] = _aggregate_window(
                report[key][window]
            )
        report[key]["deployment_gate"] = _deployment_gate(report[key]["test"])

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else artifacts_dir / "strategy_optimization_robust.json"
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[BEST]")
    print(json.dumps(_candidate_dict(best), ensure_ascii=False, indent=2))
    print("[BEST TEST AGGREGATE]")
    print(json.dumps(report["best"]["test_aggregate"], ensure_ascii=False, indent=2))
    print("[BEST NO-T TEST AGGREGATE]")
    print(json.dumps(report["best_no_t"]["test_aggregate"], ensure_ascii=False, indent=2))
    print("[BEST WITH-T TEST AGGREGATE]")
    print(json.dumps(report["best_with_t"]["test_aggregate"], ensure_ascii=False, indent=2))
    print(f"[OK] {output}")


if __name__ == "__main__":
    main()
