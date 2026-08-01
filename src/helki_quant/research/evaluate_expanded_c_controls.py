from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from concentration_constraints import load_group_metadata
from capital_aware_allocation import ALLOCATION_MODES
from evaluate_daily_topk_grid import load_middle_predictions, parse_csv_numbers
from minute_mapped_topk_replay import (
    MappedProfile,
    MappedReplayConfig,
    fold_summary,
    prepare_daily_frame,
    replay_mapped,
)
from portfolio_experiments import BASE_COST, STRESS_COST
from universe import load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_MIDDLE = HERE / "outputs" / "oof_combined" / "middle_oof_pit_de2_srfs_es.csv"
DEFAULT_GROUP_METADATA = DATA / "industry_theme_pit.csv"
DEFAULT_WINDOWS = DATA / "_research_topk_minute_windows_2025_2026.csv"
DEFAULT_FORBIDDEN = REPO_ROOT / "gm_c_forbidden_symbols.csv"


def gm_to_local_symbol(value: object) -> str:
    text = str(value).strip().upper()
    if text.startswith("SHSE."):
        return f"SH{text[-6:]}"
    if text.startswith("SZSE."):
        return f"SZ{text[-6:]}"
    return text


def load_forbidden_instruments(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    frame = pd.read_csv(path, dtype=str).fillna("")
    values: set[str] = set()
    for row in frame.to_dict(orient="records"):
        local = next(
            (
                str(row.get(column, "")).strip().upper()
                for column in ("instrument", "local_instrument")
                if str(row.get(column, "")).strip()
            ),
            "",
        )
        if local and local != "NAN":
            values.add(local)
            continue
        gm_symbol = next(
            (
                row.get(column, "")
                for column in ("gm_symbol", "symbol")
                if str(row.get(column, "")).strip()
            ),
            "",
        )
        values.add(gm_to_local_symbol(gm_symbol))
    return {value for value in values if value and value != "NAN"}


def control_score(row: dict) -> float:
    metrics = row["metrics"]
    folds = row["fold_summary"]
    return (
        1.5 * (folds["worst_fold_return"] or 0.0)
        + (folds["median_fold_return"] or 0.0)
        + 0.25 * metrics["total_return"]
        - 0.8 * metrics["max_drawdown"]
        - 0.015 * metrics["turnover"]
    )


def evaluate_candidate(
    frame: pd.DataFrame,
    minute_windows: pd.DataFrame,
    minute_pool: set[str],
    group_metadata: pd.DataFrame,
    folds: list[dict],
    profile: MappedProfile,
    cfg: MappedReplayConfig,
    cost,
    start_signal: str,
    end_signal: str,
    allocation_mode: str,
    outer_risk_threshold: float | None,
    outer_risk_floor: float | None,
) -> dict:
    full = replay_mapped(
        frame,
        minute_windows,
        minute_pool,
        group_metadata,
        profile,
        cfg,
        cost,
        "preserve_target_weights",
        allocation_mode,
        outer_risk_threshold,
        outer_risk_floor,
    )
    fold_rows = []
    for fold in folds:
        lo = max(pd.Timestamp(fold["test_start"]), pd.Timestamp(start_signal))
        hi = min(pd.Timestamp(fold["test_end"]), pd.Timestamp(end_signal))
        if lo > hi:
            continue
        part = frame[frame["signal_date"].between(lo, hi)].copy()
        if part.empty:
            continue
        result = replay_mapped(
            part,
            minute_windows,
            minute_pool,
            group_metadata,
            profile,
            cfg,
            cost,
            "preserve_target_weights",
            allocation_mode,
            outer_risk_threshold,
            outer_risk_floor,
        )
        fold_rows.append(
            {
                "fold": fold["fold"],
                "signal_start": str(lo.date()),
                "signal_end": str(hi.date()),
                "metrics": result["metrics"],
            }
        )
    full["folds"] = fold_rows
    full["fold_summary"] = fold_summary(fold_rows, cfg.initial_cash)
    full["control_score"] = control_score(full)
    return full


def write_summary(rows: list[dict], path: Path) -> None:
    fields = [
        "rank",
        "profile",
        "cost",
        "top_k",
        "rebalance_every",
        "risk_budget",
        "industry_cap",
        "buffer_multiple",
        "allocation_mode",
        "outer_risk_threshold",
        "outer_risk_floor",
        "control_score",
        "total_return",
        "worst_fold_return",
        "median_fold_return",
        "positive_fold_ratio",
        "max_drawdown",
        "turnover",
        "avg_mapped_count",
        "min_mapped_count",
    ]
    ranked = sorted(rows, key=lambda row: row["control_score"], reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            profile = row["profile"]
            metrics = row["metrics"]
            folds = row["fold_summary"]
            writer.writerow(
                {
                    "rank": rank,
                    "profile": profile["name"],
                    "cost": row["cost"]["name"],
                    "top_k": profile["top_k"],
                    "rebalance_every": profile["rebalance_every"],
                    "risk_budget": profile["risk_budget"],
                    "industry_cap": profile["industry_cap"],
                    "buffer_multiple": row["config"]["buffer_multiple"],
                    "allocation_mode": row["allocation_mode"],
                    "outer_risk_threshold": row.get("outer_risk_threshold"),
                    "outer_risk_floor": row.get("outer_risk_floor"),
                    "control_score": row["control_score"],
                    "total_return": metrics["total_return"],
                    "worst_fold_return": folds["worst_fold_return"],
                    "median_fold_return": folds["median_fold_return"],
                    "positive_fold_ratio": folds["positive_fold_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "turnover": metrics["turnover"],
                    "avg_mapped_count": metrics["avg_mapped_count"],
                    "min_mapped_count": metrics["min_mapped_count"],
                }
            )


def run(
    middle_path: Path,
    windows_path: Path,
    folds_path: Path,
    group_metadata_path: Path,
    output_path: Path,
    summary_path: Path,
    start_signal: str,
    end_signal: str,
    top_ks: list[int],
    rebalances: list[int],
    risk_budgets: list[float],
    industry_caps: list[float],
    buffer_multiples: list[int],
    initial_cash: float,
    raw_daily_dir: Path,
    price_start: str,
    price_end: str,
    allocation_mode: str = "fixed_topk",
    outer_path: Path | None = None,
    outer_risk_threshold: float | None = None,
    outer_risk_floor: float | None = None,
    forbidden_path: Path | None = DEFAULT_FORBIDDEN,
) -> dict:
    if allocation_mode not in ALLOCATION_MODES:
        raise ValueError(f"unsupported allocation mode: {allocation_mode}")
    predictions = load_middle_predictions(middle_path)
    forbidden = load_forbidden_instruments(forbidden_path)
    forbidden_prediction_rows = 0
    if forbidden:
        before = len(predictions)
        predictions = predictions[
            ~predictions["instrument"].astype(str).str.upper().isin(forbidden)
        ].copy()
        forbidden_prediction_rows = before - len(predictions)
        print(
            f"[expanded controls] forbidden instruments={len(forbidden)} "
            f"removed_prediction_rows={forbidden_prediction_rows}",
            flush=True,
        )
    if outer_path is not None:
        outer = pd.read_csv(outer_path, parse_dates=["datetime"])
        if "outer" not in outer.columns and "pred_outer" in outer.columns:
            outer = outer.rename(columns={"pred_outer": "outer"})
        if "outer" not in outer.columns:
            raise ValueError(f"{outer_path} must contain outer or pred_outer")
        outer["instrument"] = outer["instrument"].astype(str).str.upper()
        predictions = predictions.drop(
            columns=["outer", "pred_outer", "outer_risk_probability"],
            errors="ignore",
        )
        predictions = predictions.merge(
            outer[["datetime", "instrument", "outer"]],
            on=["datetime", "instrument"],
            how="left",
        )
        daily_outer = predictions.groupby("datetime")["outer"].median()
        predictions = predictions.merge(
            daily_outer.rename("outer_risk_probability"),
            left_on="datetime",
            right_index=True,
            how="left",
        )
    predictions = predictions[predictions["datetime"].between(start_signal, end_signal)].copy()
    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(raw_daily_dir, instruments, start=price_start, end=price_end)
    minute_windows = pd.read_csv(windows_path, parse_dates=["trade_date"])
    minute_windows["instrument"] = minute_windows["instrument"].astype(str).str.upper()
    minute_pool = set(minute_windows["instrument"].drop_duplicates())
    group_metadata = load_group_metadata(group_metadata_path, "industry")
    folds = json.loads(folds_path.read_text(encoding="utf-8"))

    rows = []
    for top_k in top_ks:
        for rebalance in rebalances:
            for risk_budget in risk_budgets:
                for industry_cap in industry_caps:
                    for buffer_multiple in buffer_multiples:
                        profile = MappedProfile(
                            name=f"c_top{top_k}_rb{rebalance}_risk{risk_budget:.2f}_cap{industry_cap:.2f}",
                            top_k=top_k,
                            rebalance_every=rebalance,
                            risk_budget=risk_budget,
                            min_avg_amount=100_000_000.0,
                            industry_cap=industry_cap,
                        )
                        cfg = MappedReplayConfig(
                            initial_cash=initial_cash,
                            buffer_multiple=buffer_multiple,
                        )
                        frame = prepare_daily_frame(predictions, prices, profile, cfg)
                        for cost in (BASE_COST, STRESS_COST):
                            row = evaluate_candidate(
                                frame,
                                minute_windows,
                                minute_pool,
                                group_metadata,
                                folds,
                                profile,
                                cfg,
                                cost,
                                start_signal,
                                end_signal,
                                allocation_mode,
                                outer_risk_threshold,
                                outer_risk_floor,
                            )
                            row["config"] = asdict(cfg)
                            row["allocation_mode"] = allocation_mode
                            row["outer_risk_threshold"] = outer_risk_threshold
                            row["outer_risk_floor"] = outer_risk_floor
                            rows.append(row)
                            metrics = row["metrics"]
                            fold_metrics = row["fold_summary"]
                            print(
                                f"[expanded controls] cost={cost.name} top={top_k} "
                                f"rb={rebalance} risk={risk_budget:.2f} cap={industry_cap:.0%} "
                                f"buf={buffer_multiple} ret={metrics['total_return']:+.2%} "
                                f"worst={fold_metrics['worst_fold_return']:+.2%} "
                                f"mdd={metrics['max_drawdown']:.2%} turn={metrics['turnover']:.2f} "
                                f"score={row['control_score']:.4f}",
                                flush=True,
                            )

    report = {
        "status": "expanded_c_control_grid_research_only",
        "middle_prediction": str(middle_path),
        "outer_prediction": str(outer_path) if outer_path else None,
        "outer_risk_threshold": outer_risk_threshold,
        "outer_risk_floor": outer_risk_floor,
        "forbidden_path": str(forbidden_path) if forbidden_path else None,
        "forbidden_instruments": len(forbidden),
        "forbidden_prediction_rows": forbidden_prediction_rows,
        "minute_windows": str(windows_path),
        "folds_path": str(folds_path),
        "group_metadata": str(group_metadata_path),
        "raw_daily_dir": str(raw_daily_dir),
        "price_start": price_start,
        "price_end": price_end,
        "window": {"start_signal": start_signal, "end_signal": end_signal},
        "grid": {
            "top_k": top_ks,
            "rebalance_every": rebalances,
            "risk_budget": risk_budgets,
            "industry_cap": industry_caps,
            "buffer_multiple": buffer_multiples,
            "initial_cash": initial_cash,
            "allocation_mode": allocation_mode,
            "costs": [BASE_COST.name, STRESS_COST.name],
        },
        "score_policy": (
            "1.5*worst_fold + median_fold + 0.25*total_return - "
            "0.8*max_drawdown - 0.015*turnover; stress-cost rows are preferred "
            "for final decisions."
        ),
        "candidates": sorted(rows, key=lambda row: row["control_score"], reverse=True),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(rows, summary_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--middle", default=str(DEFAULT_MIDDLE))
    parser.add_argument("--minute-windows", default=str(DEFAULT_WINDOWS))
    parser.add_argument("--folds", default=str(HERE / "outputs" / "purged_folds.json"))
    parser.add_argument("--group-metadata", default=str(DEFAULT_GROUP_METADATA))
    parser.add_argument("--output", default=str(HERE / "outputs" / "expanded_c_control_grid.json"))
    parser.add_argument("--summary", default=str(HERE / "outputs" / "expanded_c_control_grid_summary.csv"))
    parser.add_argument("--start-signal", default="2025-01-03")
    parser.add_argument("--end-signal", default="2026-04-02")
    parser.add_argument("--top-k", default="100,150")
    parser.add_argument("--rebalance", default="30,45,60")
    parser.add_argument("--risk-budget", default="0.6,0.8,1.0")
    parser.add_argument("--industry-cap", default="0.3,0.4")
    parser.add_argument("--buffer-multiple", default="2")
    parser.add_argument("--initial-cash", type=float, default=500_000.0)
    parser.add_argument("--raw-daily-dir", default=str(DATA / "A_Stock_daily_qfq"))
    parser.add_argument("--price-start", default="2022-01-04")
    parser.add_argument("--price-end", default="2026-04-28")
    parser.add_argument(
        "--allocation-mode",
        choices=sorted(ALLOCATION_MODES),
        default="fixed_topk",
    )
    parser.add_argument("--outer-predictions", default="")
    parser.add_argument("--outer-risk-threshold", type=float, default=None)
    parser.add_argument("--outer-risk-floor", type=float, default=None)
    parser.add_argument("--forbidden-symbols", default=str(DEFAULT_FORBIDDEN))
    args = parser.parse_args()
    report = run(
        Path(args.middle).resolve(),
        Path(args.minute_windows).resolve(),
        Path(args.folds).resolve(),
        Path(args.group_metadata).resolve(),
        Path(args.output).resolve(),
        Path(args.summary).resolve(),
        args.start_signal,
        args.end_signal,
        parse_csv_numbers(args.top_k, int),
        parse_csv_numbers(args.rebalance, int),
        parse_csv_numbers(args.risk_budget, float),
        parse_csv_numbers(args.industry_cap, float),
        parse_csv_numbers(args.buffer_multiple, int),
        args.initial_cash,
        Path(args.raw_daily_dir).resolve(),
        args.price_start,
        args.price_end,
        args.allocation_mode,
        Path(args.outer_predictions).resolve() if args.outer_predictions else None,
        args.outer_risk_threshold,
        args.outer_risk_floor,
        Path(args.forbidden_symbols).resolve() if args.forbidden_symbols else None,
    )
    print("[expanded controls] best candidates:")
    for row in report["candidates"][:12]:
        profile = row["profile"]
        metrics = row["metrics"]
        folds = row["fold_summary"]
        print(
            f"  {row['cost']['name']} top={profile['top_k']} "
            f"rb={profile['rebalance_every']} risk={profile['risk_budget']:.2f} "
            f"cap={profile['industry_cap']:.0%} ret={metrics['total_return']:+.2%} "
            f"worst={folds['worst_fold_return']:+.2%} "
            f"mdd={metrics['max_drawdown']:.2%} turn={metrics['turnover']:.2f} "
            f"score={row['control_score']:.4f}"
        )


if __name__ == "__main__":
    main()
