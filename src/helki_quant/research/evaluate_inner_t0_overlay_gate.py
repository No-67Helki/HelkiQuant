from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_inner_oof import load_fold_frame


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"


def read_calendar(path: Path) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.read_csv(path, header=None, names=["date"], parse_dates=["date"])["date"]
    )


def next_trade_date_map(calendar: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp]:
    calendar = calendar.drop_duplicates().sort_values()
    return dict(zip(calendar[:-1], calendar[1:]))


def load_oof_frames(config_dir: Path, prediction_dir: Path, variant: str) -> pd.DataFrame:
    frames = []
    for fold in range(1, 7):
        fold_dir = config_dir / f"fold_{fold:02d}"
        config_path = fold_dir / ("simple.yaml" if variant.endswith("_simple") else "de2_srfs_es.yaml")
        prediction_path = prediction_dir / f"fold_{fold:02d}.csv"
        frame = load_fold_frame(config_path, prediction_path).reset_index()
        frame["fold"] = fold
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    return out


def threshold_rows(frame: pd.DataFrame, thresholds: list[float], quantiles: list[float]) -> list[tuple[str, pd.Series]]:
    rows: list[tuple[str, pd.Series]] = []
    for threshold in thresholds:
        rows.append((f"score_ge_{threshold:.2f}", frame["score"] >= threshold))
    for quantile in quantiles:
        cutoff = float(frame["score"].quantile(quantile))
        rows.append((f"score_q{quantile:.2f}_cut_{cutoff:.6f}", frame["score"] >= cutoff))
    return rows


def summarize_selection(name: str, sub: pd.DataFrame, base_rows: int) -> dict:
    if sub.empty:
        return {
            "gate": name,
            "rows": 0,
            "coverage_ratio": 0.0,
            "days": 0,
            "instruments": 0,
        }
    daily = sub.groupby("trade_date")["label"].mean()
    return {
        "gate": name,
        "rows": int(len(sub)),
        "coverage_ratio": float(len(sub) / max(base_rows, 1)),
        "days": int(sub["trade_date"].nunique()),
        "instruments": int(sub["instrument"].nunique()),
        "score_mean": float(sub["score"].mean()),
        "label_mean": float(sub["label"].mean()),
        "label_median": float(sub["label"].median()),
        "label_positive_ratio_0": float((sub["label"] > 0).mean()),
        "label_positive_ratio_1pct": float((sub["label"] > 0.01).mean()),
        "label_q05": float(sub["label"].quantile(0.05)),
        "label_q25": float(sub["label"].quantile(0.25)),
        "label_q75": float(sub["label"].quantile(0.75)),
        "label_q95": float(sub["label"].quantile(0.95)),
        "daily_label_mean": float(daily.mean()),
        "daily_positive_ratio": float((daily > 0).mean()),
        "daily_above_1pct_ratio": float((daily > 0.01).mean()),
    }


def evaluate(
    config_dir: Path,
    prediction_dir: Path,
    variant: str,
    holdings_path: Path,
    calendar_path: Path,
    output_path: Path,
) -> dict:
    oof = load_oof_frames(config_dir, prediction_dir, variant)
    calendar = read_calendar(calendar_path)
    next_map = next_trade_date_map(calendar)
    oof["signal_date"] = oof["datetime"]
    oof["trade_date"] = oof["signal_date"].map(next_map)
    oof = oof.dropna(subset=["trade_date"]).copy()
    oof["trade_date"] = pd.to_datetime(oof["trade_date"]).dt.normalize()

    holdings = pd.read_csv(holdings_path, parse_dates=["trade_date"])
    holdings["trade_date"] = holdings["trade_date"].dt.normalize()
    holdings["instrument"] = holdings["instrument"].astype(str).str.upper()
    holdings = holdings[holdings["shares"] > 0][["trade_date", "instrument", "shares", "weight"]]
    holdings = holdings.rename(columns={"trade_date": "signal_date", "shares": "held_shares", "weight": "held_weight"})

    frame = oof.merge(holdings, on=["signal_date", "instrument"], how="inner")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "label"])
    gate_rows = [
        summarize_selection(name, frame.loc[mask].copy(), len(frame))
        for name, mask in threshold_rows(frame, [0.55, 0.60, 0.65, 0.70], [0.70, 0.80, 0.90])
    ]
    gate_rows = sorted(gate_rows, key=lambda row: row.get("label_mean", -999), reverse=True)
    result = {
        "status": "inner_t0_overlay_gate_evaluated",
        "variant": variant,
        "config_dir": str(config_dir.resolve()),
        "prediction_dir": str(prediction_dir.resolve()),
        "holdings_path": str(holdings_path.resolve()),
        "calendar_path": str(calendar_path.resolve()),
        "base_rows": int(len(frame)),
        "base_days": int(frame["trade_date"].nunique()) if len(frame) else 0,
        "base_instruments": int(frame["instrument"].nunique()) if len(frame) else 0,
        "base_label_mean": float(frame["label"].mean()) if len(frame) else None,
        "base_label_positive_ratio_1pct": float((frame["label"] > 0.01).mean()) if len(frame) else None,
        "best_gate": gate_rows[0] if gate_rows else None,
        "gates": gate_rows,
        "decision": (
            "candidate_for_portfolio_t0_replay"
            if gate_rows
            and gate_rows[0].get("rows", 0) >= 100
            and gate_rows[0].get("label_mean", 0.0) > (result_base := float(frame["label"].mean()) if len(frame) else 0.0)
            else "keep_inner_gate_research_only"
        ),
        "decision_rule": (
            "This is a signal gate, not portfolio PnL. It requires at least 100 held-stock "
            "opportunities and selected mean label above the held-stock base mean before "
            "building a portfolio-level reverse-T replay."
        ),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--holdings", required=True)
    parser.add_argument(
        "--calendar",
        default=str(DATA / "cn_data_pool_inner_canonical_20260605" / "calendars" / "day.txt"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(
        Path(args.config_dir).resolve(),
        Path(args.prediction_dir).resolve(),
        args.variant,
        Path(args.holdings).resolve(),
        Path(args.calendar).resolve(),
        Path(args.output).resolve(),
    )
    best = report.get("best_gate") or {}
    print(
        "[inner t0 gate] "
        f"decision={report['decision']} base_rows={report['base_rows']} "
        f"base_mean={report['base_label_mean']} best={best.get('gate')} "
        f"rows={best.get('rows')} mean={best.get('label_mean')}"
    )


if __name__ == "__main__":
    main()
