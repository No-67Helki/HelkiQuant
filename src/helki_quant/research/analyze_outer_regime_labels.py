from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .portfolio_experiments import load_predictions
    from .universe import (
        UniverseRules,
        add_point_in_time_eligibility,
        load_price_panel,
    )
except ImportError:
    from portfolio_experiments import load_predictions
    from universe import (
        UniverseRules,
        add_point_in_time_eligibility,
        load_price_panel,
    )


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_RAW_DAILY = REPO_ROOT / "data" / "A_Stock_daily_qfq" / "daily_qfq_6.5"


def max_drawdown(path: np.ndarray) -> float:
    if len(path) == 0:
        return np.nan
    nav = 1.0 + np.asarray(path, dtype=float)
    peak = np.maximum.accumulate(nav)
    return float(np.nanmax(1.0 - nav / peak))


def downside_vol(path: np.ndarray) -> float:
    if len(path) < 2:
        return np.nan
    returns = pd.Series(1.0 + np.asarray(path, dtype=float)).pct_change().dropna()
    downside = returns[returns < 0]
    return float(downside.std()) if len(downside) else 0.0


def add_forward_paths(prices: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = prices.copy().sort_values(["instrument", "datetime"])
    grouped = out.groupby("instrument", sort=False)["close"]
    entry = grouped.shift(-1)
    max_horizon = max(horizons)
    for step in range(1, max_horizon + 1):
        out[f"cumret_{step}d"] = grouped.shift(-(step + 1)) / entry - 1.0
    return out


def path_outcomes(part: pd.DataFrame, prefix: str, horizons: list[int]) -> dict:
    result = {}
    for horizon in horizons:
        cols = [f"cumret_{step}d" for step in range(1, horizon + 1)]
        path = part[cols].mean(axis=0, skipna=True).to_numpy(dtype=float)
        result[f"{prefix}_fwd_{horizon}d"] = float(path[-1]) if len(path) else np.nan
        result[f"{prefix}_min_{horizon}d"] = float(np.nanmin(path)) if len(path) else np.nan
        result[f"{prefix}_mdd_{horizon}d"] = max_drawdown(path)
        result[f"{prefix}_downside_vol_{horizon}d"] = downside_vol(path)
    return result


def rolling_z(series: pd.Series, lookback: int) -> pd.Series:
    mean = series.rolling(lookback, min_periods=max(10, lookback // 2)).mean()
    std = series.rolling(lookback, min_periods=max(10, lookback // 2)).std()
    return ((series - mean) / std.replace(0.0, np.nan)).astype(float)


def correlations(frame: pd.DataFrame, signals: list[str], outcomes: list[str]) -> dict:
    return {
        signal: {
            outcome: float(frame[signal].corr(frame[outcome], method="spearman"))
            for outcome in outcomes
            if frame[signal].notna().sum() and frame[outcome].notna().sum()
        }
        for signal in signals
    }


def quantile_summary(frame: pd.DataFrame, signal: str, outcomes: list[str]) -> list[dict]:
    sample = frame.dropna(subset=[signal, *outcomes]).copy()
    if sample.empty:
        return []
    sample["quintile"] = pd.qcut(sample[signal].rank(method="first"), 5, labels=False)
    rows = []
    for quintile, part in sample.groupby("quintile", sort=True):
        rows.append(
            {
                "quintile": int(quintile),
                "days": int(len(part)),
                "signal_mean": float(part[signal].mean()),
                **{f"{outcome}_mean": float(part[outcome].mean()) for outcome in outcomes},
            }
        )
    return rows


def evaluate(
    artifacts_dir: Path,
    raw_daily_dir: Path,
    output_path: Path,
    start: str,
    end: str,
    top_k: int,
    horizons: list[int],
) -> dict:
    predictions = load_predictions(artifacts_dir)
    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(raw_daily_dir, instruments, start=start, end=end)
    prices = add_forward_paths(prices, horizons)
    eligible = add_point_in_time_eligibility(
        prices,
        UniverseRules(min_listing_days=250, min_avg_amount=100_000_000.0),
    )
    sample = predictions.merge(
        eligible,
        on=["datetime", "instrument"],
        how="inner",
    )
    sample = sample[sample["eligible"].fillna(False)].copy()
    daily_rows = []
    for date, day in sample.groupby("datetime", sort=True):
        broad = day
        top = day.sort_values("middle", ascending=False).head(top_k)
        if broad.empty or top.empty:
            continue
        row = {
            "datetime": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "eligible_count": int(len(broad)),
            "top_k": int(len(top)),
            "outer_median": float(broad["outer"].median()),
            "outer_top_median": float(top["outer"].median()),
            "outer_p20": float(broad["outer"].quantile(0.20)),
            "outer_p80": float(broad["outer"].quantile(0.80)),
        }
        row.update(path_outcomes(broad, "broad", horizons))
        row.update(path_outcomes(top, f"top{top_k}", horizons))
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows).sort_values("datetime")
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    daily["outer_z_20"] = rolling_z(daily["outer_median"], 20)
    daily["outer_z_40"] = rolling_z(daily["outer_median"], 40)
    daily["outer_z_60"] = rolling_z(daily["outer_median"], 60)
    signals = ["outer_median", "outer_top_median", "outer_z_20", "outer_z_40", "outer_z_60"]
    outcomes = []
    for horizon in horizons:
        outcomes.extend(
            [
                f"broad_fwd_{horizon}d",
                f"broad_mdd_{horizon}d",
                f"broad_downside_vol_{horizon}d",
                f"top{top_k}_fwd_{horizon}d",
                f"top{top_k}_mdd_{horizon}d",
                f"top{top_k}_downside_vol_{horizon}d",
            ]
        )
    adverse = {}
    for horizon in horizons:
        mdd_col = f"top{top_k}_mdd_{horizon}d"
        fwd_col = f"top{top_k}_fwd_{horizon}d"
        adverse[f"top{top_k}_adverse_{horizon}d"] = {
            "mdd_gt_3pct": float((daily[mdd_col] > 0.03).mean()),
            "mdd_gt_5pct": float((daily[mdd_col] > 0.05).mean()),
            "fwd_lt_minus_3pct": float((daily[fwd_col] < -0.03).mean()),
            "fwd_lt_minus_5pct": float((daily[fwd_col] < -0.05).mean()),
        }
    report = {
        "status": "outer_regime_label_diagnostic_research_only",
        "artifacts_dir": str(artifacts_dir.resolve()),
        "raw_daily_dir": str(raw_daily_dir.resolve()),
        "date_start": str(daily["datetime"].min().date()) if len(daily) else None,
        "date_end": str(daily["datetime"].max().date()) if len(daily) else None,
        "days": int(len(daily)),
        "top_k": top_k,
        "horizons": horizons,
        "spearman_correlation": correlations(daily, signals, outcomes),
        "adverse_base_rates": adverse,
        "quantiles": {
            signal: quantile_summary(daily, signal, outcomes)
            for signal in signals
        },
        "interpretation": (
            "Use this to choose a redesigned outer risk/regime label. It is "
            "diagnostic only and does not promote any outer overlay to production."
        ),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    daily_path = output_path.with_suffix(".daily.csv")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    report["daily_csv"] = str(daily_path.resolve())
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_horizons(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--raw-daily-dir", default=str(DEFAULT_RAW_DAILY))
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", default="2022-01-04")
    parser.add_argument("--end", default="2026-06-05")
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--horizons", default="5,10,20")
    args = parser.parse_args()
    report = evaluate(
        Path(args.artifacts_dir).resolve(),
        Path(args.raw_daily_dir).resolve(),
        Path(args.output).resolve(),
        args.start,
        args.end,
        args.top_k,
        parse_horizons(args.horizons),
    )
    print(
        f"[outer regime labels] days={report['days']} "
        f"daily={report['daily_csv']}"
    )
    for signal, rows in report["spearman_correlation"].items():
        top20_mdd = rows.get(f"top{args.top_k}_mdd_20d")
        top20_fwd = rows.get(f"top{args.top_k}_fwd_20d")
        print(
            f"  {signal}: top{args.top_k}_fwd20={top20_fwd:+.4f} "
            f"top{args.top_k}_mdd20={top20_mdd:+.4f}"
        )


if __name__ == "__main__":
    main()
