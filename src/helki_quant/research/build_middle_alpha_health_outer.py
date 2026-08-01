from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_daily_topk_grid import load_middle_predictions
from evaluate_expanded_c_controls import DEFAULT_FORBIDDEN, load_forbidden_instruments
from universe import UniverseRules, add_point_in_time_eligibility, load_price_panel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def artifact(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"alpha-health artifact not found: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def load_outer_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime"])
    if "outer" not in frame.columns and "pred_outer" in frame.columns:
        frame = frame.rename(columns={"pred_outer": "outer"})
    if "outer" not in frame.columns:
        raise ValueError(f"{path} must contain outer or pred_outer")
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    frame["outer"] = pd.to_numeric(frame["outer"], errors="coerce")
    spread = frame.groupby("datetime")["outer"].agg(
        lambda values: float(values.max() - values.min())
    )
    if spread.empty or not np.isfinite(spread.max()) or spread.max() > 1e-10:
        raise ValueError(f"outer predictions are not daily-constant: max spread={spread.max()}")
    return (
        frame.groupby("datetime", as_index=False)["outer"]
        .median()
        .rename(columns={"outer": "broad_outer"})
        .sort_values("datetime")
    )


def build_label_calendar(
    market_dates: pd.DatetimeIndex,
    signal_dates: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    positions = market_dates.get_indexer(signal_dates)
    if (positions < 0).any():
        missing = signal_dates[positions < 0]
        raise ValueError(f"signal dates missing from market calendar: {missing[:5].tolist()}")
    entry_positions = positions + 1
    exit_positions = positions + horizon + 1
    valid = exit_positions < len(market_dates)
    return pd.DataFrame(
        {
            "datetime": signal_dates[valid],
            "entry_date": market_dates[entry_positions[valid]],
            "available_date": market_dates[exit_positions[valid]],
        }
    )


def compute_daily_ic(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
    min_cross_section: int,
) -> pd.DataFrame:
    market_dates = pd.DatetimeIndex(prices["datetime"].drop_duplicates()).sort_values()
    signal_dates = pd.DatetimeIndex(predictions["datetime"].drop_duplicates()).sort_values()
    label_calendar = build_label_calendar(market_dates, signal_dates, horizon)
    entry = prices[["datetime", "instrument", "close"]].rename(
        columns={"datetime": "entry_date", "close": "entry_close"}
    )
    exit_prices = prices[["datetime", "instrument", "close"]].rename(
        columns={"datetime": "available_date", "close": "exit_close"}
    )
    eligible = add_point_in_time_eligibility(prices, UniverseRules())[
        ["datetime", "instrument", "eligible"]
    ]
    sample = (
        predictions.merge(label_calendar, on="datetime", how="inner")
        .merge(entry, on=["entry_date", "instrument"], how="left")
        .merge(exit_prices, on=["available_date", "instrument"], how="left")
        .merge(eligible, on=["datetime", "instrument"], how="left")
    )
    sample["forward_return"] = sample["exit_close"] / sample["entry_close"] - 1.0
    sample = sample[
        sample["eligible"].astype("boolean").fillna(False)
        & sample["middle"].notna()
        & sample["forward_return"].notna()
    ].copy()

    rows = []
    for signal_date, day in sample.groupby("datetime", sort=True):
        if len(day) < min_cross_section:
            continue
        ic = day["middle"].corr(day["forward_return"], method="spearman")
        if not np.isfinite(ic):
            continue
        available_dates = day["available_date"].drop_duplicates()
        if len(available_dates) != 1:
            raise ValueError(f"inconsistent availability date for {signal_date}")
        rows.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "available_date": pd.Timestamp(available_dates.iloc[0]),
                "ic": float(ic),
                "cross_section": int(len(day)),
            }
        )
    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def health_snapshot(
    daily_ic: pd.DataFrame,
    decision_date: pd.Timestamp,
    rolling_window: int,
    min_observations: int,
    health_threshold: float,
) -> tuple[pd.DataFrame, float, bool]:
    known = daily_ic[daily_ic["available_date"] <= decision_date].tail(rolling_window)
    health = float(known["ic"].mean()) if len(known) else np.nan
    trigger = bool(
        len(known) >= min_observations
        and np.isfinite(health)
        and health < health_threshold
    )
    return known, health, trigger


def build(
    middle_path: Path,
    broad_outer_path: Path,
    raw_daily_dir: Path,
    forbidden_path: Path | None,
    output_path: Path,
    daily_output_path: Path,
    manifest_path: Path,
    price_start: str,
    price_end: str,
    horizon: int,
    rolling_window: int,
    min_observations: int,
    min_cross_section: int,
    health_threshold: float,
    trigger_value: float,
) -> dict:
    if horizon < 1 or rolling_window < 1 or min_observations < 1:
        raise ValueError("horizon, rolling_window, and min_observations must be positive")
    if min_observations > rolling_window:
        raise ValueError("min_observations cannot exceed rolling_window")
    middle_all = load_middle_predictions(middle_path)[
        ["datetime", "instrument", "middle"]
    ].copy()
    middle_all["datetime"] = pd.to_datetime(middle_all["datetime"]).dt.normalize()
    middle_all["instrument"] = middle_all["instrument"].astype(str).str.upper()
    if middle_all.duplicated(["datetime", "instrument"]).any():
        raise ValueError("middle predictions contain duplicate datetime/instrument keys")

    forbidden = load_forbidden_instruments(forbidden_path)
    health_predictions = middle_all[
        ~middle_all["instrument"].isin(forbidden)
    ].copy()
    instruments = health_predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(
        raw_daily_dir,
        instruments,
        start=price_start,
        end=price_end,
    ).sort_values(["datetime", "instrument"])
    daily_ic = compute_daily_ic(
        health_predictions,
        prices,
        horizon,
        min_cross_section,
    )
    broad_daily = load_outer_daily(broad_outer_path)
    decision_dates = pd.DatetimeIndex(middle_all["datetime"].drop_duplicates()).sort_values()

    health_rows = []
    for decision_date in decision_dates:
        known, health, trigger = health_snapshot(
            daily_ic,
            pd.Timestamp(decision_date),
            rolling_window,
            min_observations,
            health_threshold,
        )
        health_rows.append(
            {
                "datetime": pd.Timestamp(decision_date),
                "known_ic_observations": int(len(known)),
                "oldest_known_signal_date": (
                    str(pd.Timestamp(known.iloc[0]["signal_date"]).date()) if len(known) else None
                ),
                "latest_known_signal_date": (
                    str(pd.Timestamp(known.iloc[-1]["signal_date"]).date()) if len(known) else None
                ),
                "latest_available_date": (
                    str(pd.Timestamp(known["available_date"].max()).date()) if len(known) else None
                ),
                "alpha_health_ic_mean": health,
                "alpha_health_trigger": int(trigger),
            }
        )
    health_daily = pd.DataFrame(health_rows)
    daily = health_daily.merge(broad_daily, on="datetime", how="left", validate="one_to_one")
    if daily["broad_outer"].isna().any():
        missing = daily.loc[daily["broad_outer"].isna(), "datetime"].head(5).tolist()
        raise ValueError(f"broad outer missing decision dates: {missing}")
    daily["combined_outer"] = np.where(
        daily["alpha_health_trigger"].eq(1),
        np.maximum(daily["broad_outer"], trigger_value),
        daily["broad_outer"],
    )

    output = middle_all[["datetime", "instrument"]].merge(
        daily[["datetime", "combined_outer"]],
        on="datetime",
        how="left",
        validate="many_to_one",
    )
    output = output.rename(columns={"combined_outer": "outer"})
    if output["outer"].isna().any():
        raise ValueError("combined outer contains missing values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    daily_output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_output_path, index=False)
    report = {
        "schema_version": 2,
        "status": "causal_middle_alpha_health_outer_research_only",
        "middle": str(middle_path.resolve()),
        "broad_outer": str(broad_outer_path.resolve()),
        "raw_daily_dir": str(raw_daily_dir.resolve()),
        "forbidden_path": str(forbidden_path.resolve()) if forbidden_path else None,
        "forbidden_instruments": len(forbidden),
        "price_window": {"start": price_start, "end": price_end},
        "policy": {
            "label_horizon_trading_days": horizon,
            "label_entry": "next market-session close",
            "label_exit": f"market-session close at signal+{horizon + 1}",
            "label_known_on": "exit close (available_date)",
            "rolling_ic_observations": rolling_window,
            "min_observations": min_observations,
            "min_cross_section": min_cross_section,
            "health_threshold": health_threshold,
            "trigger_value": trigger_value,
            "causal_filter": "available_date <= decision_date",
        },
        "daily_ic_rows": int(len(daily_ic)),
        "decision_dates": int(len(daily)),
        "trigger_dates": int(daily["alpha_health_trigger"].sum()),
        "trigger_ratio": float(daily["alpha_health_trigger"].mean()),
        "first_trigger_date": (
            str(daily.loc[daily["alpha_health_trigger"].eq(1), "datetime"].min().date())
            if daily["alpha_health_trigger"].any()
            else None
        ),
        "output": str(output_path.resolve()),
        "daily_output": str(daily_output_path.resolve()),
        "source_provenance": {
            "generator": artifact(Path(__file__)),
            "middle_prediction": artifact(middle_path),
            "broad_outer_prediction": artifact(broad_outer_path),
            "forbidden_symbols": artifact(forbidden_path),
            "combined_outer": artifact(output_path),
            "daily_health": artifact(daily_output_path),
        },
        "deployment_allowed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--middle", required=True)
    parser.add_argument("--broad-outer", required=True)
    parser.add_argument("--raw-daily-dir", required=True)
    parser.add_argument("--forbidden-symbols", default=str(DEFAULT_FORBIDDEN))
    parser.add_argument("--output", required=True)
    parser.add_argument("--daily-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--price-start", default="2025-01-03")
    parser.add_argument("--price-end", default="2026-06-05")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--min-observations", type=int, default=10)
    parser.add_argument("--min-cross-section", type=int, default=100)
    parser.add_argument("--health-threshold", type=float, default=0.0)
    parser.add_argument("--trigger-value", type=float, default=1.0)
    args = parser.parse_args()
    report = build(
        Path(args.middle).resolve(),
        Path(args.broad_outer).resolve(),
        Path(args.raw_daily_dir).resolve(),
        Path(args.forbidden_symbols).resolve() if args.forbidden_symbols else None,
        Path(args.output).resolve(),
        Path(args.daily_output).resolve(),
        Path(args.manifest).resolve(),
        args.price_start,
        args.price_end,
        args.horizon,
        args.rolling_window,
        args.min_observations,
        args.min_cross_section,
        args.health_threshold,
        args.trigger_value,
    )
    print(
        f"[middle alpha health outer] dates={report['decision_dates']} "
        f"trigger={report['trigger_dates']} ({report['trigger_ratio']:.1%}) "
        f"first={report['first_trigger_date']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
