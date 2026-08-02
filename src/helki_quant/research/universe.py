from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_sources.rqdata_source import read_price_csv
except ImportError:
    from data_sources.rqdata_source import read_price_csv


@dataclass(frozen=True)
class UniverseRules:
    board_prefixes: tuple[str, ...] = ("30", "68")
    min_listing_days: int = 120
    liquidity_window: int = 20
    min_avg_amount: float = 50_000_000.0
    suspend_window: int = 20
    max_suspend_ratio: float = 0.10


def instrument_to_code(instrument: str) -> str:
    return str(instrument).upper().replace(".", "")[-6:]


def load_price_panel(
    raw_dir: Path,
    instruments: list[str],
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for pos, instrument in enumerate(sorted(set(instruments)), start=1):
        code = instrument_to_code(instrument)
        path = raw_dir / f"{code}_daily_qfq.csv"
        if not path.exists():
            continue
        frame = read_price_csv(path, frequency="1d").rename(
            columns={"date": "datetime"}
        )
        frame = frame[frame["datetime"].between(start_ts, end_ts)].copy()
        if frame.empty:
            continue
        frame["instrument"] = instrument
        rows.append(frame)
        if pos % 250 == 0:
            print(f"[prices] {pos}/{len(instruments)}", flush=True)
    if not rows:
        raise ValueError("No daily price data loaded")
    return pd.concat(rows, ignore_index=True).sort_values(["instrument", "datetime"])


def add_point_in_time_eligibility(
    panel: pd.DataFrame,
    rules: UniverseRules,
) -> pd.DataFrame:
    """Compute eligibility using only information known by each day's close."""
    out = panel.copy().sort_values(["instrument", "datetime"])
    code = out["instrument"].map(instrument_to_code)
    out["board_ok"] = code.str.startswith(rules.board_prefixes)
    grouped = out.groupby("instrument", sort=False)
    out["listing_days"] = grouped.cumcount() + 1
    out["avg_amount"] = grouped["amount"].transform(
        lambda s: s.rolling(rules.liquidity_window, min_periods=rules.liquidity_window).mean()
    )
    out["suspend_ratio"] = grouped["volume"].transform(
        lambda s: s.eq(0).rolling(rules.suspend_window, min_periods=rules.suspend_window).mean()
    )
    finite_prices = np.isfinite(out["open"]) & np.isfinite(out["close"])
    out["eligible"] = (
        out["board_ok"]
        & (out["listing_days"] >= rules.min_listing_days)
        & (out["avg_amount"] >= rules.min_avg_amount)
        & (out["suspend_ratio"] <= rules.max_suspend_ratio)
        & finite_prices
        & (out["open"] > 0)
        & (out["close"] > 0)
    )
    return out
