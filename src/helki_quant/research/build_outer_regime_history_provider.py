from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .analyze_outer_regime_labels import add_forward_paths, path_outcomes
    from .augment_outer_regime_labels import (
        read_calendar,
        read_instruments,
        write_qlib_bin,
    )
    from .universe import (
        UniverseRules,
        add_point_in_time_eligibility,
        load_price_panel,
    )
except ImportError:
    from analyze_outer_regime_labels import add_forward_paths, path_outcomes
    from augment_outer_regime_labels import (
        read_calendar,
        read_instruments,
        write_qlib_bin,
    )
    from universe import (
        UniverseRules,
        add_point_in_time_eligibility,
        load_price_panel,
    )


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_SOURCE_PROVIDER = DATA / "cn_data_canonical_pit_20260605"
DEFAULT_RAW_DAILY = DATA / "A_Stock_daily_qfq" / "daily_qfq_6.5"


def parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons:
        raise ValueError("at least one horizon is required")
    return sorted(set(horizons))


def copy_provider(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing provider: {output}")
    shutil.copytree(source, output)


def make_label_specs(horizons: list[int]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for horizon in horizons:
        prefix = f"OUTER_BROAD_{horizon}D"
        specs[f"{prefix}_FWD"] = f"broad_fwd_{horizon}d"
        specs[f"{prefix}_MDD"] = f"broad_mdd_{horizon}d"
        specs[f"{prefix}_DOWNSIDE_VOL"] = f"broad_downside_vol_{horizon}d"
        specs[f"{prefix}_ADVERSE_MDD5"] = f"broad_adverse_mdd5_{horizon}d"
        specs[f"{prefix}_ADVERSE_LOSS5"] = f"broad_adverse_loss5_{horizon}d"
        specs[f"{prefix}_ADVERSE_COMBO"] = f"broad_adverse_combo_{horizon}d"
    return specs


def make_feature_specs() -> dict[str, str]:
    return {
        "OUTER_REGIME_RET_1D": "regime_ret_1d",
        "OUTER_REGIME_RET_5D": "regime_ret_5d",
        "OUTER_REGIME_RET_10D": "regime_ret_10d",
        "OUTER_REGIME_RET_20D": "regime_ret_20d",
        "OUTER_REGIME_VOL_5D": "regime_vol_5d",
        "OUTER_REGIME_VOL_20D": "regime_vol_20d",
        "OUTER_REGIME_DOWNSIDE_VOL_20D": "regime_downside_vol_20d",
        "OUTER_REGIME_MDD_20D": "regime_mdd_20d",
        "OUTER_REGIME_BREADTH_UP_1D": "regime_breadth_up_1d",
        "OUTER_REGIME_BREADTH_UP_5D": "regime_breadth_up_5d",
        "OUTER_REGIME_DISPERSION_1D": "regime_dispersion_1d",
        "OUTER_REGIME_DISPERSION_20D": "regime_dispersion_20d",
        "OUTER_REGIME_AVG_AMOUNT_20D": "regime_avg_amount_20d",
        "OUTER_REGIME_AMOUNT_CHG_20D": "regime_amount_chg_20d",
        "OUTER_REGIME_ELIGIBLE_COUNT_Z60": "regime_eligible_count_z60",
    }


def rolling_mdd(returns: pd.Series, window: int) -> pd.Series:
    def _mdd(values: np.ndarray) -> float:
        nav = np.cumprod(1.0 + values)
        peak = np.maximum.accumulate(nav)
        return float(np.max(1.0 - nav / peak))

    return returns.rolling(window, min_periods=max(5, window // 2)).apply(_mdd, raw=True)


def rolling_downside_vol(returns: pd.Series, window: int) -> pd.Series:
    def _downside(values: np.ndarray) -> float:
        downside = values[values < 0]
        return float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0

    return returns.rolling(window, min_periods=max(5, window // 2)).apply(_downside, raw=True)


def add_regime_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.sort_values("datetime").copy()
    ret = out["regime_ret_1d"]
    out["regime_ret_5d"] = (1.0 + ret).rolling(5, min_periods=3).apply(np.prod, raw=True) - 1.0
    out["regime_ret_10d"] = (1.0 + ret).rolling(10, min_periods=5).apply(np.prod, raw=True) - 1.0
    out["regime_ret_20d"] = (1.0 + ret).rolling(20, min_periods=10).apply(np.prod, raw=True) - 1.0
    out["regime_vol_5d"] = ret.rolling(5, min_periods=3).std()
    out["regime_vol_20d"] = ret.rolling(20, min_periods=10).std()
    out["regime_downside_vol_20d"] = rolling_downside_vol(ret, 20)
    out["regime_mdd_20d"] = rolling_mdd(ret.fillna(0.0), 20)
    out["regime_breadth_up_5d"] = out["regime_breadth_up_1d"].rolling(5, min_periods=3).mean()
    out["regime_dispersion_20d"] = out["regime_dispersion_1d"].rolling(20, min_periods=10).mean()
    out["regime_avg_amount_20d"] = out["regime_avg_amount_1d"].rolling(20, min_periods=10).mean()
    out["regime_amount_chg_20d"] = (
        out["regime_avg_amount_1d"] / out["regime_avg_amount_20d"].shift(1) - 1.0
    )
    count_mean = out["eligible_count"].rolling(60, min_periods=20).mean()
    count_std = out["eligible_count"].rolling(60, min_periods=20).std()
    out["regime_eligible_count_z60"] = (out["eligible_count"] - count_mean) / count_std.replace(0.0, np.nan)
    return out


def build_daily_labels(
    source_provider: Path,
    raw_daily_dir: Path,
    start: str,
    end: str,
    horizons: list[int],
    min_listing_days: int,
    min_avg_amount: float,
) -> pd.DataFrame:
    instruments = [inst.upper() for inst in read_instruments(source_provider)]
    prices = load_price_panel(raw_daily_dir, instruments, start=start, end=end)
    prices = prices.sort_values(["instrument", "datetime"]).copy()
    prices["ret_1d"] = prices.groupby("instrument", sort=False)["close"].pct_change()
    prices = add_forward_paths(prices, horizons)
    eligible = add_point_in_time_eligibility(
        prices,
        UniverseRules(
            min_listing_days=min_listing_days,
            min_avg_amount=min_avg_amount,
        ),
    )
    daily_rows: list[dict] = []
    for date, day in eligible.groupby("datetime", sort=True):
        broad = day[day["eligible"].fillna(False)].copy()
        if broad.empty:
            continue
        stock_ret = broad["ret_1d"]
        row = {
            "datetime": pd.Timestamp(date).normalize(),
            "eligible_count": int(len(broad)),
            "regime_ret_1d": float(stock_ret.mean()),
            "regime_breadth_up_1d": float((stock_ret > 0).mean()),
            "regime_dispersion_1d": float(stock_ret.std()),
            "regime_avg_amount_1d": float(broad["amount"].mean()),
        }
        row.update(path_outcomes(broad, "broad", horizons))
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows).sort_values("datetime")
    daily = add_regime_features(daily)
    for horizon in horizons:
        mdd_col = f"broad_mdd_{horizon}d"
        fwd_col = f"broad_fwd_{horizon}d"
        finite = daily[mdd_col].notna() & daily[fwd_col].notna()
        daily[f"broad_adverse_mdd5_{horizon}d"] = np.where(
            finite, (daily[mdd_col] > 0.05).astype(float), np.nan
        )
        daily[f"broad_adverse_loss5_{horizon}d"] = np.where(
            finite, (daily[fwd_col] < -0.05).astype(float), np.nan
        )
        daily[f"broad_adverse_combo_{horizon}d"] = np.where(
            finite,
            ((daily[mdd_col] > 0.05) | (daily[fwd_col] < -0.05)).astype(float),
            np.nan,
        )
    return daily


def write_labels_to_provider(
    provider: Path,
    daily: pd.DataFrame,
    field_specs: dict[str, str],
) -> dict:
    calendar, cal_index = read_calendar(provider)
    labels = daily.copy()
    labels["datetime"] = pd.to_datetime(labels["datetime"]).dt.normalize()
    labels = labels.drop_duplicates("datetime", keep="last").set_index("datetime")
    valid_dates = [date for date in labels.index if date in cal_index]
    labels = labels.loc[valid_dates]
    if labels.empty:
        raise ValueError("no label dates overlap provider calendar")
    start_idx = cal_index[labels.index.min()]
    end_idx = cal_index[labels.index.max()]
    aligned = labels.reindex(calendar[start_idx : end_idx + 1])
    instruments = read_instruments(provider)
    for pos, inst in enumerate(instruments, start=1):
        feature_dir = provider / "features" / inst
        for field, source_col in field_specs.items():
            write_qlib_bin(
                aligned[source_col].to_numpy(dtype=np.float32),
                start_idx,
                feature_dir / f"{field.lower()}.day.bin",
            )
        if pos % 250 == 0:
            print(f"[outer broad labels] wrote {pos}/{len(instruments)}", flush=True)
    return {
        "date_start": str(labels.index.min().date()),
        "date_end": str(labels.index.max().date()),
        "calendar_start_idx": int(start_idx),
        "calendar_end_idx": int(end_idx),
        "instruments": int(len(instruments)),
    }


def build_provider(
    source_provider: Path,
    output_provider: Path,
    raw_daily_dir: Path,
    daily_output: Path,
    report_path: Path,
    start: str,
    end: str,
    horizons: list[int],
    min_listing_days: int,
    min_avg_amount: float,
) -> dict:
    copy_provider(source_provider, output_provider)
    label_specs = make_label_specs(horizons)
    feature_specs = make_feature_specs()
    daily = build_daily_labels(
        source_provider,
        raw_daily_dir,
        start,
        end,
        horizons,
        min_listing_days,
        min_avg_amount,
    )
    daily_output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_output, index=False, encoding="utf-8-sig")
    field_specs = {**label_specs, **feature_specs}
    write_report = write_labels_to_provider(output_provider, daily, field_specs)
    main_horizon = max(horizons)
    main_col = f"broad_adverse_mdd5_{main_horizon}d"
    report = {
        "status": "outer_broad_regime_history_provider_research_only",
        "source_provider": str(source_provider.resolve()),
        "output_provider": str(output_provider.resolve()),
        "raw_daily_dir": str(raw_daily_dir.resolve()),
        "daily_output": str(daily_output.resolve()),
        "date_requested_start": start,
        "date_requested_end": end,
        "horizons": horizons,
        "min_listing_days": int(min_listing_days),
        "min_avg_amount": float(min_avg_amount),
        "daily_rows": int(len(daily)),
        "valid_main_label_days": int(daily[main_col].notna().sum()) if main_col in daily else 0,
        "main_label_base_rate": float(daily[main_col].mean()) if main_col in daily else None,
        "label_fields": label_specs,
        "feature_fields": feature_specs,
        **write_report,
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-provider", default=str(DEFAULT_SOURCE_PROVIDER))
    parser.add_argument("--raw-daily-dir", default=str(DEFAULT_RAW_DAILY))
    parser.add_argument("--output-provider", required=True)
    parser.add_argument("--daily-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--start", default="2021-01-04")
    parser.add_argument("--end", default="2026-06-05")
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--min-listing-days", type=int, default=250)
    parser.add_argument("--min-avg-amount", type=float, default=100_000_000.0)
    args = parser.parse_args()
    report = build_provider(
        Path(args.source_provider).resolve(),
        Path(args.output_provider).resolve(),
        Path(args.raw_daily_dir).resolve(),
        Path(args.daily_output).resolve(),
        Path(args.report).resolve(),
        args.start,
        args.end,
        parse_horizons(args.horizons),
        args.min_listing_days,
        args.min_avg_amount,
    )
    print(
        "[outer broad labels] "
        f"provider={report['output_provider']} rows={report['daily_rows']} "
        f"valid_main={report['valid_main_label_days']} base_rate={report['main_label_base_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
