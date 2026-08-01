from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_STAGE_CSV = DATA / "_research_1min_pool_csv_2026"
DEFAULT_INNER_DAY_PROVIDER = DATA / "cn_data_pool_inner_research_2026"
DEFAULT_POOL_FILE = DATA / "cn_data_1min_pool" / "instruments" / "all.txt"

EXEC_LABELS = [
    "INTRADAY_T_EXEC_RET",
    "INTRADAY_T_EXEC_NET_RET",
    "INTRADAY_T_REVERSE_NET_RET",
    "INTRADAY_T0_SELL_OPEN_BUY_AM_NET_RET",
    "INTRADAY_T0_SELL_OPEN_BUY_PM_NET_RET",
    "INTRADAY_T0_SELL_AM_BUY_PM_NET_RET",
    "INTRADAY_T0_SELL_AM_BUY_CLOSE_NET_RET",
    "INTRADAY_T0_BEST_BUCKET_NET_RET",
    "INTRADAY_T0_BEST2_MEAN_NET_RET",
    "INTRADAY_T0_BUCKET_HIT_RATIO",
]


def read_calendar(day_provider: Path) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, int]]:
    calendar = pd.read_csv(
        day_provider / "calendars" / "day.txt",
        header=None,
        names=["date"],
        parse_dates=["date"],
    )["date"].tolist()
    return calendar, {date: idx for idx, date in enumerate(calendar)}


def read_instruments(path: Path) -> list[str]:
    return [
        line.split()[0].lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_qlib_bin(values: np.ndarray, start_idx: int, bin_path: Path) -> None:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    np.hstack([np.array([start_idx], dtype=np.float32), values.astype(np.float32)]).tofile(
        str(bin_path.resolve())
    )


def window_vwap(day: pd.DataFrame, start_minute: int, end_minute: int) -> float:
    mask = (day["minute_of_day"] >= start_minute) & (day["minute_of_day"] <= end_minute)
    if not mask.any():
        return np.nan
    sub = day.loc[mask]
    volume = pd.to_numeric(sub["volume"], errors="coerce").to_numpy(dtype=float)
    amount = pd.to_numeric(sub["amount"], errors="coerce").to_numpy(dtype=float)
    vol_sum = float(np.nansum(volume))
    if vol_sum <= 0:
        return np.nan
    return float(np.nansum(amount) / (vol_sum + 1e-12))


def round_trip_sell_buy_ret(
    sell_vwap: float,
    buy_vwap: float,
    sell_cost: float,
    buy_cost: float,
    slippage: float,
) -> float:
    if not np.isfinite(sell_vwap) or not np.isfinite(buy_vwap):
        return np.nan
    if sell_vwap <= 0 or buy_vwap <= 0:
        return np.nan
    return float(
        sell_vwap
        * (1.0 - sell_cost - slippage)
        / (buy_vwap * (1.0 + buy_cost + slippage) + 1e-12)
        - 1.0
    )


def best_sell_before_buy_bucket_ret(
    day: pd.DataFrame,
    buy_cost: float,
    sell_cost: float,
    slippage: float,
) -> float:
    # Fixed non-overlapping buckets keep the label executable enough for
    # research; it is an opportunity label, not an oracle tick high/low label.
    buckets = [
        ("open", 9 * 60 + 31, 9 * 60 + 35),
        ("am1", 10 * 60, 10 * 60 + 5),
        ("am2", 10 * 60 + 30, 10 * 60 + 35),
        ("pm1", 13 * 60 + 30, 13 * 60 + 35),
        ("pm2", 14 * 60, 14 * 60 + 5),
        ("close", 14 * 60 + 45, 14 * 60 + 50),
    ]
    vwaps = [window_vwap(day, start, end) for _, start, end in buckets]
    best = np.nan
    for sell_idx in range(len(vwaps) - 1):
        for buy_idx in range(sell_idx + 1, len(vwaps)):
            value = round_trip_sell_buy_ret(
                vwaps[sell_idx],
                vwaps[buy_idx],
                sell_cost=sell_cost,
                buy_cost=buy_cost,
                slippage=slippage,
            )
            if np.isfinite(value) and (not np.isfinite(best) or value > best):
                best = value
    return float(best) if np.isfinite(best) else np.nan


def sell_before_buy_bucket_returns(
    day: pd.DataFrame,
    buy_cost: float,
    sell_cost: float,
    slippage: float,
) -> list[float]:
    buckets = [
        ("open", 9 * 60 + 31, 9 * 60 + 35),
        ("am1", 10 * 60, 10 * 60 + 5),
        ("am2", 10 * 60 + 30, 10 * 60 + 35),
        ("pm1", 13 * 60 + 30, 13 * 60 + 35),
        ("pm2", 14 * 60, 14 * 60 + 5),
        ("close", 14 * 60 + 45, 14 * 60 + 50),
    ]
    vwaps = [window_vwap(day, start, end) for _, start, end in buckets]
    values: list[float] = []
    for sell_idx in range(len(vwaps) - 1):
        for buy_idx in range(sell_idx + 1, len(vwaps)):
            value = round_trip_sell_buy_ret(
                vwaps[sell_idx],
                vwaps[buy_idx],
                sell_cost=sell_cost,
                buy_cost=buy_cost,
                slippage=slippage,
            )
            if np.isfinite(value):
                values.append(float(value))
    return values


def best2_mean(values: list[float]) -> float:
    clean = sorted([value for value in values if np.isfinite(value)], reverse=True)
    if not clean:
        return np.nan
    if len(clean) == 1:
        return float(clean[0])
    return float(np.mean(clean[:2]))


def hit_ratio(values: list[float], threshold: float = 0.0) -> float:
    clean = [value for value in values if np.isfinite(value)]
    if not clean:
        return np.nan
    return float(np.mean([value > threshold for value in clean]))


def compute_exec_labels(
    minute_frame: pd.DataFrame,
    buy_cost: float,
    sell_cost: float,
    slippage: float,
) -> pd.DataFrame:
    frame = minute_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["trade_date"] = frame["date"].dt.normalize()
    frame["minute_of_day"] = frame["date"].dt.hour * 60 + frame["date"].dt.minute

    def window_series(start_minute: int, end_minute: int) -> pd.Series:
        mask = (frame["minute_of_day"] >= start_minute) & (frame["minute_of_day"] <= end_minute)
        if not mask.any():
            return pd.Series(dtype=float)
        sub = frame.loc[mask, ["trade_date", "amount", "volume"]].copy()
        sub["amount"] = pd.to_numeric(sub["amount"], errors="coerce")
        sub["volume"] = pd.to_numeric(sub["volume"], errors="coerce")
        grouped = sub.groupby("trade_date", sort=True)[["amount", "volume"]].sum(min_count=1)
        out = grouped["amount"] / (grouped["volume"] + 1e-12)
        return out.where(grouped["volume"] > 0)

    buckets = {
        "open": window_series(9 * 60 + 31, 9 * 60 + 35),
        "am1": window_series(10 * 60, 10 * 60 + 5),
        "am2": window_series(10 * 60 + 30, 10 * 60 + 35),
        "pm1": window_series(13 * 60 + 30, 13 * 60 + 35),
        "pm2": window_series(14 * 60, 14 * 60 + 5),
        "close": window_series(14 * 60 + 45, 14 * 60 + 50),
    }
    prices = pd.DataFrame(buckets).sort_index()
    if prices.empty:
        return pd.DataFrame(columns=EXEC_LABELS)

    def rt(sell: pd.Series, buy: pd.Series) -> pd.Series:
        return sell * (1.0 - sell_cost - slippage) / (
            buy * (1.0 + buy_cost + slippage) + 1e-12
        ) - 1.0

    out = pd.DataFrame(index=prices.index)
    out["INTRADAY_T_EXEC_RET"] = prices["close"] / (prices["open"] + 1e-12) - 1.0
    out["INTRADAY_T_EXEC_NET_RET"] = prices["close"] * (1.0 - sell_cost - slippage) / (
        prices["open"] * (1.0 + buy_cost + slippage) + 1e-12
    ) - 1.0
    out["INTRADAY_T_REVERSE_NET_RET"] = rt(prices["open"], prices["close"])
    out["INTRADAY_T0_SELL_OPEN_BUY_AM_NET_RET"] = rt(prices["open"], prices["am2"])
    out["INTRADAY_T0_SELL_OPEN_BUY_PM_NET_RET"] = rt(prices["open"], prices["pm2"])
    out["INTRADAY_T0_SELL_AM_BUY_PM_NET_RET"] = rt(prices["am1"], prices["pm2"])
    out["INTRADAY_T0_SELL_AM_BUY_CLOSE_NET_RET"] = rt(prices["am1"], prices["close"])

    bucket_order = ["open", "am1", "am2", "pm1", "pm2", "close"]
    pair_returns = []
    for sell_idx in range(len(bucket_order) - 1):
        for buy_idx in range(sell_idx + 1, len(bucket_order)):
            pair_returns.append(rt(prices[bucket_order[sell_idx]], prices[bucket_order[buy_idx]]))
    pair_frame = pd.concat(pair_returns, axis=1)
    out["INTRADAY_T0_BEST_BUCKET_NET_RET"] = pair_frame.max(axis=1, skipna=True)
    values = pair_frame.to_numpy(dtype=float)
    best2 = []
    for row in values:
        clean = row[np.isfinite(row)]
        if clean.size == 0:
            best2.append(np.nan)
        elif clean.size == 1:
            best2.append(float(clean[0]))
        else:
            best2.append(float(np.mean(np.sort(clean)[-2:])))
    out["INTRADAY_T0_BEST2_MEAN_NET_RET"] = best2
    out["INTRADAY_T0_BUCKET_HIT_RATIO"] = (pair_frame > 0).sum(axis=1) / pair_frame.notna().sum(axis=1).replace(0, np.nan)
    return out[EXEC_LABELS]


def augment(
    stage_dir: Path,
    inner_day_provider: Path,
    pool_file: Path,
    output_path: Path,
    buy_cost: float,
    sell_cost: float,
    slippage: float,
) -> dict:
    calendar, cal_index = read_calendar(inner_day_provider)
    instruments = read_instruments(pool_file)
    details = []
    for pos, inst in enumerate(instruments, start=1):
        csv_path = stage_dir / f"{inst}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing staged minute csv: {csv_path}")
        minute_frame = pd.read_csv(csv_path)
        labels = compute_exec_labels(minute_frame, buy_cost, sell_cost, slippage)
        labels.index = pd.to_datetime(labels.index).normalize()
        valid_dates = [date for date in labels.index if date in cal_index]
        labels = labels.loc[valid_dates]
        if labels.empty:
            raise ValueError(f"no executable labels in calendar for {inst}")
        start_idx = cal_index[labels.index.min()]
        end_idx = cal_index[labels.index.max()]
        aligned = labels.reindex(calendar[start_idx : end_idx + 1])
        feature_dir = inner_day_provider / "features" / inst
        for label in EXEC_LABELS:
            write_qlib_bin(
                aligned[label].to_numpy(dtype=np.float32),
                start_idx,
                feature_dir / f"{label.lower()}.day.bin",
            )
        net = labels["INTRADAY_T_EXEC_NET_RET"]
        reverse = labels["INTRADAY_T_REVERSE_NET_RET"]
        details.append(
            {
                "instrument": inst.upper(),
                "first": str(labels.index.min().date()),
                "last": str(labels.index.max().date()),
                "days": int(len(labels)),
                "net_positive_ratio": float((net > 0).mean()),
                "reverse_positive_ratio": float((reverse > 0).mean()),
                "net_mean": float(net.mean()),
                "net_median": float(net.median()),
                "gross_mean": float(labels["INTRADAY_T_EXEC_RET"].mean()),
            }
        )
        print(f"[inner labels] {pos}/{len(instruments)} {inst} {details[-1]['last']}", flush=True)

    report = {
        "status": "inner_exec_labels_augmented",
        "stage_csv": str(stage_dir),
        "inner_day_provider": str(inner_day_provider),
        "pool_file": str(pool_file),
        "label_fields": EXEC_LABELS,
        "open_window": "09:31-09:35",
        "close_window": "14:45-14:50",
        "costs": {
            "buy_cost": buy_cost,
            "sell_cost": sell_cost,
            "slippage_each_side": slippage,
        },
        "instruments": len(instruments),
        "details": details,
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE_CSV))
    parser.add_argument("--inner-day-provider", default=str(DEFAULT_INNER_DAY_PROVIDER))
    parser.add_argument("--pool-file", default=str(DEFAULT_POOL_FILE))
    parser.add_argument("--buy-cost", type=float, default=0.0005)
    parser.add_argument("--sell-cost", type=float, default=0.0015)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument(
        "--output", default=str(HERE / "outputs" / "inner_exec_label_augment.json")
    )
    args = parser.parse_args()
    report = augment(
        Path(args.stage_dir).resolve(),
        Path(args.inner_day_provider).resolve(),
        Path(args.pool_file).resolve(),
        Path(args.output).resolve(),
        args.buy_cost,
        args.sell_cost,
        args.slippage,
    )
    print(
        f"[inner labels] status={report['status']} instruments={report['instruments']} "
        f"fields={','.join(report['label_fields'])}"
    )


if __name__ == "__main__":
    main()
