# Copyright (c) HelkiQuant contributors.
# Licensed under the MIT License.
"""
预计算 8 个真·1min 聚合因子 + 1 个日内T标签，写入 qlib 二进制格式到 cn_data_pool/features/<inst>/。

输入：data/A_Stock_1min/<YYYY>_1min/<inst>_<YYYY>.csv（中文列名）
输出：data/cn_data_pool/features/<inst>/<factor>.day.bin（与 close.day.bin 对齐）

14 个因子：
    1. FIRST30M_RET     : 开盘 30min 收益 = (price@10:00 - open) / open
    2. LAST30M_RET      : 尾盘 30min 收益 = (close - price@14:30) / price@14:30
    3. INTRA_MOM_DIFF   : (上午收盘价/开盘 - 1) - (收盘/下午开盘 - 1)
    4. MIN_VOL_STD      : std(1min log return) × √240
    5. VWAP_DEV         : (close - vwap) / vwap，vwap = sum(amount)/sum(volume)
    6. MAX_MIN_VOL_RATIO: max(1min_volume) / mean(1min_volume)
    7. PRICE_VOL_CORR_MIN: corr(1min_ret, log(1min_volume+1))
    8. VWAP_REVERT      : (close - vwap) / std(1min_close + 1e-8)
    9. T_OPEN30_RET     : 早盘稳定窗口 VWAP / 09:31~09:35 VWAP - 1
   10. T_MID_RET        : 10:00~11:30 VWAP / 09:31~10:00 VWAP - 1
   11. T_VWAP_SLOPE     : 下午 VWAP / 上午 VWAP - 1
   12. T_RANGE_POS      : 收盘价在当日 1min 高低区间的位置
   13. T_VOL_CONC       : 开盘 30min 成交量 / 全天成交量
   14. T_LATE_MOM       : 尾盘稳定窗口 VWAP / 13:00~14:20 VWAP - 1

2 个标签：
    INTRADAY_T_RET      : 尾盘T窗口 VWAP / 早盘T窗口 VWAP - 1
                          默认早盘 09:31~09:35，尾盘 14:45~14:50
    INTRADAY_T_RET_STABLE:
                          稳定版日内T标签，14:20~14:50 VWAP / 09:31~10:00 VWAP - 1

仅处理 cn_data_1min_pool/instruments/all.txt 中列出的股票。
其它股票的 .day.bin 不会生成（推断时这些列为 NaN）。

用法：
    python scripts/precompute_intra_real.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ONEMIN_RAW_DIR = DATA_DIR / "A_Stock_1min"
POOL_INST_FILE = DATA_DIR / "cn_data_1min_pool" / "instruments" / "all.txt"
QLIB_DAY_DIR = DATA_DIR / "cn_data_pool"
CALENDAR_FILE = QLIB_DAY_DIR / "calendars" / "day.txt"
FEATURES_DIR = QLIB_DAY_DIR / "features"

INTRA_FEATURE_NAMES = [
    "FIRST30M_RET", "LAST30M_RET", "INTRA_MOM_DIFF", "MIN_VOL_STD",
    "VWAP_DEV", "MAX_MIN_VOL_RATIO", "PRICE_VOL_CORR_MIN", "VWAP_REVERT",
    "T_OPEN30_RET", "T_MID_RET", "T_VWAP_SLOPE", "T_RANGE_POS",
    "T_VOL_CONC", "T_LATE_MOM",
    "OPEN15_RET", "OPEN30_RANGE", "OPEN30_VWAP_DEV", "AM_RANGE_POS",
    "PM_RANGE_POS", "AM_PM_VOL_RATIO", "TOP5_VOL_CONC", "VWAP_CROSS_COUNT",
    "RET_AUTOCORR_MIN", "DOWNSIDE_VOL_RATIO", "AM_PRICE_VOL_CORR",
    "PM_PRICE_VOL_CORR", "ILLIQUIDITY_MIN", "CLOSE_PULLUP",
]
LABEL_NAMES = ["INTRADAY_T_RET", "INTRADAY_T_RET_STABLE"]
FACTOR_NAMES = INTRA_FEATURE_NAMES + LABEL_NAMES

# 中文列名（A股 1min CSV）→ 标准列名
COL_MAP = {
    "时间": "datetime",
    "代码": "instrument",
    "开盘价": "open",
    "收盘价": "close",
    "最高价": "high",
    "最低价": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def load_pool_instruments() -> list[str]:
    """读取 80 只精选池，返回小写代码列表（与 csv 文件名一致）。"""
    if not POOL_INST_FILE.exists():
        raise FileNotFoundError(f"pool list not found: {POOL_INST_FILE}")
    insts: list[str] = []
    with POOL_INST_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 形如: SZ301536  2024-03-28 09:30:00  2025-12-31 14:58:00
            insts.append(parts[0].lower())
    return insts


def load_calendar() -> tuple[list[pd.Timestamp], dict[pd.Timestamp, int]]:
    if not CALENDAR_FILE.exists():
        raise FileNotFoundError(f"calendar not found: {CALENDAR_FILE}")
    cal = pd.read_csv(CALENDAR_FILE, header=None, names=["date"], parse_dates=["date"])
    dates = list(cal["date"])
    return dates, {d: i for i, d in enumerate(dates)}


def find_1min_csv_files(inst: str) -> list[Path]:
    """收集该股票所有年份的 1min CSV 文件。

    inst 已是小写如 'sz301536'，文件命名形如 sz301536_2025.csv 位于
    A_Stock_1min/2025_1min/。
    """
    files: list[Path] = []
    for ydir in sorted(ONEMIN_RAW_DIR.glob("*_1min")):
        if not ydir.is_dir():
            continue
        # 优先全名匹配；CSV 文件名形如 sz301536_2025.csv
        for fname_pattern in (f"{inst}_*.csv", f"{inst.upper()}_*.csv"):
            files.extend(ydir.glob(fname_pattern))
    # 去重
    return sorted(set(files))


def read_1min_concat(files: list[Path]) -> pd.DataFrame | None:
    if not files:
        return None
    parts: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype={"代码": str}, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(f, dtype={"代码": str}, encoding="gbk")
        # 仅保留我们需要的列
        keep = [c for c in COL_MAP.keys() if c in df.columns]
        df = df[keep].rename(columns=COL_MAP)
        if "datetime" not in df.columns:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"])
        parts.append(df)
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["datetime"])
    out = out.sort_values("datetime").reset_index(drop=True)
    # 强制 OHLC + volume + amount 为 float
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def compute_daily_factors(min_df: pd.DataFrame) -> pd.DataFrame:
    """对单个股票的 1min 数据按交易日聚合，输出日级因子和日内T标签。

    Parameters
    ----------
    min_df : pd.DataFrame
        必含列：datetime, open, high, low, close, volume, amount

    Returns
    -------
    pd.DataFrame
        索引 = 交易日（pd.Timestamp，零点对齐），列 = FACTOR_NAMES
    """
    if min_df is None or min_df.empty:
        return pd.DataFrame(columns=FACTOR_NAMES)

    df = min_df.copy()
    df["date"] = df["datetime"].dt.normalize()
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute

    rows: list[dict] = []
    for date, day_df in df.groupby("date", sort=True):
        if len(day_df) < 30:
            # 数据太少跳过（可能是停牌当天）
            continue
        day_df = day_df.sort_values("minute_of_day").reset_index(drop=True)

        o = day_df["open"].values
        c_arr = day_df["close"].values
        h_arr = day_df["high"].values
        l_arr = day_df["low"].values
        v = day_df["volume"].values
        amt = day_df["amount"].values if "amount" in day_df.columns else (c_arr * v)

        # 通用聚合
        open_price = o[0]
        close_price = c_arr[-1]
        sum_v = float(v.sum())
        sum_amt = float(np.nansum(amt))
        def _normalize_price_scale(value, ref_price):
            if not np.isfinite(value) or not np.isfinite(ref_price) or ref_price <= 0:
                return value
            ratio = value / (ref_price + 1e-12)
            if ratio > 20:
                return value / 100.0
            if ratio < 0.05:
                return value * 100.0
            return value

        vwap = sum_amt / (sum_v + 1e-12) if sum_v > 0 else np.nan
        vwap = _normalize_price_scale(vwap, open_price)

        # 1. FIRST30M_RET: 开盘 30min 收益（09:30~09:59 收盘价 / open - 1）
        mod = day_df["minute_of_day"].values
        first30_mask = mod < (9 * 60 + 30 + 30)  # < 10:00
        if first30_mask.any():
            first30_close = c_arr[first30_mask][-1]
            first30_ret = first30_close / (open_price + 1e-12) - 1
        else:
            first30_ret = np.nan

        # 2. LAST30M_RET: 尾盘 30min 收益（14:30~15:00 close 对 14:30 close 的收益）
        last30_mask = mod >= (14 * 60 + 30)
        if last30_mask.any():
            anchor = c_arr[last30_mask][0]
            last30_ret = close_price / (anchor + 1e-12) - 1
        else:
            last30_ret = np.nan

        # 3. INTRA_MOM_DIFF: 上午 vs 下午动量差
        morning_mask = mod < (11 * 60 + 30)
        afternoon_mask = mod >= (13 * 60)
        if morning_mask.any() and afternoon_mask.any():
            mor_open = o[morning_mask][0]
            mor_close = c_arr[morning_mask][-1]
            aft_open = o[afternoon_mask][0]
            aft_close = c_arr[afternoon_mask][-1]
            mor_ret = mor_close / (mor_open + 1e-12) - 1
            aft_ret = aft_close / (aft_open + 1e-12) - 1
            intra_mom_diff = mor_ret - aft_ret
        else:
            intra_mom_diff = np.nan

        # 4. MIN_VOL_STD: 1min log-return std × √240
        log_ret = np.diff(np.log(c_arr + 1e-12))
        if len(log_ret) > 5:
            min_vol_std = float(np.nanstd(log_ret) * np.sqrt(240))
        else:
            min_vol_std = np.nan

        # 5. VWAP_DEV: (close - vwap) / vwap
        vwap_dev = (close_price - vwap) / (vwap + 1e-12) if not np.isnan(vwap) else np.nan

        # 6. MAX_MIN_VOL_RATIO
        if sum_v > 0:
            max_min_vol_ratio = float(v.max() / (v.mean() + 1e-12))
        else:
            max_min_vol_ratio = np.nan

        # 7. PRICE_VOL_CORR_MIN: corr(1min_ret, log(volume+1))
        if len(log_ret) > 5:
            log_v = np.log(v[1:] + 1.0)
            if np.nanstd(log_ret) > 1e-12 and np.nanstd(log_v) > 1e-12:
                price_vol_corr_min = float(np.corrcoef(log_ret, log_v)[0, 1])
            else:
                price_vol_corr_min = np.nan
        else:
            price_vol_corr_min = np.nan

        # 8. VWAP_REVERT: (close - vwap) / std(1min_close)
        std_c = float(np.nanstd(c_arr))
        if std_c > 1e-8 and not np.isnan(vwap):
            vwap_revert = (close_price - vwap) / std_c
        else:
            vwap_revert = np.nan

        def _window_vwap(mask):
            if not mask.any():
                return np.nan
            v_sub = v[mask]
            amt_sub = amt[mask]
            vol_sum = float(np.nansum(v_sub))
            if vol_sum <= 0:
                return np.nan
            raw = float(np.nansum(amt_sub) / (vol_sum + 1e-12))
            ref = float(c_arr[mask][0]) if mask.any() else open_price
            return float(_normalize_price_scale(raw, ref))

        def _window_volume(mask):
            if not mask.any():
                return np.nan
            return float(np.nansum(v[mask]))

        def _window_ret(mask):
            if not mask.any():
                return np.nan
            first = float(o[mask][0])
            last = float(c_arr[mask][-1])
            return last / (first + 1e-12) - 1.0

        def _window_range(mask):
            if not mask.any():
                return np.nan
            return (float(np.nanmax(h_arr[mask])) - float(np.nanmin(l_arr[mask]))) / (
                open_price + 1e-12
            )

        def _safe_corr(left, right):
            left = np.asarray(left, dtype=float)
            right = np.asarray(right, dtype=float)
            mask = np.isfinite(left) & np.isfinite(right)
            if int(mask.sum()) < 6:
                return np.nan
            left = left[mask]
            right = right[mask]
            if np.nanstd(left) <= 1e-12 or np.nanstd(right) <= 1e-12:
                return np.nan
            return float(np.corrcoef(left, right)[0, 1])

        open_t_mask = (mod >= (9 * 60 + 31)) & (mod <= (9 * 60 + 35))
        open15_mask = (mod >= (9 * 60 + 31)) & (mod <= (9 * 60 + 45))
        close_t_mask = (mod >= (14 * 60 + 45)) & (mod <= (14 * 60 + 50))
        open_stable_mask = (mod >= (9 * 60 + 31)) & (mod <= (10 * 60))
        mid_mask = (mod >= (10 * 60)) & (mod <= (11 * 60 + 30))
        morning_stable_mask = mod < (11 * 60 + 30)
        afternoon_stable_mask = mod >= (13 * 60)
        late_anchor_mask = (mod >= (13 * 60)) & (mod < (14 * 60 + 20))
        close_stable_mask = (mod >= (14 * 60 + 20)) & (mod <= (14 * 60 + 50))

        open_t_vwap = _window_vwap(open_t_mask)
        open15_vwap = _window_vwap(open15_mask)
        close_t_vwap = _window_vwap(close_t_mask)
        open_stable_vwap = _window_vwap(open_stable_mask)
        mid_vwap = _window_vwap(mid_mask)
        morning_vwap = _window_vwap(morning_stable_mask)
        afternoon_vwap = _window_vwap(afternoon_stable_mask)
        late_anchor_vwap = _window_vwap(late_anchor_mask)
        close_stable_vwap = _window_vwap(close_stable_mask)

        if np.isfinite(open_t_vwap) and np.isfinite(open_stable_vwap) and open_t_vwap > 0:
            t_open30_ret = open_stable_vwap / (open_t_vwap + 1e-12) - 1
        else:
            t_open30_ret = np.nan
        if np.isfinite(mid_vwap) and np.isfinite(open_stable_vwap) and open_stable_vwap > 0:
            t_mid_ret = mid_vwap / (open_stable_vwap + 1e-12) - 1
        else:
            t_mid_ret = np.nan
        if np.isfinite(afternoon_vwap) and np.isfinite(morning_vwap) and morning_vwap > 0:
            t_vwap_slope = afternoon_vwap / (morning_vwap + 1e-12) - 1
        else:
            t_vwap_slope = np.nan
        day_high = float(np.nanmax(day_df["high"].values))
        day_low = float(np.nanmin(day_df["low"].values))
        t_range_pos = (close_price - day_low) / (day_high - day_low + 1e-12)
        open30_vol = _window_volume(first30_mask)
        t_vol_conc = open30_vol / (sum_v + 1e-12) if np.isfinite(open30_vol) and sum_v > 0 else np.nan
        if np.isfinite(close_stable_vwap) and np.isfinite(late_anchor_vwap) and late_anchor_vwap > 0:
            t_late_mom = close_stable_vwap / (late_anchor_vwap + 1e-12) - 1
        else:
            t_late_mom = np.nan

        if np.isfinite(open_t_vwap) and np.isfinite(close_t_vwap) and open_t_vwap > 0:
            intraday_t_ret = close_t_vwap / (open_t_vwap + 1e-12) - 1
        else:
            intraday_t_ret = np.nan
        if np.isfinite(open_stable_vwap) and np.isfinite(close_stable_vwap) and open_stable_vwap > 0:
            intraday_t_ret_stable = close_stable_vwap / (open_stable_vwap + 1e-12) - 1
        else:
            intraday_t_ret_stable = np.nan

        open15_ret = (
            open15_vwap / (open_price + 1e-12) - 1.0 if np.isfinite(open15_vwap) else np.nan
        )
        open30_range = _window_range(first30_mask)
        open30_vwap_dev = (
            open_stable_vwap / (open_price + 1e-12) - 1.0
            if np.isfinite(open_stable_vwap)
            else np.nan
        )
        if morning_mask.any():
            mor_high = float(np.nanmax(h_arr[morning_mask]))
            mor_low = float(np.nanmin(l_arr[morning_mask]))
            mor_close = float(c_arr[morning_mask][-1])
            am_range_pos = (mor_close - mor_low) / (mor_high - mor_low + 1e-12)
        else:
            am_range_pos = np.nan
        if afternoon_mask.any():
            pm_high = float(np.nanmax(h_arr[afternoon_mask]))
            pm_low = float(np.nanmin(l_arr[afternoon_mask]))
            pm_close = float(c_arr[afternoon_mask][-1])
            pm_range_pos = (pm_close - pm_low) / (pm_high - pm_low + 1e-12)
        else:
            pm_range_pos = np.nan
        morning_vol = _window_volume(morning_stable_mask)
        afternoon_vol = _window_volume(afternoon_stable_mask)
        am_pm_vol_ratio = (
            morning_vol / (afternoon_vol + 1e-12)
            if np.isfinite(morning_vol) and np.isfinite(afternoon_vol) and afternoon_vol > 0
            else np.nan
        )
        top5_vol_conc = (
            float(np.nansum(np.sort(v[np.isfinite(v)])[-5:]) / (sum_v + 1e-12))
            if sum_v > 0 and np.isfinite(v).any()
            else np.nan
        )
        cum_amt = np.nancumsum(amt)
        cum_vol = np.nancumsum(v)
        cum_vwap = cum_amt / (cum_vol + 1e-12)
        if len(cum_vwap):
            scale_ref = open_price if np.isfinite(open_price) and open_price > 0 else close_price
            cum_ratio = np.nanmedian(cum_vwap / (scale_ref + 1e-12))
            if np.isfinite(cum_ratio) and cum_ratio > 20:
                cum_vwap = cum_vwap / 100.0
            elif np.isfinite(cum_ratio) and cum_ratio < 0.05:
                cum_vwap = cum_vwap * 100.0
        spread = c_arr - cum_vwap
        spread_sign = np.sign(spread[np.isfinite(spread)])
        spread_sign = spread_sign[spread_sign != 0]
        if len(spread_sign) > 1:
            vwap_cross_count = float(np.sum(spread_sign[1:] != spread_sign[:-1]))
        else:
            vwap_cross_count = np.nan
        ret_autocorr_min = _safe_corr(log_ret[1:], log_ret[:-1]) if len(log_ret) > 6 else np.nan
        neg_ret = log_ret[log_ret < 0]
        pos_ret = log_ret[log_ret > 0]
        downside_vol_ratio = (
            float(np.nanstd(neg_ret) / (np.nanstd(pos_ret) + 1e-12))
            if len(neg_ret) > 3 and len(pos_ret) > 3
            else np.nan
        )
        ret_series = np.r_[np.nan, log_ret]
        log_volume = np.log(v + 1.0)
        am_price_vol_corr = _safe_corr(ret_series[morning_stable_mask], log_volume[morning_stable_mask])
        pm_price_vol_corr = _safe_corr(ret_series[afternoon_stable_mask], log_volume[afternoon_stable_mask])
        illiquidity_min = (
            float(np.nansum(np.abs(log_ret)) / (sum_amt + 1e-12))
            if len(log_ret) > 5 and sum_amt > 0
            else np.nan
        )
        close_pullup = (
            close_price / (close_stable_vwap + 1e-12) - 1.0
            if np.isfinite(close_stable_vwap)
            else np.nan
        )

        rows.append({
            "date": date,
            "FIRST30M_RET": first30_ret,
            "LAST30M_RET": last30_ret,
            "INTRA_MOM_DIFF": intra_mom_diff,
            "MIN_VOL_STD": min_vol_std,
            "VWAP_DEV": vwap_dev,
            "MAX_MIN_VOL_RATIO": max_min_vol_ratio,
            "PRICE_VOL_CORR_MIN": price_vol_corr_min,
            "VWAP_REVERT": vwap_revert,
            "T_OPEN30_RET": t_open30_ret,
            "T_MID_RET": t_mid_ret,
            "T_VWAP_SLOPE": t_vwap_slope,
            "T_RANGE_POS": t_range_pos,
            "T_VOL_CONC": t_vol_conc,
            "T_LATE_MOM": t_late_mom,
            "OPEN15_RET": open15_ret,
            "OPEN30_RANGE": open30_range,
            "OPEN30_VWAP_DEV": open30_vwap_dev,
            "AM_RANGE_POS": am_range_pos,
            "PM_RANGE_POS": pm_range_pos,
            "AM_PM_VOL_RATIO": am_pm_vol_ratio,
            "TOP5_VOL_CONC": top5_vol_conc,
            "VWAP_CROSS_COUNT": vwap_cross_count,
            "RET_AUTOCORR_MIN": ret_autocorr_min,
            "DOWNSIDE_VOL_RATIO": downside_vol_ratio,
            "AM_PRICE_VOL_CORR": am_price_vol_corr,
            "PM_PRICE_VOL_CORR": pm_price_vol_corr,
            "ILLIQUIDITY_MIN": illiquidity_min,
            "CLOSE_PULLUP": close_pullup,
            "INTRADAY_T_RET": intraday_t_ret,
            "INTRADAY_T_RET_STABLE": intraday_t_ret_stable,
        })

    if not rows:
        return pd.DataFrame(columns=FACTOR_NAMES)
    out = pd.DataFrame(rows).set_index("date")
    return out[FACTOR_NAMES]


def write_qlib_bin(values: np.ndarray, start_idx: int, bin_path: Path):
    """qlib 日频二进制格式： [start_idx(<f), values(<f) ...]。"""
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    np.hstack([np.array([start_idx], dtype=np.float32),
               values.astype(np.float32)]).tofile(str(bin_path.resolve()))


def process_one_instrument(inst_lower: str, calendar: list[pd.Timestamp],
                           cal_index: dict[pd.Timestamp, int]) -> bool:
    """处理单只股票：聚合 1min → 写 8 个 .day.bin 文件。"""
    files = find_1min_csv_files(inst_lower)
    if not files:
        print(f"  [warn] {inst_lower}: no 1min csv found, skip")
        return False
    min_df = read_1min_concat(files)
    if min_df is None or min_df.empty:
        print(f"  [warn] {inst_lower}: empty 1min data after concat, skip")
        return False

    daily = compute_daily_factors(min_df)
    if daily.empty:
        print(f"  [warn] {inst_lower}: aggregated empty, skip")
        return False

    # 对齐到 qlib calendar
    daily.index = pd.to_datetime(daily.index).normalize()
    # 仅保留 calendar 中存在的日期
    valid_dates = [d for d in daily.index if d in cal_index]
    daily = daily.loc[valid_dates]
    if daily.empty:
        print(f"  [warn] {inst_lower}: no dates in calendar, skip")
        return False

    # qlib bin 需要：start_idx + 从该 idx 到末尾的连续值（缺失日期填 NaN）
    start_idx = cal_index[daily.index.min()]
    end_idx = cal_index[daily.index.max()]
    length = end_idx - start_idx + 1
    # 构造对齐的全列
    aligned_dates = calendar[start_idx:end_idx + 1]
    aligned = daily.reindex(aligned_dates)

    feat_dir = FEATURES_DIR / inst_lower
    feat_dir.mkdir(parents=True, exist_ok=True)
    for factor in FACTOR_NAMES:
        vals = aligned[factor].values.astype(np.float32)
        bin_path = feat_dir / f"{factor.lower()}.day.bin"
        write_qlib_bin(vals, start_idx, bin_path)

    print(f"  [ok] {inst_lower}: {length} days written ({daily.index.min().date()} ~ {daily.index.max().date()})")
    return True


def main():
    parser = argparse.ArgumentParser(description="离线聚合 1min → 8 个真·日内因子 → qlib bin")
    parser.add_argument("--instruments", nargs="*", default=None,
                        help="指定股票列表（小写），默认使用 cn_data_1min_pool 全部")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅处理前 N 只（调试用）")
    args = parser.parse_args()

    print(f"[precompute] project_root = {PROJECT_ROOT}")
    print(f"[precompute] features_dir = {FEATURES_DIR}")

    insts = args.instruments if args.instruments else load_pool_instruments()
    if args.limit:
        insts = insts[: args.limit]
    print(f"[precompute] 待处理股票数: {len(insts)}")

    calendar, cal_index = load_calendar()
    print(f"[precompute] 日历范围: {calendar[0].date()} ~ {calendar[-1].date()} "
          f"({len(calendar)} 个交易日)")

    t0 = time.time()
    n_ok = 0
    for i, inst in enumerate(insts, 1):
        print(f"[{i}/{len(insts)}] {inst}", flush=True)
        try:
            if process_one_instrument(inst, calendar, cal_index):
                n_ok += 1
        except Exception as e:
            import traceback
            print(f"  [error] {inst}: {e}")
            traceback.print_exc()
        if i % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(insts) - i)
            print(f"  ...elapsed {elapsed:.0f}s, eta {eta:.0f}s")
    print(f"[precompute] done. success={n_ok}/{len(insts)}, "
          f"total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
