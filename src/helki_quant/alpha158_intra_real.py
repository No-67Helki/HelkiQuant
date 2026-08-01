# Copyright (c) HelkiQuant contributors.
# Licensed under the MIT License.
"""
Alpha158IntraReal: 面向内层（日内T方向二分类）的真·日内特征。

设计理念：
    - 内层做T，需要日内行为信息
    - 短窗口（5/10/20）+ K线形态 + 5个HF代理 + 8个日内代理 + 14个真·1min聚合因子
    - 真·1min 因子从 cn_data_1min_pool 离线聚合，作为日级 $-字段供 qlib 直接使用
    - 内层标签使用预计算的 Ref($INTRADAY_T_RET_STABLE, -1)，即更稳定的下一交易日日内T收益

14 个真·1min 因子（需先运行 scripts/precompute_intra_real.py 生成）：
    1. FIRST30M_RET     : 昨日开盘 30min 收益
    2. LAST30M_RET      : 昨日尾盘 30min 收益
    3. INTRA_MOM_DIFF   : 上午收益 - 下午收益
    4. MIN_VOL_STD      : 1min log-return std × √240（年化日内波动率近似）
    5. VWAP_DEV         : (close - vwap_intraday) / vwap
    6. MAX_MIN_VOL_RATIO: max(1min_volume) / mean(1min_volume)（大单冲击代理）
    7. PRICE_VOL_CORR_MIN: 1min ret 与 log(volume) 相关性
    8. VWAP_REVERT      : (close - vwap) / std(1min_close)（VWAP 偏差归一化）
    9. T_OPEN30_RET     : 早盘稳定窗口相对 T 开仓窗口收益
    10. T_MID_RET       : 上午中段相对早盘稳定窗口收益
    11. T_VWAP_SLOPE    : 下午 VWAP / 上午 VWAP - 1
    12. T_RANGE_POS     : 收盘价在当日 1min 高低区间的位置
    13. T_VOL_CONC      : 开盘 30min 成交量占比
    14. T_LATE_MOM      : 尾盘稳定窗口相对下午锚定窗口收益
标签字段：
    INTRADAY_T_RET      : 尾盘T窗口 VWAP / 早盘T窗口 VWAP - 1（仅做 label，不进入特征）
    INTRADAY_T_RET_STABLE: 稳定版标签，14:20~14:50 VWAP / 09:31~10:00 VWAP - 1

输出特征：~108
    - kbar: 9
    - price: 4
    - rolling: 15 ops × 3 windows = 45
    - HF: 5
    - DAILY_INNER: 8
    - INTRA_REAL: 14
"""
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL


# 复用 Alpha158HF 的 5 个高频代理因子
HF_FIELDS = [
    "$open / Ref($close, 1) - 1",
    "Mean(($close - $open) / $open, 5)",
    "($high - $low) / $open",
    "Std($close / Ref($close, 1) - 1, 5)",
    "Corr($close / Ref($close, 1) - 1, Log($volume / Ref($volume, 1) + 1e-12), 5)",
]
HF_NAMES = ["OVN_GAP", "MOM_TAIL5", "RNG_OPEN", "INTRA_VOL5", "PV_DIV5"]

# 复用 alpha158_inner 的 8 个日级日内代理 + 新增 DOWN_SHADOW_REAL
DAILY_INNER_FIELDS = [
    "Mean($open / Ref($close, 1) - 1, 3)",
    "Mean(($close - $open) / ($high - $low + 1e-8), 5)",
    "($close - $low) / ($high - $low + 1e-8)",
    "$volume / (Mean($volume, 5) + 1e-8)",
    "($high - $close) / ($high - $low + 1e-8)",
    # K线实体比例（原 BODY_RATIO，注释纠正）
    "Abs($close - $open) / ($high - $low + 1e-8)",
    "($close / Ref($close, 1) - 1 - Mean($close / Ref($close, 1) - 1, 20)) * "
    "($close / Ref($close, 1) - 1 - Mean($close / Ref($close, 1) - 1, 20)) * "
    "($close / Ref($close, 1) - 1 - Mean($close / Ref($close, 1) - 1, 20)) / "
    "(Std($close / Ref($close, 1) - 1, 20) + 1e-8)",
    "Mean(($high - $low) / $open, 5)",
    # 新增：真·下影线比例
    "(If($close > $open, $open, $close) - $low) / ($high - $low + 1e-8)",
]
DAILY_INNER_NAMES = [
    "GAP_MOM3", "TAIL_STR5", "INTRA_POS", "VOL_RATIO5",
    "UP_SHADOW", "BODY_RATIO", "RET_SKEW20", "AMPLITUDE_MA5",
    "DOWN_SHADOW_REAL",
]

# 真·1min 聚合因子（从 cn_data_1min_pool 离线生成，写入 cn_data_pool/features/）
# 推断时直接以 $-字段使用，但要求 scripts/precompute_intra_real.py 已执行
INTRA_REAL_FIELDS = [
    "$FIRST30M_RET",
    "$LAST30M_RET",
    "$INTRA_MOM_DIFF",
    "$MIN_VOL_STD",
    "$VWAP_DEV",
    "$MAX_MIN_VOL_RATIO",
    "$PRICE_VOL_CORR_MIN",
    "$VWAP_REVERT",
    "$T_OPEN30_RET",
    "$T_MID_RET",
    "$T_VWAP_SLOPE",
    "$T_RANGE_POS",
    "$T_VOL_CONC",
    "$T_LATE_MOM",
    "$OPEN15_RET",
    "$OPEN30_RANGE",
    "$OPEN30_VWAP_DEV",
    "$AM_RANGE_POS",
    "$PM_RANGE_POS",
    "$AM_PM_VOL_RATIO",
    "$TOP5_VOL_CONC",
    "$VWAP_CROSS_COUNT",
    "$RET_AUTOCORR_MIN",
    "$DOWNSIDE_VOL_RATIO",
    "$AM_PRICE_VOL_CORR",
    "$PM_PRICE_VOL_CORR",
    "$ILLIQUIDITY_MIN",
    "$CLOSE_PULLUP",
]
INTRA_REAL_NAMES = [
    "FIRST30M_RET", "LAST30M_RET", "INTRA_MOM_DIFF", "MIN_VOL_STD",
    "VWAP_DEV", "MAX_MIN_VOL_RATIO", "PRICE_VOL_CORR_MIN", "VWAP_REVERT",
    "T_OPEN30_RET", "T_MID_RET", "T_VWAP_SLOPE", "T_RANGE_POS",
    "T_VOL_CONC", "T_LATE_MOM",
    "OPEN15_RET", "OPEN30_RANGE", "OPEN30_VWAP_DEV", "AM_RANGE_POS",
    "PM_RANGE_POS", "AM_PM_VOL_RATIO", "TOP5_VOL_CONC", "VWAP_CROSS_COUNT",
    "RET_AUTOCORR_MIN", "DOWNSIDE_VOL_RATIO", "AM_PRICE_VOL_CORR",
    "PM_PRICE_VOL_CORR", "ILLIQUIDITY_MIN", "CLOSE_PULLUP",
]


class Alpha158IntraReal(Alpha158):
    """短窗 Alpha158 + 5HF + 9日内代理 + 14真·1min，共 ~108 个特征。

    使用前必须先运行：
        python scripts/precompute_intra_real.py
    生成 8 个真·1min 因子的 .day.bin 文件到 cn_data_pool/features/<inst>/
    """

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {
                "windows": [5, 10, 20],
                "include": [
                    "ROC", "MA", "STD", "BETA", "RSQR",
                    "RSV", "IMXD", "CORR", "CNTD",
                    "VMA", "VSTD", "WVMA", "SUMD",
                    "QTLU", "QTLD",
                ],
            },
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        fields = (list(fields)
                  + list(HF_FIELDS)
                  + list(DAILY_INNER_FIELDS)
                  + list(INTRA_REAL_FIELDS))
        names = (list(names)
                 + list(HF_NAMES)
                 + list(DAILY_INNER_NAMES)
                 + list(INTRA_REAL_NAMES))
        return fields, names
