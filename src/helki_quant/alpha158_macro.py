# Copyright (c) HelkiQuant contributors.
# Licensed under the MIT License.
"""
Alpha158Macro: 面向外层（21日远期收益分类）的宏观/中长周期特征。

设计理念：
    - 外层判断中期趋势（主升 / 震荡 / 主跌），不需要短期噪声
    - 砍掉 Alpha158 的短窗口 5/10，保留并扩展长窗口 30/60/120/250
    - 精简 rolling 算子，只保留对中长趋势有判别力的
    - 追加 5 个截面排名因子（CSRank），让模型学到行业相对强弱

输出特征：~78 个
    - kbar: 9（K线形态，无窗口）
    - price: 4（OPEN0/HIGH0/LOW0/VWAP0）
    - rolling: 15 ops × 4 windows = 60
    - CSRank 截面特征: 5

单股推理时：CSRank 特征全部填 0.5（中位数占位）。
"""
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL


# 原计划中的 CSRank 截面排名因子被下掉：qlib 原生 Operators 未注册 CSRank
# （只有时间序列 Rank 算子）。如需補回可后期注册自定义 op。
CS_FIELDS: list[str] = []
CS_NAMES: list[str] = []


class Alpha158Macro(Alpha158):
    """长窗口（30/60/120/250）+ 截面排名因子，面向外层 21d 趋势分类。"""

    def get_feature_config(self):
        # 仅保留长窗口；rolling 算子用 include 精简
        # 默认 Alpha158 rolling 包含 29 个算子，这里只保留对中长趋势有判别力的 15 个
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {
                "windows": [30, 60, 120, 250],
                "include": [
                    "ROC", "MA", "STD", "BETA", "RSQR",
                    "RSV", "IMXD", "CORR", "CNTD",
                    "VMA", "VSTD", "WVMA", "SUMD",
                    "QTLU", "QTLD",
                ],
            },
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        fields = list(fields) + list(CS_FIELDS)
        names = list(names) + list(CS_NAMES)
        return fields, names
