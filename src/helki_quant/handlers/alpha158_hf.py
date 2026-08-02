from __future__ import annotations

from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL


class Alpha158HF(Alpha158):
    """Alpha158 plus five daily features used by the frozen middle model."""

    HF_FIELDS = [
        "$open / Ref($close, 1) - 1",
        "Mean(($close - $open) / $open, 5)",
        "($high - $low) / $open",
        "Std($close / Ref($close, 1) - 1, 5)",
        (
            "Corr($close / Ref($close, 1) - 1, "
            "Log($volume / Ref($volume, 1) + 1e-12), 5)"
        ),
    ]
    HF_NAMES = ["OVN_GAP", "MOM_TAIL5", "RNG_OPEN", "INTRA_VOL5", "PV_DIV5"]

    def get_feature_config(self):
        config = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        fields, names = Alpha158DL.get_feature_config(config)
        return list(fields) + self.HF_FIELDS, list(names) + self.HF_NAMES
