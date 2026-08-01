from __future__ import annotations

from qlib.contrib.data.handler import Alpha158


REGIME_FIELDS = [
    "OUTER_REGIME_RET_1D",
    "OUTER_REGIME_RET_5D",
    "OUTER_REGIME_RET_10D",
    "OUTER_REGIME_RET_20D",
    "OUTER_REGIME_VOL_5D",
    "OUTER_REGIME_VOL_20D",
    "OUTER_REGIME_DOWNSIDE_VOL_20D",
    "OUTER_REGIME_MDD_20D",
    "OUTER_REGIME_BREADTH_UP_1D",
    "OUTER_REGIME_BREADTH_UP_5D",
    "OUTER_REGIME_DISPERSION_1D",
    "OUTER_REGIME_DISPERSION_20D",
    "OUTER_REGIME_AVG_AMOUNT_20D",
    "OUTER_REGIME_AMOUNT_CHG_20D",
    "OUTER_REGIME_ELIGIBLE_COUNT_Z60",
]


class Alpha158Regime(Alpha158):
    """Date-level market regime features for the outer risk overlay.

    These fields are written into every instrument's feature directory with the
    same daily value. The outer model therefore learns broad market risk state
    instead of cross-sectional stock selection.
    """

    def get_feature_config(self):
        fields = [f"${field}" for field in REGIME_FIELDS]
        names = [field.replace("OUTER_", "") for field in REGIME_FIELDS]
        return fields, names
