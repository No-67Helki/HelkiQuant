from __future__ import annotations

import pandas as pd


def collapse_identical_datetime_rows(
    frame: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or "datetime" not in frame.index.names:
        raise ValueError(f"{context}: collapse_by_datetime requires a datetime MultiIndex")
    if "feature" not in frame.columns.get_level_values(0):
        raise ValueError(f"{context}: missing feature column group")
    if "label" not in frame.columns.get_level_values(0):
        raise ValueError(f"{context}: missing label column group")

    datetimes = frame.index.get_level_values("datetime")
    feature_hash = pd.util.hash_pandas_object(frame["feature"], index=False)
    feature_hash.index = datetimes
    max_feature_variants = int(feature_hash.groupby(level=0).nunique().max())

    labels = frame["label"]
    label_hash = pd.util.hash_pandas_object(labels, index=False)
    label_hash.index = datetimes
    max_label_variants = int(label_hash.groupby(level=0).nunique().max())
    if max_feature_variants != 1 or max_label_variants != 1:
        raise ValueError(
            f"{context}: rows are not identical within datetime "
            f"feature_variants={max_feature_variants} label_variants={max_label_variants}"
        )

    collapsed = frame.groupby(level="datetime", sort=True).first()
    collapsed.index.name = "datetime"
    return collapsed
