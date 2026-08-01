from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_held_intraday_decision_dataset import add_cross_sectional_features
from held_intraday_factor_engineering import (
    REALTIME_ENGINEERED_FEATURES,
    add_realtime_reproducible_factors,
)


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def augment(
    input_csv: Path,
    held_context_csv: Path,
    output_csv: Path,
    output_json: Path,
) -> dict:
    frame = pd.read_csv(input_csv)
    frame["trade_date"] = _normalize_date(frame["trade_date"])
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    context = pd.read_csv(
        held_context_csv,
        usecols=["trade_date", "instrument", "group"],
    )
    context["trade_date"] = _normalize_date(context["trade_date"])
    context["instrument"] = context["instrument"].astype(str).str.upper()
    context = context.drop_duplicates(["trade_date", "instrument"], keep="last")
    if "group" in frame.columns:
        frame = frame.drop(columns=["group"])
    before = len(frame)
    frame = frame.merge(context, on=["trade_date", "instrument"], how="left", validate="many_to_one")
    if len(frame) != before:
        raise ValueError(f"industry merge changed row count before={before} after={len(frame)}")
    group_text = frame["group"].astype(str).str.strip().str.upper()
    known_group = ~group_text.isin({"", "NAN", "NONE", "UNKNOWN", "OTHER"})
    frame = add_realtime_reproducible_factors(frame)
    frame = add_cross_sectional_features(frame)
    frame = frame.sort_values(["trade_date", "instrument", "decision_time"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")

    feature_audit = {}
    for col in REALTIME_ENGINEERED_FEATURES:
        values = pd.to_numeric(frame[col], errors="coerce")
        finite = np.isfinite(values)
        feature_audit[col] = {
            "finite_ratio": float(finite.mean()),
            "nunique": int(values[finite].nunique()),
            "mean": float(values[finite].mean()) if finite.any() else None,
            "std": float(values[finite].std()) if finite.any() else None,
        }
    report = {
        "status": "held_intraday_live_factors_augmented",
        "input_csv": str(input_csv.resolve()),
        "held_context_csv": str(held_context_csv.resolve()),
        "output_csv": str(output_csv.resolve()),
        "rows": int(len(frame)),
        "dates": int(frame["trade_date"].nunique()),
        "instruments": int(frame["instrument"].nunique()),
        "industry_known_ratio": float(known_group.mean()),
        "realtime_reproducible": True,
        "future_label_columns_used": False,
        "feature_audit": feature_audit,
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--held-context-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = augment(
        Path(args.input_csv).resolve(),
        Path(args.held_context_csv).resolve(),
        Path(args.output_csv).resolve(),
        Path(args.output_json).resolve(),
    )
    print(
        f"[held live factors] rows={report['rows']} dates={report['dates']} "
        f"industry_known={report['industry_known_ratio']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
