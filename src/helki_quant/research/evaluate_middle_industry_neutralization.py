from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from concentration_constraints import groups_on_date, load_group_metadata
from evaluate_middle_oof import score_sample
from universe import UniverseRules, add_point_in_time_eligibility, load_price_panel


def parse_weights(raw: str) -> list[float]:
    weights = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not weights or any(value < 0.0 or value > 1.0 for value in weights):
        raise ValueError("weights must be non-empty values in [0, 1]")
    return weights


def attach_groups(prediction: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for date, part in prediction.groupby("datetime", sort=True):
        mapped = part.copy()
        group_map = groups_on_date(metadata, pd.Timestamp(date))
        mapped["group"] = mapped["instrument"].map(group_map).fillna("__UNKNOWN__")
        parts.append(mapped)
    return pd.concat(parts, ignore_index=True)


def add_neutral_scores(frame: pd.DataFrame, weights: list[float]) -> pd.DataFrame:
    out = frame.copy()
    daily = out.groupby("datetime", sort=False)["middle"]
    daily_std = daily.transform("std").replace(0.0, np.nan)
    out["middle_global_z"] = (out["middle"] - daily.transform("mean")) / daily_std
    grouped = out.groupby(["datetime", "group"], sort=False)["middle"]
    group_count = grouped.transform("count")
    group_std = grouped.transform("std").replace(0.0, np.nan)
    group_z = (out["middle"] - grouped.transform("mean")) / group_std
    out["middle_industry_z"] = group_z.where(group_count >= 5, out["middle_global_z"])
    out["middle_industry_z"] = out["middle_industry_z"].fillna(out["middle_global_z"])
    for weight in weights:
        code = int(round(weight * 100))
        out[f"middle_neutral_{code:03d}"] = (
            (1.0 - weight) * out["middle_global_z"] + weight * out["middle_industry_z"]
        )
    return out


def assign_folds(frame: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    out = frame.copy()
    out["fold"] = np.nan
    for fold in folds:
        mask = out["datetime"].between(fold["test_start"], fold["test_end"])
        out.loc[mask, "fold"] = int(fold["fold"])
    return out.dropna(subset=["fold"]).assign(fold=lambda value: value["fold"].astype(int))


def evaluate(
    middle_path: Path,
    metadata_path: Path,
    folds_path: Path,
    raw_daily_dir: Path,
    output_dir: Path,
    report_path: Path,
    *,
    start: str,
    end: str,
    price_start: str,
    price_end: str,
    weights: list[float],
) -> dict:
    prediction = pd.read_csv(middle_path, parse_dates=["datetime"])
    prediction["datetime"] = prediction["datetime"].dt.normalize()
    prediction["instrument"] = prediction["instrument"].astype(str).str.upper()
    prediction = prediction[prediction["datetime"].between(start, end)].copy()
    if prediction.duplicated(["datetime", "instrument"]).any():
        raise ValueError("middle prediction has duplicate date/instrument rows")
    metadata = load_group_metadata(metadata_path, "industry")
    prediction = add_neutral_scores(attach_groups(prediction, metadata), weights)

    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    prediction = assign_folds(prediction, folds)
    instruments = prediction["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(raw_daily_dir, instruments, start=price_start, end=price_end)
    prices = prices.sort_values(["instrument", "datetime"])
    grouped = prices.groupby("instrument", sort=False)
    prices["forward_5d"] = grouped["close"].shift(-6) / grouped["close"].shift(-1) - 1.0
    eligible = add_point_in_time_eligibility(prices, UniverseRules())
    sample = prediction.merge(
        eligible[["datetime", "instrument", "eligible"]],
        on=["datetime", "instrument"],
        how="inner",
    ).merge(
        prices[["datetime", "instrument", "forward_5d"]],
        on=["datetime", "instrument"],
        how="inner",
    )
    sample = sample[sample["eligible"]].dropna(subset=["forward_5d"]).copy()

    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for weight in weights:
        code = int(round(weight * 100))
        score_col = f"middle_neutral_{code:03d}"
        scored = sample.copy()
        scored["middle"] = scored[score_col]
        aggregate = score_sample(scored)
        fold_metrics = []
        for fold, part in scored.groupby("fold", sort=True):
            metrics = score_sample(part)
            fold_metrics.append({"fold": int(fold), **metrics})
        fold_ics = np.asarray([row["daily_ic_mean"] for row in fold_metrics], dtype=float)
        fold_spreads = np.asarray([row["top_minus_bottom_mean"] for row in fold_metrics], dtype=float)
        selection_score = float(
            np.nanmedian(fold_ics)
            + np.nanmin(fold_ics)
            + 0.25 * np.nanmedian(fold_spreads)
            + 0.25 * np.nanmin(fold_spreads)
        )
        output_path = output_dir / f"middle_industry_neutral_w{code:03d}.csv"
        prediction[["datetime", "instrument", score_col]].rename(
            columns={score_col: "middle"}
        ).to_csv(output_path, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "industry_neutral_weight": weight,
                "output_prediction": str(output_path.resolve()),
                "aggregate": aggregate,
                "folds": fold_metrics,
                "worst_fold_ic": float(np.nanmin(fold_ics)),
                "median_fold_ic": float(np.nanmedian(fold_ics)),
                "positive_fold_ic_ratio": float(np.mean(fold_ics > 0.0)),
                "worst_fold_spread": float(np.nanmin(fold_spreads)),
                "selection_score": selection_score,
            }
        )

    ranked = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    report = {
        "status": "middle_industry_neutralization_oof_evaluated",
        "middle_prediction": str(middle_path.resolve()),
        "industry_metadata": str(metadata_path.resolve()),
        "folds_path": str(folds_path.resolve()),
        "window": {"start": start, "end": end},
        "weights": weights,
        "group_known_ratio": float((prediction["group"] != "__UNKNOWN__").mean()),
        "selection_policy": (
            "median_fold_ic + worst_fold_ic + 0.25*median_fold_spread + "
            "0.25*worst_fold_spread; historical OOF only"
        ),
        "candidates": ranked,
        "selected_weight": ranked[0]["industry_neutral_weight"],
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--middle", required=True)
    parser.add_argument("--industry-metadata", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--raw-daily-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--start", default="2025-01-03")
    parser.add_argument("--end", default="2026-04-02")
    parser.add_argument("--price-start", default="2022-01-04")
    parser.add_argument("--price-end", default="2026-06-05")
    parser.add_argument("--weights", default="0,0.25,0.5,0.75,1")
    args = parser.parse_args()
    report = evaluate(
        Path(args.middle).resolve(),
        Path(args.industry_metadata).resolve(),
        Path(args.folds).resolve(),
        Path(args.raw_daily_dir).resolve(),
        Path(args.output_dir).resolve(),
        Path(args.report).resolve(),
        start=args.start,
        end=args.end,
        price_start=args.price_start,
        price_end=args.price_end,
        weights=parse_weights(args.weights),
    )
    for row in report["candidates"]:
        print(
            "[middle neutral] "
            f"weight={row['industry_neutral_weight']:.2f} "
            f"ic={row['aggregate']['daily_ic_mean']:+.4f} "
            f"worst_ic={row['worst_fold_ic']:+.4f} "
            f"spread={row['aggregate']['top_minus_bottom_mean']:+.3%} "
            f"score={row['selection_score']:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
