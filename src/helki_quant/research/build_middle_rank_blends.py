from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_scores(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime"])
    score_column = "middle" if "middle" in frame.columns else "pred_middle"
    if score_column not in frame.columns:
        raise ValueError(f"{path} must contain middle or pred_middle")
    frame = frame[["datetime", "instrument", score_column]].rename(
        columns={score_column: name}
    )
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    duplicates = frame.duplicated(["datetime", "instrument"], keep=False)
    if duplicates.any():
        raise ValueError(f"{path} contains {int(duplicates.sum())} duplicate rows")
    if frame[name].isna().any():
        raise ValueError(f"{path} contains missing scores")
    return frame.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def rank_by_date(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("datetime", sort=False)[column].rank(
        pct=True, method="average"
    )


def parse_weights(value: str) -> list[float]:
    weights = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not weights or any(weight <= 0.0 or weight >= 1.0 for weight in weights):
        raise ValueError("weights must contain values strictly between zero and one")
    return weights


def build(
    baseline_path: Path,
    candidate_path: Path,
    output_dir: Path,
    weights: list[float],
) -> dict:
    baseline = load_scores(baseline_path, "baseline_score")
    candidate = load_scores(candidate_path, "candidate_score")
    merged = baseline.merge(
        candidate,
        on=["datetime", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("baseline and candidate predictions do not overlap")

    baseline_keys = baseline[["datetime", "instrument"]]
    candidate_keys = candidate[["datetime", "instrument"]]
    baseline_only = len(
        baseline_keys.merge(
            candidate_keys,
            on=["datetime", "instrument"],
            how="left",
            indicator=True,
        ).query("_merge == 'left_only'")
    )
    candidate_only = len(
        candidate_keys.merge(
            baseline_keys,
            on=["datetime", "instrument"],
            how="left",
            indicator=True,
        ).query("_merge == 'left_only'")
    )

    merged["baseline_rank"] = rank_by_date(merged, "baseline_score")
    merged["candidate_rank"] = rank_by_date(merged, "candidate_score")
    daily_correlation = merged.groupby("datetime", sort=False)[
        ["baseline_rank", "candidate_rank"]
    ].apply(lambda group: group["baseline_rank"].corr(group["candidate_rank"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for weight in weights:
        weight_tag = f"{int(round(weight * 100)):03d}"
        output_path = output_dir / f"middle_rank_blend_candidate_w{weight_tag}.csv"
        result = merged[["datetime", "instrument"]].copy()
        result["middle"] = (
            (1.0 - weight) * merged["baseline_rank"]
            + weight * merged["candidate_rank"]
        )
        result.to_csv(output_path, index=False)
        outputs.append(
            {
                "candidate_weight": weight,
                "baseline_weight": 1.0 - weight,
                "path": str(output_path.resolve()),
            }
        )

    report = {
        "status": "middle_oof_rank_blends_research_only",
        "baseline": str(baseline_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "rows": len(merged),
        "dates": int(merged["datetime"].nunique()),
        "date_start": str(merged["datetime"].min().date()),
        "date_end": str(merged["datetime"].max().date()),
        "instruments": int(merged["instrument"].nunique()),
        "baseline_only_rows": baseline_only,
        "candidate_only_rows": candidate_only,
        "daily_rank_correlation_mean": float(daily_correlation.mean()),
        "daily_rank_correlation_min": float(daily_correlation.min()),
        "outputs": outputs,
        "deployment_allowed": False,
    }
    report_path = output_dir / "middle_rank_blend_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", default="0.25,0.50,0.75")
    args = parser.parse_args()
    report = build(
        Path(args.baseline).resolve(),
        Path(args.candidate).resolve(),
        Path(args.output_dir).resolve(),
        parse_weights(args.weights),
    )
    print(
        f"[middle rank blend] rows={report['rows']} dates={report['dates']} "
        f"rank_corr={report['daily_rank_correlation_mean']:+.4f}"
    )
    for row in report["outputs"]:
        print(
            f"  candidate_weight={row['candidate_weight']:.0%} -> {row['path']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
