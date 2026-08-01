from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from evaluate_held_intraday_decision_model import auc_score
from inner_t0_bidirectional_engine import percentile_score


def score_models(
    input_csv: Path,
    model_dir: Path,
    output_dir: Path,
    output_json: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    fold_id: int = 7,
    period_name: str | None = None,
) -> dict:
    manifest = json.loads((model_dir / "frozen_models_manifest.json").read_text(encoding="utf-8"))
    feature_cols = list(manifest["feature_cols"])
    directions = manifest["models"]
    required = {"datetime", "trade_date", "instrument", "decision_time", *feature_cols}
    for item in directions.values():
        required.add(item["label_col"])
        required.add(item["edge_col"])
    frame = pd.read_csv(
        input_csv,
        usecols=lambda col: col in required,
        parse_dates=["datetime", "trade_date"],
    ).replace([np.inf, -np.inf], np.nan)
    frame["decision_time"] = (
        frame["decision_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    )
    frame = frame[frame["decision_time"] == "1000"].copy()
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[feature_cols] = frame[feature_cols].fillna(0.0)
    calibration_starts = [pd.Timestamp(item["split"]["calibration_start"]) for item in directions.values()]
    calibration_ends = [pd.Timestamp(item["split"]["calibration_end"]) for item in directions.values()]
    calibration_start = max(calibration_starts)
    calibration_end = min(calibration_ends)
    if (start is None) != (end is None):
        raise ValueError("--start and --end must be provided together")
    is_forward = start is not None
    evaluation_start = pd.Timestamp(start) if start is not None else calibration_start
    evaluation_end = pd.Timestamp(end) if end is not None else calibration_end
    if evaluation_start > evaluation_end:
        raise ValueError("evaluation start must not exceed end")
    if is_forward and evaluation_start <= calibration_end:
        raise ValueError(
            "forward evaluation must start strictly after frozen calibration end "
            f"{calibration_end.date()}"
        )
    holdout = frame[frame["trade_date"].between(evaluation_start, evaluation_end)].copy()
    if holdout.empty:
        raise ValueError(
            f"no frozen evaluation rows between {evaluation_start.date()} and {evaluation_end.date()}"
        )
    period = period_name or ("forward" if is_forward else "calibration")
    period = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in period)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    reports = {}
    score_frames = {}
    for direction, item in directions.items():
        model = CatBoostClassifier()
        model.load_model(str((model_dir / Path(item["model_path"]).name).resolve()))
        calibration = np.load(model_dir / Path(item["calibration_path"]).name)
        label_col = item["label_col"]
        edge_col = item["edge_col"]
        eligible = holdout.dropna(subset=[label_col, edge_col]).copy()
        raw = model.predict_proba(eligible[feature_cols])[:, 1]
        score = np.asarray([percentile_score(value, calibration) for value in raw])
        output = eligible[
            ["datetime", "trade_date", "instrument", "decision_time", label_col, edge_col]
        ].copy()
        output["fold"] = int(fold_id)
        output["label"] = (pd.to_numeric(output[label_col], errors="coerce") > 0.5).astype(int)
        output["raw_score"] = raw
        output["score"] = score
        output_path = output_dir / f"frozen_{direction}_{period}_predictions.csv"
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        selected = output[output["score"] >= float(item["score_threshold"])]
        reports[direction] = {
            "output_predictions": str(output_path.resolve()),
            "rows": int(len(output)),
            "dates": int(output["trade_date"].nunique()),
            "instruments": int(output["instrument"].nunique()),
            "auc": auc_score(output["label"].to_numpy(), output["raw_score"].to_numpy()),
            "spearman_edge": float(output["raw_score"].corr(output[edge_col], method="spearman")),
            "threshold": float(item["score_threshold"]),
            "daily_top_n": int(item["daily_top_n"]),
            "selected_rows_before_top_n": int(len(selected)),
            "selected_dates_before_top_n": int(selected["trade_date"].nunique()),
        }
        score_frames[direction] = output
    buy = score_frames["buy_first"]
    sell = score_frames["sell_first"]
    buy_top = (
        buy[buy["score"] >= directions["buy_first"]["score_threshold"]]
        .sort_values(["trade_date", "score"], ascending=[True, False])
        .groupby("trade_date", sort=False)
        .head(int(directions["buy_first"]["daily_top_n"]))
    )
    sell_top = (
        sell[sell["score"] >= directions["sell_first"]["score_threshold"]]
        .sort_values(["trade_date", "score"], ascending=[True, False])
        .groupby("trade_date", sort=False)
        .head(int(directions["sell_first"]["daily_top_n"]))
    )
    conflicts = buy_top[["trade_date", "instrument"]].merge(
        sell_top[["trade_date", "instrument"]],
        on=["trade_date", "instrument"],
        how="inner",
    )
    report = {
        "status": (
            "held_intraday_frozen_models_forward_scored"
            if is_forward
            else "held_intraday_frozen_models_calibration_scored"
        ),
        "input_csv": str(input_csv.resolve()),
        "model_dir": str(model_dir.resolve()),
        "frozen_calibration_start": str(calibration_start.date()),
        "frozen_calibration_end": str(calibration_end.date()),
        "evaluation_start": str(evaluation_start.date()),
        "evaluation_end": str(evaluation_end.date()),
        "evaluation_dates": int(holdout["trade_date"].nunique()),
        "evaluation_fold": int(fold_id),
        "evaluation_period_name": period,
        "strictly_after_calibration": bool(is_forward),
        "outcomes_never_used_for_fit_early_stopping_or_percentile_cdf": True,
        "profiles_not_reoptimized_on_calibration_outcomes": True,
        "directions": reports,
        "pre_trigger_topn_conflicts": int(len(conflicts)),
        "deployment_allowed": False,
    }
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--fold-id", type=int, default=7)
    parser.add_argument("--period-name", default=None)
    args = parser.parse_args()
    report = score_models(
        Path(args.input_csv).resolve(),
        Path(args.model_dir).resolve(),
        Path(args.output_dir).resolve(),
        Path(args.output_json).resolve(),
        start=args.start,
        end=args.end,
        fold_id=args.fold_id,
        period_name=args.period_name,
    )
    print(
        "[inner frozen score] "
        + " ".join(
            f"{direction}_auc={item['auc']:.4f} selected={item['selected_rows_before_top_n']}"
            for direction, item in report["directions"].items()
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
