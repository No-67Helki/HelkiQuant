from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from evaluate_held_intraday_decision_model import metrics, select_feature_cols


def normalize_decision_time(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)


def train(
    input_csv: Path,
    output_dir: Path,
    *,
    decision_time: str,
    label_col: str,
    threshold: float,
    trade_fraction: float,
    feature_mode: str,
    model_name_stem: str,
) -> dict:
    frame = pd.read_csv(input_csv, parse_dates=["trade_date", "datetime"]).replace([np.inf, -np.inf], np.nan)
    frame["decision_time"] = normalize_decision_time(frame["decision_time"])
    wanted = str(decision_time).zfill(4)
    frame = frame[frame["decision_time"] == wanted].copy()
    frame = frame.dropna(subset=[label_col, "t0_best_edge"]).copy()
    frame["label"] = (frame[label_col] > 0.5).astype(int)
    feature_cols = select_feature_cols(frame, feature_mode)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[feature_cols] = frame[feature_cols].fillna(0.0)
    model = CatBoostClassifier(
        iterations=180,
        depth=4,
        learning_rate=0.045,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=20260611,
        task_type="CPU",
        allow_writing_files=False,
        verbose=50,
    )
    model.fit(Pool(frame[feature_cols], label=frame["label"]))
    frame["score"] = model.predict_proba(frame[feature_cols])[:, 1]
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name_stem}_catboost.cbm"
    meta_path = output_dir / f"{model_name_stem}_model_meta.json"
    sample_path = output_dir / f"{model_name_stem}_train_scores.csv"
    model.save_model(str(model_path))
    report = {
        "status": "held_intraday_live_model_trained",
        "input_csv": str(input_csv.resolve()),
        "model_path": str(model_path.resolve()),
        "model_name_stem": model_name_stem,
        "decision_time": wanted,
        "label_col": label_col,
        "threshold": threshold,
        "trade_fraction": trade_fraction,
        "feature_cols": feature_cols,
        "feature_mode": feature_mode,
        "train_metrics": metrics(frame),
        "best_iteration": model.get_best_iteration(),
        "deployment_allowed": False,
        "paper_candidate_only": True,
    }
    meta_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    frame[["datetime", "trade_date", "instrument", "decision_time", label_col, "label", "t0_best_edge", "score"]].to_csv(
        sample_path, index=False, encoding="utf-8-sig"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decision-time", default="0935")
    parser.add_argument("--label-col", default="t0_hit_1445_1450")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--trade-fraction", type=float, default=0.30)
    parser.add_argument(
        "--feature-mode",
        choices=[
            "all",
            "live",
            "live_core",
            "live_limit",
            "live_industry",
            "live_compact_core",
            "live_compact_limit",
        ],
        default="all",
    )
    parser.add_argument("--model-name-stem", default="inner_t0_0935_1445")
    args = parser.parse_args()
    report = train(
        Path(args.input_csv).resolve(),
        Path(args.output_dir).resolve(),
        decision_time=args.decision_time,
        label_col=args.label_col,
        threshold=args.threshold,
        trade_fraction=args.trade_fraction,
        feature_mode=args.feature_mode,
        model_name_stem=args.model_name_stem,
    )
    print(
        "[held intraday live model] "
        f"rows={report['train_metrics']['rows']} auc={report['train_metrics'].get('auc')} "
        f"model={report['model_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
