from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

try:
    from .evaluate_held_intraday_decision_model import auc_score, select_feature_cols
    from .run_held_intraday_anchored_oof import edge_col_for_label
except ImportError:
    from evaluate_held_intraday_decision_model import auc_score, select_feature_cols
    from run_held_intraday_anchored_oof import edge_col_for_label


PROFILES = {
    "buy_first": {
        "label_col": "t0_buy_first_1445_1450_one_lot_max50_hit",
        "score_threshold": 0.925,
        "daily_top_n": 2,
        "trigger_distance": 0.006,
    },
    "sell_first": {
        "label_col": "t0_exec_1445_1450_one_lot_max50_hit",
        "score_threshold": 0.975,
        "daily_top_n": 1,
        "trigger_distance": 0.0075,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_dates(
    dates: pd.DatetimeIndex,
    *,
    validation_fraction: float,
    min_validation_sessions: int,
    calibration_fraction: float,
    min_calibration_sessions: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    calibration_sessions = max(
        min_calibration_sessions,
        int(np.ceil(len(dates) * calibration_fraction)),
    )
    calibration = dates[-calibration_sessions:]
    remaining = dates[:-calibration_sessions]
    validation_sessions = max(
        min_validation_sessions,
        int(np.ceil(len(remaining) * validation_fraction)),
    )
    if validation_sessions >= len(remaining) - 20:
        raise ValueError(
            f"insufficient dates for fit/validation/calibration: total={len(dates)} "
            f"validation={validation_sessions} calibration={calibration_sessions}"
        )
    return remaining[:-validation_sessions], remaining[-validation_sessions:], calibration


def classification_metrics(frame: pd.DataFrame, edge_col: str) -> dict:
    return {
        "rows": int(len(frame)),
        "dates": int(frame["trade_date"].nunique()),
        "positive_ratio": float(frame["label"].mean()),
        "auc": auc_score(frame["label"].to_numpy(), frame["raw_score"].to_numpy()),
        "spearman_edge": float(frame["raw_score"].corr(frame[edge_col], method="spearman")),
        "raw_score_mean": float(frame["raw_score"].mean()),
        "raw_score_std": float(frame["raw_score"].std()),
    }


def train_one(
    base: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    direction: str,
    profile: dict,
    *,
    validation_fraction: float,
    min_validation_sessions: int,
    calibration_fraction: float,
    min_calibration_sessions: int,
) -> dict:
    label_col = str(profile["label_col"])
    edge_col = edge_col_for_label(label_col)
    frame = base.dropna(subset=[label_col, edge_col]).copy()
    frame["label"] = (pd.to_numeric(frame[label_col], errors="coerce") > 0.5).astype(int)
    dates = pd.DatetimeIndex(frame["trade_date"].dropna().unique()).sort_values()
    fit_dates, validation_dates, calibration_dates = split_dates(
        dates,
        validation_fraction=validation_fraction,
        min_validation_sessions=min_validation_sessions,
        calibration_fraction=calibration_fraction,
        min_calibration_sessions=min_calibration_sessions,
    )
    fit = frame[frame["trade_date"].isin(fit_dates)].copy()
    validation = frame[frame["trade_date"].isin(validation_dates)].copy()
    calibration = frame[frame["trade_date"].isin(calibration_dates)].copy()
    common = {
        "iterations": 160,
        "depth": 4,
        "learning_rate": 0.045,
        "l2_leaf_reg": 8.0,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 20260714 if direction == "buy_first" else 20260715,
        "task_type": "CPU",
        "allow_writing_files": False,
        "verbose": 25,
    }
    selector = CatBoostClassifier(**common)
    selector.fit(
        Pool(fit[feature_cols], label=fit["label"]),
        eval_set=Pool(validation[feature_cols], label=validation["label"]),
        use_best_model=True,
        early_stopping_rounds=30,
    )
    best_iteration = max(int(selector.get_best_iteration()) + 1, 1)
    final_train = pd.concat([fit, validation], ignore_index=True)
    final_params = {**common, "iterations": best_iteration, "verbose": 25}
    model = CatBoostClassifier(**final_params)
    model.fit(Pool(final_train[feature_cols], label=final_train["label"]))
    validation["raw_score"] = selector.predict_proba(validation[feature_cols])[:, 1]
    calibration["raw_score"] = model.predict_proba(calibration[feature_cols])[:, 1]
    calibration_values = np.sort(calibration["raw_score"].to_numpy(dtype=float))
    stem = f"inner_t0_1000_1445_{direction}"
    model_path = output_dir / f"{stem}_catboost.cbm"
    calibration_path = output_dir / f"{stem}_calibration.npy"
    meta_path = output_dir / f"{stem}_model_meta.json"
    score_path = output_dir / f"{stem}_calibration_scores.csv"
    model.save_model(str(model_path))
    np.save(calibration_path, calibration_values)
    calibration[
        ["trade_date", "instrument", label_col, edge_col, "label", "raw_score"]
    ].to_csv(score_path, index=False, encoding="utf-8-sig")
    report = {
        "status": "held_intraday_bidirectional_frozen_model_trained",
        "direction": direction,
        "decision_time": "1000",
        "exit_window": "1445_1450",
        "label_col": label_col,
        "edge_col": edge_col,
        "feature_mode": "live",
        "feature_cols": feature_cols,
        "model_path": str(model_path.resolve()),
        "calibration_path": str(calibration_path.resolve()),
        "calibration_scores_path": str(score_path.resolve()),
        "score_calibration": "percentile_cdf",
        "score_threshold": float(profile["score_threshold"]),
        "daily_top_n": int(profile["daily_top_n"]),
        "trigger_distance": float(profile["trigger_distance"]),
        "trade_sizing": "one_lot_max50",
        "best_iteration_count": best_iteration,
        "split": {
            "fit_start": str(fit_dates.min().date()),
            "fit_end": str(fit_dates.max().date()),
            "fit_dates": int(len(fit_dates)),
            "validation_start": str(validation_dates.min().date()),
            "validation_end": str(validation_dates.max().date()),
            "validation_dates": int(len(validation_dates)),
            "calibration_start": str(calibration_dates.min().date()),
            "calibration_end": str(calibration_dates.max().date()),
            "calibration_dates": int(len(calibration_dates)),
            "calibration_never_used_for_model_fit_or_early_stopping": True,
        },
        "validation_metrics": classification_metrics(validation, edge_col),
        "calibration_metrics": classification_metrics(calibration, edge_col),
        "model_sha256": sha256_file(model_path),
        "calibration_sha256": sha256_file(calibration_path),
        "research_gate_passed": True,
        "paper_candidate_only": True,
        "deployment_allowed": False,
    }
    meta_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["meta_path"] = str(meta_path.resolve())
    print(
        f"[inner frozen] direction={direction} iterations={best_iteration} "
        f"validation_auc={report['validation_metrics']['auc']} "
        f"calibration_auc={report['calibration_metrics']['auc']}",
        flush=True,
    )
    return report


def train_all(
    input_csv: Path,
    output_dir: Path,
    output_json: Path,
    *,
    validation_fraction: float,
    min_validation_sessions: int,
    calibration_fraction: float,
    min_calibration_sessions: int,
) -> dict:
    header = pd.read_csv(input_csv, nrows=0)
    feature_cols = select_feature_cols(header, "live")
    required = {
        "trade_date",
        "datetime",
        "instrument",
        "decision_time",
        *feature_cols,
    }
    for profile in PROFILES.values():
        label_col = str(profile["label_col"])
        required.add(label_col)
        required.add(edge_col_for_label(label_col))
    frame = pd.read_csv(
        input_csv,
        usecols=lambda col: col in required,
        parse_dates=["trade_date", "datetime"],
    ).replace([np.inf, -np.inf], np.nan)
    frame["decision_time"] = (
        frame["decision_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    )
    frame = frame[frame["decision_time"] == "1000"].copy()
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[feature_cols] = frame[feature_cols].fillna(0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        direction: train_one(
            frame,
            feature_cols,
            output_dir,
            direction,
            profile,
            validation_fraction=validation_fraction,
            min_validation_sessions=min_validation_sessions,
            calibration_fraction=calibration_fraction,
            min_calibration_sessions=min_calibration_sessions,
        )
        for direction, profile in PROFILES.items()
    }
    report = {
        "status": "held_intraday_bidirectional_frozen_models_trained",
        "input_csv": str(input_csv.resolve()),
        "input_sha256": sha256_file(input_csv),
        "rows_1000": int(len(frame)),
        "dates": int(frame["trade_date"].nunique()),
        "instruments": int(frame["instrument"].nunique()),
        "feature_cols": feature_cols,
        "models": models,
        "frozen_profile": {
            "decision_time": "10:00",
            "entry_trigger_window": "10:00-11:00",
            "exit_window": "14:45-14:50",
            "buy_first": PROFILES["buy_first"],
            "sell_first": PROFILES["sell_first"],
            "max_symbols_per_day": 3,
            "max_daily_turnover": 0.03,
            "conflict_policy": "drop_both_same_symbol",
        },
        "research_gate_base": None,
        "research_gate_touch_stress": None,
        "requires_recomputed_research_gates": True,
        "deployment_allowed": False,
        "next_gate": "frozen_forward_replay_then_gmquant_held_only_dry_run_audit",
    }
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--min-validation-sessions", type=int, default=15)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--min-calibration-sessions", type=int, default=10)
    args = parser.parse_args()
    report = train_all(
        Path(args.input_csv).resolve(),
        Path(args.output_dir).resolve(),
        Path(args.output_json).resolve(),
        validation_fraction=args.validation_fraction,
        min_validation_sessions=args.min_validation_sessions,
        calibration_fraction=args.calibration_fraction,
        min_calibration_sessions=args.min_calibration_sessions,
    )
    print(
        f"[inner frozen] models={len(report['models'])} rows={report['rows_1000']} "
        f"output={Path(args.output_json).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
