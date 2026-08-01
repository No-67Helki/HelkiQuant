from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.isotonic import IsotonicRegression

from evaluate_held_intraday_decision_model import auc_score, metrics, select_feature_cols
from run_held_intraday_anchored_oof import (
    anchored_train_validation_calibration_dates,
    normalize_decision_time,
    parse_dates,
)


def expected_hurdle_score(
    touch_probability: np.ndarray,
    conditional_edge: np.ndarray,
) -> np.ndarray:
    probability = np.asarray(touch_probability, dtype=float)
    edge = np.asarray(conditional_edge, dtype=float)
    if probability.shape != edge.shape:
        raise ValueError("touch probability and conditional edge shapes differ")
    return probability * edge


def _load_inputs(input_csv: Path, append_csvs: tuple[Path, ...]) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(input_csv, parse_dates=["trade_date", "datetime"])
    base_max_date = pd.Timestamp(frame["trade_date"].max()).normalize()
    input_parts = [str(input_csv.resolve())]
    for append_csv in append_csvs:
        appended = pd.read_csv(append_csv, parse_dates=["trade_date", "datetime"])
        append_min_date = pd.Timestamp(appended["trade_date"].min()).normalize()
        if append_min_date <= base_max_date:
            raise ValueError(
                "append CSV must be strictly later than the base dataset: "
                f"base_max={base_max_date.date()} append_min={append_min_date.date()} "
                f"path={append_csv}"
            )
        frame = pd.concat([frame, appended], ignore_index=True, sort=False)
        base_max_date = pd.Timestamp(appended["trade_date"].max()).normalize()
        input_parts.append(str(append_csv.resolve()))
    return frame, input_parts


def _fit_fold(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    touch_col: str,
    conditional_edge_col: str,
    realized_edge_col: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fold_id: int,
    validation_fraction: float,
    min_validation_sessions: int,
    calibration_fraction: float,
    min_calibration_sessions: int,
    purge_sessions: int,
    score_calibration: str,
    calibration_target_clip: float,
) -> tuple[pd.DataFrame, dict]:
    fit_dates, validation_dates, calibration_dates, purged_dates = (
        anchored_train_validation_calibration_dates(
            frame,
            test_start,
            validation_fraction=validation_fraction,
            min_validation_sessions=min_validation_sessions,
            calibration_fraction=calibration_fraction,
            min_calibration_sessions=min_calibration_sessions,
            purge_sessions=purge_sessions,
        )
    )
    train = frame[frame["trade_date"].isin(fit_dates)].copy()
    validation = frame[frame["trade_date"].isin(validation_dates)].copy()
    calibration = frame[frame["trade_date"].isin(calibration_dates)].copy()
    test = frame[(frame["trade_date"] >= test_start) & (frame["trade_date"] <= test_end)].copy()
    if len(train) < 500 or len(validation) < 100 or len(calibration) < 100 or len(test) < 100:
        raise ValueError(
            f"fold {fold_id} has too few rows: train={len(train)} validation={len(validation)} "
            f"calibration={len(calibration)} test={len(test)}"
        )

    train_touch = pd.to_numeric(train[touch_col], errors="coerce").astype(int)
    validation_touch = pd.to_numeric(validation[touch_col], errors="coerce").astype(int)
    if train_touch.nunique() < 2 or validation_touch.nunique() < 2:
        raise ValueError(f"fold {fold_id} touch classifier requires both classes")
    touch_model = CatBoostClassifier(
        iterations=180,
        depth=4,
        learning_rate=0.045,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=100 + fold_id,
        task_type="CPU",
        allow_writing_files=False,
        verbose=50,
    )
    print(
        f"[held hurdle] fold={fold_id} touch train={len(train)} validation={len(validation)}",
        flush=True,
    )
    touch_model.fit(
        Pool(train[feature_cols], label=train_touch),
        eval_set=Pool(validation[feature_cols], label=validation_touch),
        use_best_model=True,
        early_stopping_rounds=30,
    )

    train_conditional = train[
        (pd.to_numeric(train[touch_col], errors="coerce") > 0.5)
        & pd.to_numeric(train[conditional_edge_col], errors="coerce").notna()
    ].copy()
    validation_conditional = validation[
        (pd.to_numeric(validation[touch_col], errors="coerce") > 0.5)
        & pd.to_numeric(validation[conditional_edge_col], errors="coerce").notna()
    ].copy()
    if len(train_conditional) < 200 or len(validation_conditional) < 50:
        raise ValueError(
            f"fold {fold_id} has too few touched rows for edge model: "
            f"train={len(train_conditional)} validation={len(validation_conditional)}"
        )
    edge_model = CatBoostRegressor(
        iterations=240,
        depth=4,
        learning_rate=0.04,
        l2_leaf_reg=10.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=200 + fold_id,
        task_type="CPU",
        allow_writing_files=False,
        verbose=50,
    )
    print(
        f"[held hurdle] fold={fold_id} conditional train={len(train_conditional)} "
        f"validation={len(validation_conditional)}",
        flush=True,
    )
    edge_model.fit(
        Pool(
            train_conditional[feature_cols],
            label=train_conditional[conditional_edge_col],
        ),
        eval_set=Pool(
            validation_conditional[feature_cols],
            label=validation_conditional[conditional_edge_col],
        ),
        use_best_model=True,
        early_stopping_rounds=30,
    )

    def component_scores(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        touch_probability = touch_model.predict_proba(part[feature_cols])[:, 1]
        conditional_edge = edge_model.predict(part[feature_cols])
        return (
            touch_probability,
            conditional_edge,
            expected_hurdle_score(touch_probability, conditional_edge),
        )

    test_touch_probability, test_conditional_edge, test_raw_score = component_scores(test)
    test["touch_probability"] = test_touch_probability
    test["conditional_edge_score"] = test_conditional_edge
    test["raw_score"] = test_raw_score
    if score_calibration == "isotonic":
        _, _, calibration_raw = component_scores(calibration)
        calibration_target = pd.to_numeric(
            calibration[realized_edge_col], errors="coerce"
        ).to_numpy(dtype=float)
        finite = np.isfinite(calibration_raw) & np.isfinite(calibration_target)
        if finite.sum() < 100 or len(np.unique(calibration_raw[finite])) < 2:
            raise ValueError(f"fold {fold_id} has insufficient hurdle calibration rows")
        if calibration_target_clip > 0:
            calibration_target = np.clip(
                calibration_target,
                -calibration_target_clip,
                calibration_target_clip,
            )
        calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
        calibrator.fit(calibration_raw[finite], calibration_target[finite])
        test["score"] = calibrator.predict(test_raw_score)
    elif score_calibration == "none":
        test["score"] = test_raw_score
    else:
        raise ValueError(f"unsupported score calibration: {score_calibration}")
    test["fold"] = fold_id
    test["label"] = (pd.to_numeric(test[realized_edge_col], errors="coerce") > 0.0).astype(int)
    test["profit_label"] = test["label"]
    fold_metrics = metrics(test, edge_col=realized_edge_col)
    touched_test = pd.to_numeric(test[touch_col], errors="coerce") > 0.5
    fold_metrics.update(
        {
            "fold": fold_id,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "calibration_rows": int(len(calibration)),
            "test_touched_rows": int(touched_test.sum()),
            "test_touch_ratio": float(touched_test.mean()),
            "touch_auc": auc_score(
                pd.to_numeric(test[touch_col], errors="coerce").to_numpy(dtype=int),
                test_touch_probability,
            ),
            "conditional_edge_spearman": float(
                pd.Series(test_conditional_edge[touched_test.to_numpy()]).corr(
                    pd.to_numeric(
                        test.loc[touched_test, conditional_edge_col], errors="coerce"
                    ).reset_index(drop=True),
                    method="spearman",
                )
            )
            if touched_test.sum() > 2
            else None,
            "fit_dates": int(len(fit_dates)),
            "validation_dates": int(len(validation_dates)),
            "calibration_dates": int(len(calibration_dates)),
            "purged_dates": [str(date.date()) for date in purged_dates],
            "touch_best_iteration": int(touch_model.get_best_iteration()),
            "conditional_edge_best_iteration": int(edge_model.get_best_iteration()),
            "conditional_train_rows": int(len(train_conditional)),
            "conditional_validation_rows": int(len(validation_conditional)),
        }
    )
    return test, fold_metrics


def evaluate(
    input_csv: Path,
    output_json: Path,
    output_predictions: Path,
    *,
    append_csvs: tuple[Path, ...],
    decision_time: str,
    touch_col: str,
    conditional_edge_col: str,
    realized_edge_col: str,
    test_starts: list[pd.Timestamp],
    feature_mode: str,
    validation_fraction: float,
    min_validation_sessions: int,
    calibration_fraction: float,
    min_calibration_sessions: int,
    purge_sessions: int,
    score_calibration: str,
    calibration_target_clip: float,
) -> dict:
    frame, input_parts = _load_inputs(input_csv, append_csvs)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame["trade_date"] = frame["trade_date"].dt.normalize()
    frame["datetime"] = frame["datetime"].dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame["decision_time"] = normalize_decision_time(frame["decision_time"])
    duplicate_keys = ["datetime", "trade_date", "instrument", "decision_time"]
    duplicate_rows = int(frame.duplicated(duplicate_keys, keep=False).sum())
    if duplicate_rows:
        raise ValueError(f"duplicate decision rows after appending datasets: {duplicate_rows}")
    wanted = str(decision_time).zfill(4)
    frame = frame[frame["decision_time"] == wanted].copy()
    required = [touch_col, conditional_edge_col, realized_edge_col]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"hurdle target columns not found: {missing}")
    frame = frame.dropna(subset=[touch_col, realized_edge_col]).copy()
    feature_cols = select_feature_cols(frame, feature_mode)
    for column in feature_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[feature_cols] = frame[feature_cols].fillna(0.0)
    frame["label"] = (pd.to_numeric(frame[realized_edge_col], errors="coerce") > 0).astype(int)

    starts = sorted(test_starts)
    max_date = frame["trade_date"].max()
    predictions = []
    folds = []
    for index, start in enumerate(starts, start=1):
        end = starts[index] - pd.Timedelta(days=1) if index < len(starts) else max_date
        prediction, fold = _fit_fold(
            frame,
            feature_cols,
            touch_col=touch_col,
            conditional_edge_col=conditional_edge_col,
            realized_edge_col=realized_edge_col,
            test_start=start,
            test_end=end,
            fold_id=index,
            validation_fraction=validation_fraction,
            min_validation_sessions=min_validation_sessions,
            calibration_fraction=calibration_fraction,
            min_calibration_sessions=min_calibration_sessions,
            purge_sessions=purge_sessions,
            score_calibration=score_calibration,
            calibration_target_clip=calibration_target_clip,
        )
        predictions.append(prediction)
        folds.append(fold)
        print(
            f"[held hurdle] fold={index} rows={fold['rows']} auc={fold.get('auc')} "
            f"touch_auc={fold.get('touch_auc')} edge_rho={fold.get('spearman_edge')}",
            flush=True,
        )
    prediction_all = pd.concat(predictions, ignore_index=True).sort_values(
        ["trade_date", "instrument"]
    )
    prediction_columns = [
        "datetime",
        "trade_date",
        "instrument",
        "decision_time",
        "fold",
        touch_col,
        conditional_edge_col,
        realized_edge_col,
        "label",
        "profit_label",
        "touch_probability",
        "conditional_edge_score",
        "raw_score",
        "score",
    ]
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    prediction_all[prediction_columns].to_csv(
        output_predictions, index=False, encoding="utf-8-sig"
    )
    overall = metrics(prediction_all, edge_col=realized_edge_col)
    overall["touch_auc"] = auc_score(
        pd.to_numeric(prediction_all[touch_col], errors="coerce").to_numpy(dtype=int),
        prediction_all["touch_probability"].to_numpy(dtype=float),
    )
    touched = pd.to_numeric(prediction_all[touch_col], errors="coerce") > 0.5
    overall["conditional_edge_spearman"] = float(
        prediction_all.loc[touched, "conditional_edge_score"].corr(
            pd.to_numeric(
                prediction_all.loc[touched, conditional_edge_col], errors="coerce"
            ),
            method="spearman",
        )
    )
    report = {
        "status": "held_intraday_hurdle_oof_evaluated",
        "input_csv": str(input_csv.resolve()),
        "input_parts": input_parts,
        "output_predictions": str(output_predictions.resolve()),
        "decision_time": wanted,
        "touch_col": touch_col,
        "conditional_edge_col": conditional_edge_col,
        "realized_edge_col": realized_edge_col,
        "model_structure": "touch_classifier_x_conditional_edge_regressor",
        "score_calibration": score_calibration,
        "feature_mode": feature_mode,
        "feature_cols": feature_cols,
        "split_policy": {
            "anchored_walk_forward": True,
            "test_used_for_early_stopping": False,
            "validation_fraction": validation_fraction,
            "min_validation_sessions": min_validation_sessions,
            "calibration_fraction": calibration_fraction,
            "min_calibration_sessions": min_calibration_sessions,
            "purge_sessions": purge_sessions,
            "calibration_target": realized_edge_col,
            "calibration_target_clip": calibration_target_clip,
        },
        "folds": folds,
        "overall": overall,
        "selection_policy": "pre-registered threshold=0.0 and daily Top-2; no grid search",
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[held hurdle] complete rows={overall['rows']} auc={overall['auc']} "
        f"touch_auc={overall['touch_auc']} edge_rho={overall['spearman_edge']}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--append-csv", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--decision-time", default="1000")
    parser.add_argument("--touch-col", required=True)
    parser.add_argument("--conditional-edge-col", required=True)
    parser.add_argument("--realized-edge-col", required=True)
    parser.add_argument(
        "--test-starts",
        default="2025-06-03,2025-07-22,2025-09-09,2025-10-29,2025-12-17,2026-02-05",
    )
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
        default="live_compact_limit",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--min-validation-sessions", type=int, default=15)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--min-calibration-sessions", type=int, default=10)
    parser.add_argument("--purge-sessions", type=int, default=1)
    parser.add_argument("--score-calibration", choices=["none", "isotonic"], default="isotonic")
    parser.add_argument("--calibration-target-clip", type=float, default=0.05)
    args = parser.parse_args()
    evaluate(
        Path(args.input_csv).resolve(),
        Path(args.output_json).resolve(),
        Path(args.output_predictions).resolve(),
        append_csvs=tuple(Path(path).resolve() for path in args.append_csv),
        decision_time=args.decision_time,
        touch_col=args.touch_col,
        conditional_edge_col=args.conditional_edge_col,
        realized_edge_col=args.realized_edge_col,
        test_starts=parse_dates(args.test_starts),
        feature_mode=args.feature_mode,
        validation_fraction=args.validation_fraction,
        min_validation_sessions=args.min_validation_sessions,
        calibration_fraction=args.calibration_fraction,
        min_calibration_sessions=args.min_calibration_sessions,
        purge_sessions=args.purge_sessions,
        score_calibration=args.score_calibration,
        calibration_target_clip=args.calibration_target_clip,
    )


if __name__ == "__main__":
    main()
