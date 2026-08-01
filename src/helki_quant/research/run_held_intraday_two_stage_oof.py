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


def expected_value_from_win_probability(
    probability: np.ndarray,
    positive_mean: float,
    negative_mean: float,
) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    return probability * float(positive_mean) + (1.0 - probability) * float(negative_mean)


def calibrate_probability(
    calibration_score: np.ndarray,
    calibration_label: np.ndarray,
    test_score: np.ndarray,
    *,
    mode: str,
    stage: str,
) -> tuple[np.ndarray, dict]:
    raw = np.asarray(calibration_score, dtype=float)
    label = np.asarray(calibration_label, dtype=int)
    test = np.asarray(test_score, dtype=float)
    finite = np.isfinite(raw) & np.isfinite(label)
    raw = raw[finite]
    label = label[finite]
    if mode == "none":
        return test, {"mode": "none", "rows": int(len(raw))}
    if mode != "isotonic":
        raise ValueError(f"unsupported probability calibration: {mode}")
    if len(raw) < 100 or len(np.unique(raw)) < 2 or len(np.unique(label)) < 2:
        raise ValueError(
            f"{stage} calibration is insufficient: rows={len(raw)} "
            f"scores={len(np.unique(raw))} labels={len(np.unique(label))}"
        )
    calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
    calibrator.fit(raw, label)
    calibrated = calibrator.predict(test)
    return calibrated, {
        "mode": "isotonic",
        "rows": int(len(raw)),
        "positive_ratio": float(label.mean()),
        "raw_auc": auc_score(label, raw),
    }


def _classifier(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=180,
        depth=4,
        learning_rate=0.045,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        task_type="CPU",
        allow_writing_files=False,
        verbose=50,
    )


def _regressor(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=240,
        depth=4,
        learning_rate=0.04,
        l2_leaf_reg=10.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        task_type="CPU",
        allow_writing_files=False,
        verbose=50,
    )


def _fit_classifier(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    *,
    seed: int,
) -> CatBoostClassifier:
    for name, part in (("train", train), ("validation", validation)):
        if part[label_col].nunique() < 2:
            raise ValueError(f"{name} {label_col} has fewer than two classes")
    model = _classifier(seed)
    model.fit(
        Pool(train[feature_cols], label=train[label_col]),
        eval_set=Pool(validation[feature_cols], label=validation[label_col]),
        use_best_model=True,
        early_stopping_rounds=30,
    )
    return model


def _fit_regressor(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    seed: int,
) -> CatBoostRegressor:
    model = _regressor(seed)
    model.fit(
        Pool(train[feature_cols], label=train[target_col]),
        eval_set=Pool(validation[feature_cols], label=validation[target_col]),
        use_best_model=True,
        early_stopping_rounds=30,
    )
    return model


def _stage_columns(label_prefix: str, conditional_target: str) -> dict[str, str]:
    if conditional_target not in {"edge", "pnl"}:
        raise ValueError(f"unsupported conditional target: {conditional_target}")
    return {
        "touched": f"{label_prefix}_touched",
        "conditional_hit": f"{label_prefix}_conditional_hit",
        "conditional_target": f"{label_prefix}_conditional_{conditional_target}",
        "realized_hit": f"{label_prefix}_realized_hit",
        "realized_edge": f"{label_prefix}_realized_edge",
        "realized_pnl": f"{label_prefix}_realized_pnl",
    }


def train_predict_fold(
    frame: pd.DataFrame,
    feature_cols: list[str],
    columns: dict[str, str],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fold_id: int,
    *,
    validation_fraction: float,
    min_validation_sessions: int,
    calibration_fraction: float,
    min_calibration_sessions: int,
    purge_sessions: int,
    conditional_objective: str,
    target_clip: float,
    probability_calibration: str,
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
    fit = frame[frame["trade_date"].isin(fit_dates)].copy()
    validation = frame[frame["trade_date"].isin(validation_dates)].copy()
    calibration = frame[frame["trade_date"].isin(calibration_dates)].copy()
    test = frame[(frame["trade_date"] >= test_start) & (frame["trade_date"] <= test_end)].copy()
    if len(fit) < 500 or len(validation) < 100 or len(calibration) < 100 or len(test) < 100:
        raise ValueError(
            f"fold {fold_id} has too few rows: fit={len(fit)} validation={len(validation)} "
            f"calibration={len(calibration)} test={len(test)}"
        )

    touch_col = columns["touched"]
    conditional_hit_col = columns["conditional_hit"]
    conditional_target_col = columns["conditional_target"]
    touched_fit = fit[(fit[touch_col] > 0.5) & fit[conditional_target_col].notna()].copy()
    touched_validation = validation[
        (validation[touch_col] > 0.5) & validation[conditional_target_col].notna()
    ].copy()
    touched_calibration = calibration[
        (calibration[touch_col] > 0.5) & calibration[conditional_target_col].notna()
    ].copy()
    if min(len(touched_fit), len(touched_validation), len(touched_calibration)) < 100:
        raise ValueError(
            f"fold {fold_id} has too few touched rows: fit={len(touched_fit)} "
            f"validation={len(touched_validation)} calibration={len(touched_calibration)}"
        )

    touch_model = _fit_classifier(
        fit,
        validation,
        feature_cols,
        touch_col,
        seed=4200 + fold_id,
    )
    calibration_touch_raw = touch_model.predict_proba(calibration[feature_cols])[:, 1]
    test_touch_raw = touch_model.predict_proba(test[feature_cols])[:, 1]
    test["touch_probability"], touch_calibration = calibrate_probability(
        calibration_touch_raw,
        calibration[touch_col].to_numpy(dtype=int),
        test_touch_raw,
        mode=probability_calibration,
        stage="touch",
    )

    if target_clip > 0:
        for part in (touched_fit, touched_validation):
            part["conditional_model_target"] = part[conditional_target_col].clip(
                -target_clip,
                target_clip,
            )
    else:
        for part in (touched_fit, touched_validation):
            part["conditional_model_target"] = part[conditional_target_col]

    conditional_calibration: dict
    if conditional_objective == "regression":
        conditional_model = _fit_regressor(
            touched_fit,
            touched_validation,
            feature_cols,
            "conditional_model_target",
            seed=4300 + fold_id,
        )
        test["conditional_expected_value"] = conditional_model.predict(test[feature_cols])
        test["conditional_probability"] = np.nan
        conditional_calibration = {"mode": "not_applicable"}
        conditional_best_iteration = conditional_model.get_best_iteration()
    elif conditional_objective == "classification":
        conditional_model = _fit_classifier(
            touched_fit,
            touched_validation,
            feature_cols,
            conditional_hit_col,
            seed=4400 + fold_id,
        )
        calibration_conditional_raw = conditional_model.predict_proba(
            touched_calibration[feature_cols]
        )[:, 1]
        test_conditional_raw = conditional_model.predict_proba(test[feature_cols])[:, 1]
        test["conditional_probability"], conditional_calibration = calibrate_probability(
            calibration_conditional_raw,
            touched_calibration[conditional_hit_col].to_numpy(dtype=int),
            test_conditional_raw,
            mode=probability_calibration,
            stage="conditional",
        )
        fit_target = touched_fit["conditional_model_target"]
        fit_positive = fit_target[touched_fit[conditional_hit_col] > 0.5]
        fit_negative = fit_target[touched_fit[conditional_hit_col] <= 0.5]
        if fit_positive.empty or fit_negative.empty:
            raise ValueError(f"fold {fold_id} conditional classes lack target values")
        positive_mean = float(fit_positive.mean())
        negative_mean = float(fit_negative.mean())
        test["conditional_expected_value"] = expected_value_from_win_probability(
            test["conditional_probability"].to_numpy(),
            positive_mean,
            negative_mean,
        )
        conditional_calibration.update(
            {
                "positive_target_mean": positive_mean,
                "negative_target_mean": negative_mean,
            }
        )
        conditional_best_iteration = conditional_model.get_best_iteration()
    else:
        raise ValueError(f"unsupported conditional objective: {conditional_objective}")

    test["raw_score"] = test["touch_probability"] * test["conditional_expected_value"]
    test["score"] = test["raw_score"]
    test["label"] = (test[columns["realized_edge"]] > 0.0).astype(int)
    test["fold"] = fold_id
    fold_metrics = metrics(test, edge_col=columns["realized_edge"])
    touched_test = test[test[touch_col] > 0.5]
    fold_metrics.update(
        {
            "fold": fold_id,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "train_rows": int(len(fit)),
            "train_dates": int(len(fit_dates)),
            "validation_rows": int(len(validation)),
            "validation_dates": int(len(validation_dates)),
            "calibration_rows": int(len(calibration)),
            "calibration_dates": int(len(calibration_dates)),
            "touched_train_rows": int(len(touched_fit)),
            "touched_validation_rows": int(len(touched_validation)),
            "touched_calibration_rows": int(len(touched_calibration)),
            "touch_auc": auc_score(test[touch_col].to_numpy(), test["touch_probability"].to_numpy()),
            "conditional_spearman": float(
                touched_test["conditional_expected_value"].corr(
                    touched_test[conditional_target_col], method="spearman"
                )
            ),
            "score_positive_rows": int((test["score"] > 0.0).sum()),
            "score_positive_dates": int(test.loc[test["score"] > 0.0, "trade_date"].nunique()),
            "score_positive_edge_mean": float(
                test.loc[test["score"] > 0.0, columns["realized_edge"]].mean()
            )
            if (test["score"] > 0.0).any()
            else None,
            "touch_best_iteration": touch_model.get_best_iteration(),
            "conditional_best_iteration": conditional_best_iteration,
            "touch_calibration": touch_calibration,
            "conditional_calibration": conditional_calibration,
            "purged_dates": [str(date.date()) for date in purged_dates],
        }
    )
    return test, fold_metrics


def evaluate(
    input_csv: Path,
    output_json: Path,
    output_predictions: Path,
    *,
    decision_time: str,
    label_prefix: str,
    test_starts: list[pd.Timestamp],
    feature_mode: str,
    conditional_objective: str,
    conditional_target: str,
    target_clip: float,
    probability_calibration: str,
    validation_fraction: float,
    min_validation_sessions: int,
    calibration_fraction: float,
    min_calibration_sessions: int,
    purge_sessions: int,
) -> dict:
    frame = pd.read_csv(input_csv, parse_dates=["trade_date", "datetime"]).replace(
        [np.inf, -np.inf], np.nan
    )
    frame["trade_date"] = frame["trade_date"].dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame["decision_time"] = normalize_decision_time(frame["decision_time"])
    wanted = str(decision_time).zfill(4)
    frame = frame[frame["decision_time"] == wanted].copy()
    columns = _stage_columns(label_prefix, conditional_target)
    for name, col in columns.items():
        if col not in frame.columns:
            raise KeyError(f"{name} column not found: {col}")
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(
        subset=[columns["touched"], columns["realized_hit"], columns["realized_edge"]]
    ).copy()
    feature_cols = select_feature_cols(frame, feature_mode)
    if not feature_cols:
        raise ValueError("no model features selected")
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[feature_cols] = frame[feature_cols].fillna(0.0)

    starts = sorted(test_starts)
    max_date = frame["trade_date"].max()
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    for idx, start in enumerate(starts, start=1):
        end = starts[idx] - pd.Timedelta(days=1) if idx < len(starts) else max_date
        prediction, fold = train_predict_fold(
            frame,
            feature_cols,
            columns,
            start,
            end,
            idx,
            validation_fraction=validation_fraction,
            min_validation_sessions=min_validation_sessions,
            calibration_fraction=calibration_fraction,
            min_calibration_sessions=min_calibration_sessions,
            purge_sessions=purge_sessions,
            conditional_objective=conditional_objective,
            target_clip=target_clip,
            probability_calibration=probability_calibration,
        )
        predictions.append(prediction)
        fold_rows.append(fold)
        print(
            f"[held two-stage] fold={idx} rows={fold['rows']} "
            f"touch_auc={fold['touch_auc']} spearman={fold['spearman_edge']} "
            f"positive={fold['score_positive_rows']}",
            flush=True,
        )

    pred_all = pd.concat(predictions, ignore_index=True).sort_values(
        ["trade_date", "instrument"]
    )
    pred_cols = [
        "datetime",
        "trade_date",
        "instrument",
        "decision_time",
        "fold",
        columns["touched"],
        columns["conditional_hit"],
        columns["conditional_target"],
        columns["realized_hit"],
        columns["realized_edge"],
        columns["realized_pnl"],
        "label",
        "touch_probability",
        "conditional_probability",
        "conditional_expected_value",
        "raw_score",
        "score",
    ]
    pred_cols = list(dict.fromkeys(col for col in pred_cols if col in pred_all.columns))
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    pred_all[pred_cols].to_csv(output_predictions, index=False, encoding="utf-8-sig")
    overall = metrics(pred_all, edge_col=columns["realized_edge"])
    report = {
        "status": "held_intraday_two_stage_oof_evaluated",
        "input_csv": str(input_csv.resolve()),
        "output_predictions": str(output_predictions.resolve()),
        "decision_time": wanted,
        "label_prefix": label_prefix,
        "columns": columns,
        "conditional_objective": conditional_objective,
        "conditional_target": conditional_target,
        "target_clip": target_clip,
        "probability_calibration": probability_calibration,
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
        },
        "folds": fold_rows,
        "overall": overall,
        "deployment_allowed": False,
        "research_only_reason": "Two-stage held-only T+0 model has not passed portfolio gates.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--decision-time", required=True)
    parser.add_argument("--label-prefix", required=True)
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
        default="live",
    )
    parser.add_argument(
        "--conditional-objective",
        choices=["regression", "classification"],
        default="regression",
    )
    parser.add_argument("--conditional-target", choices=["edge", "pnl"], default="edge")
    parser.add_argument("--target-clip", type=float, default=0.05)
    parser.add_argument(
        "--probability-calibration", choices=["none", "isotonic"], default="none"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--min-validation-sessions", type=int, default=15)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--min-calibration-sessions", type=int, default=10)
    parser.add_argument("--purge-sessions", type=int, default=1)
    args = parser.parse_args()
    report = evaluate(
        Path(args.input_csv).resolve(),
        Path(args.output_json).resolve(),
        Path(args.output_predictions).resolve(),
        decision_time=args.decision_time,
        label_prefix=args.label_prefix,
        test_starts=parse_dates(args.test_starts),
        feature_mode=args.feature_mode,
        conditional_objective=args.conditional_objective,
        conditional_target=args.conditional_target,
        target_clip=args.target_clip,
        probability_calibration=args.probability_calibration,
        validation_fraction=args.validation_fraction,
        min_validation_sessions=args.min_validation_sessions,
        calibration_fraction=args.calibration_fraction,
        min_calibration_sessions=args.min_calibration_sessions,
        purge_sessions=args.purge_sessions,
    )
    overall = report["overall"]
    print(
        f"[held two-stage] rows={overall['rows']} auc={overall.get('auc')} "
        f"spearman={overall.get('spearman_edge')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
