from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


HERE = Path(__file__).resolve().parent
MULTI_LAYER = HERE.parent
if str(MULTI_LAYER) not in sys.path:
    sys.path.insert(0, str(MULTI_LAYER))

from catboost_densemble import CatBoostDEnsemble  # noqa: E402
from evaluate_held_intraday_decision_model import metrics, select_feature_cols  # noqa: E402
from run_held_intraday_anchored_oof import (  # noqa: E402
    anchored_train_validation_calibration_dates,
    edge_col_for_label,
    normalize_decision_time,
    parse_dates,
)


class FrameDatasetAdapter:
    def __init__(self, segments: dict[str, pd.DataFrame]):
        self.segments = segments

    def prepare(self, segment, col_set=None, data_key=None):
        del data_key
        if isinstance(segment, (list, tuple)):
            return tuple(self.prepare(item, col_set=col_set) for item in segment)
        table = self.segments[str(segment)]
        if col_set is None:
            return table
        wanted = [col_set] if isinstance(col_set, str) else list(col_set)
        mask = table.columns.get_level_values(0).isin(wanted)
        selected = table.loc[:, mask]
        if isinstance(col_set, str):
            return selected[col_set]
        return selected


def qlib_table(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    feature = frame[feature_cols].copy()
    feature.columns = pd.MultiIndex.from_product([["feature"], feature_cols])
    label = pd.DataFrame(
        frame["model_target"].to_numpy(dtype=float),
        index=frame.index,
        columns=pd.MultiIndex.from_tuples([("label", "label")]),
    )
    return pd.concat([feature, label], axis=1)


def ensemble_probability(model: CatBoostDEnsemble, frame: pd.DataFrame) -> np.ndarray:
    total = np.zeros(len(frame), dtype=float)
    weight_sum = 0.0
    for submodel, features, weight in zip(
        model.ensemble,
        model.sub_features,
        model.sub_weights,
    ):
        probability = submodel.predict(
            frame.loc[:, features].to_numpy(),
            prediction_type="Probability",
        )
        total += np.asarray(probability, dtype=float)[:, 1] * float(weight)
        weight_sum += float(weight)
    if weight_sum <= 0:
        raise ValueError("DEnsemble submodel weight sum must be positive")
    return total / weight_sum


def train_predict_fold(
    frame: pd.DataFrame,
    feature_cols: list[str],
    edge_col: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fold_id: int,
    *,
    validation_fraction: float,
    min_validation_sessions: int,
    calibration_fraction: float,
    min_calibration_sessions: int,
    purge_sessions: int,
    num_models: int,
    epochs: int,
    enable_sr: bool,
    enable_fs: bool,
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
    fit = frame[frame["trade_date"].isin(fit_dates)].copy()
    validation = frame[frame["trade_date"].isin(validation_dates)].copy()
    calibration = frame[frame["trade_date"].isin(calibration_dates)].copy()
    test = frame[(frame["trade_date"] >= test_start) & (frame["trade_date"] <= test_end)].copy()
    if min(len(fit), len(validation), len(calibration), len(test)) < 100:
        raise ValueError(
            f"fold {fold_id} has too few rows fit={len(fit)} validation={len(validation)} "
            f"calibration={len(calibration)} test={len(test)}"
        )
    dataset = FrameDatasetAdapter(
        {
            "train": qlib_table(fit, feature_cols),
            "valid": qlib_table(validation, feature_cols),
        }
    )
    model = CatBoostDEnsemble(
        loss="Logloss",
        binary_threshold=0.5,
        num_models=num_models,
        enable_sr=enable_sr,
        enable_fs=enable_fs,
        ensemble_eval_rows=None,
        random_seed=4200 + fold_id,
        depth=4,
        learning_rate=0.045,
        l2_leaf_reg=8.0,
        eval_metric="AUC",
        allow_writing_files=False,
        task_type="CPU",
        verbose=max(10, epochs // 4),
    )
    model.fit(dataset, epochs=epochs)
    calibration_raw = ensemble_probability(model, calibration[feature_cols])
    test["raw_score"] = ensemble_probability(model, test[feature_cols])
    if score_calibration == "percentile":
        calibration_values = np.sort(calibration_raw[np.isfinite(calibration_raw)])
        if len(calibration_values) < 100:
            raise ValueError(f"fold {fold_id} has too few calibration predictions")
        test["score"] = np.searchsorted(
            calibration_values,
            test["raw_score"].to_numpy(dtype=float),
            side="right",
        ) / float(len(calibration_values))
    elif score_calibration == "isotonic":
        calibration_target = pd.to_numeric(calibration[edge_col], errors="coerce").to_numpy(
            dtype=float
        )
        finite = np.isfinite(calibration_raw) & np.isfinite(calibration_target)
        if finite.sum() < 100 or len(np.unique(calibration_raw[finite])) < 2:
            raise ValueError(f"fold {fold_id} has insufficient isotonic calibration rows")
        if calibration_target_clip > 0:
            calibration_target = np.clip(
                calibration_target,
                -calibration_target_clip,
                calibration_target_clip,
            )
        calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
        calibrator.fit(calibration_raw[finite], calibration_target[finite])
        test["score"] = calibrator.predict(test["raw_score"].to_numpy(dtype=float))
    else:
        raise ValueError(f"unsupported score calibration: {score_calibration}")
    test["label"] = (test[edge_col] > 0.0).astype(int)
    test["fold"] = fold_id
    fold_metrics = metrics(test, edge_col=edge_col)
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
            "purged_dates": [str(date.date()) for date in purged_dates],
            "submodel_count": int(len(model.ensemble)),
            "submodel_feature_counts": [int(len(features)) for features in model.sub_features],
            "submodel_best_iterations": [
                int(submodel.get_best_iteration()) for submodel in model.ensemble
            ],
            "score_calibration": score_calibration,
        }
    )
    return test, fold_metrics


def evaluate(
    input_csv: Path,
    output_json: Path,
    output_predictions: Path,
    *,
    decision_time: str,
    label_col: str,
    edge_col: str | None,
    test_starts: list[pd.Timestamp],
    feature_mode: str,
    validation_fraction: float,
    min_validation_sessions: int,
    calibration_fraction: float,
    min_calibration_sessions: int,
    purge_sessions: int,
    num_models: int,
    epochs: int,
    enable_sr: bool,
    enable_fs: bool,
    score_calibration: str,
    calibration_target_clip: float,
    append_csvs: tuple[Path, ...] = (),
) -> dict:
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
    edge_col = edge_col or edge_col_for_label(label_col)
    for col in (label_col, edge_col):
        if col not in frame.columns:
            raise KeyError(f"target column not found: {col}")
    frame = frame.dropna(subset=[label_col, edge_col]).copy()
    frame["model_target"] = (frame[label_col] > 0.5).astype(int)
    feature_cols = select_feature_cols(frame, feature_mode)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[feature_cols] = frame[feature_cols].fillna(0.0)

    starts = sorted(test_starts)
    max_date = frame["trade_date"].max()
    predictions = []
    folds = []
    for idx, start in enumerate(starts, start=1):
        end = starts[idx] - pd.Timedelta(days=1) if idx < len(starts) else max_date
        prediction, fold = train_predict_fold(
            frame,
            feature_cols,
            edge_col,
            start,
            end,
            idx,
            validation_fraction=validation_fraction,
            min_validation_sessions=min_validation_sessions,
            calibration_fraction=calibration_fraction,
            min_calibration_sessions=min_calibration_sessions,
            purge_sessions=purge_sessions,
            num_models=num_models,
            epochs=epochs,
            enable_sr=enable_sr,
            enable_fs=enable_fs,
            score_calibration=score_calibration,
            calibration_target_clip=calibration_target_clip,
        )
        predictions.append(prediction)
        folds.append(fold)
        print(
            f"[held DEnsemble] fold={idx} rows={fold['rows']} "
            f"auc={fold.get('auc')} spearman={fold.get('spearman_edge')}",
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
        label_col,
        edge_col,
        "label",
        "raw_score",
        "score",
    ]
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    pred_all[pred_cols].to_csv(output_predictions, index=False, encoding="utf-8-sig")
    report = {
        "status": "held_intraday_catboost_densemble_oof_evaluated",
        "input_csv": str(input_csv.resolve()),
        "input_parts": input_parts,
        "output_predictions": str(output_predictions.resolve()),
        "decision_time": wanted,
        "label_col": label_col,
        "edge_col": edge_col,
        "feature_mode": feature_mode,
        "feature_cols": feature_cols,
        "model": {
            "class": "CatBoostDEnsemble",
            "num_models": num_models,
            "epochs": epochs,
            "enable_sr": enable_sr,
            "enable_fs": enable_fs,
            "score_calibration": score_calibration,
            "calibration_target_clip": calibration_target_clip,
        },
        "split_policy": {
            "anchored_walk_forward": True,
            "test_used_for_early_stopping": False,
            "validation_fraction": validation_fraction,
            "calibration_fraction": calibration_fraction,
            "purge_sessions": purge_sessions,
        },
        "folds": folds,
        "overall": metrics(pred_all, edge_col=edge_col),
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument(
        "--append-csv",
        action="append",
        default=[],
        help="Append a strictly later, disjoint dataset without creating a merged CSV.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--decision-time", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--edge-col", default=None)
    parser.add_argument(
        "--test-starts",
        default="2025-06-03,2025-07-22,2025-09-09,2025-10-29,2025-12-17,2026-02-05",
    )
    parser.add_argument(
        "--feature-mode",
        choices=[
            "live",
            "live_core",
            "live_limit",
            "live_industry",
            "live_compact_core",
            "live_compact_limit",
        ],
        default="live_limit",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--min-validation-sessions", type=int, default=15)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--min-calibration-sessions", type=int, default=10)
    parser.add_argument("--purge-sessions", type=int, default=1)
    parser.add_argument("--num-models", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument(
        "--score-calibration",
        choices=["percentile", "isotonic"],
        default="percentile",
    )
    parser.add_argument("--calibration-target-clip", type=float, default=0.05)
    parser.add_argument("--disable-sr", action="store_true")
    parser.add_argument("--disable-fs", action="store_true")
    args = parser.parse_args()
    report = evaluate(
        Path(args.input_csv).resolve(),
        Path(args.output_json).resolve(),
        Path(args.output_predictions).resolve(),
        decision_time=args.decision_time,
        label_col=args.label_col,
        edge_col=args.edge_col,
        test_starts=parse_dates(args.test_starts),
        feature_mode=args.feature_mode,
        validation_fraction=args.validation_fraction,
        min_validation_sessions=args.min_validation_sessions,
        calibration_fraction=args.calibration_fraction,
        min_calibration_sessions=args.min_calibration_sessions,
        purge_sessions=args.purge_sessions,
        num_models=args.num_models,
        epochs=args.epochs,
        enable_sr=not args.disable_sr,
        enable_fs=not args.disable_fs,
        score_calibration=args.score_calibration,
        calibration_target_clip=args.calibration_target_clip,
        append_csvs=tuple(Path(path).resolve() for path in args.append_csv),
    )
    overall = report["overall"]
    print(
        f"[held DEnsemble] rows={overall['rows']} auc={overall.get('auc')} "
        f"spearman={overall.get('spearman_edge')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
