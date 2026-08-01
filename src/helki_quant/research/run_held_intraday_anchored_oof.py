from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool
from sklearn.isotonic import IsotonicRegression

from evaluate_held_intraday_decision_model import auc_score, metrics, select_feature_cols


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_dates(raw: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(item.strip()).normalize() for item in raw.split(",") if item.strip()]


def normalize_decision_time(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)


def edge_col_for_label(label_col: str) -> str:
    if label_col.startswith("trigger_") and label_col.endswith("_realized_hit"):
        return f"{label_col[:-13]}_realized_edge"
    if label_col.startswith("trigger_") and label_col.endswith("_conditional_hit"):
        return f"{label_col[:-16]}_conditional_edge"
    if label_col.startswith(("t0_exec_", "t0_buy_first_")) and label_col.endswith("_hit"):
        return f"{label_col[:-4]}_edge"
    if label_col.startswith("t0_hit_"):
        return label_col.replace("t0_hit_", "t0_edge_", 1)
    if label_col == "t0_best_hit":
        return "t0_best_edge"
    raise ValueError(f"cannot infer edge column from label: {label_col}")


def classification_target(
    frame: pd.DataFrame,
    *,
    label_col: str,
    edge_col: str,
    positive_edge_threshold: float,
) -> pd.Series:
    """Build a future-only classification target with an optional net-edge buffer."""
    if positive_edge_threshold < 0:
        raise ValueError("classification positive edge threshold must be >= 0")
    if positive_edge_threshold > 0:
        edge = pd.to_numeric(frame[edge_col], errors="coerce")
        return (edge > float(positive_edge_threshold)).astype(int)
    label = pd.to_numeric(frame[label_col], errors="coerce")
    return (label > 0.5).astype(int)


def time_decay_session_weights(
    trade_dates: pd.Series,
    half_life_sessions: float,
) -> pd.Series:
    """Return causal recency weights indexed by completed training sessions."""
    if half_life_sessions < 0:
        raise ValueError("time decay half-life sessions must be >= 0")
    if half_life_sessions == 0:
        return pd.Series(1.0, index=trade_dates.index, dtype=float)
    normalized = pd.to_datetime(trade_dates, errors="coerce").dt.normalize()
    if normalized.isna().any():
        raise ValueError("time decay received invalid trade dates")
    sessions = pd.DatetimeIndex(normalized.unique()).sort_values()
    session_number = {date: pos for pos, date in enumerate(sessions)}
    age = normalized.map(lambda date: len(sessions) - 1 - session_number[date]).astype(float)
    return np.exp(-np.log(2.0) * age / float(half_life_sessions))


def anchored_train_validation_dates(
    frame: pd.DataFrame,
    test_start: pd.Timestamp,
    *,
    validation_fraction: float,
    min_validation_sessions: int,
    purge_sessions: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    dates = pd.DatetimeIndex(
        frame.loc[frame["trade_date"] < test_start, "trade_date"].dropna().unique()
    ).sort_values()
    if purge_sessions < 0:
        raise ValueError("purge_sessions must be >= 0")
    if purge_sessions:
        purged = dates[-purge_sessions:]
        dates = dates[:-purge_sessions]
    else:
        purged = pd.DatetimeIndex([])
    validation_sessions = max(
        int(min_validation_sessions),
        int(np.ceil(len(dates) * float(validation_fraction))),
    )
    if validation_sessions >= len(dates) - 5:
        raise ValueError(
            f"not enough pre-test sessions for fit/validation: dates={len(dates)} "
            f"validation={validation_sessions}"
        )
    return dates[:-validation_sessions], dates[-validation_sessions:], purged


def anchored_train_validation_calibration_dates(
    frame: pd.DataFrame,
    test_start: pd.Timestamp,
    *,
    validation_fraction: float,
    min_validation_sessions: int,
    calibration_fraction: float,
    min_calibration_sessions: int,
    purge_sessions: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    dates = pd.DatetimeIndex(
        frame.loc[frame["trade_date"] < test_start, "trade_date"].dropna().unique()
    ).sort_values()
    if purge_sessions < 0:
        raise ValueError("purge_sessions must be >= 0")
    if purge_sessions:
        purged = dates[-purge_sessions:]
        dates = dates[:-purge_sessions]
    else:
        purged = pd.DatetimeIndex([])
    calibration_sessions = max(
        int(min_calibration_sessions),
        int(np.ceil(len(dates) * float(calibration_fraction))),
    )
    calibration_dates = dates[-calibration_sessions:]
    remaining = dates[:-calibration_sessions]
    validation_sessions = max(
        int(min_validation_sessions),
        int(np.ceil(len(remaining) * float(validation_fraction))),
    )
    if validation_sessions >= len(remaining) - 5:
        raise ValueError(
            f"not enough pre-test sessions for fit/validation/calibration: dates={len(dates)} "
            f"validation={validation_sessions} calibration={calibration_sessions}"
        )
    return (
        remaining[:-validation_sessions],
        remaining[-validation_sessions:],
        calibration_dates,
        purged,
    )


def train_predict_fold(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fold_id: int,
    validation_fraction: float,
    min_validation_sessions: int,
    purge_sessions: int,
    objective: str,
    edge_col: str,
    score_calibration: str,
    calibration_fraction: float,
    min_calibration_sessions: int,
    calibration_target_clip: float,
    sample_weight_mode: str,
    ranking_target_mode: str,
    time_decay_half_life_sessions: float,
    artifact_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    if score_calibration in {"isotonic", "percentile"}:
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
    else:
        fit_dates, validation_dates, purged_dates = anchored_train_validation_dates(
            frame,
            test_start,
            validation_fraction=validation_fraction,
            min_validation_sessions=min_validation_sessions,
            purge_sessions=purge_sessions,
        )
        calibration_dates = pd.DatetimeIndex([])
    train = frame[frame["trade_date"].isin(fit_dates)].copy()
    validation = frame[frame["trade_date"].isin(validation_dates)].copy()
    calibration = frame[frame["trade_date"].isin(calibration_dates)].copy()
    test = frame[(frame["trade_date"] >= test_start) & (frame["trade_date"] <= test_end)].copy()
    if len(train) < 500 or len(validation) < 100 or len(test) < 100:
        raise ValueError(
            f"fold {fold_id} has too few rows: train={len(train)} "
            f"validation={len(validation)} test={len(test)} "
            f"start={test_start.date()} end={test_end.date()}"
        )
    common = {
        "iterations": 240 if objective in {"regression", "ranking"} else 160,
        "depth": 4,
        "learning_rate": 0.04 if objective == "regression" else 0.045,
        "l2_leaf_reg": 10.0 if objective == "regression" else 8.0,
        "random_seed": 42 + fold_id,
        "task_type": "CPU",
        "allow_writing_files": False,
        "verbose": 50,
    }
    if objective == "regression":
        model = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", **common)
    elif objective == "ranking":
        model = CatBoostRanker(
            loss_function="YetiRankPairwise",
            eval_metric="NDCG:top=5",
            **common,
        )
    else:
        eval_metric = "Logloss" if sample_weight_mode != "none" else "AUC"
        model = CatBoostClassifier(loss_function="Logloss", eval_metric=eval_metric, **common)
    decay_weight = time_decay_session_weights(
        train["trade_date"],
        time_decay_half_life_sessions,
    )
    train_weight = train["model_weight"] * decay_weight
    if sample_weight_mode == "none" and time_decay_half_life_sessions == 0:
        train_weight = None
    validation_weight = validation["model_weight"] if sample_weight_mode != "none" else None
    train_group = pd.factorize(train["trade_date"], sort=True)[0] if objective == "ranking" else None
    validation_group = (
        pd.factorize(validation["trade_date"], sort=True)[0] if objective == "ranking" else None
    )
    model.fit(
        Pool(
            train[feature_cols],
            label=train["model_target"],
            weight=train_weight,
            group_id=train_group,
        ),
        eval_set=Pool(
            validation[feature_cols],
            label=validation["model_target"],
            weight=validation_weight,
            group_id=validation_group,
        ),
        use_best_model=True,
        early_stopping_rounds=30,
    )
    if objective in {"regression", "ranking"}:
        test["raw_score"] = model.predict(test[feature_cols])
        calibration_raw = model.predict(calibration[feature_cols]) if not calibration.empty else None
    else:
        test["raw_score"] = model.predict_proba(test[feature_cols])[:, 1]
        calibration_raw = (
            model.predict_proba(calibration[feature_cols])[:, 1] if not calibration.empty else None
        )
    calibrator = None
    calibration_values = None
    if score_calibration == "isotonic":
        if calibration_raw is None or len(np.unique(calibration_raw)) < 2:
            raise ValueError(f"fold {fold_id} has insufficient calibration score variation")
        calibration_target = pd.to_numeric(calibration[edge_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(calibration_raw) & np.isfinite(calibration_target)
        if finite.sum() < 100:
            raise ValueError(f"fold {fold_id} has too few finite calibration rows: {finite.sum()}")
        if calibration_target_clip > 0:
            calibration_target = np.clip(
                calibration_target,
                -calibration_target_clip,
                calibration_target_clip,
            )
        calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
        calibrator.fit(np.asarray(calibration_raw)[finite], calibration_target[finite])
        test["score"] = calibrator.predict(test["raw_score"].to_numpy(dtype=float))
    elif score_calibration == "percentile":
        if calibration_raw is None:
            raise ValueError(f"fold {fold_id} has no calibration scores")
        calibration_values = np.asarray(calibration_raw, dtype=float)
        calibration_values = np.sort(calibration_values[np.isfinite(calibration_values)])
        if len(calibration_values) < 100:
            raise ValueError(
                f"fold {fold_id} has too few finite calibration scores: {len(calibration_values)}"
            )
        test["score"] = np.searchsorted(
            calibration_values,
            test["raw_score"].to_numpy(dtype=float),
            side="right",
        ) / float(len(calibration_values))
    else:
        test["score"] = test["raw_score"]
    test["fold"] = fold_id
    fold_metrics = metrics(test, edge_col=edge_col)
    fold_metrics.update(
        {
            "fold": fold_id,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "train_rows": int(len(train)),
            "train_dates": int(len(fit_dates)),
            "validation_rows": int(len(validation)),
            "validation_dates": int(len(validation_dates)),
            "validation_start": str(validation_dates.min().date()),
            "validation_end": str(validation_dates.max().date()),
            "calibration_rows": int(len(calibration)),
            "calibration_dates": int(len(calibration_dates)),
            "calibration_start": str(calibration_dates.min().date())
            if len(calibration_dates)
            else None,
            "calibration_end": str(calibration_dates.max().date())
            if len(calibration_dates)
            else None,
            "score_calibration": score_calibration,
            "sample_weight_mode": sample_weight_mode,
            "time_decay_half_life_sessions": float(time_decay_half_life_sessions),
            "train_weight_min": float(train_weight.min()) if train_weight is not None else 1.0,
            "train_weight_mean": float(train_weight.mean()) if train_weight is not None else 1.0,
            "train_weight_max": float(train_weight.max()) if train_weight is not None else 1.0,
            "ranking_target_mode": ranking_target_mode if objective == "ranking" else None,
            "purge_sessions": int(len(purged_dates)),
            "purged_dates": [str(date.date()) for date in purged_dates],
            "best_iteration": model.get_best_iteration(),
        }
    )
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / f"fold_{fold_id:02d}_catboost.cbm"
        model.save_model(str(model_path))
        calibration_path = None
        if calibrator is not None:
            calibration_path = artifact_dir / f"fold_{fold_id:02d}_isotonic.npz"
            np.savez(
                calibration_path,
                x_thresholds=np.asarray(calibrator.X_thresholds_, dtype=float),
                y_thresholds=np.asarray(calibrator.y_thresholds_, dtype=float),
            )
        elif calibration_values is not None:
            calibration_path = artifact_dir / f"fold_{fold_id:02d}_percentile.npy"
            np.save(calibration_path, np.asarray(calibration_values, dtype=float))
        metadata = {
            "status": "held_intraday_anchored_fold_model_frozen",
            "fold": int(fold_id),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "label_col": label_col,
            "edge_col": edge_col,
            "objective": objective,
            "feature_cols": feature_cols,
            "score_calibration": score_calibration,
            "model_path": str(model_path.resolve()),
            "model_sha256": sha256_file(model_path),
            "calibration_path": str(calibration_path.resolve()) if calibration_path else None,
            "calibration_sha256": sha256_file(calibration_path) if calibration_path else None,
            "split": fold_metrics,
            "deployment_allowed": False,
            "paper_orders_allowed": False,
        }
        metadata_path = artifact_dir / f"fold_{fold_id:02d}_model_meta.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fold_metrics["frozen_artifacts"] = {
            "model": str(model_path.resolve()),
            "calibration": str(calibration_path.resolve()) if calibration_path else None,
            "metadata": str(metadata_path.resolve()),
        }
    return test, fold_metrics


def evaluate(
    input_csv: Path,
    output_json: Path,
    output_predictions: Path,
    *,
    decision_time: str,
    label_col: str,
    test_starts: list[pd.Timestamp],
    feature_mode: str,
    validation_fraction: float,
    min_validation_sessions: int,
    purge_sessions: int,
    objective: str = "classification",
    target_col: str | None = None,
    edge_col: str | None = None,
    target_clip: float = 0.05,
    score_calibration: str = "none",
    calibration_fraction: float = 0.10,
    min_calibration_sessions: int = 10,
    calibration_target_clip: float = 0.05,
    sample_weight_mode: str = "none",
    ranking_target_mode: str = "shifted_edge",
    classification_positive_edge_threshold: float = 0.0,
    time_decay_half_life_sessions: float = 0.0,
    append_csvs: tuple[Path, ...] = (),
    fold_id_offset: int = 0,
    save_fold_artifacts_dir: Path | None = None,
    save_fold_ids: tuple[int, ...] = (),
) -> dict:
    if time_decay_half_life_sessions < 0:
        raise ValueError("time_decay_half_life_sessions must be >= 0")
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
    if label_col not in frame.columns:
        raise KeyError(f"label column not found: {label_col}")
    edge_col = edge_col or edge_col_for_label(label_col)
    target_col = target_col or (edge_col if objective == "regression" else label_col)
    if objective not in {"classification", "regression", "ranking"}:
        raise ValueError(f"unsupported objective: {objective}")
    for required in (edge_col, target_col):
        if required not in frame.columns:
            raise KeyError(f"target column not found: {required}")
    frame = frame.dropna(subset=[label_col, edge_col, target_col]).copy()
    frame["profit_label"] = (frame[edge_col] > 0.0).astype(int)
    if objective == "regression":
        frame["model_target"] = pd.to_numeric(frame[target_col], errors="coerce")
        if target_clip > 0:
            frame["model_target"] = frame["model_target"].clip(-target_clip, target_clip)
    elif objective == "ranking":
        if ranking_target_mode == "percentile":
            frame["model_target"] = frame.groupby("trade_date", sort=False)[edge_col].rank(
                method="average", pct=True
            )
        elif ranking_target_mode == "shifted_edge":
            clip = target_clip if target_clip > 0 else 0.05
            frame["model_target"] = frame[edge_col].clip(-clip, clip) + clip
        else:
            raise ValueError(f"unsupported ranking_target_mode: {ranking_target_mode}")
    else:
        frame["model_target"] = classification_target(
            frame,
            label_col=label_col,
            edge_col=edge_col,
            positive_edge_threshold=classification_positive_edge_threshold,
        )
    frame["label"] = (
        frame["model_target"].astype(int)
        if objective == "classification"
        else frame["profit_label"]
    )
    edge_values = pd.to_numeric(frame[edge_col], errors="coerce").abs()
    if sample_weight_mode == "none":
        frame["model_weight"] = 1.0
    elif sample_weight_mode == "abs_edge":
        frame["model_weight"] = edge_values.clip(lower=0.002, upper=0.05) / 0.01
    elif sample_weight_mode == "downside_edge":
        downside_multiplier = np.where(frame[edge_col] < 0.0, 1.5, 1.0)
        frame["model_weight"] = (
            edge_values.clip(lower=0.002, upper=0.05) / 0.01 * downside_multiplier
        )
    else:
        raise ValueError(f"unsupported sample_weight_mode: {sample_weight_mode}")
    frame = frame.dropna(subset=["model_target"])
    feature_cols = select_feature_cols(frame, feature_mode)
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[feature_cols] = frame[feature_cols].fillna(0.0)

    starts = sorted(test_starts)
    max_date = frame["trade_date"].max()
    predictions = []
    fold_rows = []
    for idx, start in enumerate(starts, start=1):
        end = (starts[idx] - pd.Timedelta(days=1)) if idx < len(starts) else max_date
        fold_id = idx + int(fold_id_offset)
        artifact_dir = None
        if save_fold_artifacts_dir is not None and (
            not save_fold_ids or fold_id in save_fold_ids
        ):
            artifact_dir = save_fold_artifacts_dir
        pred, row = train_predict_fold(
            frame,
            feature_cols,
            label_col,
            start,
            end,
            fold_id,
            validation_fraction,
            min_validation_sessions,
            purge_sessions,
            objective,
            edge_col,
            score_calibration,
            calibration_fraction,
            min_calibration_sessions,
            calibration_target_clip,
            sample_weight_mode,
            ranking_target_mode,
            time_decay_half_life_sessions,
            artifact_dir,
        )
        predictions.append(pred)
        fold_rows.append(row)
        print(
            f"[held intraday anchored] fold={idx} rows={row['rows']} "
            f"auc={row.get('auc')} spearman={row.get('spearman_edge')}",
            flush=True,
        )
    pred_all = pd.concat(predictions, ignore_index=True).sort_values(["trade_date", "instrument"])
    pred_cols = [
        "datetime",
        "trade_date",
        "instrument",
        "decision_time",
        "fold",
        label_col,
        "label",
        "profit_label",
        target_col,
        edge_col,
        "raw_score",
        "score",
    ]
    pred_cols = list(dict.fromkeys(col for col in pred_cols if col in pred_all.columns))
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    pred_all[pred_cols].to_csv(output_predictions, index=False, encoding="utf-8-sig")
    overall = metrics(pred_all, edge_col=edge_col)
    report = {
        "status": "held_intraday_anchored_oof_evaluated",
        "input_csv": str(input_csv.resolve()),
        "input_parts": input_parts,
        "output_predictions": str(output_predictions.resolve()),
        "decision_time": wanted,
        "label_col": label_col,
        "target_col": target_col,
        "edge_col": edge_col,
        "objective": objective,
        "target_clip": target_clip if objective == "regression" else None,
        "score_calibration": score_calibration,
        "sample_weight_mode": sample_weight_mode,
        "time_decay_half_life_sessions": float(time_decay_half_life_sessions),
        "ranking_target_mode": ranking_target_mode if objective == "ranking" else None,
        "classification_positive_edge_threshold": (
            float(classification_positive_edge_threshold)
            if objective == "classification"
            else None
        ),
        "feature_cols": feature_cols,
        "feature_mode": feature_mode,
        "split_policy": {
            "anchored_walk_forward": True,
            "test_used_for_early_stopping": False,
            "validation_fraction": validation_fraction,
            "min_validation_sessions": min_validation_sessions,
            "purge_sessions": purge_sessions,
            "score_calibration": score_calibration,
            "calibration_fraction": calibration_fraction,
            "min_calibration_sessions": min_calibration_sessions,
            "calibration_target_clip": calibration_target_clip,
            "fold_id_offset": int(fold_id_offset),
        },
        "saved_fold_artifacts_dir": (
            str(save_fold_artifacts_dir.resolve())
            if save_fold_artifacts_dir is not None
            else None
        ),
        "saved_fold_ids": [int(value) for value in save_fold_ids],
        "folds": fold_rows,
        "overall": overall,
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
        default="all",
    )
    parser.add_argument(
        "--objective",
        choices=["classification", "regression", "ranking"],
        default="classification",
    )
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--edge-col", default=None)
    parser.add_argument("--target-clip", type=float, default=0.05)
    parser.add_argument(
        "--classification-positive-edge-threshold",
        type=float,
        default=0.0,
        help="For classification, require realized net edge above this buffer for a positive target.",
    )
    parser.add_argument(
        "--score-calibration",
        choices=["none", "isotonic", "percentile"],
        default="none",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--min-calibration-sessions", type=int, default=10)
    parser.add_argument("--calibration-target-clip", type=float, default=0.05)
    parser.add_argument("--fold-id-offset", type=int, default=0)
    parser.add_argument("--save-fold-artifacts-dir")
    parser.add_argument("--save-fold-id", action="append", type=int, default=[])
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "abs_edge", "downside_edge"],
        default="none",
    )
    parser.add_argument(
        "--time-decay-half-life-sessions",
        type=float,
        default=0.0,
        help="Causal recency half-life applied only to fold training rows; 0 disables it.",
    )
    parser.add_argument(
        "--ranking-target-mode",
        choices=["shifted_edge", "percentile"],
        default="shifted_edge",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--min-validation-sessions", type=int, default=15)
    parser.add_argument("--purge-sessions", type=int, default=1)
    args = parser.parse_args()
    report = evaluate(
        Path(args.input_csv).resolve(),
        Path(args.output_json).resolve(),
        Path(args.output_predictions).resolve(),
        decision_time=args.decision_time,
        label_col=args.label_col,
        test_starts=parse_dates(args.test_starts),
        feature_mode=args.feature_mode,
        validation_fraction=args.validation_fraction,
        min_validation_sessions=args.min_validation_sessions,
        purge_sessions=args.purge_sessions,
        objective=args.objective,
        target_col=args.target_col,
        edge_col=args.edge_col,
        target_clip=args.target_clip,
        score_calibration=args.score_calibration,
        calibration_fraction=args.calibration_fraction,
        min_calibration_sessions=args.min_calibration_sessions,
        calibration_target_clip=args.calibration_target_clip,
        sample_weight_mode=args.sample_weight_mode,
        ranking_target_mode=args.ranking_target_mode,
        classification_positive_edge_threshold=args.classification_positive_edge_threshold,
        time_decay_half_life_sessions=args.time_decay_half_life_sessions,
        append_csvs=tuple(Path(path).resolve() for path in args.append_csv),
        fold_id_offset=args.fold_id_offset,
        save_fold_artifacts_dir=(
            Path(args.save_fold_artifacts_dir).resolve()
            if args.save_fold_artifacts_dir
            else None
        ),
        save_fold_ids=tuple(args.save_fold_id),
    )
    overall = report["overall"]
    print(
        "[held intraday anchored] "
        f"objective={report['objective']} rows={overall['rows']} "
        f"auc={overall.get('auc')} spearman={overall.get('spearman_edge')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
