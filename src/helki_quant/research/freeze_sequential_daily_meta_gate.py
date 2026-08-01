from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_sequential_daily_meta_gate import META_FEATURES, build_daily_meta_frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_meta_features(values: Mapping[str, Any], artifact: Mapping[str, Any]) -> float:
    features = list(artifact["meta_features"])
    vector = np.asarray([float(values[name]) for name in features], dtype=float)
    scaler = artifact["scaler"]
    mean = np.asarray(scaler["mean"], dtype=float)
    scale = np.asarray(scaler["scale"], dtype=float)
    coefficient = np.asarray(artifact["ridge"]["coefficient"], dtype=float)
    standardized = (vector - mean) / scale
    return float(standardized @ coefficient + float(artifact["ridge"]["intercept"]))


def freeze_daily_gate(
    predictions_path: Path,
    output_path: Path,
    *,
    ranking_col: str,
    realized_edge_col: str,
    history_start: pd.Timestamp,
    evaluation_start: pd.Timestamp,
    daily_top_n: int,
    ridge_alpha: float,
    minimum_training_dates: int,
    expected_daily_path: Path | None = None,
    parity_tolerance: float = 1e-12,
) -> dict[str, Any]:
    predictions = pd.read_csv(predictions_path, parse_dates=["datetime", "trade_date"])
    predictions["trade_date"] = predictions["trade_date"].dt.normalize()
    predictions["instrument"] = predictions["instrument"].astype(str).str.upper()
    required = {ranking_col, realized_edge_col}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise KeyError(f"daily gate prediction columns missing: {missing}")
    daily = build_daily_meta_frame(
        predictions,
        ranking_col=ranking_col,
        realized_edge_col=realized_edge_col,
        daily_top_n=daily_top_n,
    )
    train = daily[
        (daily["trade_date"] >= history_start)
        & (daily["trade_date"] < evaluation_start)
    ].copy()
    evaluation = daily[daily["trade_date"] >= evaluation_start].copy()
    if len(train) < minimum_training_dates:
        raise ValueError(
            f"too few earlier OOF dates: train={len(train)} minimum={minimum_training_dates}"
        )
    if evaluation.empty:
        raise ValueError("no evaluation dates on or after evaluation_start")
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=ridge_alpha)),
        ]
    )
    pipeline.fit(train[META_FEATURES], train["daily_realized_edge"])
    evaluation["meta_gate_score"] = pipeline.predict(evaluation[META_FEATURES])
    evaluation["meta_gate_enabled"] = evaluation["meta_gate_score"] > 0.0
    scaler = pipeline.named_steps["scale"]
    ridge = pipeline.named_steps["ridge"]
    report: dict[str, Any] = {
        "status": "sequential_daily_meta_gate_frozen",
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "ranking_col": ranking_col,
        "realized_edge_col": realized_edge_col,
        "daily_top_n": int(daily_top_n),
        "history_start": str(history_start.date()),
        "evaluation_start": str(evaluation_start.date()),
        "ridge_alpha": float(ridge_alpha),
        "gate_threshold": 0.0,
        "meta_features": META_FEATURES,
        "scaler": {
            "mean": [float(value) for value in scaler.mean_],
            "scale": [float(value) for value in scaler.scale_],
            "variance": [float(value) for value in scaler.var_],
            "samples_seen": int(scaler.n_samples_seen_),
        },
        "ridge": {
            "coefficient": [float(value) for value in ridge.coef_],
            "intercept": float(ridge.intercept_),
        },
        "train": {
            "start": str(train["trade_date"].min().date()),
            "end": str(train["trade_date"].max().date()),
            "dates": int(len(train)),
        },
        "evaluation": {
            "start": str(evaluation["trade_date"].min().date()),
            "end": str(evaluation["trade_date"].max().date()),
            "dates": int(len(evaluation)),
            "enabled_dates": int(evaluation["meta_gate_enabled"].sum()),
        },
        "deployment_allowed": False,
        "paper_orders_allowed": False,
    }
    manual_scores = evaluation.apply(
        lambda row: score_meta_features(row, report),
        axis=1,
    ).to_numpy(dtype=float)
    implementation_diff = np.abs(
        manual_scores - evaluation["meta_gate_score"].to_numpy(dtype=float)
    )
    report["implementation_parity"] = {
        "max_abs_score_diff": float(implementation_diff.max()),
        "passed": bool(float(implementation_diff.max()) <= parity_tolerance),
    }
    if not report["implementation_parity"]["passed"]:
        raise RuntimeError(
            "serialized daily gate does not reproduce sklearn pipeline: "
            f"max_abs_diff={implementation_diff.max()}"
        )
    if expected_daily_path is not None:
        expected = pd.read_csv(expected_daily_path, parse_dates=["trade_date"])
        expected["trade_date"] = expected["trade_date"].dt.normalize()
        compared = evaluation[["trade_date", "meta_gate_score", "meta_gate_enabled"]].merge(
            expected[["trade_date", "meta_gate_score", "meta_gate_enabled"]],
            on="trade_date",
            how="outer",
            suffixes=("_rebuilt", "_expected"),
            indicator=True,
        )
        matched = compared["_merge"] == "both"
        score_diff = (
            pd.to_numeric(compared.loc[matched, "meta_gate_score_rebuilt"])
            - pd.to_numeric(compared.loc[matched, "meta_gate_score_expected"])
        ).abs()
        enabled_mismatch = (
            compared.loc[matched, "meta_gate_enabled_rebuilt"].astype(bool)
            != compared.loc[matched, "meta_gate_enabled_expected"].astype(bool)
        )
        key_mismatch = int((~matched).sum())
        max_abs_diff = float(score_diff.max()) if not score_diff.empty else float("inf")
        parity_passed = bool(
            key_mismatch == 0
            and int(enabled_mismatch.sum()) == 0
            and max_abs_diff <= parity_tolerance
        )
        report["expected_forward_parity"] = {
            "expected_daily": str(expected_daily_path.resolve()),
            "expected_daily_sha256": sha256_file(expected_daily_path),
            "matched_dates": int(matched.sum()),
            "key_mismatch": key_mismatch,
            "enabled_mismatch": int(enabled_mismatch.sum()),
            "max_abs_score_diff": max_abs_diff,
            "tolerance": float(parity_tolerance),
            "passed": parity_passed,
        }
        if not parity_passed:
            raise RuntimeError(
                "frozen daily gate does not reproduce expected forward scores: "
                f"{report['expected_forward_parity']}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ranking-col", default="raw_score")
    parser.add_argument("--realized-edge-col", required=True)
    parser.add_argument("--history-start", default="2023-08-03")
    parser.add_argument("--evaluation-start", default="2026-04-08")
    parser.add_argument("--daily-top-n", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--minimum-training-dates", type=int, default=120)
    parser.add_argument("--expected-daily")
    parser.add_argument("--parity-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    report = freeze_daily_gate(
        Path(args.predictions).resolve(),
        Path(args.output).resolve(),
        ranking_col=args.ranking_col,
        realized_edge_col=args.realized_edge_col,
        history_start=pd.Timestamp(args.history_start).normalize(),
        evaluation_start=pd.Timestamp(args.evaluation_start).normalize(),
        daily_top_n=args.daily_top_n,
        ridge_alpha=args.ridge_alpha,
        minimum_training_dates=args.minimum_training_dates,
        expected_daily_path=(
            Path(args.expected_daily).resolve() if args.expected_daily else None
        ),
        parity_tolerance=args.parity_tolerance,
    )
    parity = report.get("expected_forward_parity", report["implementation_parity"])
    print(
        f"[freeze daily meta] train_dates={report['train']['dates']} "
        f"evaluation_dates={report['evaluation']['dates']} "
        f"enabled={report['evaluation']['enabled_dates']} "
        f"parity={parity['passed']} max_abs_diff={parity['max_abs_score_diff']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
