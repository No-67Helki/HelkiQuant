from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["datetime", "trade_date", "instrument", "decision_time"]


def blend(
    prediction_paths: list[Path],
    weights: list[float],
    output_csv: Path,
    output_json: Path,
    *,
    label_col: str,
    edge_col: str,
) -> dict:
    if len(prediction_paths) < 2 or len(prediction_paths) != len(weights):
        raise ValueError("prediction paths and weights require the same length >= 2")
    weight_array = np.asarray(weights, dtype=float)
    if not np.isfinite(weight_array).all() or (weight_array < 0).any() or weight_array.sum() <= 0:
        raise ValueError("blend weights must be finite, non-negative, and sum positive")
    weight_array /= weight_array.sum()

    parts = []
    for idx, path in enumerate(prediction_paths):
        frame = pd.read_csv(path)
        required = set(KEYS + ["fold", label_col, edge_col, "raw_score", "score"])
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"prediction {path} missing columns: {missing}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"prediction {path} has duplicate keys")
        keep = frame[KEYS + ["fold", label_col, edge_col, "raw_score", "score"]].copy()
        keep = keep.rename(
            columns={
                "fold": f"fold_{idx}",
                label_col: f"label_{idx}",
                edge_col: f"edge_{idx}",
                "raw_score": f"raw_score_{idx}",
                "score": f"score_{idx}",
            }
        )
        parts.append(keep)
    merged = parts[0]
    for idx, part in enumerate(parts[1:], start=1):
        before = len(merged)
        merged = merged.merge(part, on=KEYS, how="inner", validate="one_to_one")
        if len(merged) != before:
            raise ValueError(
                f"prediction key mismatch after model {idx}: before={before} after={len(merged)}"
            )
    reference_fold = pd.to_numeric(merged["fold_0"], errors="raise").astype(int)
    reference_label = pd.to_numeric(merged["label_0"], errors="raise")
    reference_edge = pd.to_numeric(merged["edge_0"], errors="raise")
    for idx in range(1, len(parts)):
        if not np.array_equal(
            reference_fold.to_numpy(),
            pd.to_numeric(merged[f"fold_{idx}"], errors="raise").astype(int).to_numpy(),
        ):
            raise ValueError(f"fold mismatch for prediction model {idx}")
        if not np.allclose(
            reference_label,
            pd.to_numeric(merged[f"label_{idx}"], errors="raise"),
            equal_nan=True,
        ):
            raise ValueError(f"label mismatch for prediction model {idx}")
        if not np.allclose(
            reference_edge,
            pd.to_numeric(merged[f"edge_{idx}"], errors="raise"),
            equal_nan=True,
        ):
            raise ValueError(f"edge mismatch for prediction model {idx}")

    score_matrix = np.column_stack(
        [pd.to_numeric(merged[f"score_{idx}"], errors="coerce") for idx in range(len(parts))]
    )
    raw_matrix = np.column_stack(
        [
            pd.to_numeric(merged[f"raw_score_{idx}"], errors="coerce")
            for idx in range(len(parts))
        ]
    )
    if not np.isfinite(score_matrix).all() or not np.isfinite(raw_matrix).all():
        raise ValueError("prediction scores contain non-finite values")
    output = merged[KEYS].copy()
    output["fold"] = reference_fold
    output[label_col] = reference_label
    output[edge_col] = reference_edge
    output["label"] = (reference_edge > 0.0).astype(int)
    output["raw_score"] = raw_matrix @ weight_array
    output["score"] = score_matrix @ weight_array
    component_ranks = []
    for idx in range(len(parts)):
        score_col = f"component_score_{idx}"
        rank_col = f"component_daily_rank_{idx}"
        output[score_col] = score_matrix[:, idx]
        output[rank_col] = output.groupby("trade_date", sort=False)[score_col].rank(
            method="min",
            ascending=False,
        )
        component_ranks.append(output[rank_col].to_numpy(dtype=float))
    rank_matrix = np.column_stack(component_ranks)
    output["consensus_max_rank"] = rank_matrix.max(axis=1)
    output["consensus_mean_rank"] = rank_matrix.mean(axis=1)
    output["consensus_score"] = (
        1.0 / output["consensus_max_rank"] + output["score"] * 1e-6
    )
    output = output.sort_values(["trade_date", "instrument"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8-sig")

    score_corr = pd.DataFrame(score_matrix).corr(method="spearman").to_numpy().tolist()
    report = {
        "status": "held_intraday_oof_predictions_blended",
        "prediction_paths": [str(path.resolve()) for path in prediction_paths],
        "weights": weight_array.tolist(),
        "output_csv": str(output_csv.resolve()),
        "rows": int(len(output)),
        "dates": int(pd.to_datetime(output["trade_date"]).nunique()),
        "folds": int(output["fold"].nunique()),
        "score_spearman_correlation": score_corr,
        "consensus_rows": {
            "all_top1": int((output["consensus_max_rank"] <= 1).sum()),
            "all_top3": int((output["consensus_max_rank"] <= 3).sum()),
            "all_top5": int((output["consensus_max_rank"] <= 5).sum()),
        },
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-csvs", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--edge-col", required=True)
    args = parser.parse_args()
    paths = [Path(item.strip()).resolve() for item in args.prediction_csvs.split(",") if item.strip()]
    weights = [float(item.strip()) for item in args.weights.split(",") if item.strip()]
    report = blend(
        paths,
        weights,
        Path(args.output_csv).resolve(),
        Path(args.output_json).resolve(),
        label_col=args.label_col,
        edge_col=args.edge_col,
    )
    print(
        f"[held blend] rows={report['rows']} folds={report['folds']} "
        f"weights={report['weights']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
