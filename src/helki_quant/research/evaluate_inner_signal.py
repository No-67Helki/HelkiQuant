from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

import qlib
from qlib.utils import init_instance_by_config


HERE = Path(__file__).resolve().parent
MULTI_LAYER = HERE.parent
INTRADAY = MULTI_LAYER.parent / "intraday_t"
for path in (MULTI_LAYER, INTRADAY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_yaml(path: Path) -> dict:
    return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        rank_sum = np.bincount(inv, weights=ranks)
        avg = rank_sum / counts
        ranks = avg[inv]
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def directional_buckets(frame: pd.DataFrame) -> list[dict]:
    buckets = [
        ("strong_reverse", frame["score"] <= 0.40),
        ("reverse", frame["score"] <= 0.45),
        ("neutral", (frame["score"] > 0.45) & (frame["score"] < 0.55)),
        ("buy_then_sell", frame["score"] >= 0.55),
        ("strong_buy_then_sell", frame["score"] >= 0.60),
    ]
    rows = []
    for name, mask in buckets:
        sub = frame.loc[mask]
        rows.append(
            {
                "bucket": name,
                "rows": int(len(sub)),
                "ratio": float(len(sub) / len(frame)) if len(frame) else 0.0,
                "label_mean": float(sub["label"].mean()) if len(sub) else None,
                "label_positive_ratio": float((sub["label"] > 0).mean()) if len(sub) else None,
                "reverse_label_positive_ratio": float((sub["label"] < 0).mean()) if len(sub) else None,
            }
        )
    return rows


def evaluate(config_path: Path, prediction_path: Path, output_path: Path) -> dict:
    cfg = load_yaml(config_path)
    qlib.init(**cfg.get("qlib_init_inner", cfg["qlib_init"]))
    dataset = init_instance_by_config(cfg["inner_model"]["dataset"])
    labels = dataset.prepare("test", col_set="label")
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    labels = labels.rename("label")
    pred = pd.read_csv(prediction_path, index_col=[0, 1], parse_dates=[0]).iloc[:, 0]
    pred.index.names = labels.index.names
    pred = pred.rename("score")
    frame = pd.concat([pred, labels], axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    daily_ic = frame.groupby(level="datetime").apply(
        lambda x: x["score"].corr(x["label"], method="spearman") if len(x) >= 5 else np.nan
    )
    y_binary = (frame["label"].to_numpy() > 0).astype(int)
    report = {
        "status": "evaluated" if len(frame) else "failed",
        "config": str(config_path),
        "prediction_path": str(prediction_path),
        "rows": int(len(frame)),
        "date_start": str(frame.index.get_level_values("datetime").min()) if len(frame) else None,
        "date_end": str(frame.index.get_level_values("datetime").max()) if len(frame) else None,
        "score_mean": float(frame["score"].mean()) if len(frame) else None,
        "score_std": float(frame["score"].std()) if len(frame) else None,
        "label_mean": float(frame["label"].mean()) if len(frame) else None,
        "label_positive_ratio": float((frame["label"] > 0).mean()) if len(frame) else None,
        "spearman": float(frame["score"].corr(frame["label"], method="spearman")) if len(frame) else None,
        "pearson": float(frame["score"].corr(frame["label"], method="pearson")) if len(frame) else None,
        "auc_positive": auc_score(y_binary, frame["score"].to_numpy()) if len(frame) else None,
        "daily_ic_mean": float(daily_ic.mean()) if len(daily_ic.dropna()) else None,
        "daily_ic_ir": float(daily_ic.mean() / (daily_ic.std() + 1e-12)) if len(daily_ic.dropna()) else None,
        "buckets": directional_buckets(frame),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(
        Path(args.config).resolve(),
        Path(args.prediction).resolve(),
        Path(args.output).resolve(),
    )
    print(
        f"[inner signal] status={report['status']} rows={report['rows']} "
        f"spearman={report['spearman']} auc={report['auc_positive']}"
    )


if __name__ == "__main__":
    main()
