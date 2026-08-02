from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from ruamel.yaml import YAML


LAYER_KEYS = {
    "outer": "outer_model",
    "middle": "middle_model",
    "inner": "inner_model",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))


def score_saved_model(
    config_path: Path,
    layer: str,
    model_path: Path,
    output_csv: Path,
    output_json: Path,
) -> dict[str, Any]:
    import qlib
    from qlib.utils import init_instance_by_config

    config = load_yaml(config_path)
    qlib_key = "qlib_init_inner" if layer == "inner" else "qlib_init"
    qlib.init(**config.get(qlib_key, config["qlib_init"]))
    layer_config = config[LAYER_KEYS[layer]]
    dataset = init_instance_by_config(layer_config["dataset"])
    model = joblib.load(model_path)
    prediction = model.predict(dataset, segment="test")
    if not isinstance(prediction, pd.Series):
        feature = dataset.prepare("test", col_set="feature")
        prediction = pd.Series(prediction, index=feature.index)
    prediction = prediction.rename(layer).sort_index()
    if prediction.empty:
        raise ValueError("saved model produced no test predictions")
    if not isinstance(prediction.index, pd.MultiIndex):
        raise ValueError("prediction index must be a datetime/instrument MultiIndex")
    datetime_values = pd.to_datetime(prediction.index.get_level_values("datetime"))
    instrument_values = prediction.index.get_level_values("instrument").astype(str)
    if prediction.isna().any():
        raise ValueError("saved model produced NaN predictions")

    test_segment = layer_config["dataset"]["kwargs"]["segments"]["test"]
    expected_start = pd.Timestamp(test_segment[0]).normalize()
    expected_end = pd.Timestamp(test_segment[1]).normalize()
    actual_start = datetime_values.min().normalize()
    actual_end = datetime_values.max().normalize()
    if actual_start != expected_start or actual_end != expected_end:
        raise ValueError(
            "prediction dates do not match configured test segment: "
            f"expected={expected_start.date()}..{expected_end.date()} "
            f"actual={actual_start.date()}..{actual_end.date()}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output_csv)
    report = {
        "status": "saved_frozen_model_prediction",
        "layer": layer,
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "output": str(output_csv.resolve()),
        "rows": int(len(prediction)),
        "instruments": int(pd.Index(instrument_values).nunique()),
        "prediction_start": str(actual_start.date()),
        "prediction_end": str(actual_end.date()),
        "training_performed": False,
        "profile_retuned": False,
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--layer", choices=sorted(LAYER_KEYS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = score_saved_model(
        Path(args.config).resolve(),
        args.layer,
        Path(args.model).resolve(),
        Path(args.output_csv).resolve(),
        Path(args.output_json).resolve(),
    )
    print(
        f"[saved model score] layer={report['layer']} rows={report['rows']} "
        f"date={report['prediction_start']} model={report['model_sha256'][:12]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
