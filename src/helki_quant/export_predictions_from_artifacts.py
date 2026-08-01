from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoost
from ruamel.yaml import YAML

import qlib
from qlib.utils import init_instance_by_config


DIRNAME = Path(__file__).absolute().resolve().parent
INTRADAY_DIR = DIRNAME.parent / "intraday_t"
for path in (DIRNAME, INTRADAY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _flat_cols(cols) -> list[str]:
    return [c[-1] if isinstance(c, tuple) else str(c) for c in cols]


def _predict_layer(cfg: dict, artifacts_dir: Path, layer_name: str, cfg_key: str) -> pd.Series:
    layer_dir = artifacts_dir / layer_name
    meta = json.loads((layer_dir / "meta.json").read_text(encoding="utf-8"))
    dataset = init_instance_by_config(cfg[cfg_key]["dataset"])
    x = dataset.prepare("test", col_set="feature").copy()
    x.columns = _flat_cols(x.columns)

    preds = []
    for i in range(int(meta["num_models"])):
        model = CatBoost()
        model.load_model(str(layer_dir / f"submodel_{i}.cbm"))
        sub_names = json.loads((layer_dir / f"sub_features_{i}.json").read_text(encoding="utf-8"))
        sub_x = x.reindex(columns=sub_names, fill_value=0.0).values

        if meta.get("is_cls", False):
            proba = np.asarray(model.predict(sub_x, prediction_type="Probability"))
            if proba.ndim == 1:
                signal = proba.astype(float)
            elif proba.shape[1] >= 3:
                signal = proba[:, -1] - proba[:, 0]
            elif proba.shape[1] == 2:
                signal = proba[:, 1]
            else:
                signal = proba[:, 0]
        else:
            signal = np.asarray(model.predict(sub_x), dtype=float).reshape(-1)
        preds.append(signal)

    return pd.Series(np.mean(np.vstack(preds), axis=0), index=x.index, name=f"pred_{layer_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DIRNAME / "config_densemble_v2.yaml"))
    parser.add_argument("--artifacts_dir", default=str(DIRNAME / "artifacts" / "robust_v2"))
    parser.add_argument("--layers", nargs="+", default=["outer", "middle", "inner"])
    args = parser.parse_args()

    cfg = YAML(typ="safe", pure=True).load(Path(args.config).read_text(encoding="utf-8"))
    qlib.init(**cfg["qlib_init"])

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    pred_dir = artifacts_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    layer_to_cfg = {
        "outer": "outer_model",
        "middle": "middle_model",
        "inner": "inner_model",
    }

    for layer in args.layers:
        if layer not in layer_to_cfg:
            raise ValueError(f"Unsupported layer: {layer}")
        pred = _predict_layer(cfg, artifacts_dir, layer, layer_to_cfg[layer])
        out_path = pred_dir / f"pred_{layer}.csv"
        pred.to_csv(out_path)
        print(f"[pred] {layer}: rows={len(pred)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
