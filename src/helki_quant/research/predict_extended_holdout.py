from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from ruamel.yaml import YAML


HERE = Path(__file__).resolve().parent
MULTI_LAYER = HERE.parent
INTRADAY = MULTI_LAYER.parent / "intraday_t"
for path in (MULTI_LAYER, INTRADAY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_oof import add_fold_whitelist, load_yaml


LAYER_KEYS = {
    "outer": "outer_model",
    "middle": "middle_model",
    "inner": "inner_model",
}
DEFAULT_MIDDLE_WHITELIST = (
    HERE
    / "outputs"
    / "factor_reports"
    / "pit_holdout_de2_srfs_es"
    / "fold_99"
    / "feature_whitelist_middle_v2.json"
)


def set_provider(cfg: dict, provider: Path) -> None:
    cfg["qlib_init"]["provider_uri"]["day"] = str(provider.resolve())


def set_test_window(dataset_cfg: dict, start: str, end: str) -> dict:
    out = copy.deepcopy(dataset_cfg)
    handler_kwargs = out["kwargs"]["handler"]["kwargs"]
    handler_kwargs["end_time"] = end
    out["kwargs"]["segments"]["test"] = [start, end]
    return out


def patch_model_compat(model: object) -> object:
    """Backfill fields missing from older CatBoostDEnsemble pickles."""

    loss = getattr(model, "loss", None)
    if loss is None:
        params = getattr(model, "_params", {}) or {}
        loss = params.get("loss_function", "RMSE")
        setattr(model, "loss", loss)

    if not hasattr(model, "_is_multiclass"):
        setattr(model, "_is_multiclass", loss == "MultiClass")
    if not hasattr(model, "_is_binary"):
        setattr(model, "_is_binary", loss == "Logloss")
    if not hasattr(model, "_is_cls"):
        setattr(
            model,
            "_is_cls",
            bool(getattr(model, "_is_multiclass", False) or getattr(model, "_is_binary", False)),
        )
    if not hasattr(model, "binary_threshold"):
        setattr(model, "binary_threshold", 0.0)
    if not hasattr(model, "adaptive_thresholds"):
        setattr(model, "adaptive_thresholds", None)
    if not hasattr(model, "instrument_thresholds"):
        setattr(model, "instrument_thresholds", {})
    return model


def predict(
    config_path: Path,
    model_path: Path,
    whitelist_path: Path | None,
    provider: Path,
    output_path: Path,
    start: str,
    end: str,
    layer: str = "middle",
) -> dict:
    if layer not in LAYER_KEYS:
        raise ValueError(f"unsupported layer: {layer}")
    started = time.time()
    cfg = load_yaml(config_path)
    set_provider(cfg, provider)
    qlib.init(**cfg["qlib_init"])
    model_cfg = copy.deepcopy(cfg[LAYER_KEYS[layer]])
    model_cfg["dataset"] = set_test_window(model_cfg["dataset"], start, end)
    if whitelist_path is not None:
        model_cfg["dataset"] = add_fold_whitelist(model_cfg["dataset"], whitelist_path)
    model = patch_model_compat(joblib.load(model_path))
    dataset = init_instance_by_config(model_cfg["dataset"])
    prediction = model.predict(dataset, segment="test")
    if not isinstance(prediction, pd.Series):
        feature = dataset.prepare("test", col_set="feature")
        prediction = pd.Series(prediction, index=feature.index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.rename(layer).to_csv(output_path)
    meta = {
        "status": "forward_layer_prediction_research_only",
        "layer": layer,
        "config": str(config_path),
        "model": str(model_path),
        "whitelist": str(whitelist_path) if whitelist_path else None,
        "provider": str(provider),
        "prediction": str(output_path),
        "test_start": start,
        "test_end": end,
        "rows": int(len(prediction)),
        "prediction_start": str(prediction.index.get_level_values("datetime").min()),
        "prediction_end": str(prediction.index.get_level_values("datetime").max()),
        "elapsed_seconds": time.time() - started,
        "deployment_allowed": False,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=sorted(LAYER_KEYS), default="middle")
    parser.add_argument(
        "--config",
        default=str(HERE / "outputs" / "holdout_config" / "pit_holdout_de2_srfs_es.yaml"),
    )
    parser.add_argument(
        "--model",
        default=str(HERE / "outputs" / "oof" / "pit_holdout_de2_srfs_es" / "middle" / "fold_99_model.pkl"),
    )
    parser.add_argument(
        "--whitelist",
        default=None,
        help="Optional feature whitelist. Middle keeps its historical default; outer/inner use none unless supplied.",
    )
    parser.add_argument("--provider", required=True)
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument("--start", default="2026-04-21")
    parser.add_argument("--end", default="2026-06-04")
    args = parser.parse_args()
    whitelist = Path(args.whitelist).resolve() if args.whitelist else None
    if args.layer == "middle" and whitelist is None:
        whitelist = DEFAULT_MIDDLE_WHITELIST.resolve()
    output = (
        Path(args.output).resolve()
        if args.output
        else (
            HERE
            / "outputs"
            / "oof"
            / "pit_holdout_20260605_de2_srfs_es"
            / args.layer
            / "fold_99.csv"
        ).resolve()
    )
    meta = predict(
        Path(args.config).resolve(),
        Path(args.model).resolve(),
        whitelist,
        Path(args.provider).resolve(),
        output,
        args.start,
        args.end,
        args.layer,
    )
    print(
        f"[forward layer predict] layer={meta['layer']} rows={meta['rows']} "
        f"window={meta['prediction_start']}..{meta['prediction_end']} "
        f"output={meta['prediction']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
