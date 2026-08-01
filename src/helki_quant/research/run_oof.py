from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import joblib
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

from evaluate_factors import evaluate_layer
from realtime_output import log_step, setup_realtime_output


LAYER_KEYS = {
    "outer": "outer_model",
    "middle": "middle_model",
    "inner": "inner_model",
}


def load_yaml(path: Path) -> dict:
    return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))


def add_fold_whitelist(dataset_cfg: dict, whitelist_path: Path) -> dict:
    dataset_cfg = copy.deepcopy(dataset_cfg)
    handler_kwargs = dataset_cfg["kwargs"]["handler"]["kwargs"]
    processors = [
        proc
        for proc in handler_kwargs.get("infer_processors", [])
        if proc.get("class") != "FeatureWhitelist"
    ]
    processors.append(
        {
            "class": "FeatureWhitelist",
            "module_path": "feature_processors",
            "kwargs": {
                "fields_group": "feature",
                "whitelist_path": str(whitelist_path.resolve()),
            },
        }
    )
    handler_kwargs["infer_processors"] = processors
    return dataset_cfg


def train_layer(
    config_path: Path,
    layer: str,
    fold: int,
    variant: str,
    output_dir: Path,
    select_factors: bool,
    whitelist_path: Path | None,
    feature_blacklist: set[str] | None = None,
) -> Path:
    setup_realtime_output()
    if select_factors and whitelist_path is not None:
        raise ValueError("Use either --select-factors or --whitelist-path, not both")
    cfg = load_yaml(config_path)
    qlib_key = "qlib_init_inner" if layer == "inner" else "qlib_init"
    log_step(
        f"[oof] init qlib layer={layer} fold={fold} variant={variant} "
        f"config={config_path}"
    )
    qlib.init(**cfg.get(qlib_key, cfg["qlib_init"]))

    model_cfg = copy.deepcopy(cfg[LAYER_KEYS[layer]])
    factor_dir = output_dir / "factor_reports" / variant / f"fold_{fold:02d}"
    if select_factors:
        log_step(f"[oof] factor selection start layer={layer} fold={fold}")
        evaluate_layer(
            layer=layer,
            config_path=config_path,
            output_dir=factor_dir,
            corr_threshold=0.90,
            min_features=20,
            max_features=80,
            corr_sample=100000,
            include_test_report=False,
            feature_blacklist=feature_blacklist,
        )
        whitelist_path = factor_dir / f"feature_whitelist_{layer}_v2.json"
        model_cfg["dataset"] = add_fold_whitelist(model_cfg["dataset"], whitelist_path)
        log_step(f"[oof] factor selection done whitelist={whitelist_path}")
    elif whitelist_path is not None:
        log_step(f"[oof] reuse whitelist={whitelist_path}")
        model_cfg["dataset"] = add_fold_whitelist(model_cfg["dataset"], whitelist_path)

    started = time.time()
    log_step(f"[oof] build model start layer={layer} fold={fold}")
    model = init_instance_by_config(model_cfg["model"])
    log_step(f"[oof] build dataset start layer={layer} fold={fold}")
    dataset = init_instance_by_config(model_cfg["dataset"])
    log_step(
        f"[oof] fit start layer={layer} fold={fold} "
        f"fit_params={model_cfg.get('fit_params', {})}"
    )
    model.fit(dataset, **model_cfg.get("fit_params", {}))
    log_step(f"[oof] fit done layer={layer} fold={fold} elapsed={time.time() - started:.1f}s")
    log_step(f"[oof] predict start layer={layer} fold={fold}")
    prediction = model.predict(dataset, segment="test")
    if not isinstance(prediction, pd.Series):
        feature = dataset.prepare("test", col_set="feature")
        prediction = pd.Series(prediction, index=feature.index)
    log_step(f"[oof] predict done rows={len(prediction)}")

    layer_dir = output_dir / "oof" / variant / layer
    layer_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = layer_dir / f"fold_{fold:02d}.csv"
    prediction.rename(layer).to_csv(prediction_path)
    model_path = layer_dir / f"fold_{fold:02d}_model.pkl"
    joblib.dump(model, model_path)
    log_step(f"[oof] saved prediction={prediction_path}")
    log_step(f"[oof] saved model={model_path}")
    meta = {
        "status": "fold_oof_prediction",
        "fold": fold,
        "layer": layer,
        "variant": variant,
        "config": str(config_path),
        "select_factors": select_factors,
        "whitelist_path": str(whitelist_path) if whitelist_path else None,
        "feature_blacklist": sorted(feature_blacklist or set()),
        "rows": int(len(prediction)),
        "prediction_start": str(prediction.index.get_level_values("datetime").min()),
        "prediction_end": str(prediction.index.get_level_values("datetime").max()),
        "elapsed_seconds": time.time() - started,
        "deployment_allowed": False,
    }
    prediction_path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return prediction_path


def main() -> None:
    setup_realtime_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--layer", choices=list(LAYER_KEYS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--variant",
        required=True,
        help="Output/model variant name, for example simple, densemble, or de2_srfs_es.",
    )
    parser.add_argument("--output-dir", default=str(HERE / "outputs"))
    parser.add_argument("--select-factors", action="store_true")
    parser.add_argument(
        "--whitelist-path",
        default=None,
        help="Reuse a fold-specific train/valid-only whitelist for fair model comparison.",
    )
    parser.add_argument(
        "--feature-blacklist",
        nargs="*",
        default=[],
        help="Feature names to exclude when --select-factors is used.",
    )
    args = parser.parse_args()
    path = train_layer(
        config_path=Path(args.config).resolve(),
        layer=args.layer,
        fold=args.fold,
        variant=args.variant,
        output_dir=Path(args.output_dir).resolve(),
        select_factors=args.select_factors,
        whitelist_path=Path(args.whitelist_path).resolve() if args.whitelist_path else None,
        feature_blacklist=set(args.feature_blacklist),
    )
    print(f"[oof] -> {path}")


if __name__ == "__main__":
    main()
