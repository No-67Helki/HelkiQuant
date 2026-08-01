from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from ruamel.yaml import YAML

from run_research import build_folds


HERE = Path(__file__).resolve().parent
MULTI_LAYER = HERE.parent
BASE_CONFIG = MULTI_LAYER / "config_densemble_v2.yaml"
LAYER_KEYS = {
    "outer": "outer_model",
    "middle": "middle_model",
    "inner": "inner_model",
}


def load_yaml(path: Path) -> dict:
    return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))


def make_provider_paths_absolute(cfg: dict, base_dir: Path) -> None:
    for init_key in ("qlib_init", "qlib_init_inner"):
        provider_uri = cfg.get(init_key, {}).get("provider_uri")
        if not isinstance(provider_uri, dict):
            continue
        for freq, value in provider_uri.items():
            if isinstance(value, str) and not os.path.isabs(value):
                provider_uri[freq] = str((base_dir / value).resolve())


def remove_global_whitelists(cfg: dict) -> None:
    for model_key in LAYER_KEYS.values():
        handler_kwargs = cfg[model_key]["dataset"]["kwargs"]["handler"]["kwargs"]
        processors = handler_kwargs.get("infer_processors", [])
        handler_kwargs["infer_processors"] = [
            proc for proc in processors if proc.get("class") != "FeatureWhitelist"
        ]


def set_daily_provider(cfg: dict, daily_provider: Path | None) -> None:
    if daily_provider is None:
        return
    # Outer and middle layers share qlib_init. The inner layer keeps the
    # existing daily/minute provider pair because it requires minute fields.
    cfg["qlib_init"]["provider_uri"]["day"] = str(daily_provider.resolve())


def set_inner_providers(
    cfg: dict,
    inner_day_provider: Path | None,
    inner_minute_provider: Path | None,
) -> None:
    if inner_day_provider is not None:
        cfg["qlib_init_inner"]["provider_uri"]["day"] = str(inner_day_provider.resolve())
    if inner_minute_provider is not None:
        cfg["qlib_init_inner"]["provider_uri"]["1min"] = str(
            inner_minute_provider.resolve()
        )


def configure_fold(cfg: dict, fold: dict) -> None:
    cfg["segments"] = {
        "train": [fold["train_start"], fold["train_end"]],
        "valid": [fold["valid_start"], fold["valid_end"]],
        "test": [fold["test_start"], fold["test_end"]],
    }
    for model_key in LAYER_KEYS.values():
        handler_kwargs = cfg[model_key]["dataset"]["kwargs"]["handler"]["kwargs"]
        handler_kwargs["start_time"] = fold["train_start"]
        handler_kwargs["end_time"] = fold["test_end"]
        handler_kwargs["fit_start_time"] = fold["train_start"]
        handler_kwargs["fit_end_time"] = fold["train_end"]
        cfg[model_key]["dataset"]["kwargs"]["segments"] = copy.deepcopy(cfg["segments"])
    cfg["backtest"]["start_time"] = fold["test_start"]
    cfg["backtest"]["end_time"] = fold["test_end"]


def convert_to_simple_catboost(cfg: dict) -> None:
    for layer, model_key in LAYER_KEYS.items():
        old = cfg[model_key]["model"]["kwargs"]
        thresholds = list(old.get("thresholds", [-0.02, 0.02]))
        num_classes = 3
        if layer == "inner":
            thresholds = [float(old.get("binary_threshold", 0.0))]
            num_classes = 2
        cfg[model_key]["model"] = {
            "class": "CatBoostClsModel",
            "module_path": "catboost_cls_model",
            "kwargs": {
                "num_classes": num_classes,
                "thresholds": thresholds,
                "learning_rate": float(old.get("learning_rate", 0.035)),
                "depth": int(old.get("max_depth", 5)),
                "l2_leaf_reg": float(old.get("l2_leaf_reg", 8.0)),
                "random_strength": float(old.get("random_strength", 1.0)),
                "thread_count": int(old.get("thread_count", 8)),
                "random_seed": 42,
                "task_type": "CPU",
            },
        }
        epochs = int(cfg[model_key].get("fit_params", {}).get("epochs", 220))
        cfg[model_key]["fit_params"] = {
            "num_boost_round": epochs,
            "early_stopping_rounds": 40,
            "verbose_eval": 50,
        }


def generate_configs(
    base_config: Path,
    folds_path: Path,
    output_dir: Path,
    daily_provider: Path | None = None,
    inner_day_provider: Path | None = None,
    inner_minute_provider: Path | None = None,
    run_variant_prefix: str = "",
) -> dict:
    base = load_yaml(base_config)
    make_provider_paths_absolute(base, base_config.parent)
    set_daily_provider(base, daily_provider)
    set_inner_providers(base, inner_day_provider, inner_minute_provider)
    remove_global_whitelists(base)
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    generated_layers = list(LAYER_KEYS)
    if daily_provider is not None and (
        inner_day_provider is None or inner_minute_provider is None
    ):
        generated_layers = ["outer", "middle"]
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120

    manifest = {
        "status": "research_only_not_trained",
        "base_config": str(base_config),
        "outer_middle_daily_provider": base["qlib_init"]["provider_uri"]["day"],
        "inner_daily_provider": base["qlib_init_inner"]["provider_uri"]["day"],
        "inner_minute_provider": base["qlib_init_inner"]["provider_uri"]["1min"],
        "run_variant_prefix": run_variant_prefix,
        "generated_run_layers": generated_layers,
        "feature_policy": (
            "Global robust_v2 whitelists removed. run_oof.py --select-factors creates "
            "a train/valid-only whitelist separately for every fold and layer."
        ),
        "folds": [],
    }
    for fold in folds:
        fold_dir = output_dir / f"fold_{fold['fold']:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        entry = {"fold": fold["fold"], "configs": {}, "commands": []}
        for variant in ("simple", "densemble"):
            run_variant = f"{run_variant_prefix}{variant}"
            cfg = copy.deepcopy(base)
            configure_fold(cfg, fold)
            if variant == "simple":
                convert_to_simple_catboost(cfg)
            cfg["research_metadata"] = {
                "fold": fold["fold"],
                "variant": run_variant,
                "purge_days": fold["purge_days"],
                "embargo_days": fold["embargo_days"],
                "global_feature_whitelist_removed": True,
                "outer_middle_daily_provider": base["qlib_init"]["provider_uri"]["day"],
                "inner_day_provider": base["qlib_init_inner"]["provider_uri"]["day"],
                "inner_minute_provider": base["qlib_init_inner"]["provider_uri"]["1min"],
                "deployment_allowed": False,
            }
            config_path = fold_dir / f"{variant}.yaml"
            with config_path.open("w", encoding="utf-8") as stream:
                yaml.dump(cfg, stream)
            entry["configs"][variant] = str(config_path)
            for layer in manifest["generated_run_layers"]:
                entry["commands"].append(
                    "python run_oof.py "
                    f"--config \"{config_path}\" --layer {layer} "
                    f"--fold {fold['fold']} --variant {run_variant} --select-factors"
                )
        manifest["folds"].append(entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--folds", default=str(HERE / "outputs" / "purged_folds.json"))
    parser.add_argument("--output-dir", default=str(HERE / "outputs" / "fold_configs"))
    parser.add_argument(
        "--daily-provider",
        default=None,
        help="Optional point-in-time daily provider for outer and middle layers only.",
    )
    parser.add_argument("--inner-day-provider", default=None)
    parser.add_argument("--inner-minute-provider", default=None)
    parser.add_argument(
        "--run-variant-prefix",
        default="",
        help="Prefix OOF variant names to isolate outputs from another source pool.",
    )
    args = parser.parse_args()

    folds_path = Path(args.folds).resolve()
    if not folds_path.exists():
        build_folds(folds_path.parent)
    manifest = generate_configs(
        Path(args.base_config).resolve(),
        folds_path,
        Path(args.output_dir).resolve(),
        Path(args.daily_provider).resolve() if args.daily_provider else None,
        Path(args.inner_day_provider).resolve() if args.inner_day_provider else None,
        Path(args.inner_minute_provider).resolve()
        if args.inner_minute_provider
        else None,
        args.run_variant_prefix,
    )
    print(
        f"[fold configs] folds={len(manifest['folds'])} "
        f"-> {Path(args.output_dir).resolve() / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
