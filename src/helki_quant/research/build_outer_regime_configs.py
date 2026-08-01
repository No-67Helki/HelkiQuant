from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ruamel.yaml import YAML

from build_fold_configs import (
    BASE_CONFIG,
    configure_fold,
    convert_to_simple_catboost,
    load_yaml,
    make_provider_paths_absolute,
    remove_global_whitelists,
    set_daily_provider,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_DAILY_PROVIDER = REPO_ROOT / "data" / "cn_data_outer_regime_broad_20260605_v1"
DEFAULT_FOLDS = HERE / "outputs" / "purged_folds.json"

OUTER_LABELS = {
    "broad_adverse_mdd5_20d": "$OUTER_BROAD_20D_ADVERSE_MDD5",
    "broad_adverse_loss5_20d": "$OUTER_BROAD_20D_ADVERSE_LOSS5",
    "broad_adverse_combo_20d": "$OUTER_BROAD_20D_ADVERSE_COMBO",
    "broad_adverse_mdd5_10d": "$OUTER_BROAD_10D_ADVERSE_MDD5",
    "broad_adverse_loss5_10d": "$OUTER_BROAD_10D_ADVERSE_LOSS5",
    "broad_adverse_combo_10d": "$OUTER_BROAD_10D_ADVERSE_COMBO",
}


def configure_outer_regime(
    cfg: dict,
    label_expr: str,
    threshold: float,
    *,
    collapse_by_datetime: bool = False,
) -> None:
    handler_cfg = cfg["outer_model"]["dataset"]["kwargs"]["handler"]
    handler_cfg["class"] = "Alpha158Regime"
    handler_cfg["module_path"] = "alpha158_regime"
    handler_kwargs = cfg["outer_model"]["dataset"]["kwargs"]["handler"]["kwargs"]
    handler_kwargs["label"] = [label_expr]
    handler_kwargs["infer_processors"] = [
        proc
        for proc in handler_kwargs.get("infer_processors", [])
        if proc.get("class") != "FeatureWhitelist"
    ]
    model_kwargs = cfg["outer_model"]["model"]["kwargs"]
    model_kwargs["collapse_by_datetime"] = bool(collapse_by_datetime)
    model_class = cfg["outer_model"]["model"].get("class")
    if model_class == "CatBoostDEnsemble":
        model_kwargs["loss"] = "Logloss"
        model_kwargs["binary_threshold"] = float(threshold)
        model_kwargs["task_type"] = "CPU"
        model_kwargs.pop("thresholds", None)
        model_kwargs.pop("adaptive_thresholds", None)
        model_kwargs.pop("num_classes", None)
    else:
        model_kwargs.update(
            {
                "num_classes": 2,
                "thresholds": [float(threshold)],
                "task_type": "CPU",
            }
        )


def base_config(
    daily_provider: Path,
    label_expr: str,
    threshold: float,
    *,
    collapse_by_datetime: bool = False,
) -> dict:
    cfg = load_yaml(BASE_CONFIG)
    make_provider_paths_absolute(cfg, BASE_CONFIG.parent)
    set_daily_provider(cfg, daily_provider)
    remove_global_whitelists(cfg)
    configure_outer_regime(
        cfg,
        label_expr,
        threshold,
        collapse_by_datetime=collapse_by_datetime,
    )
    return cfg


def build_folds(
    output_dir: Path,
    folds_path: Path,
    daily_provider: Path,
    label_name: str,
    threshold: float,
    collapse_by_datetime: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_expr = OUTER_LABELS[label_name]
    base = base_config(
        daily_provider,
        label_expr,
        threshold,
        collapse_by_datetime=collapse_by_datetime,
    )
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    manifest = {
        "status": "outer_regime_fold_configs_research_only",
        "label_name": label_name,
        "label": label_expr,
        "binary_threshold": float(threshold),
        "collapse_by_datetime": bool(collapse_by_datetime),
        "daily_provider": str(daily_provider.resolve()),
        "folds": [],
        "deployment_allowed": False,
    }
    for fold in folds:
        fold_dir = output_dir / f"fold_{fold['fold']:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        entry = {"fold": fold["fold"], "configs": {}, "commands": []}
        for variant in ("simple", "densemble"):
            run_variant = f"outer_regime_{label_name}_{variant}"
            cfg = copy.deepcopy(base)
            configure_fold(cfg, fold)
            if variant == "simple":
                convert_to_simple_catboost(cfg)
                configure_outer_regime(
                    cfg,
                    label_expr,
                    threshold,
                    collapse_by_datetime=collapse_by_datetime,
                )
            else:
                configure_outer_regime(cfg, label_expr, threshold)
            cfg["research_metadata"] = {
                **fold,
                "variant": run_variant,
                "label_name": label_name,
                "label": label_expr,
                "daily_provider": str(daily_provider.resolve()),
                "deployment_allowed": False,
            }
            config_path = fold_dir / f"{variant}.yaml"
            with config_path.open("w", encoding="utf-8") as stream:
                yaml.dump(cfg, stream)
            entry["configs"][variant] = str(config_path)
            entry["commands"].append(
                "python run_oof.py "
                f"--config \"{config_path}\" --layer outer --fold {fold['fold']} "
                f"--variant {run_variant}"
            )
        manifest["folds"].append(entry)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-name", choices=sorted(OUTER_LABELS), default="broad_adverse_mdd5_20d")
    parser.add_argument("--daily-provider", default=str(DEFAULT_DAILY_PROVIDER))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--collapse-by-datetime", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(HERE / "outputs" / "outer_regime_fold_configs_broad_adverse_mdd5_20d_20260609"),
    )
    args = parser.parse_args()
    manifest = build_folds(
        Path(args.output_dir).resolve(),
        Path(args.folds).resolve(),
        Path(args.daily_provider).resolve(),
        args.label_name,
        args.threshold,
        args.collapse_by_datetime,
    )
    print(
        "[outer regime configs] "
        f"label={manifest['label_name']} provider={manifest['daily_provider']} "
        f"manifest={Path(args.output_dir).resolve() / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
