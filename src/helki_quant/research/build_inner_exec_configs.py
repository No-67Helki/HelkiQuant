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
)
from build_holdout_config import HOLDOUT


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_INNER_DAY_PROVIDER = REPO_ROOT / "data" / "cn_data_pool_inner_research_2026"
DEFAULT_MINUTE_PROVIDER = REPO_ROOT / "data" / "cn_data_1min_pool_research_2026"
DEFAULT_FOLDS = HERE / "outputs" / "purged_folds.json"
INNER_LABELS = {
    "exec_net": "Ref($INTRADAY_T_EXEC_NET_RET, -1)",
    "reverse_net": "Ref($INTRADAY_T_REVERSE_NET_RET, -1)",
    "t0_open_am": "Ref($INTRADAY_T0_SELL_OPEN_BUY_AM_NET_RET, -1)",
    "t0_open_pm": "Ref($INTRADAY_T0_SELL_OPEN_BUY_PM_NET_RET, -1)",
    "t0_am_pm": "Ref($INTRADAY_T0_SELL_AM_BUY_PM_NET_RET, -1)",
    "t0_am_close": "Ref($INTRADAY_T0_SELL_AM_BUY_CLOSE_NET_RET, -1)",
    "t0_best_bucket": "Ref($INTRADAY_T0_BEST_BUCKET_NET_RET, -1)",
    "t0_best2_mean": "Ref($INTRADAY_T0_BEST2_MEAN_NET_RET, -1)",
    "t0_bucket_hit": "Ref($INTRADAY_T0_BUCKET_HIT_RATIO, -1)",
}
INNER_THRESHOLDS = {
    "exec_net": 0.0,
    "reverse_net": 0.0,
    "t0_open_am": 0.0,
    "t0_open_pm": 0.0,
    "t0_am_pm": 0.0,
    "t0_am_close": 0.0,
    # Best-bucket opportunity is intentionally stricter: near-zero positive
    # values are too common and too fragile after real execution constraints.
    "t0_best_bucket": 0.01,
    "t0_best2_mean": 0.005,
    # Hit ratio predicts breadth rather than a single high-return bucket.
    "t0_bucket_hit": 0.35,
}


def configure_inner_provider(cfg: dict, inner_day_provider: Path, minute_provider: Path) -> None:
    cfg["qlib_init_inner"]["provider_uri"]["day"] = str(inner_day_provider.resolve())
    cfg["qlib_init_inner"]["provider_uri"]["1min"] = str(minute_provider.resolve())


def configure_inner_label(cfg: dict, label_expr: str, binary_threshold: float) -> None:
    handler_kwargs = cfg["inner_model"]["dataset"]["kwargs"]["handler"]["kwargs"]
    handler_kwargs["label"] = [label_expr]
    model_kwargs = cfg["inner_model"]["model"]["kwargs"]
    model_kwargs["binary_threshold"] = float(binary_threshold)


def configure_fast_densemble(cfg: dict, fast: bool) -> None:
    kwargs = cfg["inner_model"]["model"]["kwargs"]
    kwargs.update(
        {
            "num_models": 2 if fast else 4,
            "sub_weights": [1, 1],
            "enable_sr": True,
            "enable_fs": True,
            "od_type": "Iter",
            "od_wait": 40,
            "ensemble_eval_rows": 80000,
        }
    )
    if not fast:
        kwargs["sub_weights"] = [1, 1, 1, 1]
        kwargs["ensemble_eval_rows"] = 120000
    cfg["inner_model"]["fit_params"]["epochs"] = 220 if fast else 280


def base_inner_config(
    inner_day_provider: Path,
    minute_provider: Path,
    label_expr: str,
    binary_threshold: float,
) -> dict:
    cfg = load_yaml(BASE_CONFIG)
    make_provider_paths_absolute(cfg, BASE_CONFIG.parent)
    remove_global_whitelists(cfg)
    configure_inner_provider(cfg, inner_day_provider, minute_provider)
    configure_inner_label(cfg, label_expr, binary_threshold)
    return cfg


def build_holdout(
    output_dir: Path,
    inner_day_provider: Path,
    minute_provider: Path,
    label_name: str,
    fast_densemble: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_expr = INNER_LABELS[label_name]
    binary_threshold = INNER_THRESHOLDS[label_name]
    base = base_inner_config(inner_day_provider, minute_provider, label_expr, binary_threshold)
    manifest = {
        "status": "inner_exec_holdout_configs_research_only",
        "label_name": label_name,
        "label": label_expr,
        "binary_threshold": binary_threshold,
        "inner_day_provider": str(inner_day_provider.resolve()),
        "minute_provider": str(minute_provider.resolve()),
        "holdout": HOLDOUT,
        "configs": {},
        "commands": [],
        "deployment_allowed": False,
    }
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    for variant in ("simple", "de2_srfs_es"):
        run_variant = f"inner_exec_{label_name}_holdout_{variant}"
        cfg = copy.deepcopy(base)
        configure_fold(cfg, HOLDOUT)
        if variant == "simple":
            convert_to_simple_catboost(cfg)
        else:
            configure_fast_densemble(cfg, fast_densemble)
        cfg["research_metadata"] = {
            **HOLDOUT,
            "variant": run_variant,
            "label_name": label_name,
            "label": label_expr,
            "inner_day_provider": str(inner_day_provider.resolve()),
            "minute_provider": str(minute_provider.resolve()),
            "deployment_allowed": False,
        }
        config_path = output_dir / f"{variant}.yaml"
        with config_path.open("w", encoding="utf-8") as stream:
            yaml.dump(cfg, stream)
        manifest["configs"][variant] = str(config_path)
        manifest["commands"].append(
            "python run_oof.py "
            f"--config \"{config_path}\" --layer inner --fold 99 "
            f"--variant {run_variant} --select-factors"
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_folds(
    output_dir: Path,
    folds_path: Path,
    inner_day_provider: Path,
    minute_provider: Path,
    label_name: str,
    fast_densemble: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_expr = INNER_LABELS[label_name]
    binary_threshold = INNER_THRESHOLDS[label_name]
    base = base_inner_config(inner_day_provider, minute_provider, label_expr, binary_threshold)
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    manifest = {
        "status": "inner_exec_fold_configs_research_only",
        "label_name": label_name,
        "label": label_expr,
        "binary_threshold": binary_threshold,
        "inner_day_provider": str(inner_day_provider.resolve()),
        "minute_provider": str(minute_provider.resolve()),
        "folds": [],
        "deployment_allowed": False,
    }
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    for fold in folds:
        fold_dir = output_dir / f"fold_{fold['fold']:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        entry = {"fold": fold["fold"], "configs": {}, "commands": []}
        for variant in ("simple", "de2_srfs_es"):
            run_variant = f"inner_exec_{label_name}_{variant}"
            cfg = copy.deepcopy(base)
            configure_fold(cfg, fold)
            if variant == "simple":
                convert_to_simple_catboost(cfg)
            else:
                configure_fast_densemble(cfg, fast_densemble)
            cfg["research_metadata"] = {
                **fold,
                "variant": run_variant,
                "label_name": label_name,
                "label": label_expr,
                "inner_day_provider": str(inner_day_provider.resolve()),
                "minute_provider": str(minute_provider.resolve()),
                "deployment_allowed": False,
            }
            config_path = fold_dir / f"{variant}.yaml"
            with config_path.open("w", encoding="utf-8") as stream:
                yaml.dump(cfg, stream)
            entry["configs"][variant] = str(config_path)
            entry["commands"].append(
                "python run_oof.py "
                f"--config \"{config_path}\" --layer inner --fold {fold['fold']} "
                f"--variant {run_variant} --select-factors"
            )
        manifest["folds"].append(entry)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["holdout", "folds"], default="holdout")
    parser.add_argument("--label-name", choices=sorted(INNER_LABELS), default="exec_net")
    parser.add_argument(
        "--full-densemble",
        action="store_true",
        help="Use the slower 4-submodel DEnsemble instead of the fast 2-submodel research config.",
    )
    parser.add_argument("--inner-day-provider", default=str(DEFAULT_INNER_DAY_PROVIDER))
    parser.add_argument("--minute-provider", default=str(DEFAULT_MINUTE_PROVIDER))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument(
        "--output-dir",
        default=str(HERE / "outputs" / "inner_exec_holdout_configs"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if args.mode == "holdout":
        manifest = build_holdout(
            output_dir,
            Path(args.inner_day_provider).resolve(),
            Path(args.minute_provider).resolve(),
            args.label_name,
            not args.full_densemble,
        )
    else:
        manifest = build_folds(
            output_dir,
            Path(args.folds).resolve(),
            Path(args.inner_day_provider).resolve(),
            Path(args.minute_provider).resolve(),
            args.label_name,
            not args.full_densemble,
        )
    print(f"[inner exec configs] status={manifest['status']} -> {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
