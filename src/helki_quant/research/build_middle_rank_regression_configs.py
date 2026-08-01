from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ruamel.yaml import YAML


def load_yaml(path: Path) -> dict:
    return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict) -> None:
    yaml = YAML()
    yaml.default_flow_style = False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(payload, handle)


def build_config(source: dict, *, num_models: int, label_horizon: int = 5) -> dict:
    if label_horizon < 1:
        raise ValueError("label_horizon must be >= 1")
    cfg = copy.deepcopy(source)
    middle = cfg["middle_model"]
    model = middle["model"]
    if model.get("class") != "CatBoostDEnsemble":
        raise ValueError("source middle model must be CatBoostDEnsemble")
    kwargs = model.setdefault("kwargs", {})
    kwargs["loss"] = "RMSE"
    kwargs["num_models"] = int(num_models)
    kwargs["sub_weights"] = [1.0] * int(num_models)
    kwargs.pop("thresholds", None)
    kwargs.pop("adaptive_thresholds", None)
    kwargs.pop("binary_threshold", None)
    if num_models == 1:
        kwargs["enable_sr"] = False
        kwargs["enable_fs"] = False

    handler = middle["dataset"]["kwargs"]["handler"]["kwargs"]
    handler["label"] = [
        f"Ref($close, -{label_horizon + 1}) / Ref($close, -1) - 1"
    ]
    handler["learn_processors"] = [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ]
    metadata = cfg.setdefault("research_metadata", {})
    metadata.update(
        {
            "middle_objective": "cross_sectional_rank_regression",
            "middle_label_processor": "CSRankNorm(label)",
            "middle_loss": "RMSE",
            "middle_num_models": int(num_models),
            "middle_label_horizon_trading_days": int(label_horizon),
            "test_metrics_read_during_config_build": False,
            "deployment_allowed": False,
        }
    )
    return cfg


def build(
    source_dir: Path,
    output_dir: Path,
    report_path: Path,
    num_models: int,
    label_horizon: int = 5,
) -> dict:
    rows = []
    for fold in range(1, 7):
        source_path = source_dir / f"fold_{fold:02d}" / "pit_de2_srfs_es.yaml"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        output_path = output_dir / f"fold_{fold:02d}" / "middle_rank_regression.yaml"
        write_yaml(
            output_path,
            build_config(
                load_yaml(source_path),
                num_models=num_models,
                label_horizon=label_horizon,
            ),
        )
        rows.append(
            {
                "fold": fold,
                "source": str(source_path.resolve()),
                "output": str(output_path.resolve()),
            }
        )
    report = {
        "status": "middle_rank_regression_configs_built",
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_models": int(num_models),
        "label_horizon_trading_days": int(label_horizon),
        "objective": f"CSRankNorm future_{label_horizon}d return with RMSE",
        "folds": rows,
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--num-models", type=int, default=1)
    parser.add_argument("--label-horizon", type=int, default=5)
    args = parser.parse_args()
    if args.num_models < 1:
        raise ValueError("num-models must be >= 1")
    if args.label_horizon < 1:
        raise ValueError("label-horizon must be >= 1")
    report = build(
        Path(args.source_dir).resolve(),
        Path(args.output_dir).resolve(),
        Path(args.report).resolve(),
        args.num_models,
        args.label_horizon,
    )
    print(
        "[middle rank regression configs] "
        f"folds={len(report['folds'])} num_models={report['num_models']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
