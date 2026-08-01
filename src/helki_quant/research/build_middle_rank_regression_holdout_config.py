from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_middle_rank_regression_configs import build_config, load_yaml, write_yaml


def build(
    source_path: Path,
    output_path: Path,
    report_path: Path,
    provider: Path,
    num_models: int,
) -> dict:
    cfg = build_config(load_yaml(source_path), num_models=num_models)
    cfg["qlib_init"]["provider_uri"]["day"] = str(provider.resolve())
    metadata = cfg.setdefault("research_metadata", {})
    metadata.update(
        {
            "source_config": str(source_path.resolve()),
            "canonical_provider": str(provider.resolve()),
            "holdout_window_consumed_for_diagnostics": True,
            "deployment_allowed": False,
        }
    )
    write_yaml(output_path, cfg)
    report = {
        "status": "middle_rank_regression_holdout_config_built",
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "provider": str(provider.resolve()),
        "num_models": int(num_models),
        "train": cfg["middle_model"]["dataset"]["kwargs"]["segments"]["train"],
        "valid": cfg["middle_model"]["dataset"]["kwargs"]["segments"]["valid"],
        "test": cfg["middle_model"]["dataset"]["kwargs"]["segments"]["test"],
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--num-models", type=int, default=1)
    args = parser.parse_args()
    if args.num_models < 1:
        raise ValueError("num-models must be >= 1")
    report = build(
        Path(args.source).resolve(),
        Path(args.output).resolve(),
        Path(args.report).resolve(),
        Path(args.provider).resolve(),
        args.num_models,
    )
    print(
        "[middle rank regression holdout config] "
        f"train={report['train']} valid={report['valid']} test={report['test']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
