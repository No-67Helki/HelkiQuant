from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ruamel.yaml import YAML


HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE / "outputs" / "fold_configs_canonical_20260605" / "fold_06" / "densemble.yaml"
DEFAULT_OUTPUT = HERE / "outputs" / "paper_forward_20260605" / "config" / "densemble.yaml"


def load_yaml(path: Path) -> dict:
    return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))


def set_segments(cfg: dict, train_end: str, valid_start: str, valid_end: str, test_date: str) -> None:
    segments = {
        "train": ["2022-01-04", train_end],
        "valid": [valid_start, valid_end],
        "test": [test_date, test_date],
    }
    cfg["segments"] = copy.deepcopy(segments)
    for key in ("outer_model", "middle_model", "inner_model"):
        if key not in cfg:
            continue
        dataset = cfg[key]["dataset"]
        handler_kwargs = dataset["kwargs"]["handler"]["kwargs"]
        handler_kwargs["start_time"] = "2022-01-04"
        handler_kwargs["end_time"] = test_date
        handler_kwargs["fit_start_time"] = "2022-01-04"
        handler_kwargs["fit_end_time"] = train_end
        dataset["kwargs"]["segments"] = copy.deepcopy(segments)
    if "backtest" in cfg:
        cfg["backtest"]["start_time"] = test_date
        cfg["backtest"]["end_time"] = test_date


def build(
    base_config: Path,
    output: Path,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_date: str,
    provider_day: Path | None = None,
) -> dict:
    cfg = load_yaml(base_config)
    set_segments(cfg, train_end, valid_start, valid_end, test_date)
    if provider_day is not None:
        cfg.setdefault("qlib_init", {}).setdefault("provider_uri", {})["day"] = str(
            provider_day.resolve()
        )
    cfg["research_metadata"] = {
        "status": "paper_forward_middle_config",
        "source_config": str(base_config.resolve()),
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_date": test_date,
        "provider_day": str(provider_day.resolve()) if provider_day else None,
        "intended_use": "Generate current paper-simulation target only; not OOF evidence.",
        "deployment_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    with output.open("w", encoding="utf-8") as stream:
        yaml.dump(cfg, stream)
    meta = {
        "status": "paper_forward_config_created",
        "output": str(output.resolve()),
        "base_config": str(base_config.resolve()),
        "train": ["2022-01-04", train_end],
        "valid": [valid_start, valid_end],
        "test": [test_date, test_date],
        "provider_day": str(provider_day.resolve()) if provider_day else None,
        "deployment_allowed": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=str(DEFAULT_BASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train-end", default="2026-02-27")
    parser.add_argument("--valid-start", default="2026-03-02")
    parser.add_argument("--valid-end", default="2026-05-29")
    parser.add_argument("--test-date", default="2026-06-05")
    parser.add_argument("--provider-day", default="")
    args = parser.parse_args()
    meta = build(
        Path(args.base_config).resolve(),
        Path(args.output).resolve(),
        args.train_end,
        args.valid_start,
        args.valid_end,
        args.test_date,
        Path(args.provider_day).resolve() if args.provider_day else None,
    )
    print(
        "[paper forward config] "
        f"train={meta['train']} valid={meta['valid']} test={meta['test']} "
        f"output={meta['output']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
