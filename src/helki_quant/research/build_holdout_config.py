from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ruamel.yaml import YAML

from build_fold_configs import (
    BASE_CONFIG,
    configure_fold,
    load_yaml,
    make_provider_paths_absolute,
    remove_global_whitelists,
    set_daily_provider,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_DAILY_PROVIDER = REPO_ROOT / "data" / "cn_data_research_pit"

HOLDOUT = {
    "fold": 99,
    "name": "micro_holdout_2026_04",
    "train_start": "2022-01-04",
    "train_end": "2025-09-26",
    "valid_start": "2025-09-29",
    "valid_end": "2026-03-20",
    "test_start": "2026-04-03",
    "test_end": "2026-04-20",
    "purge_days": 21,
    "embargo_days": 5,
    "warning": (
        "Short chronological holdout after OOF fold 6. It is useful as a leak "
        "check but too short to be sufficient promotion evidence."
    ),
}


def build(output_dir: Path, daily_provider: Path) -> dict:
    base = load_yaml(BASE_CONFIG)
    make_provider_paths_absolute(base, BASE_CONFIG.parent)
    set_daily_provider(base, daily_provider)
    remove_global_whitelists(base)
    cfg = copy.deepcopy(base)
    configure_fold(cfg, HOLDOUT)
    kwargs = cfg["middle_model"]["model"]["kwargs"]
    kwargs.update(
        {
            "num_models": 2,
            "sub_weights": [1, 1],
            "enable_sr": True,
            "enable_fs": True,
            "od_type": "Iter",
            "od_wait": 40,
        }
    )
    cfg["research_metadata"] = {
        **HOLDOUT,
        "variant": "pit_holdout_de2_srfs_es",
        "outer_middle_daily_provider": str(daily_provider.resolve()),
        "deployment_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "pit_holdout_de2_srfs_es.yaml"
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.dump(cfg, stream)
    manifest = {
        "status": "holdout_config_research_only",
        "holdout": HOLDOUT,
        "config": str(config_path),
        "daily_provider": str(daily_provider.resolve()),
        "commands": [
            (
                "python run_oof.py "
                f"--config \"{config_path}\" --layer middle --fold 99 "
                "--variant pit_holdout_de2_srfs_es --select-factors"
            )
        ],
        "deployment_allowed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default=str(HERE / "outputs" / "holdout_config")
    )
    parser.add_argument("--daily-provider", default=str(DEFAULT_DAILY_PROVIDER))
    args = parser.parse_args()
    manifest = build(Path(args.output_dir).resolve(), Path(args.daily_provider).resolve())
    print(f"[holdout config] {manifest['config']}")


if __name__ == "__main__":
    main()
