from __future__ import annotations

import argparse
import copy
from pathlib import Path

from ruamel.yaml import YAML


HERE = Path(__file__).resolve().parent


CANDIDATES = {
    "de2_plain_es": {
        "num_models": 2,
        "sub_weights": [1, 1],
        "enable_sr": False,
        "enable_fs": False,
    },
    "de2_srfs_es": {
        "num_models": 2,
        "sub_weights": [1, 1],
        "enable_sr": True,
        "enable_fs": True,
    },
}


def build(
    fold: int,
    output_dir: Path,
    source_config_dir: Path,
    variant_prefix: str = "",
) -> list[Path]:
    source = source_config_dir / f"fold_{fold:02d}" / "densemble.yaml"
    yaml_safe = YAML(typ="safe", pure=True)
    base = yaml_safe.load(source.read_text(encoding="utf-8"))
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    paths = []
    for base_name, params in CANDIDATES.items():
        name = f"{variant_prefix}{base_name}"
        cfg = copy.deepcopy(base)
        kwargs = cfg["middle_model"]["model"]["kwargs"]
        kwargs.update(params)
        kwargs["od_type"] = "Iter"
        kwargs["od_wait"] = 40
        cfg["research_metadata"]["variant"] = name
        cfg["research_metadata"]["densemble_candidate"] = params
        path = output_dir / f"fold_{fold:02d}" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            yaml.dump(cfg, stream)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--output-dir", default=str(HERE / "outputs" / "model_candidates")
    )
    parser.add_argument(
        "--source-config-dir", default=str(HERE / "outputs" / "fold_configs")
    )
    parser.add_argument(
        "--variant-prefix",
        default="",
        help="Prefix candidate variant names to isolate outputs by source pool.",
    )
    args = parser.parse_args()
    paths = build(
        args.fold,
        Path(args.output_dir).resolve(),
        Path(args.source_config_dir).resolve(),
        args.variant_prefix,
    )
    for path in paths:
        print(f"[candidate config] {path}")


if __name__ == "__main__":
    main()
