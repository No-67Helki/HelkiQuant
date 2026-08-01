from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
LAYERS = ("outer", "middle", "inner")


def read_layer(layer_dir: Path, layer: str) -> tuple[pd.DataFrame, list[int]]:
    frames = []
    folds = []
    for path in sorted(layer_dir.glob("fold_*.csv")):
        frame = pd.read_csv(path)
        if layer not in frame.columns:
            raise ValueError(f"{path} does not contain prediction column {layer!r}")
        frames.append(frame[["datetime", "instrument", layer]])
        folds.append(int(path.stem.split("_")[-1]))
    if not frames:
        return pd.DataFrame(columns=["datetime", "instrument", layer]), []
    result = pd.concat(frames, ignore_index=True)
    result["datetime"] = pd.to_datetime(result["datetime"])
    duplicates = result.duplicated(["datetime", "instrument"], keep=False)
    if duplicates.any():
        raise ValueError(
            f"{layer} OOF predictions overlap on {int(duplicates.sum())} rows"
        )
    return result.sort_values(["datetime", "instrument"]), folds


def assemble(
    source_dir: Path,
    variant: str,
    output_dir: Path,
    layer_variants: dict[str, str] | None = None,
) -> dict:
    if layer_variants is None:
        layer_variants = {layer: variant for layer in LAYERS}
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "assembled_oof_research_only",
        "variant": variant,
        "layer_variants": layer_variants,
        "layers": {},
        "deployment_allowed": False,
    }
    for layer in LAYERS:
        frame, folds = read_layer(source_dir / layer_variants[layer] / layer, layer)
        frame = frame.rename(columns={layer: f"pred_{layer}"})
        frame.to_csv(prediction_dir / f"pred_{layer}.csv", index=False)
        manifest["layers"][layer] = {"folds": folds, "rows": len(frame)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oof_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(HERE / "outputs" / "oof"))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--outer-variant", default=None)
    parser.add_argument("--middle-variant", default=None)
    parser.add_argument("--inner-variant", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: outputs/oof_artifacts/<variant>",
    )
    args = parser.parse_args()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (HERE / "outputs" / "oof_artifacts" / args.variant).resolve()
    )
    layer_variants = {
        "outer": args.outer_variant or args.variant,
        "middle": args.middle_variant or args.variant,
        "inner": args.inner_variant or args.variant,
    }
    manifest = assemble(
        Path(args.source_dir).resolve(),
        args.variant,
        output_dir,
        layer_variants=layer_variants,
    )
    print(f"[assemble OOF] {manifest['layers']} -> {output_dir}")


if __name__ == "__main__":
    main()
