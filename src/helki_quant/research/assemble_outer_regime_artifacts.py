from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def assemble(
    baseline_artifacts: Path,
    outer_oof_dir: Path,
    output_dir: Path,
    variant: str,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing artifacts: {output_dir}")
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True)
    baseline_pred = baseline_artifacts / "predictions"
    for layer in ("middle", "inner"):
        source = baseline_pred / f"pred_{layer}.csv"
        if source.exists():
            shutil.copy2(source, pred_dir / f"pred_{layer}.csv")
    frames = []
    folds = []
    for path in sorted(outer_oof_dir.glob("fold_*.csv")):
        fold = int(path.stem.split("_")[-1])
        frame = pd.read_csv(path, parse_dates=["datetime"])
        frame = frame.rename(columns={"outer": "pred_outer"})
        frames.append(frame[["datetime", "instrument", "pred_outer"]])
        folds.append(fold)
    if not frames:
        raise FileNotFoundError(f"no fold_*.csv files under {outer_oof_dir}")
    outer = pd.concat(frames, ignore_index=True).sort_values(["datetime", "instrument"])
    outer.to_csv(pred_dir / "pred_outer.csv", index=False, encoding="utf-8")
    manifest = {
        "status": "assembled_oof_research_only",
        "variant": variant,
        "baseline_artifacts": str(baseline_artifacts.resolve()),
        "outer_oof_dir": str(outer_oof_dir.resolve()),
        "layer_variants": {
            "outer": variant,
            "middle": "copied_from_baseline_artifacts",
            "inner": "copied_from_baseline_artifacts",
        },
        "layers": {
            "outer": {"folds": folds, "rows": int(len(outer))},
            "middle": {
                "source": str((baseline_pred / "pred_middle.csv").resolve()),
                "rows": int(pd.read_csv(baseline_pred / "pred_middle.csv").shape[0]),
            },
            "inner": {
                "source": str((baseline_pred / "pred_inner.csv").resolve()),
                "rows": int(pd.read_csv(baseline_pred / "pred_inner.csv").shape[0]),
            },
        },
        "deployment_allowed": False,
    }
    (output_dir / "oof_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-artifacts", required=True)
    parser.add_argument("--outer-oof-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    manifest = assemble(
        Path(args.baseline_artifacts).resolve(),
        Path(args.outer_oof_dir).resolve(),
        Path(args.output_dir).resolve(),
        args.variant,
    )
    print(
        "[assemble outer regime] "
        f"variant={manifest['variant']} outer_rows={manifest['layers']['outer']['rows']} "
        f"output={Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
