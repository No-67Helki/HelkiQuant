from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(source_dir: Path, output_dir: Path, report_path: Path) -> dict:
    source_paths = sorted(source_dir.glob("fold_*/feature_whitelist_middle_v2.json"))
    if not source_paths:
        raise ValueError(f"no middle whitelists found under {source_dir}")

    cumulative: set[str] | None = None
    rows = []
    previous_fold = 0
    for source_path in source_paths:
        fold = int(source_path.parent.name.split("_")[-1])
        if fold != previous_fold + 1:
            raise ValueError(f"non-contiguous whitelist folds: previous={previous_fold} current={fold}")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        ordered = [str(feature) for feature in source.get("kept", [])]
        current = set(ordered)
        if not current:
            raise ValueError(f"empty whitelist: {source_path}")
        cumulative = current if cumulative is None else cumulative & current
        if not cumulative:
            raise ValueError(f"expanding intersection became empty at fold {fold}")

        # Keep the current fold's original feature order while applying only
        # information available no later than that fold.
        kept = [feature for feature in ordered if feature in cumulative]
        output = dict(source)
        output.update(
            {
                "kept": kept,
                "n_kept": len(kept),
                "selection_policy": "causal_expanding_intersection",
                "source_folds": list(range(1, fold + 1)),
                "source_whitelists": [
                    str(
                        (
                            source_dir
                            / f"fold_{source_fold:02d}"
                            / "feature_whitelist_middle_v2.json"
                        ).resolve()
                    )
                    for source_fold in range(1, fold + 1)
                ],
                "test_metrics_read_during_selection": False,
                "deployment_allowed": False,
            }
        )
        destination = output_dir / f"fold_{fold:02d}" / "feature_whitelist_middle_v2.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(
            {
                "fold": fold,
                "source_count": len(current),
                "expanding_intersection_count": len(kept),
                "output": str(destination.resolve()),
            }
        )
        previous_fold = fold

    report = {
        "status": "causal_expanding_middle_whitelists_built",
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "folds": rows,
        "final_stable_features": kept,
        "final_stable_count": len(kept),
        "leakage_policy": (
            "Fold i intersects only train/valid-selected whitelists from folds 1..i. "
            "No later-fold whitelist is used for an earlier test fold."
        ),
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
    args = parser.parse_args()
    report = build(
        Path(args.source_dir).resolve(),
        Path(args.output_dir).resolve(),
        Path(args.report).resolve(),
    )
    counts = ", ".join(
        f"f{row['fold']}={row['expanding_intersection_count']}" for row in report["folds"]
    )
    print(f"[middle stable whitelist] {counts}", flush=True)


if __name__ == "__main__":
    main()
