from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    overall = report.get("overall", {})
    return {
        "variant": report.get("variant", path.stem),
        "path": str(path.resolve()),
        "rows": int(overall.get("rows", 0) or 0),
        "auc": overall.get("auc_positive"),
        "spearman": overall.get("spearman"),
        "daily_ic_mean": overall.get("daily_ic_mean"),
        "daily_ic_ir": overall.get("daily_ic_ir"),
        "worst_fold_auc": report.get("worst_fold_auc"),
        "worst_fold_spearman": report.get("worst_fold_spearman"),
        "label_positive_ratio": overall.get("label_positive_ratio"),
        "positive_threshold": report.get("positive_threshold"),
        "deployment_allowed": bool(report.get("deployment_allowed", False)),
    }


def score(row: dict) -> float:
    auc = float(row.get("auc") or 0.0)
    worst_auc = float(row.get("worst_fold_auc") or 0.0)
    spearman = float(row.get("spearman") or 0.0)
    worst_spearman = float(row.get("worst_fold_spearman") or 0.0)
    return auc + worst_auc + 0.25 * spearman + 0.10 * worst_spearman


def compare(paths: list[Path], output_json: Path, output_csv: Path) -> dict:
    rows = [load_report(path) for path in paths]
    for row in rows:
        row["diagnostic_score"] = score(row)
    rows = sorted(rows, key=lambda item: item["diagnostic_score"], reverse=True)
    result = {
        "status": "inner_oof_variants_compared",
        "best_variant": rows[0]["variant"] if rows else None,
        "decision": (
            "inner_candidate_for_replay"
            if rows
            and (rows[0].get("worst_fold_auc") or 0.0) >= 0.55
            and (rows[0].get("spearman") or 0.0) > 0.05
            else "keep_inner_research_only"
        ),
        "decision_rule": "Candidate requires worst_fold_auc >= 0.55 and overall spearman > 0.05 before replay integration.",
        "deployment_allowed": False,
        "variants": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    result = compare(
        [Path(path).resolve() for path in args.reports],
        Path(args.output_json).resolve(),
        Path(args.output_csv).resolve(),
    )
    best = result["variants"][0] if result["variants"] else {}
    print(
        "[inner compare] "
        f"decision={result['decision']} best={best.get('variant')} "
        f"auc={best.get('auc')} worst_auc={best.get('worst_fold_auc')} "
        f"spearman={best.get('spearman')}"
    )


if __name__ == "__main__":
    main()
