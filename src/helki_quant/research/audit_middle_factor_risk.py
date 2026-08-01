from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


SCALE_RISK_PATTERNS = ("VWAP", "WVMA")


def _load_whitelist(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("kept", []))


def _risk_reason(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    feature = str(row["feature"]).upper()
    if any(pattern in feature for pattern in SCALE_RISK_PATTERNS):
        reasons.append("qfq_amount_volume_scale_risk")
    if bool(row.get("kept", False)) and not bool(row.get("train_valid_same_sign", False)):
        reasons.append("train_valid_ic_sign_flip")
    if float(row.get("stable_ic", 0.0) or 0.0) <= 0.0:
        reasons.append("non_positive_stable_ic")
    if float(row.get("valid_nan_ratio", 0.0) or 0.0) > 0.05:
        reasons.append("valid_nan_ratio_gt_5pct")
    if abs(float(row.get("valid_mean_shift", 0.0) or 0.0)) > 0.75:
        reasons.append("large_valid_mean_shift")
    return reasons


def audit(
    factor_report: Path,
    whitelist_path: Path,
    output_path: Path,
    sanitized_whitelist_path: Path | None = None,
) -> dict:
    report = pd.read_csv(factor_report)
    kept = _load_whitelist(whitelist_path)
    report["kept"] = report["feature"].isin(kept)
    kept_report = report[report["kept"]].copy()
    kept_report["risk_reasons"] = kept_report.apply(_risk_reason, axis=1)
    flagged = kept_report[kept_report["risk_reasons"].map(bool)].copy()
    flagged["risk_reason"] = flagged["risk_reasons"].map(lambda values: ",".join(values))

    recommended_exclusions = sorted(
        flagged[
            flagged["risk_reasons"].map(
                lambda reasons: any(
                    reason
                    in {
                        "qfq_amount_volume_scale_risk",
                        "train_valid_ic_sign_flip",
                        "non_positive_stable_ic",
                    }
                    for reason in reasons
                )
            )
        ]["feature"].tolist()
    )

    sanitized_kept = [feature for feature in sorted(kept) if feature not in recommended_exclusions]
    top_positive = (
        kept_report.sort_values("stable_ic", ascending=False)
        .head(20)[
            [
                "feature",
                "stable_ic",
                "valid_cs_ic_mean",
                "valid_cs_positive_ratio",
                "stability_score",
            ]
        ]
        .to_dict(orient="records")
    )
    top_flagged = (
        flagged.sort_values(["risk_reason", "stable_ic"], ascending=[True, True])[
            [
                "feature",
                "stable_ic",
                "valid_cs_ic_mean",
                "valid_cs_positive_ratio",
                "valid_mean_shift",
                "valid_nan_ratio",
                "risk_reason",
            ]
        ]
        .to_dict(orient="records")
    )

    result = {
        "status": "middle_factor_risk_audit_research_only",
        "factor_report": str(factor_report.resolve()),
        "whitelist": str(whitelist_path.resolve()),
        "n_total_reported": int(len(report)),
        "n_kept": int(len(kept_report)),
        "n_flagged_kept": int(len(flagged)),
        "recommended_exclusions_for_next_oof_training": recommended_exclusions,
        "sanitized_whitelist_candidate": sanitized_kept,
        "sanitized_whitelist_count": int(len(sanitized_kept)),
        "top_positive_kept_factors": top_positive,
        "flagged_kept_factors": top_flagged,
        "interpretation": (
            "This audit does not change current model predictions. It defines the "
            "next OOF retraining candidate: remove scale-risk or unstable kept "
            "factors, then compare the sanitized middle model against the current "
            "frozen profile before touching any holdout."
        ),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if sanitized_whitelist_path is not None:
        source = json.loads(whitelist_path.read_text(encoding="utf-8"))
        source["kept"] = sanitized_kept
        source["n_kept"] = len(sanitized_kept)
        source["removed_by_middle_factor_risk_audit"] = recommended_exclusions
        source["source_whitelist"] = str(whitelist_path.resolve())
        source["audit"] = str(output_path.resolve())
        source["test_metrics_read_during_selection"] = False
        source["deployment_allowed"] = False
        sanitized_whitelist_path.parent.mkdir(parents=True, exist_ok=True)
        sanitized_whitelist_path.write_text(
            json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factor-report",
        default=str(
            HERE
            / "outputs"
            / "factor_reports"
            / "pit_holdout_de2_srfs_es"
            / "fold_99"
            / "factor_report_middle.csv"
        ),
    )
    parser.add_argument(
        "--whitelist",
        default=str(
            HERE
            / "outputs"
            / "factor_reports"
            / "pit_holdout_de2_srfs_es"
            / "fold_99"
            / "feature_whitelist_middle_v2.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(HERE / "outputs" / "middle_factor_risk_audit_20260606.json"),
    )
    parser.add_argument(
        "--sanitized-whitelist-output",
        default=str(
            HERE
            / "outputs"
            / "factor_reports"
            / "pit_holdout_de2_srfs_es"
            / "fold_99"
            / "feature_whitelist_middle_v2_sanitized_20260606.json"
        ),
    )
    args = parser.parse_args()
    result = audit(
        Path(args.factor_report).resolve(),
        Path(args.whitelist).resolve(),
        Path(args.output).resolve(),
        Path(args.sanitized_whitelist_output).resolve(),
    )
    print(
        "[middle factor audit] "
        f"kept={result['n_kept']} flagged={result['n_flagged_kept']} "
        f"exclude_next={len(result['recommended_exclusions_for_next_oof_training'])}"
    )
    for feature in result["recommended_exclusions_for_next_oof_training"][:20]:
        print(f"  exclude_next: {feature}")


if __name__ == "__main__":
    main()
