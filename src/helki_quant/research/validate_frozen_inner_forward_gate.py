from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, value: Any, op: str, threshold: Any) -> dict[str, Any]:
    if op == "is":
        passed = value is threshold
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = np.nan
        if op == ">=":
            passed = bool(np.isfinite(number) and number >= float(threshold))
        elif op == ">":
            passed = bool(np.isfinite(number) and number > float(threshold))
        elif op == "==":
            passed = bool(np.isfinite(number) and np.isclose(number, float(threshold)))
        else:
            raise ValueError(f"unsupported operation: {op}")
        value = None if not np.isfinite(number) else number
    return {"name": name, "value": value, "op": op, "threshold": threshold, "passed": passed}


def _result(report: dict[str, Any], touch_buffer: float) -> dict[str, Any]:
    for row in report.get("results", []):
        if np.isclose(float(row.get("touch_buffer", np.nan)), touch_buffer):
            return row
    raise ValueError(f"tick replay lacks touch_buffer={touch_buffer}")


def validate(
    score_path: Path,
    buy_tick_path: Path,
    sell_tick_path: Path,
    frozen_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    scores = _load(score_path)
    buy_tick = _load(buy_tick_path)
    sell_tick = _load(sell_tick_path)
    frozen = _load(frozen_manifest_path)
    buy_score = scores["directions"]["buy_first"]
    sell_score = scores["directions"]["sell_first"]
    buy_base = _result(buy_tick, 0.0)
    buy_stress = _result(buy_tick, 0.001)
    sell_base = _result(sell_tick, 0.0)
    sell_stress = _result(sell_tick, 0.001)
    profile = frozen["frozen_profile"]

    checks = [
        _check("strictly_after_calibration", scores.get("strictly_after_calibration"), "is", True),
        _check("evaluation_dates", scores.get("evaluation_dates"), ">=", 30),
        _check("buy_auc", buy_score.get("auc"), ">=", 0.55),
        _check("sell_auc", sell_score.get("auc"), ">=", 0.55),
        _check("buy_selected_dates", buy_score.get("selected_dates_before_top_n"), ">=", 10),
        _check("sell_selected_dates", sell_score.get("selected_dates_before_top_n"), ">=", 10),
        _check("buy_base_round_trips", buy_base.get("round_trips"), ">=", 20),
        _check("sell_base_round_trips", sell_base.get("round_trips"), ">=", 10),
        _check("buy_base_pnl", buy_base.get("cum_pnl"), ">", 0.0),
        _check("sell_base_pnl", sell_base.get("cum_pnl"), ">", 0.0),
        _check("buy_base_profit_factor", buy_base.get("profit_factor"), ">=", 1.10),
        _check("sell_base_profit_factor", sell_base.get("profit_factor"), ">=", 1.10),
        _check("buy_stress_pnl", buy_stress.get("cum_pnl"), ">", 0.0),
        _check("sell_stress_pnl", sell_stress.get("cum_pnl"), ">", 0.0),
        _check(
            "combined_base_pnl",
            float(buy_base.get("cum_pnl", 0.0)) + float(sell_base.get("cum_pnl", 0.0)),
            ">",
            0.0,
        ),
        _check(
            "combined_stress_pnl",
            float(buy_stress.get("cum_pnl", 0.0)) + float(sell_stress.get("cum_pnl", 0.0)),
            ">",
            0.0,
        ),
        _check(
            "buy_trigger_frozen",
            buy_tick["grid"]["trigger_distances"][0],
            "==",
            profile["buy_first"]["trigger_distance"],
        ),
        _check(
            "sell_trigger_frozen",
            sell_tick["grid"]["trigger_distances"][0],
            "==",
            profile["sell_first"]["trigger_distance"],
        ),
    ]
    failed = [row for row in checks if not row["passed"]]
    report = {
        "status": "frozen_inner_t0_forward_economic_gate_validated",
        "sources": {
            "scores": str(score_path.resolve()),
            "buy_tick": str(buy_tick_path.resolve()),
            "sell_tick": str(sell_tick_path.resolve()),
            "frozen_manifest": str(frozen_manifest_path.resolve()),
        },
        "evaluation_window": {
            "start": scores.get("evaluation_start"),
            "end": scores.get("evaluation_end"),
            "dates": scores.get("evaluation_dates"),
            "strictly_after_calibration": scores.get("strictly_after_calibration"),
        },
        "profile_was_not_reoptimized_on_forward_outcomes": True,
        "base": {"buy_first": buy_base, "sell_first": sell_base},
        "touch_stress_0p1pct": {"buy_first": buy_stress, "sell_first": sell_stress},
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
        "research_gate_passed": not failed,
        "paper_ready": False,
        "decision": (
            "advance_to_held_only_platform_dry_run"
            if not failed
            else "withdraw_frozen_candidate_and_rebuild_trigger_aligned_labels"
        ),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--buy-tick", required=True)
    parser.add_argument("--sell-tick", required=True)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = validate(
        Path(args.scores).resolve(),
        Path(args.buy_tick).resolve(),
        Path(args.sell_tick).resolve(),
        Path(args.frozen_manifest).resolve(),
        Path(args.output).resolve(),
    )
    print(
        "[inner frozen forward gate] "
        f"passed={report['passed']} failed={len(report['failed_checks'])} "
        f"decision={report['decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
