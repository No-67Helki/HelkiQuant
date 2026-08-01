from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return out


def check(name: str, value: Any, op: str, threshold: float | int) -> dict[str, Any]:
    num = as_float(value)
    if num is None:
        passed = False
    elif op == ">=":
        passed = num >= float(threshold)
    elif op == ">":
        passed = num > float(threshold)
    elif op == "<=":
        passed = num <= float(threshold)
    elif op == "<":
        passed = num < float(threshold)
    else:
        raise ValueError(f"unsupported op: {op}")
    return {"name": name, "value": num, "op": op, "threshold": threshold, "passed": passed}


def check_equal(name: str, value: Any, expected: Any) -> dict[str, Any]:
    if isinstance(expected, float):
        number = as_float(value)
        passed = number is not None and math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12)
    else:
        number = value
        passed = value == expected
    return {
        "name": name,
        "value": number,
        "op": "==",
        "threshold": expected,
        "passed": passed,
    }


def replay_profile_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is not None:
        return value

    profile = row.get("profile")
    if not isinstance(profile, dict):
        return None
    aliases = {
        "threshold": "score_threshold",
        "daily_top_n": "daily_top_n",
        "trade_fraction": "trade_fraction",
        "sizing_mode": "sizing_mode",
        "trade_direction": "direction",
    }
    profile_key = aliases.get(key, key)
    return profile.get(profile_key)


def select_replay_setting(
    replay: dict[str, Any],
    *,
    expected_threshold: float | None = None,
    expected_daily_top_n: int | None = None,
    expected_trade_fraction: float | None = None,
    expected_sizing_mode: str | None = None,
    expected_trade_direction: str | None = None,
) -> dict[str, Any]:
    expected = {
        "threshold": expected_threshold,
        "daily_top_n": expected_daily_top_n,
        "trade_fraction": expected_trade_fraction,
        "sizing_mode": expected_sizing_mode,
        "trade_direction": expected_trade_direction,
    }
    if all(value is None for value in expected.values()):
        selected = replay.get("best") or replay
        if not isinstance(selected, dict):
            raise ValueError("replay does not contain a selectable result")
        return selected

    candidates = replay.get("results") or []
    if not candidates and isinstance(replay.get("best"), dict):
        candidates = [replay["best"]]
    if not candidates and isinstance(replay.get("profile"), dict):
        candidates = [replay]
    if not candidates and isinstance(replay, dict):
        candidates = [replay]

    matches: list[dict[str, Any]] = []
    for row in candidates:
        matched = True
        for key, wanted in expected.items():
            if wanted is None:
                continue
            actual = replay_profile_value(row, key)
            if isinstance(wanted, float):
                number = as_float(actual)
                if number is None or not math.isclose(
                    number, wanted, rel_tol=0.0, abs_tol=1e-12
                ):
                    matched = False
                    break
            elif actual != wanted:
                matched = False
                break
        if matched:
            matches.append(row)

    if len(matches) != 1:
        requested = {key: value for key, value in expected.items() if value is not None}
        raise ValueError(
            f"expected exactly one replay result for {requested}, found {len(matches)}"
        )
    return matches[0]


def validate(
    replay_path: Path,
    output_path: Path,
    *,
    min_cum_pnl: float,
    min_incremental_return: float,
    min_round_trips: int,
    min_symbols_traded: int,
    min_active_months: int,
    min_profit_factor: float,
    max_top_symbol_positive_share: float,
    max_top3_positive_share: float,
    max_daily_turnover: float,
    max_overlay_drawdown: float,
    max_incremental_drawdown: float,
    min_fold_count: int,
    min_profitable_folds: int,
    min_worst_fold_return: float,
    min_round_trips_per_fold: int,
    gate_stage: str,
    expected_threshold: float | None = None,
    expected_daily_top_n: int | None = None,
    expected_trade_fraction: float | None = None,
    expected_sizing_mode: str | None = None,
    expected_trade_direction: str | None = None,
    forward_replay_path: Path | None = None,
    min_forward_cum_pnl: float = 0.0,
    min_forward_round_trips: int = 20,
    min_forward_symbols_traded: int = 10,
    min_forward_active_months: int = 3,
    min_forward_profit_factor: float = 1.2,
    max_forward_losing_month_fraction: float = 0.4,
    max_forward_top3_positive_share: float = 0.6,
) -> dict:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    selection_kwargs = {
        "expected_threshold": expected_threshold,
        "expected_daily_top_n": expected_daily_top_n,
        "expected_trade_fraction": expected_trade_fraction,
        "expected_sizing_mode": expected_sizing_mode,
        "expected_trade_direction": expected_trade_direction,
    }
    best = select_replay_setting(replay, **selection_kwargs)
    best_profile = {
        key: replay_profile_value(best, key)
        for key in (
            "threshold",
            "daily_top_n",
            "trade_fraction",
            "sizing_mode",
            "trade_direction",
        )
    }
    fold_rows = best.get("folds") or []
    minimum_fold_round_trips = min(
        (int(row.get("round_trips", 0)) for row in fold_rows),
        default=0,
    )
    checks = [
        check("cum_pnl", best.get("cum_pnl"), ">=", min_cum_pnl),
        check("incremental_return", best.get("incremental_return"), ">=", min_incremental_return),
        check("round_trips", best.get("round_trips"), ">=", min_round_trips),
        check("symbols_traded", best.get("symbols_traded"), ">=", min_symbols_traded),
        check("active_months", best.get("active_months"), ">=", min_active_months),
        check("profit_factor", best.get("profit_factor"), ">=", min_profit_factor),
        check("top_symbol_positive_pnl_share", best.get("top_symbol_positive_pnl_share"), "<=", max_top_symbol_positive_share),
        check("top3_positive_pnl_share", best.get("top3_positive_pnl_share"), "<=", max_top3_positive_share),
        check("max_daily_turnover", best.get("max_daily_turnover"), "<=", max_daily_turnover),
        check("max_overlay_drawdown", best.get("max_overlay_drawdown"), "<=", max_overlay_drawdown),
        check(
            "incremental_max_drawdown",
            best.get("incremental_max_drawdown"),
            "<=",
            max_incremental_drawdown,
        ),
        check("fold_count", best.get("fold_count"), ">=", min_fold_count),
        check("profitable_folds", best.get("profitable_folds"), ">=", min_profitable_folds),
        check("worst_fold_return", best.get("worst_fold_return"), ">=", min_worst_fold_return),
        check(
            "minimum_fold_round_trips",
            minimum_fold_round_trips,
            ">=",
            min_round_trips_per_fold,
        ),
    ]

    forward_best: dict[str, Any] | None = None
    if forward_replay_path is not None:
        forward_replay = json.loads(forward_replay_path.read_text(encoding="utf-8"))
        forward_best = select_replay_setting(forward_replay, **selection_kwargs)
        forward_profile = {
            key: replay_profile_value(forward_best, key)
            for key in best_profile
        }
        active_months = int(forward_best.get("active_months", 0) or 0)
        losing_months = int(forward_best.get("losing_months", 0) or 0)
        losing_month_fraction = (
            losing_months / active_months if active_months > 0 else None
        )
        profile_checks = [
            check_equal(
                f"forward_{key}_matches_oof",
                forward_profile[key],
                expected,
            )
            for key, expected in best_profile.items()
            if expected is not None
        ]
        checks.extend(
            profile_checks
            + [
                check("forward_cum_pnl", forward_best.get("cum_pnl"), ">", min_forward_cum_pnl),
                check(
                    "forward_round_trips",
                    forward_best.get("round_trips"),
                    ">=",
                    min_forward_round_trips,
                ),
                check(
                    "forward_symbols_traded",
                    forward_best.get("symbols_traded"),
                    ">=",
                    min_forward_symbols_traded,
                ),
                check(
                    "forward_active_months",
                    forward_best.get("active_months"),
                    ">=",
                    min_forward_active_months,
                ),
                check(
                    "forward_profit_factor",
                    forward_best.get("profit_factor"),
                    ">=",
                    min_forward_profit_factor,
                ),
                check(
                    "forward_losing_month_fraction",
                    losing_month_fraction,
                    "<=",
                    max_forward_losing_month_fraction,
                ),
                check(
                    "forward_top3_positive_pnl_share",
                    forward_best.get("top3_positive_pnl_share"),
                    "<=",
                    max_forward_top3_positive_share,
                ),
            ]
        )
    failed = [row for row in checks if not row["passed"]]
    result = {
        "status": "held_intraday_t0_gate_validated",
        "replay": str(replay_path.resolve()),
        "forward_replay": (
            str(forward_replay_path.resolve()) if forward_replay_path is not None else None
        ),
        "gate_stage": gate_stage,
        "selected_setting": {
            "selection_mode": best.get("selection_mode"),
            "threshold": best_profile["threshold"],
            "daily_top_n": best_profile["daily_top_n"],
            "trade_fraction": best_profile["trade_fraction"],
            "sizing_mode": best_profile["sizing_mode"],
            "trade_direction": best_profile["trade_direction"],
            "components": best.get("components"),
        },
        "selected_metrics": best,
        "selected_forward_metrics": forward_best,
        "checks": checks,
        "passed": not failed,
        "failed_checks": failed,
        "research_gate_passed": not failed,
        "paper_ready": bool(not failed and gate_stage == "paper_dry_run"),
        "decision": (
            "advance_to_gmquant_dry_run"
            if not failed and gate_stage == "research_oof"
            else "paper_candidate_allowed"
            if not failed
            else "keep_held_intraday_t0_research_only"
        ),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-cum-pnl", type=float, default=3000.0)
    parser.add_argument("--min-incremental-return", type=float, default=0.003)
    parser.add_argument("--min-round-trips", type=int, default=80)
    parser.add_argument("--min-symbols-traded", type=int, default=20)
    parser.add_argument("--min-active-months", type=int, default=4)
    parser.add_argument("--min-profit-factor", type=float, default=1.4)
    parser.add_argument("--max-top-symbol-positive-share", type=float, default=0.30)
    parser.add_argument("--max-top3-positive-share", type=float, default=0.55)
    parser.add_argument("--max-daily-turnover", type=float, default=0.10)
    parser.add_argument("--max-overlay-drawdown", type=float, default=0.04)
    parser.add_argument("--max-incremental-drawdown", type=float, default=0.003)
    parser.add_argument("--min-fold-count", type=int, default=6)
    parser.add_argument("--min-profitable-folds", type=int, default=6)
    parser.add_argument("--min-worst-fold-return", type=float, default=0.0)
    parser.add_argument("--min-round-trips-per-fold", type=int, default=3)
    parser.add_argument("--gate-stage", choices=["research_oof", "paper_dry_run"], default="research_oof")
    parser.add_argument("--expected-threshold", type=float, default=None)
    parser.add_argument("--expected-daily-top-n", type=int, default=None)
    parser.add_argument("--expected-trade-fraction", type=float, default=None)
    parser.add_argument("--expected-sizing-mode", default=None)
    parser.add_argument("--expected-trade-direction", default=None)
    parser.add_argument("--forward-replay", default="")
    parser.add_argument("--min-forward-cum-pnl", type=float, default=0.0)
    parser.add_argument("--min-forward-round-trips", type=int, default=20)
    parser.add_argument("--min-forward-symbols-traded", type=int, default=10)
    parser.add_argument("--min-forward-active-months", type=int, default=3)
    parser.add_argument("--min-forward-profit-factor", type=float, default=1.2)
    parser.add_argument("--max-forward-losing-month-fraction", type=float, default=0.4)
    parser.add_argument("--max-forward-top3-positive-share", type=float, default=0.6)
    args = parser.parse_args()
    report = validate(
        Path(args.replay).resolve(),
        Path(args.output).resolve(),
        min_cum_pnl=args.min_cum_pnl,
        min_incremental_return=args.min_incremental_return,
        min_round_trips=args.min_round_trips,
        min_symbols_traded=args.min_symbols_traded,
        min_active_months=args.min_active_months,
        min_profit_factor=args.min_profit_factor,
        max_top_symbol_positive_share=args.max_top_symbol_positive_share,
        max_top3_positive_share=args.max_top3_positive_share,
        max_daily_turnover=args.max_daily_turnover,
        max_overlay_drawdown=args.max_overlay_drawdown,
        max_incremental_drawdown=args.max_incremental_drawdown,
        min_fold_count=args.min_fold_count,
        min_profitable_folds=args.min_profitable_folds,
        min_worst_fold_return=args.min_worst_fold_return,
        min_round_trips_per_fold=args.min_round_trips_per_fold,
        gate_stage=args.gate_stage,
        expected_threshold=args.expected_threshold,
        expected_daily_top_n=args.expected_daily_top_n,
        expected_trade_fraction=args.expected_trade_fraction,
        expected_sizing_mode=args.expected_sizing_mode,
        expected_trade_direction=args.expected_trade_direction,
        forward_replay_path=(
            Path(args.forward_replay).resolve() if args.forward_replay else None
        ),
        min_forward_cum_pnl=args.min_forward_cum_pnl,
        min_forward_round_trips=args.min_forward_round_trips,
        min_forward_symbols_traded=args.min_forward_symbols_traded,
        min_forward_active_months=args.min_forward_active_months,
        min_forward_profit_factor=args.min_forward_profit_factor,
        max_forward_losing_month_fraction=args.max_forward_losing_month_fraction,
        max_forward_top3_positive_share=args.max_forward_top3_positive_share,
    )
    print(
        "[held intraday gate] "
        f"passed={report['passed']} decision={report['decision']} "
        f"failed={len(report['failed_checks'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
