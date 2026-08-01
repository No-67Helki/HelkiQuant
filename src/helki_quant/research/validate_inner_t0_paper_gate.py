from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return out


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def check(name: str, value: Any, op: str, threshold: float | int | bool) -> dict[str, Any]:
    if isinstance(threshold, bool):
        passed = bool(value) is threshold
        return {"name": name, "value": bool(value), "op": "is", "threshold": threshold, "passed": passed}
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(
    replay_path: Path,
    audit_path: Path | None,
    output_path: Path,
    *,
    min_incremental_return: float,
    min_cum_pnl: float,
    min_round_trips: int,
    min_symbols_traded: int,
    min_active_months: int,
    min_profit_factor: float,
    max_losing_months: int,
    max_daily_turnover: float,
    max_overlay_drawdown: float,
    max_top_symbol_positive_share: float,
    max_top3_positive_share: float,
    max_top3_net_share: float,
) -> dict[str, Any]:
    replay = load_json(replay_path)
    best = replay.get("best_qualified") or replay.get("best") or {}
    audit = load_json(audit_path) if audit_path and audit_path.exists() else {}

    selected = dict(best)
    selected.update(
        {
            "audit_round_trips": audit.get("round_trips"),
            "audit_orders": audit.get("orders"),
            "audit_pnl_total": audit.get("pnl_total"),
            "audit_profit_factor": audit.get("profit_factor"),
            "audit_symbols_traded": audit.get("symbols_traded"),
            "audit_active_months": audit.get("active_months"),
            "audit_losing_months": audit.get("losing_months"),
            "audit_max_overlay_drawdown": audit.get("max_overlay_drawdown"),
            "audit_top_symbol_pnl_share": audit.get("top_symbol_pnl_share"),
            "audit_top3_symbol_pnl_share": audit.get("top3_symbol_pnl_share"),
        }
    )

    top_symbol_positive = best.get("top_symbol_positive_pnl_share")
    top3_positive = best.get("top3_positive_pnl_share")
    top3_net = best.get("top3_net_pnl_share")
    if top_symbol_positive is None:
        top_symbol_positive = audit.get("top_symbol_pnl_share")
    if top3_positive is None:
        top3_positive = audit.get("top3_symbol_pnl_share")
    if top3_net is None:
        top3_net = audit.get("top3_symbol_pnl_share")

    checks = [
        check("incremental_return", best.get("incremental_return"), ">=", min_incremental_return),
        check("cum_pnl", best.get("cum_pnl", audit.get("pnl_total")), ">=", min_cum_pnl),
        check("round_trips", best.get("round_trips", audit.get("round_trips")), ">=", min_round_trips),
        check("symbols_traded", best.get("symbols_traded", audit.get("symbols_traded")), ">=", min_symbols_traded),
        check("active_months", best.get("active_months", audit.get("active_months")), ">=", min_active_months),
        check("profit_factor", best.get("profit_factor", audit.get("profit_factor")), ">=", min_profit_factor),
        check("losing_months", audit.get("losing_months"), "<=", max_losing_months),
        check("max_daily_turnover", best.get("max_daily_turnover"), "<=", max_daily_turnover),
        check("max_overlay_drawdown", audit.get("max_overlay_drawdown", best.get("max_drawdown")), "<=", max_overlay_drawdown),
        check("top_symbol_positive_pnl_share", top_symbol_positive, "<=", max_top_symbol_positive_share),
        check("top3_positive_pnl_share", top3_positive, "<=", max_top3_positive_share),
        check("top3_net_pnl_share", top3_net, "<=", max_top3_net_share),
    ]
    failed = [row for row in checks if not row["passed"]]
    result = {
        "status": "inner_t0_paper_gate_validated",
        "replay": str(replay_path.resolve()),
        "audit": str(audit_path.resolve()) if audit_path else None,
        "selected_setting": {
            "threshold": best.get("threshold"),
            "trade_fraction": best.get("trade_fraction"),
        },
        "selected_metrics": selected,
        "checks": checks,
        "passed": not failed,
        "failed_checks": failed,
        "deployment_allowed": not failed,
        "decision": "paper_candidate_allowed" if not failed else "keep_inner_t0_research_only",
        "next_required_if_failed": [
            "Improve signal breadth so more symbols and months trade.",
            "Reduce top-symbol and top-3 PnL concentration.",
            "Rerun portfolio replay with strict gates before packaging a GmQuant PAPER entrypoint.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--audit", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-incremental-return", type=float, default=0.003)
    parser.add_argument("--min-cum-pnl", type=float, default=3000.0)
    parser.add_argument("--min-round-trips", type=int, default=40)
    parser.add_argument("--min-symbols-traded", type=int, default=20)
    parser.add_argument("--min-active-months", type=int, default=8)
    parser.add_argument("--min-profit-factor", type=float, default=1.40)
    parser.add_argument("--max-losing-months", type=int, default=3)
    parser.add_argument("--max-daily-turnover", type=float, default=0.10)
    parser.add_argument("--max-overlay-drawdown", type=float, default=0.04)
    parser.add_argument("--max-top-symbol-positive-share", type=float, default=0.30)
    parser.add_argument("--max-top3-positive-share", type=float, default=0.55)
    parser.add_argument("--max-top3-net-share", type=float, default=1.00)
    args = parser.parse_args()
    result = validate(
        Path(args.replay).resolve(),
        Path(args.audit).resolve() if args.audit else None,
        Path(args.output).resolve(),
        min_incremental_return=args.min_incremental_return,
        min_cum_pnl=args.min_cum_pnl,
        min_round_trips=args.min_round_trips,
        min_symbols_traded=args.min_symbols_traded,
        min_active_months=args.min_active_months,
        min_profit_factor=args.min_profit_factor,
        max_losing_months=args.max_losing_months,
        max_daily_turnover=args.max_daily_turnover,
        max_overlay_drawdown=args.max_overlay_drawdown,
        max_top_symbol_positive_share=args.max_top_symbol_positive_share,
        max_top3_positive_share=args.max_top3_positive_share,
        max_top3_net_share=args.max_top3_net_share,
    )
    print(
        "[inner t0 paper gate] "
        f"passed={result['passed']} decision={result['decision']} "
        f"failed={len(result['failed_checks'])} output={Path(args.output).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
