from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_audit(log_dir: Path, profile_dir_name: str) -> dict:
    path = log_dir / profile_dir_name / "audit.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def final_holdings_count(log_dir: Path, profile_dir_name: str) -> int:
    path = log_dir / profile_dir_name / "holdings.csv"
    if not path.exists():
        return 0
    frame = pd.read_csv(path)
    if frame.empty:
        return 0
    last_date = frame["trade_date"].max()
    return int(frame[frame["trade_date"].eq(last_date)]["instrument"].nunique())


def summarize(
    base_log_dir: Path,
    stress_log_dir: Path,
    profile_dir_name: str,
    prefix: str,
    output_csv: Path,
    output_json: Path | None,
) -> dict:
    base = load_audit(base_log_dir, profile_dir_name)
    stress = load_audit(stress_log_dir, profile_dir_name)
    base_final_holdings = final_holdings_count(base_log_dir, profile_dir_name)
    stress_final_holdings = final_holdings_count(stress_log_dir, profile_dir_name)
    row = {
        "profile": profile_dir_name,
        "base_return": float(base["total_return"]),
        f"{prefix}_return": float(stress["total_return"]),
        "return_delta": float(stress["total_return"]) - float(base["total_return"]),
        "base_mdd": float(base["max_drawdown"]),
        f"{prefix}_mdd": float(stress["max_drawdown"]),
        "mdd_delta": float(stress["max_drawdown"]) - float(base["max_drawdown"]),
        "base_turnover": float(base["turnover"]),
        f"{prefix}_turnover": float(stress["turnover"]),
        "turnover_delta": float(stress["turnover"]) - float(base["turnover"]),
        "base_trades": int(base["trades"]),
        f"{prefix}_trades": int(stress["trades"]),
        "blocked_order_events": int(stress.get("blocked_order_events", 0)),
        "blocked_order_keys": int(stress.get("blocked_order_keys", 0)),
        "base_min_cash": float(base["min_cash"]),
        f"{prefix}_min_cash": float(stress["min_cash"]),
        "base_final_holdings": base_final_holdings,
        f"{prefix}_final_holdings": stress_final_holdings,
        "final_holdings_delta": stress_final_holdings - base_final_holdings,
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_csv, index=False, encoding="utf-8-sig")
    report = {
        "status": f"{prefix}_stress_summary_research_only",
        "base_log_dir": str(base_log_dir.resolve()),
        "stress_log_dir": str(stress_log_dir.resolve()),
        "profile_dir_name": profile_dir_name,
        "rows": [row],
        "deployment_allowed": False,
    }
    if output_json is not None:
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-log-dir", required=True)
    parser.add_argument("--stress-log-dir", required=True)
    parser.add_argument("--profile-dir-name", required=True)
    parser.add_argument("--prefix", required=True, choices=["sellblock", "staleexit"])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    report = summarize(
        Path(args.base_log_dir).resolve(),
        Path(args.stress_log_dir).resolve(),
        args.profile_dir_name,
        args.prefix,
        Path(args.output_csv).resolve(),
        Path(args.output_json).resolve() if args.output_json else None,
    )
    row = report["rows"][0]
    print(
        f"[{args.prefix} summary] "
        f"return={row[f'{args.prefix}_return']:+.2%} "
        f"mdd={row[f'{args.prefix}_mdd']:.2%} "
        f"delta={row['return_delta']:+.2%} "
        f"holdings_delta={row['final_holdings_delta']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
