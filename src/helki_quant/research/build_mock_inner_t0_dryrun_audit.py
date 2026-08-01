from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_REPLAY_TRADES = (
    HERE
    / "outputs"
    / "held_intraday_anchored_replay_0935_to_1445_live_features_20260611_trades.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "gm_inner_t0_dryrun_audit"


def local_to_gm(symbol: object) -> str:
    text = str(symbol).strip().upper()
    code = text[-6:]
    if text.startswith("SH") or code.startswith(("6", "9")):
        return f"SHSE.{code}"
    return f"SZSE.{code}"


def build_mock_audit(
    replay_trades: Path,
    output_root: Path,
    *,
    run_id: str,
    max_rows: int,
    threshold: float,
    trade_fraction: float,
) -> dict:
    trades = pd.read_csv(replay_trades)
    trades = trades[
        (pd.to_numeric(trades["threshold"], errors="coerce").round(6) == round(threshold, 6))
        & (pd.to_numeric(trades["trade_fraction"], errors="coerce").round(6) == round(trade_fraction, 6))
    ].copy()
    if trades.empty:
        raise ValueError("no matching replay trades for requested threshold/fraction")
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.strftime("%Y-%m-%d")
    trades = trades.sort_values(["trade_date", "instrument", "score"], ascending=[True, True, False]).head(max_rows)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sell_rows = []
    buy_rows = []
    for row in trades.itertuples(index=False):
        gm_symbol = local_to_gm(row.instrument)
        sell_rows.append(
            {
                "date": row.trade_date,
                "time": "09:35:00",
                "phase": "morning_sell",
                "dry_run": True,
                "symbol": gm_symbol,
                "local_symbol": row.instrument,
                "score": float(row.score),
                "score_threshold": threshold,
                "held_volume": int(row.held_shares),
                "available_volume": int(row.held_shares),
                "intent_volume": int(row.t0_volume),
                "sell_price_ref": float(row.sell_price),
                "sell_value_ref": float(row.t0_volume) * float(row.sell_price),
                "sell_fee_est": float(row.sell_fee),
                "trade_fraction": trade_fraction,
                "features": "{}",
                "action": "SELL_INTENT",
            }
        )
        buy_rows.append(
            {
                "date": row.trade_date,
                "time": "14:45:00",
                "phase": "afternoon_buyback",
                "dry_run": True,
                "symbol": gm_symbol,
                "intent_volume": int(row.t0_volume),
                "buy_price_ref": float(row.buy_price),
                "buy_value_ref": float(row.t0_volume) * float(row.buy_price),
                "buy_fee_est": float(row.buy_fee),
                "action": "BUYBACK_INTENT",
                "source_sell_score": float(row.score),
                "source_sell_price": float(row.sell_price),
            }
        )
    pd.DataFrame(sell_rows).to_csv(out_dir / "sell_intents.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(buy_rows).to_csv(out_dir / "buyback_intents.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "inner_t0_dryrun_audit",
        "dry_run": True,
        "allow_orders": False,
        "score_threshold": threshold,
        "trade_fraction": trade_fraction,
        "sell_intents": len(sell_rows),
        "sell_selected": len(sell_rows),
        "buyback_intents": len(buy_rows),
        "state_symbols": sorted({row["symbol"] for row in sell_rows}),
        "deployment_allowed": False,
        "mock_from_replay": str(replay_trades.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "mock_audit_built",
        "output_dir": str(out_dir.resolve()),
        "rows": len(sell_rows),
        "dates": sorted({row["date"] for row in sell_rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-trades", default=str(DEFAULT_REPLAY_TRADES))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="mock_live_feature_pass_20260611")
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--trade-fraction", type=float, default=0.30)
    args = parser.parse_args()
    report = build_mock_audit(
        Path(args.replay_trades).resolve(),
        Path(args.output_root).resolve(),
        run_id=args.run_id,
        max_rows=args.max_rows,
        threshold=args.threshold,
        trade_fraction=args.trade_fraction,
    )
    print(
        "[inner t0 mock audit] "
        f"rows={report['rows']} output={report['output_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
