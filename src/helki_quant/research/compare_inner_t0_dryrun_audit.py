from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_AUDIT_DIR = REPO_ROOT / "outputs" / "gm_inner_t0_dryrun_audit"
DEFAULT_REPLAY_TRADES = (
    HERE
    / "outputs"
    / "held_intraday_anchored_replay_0935_to_1445_live_features_20260611_trades.csv"
)
DEFAULT_FORBIDDEN = REPO_ROOT / "gm_c_forbidden_symbols.csv"
DEFAULT_OUTPUT = HERE / "outputs" / "inner_t0_dryrun_audit_compare_latest.json"


def local_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if "." in text:
        exchange, code = text.split(".", 1)
        return ("SH" if exchange in {"SHSE", "SH"} else "SZ") + code
    if text.startswith(("SH", "SZ")):
        return text
    code = text[-6:]
    return ("SH" if code.startswith(("6", "9")) else "SZ") + code


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_audit_run(audit_root: Path, include_mock: bool = False) -> Path:
    if not audit_root.exists():
        raise FileNotFoundError(f"audit root not found: {audit_root}")
    runs = [path for path in audit_root.iterdir() if path.is_dir()]
    if not include_mock:
        runs = [path for path in runs if not path.name.lower().startswith("mock_")]
    if not runs:
        raise FileNotFoundError(f"no audit run directories under: {audit_root} include_mock={include_mock}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_forbidden(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, dtype=str).fillna("")
    symbols: set[str] = set()
    for col in frame.columns:
        if "symbol" in col.lower() or "code" in col.lower() or "instrument" in col.lower():
            symbols.update(frame[col].map(local_symbol).tolist())
    return {item for item in symbols if item}


def normalize_sell_intents(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "instrument",
                "score",
                "intent_volume",
                "sell_price_ref",
                "action",
                "dry_run",
            ]
        )
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    symbol_col = "local_symbol" if "local_symbol" in out.columns else "symbol"
    out["instrument"] = out[symbol_col].map(local_symbol)
    out["score"] = pd.to_numeric(out.get("score"), errors="coerce")
    out["intent_volume"] = pd.to_numeric(out.get("intent_volume"), errors="coerce").fillna(0).astype(int)
    out["sell_price_ref"] = pd.to_numeric(out.get("sell_price_ref"), errors="coerce")
    return out


def normalize_buybacks(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "instrument", "intent_volume", "buy_price_ref", "action"])
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["instrument"] = out["symbol"].map(local_symbol)
    out["intent_volume"] = pd.to_numeric(out.get("intent_volume"), errors="coerce").fillna(0).astype(int)
    out["buy_price_ref"] = pd.to_numeric(out.get("buy_price_ref"), errors="coerce")
    return out


def normalize_replay(frame: pd.DataFrame, threshold: float, trade_fraction: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "instrument", "score", "t0_volume", "sell_price", "buy_price"])
    out = frame.copy()
    out = out[
        (pd.to_numeric(out["threshold"], errors="coerce").round(6) == round(threshold, 6))
        & (pd.to_numeric(out["trade_fraction"], errors="coerce").round(6) == round(trade_fraction, 6))
    ].copy()
    if out.empty:
        return out
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
    out["instrument"] = out["instrument"].map(local_symbol)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["t0_volume"] = pd.to_numeric(out["t0_volume"], errors="coerce").fillna(0).astype(int)
    out["sell_price"] = pd.to_numeric(out["sell_price"], errors="coerce")
    out["buy_price"] = pd.to_numeric(out["buy_price"], errors="coerce")
    return out


def compare(
    audit_run: Path,
    replay_trades: Path,
    forbidden_path: Path,
    output_json: Path,
    threshold: float,
    trade_fraction: float,
    max_symbols_per_day: int,
    max_daily_turnover: float,
) -> dict:
    summary = load_json(audit_run / "summary.json")
    sells = normalize_sell_intents(read_csv_or_empty(audit_run / "sell_intents.csv"))
    buybacks = normalize_buybacks(read_csv_or_empty(audit_run / "buyback_intents.csv"))
    replay = normalize_replay(read_csv_or_empty(replay_trades), threshold, trade_fraction)
    forbidden = load_forbidden(forbidden_path)

    selected = sells[sells["action"].astype(str).str.upper().eq("SELL_INTENT")].copy()
    skipped = sells[~sells.index.isin(selected.index)].copy()
    action_counts = (
        sells["action"].astype(str).str.upper().value_counts().sort_index().to_dict()
        if "action" in sells.columns and not sells.empty
        else {}
    )
    buyback_action_counts = (
        buybacks["action"].astype(str).str.upper().value_counts().sort_index().to_dict()
        if "action" in buybacks.columns and not buybacks.empty
        else {}
    )

    sell_keys = selected[["trade_date", "instrument"]].drop_duplicates()
    valid_buybacks = buybacks[buybacks["action"].astype(str).str.upper().eq("BUYBACK_INTENT")].copy()
    buy_keys = valid_buybacks[["trade_date", "instrument"]].drop_duplicates()
    buy_match = sell_keys.merge(buy_keys, on=["trade_date", "instrument"], how="left", indicator=True)
    unmatched_sell_buyback = buy_match[buy_match["_merge"] == "left_only"][["trade_date", "instrument"]]
    blocked_buybacks = buybacks[
        buybacks["action"].astype(str).str.upper().ne("BUYBACK_INTENT")
    ].copy()

    forbidden_hits = sorted(set(selected["instrument"]) & forbidden)
    below_threshold = selected[pd.to_numeric(selected["score"], errors="coerce") < threshold]
    non_dry_run_rows = sells[sells.get("dry_run", True).astype(str).str.upper().isin({"FALSE", "0", "NO"})]

    by_day = (
        selected.groupby("trade_date", as_index=False)
        .agg(
            selected_symbols=("instrument", "nunique"),
            selected_rows=("instrument", "size"),
            turnover_value_ref=("sell_value_ref", "sum") if "sell_value_ref" in selected.columns else ("intent_volume", "sum"),
        )
        if not selected.empty
        else pd.DataFrame(columns=["trade_date", "selected_symbols", "selected_rows", "turnover_value_ref"])
    )
    if "turnover_value_ref" in by_day.columns:
        by_day["turnover_ref_1m"] = pd.to_numeric(by_day["turnover_value_ref"], errors="coerce").fillna(0.0) / 1_000_000.0
    max_symbols_observed = int(by_day["selected_symbols"].max()) if not by_day.empty else 0
    max_turnover_observed = float(by_day["turnover_ref_1m"].max()) if not by_day.empty else 0.0

    overlap_report: dict[str, object] = {
        "overlap_dates": 0,
        "gm_selected_on_overlap": 0,
        "local_selected_on_overlap": 0,
        "matched_keys": 0,
        "gm_not_in_local": [],
        "local_not_in_gm": [],
    }
    if not selected.empty and not replay.empty:
        overlap_dates = sorted(set(selected["trade_date"]) & set(replay["trade_date"]))
        if overlap_dates:
            gm_overlap = sell_keys[sell_keys["trade_date"].isin(overlap_dates)]
            local_overlap = replay[replay["trade_date"].isin(overlap_dates)][["trade_date", "instrument"]].drop_duplicates()
            merged = gm_overlap.merge(local_overlap, on=["trade_date", "instrument"], how="outer", indicator=True)
            overlap_report = {
                "overlap_dates": len(overlap_dates),
                "gm_selected_on_overlap": int(len(gm_overlap)),
                "local_selected_on_overlap": int(len(local_overlap)),
                "matched_keys": int((merged["_merge"] == "both").sum()),
                "gm_not_in_local": merged.loc[merged["_merge"] == "left_only", ["trade_date", "instrument"]]
                .head(50)
                .to_dict("records"),
                "local_not_in_gm": merged.loc[merged["_merge"] == "right_only", ["trade_date", "instrument"]]
                .head(50)
                .to_dict("records"),
            }

    errors = []
    warnings = []
    if summary.get("deployment_allowed") is not False:
        errors.append("summary.deployment_allowed is not false")
    if summary.get("dry_run") is not True:
        errors.append("summary.dry_run is not true")
    if len(non_dry_run_rows):
        errors.append("sell_intents contains non-dry-run rows")
    if forbidden_hits:
        errors.append("selected intents include forbidden symbols")
    if len(below_threshold):
        errors.append("selected intents include scores below threshold")
    if len(unmatched_sell_buyback):
        errors.append("some sell intents have no same-day successful buyback intent")
    if len(blocked_buybacks):
        errors.append("buyback contains blocked or non-intent rows")
    if max_symbols_observed > max_symbols_per_day:
        errors.append("selected symbols per day exceed limit")
    if max_turnover_observed > max_daily_turnover:
        warnings.append("reference turnover exceeds dry-run limit")
    if selected.empty:
        warnings.append("no selected sell intents; this can be normal if no held symbol crosses threshold")
    if not replay_trades.exists():
        warnings.append("local replay trades file missing; overlap comparison skipped")
    elif not overlap_report["overlap_dates"]:
        warnings.append("no date overlap with local replay; only self-consistency checks were applied")

    report = {
        "status": "passed" if not errors else "failed",
        "audit_run": str(audit_run.resolve()),
        "replay_trades": str(replay_trades.resolve()),
        "threshold": threshold,
        "trade_fraction": trade_fraction,
        "summary": summary,
        "counts": {
            "sell_intent_rows": int(len(sells)),
            "sell_selected_rows": int(len(selected)),
            "sell_skipped_rows": int(len(skipped)),
            "sell_action_counts": action_counts,
            "buyback_rows": int(len(buybacks)),
            "buyback_action_counts": buyback_action_counts,
            "blocked_buyback_rows": int(len(blocked_buybacks)),
            "selected_symbols": int(selected["instrument"].nunique()) if not selected.empty else 0,
            "selected_dates": int(selected["trade_date"].nunique()) if not selected.empty else 0,
            "forbidden_hits": len(forbidden_hits),
            "below_threshold_rows": int(len(below_threshold)),
            "unmatched_sell_buyback_rows": int(len(unmatched_sell_buyback)),
        },
        "limits": {
            "max_symbols_per_day": max_symbols_per_day,
            "max_daily_turnover": max_daily_turnover,
            "observed_max_symbols_per_day": max_symbols_observed,
            "observed_max_turnover_ref_1m": max_turnover_observed,
        },
        "forbidden_hits": forbidden_hits[:50],
        "unmatched_sell_buyback": unmatched_sell_buyback.head(50).to_dict("records"),
        "blocked_buybacks": blocked_buybacks.head(50).to_dict("records"),
        "overlap": overlap_report,
        "errors": errors,
        "warnings": warnings,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-run", default="")
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--replay-trades", default=str(DEFAULT_REPLAY_TRADES))
    parser.add_argument("--forbidden", default=str(DEFAULT_FORBIDDEN))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--trade-fraction", type=float, default=0.30)
    parser.add_argument("--max-symbols-per-day", type=int, default=20)
    parser.add_argument("--max-daily-turnover", type=float, default=0.03)
    parser.add_argument("--include-mock", action="store_true")
    args = parser.parse_args()

    try:
        audit_run = (
            Path(args.audit_run).resolve()
            if args.audit_run
            else latest_audit_run(Path(args.audit_root).resolve(), include_mock=args.include_mock)
        )
        if not audit_run.exists():
            raise FileNotFoundError(f"audit run not found: {audit_run}")
    except FileNotFoundError as exc:
        output_path = Path(args.output_json).resolve()
        report = {
            "status": "waiting_for_real_audit",
            "audit_root": str(Path(args.audit_root).resolve()),
            "include_mock": args.include_mock,
            "message": str(exc),
            "errors": [],
            "warnings": ["no real GmQuant dry-run audit found"],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            "[inner t0 dryrun compare] "
            f"status={report['status']} output={output_path}",
            flush=True,
        )
        return
    report = compare(
        audit_run,
        Path(args.replay_trades).resolve(),
        Path(args.forbidden).resolve(),
        Path(args.output_json).resolve(),
        args.threshold,
        args.trade_fraction,
        args.max_symbols_per_day,
        args.max_daily_turnover,
    )
    print(
        "[inner t0 dryrun compare] "
        f"status={report['status']} selected={report['counts']['sell_selected_rows']} "
        f"buybacks={report['counts']['buyback_rows']} errors={len(report['errors'])} "
        f"warnings={len(report['warnings'])} output={Path(args.output_json).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
