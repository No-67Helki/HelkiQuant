from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_held_intraday_decision_dataset import (  # noqa: E402
    CONTEXT_FEATURE_COLUMNS,
    TRIGGER_BUYBACK_WINDOWS,
    add_cross_sectional_features,
    add_execution_aligned_labels,
    add_position_dependent_features,
    add_trigger_aligned_labels,
    build_dataset,
    normalize_inst,
)
from held_intraday_factor_engineering import add_realtime_reproducible_factors  # noqa: E402


CONTEXT_KEYS = ["datetime", "trade_date", "instrument"]
DECISION_KEYS = CONTEXT_KEYS + ["decision_minute"]


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    missing = [col for col in CONTEXT_KEYS if col not in out]
    if missing:
        raise ValueError(f"missing context key columns: {missing}")
    out["datetime"] = pd.to_datetime(out["datetime"], errors="raise").dt.normalize()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="raise").dt.normalize()
    out["instrument"] = out["instrument"].map(normalize_inst)
    return out


def _anti_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    marker = left.merge(
        right[CONTEXT_KEYS].drop_duplicates(),
        on=CONTEXT_KEYS,
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    return marker.loc[marker["_merge"] == "left_only", left.columns].copy()


def partition_context(
    old_context: pd.DataFrame,
    new_context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old = _normalize_keys(old_context)
    new = _normalize_keys(new_context)
    for name, frame in (("old", old), ("new", new)):
        duplicate = frame.duplicated(CONTEXT_KEYS, keep=False)
        if duplicate.any():
            sample = frame.loc[duplicate, CONTEXT_KEYS].head(5).to_dict("records")
            raise ValueError(f"{name} context has duplicate keys: {sample}")
    shared = new.merge(
        old[CONTEXT_KEYS],
        on=CONTEXT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    new_only = _anti_join(new, old)
    old_only = _anti_join(old, new)
    return shared, new_only, old_only


def refresh_context_dependent_columns(
    cached_rows: pd.DataFrame,
    new_context: pd.DataFrame,
    *,
    sell_cost: float,
    buy_cost: float,
    slippage: float,
) -> pd.DataFrame:
    if cached_rows.empty:
        return cached_rows.copy()
    original_columns = list(cached_rows.columns)
    rows = _normalize_keys(cached_rows)
    context = _normalize_keys(new_context)
    missing_context_cols = [col for col in CONTEXT_FEATURE_COLUMNS if col not in context]
    if missing_context_cols:
        raise ValueError(f"new context missing feature columns: {missing_context_cols}")
    context_values = context[CONTEXT_KEYS + CONTEXT_FEATURE_COLUMNS].copy()
    rows = rows.drop(
        columns=[col for col in CONTEXT_FEATURE_COLUMNS if col in rows],
        errors="ignore",
    ).merge(
        context_values,
        on=CONTEXT_KEYS,
        how="left",
        validate="many_to_one",
        indicator="_context_merge",
    )
    missing = rows["_context_merge"] != "both"
    if missing.any():
        sample = rows.loc[missing, CONTEXT_KEYS].head(5).to_dict("records")
        raise ValueError(f"decision rows missing new context: {sample}")
    rows = rows.drop(columns="_context_merge").replace([np.inf, -np.inf], np.nan)
    rows = add_position_dependent_features(
        rows,
        sell_cost=sell_cost,
        buy_cost=buy_cost,
    )
    rows = add_execution_aligned_labels(
        rows,
        sell_cost=sell_cost,
        buy_cost=buy_cost,
        slippage=slippage,
    )
    for buyback_window in TRIGGER_BUYBACK_WINDOWS:
        rows = add_trigger_aligned_labels(
            rows,
            sell_cost=sell_cost,
            buy_cost=buy_cost,
            slippage=slippage,
            buyback_window=buyback_window,
        )
    rows = add_realtime_reproducible_factors(rows)
    rows = add_cross_sectional_features(rows)
    ordered = [col for col in original_columns if col in rows]
    ordered.extend(col for col in rows.columns if col not in ordered)
    return rows[ordered]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild_from_cache(
    *,
    cached_decision_path: Path,
    cached_context_path: Path,
    new_context_path: Path,
    output_csv: Path,
    output_json: Path,
    incremental_dir: Path,
    stage_dir: Path | None,
    stage_only: bool,
    sell_cost: float,
    buy_cost: float,
    slippage: float,
    minute_builder: Callable[..., dict] = build_dataset,
) -> dict:
    print("[held cache] loading contexts", flush=True)
    old_context = pd.read_csv(cached_context_path)
    new_context = pd.read_csv(new_context_path)
    shared, new_only, old_only = partition_context(old_context, new_context)
    print(
        "[held cache] "
        f"old={len(old_context)} new={len(new_context)} shared={len(shared)} "
        f"new_only={len(new_only)} old_only={len(old_only)} "
        f"new_only_instruments={new_only['instrument'].nunique()}",
        flush=True,
    )

    cached = _normalize_keys(pd.read_csv(cached_decision_path))
    cached_shared = cached.merge(
        shared,
        on=CONTEXT_KEYS,
        how="inner",
        validate="many_to_one",
    )
    incremental_dir.mkdir(parents=True, exist_ok=True)
    new_only_context_path = incremental_dir / "new_only_context.csv"
    incremental_csv = incremental_dir / "new_only_decision.csv"
    incremental_json = incremental_dir / "new_only_decision.json"
    new_only.to_csv(new_only_context_path, index=False, encoding="utf-8-sig")

    incremental_report = None
    if new_only.empty:
        incremental = pd.DataFrame(columns=cached.columns)
        incremental.to_csv(incremental_csv, index=False, encoding="utf-8-sig")
    else:
        start = new_only["trade_date"].min().strftime("%Y-%m-%d")
        end = new_only["trade_date"].max().strftime("%Y-%m-%d")
        print(
            "[held cache] reading raw minute data only for "
            f"{new_only['instrument'].nunique()} incremental instruments {start}..{end}",
            flush=True,
        )
        incremental_report = minute_builder(
            new_only_context_path,
            incremental_csv,
            incremental_json,
            start=start,
            end=end,
            max_instruments=0,
            stage_dir=stage_dir,
            stage_only=stage_only,
            sell_cost=sell_cost,
            buy_cost=buy_cost,
            slippage=slippage,
        )
        incremental = _normalize_keys(pd.read_csv(incremental_csv))

    print(
        f"[held cache] cached_shared_rows={len(cached_shared)} "
        f"incremental_rows={len(incremental)}; recomputing labels and snapshots",
        flush=True,
    )
    combined = pd.concat([cached_shared, incremental], ignore_index=True, sort=False)
    if combined.duplicated(DECISION_KEYS, keep=False).any():
        sample = combined.loc[
            combined.duplicated(DECISION_KEYS, keep=False), DECISION_KEYS
        ].head(5)
        raise ValueError(f"duplicate decision keys after cache merge: {sample.to_dict('records')}")
    refreshed = refresh_context_dependent_columns(
        combined,
        new_context,
        sell_cost=sell_cost,
        buy_cost=buy_cost,
        slippage=slippage,
    ).sort_values(["trade_date", "instrument", "decision_minute"])

    represented = refreshed[CONTEXT_KEYS].drop_duplicates()
    normalized_new = _normalize_keys(new_context)
    missing_context = _anti_join(normalized_new, represented)
    for col in ("datetime", "trade_date"):
        refreshed[col] = refreshed[col].dt.strftime("%Y-%m-%d")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    refreshed.to_csv(output_csv, index=False, encoding="utf-8-sig")

    report = {
        "status": "held_intraday_cache_increment_rebuilt",
        "cached_decision_path": str(cached_decision_path.resolve()),
        "cached_context_path": str(cached_context_path.resolve()),
        "new_context_path": str(new_context_path.resolve()),
        "output_csv": str(output_csv.resolve()),
        "incremental_dir": str(incremental_dir.resolve()),
        "costs": {
            "sell_cost": sell_cost,
            "buy_cost": buy_cost,
            "slippage": slippage,
        },
        "context_keys": {
            "old": int(len(old_context)),
            "new": int(len(new_context)),
            "shared": int(len(shared)),
            "new_only": int(len(new_only)),
            "old_only": int(len(old_only)),
            "new_only_instruments": int(new_only["instrument"].nunique()),
            "represented": int(len(represented)),
            "missing_minute_or_snapshot": int(len(missing_context)),
        },
        "decision_rows": {
            "cached_shared": int(len(cached_shared)),
            "incremental": int(len(incremental)),
            "final": int(len(refreshed)),
            "dates": int(refreshed["trade_date"].nunique()),
            "instruments": int(refreshed["instrument"].nunique()),
            "duplicate_keys": int(refreshed.duplicated(DECISION_KEYS).sum()),
        },
        "incremental_report": incremental_report,
        "source_sha256": {
            "cached_decision": _sha256(cached_decision_path),
            "cached_context": _sha256(cached_context_path),
            "new_context": _sha256(new_context_path),
        },
        "selection_policy": "frozen profile only; no threshold or Top-N reselection",
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[held cache] complete "
        f"rows={len(refreshed)} dates={report['decision_rows']['dates']} "
        f"instruments={report['decision_rows']['instruments']} "
        f"missing_context={len(missing_context)}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-decision", required=True)
    parser.add_argument("--cached-context", required=True)
    parser.add_argument("--new-context", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--incremental-dir", required=True)
    parser.add_argument("--stage-dir", default=None)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--sell-cost", type=float, default=0.0025)
    parser.add_argument("--buy-cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    args = parser.parse_args()
    rebuild_from_cache(
        cached_decision_path=Path(args.cached_decision).resolve(),
        cached_context_path=Path(args.cached_context).resolve(),
        new_context_path=Path(args.new_context).resolve(),
        output_csv=Path(args.output_csv).resolve(),
        output_json=Path(args.output_json).resolve(),
        incremental_dir=Path(args.incremental_dir).resolve(),
        stage_dir=Path(args.stage_dir).resolve() if args.stage_dir else None,
        stage_only=args.stage_only,
        sell_cost=args.sell_cost,
        buy_cost=args.buy_cost,
        slippage=args.slippage,
    )


if __name__ == "__main__":
    main()
