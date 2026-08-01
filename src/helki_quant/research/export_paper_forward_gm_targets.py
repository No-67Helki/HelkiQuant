from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .capital_aware_allocation import ALLOCATION_MODES, allocate_equal_weight_lots
    from .concentration_constraints import (
        ConcentrationRules,
        concentration_snapshot,
        groups_on_date,
        load_group_metadata,
        select_with_group_cap,
    )
    from .evaluate_daily_topk_grid import load_middle_predictions
    from .export_gm_c_baseline_targets import to_gm_symbol
    from .minute_mapped_topk_replay import min_buy_lot, safe_price
    from .universe import UniverseRules, add_point_in_time_eligibility, load_price_panel
except ImportError:
    from capital_aware_allocation import ALLOCATION_MODES, allocate_equal_weight_lots
    from concentration_constraints import (
        ConcentrationRules,
        concentration_snapshot,
        groups_on_date,
        load_group_metadata,
        select_with_group_cap,
    )
    from evaluate_daily_topk_grid import load_middle_predictions
    from export_gm_c_baseline_targets import to_gm_symbol
    from minute_mapped_topk_replay import min_buy_lot, safe_price
    from universe import UniverseRules, add_point_in_time_eligibility, load_price_panel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_PREDICTION = HERE / "outputs" / "oof" / "canonical_20260605_paper_forward" / "middle" / "fold_99.csv"
DEFAULT_RAW_DAILY = DATA / "A_Stock_daily_qfq" / "daily_qfq_6.5"
DEFAULT_GROUP_METADATA = DATA / "industry_theme_pit_ffill_20260605.csv"
DEFAULT_FORBIDDEN = DATA / "forbidden_st_symbols_20260605.csv"
DEFAULT_OUTPUT_DIR = HERE / "outputs" / "gm_c_baseline_targets_paper_forward_20260605_v1_nost"


def gm_to_local_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if "." not in text:
        return text
    exchange, code = text.split(".", 1)
    prefix = "SH" if exchange in {"SHSE", "SH"} else "SZ"
    return f"{prefix}{code}"


def load_forbidden(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    frame = pd.read_csv(path, dtype=str).fillna("")
    values: set[str] = set()
    for col in ("instrument", "local_instrument"):
        if col in frame.columns:
            values.update(frame[col].astype(str).str.upper())
    for col in ("gm_symbol", "symbol"):
        if col in frame.columns:
            values.update(frame[col].map(gm_to_local_symbol).astype(str).str.upper())
    return {v for v in values if v and v != "NAN"}


def load_previous_selection(path: Path | None) -> set[str]:
    if path is None:
        return set()
    frame = load_previous_target_frame(path)
    return set(frame["instrument"])


def load_previous_target_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"previous target not found: {path}")
    frame = pd.read_csv(path, dtype={"symbol": str, "instrument": str})
    if "target_shares" in frame.columns:
        shares = pd.to_numeric(frame["target_shares"], errors="coerce").fillna(0)
        frame = frame[shares > 0].copy()
    if "instrument" in frame.columns:
        frame["instrument"] = frame["instrument"].astype(str).str.upper()
    elif "symbol" in frame.columns:
        frame["instrument"] = frame["symbol"].map(gm_to_local_symbol).astype(str).str.upper()
    else:
        raise ValueError(f"{path} must contain instrument or symbol")
    frame = frame[
        frame["instrument"].astype(str).ne("")
        & frame["instrument"].astype(str).ne("NAN")
    ].copy()
    if frame["instrument"].duplicated().any():
        raise ValueError(f"{path} contains duplicate instruments")
    return frame.reset_index(drop=True)


def load_outer_probability(path: Path, signal_ts: pd.Timestamp) -> dict:
    frame = pd.read_csv(path, parse_dates=["datetime"])
    prediction_col = next(
        (name for name in ("outer", "pred_outer") if name in frame.columns),
        None,
    )
    if prediction_col is None:
        raise ValueError(f"{path} must contain outer or pred_outer")
    values = pd.to_numeric(
        frame.loc[frame["datetime"].eq(signal_ts), prediction_col],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError(f"{path} has no finite outer prediction on {signal_ts.date()}")
    return {
        "probability": float(values.median()),
        "rows": int(len(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _previous_effective_risk_budget(frame: pd.DataFrame, top_k: int, fallback: float) -> float:
    if "effective_risk_budget" in frame.columns:
        values = pd.to_numeric(frame["effective_risk_budget"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    if "nominal_target_weight" in frame.columns:
        values = pd.to_numeric(frame["nominal_target_weight"], errors="coerce").dropna()
        if not values.empty:
            return float(values.median() * top_k)
    return float(fallback)


def _export_scheduled_carry(
    *,
    prediction_path: Path,
    previous_target_path: Path,
    forbidden_path: Path | None,
    output_dir: Path,
    forbidden: set[str],
    signal_ts: pd.Timestamp,
    trade_date: str,
    top_k: int,
    risk_budget: float,
    industry_cap: float,
    min_avg_amount: float,
    initial_cash: float,
    outer_prediction_path: Path | None,
    require_outer: bool,
    outer_stats: dict | None,
    outer_risk_threshold: float,
    outer_risk_floor: float,
    allocation_mode: str,
    min_effective_exposure_ratio: float,
    max_name_weight: float,
    rebalance_every: int,
    buffer_multiple: int,
    pause_buys_on_sell_reject: bool,
    removed_forbidden_prediction_rows: int,
    middle_last_rebalance_signal_date: str,
    trading_sessions_since_rebalance: int,
) -> dict:
    previous = load_previous_target_frame(previous_target_path)
    previous_names = set(previous["instrument"])
    carried = previous[
        ~previous["instrument"].astype(str).str.upper().isin(forbidden)
    ].copy()
    forced_forbidden_exits = sorted(previous_names - set(carried["instrument"]))
    if carried.empty:
        raise ValueError("scheduled carry has no target rows after forbidden-symbol exits")

    carried["trade_date"] = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    carried["signal_date"] = signal_ts.strftime("%Y-%m-%d")
    carried["middle_rebalance_due"] = False
    carried["middle_last_rebalance_signal_date"] = pd.Timestamp(
        middle_last_rebalance_signal_date
    ).strftime("%Y-%m-%d")
    carried["middle_trading_sessions_since_rebalance"] = int(
        trading_sessions_since_rebalance
    )
    effective_risk_budget = _previous_effective_risk_budget(
        carried, top_k, risk_budget
    )
    carried["effective_risk_budget"] = effective_risk_budget

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"paper_forward_top{top_k}_nost.csv"
    carried.to_csv(target_path, index=False, encoding="utf-8-sig")

    effective_weights = pd.to_numeric(
        carried.get("effective_weight_ref", carried.get("target_weight", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    effective_weight_sum = float(effective_weights.sum())
    requested_weight = float(effective_risk_budget)
    effective_exposure_ratio = (
        effective_weight_sum / requested_weight if requested_weight > 0 else 0.0
    )
    max_effective_name_weight = float(effective_weights.max())
    allocation_gate_passed = bool(
        effective_exposure_ratio >= min_effective_exposure_ratio
        and max_effective_name_weight <= max_name_weight
    )
    group_values = (
        carried["group"].fillna("__UNKNOWN__").astype(str)
        if "group" in carried.columns
        else pd.Series("__UNKNOWN__", index=carried.index)
    )
    groups = dict(zip(carried["instrument"].astype(str), group_values))
    weighted = pd.DataFrame(
        {"group": group_values, "weight": effective_weights}
    ).groupby("group", dropna=False)["weight"].sum()
    current_outer_triggered = bool(
        outer_stats is not None
        and outer_stats["probability"] >= outer_risk_threshold
    )
    applied_outer_triggered = bool(
        effective_risk_budget <= outer_risk_floor + 1e-12
    )
    allocated_notional = float(
        pd.to_numeric(
            carried.get("target_notional_ref", effective_weights * initial_cash),
            errors="coerce",
        ).fillna(0.0).sum()
    )
    allocation = {
        "mode": allocation_mode,
        "requested_count": top_k,
        "valid_price_count": int(len(carried)),
        "selected_count": int(len(carried)),
        "allocated_count": int(len(carried)),
        "budget_value": float(initial_cash * effective_risk_budget),
        "allocated_notional": allocated_notional,
        "effective_weight": effective_weight_sum,
        "budget_utilization": (
            allocated_notional / (initial_cash * effective_risk_budget)
            if effective_risk_budget > 0
            else 0.0
        ),
        "ideal_value_per_name": float(initial_cash * effective_risk_budget / top_k),
        "residual_value": float(
            max(0.0, initial_cash * effective_risk_budget - allocated_notional)
        ),
        "topup_lots": 0,
    }
    manifest = {
        "status": "paper_forward_gm_targets_exported",
        "deployment_allowed": False,
        "paper_simulation_candidate": True,
        "prediction": str(prediction_path.resolve()),
        "target": str(target_path.resolve()),
        "signal_date": signal_ts.strftime("%Y-%m-%d"),
        "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
        "profile": (
            f"paper_forward_top{top_k}_rb{rebalance_every}_risk{effective_risk_budget:.2f}_"
            f"cap{industry_cap:.2f}_b{buffer_multiple}_nost_{allocation_mode}"
        ),
        "selection_mode": "scheduled_carry",
        "middle_rebalance": {
            "due": False,
            "last_rebalance_signal_date": pd.Timestamp(
                middle_last_rebalance_signal_date
            ).strftime("%Y-%m-%d"),
            "trading_sessions_since_rebalance": int(
                trading_sessions_since_rebalance
            ),
            "policy": "membership_and_target_shares_change_only_on_scheduled_rebalance",
            "forced_forbidden_exits": forced_forbidden_exits,
        },
        "top_k": top_k,
        "rebalance_every": rebalance_every,
        "buffer_multiple": buffer_multiple,
        "previous_target": str(previous_target_path.resolve()),
        "previous_selection_count": int(len(previous)),
        "retained_previous_count": int(len(carried)),
        "retained_previous_instruments": sorted(carried["instrument"].tolist()),
        "risk_budget": effective_risk_budget,
        "base_risk_budget": risk_budget,
        "industry_cap": industry_cap,
        "min_avg_amount": min_avg_amount,
        "initial_cash_for_share_estimate": initial_cash,
        "rows": int(len(carried)),
        "symbols": int(carried["symbol"].nunique()),
        "target_weight_sum": float(
            pd.to_numeric(carried.get("target_weight", 0.0), errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        "effective_weight_sum": effective_weight_sum,
        "effective_exposure_ratio": effective_exposure_ratio,
        "outer_overlay": {
            "required": bool(require_outer),
            "prediction": (
                str(outer_prediction_path.resolve()) if outer_prediction_path else None
            ),
            "probability": outer_stats["probability"] if outer_stats else None,
            "prediction_rows": outer_stats["rows"] if outer_stats else 0,
            "prediction_minimum": outer_stats["minimum"] if outer_stats else None,
            "prediction_maximum": outer_stats["maximum"] if outer_stats else None,
            "threshold": outer_risk_threshold,
            "risk_floor": outer_risk_floor,
            "triggered": applied_outer_triggered,
            "current_signal_triggered": current_outer_triggered,
            "applied_on_this_release": False,
            "deferred_until_middle_rebalance": (
                current_outer_triggered != applied_outer_triggered
            ),
        },
        "allocation": allocation,
        "execution_risk_controls": {
            "pause_buys_on_sell_reject": bool(pause_buys_on_sell_reject),
            "sell_first_required": bool(pause_buys_on_sell_reject),
        },
        "allocation_gate": {
            "passed": allocation_gate_passed,
            "min_effective_exposure_ratio": min_effective_exposure_ratio,
            "max_name_weight": max_name_weight,
            "observed_effective_exposure_ratio": effective_exposure_ratio,
            "observed_max_name_weight": max_effective_name_weight,
        },
        "forbidden_path": str(forbidden_path.resolve()) if forbidden_path else None,
        "removed_forbidden_prediction_rows": int(removed_forbidden_prediction_rows),
        "forbidden_order_hits": 0,
        "group_snapshot": concentration_snapshot(carried["instrument"].tolist(), groups),
        "weighted_group_snapshot": {
            "max_group": str(weighted.idxmax()) if len(weighted) else None,
            "max_group_weight": float(weighted.max()) if len(weighted) else 0.0,
            "groups": {str(key): float(value) for key, value in weighted.items()},
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def export_targets(
    prediction_path: Path,
    output_dir: Path,
    raw_daily_dir: Path,
    group_metadata_path: Path,
    forbidden_path: Path | None,
    signal_date: str,
    trade_date: str,
    top_k: int,
    risk_budget: float,
    industry_cap: float,
    min_avg_amount: float,
    initial_cash: float,
    outer_prediction_path: Path | None = None,
    require_outer: bool = False,
    outer_risk_threshold: float = 0.50,
    outer_risk_floor: float = 0.30,
    allocation_mode: str = "fixed_topk",
    min_effective_exposure_ratio: float = 0.0,
    max_name_weight: float = 0.03,
    require_allocation_gate: bool = False,
    rebalance_every: int = 20,
    buffer_multiple: int = 2,
    pause_buys_on_sell_reject: bool = False,
    previous_target_path: Path | None = None,
    require_previous_target: bool = False,
    middle_rebalance_due: bool = True,
    middle_last_rebalance_signal_date: str | None = None,
    trading_sessions_since_rebalance: int = 0,
) -> dict:
    if allocation_mode not in ALLOCATION_MODES:
        raise ValueError(f"unsupported allocation mode: {allocation_mode}")
    if not 0 < outer_risk_floor <= risk_budget <= 1:
        raise ValueError("outer risk floor must be in (0, risk_budget]")
    if not 0 <= min_effective_exposure_ratio <= 1:
        raise ValueError("min effective exposure ratio must be in [0, 1]")
    if not 0 < max_name_weight <= 1:
        raise ValueError("max name weight must be in (0, 1]")

    predictions = load_middle_predictions(prediction_path)
    signal_ts = pd.Timestamp(signal_date)
    predictions = predictions[predictions["datetime"].eq(signal_ts)].copy()
    if predictions.empty:
        raise ValueError(f"{prediction_path} has no predictions on {signal_date}")
    forbidden = load_forbidden(forbidden_path)
    previous_selection = load_previous_selection(previous_target_path)
    previous_selection -= forbidden
    if require_previous_target and not previous_selection:
        raise ValueError("a non-empty previous target is required for buffered forward selection")
    before_forbidden = len(predictions)
    predictions = predictions[~predictions["instrument"].astype(str).str.upper().isin(forbidden)].copy()
    removed_forbidden_prediction_rows = before_forbidden - len(predictions)

    outer_stats = None
    effective_risk_budget = float(risk_budget)
    if outer_prediction_path is not None:
        outer_stats = load_outer_probability(outer_prediction_path, signal_ts)
        if outer_stats["probability"] >= outer_risk_threshold:
            effective_risk_budget = float(outer_risk_floor)
    elif require_outer:
        raise ValueError("outer prediction is required for this target export")
    if not middle_rebalance_due:
        if previous_target_path is None:
            raise ValueError("scheduled carry requires a previous target")
        if middle_last_rebalance_signal_date is None:
            raise ValueError("scheduled carry requires the last middle rebalance date")
        manifest = _export_scheduled_carry(
            prediction_path=prediction_path,
            previous_target_path=previous_target_path,
            forbidden_path=forbidden_path,
            output_dir=output_dir,
            forbidden=forbidden,
            signal_ts=signal_ts,
            trade_date=trade_date,
            top_k=top_k,
            risk_budget=risk_budget,
            industry_cap=industry_cap,
            min_avg_amount=min_avg_amount,
            initial_cash=initial_cash,
            outer_prediction_path=outer_prediction_path,
            require_outer=require_outer,
            outer_stats=outer_stats,
            outer_risk_threshold=outer_risk_threshold,
            outer_risk_floor=outer_risk_floor,
            allocation_mode=allocation_mode,
            min_effective_exposure_ratio=min_effective_exposure_ratio,
            max_name_weight=max_name_weight,
            rebalance_every=rebalance_every,
            buffer_multiple=buffer_multiple,
            pause_buys_on_sell_reject=pause_buys_on_sell_reject,
            removed_forbidden_prediction_rows=removed_forbidden_prediction_rows,
            middle_last_rebalance_signal_date=middle_last_rebalance_signal_date,
            trading_sessions_since_rebalance=trading_sessions_since_rebalance,
        )
        if require_allocation_gate and not manifest["allocation_gate"]["passed"]:
            raise RuntimeError("allocation gate failed for scheduled carry")
        return manifest

    instruments = predictions["instrument"].drop_duplicates().tolist()
    prices = load_price_panel(
        raw_daily_dir,
        instruments,
        start="2022-01-04",
        end=signal_date,
    )
    rules = UniverseRules(
        board_prefixes=("30", "68"),
        min_listing_days=250,
        liquidity_window=20,
        min_avg_amount=min_avg_amount,
        suspend_window=20,
        max_suspend_ratio=0.10,
    )
    eligible = add_point_in_time_eligibility(prices, rules)
    latest = eligible[eligible["datetime"].eq(signal_ts)].copy()
    latest = latest[latest["eligible"].fillna(False)].copy()
    latest = latest[~latest["instrument"].astype(str).str.upper().isin(forbidden)].copy()
    signal = predictions.merge(
        latest[
            [
                "instrument",
                "close",
                "avg_amount",
                "listing_days",
                "suspend_ratio",
            ]
        ],
        on="instrument",
        how="inner",
    ).sort_values("middle", ascending=False)
    if signal.empty:
        raise ValueError("No eligible forward candidates after filters")

    group_metadata = load_group_metadata(group_metadata_path, "industry")
    groups = groups_on_date(group_metadata, signal_ts)
    ranked = signal["instrument"].tolist()
    selected = select_with_group_cap(
        ranked,
        previous_selection=previous_selection,
        top_k=top_k,
        buffer_k=top_k * buffer_multiple,
        groups=groups,
        rules=ConcentrationRules(max_group_fraction=industry_cap),
    )
    selected_signal = signal.set_index("instrument").loc[selected].reset_index()
    selected_signal = selected_signal.reset_index(drop=True)
    selected_instruments = selected_signal["instrument"].astype(str).str.upper().tolist()
    price_by_instrument = {
        str(row.instrument).upper(): safe_price(row.close)
        for row in selected_signal.itertuples(index=False)
    }
    min_lots = {
        instrument: min_buy_lot(instrument, 100)
        for instrument in selected_instruments
    }
    allocation = allocate_equal_weight_lots(
        selected_instruments,
        price_by_instrument,
        min_lots,
        initial_cash,
        effective_risk_budget,
        lot_size=100,
        denominator_count=top_k,
        mode=allocation_mode,
    )
    shares_by_instrument = allocation["shares"]
    allocation_weights = allocation["weights"]
    allocation_notionals = allocation["notional"]
    nominal_weight = effective_risk_budget / top_k
    rows = []
    for rank, row in enumerate(selected_signal.itertuples(index=False), start=1):
        instrument = str(row.instrument).upper()
        price = safe_price(getattr(row, "close"))
        shares = int(shares_by_instrument.get(instrument, 0))
        if shares <= 0:
            continue
        actual_weight = float(allocation_weights[instrument])
        rows.append(
            {
                "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
                "symbol": to_gm_symbol(instrument),
                "instrument": instrument,
                "rank": rank,
                "middle": float(getattr(row, "middle")),
                "target_weight": nominal_weight if allocation_mode == "fixed_topk" else actual_weight,
                "target_shares": int(shares),
                "nominal_target_weight": nominal_weight,
                "effective_weight_ref": actual_weight,
                "target_notional_ref": float(allocation_notionals[instrument]),
                "group": groups.get(instrument, "__UNKNOWN__"),
                "signal_date": signal_ts.strftime("%Y-%m-%d"),
                "price_ref_close": price,
                "avg_amount": float(getattr(row, "avg_amount")),
                "listing_days": int(getattr(row, "listing_days")),
                "effective_risk_budget": effective_risk_budget,
                "middle_rebalance_due": True,
                "middle_last_rebalance_signal_date": (
                    pd.Timestamp(
                        middle_last_rebalance_signal_date or signal_ts
                    ).strftime("%Y-%m-%d")
                ),
                "middle_trading_sessions_since_rebalance": int(
                    trading_sessions_since_rebalance
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Selected candidates all failed target-share rounding")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"paper_forward_top{top_k}_nost.csv"
    out.to_csv(target_path, index=False, encoding="utf-8-sig")
    no_st_hits = int(out["instrument"].astype(str).str.upper().isin(forbidden).sum())
    retained_previous = sorted(set(out["instrument"].astype(str).str.upper()) & previous_selection)
    group_snapshot = concentration_snapshot(out["instrument"].tolist(), groups)
    weighted_groups = out.groupby("group", dropna=False)["effective_weight_ref"].sum()
    effective_weight_sum = float(out["effective_weight_ref"].sum())
    requested_weight = float(effective_risk_budget)
    effective_exposure_ratio = effective_weight_sum / requested_weight
    max_effective_name_weight = float(out["effective_weight_ref"].max())
    allocation_gate_passed = bool(
        effective_exposure_ratio >= min_effective_exposure_ratio
        and max_effective_name_weight <= max_name_weight
    )
    manifest = {
        "status": "paper_forward_gm_targets_exported",
        "deployment_allowed": False,
        "paper_simulation_candidate": True,
        "prediction": str(prediction_path.resolve()),
        "target": str(target_path.resolve()),
        "signal_date": signal_ts.strftime("%Y-%m-%d"),
        "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
        "profile": (
            f"paper_forward_top{top_k}_rb{rebalance_every}_risk{effective_risk_budget:.2f}_"
            f"cap{industry_cap:.2f}_b{buffer_multiple}_nost_{allocation_mode}"
        ),
        "selection_mode": "scheduled_rebalance",
        "middle_rebalance": {
            "due": True,
            "last_rebalance_signal_date": pd.Timestamp(
                middle_last_rebalance_signal_date or signal_ts
            ).strftime("%Y-%m-%d"),
            "trading_sessions_since_rebalance": int(
                trading_sessions_since_rebalance
            ),
            "policy": "membership_and_target_shares_change_only_on_scheduled_rebalance",
            "forced_forbidden_exits": [],
        },
        "top_k": top_k,
        "rebalance_every": rebalance_every,
        "buffer_multiple": buffer_multiple,
        "previous_target": str(previous_target_path.resolve()) if previous_target_path else None,
        "previous_selection_count": len(previous_selection),
        "retained_previous_count": len(retained_previous),
        "retained_previous_instruments": retained_previous,
        "risk_budget": effective_risk_budget,
        "base_risk_budget": risk_budget,
        "industry_cap": industry_cap,
        "min_avg_amount": min_avg_amount,
        "initial_cash_for_share_estimate": initial_cash,
        "rows": int(len(out)),
        "symbols": int(out["symbol"].nunique()),
        "target_weight_sum": float(out["target_weight"].sum()),
        "effective_weight_sum": effective_weight_sum,
        "effective_exposure_ratio": effective_exposure_ratio,
        "outer_overlay": {
            "required": bool(require_outer),
            "prediction": str(outer_prediction_path.resolve()) if outer_prediction_path else None,
            "probability": outer_stats["probability"] if outer_stats else None,
            "prediction_rows": outer_stats["rows"] if outer_stats else 0,
            "prediction_minimum": outer_stats["minimum"] if outer_stats else None,
            "prediction_maximum": outer_stats["maximum"] if outer_stats else None,
            "threshold": outer_risk_threshold,
            "risk_floor": outer_risk_floor,
            "triggered": bool(
                outer_stats is not None
                and outer_stats["probability"] >= outer_risk_threshold
            ),
        },
        "allocation": allocation["diagnostics"],
        "execution_risk_controls": {
            "pause_buys_on_sell_reject": bool(pause_buys_on_sell_reject),
            "sell_first_required": bool(pause_buys_on_sell_reject),
        },
        "allocation_gate": {
            "passed": allocation_gate_passed,
            "min_effective_exposure_ratio": min_effective_exposure_ratio,
            "max_name_weight": max_name_weight,
            "observed_effective_exposure_ratio": effective_exposure_ratio,
            "observed_max_name_weight": max_effective_name_weight,
        },
        "forbidden_path": str(forbidden_path.resolve()) if forbidden_path else None,
        "removed_forbidden_prediction_rows": int(removed_forbidden_prediction_rows),
        "forbidden_order_hits": no_st_hits,
        "group_snapshot": group_snapshot,
        "weighted_group_snapshot": {
            "max_group": str(weighted_groups.idxmax()) if len(weighted_groups) else None,
            "max_group_weight": float(weighted_groups.max()) if len(weighted_groups) else 0.0,
            "groups": {str(key): float(value) for key, value in weighted_groups.items()},
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if require_allocation_gate and not allocation_gate_passed:
        raise RuntimeError(
            "allocation gate failed: "
            f"effective_ratio={effective_exposure_ratio:.2%} "
            f"max_name_weight={max_effective_name_weight:.2%}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", default=str(DEFAULT_PREDICTION))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--raw-daily-dir", default=str(DEFAULT_RAW_DAILY))
    parser.add_argument("--group-metadata", default=str(DEFAULT_GROUP_METADATA))
    parser.add_argument("--forbidden-symbols", default=str(DEFAULT_FORBIDDEN))
    parser.add_argument("--signal-date", default="2026-06-05")
    parser.add_argument("--trade-date", default="2026-06-09")
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--risk-budget", type=float, default=0.60)
    parser.add_argument("--industry-cap", type=float, default=0.30)
    parser.add_argument("--min-avg-amount", type=float, default=100_000_000.0)
    parser.add_argument("--initial-cash", type=float, default=500_000.0)
    parser.add_argument("--outer-prediction", default="")
    parser.add_argument("--require-outer", action="store_true")
    parser.add_argument("--outer-risk-threshold", type=float, default=0.50)
    parser.add_argument("--outer-risk-floor", type=float, default=0.30)
    parser.add_argument(
        "--allocation-mode",
        choices=sorted(ALLOCATION_MODES),
        default="fixed_topk",
    )
    parser.add_argument("--min-effective-exposure-ratio", type=float, default=0.0)
    parser.add_argument("--max-name-weight", type=float, default=0.03)
    parser.add_argument("--require-allocation-gate", action="store_true")
    parser.add_argument("--rebalance-every", type=int, default=20)
    parser.add_argument("--buffer-multiple", type=int, default=2)
    parser.add_argument("--pause-buys-on-sell-reject", action="store_true")
    parser.add_argument("--previous-target", default="")
    parser.add_argument("--require-previous-target", action="store_true")
    args = parser.parse_args()
    manifest = export_targets(
        Path(args.prediction).resolve(),
        Path(args.output_dir).resolve(),
        Path(args.raw_daily_dir).resolve(),
        Path(args.group_metadata).resolve(),
        Path(args.forbidden_symbols).resolve() if args.forbidden_symbols else None,
        args.signal_date,
        args.trade_date,
        args.top_k,
        args.risk_budget,
        args.industry_cap,
        args.min_avg_amount,
        args.initial_cash,
        Path(args.outer_prediction).resolve() if args.outer_prediction else None,
        args.require_outer,
        args.outer_risk_threshold,
        args.outer_risk_floor,
        args.allocation_mode,
        args.min_effective_exposure_ratio,
        args.max_name_weight,
        args.require_allocation_gate,
        args.rebalance_every,
        args.buffer_multiple,
        args.pause_buys_on_sell_reject,
        Path(args.previous_target).resolve() if args.previous_target else None,
        args.require_previous_target,
    )
    print(
        "[paper forward targets] "
        f"rows={manifest['rows']} symbols={manifest['symbols']} "
        f"weight={manifest['target_weight_sum']:.2%} "
        f"effective={manifest['effective_weight_sum']:.2%} "
        f"allocation_gate={manifest['allocation_gate']['passed']} "
        f"forbidden_hits={manifest['forbidden_order_hits']} "
        f"target={manifest['target']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
