from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_held_intraday_decision_dataset import (
    add_trigger_aligned_labels,
    normalize_inst,
    trigger_label_prefix,
)


KEYS = ["datetime", "trade_date", "instrument", "decision_time"]


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="raise").dt.normalize()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="raise").dt.normalize()
    out["instrument"] = out["instrument"].map(normalize_inst)
    out["decision_time"] = (
        out["decision_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    )
    return out


def _profit_factor(pnl: pd.Series) -> float | None:
    values = pd.to_numeric(pnl, errors="coerce").dropna()
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses <= 0:
        return None
    return gains / losses


def summarize_executed(frame: pd.DataFrame, pnl_col: str) -> dict:
    if frame.empty:
        return {
            "round_trips": 0,
            "cum_pnl": 0.0,
            "profit_factor": None,
            "positive_trade_ratio": 0.0,
            "active_dates": 0,
            "symbols": 0,
        }
    pnl = pd.to_numeric(frame[pnl_col], errors="coerce")
    return {
        "round_trips": int(len(frame)),
        "cum_pnl": float(pnl.sum()),
        "profit_factor": _profit_factor(pnl),
        "positive_trade_ratio": float((pnl > 0).mean()),
        "active_dates": int(frame["trade_date"].nunique()),
        "symbols": int(frame["instrument"].nunique()),
    }


def _apply_turnover_limit(
    selected: pd.DataFrame,
    *,
    prefix: str,
    buyback_col: str,
    nav_by_date: dict[pd.Timestamp, float],
    max_daily_turnover: float,
) -> tuple[pd.DataFrame, int]:
    accepted = []
    rejected = 0
    for trade_date, part in selected.groupby("trade_date", sort=True):
        nav = nav_by_date.get(pd.Timestamp(trade_date).normalize())
        if nav is None or not np.isfinite(nav) or nav <= 0:
            raise ValueError(f"daily account NAV missing for {pd.Timestamp(trade_date).date()}")
        used = 0.0
        for index, row in part.iterrows():
            volume = float(row["t0_exec_volume_one_lot_max50"])
            entry = float(row[f"{prefix}_entry_price"])
            exit_price = float(row[buyback_col]) * 1.0005
            turnover = volume * (entry + exit_price)
            if (used + turnover) / nav > max_daily_turnover:
                rejected += 1
                continue
            accepted.append(index)
            used += turnover
    return selected.loc[accepted].copy(), rejected


def _selected_profile(
    frame: pd.DataFrame,
    *,
    prefix: str,
    buyback_col: str,
    score_col: str,
    threshold: float,
    daily_top_n: int,
    nav_by_date: dict[pd.Timestamp, float],
    max_daily_turnover: float,
) -> dict:
    selected = (
        frame[pd.to_numeric(frame[score_col], errors="coerce") >= threshold]
        .sort_values(["trade_date", score_col, "instrument"], ascending=[True, False, True])
        .groupby("trade_date", sort=False)
        .head(daily_top_n)
    )
    touched = selected[pd.to_numeric(selected[f"{prefix}_touched"], errors="coerce") > 0.5]
    executed, rejected = _apply_turnover_limit(
        touched,
        prefix=prefix,
        buyback_col=buyback_col,
        nav_by_date=nav_by_date,
        max_daily_turnover=max_daily_turnover,
    )
    result = {
        "candidate_rows": int(len(selected)),
        "candidate_dates": int(selected["trade_date"].nunique()),
        "selected_touch_ratio": float(len(touched) / len(selected)) if len(selected) else 0.0,
        "turnover_rejections": rejected,
    }
    result.update(summarize_executed(executed, f"{prefix}_realized_pnl"))
    return result


def _oracle_positive_top_n(
    frame: pd.DataFrame,
    *,
    prefix: str,
    buyback_col: str,
    daily_top_n: int,
    nav_by_date: dict[pd.Timestamp, float],
    max_daily_turnover: float,
) -> dict:
    pnl_col = f"{prefix}_realized_pnl"
    eligible = frame[
        (pd.to_numeric(frame[f"{prefix}_touched"], errors="coerce") > 0.5)
        & (pd.to_numeric(frame[pnl_col], errors="coerce") > 0.0)
    ]
    selected = (
        eligible.sort_values(
            ["trade_date", pnl_col, "instrument"], ascending=[True, False, True]
        )
        .groupby("trade_date", sort=False)
        .head(daily_top_n)
    )
    executed, rejected = _apply_turnover_limit(
        selected,
        prefix=prefix,
        buyback_col=buyback_col,
        nav_by_date=nav_by_date,
        max_daily_turnover=max_daily_turnover,
    )
    result = {"turnover_rejections": rejected}
    result.update(summarize_executed(executed, pnl_col))
    return result


def parse_prediction_spec(raw: str) -> tuple[str, Path, float]:
    parts = raw.split("::")
    if len(parts) != 3:
        raise ValueError("prediction spec must be NAME::PATH::THRESHOLD")
    return parts[0], Path(parts[1]).resolve(), float(parts[2])


def audit(
    dataset_path: Path,
    daily_account_path: Path,
    output_json: Path,
    *,
    prediction_specs: list[tuple[str, Path, float]],
    decision_time: str,
    trigger_distances: list[float],
    daily_top_n: int,
    max_daily_turnover: float,
    sell_cost: float,
    buy_cost: float,
    slippage: float,
) -> dict:
    predictions = {}
    union_keys = []
    for name, path, threshold in prediction_specs:
        prediction = _normalize_keys(pd.read_csv(path))
        prediction = prediction[prediction["decision_time"] == str(decision_time).zfill(4)].copy()
        if prediction.duplicated(KEYS).any():
            raise ValueError(f"prediction {name} has duplicate keys")
        predictions[name] = {"frame": prediction, "path": path, "threshold": threshold}
        union_keys.append(prediction[KEYS])
    key_frame = pd.concat(union_keys, ignore_index=True).drop_duplicates()
    required = {
        *KEYS,
        "shares",
        "sell_price_decision",
        "label_trigger_window_low",
        "label_trigger_window_high",
        "buyback_1445_1450_price",
        "t0_exec_volume_one_lot_max50",
    }
    dataset = pd.read_csv(dataset_path, usecols=lambda col: col in required)
    dataset = _normalize_keys(dataset)
    dataset = dataset.merge(key_frame, on=KEYS, how="inner", validate="one_to_one")
    if len(dataset) != len(key_frame):
        raise ValueError(f"dataset coverage mismatch keys={len(key_frame)} rows={len(dataset)}")
    account = pd.read_csv(daily_account_path, parse_dates=["trade_date"])
    account["trade_date"] = account["trade_date"].dt.normalize()
    nav_by_date = dict(
        zip(
            account.drop_duplicates("trade_date", keep="last")["trade_date"],
            pd.to_numeric(account.drop_duplicates("trade_date", keep="last")["nav"], errors="coerce"),
        )
    )

    distance_results = {}
    buyback_col = "buyback_1445_1450_price"
    for distance in trigger_distances:
        enriched = add_trigger_aligned_labels(
            dataset,
            sell_cost=sell_cost,
            buy_cost=buy_cost,
            slippage=slippage,
            trigger_distances={"sell_first": float(distance)},
            touch_buffers=(0.0,),
            buyback_window="1445_1450",
        )
        prefix = trigger_label_prefix("sell_first", float(distance), 0.0, "1445_1450")
        valid = pd.to_numeric(enriched[f"{prefix}_touched"], errors="coerce").notna()
        touched = pd.to_numeric(enriched[f"{prefix}_touched"], errors="coerce") > 0.5
        profitable = pd.to_numeric(enriched[f"{prefix}_realized_pnl"], errors="coerce") > 0.0
        result = {
            "valid_rows": int(valid.sum()),
            "valid_dates": int(enriched.loc[valid, "trade_date"].nunique()),
            "touch_rows": int(touched.sum()),
            "touch_ratio": float(touched.loc[valid].mean()) if valid.any() else None,
            "profitable_touch_rows": int((touched & profitable).sum()),
            "profitable_realized_ratio": float(profitable.loc[valid].mean()) if valid.any() else None,
            "oracle_positive_top_n": _oracle_positive_top_n(
                enriched,
                prefix=prefix,
                buyback_col=buyback_col,
                daily_top_n=daily_top_n,
                nav_by_date=nav_by_date,
                max_daily_turnover=max_daily_turnover,
            ),
            "prediction_profiles": {},
        }
        for name, item in predictions.items():
            scored = enriched.merge(
                item["frame"][KEYS + ["fold", "score"]],
                on=KEYS,
                how="inner",
                validate="one_to_one",
            )
            result["prediction_profiles"][name] = {
                "prediction_path": str(item["path"]),
                "score_threshold": item["threshold"],
                **_selected_profile(
                    scored,
                    prefix=prefix,
                    buyback_col=buyback_col,
                    score_col="score",
                    threshold=item["threshold"],
                    daily_top_n=daily_top_n,
                    nav_by_date=nav_by_date,
                    max_daily_turnover=max_daily_turnover,
                ),
            }
        distance_results[f"{distance:.6f}"] = result
        print(
            f"[trigger ceiling] distance={distance:.4%} touch={result['touch_ratio']:.2%} "
            f"oracle_trips={result['oracle_positive_top_n']['round_trips']} "
            f"oracle_pnl={result['oracle_positive_top_n']['cum_pnl']:.2f}",
            flush=True,
        )
    report = {
        "status": "trigger_opportunity_ceiling_audited",
        "dataset": str(dataset_path.resolve()),
        "daily_account": str(daily_account_path.resolve()),
        "decision_time": str(decision_time).zfill(4),
        "date_range": [
            str(dataset["trade_date"].min().date()),
            str(dataset["trade_date"].max().date()),
        ],
        "rows": int(len(dataset)),
        "dates": int(dataset["trade_date"].nunique()),
        "daily_top_n": daily_top_n,
        "max_daily_turnover": max_daily_turnover,
        "distances": distance_results,
        "warning": "Oracle metrics use future outcomes and are diagnostic upper bounds only.",
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--daily-account", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="NAME::PATH::THRESHOLD",
    )
    parser.add_argument("--decision-time", default="1000")
    parser.add_argument("--trigger-distances", default="0.005,0.00625,0.0075,0.01")
    parser.add_argument("--daily-top-n", type=int, default=2)
    parser.add_argument("--max-daily-turnover", type=float, default=0.03)
    parser.add_argument("--sell-cost", type=float, default=0.0025)
    parser.add_argument("--buy-cost", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    args = parser.parse_args()
    audit(
        Path(args.dataset).resolve(),
        Path(args.daily_account).resolve(),
        Path(args.output_json).resolve(),
        prediction_specs=[parse_prediction_spec(raw) for raw in args.prediction],
        decision_time=args.decision_time,
        trigger_distances=[float(item) for item in args.trigger_distances.split(",")],
        daily_top_n=args.daily_top_n,
        max_daily_turnover=args.max_daily_turnover,
        sell_cost=args.sell_cost,
        buy_cost=args.buy_cost,
        slippage=args.slippage,
    )


if __name__ == "__main__":
    main()
