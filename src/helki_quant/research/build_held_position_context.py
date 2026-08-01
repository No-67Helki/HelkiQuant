from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"


def read_calendar(path: Path) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.read_csv(path, header=None, names=["date"], parse_dates=["date"])["date"]
    ).drop_duplicates().sort_values()


def next_trade_date_map(calendar: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp]:
    return dict(zip(calendar[:-1], calendar[1:]))


def normalize_instrument(value: object) -> str:
    raw = str(value).upper()
    if raw.startswith("SZSE."):
        return "SZ" + raw.split(".", 1)[1]
    if raw.startswith("SHSE."):
        return "SH" + raw.split(".", 1)[1]
    return raw


def load_holdings(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    frame["holding_date"] = frame["trade_date"].dt.normalize()
    frame["datetime"] = frame["holding_date"]
    frame = frame.drop(columns=["trade_date"])
    frame["instrument"] = frame["instrument"].map(normalize_instrument)
    frame = frame[frame["shares"] > 0].copy()
    return frame.sort_values(["instrument", "datetime"]).reset_index(drop=True)


def add_holding_streak_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for instrument, part in frame.groupby("instrument", sort=False):
        part = part.sort_values("datetime").copy()
        date_diff = part["datetime"].diff().dt.days.fillna(9999)
        share_changed = part["shares"].ne(part["shares"].shift()).fillna(True)
        new_streak = (date_diff > 10) | share_changed
        streak_id = new_streak.cumsum()
        part["held_age_days"] = part.groupby(streak_id).cumcount() + 1
        part["entry_mark_price_approx"] = part.groupby(streak_id)["mark_price"].transform("first")
        part["prev_mark_price"] = part["mark_price"].shift(1)
        part["held_unrealized_ret_approx"] = (
            part["mark_price"] / (part["entry_mark_price_approx"] + 1e-12) - 1.0
        )
        part["held_prev_day_ret"] = part["mark_price"] / (part["prev_mark_price"] + 1e-12) - 1.0
        part["held_prev_day_ret"] = part["held_prev_day_ret"].replace([np.inf, -np.inf], np.nan)
        rows.append(part)
    if not rows:
        return frame
    return pd.concat(rows, ignore_index=True)


def load_targets(path: Path, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "trade_date" not in frame.columns:
        raise ValueError(f"{path} missing trade_date")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    if "signal_date" in frame.columns:
        frame["target_signal_date"] = pd.to_datetime(
            frame["signal_date"], errors="coerce"
        ).dt.normalize()
        frame["target_trade_date"] = frame["trade_date"]
    else:
        # Production-style targets are stamped with the signal/planning date;
        # the whole snapshot becomes effective on the next exchange session.
        frame["target_signal_date"] = frame["trade_date"]
        frame["target_trade_date"] = frame["trade_date"].map(next_trade_date_map(calendar))
    frame["instrument"] = frame["instrument"].map(normalize_instrument)
    if "target_shares" in frame.columns:
        frame["target_shares"] = pd.to_numeric(frame["target_shares"], errors="coerce").fillna(0)
        frame = frame[frame["target_shares"] > 0].copy()
    keep = [
        "target_signal_date",
        "target_trade_date",
        "instrument",
        "rank",
        "middle",
        "target_weight",
        "target_shares",
        "group",
    ]
    return (
        frame[[col for col in keep if col in frame.columns]]
        .dropna(subset=["target_trade_date", "instrument"])
        .drop_duplicates(["target_trade_date", "instrument"], keep="last")
        .sort_values(["target_trade_date", "instrument"])
        .reset_index(drop=True)
    )


def expand_active_target_snapshots(
    targets: pd.DataFrame,
    holding_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Carry the latest complete portfolio snapshot to each holding date."""
    if targets.empty or len(holding_dates) == 0:
        return targets.assign(datetime=pd.NaT).iloc[0:0].copy()
    snapshot_dates = pd.DatetimeIndex(targets["target_trade_date"].dropna().unique()).sort_values()
    rows: list[pd.DataFrame] = []
    for holding_date in pd.DatetimeIndex(holding_dates).normalize().unique().sort_values():
        pos = int(snapshot_dates.searchsorted(holding_date, side="right")) - 1
        if pos < 0:
            continue
        snapshot_date = snapshot_dates[pos]
        part = targets[targets["target_trade_date"] == snapshot_date].copy()
        part["datetime"] = holding_date
        rows.append(part)
    if not rows:
        return targets.assign(datetime=pd.NaT).iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True).sort_values(["datetime", "instrument"]).reset_index(drop=True)


def load_windows(path: Path, calendar_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    frame["trade_date"] = frame["trade_date"].dt.normalize()
    frame["instrument"] = frame["instrument"].map(normalize_instrument)
    next_map = next_trade_date_map(read_calendar(calendar_path))
    inverse = {value: key for key, value in next_map.items()}
    frame["datetime"] = frame["trade_date"].map(inverse)
    return frame.dropna(subset=["datetime"]).copy()


def build_context(
    holdings_path: Path,
    targets_path: Path,
    minute_windows_path: Path,
    calendar_path: Path,
    output_csv: Path,
    output_json: Path,
    *,
    buy_cost: float,
    sell_cost: float,
    slippage: float,
) -> dict:
    calendar = read_calendar(calendar_path)
    holdings = add_holding_streak_features(load_holdings(holdings_path))
    target_snapshots = load_targets(targets_path, calendar)
    targets = expand_active_target_snapshots(
        target_snapshots,
        pd.DatetimeIndex(holdings["datetime"].dropna().unique()),
    )
    windows = load_windows(minute_windows_path, calendar_path)

    frame = holdings.merge(targets, on=["datetime", "instrument"], how="left")
    frame = frame.merge(
        windows[["datetime", "trade_date", "instrument", "open_exec", "close_exec", "mark_close"]],
        on=["datetime", "instrument"],
        how="left",
    )
    if "trade_date" not in frame.columns:
        raise ValueError("execution trade_date missing after minute-window merge")
    invalid_execution_date = frame["trade_date"].notna() & (frame["trade_date"] <= frame["datetime"])
    if invalid_execution_date.any():
        sample = frame.loc[invalid_execution_date, ["datetime", "trade_date", "instrument"]].head(5)
        raise ValueError(
            "held context leaks same/prior-day execution data; sample="
            + sample.to_dict(orient="records").__repr__()
        )
    frame["target_weight"] = frame["target_weight"].fillna(0.0)
    frame["target_shares"] = frame["target_shares"].fillna(0.0)
    frame["target_missing"] = frame["rank"].isna().astype(int)
    frame["held_weight_gap_to_target"] = frame["target_weight"] - frame["weight"]
    frame["held_share_gap_to_target"] = frame["target_shares"] - frame["shares"]
    frame["held_abs_weight_gap_to_target"] = frame["held_weight_gap_to_target"].abs()
    frame["held_target_share_ratio"] = frame["target_shares"] / (frame["shares"] + 1e-12)
    frame["held_open_gap_vs_mark"] = frame["open_exec"] / (frame["mark_price"] + 1e-12) - 1.0
    frame["held_close_gap_vs_mark"] = frame["close_exec"] / (frame["mark_price"] + 1e-12) - 1.0
    frame["held_exec_intraday_ret"] = frame["close_exec"] / (frame["open_exec"] + 1e-12) - 1.0

    sell_price = frame["open_exec"] * (1.0 - slippage)
    buy_price = frame["close_exec"] * (1.0 + slippage)
    frame["held_t0_sell_open_buy_close_edge"] = (
        sell_price - buy_price
    ) / (frame["mark_price"] + 1e-12) - sell_cost - buy_cost
    frame["held_t0_sell_open_buy_close_hit"] = (
        frame["held_t0_sell_open_buy_close_edge"] > 0
    ).astype(float)
    frame["held_t0_sell_open_buy_close_strong_hit"] = (
        frame["held_t0_sell_open_buy_close_edge"] > 0.002
    ).astype(float)

    feature_cols = [
        "held_age_days",
        "held_unrealized_ret_approx",
        "held_prev_day_ret",
        "held_weight_gap_to_target",
        "held_abs_weight_gap_to_target",
        "held_share_gap_to_target",
        "held_target_share_ratio",
        "held_open_gap_vs_mark",
        "held_close_gap_vs_mark",
        "held_exec_intraday_ret",
        "target_missing",
        "rank",
        "middle",
    ]
    label_cols = [
        "held_t0_sell_open_buy_close_edge",
        "held_t0_sell_open_buy_close_hit",
        "held_t0_sell_open_buy_close_strong_hit",
    ]
    keep_cols = [
        "datetime",
        "holding_date",
        "trade_date",
        "instrument",
        "shares",
        "mark_price",
        "weight",
        "target_signal_date",
        "target_trade_date",
        "target_weight",
        "target_shares",
        "group",
        "open_exec",
        "close_exec",
        "mark_close",
    ] + feature_cols + label_cols
    frame = frame[[col for col in keep_cols if col in frame.columns]].replace([np.inf, -np.inf], np.nan)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
    usable = frame.dropna(subset=["open_exec", "close_exec", "held_t0_sell_open_buy_close_edge"])
    report = {
        "status": "held_position_context_built",
        "holdings_path": str(holdings_path.resolve()),
        "targets_path": str(targets_path.resolve()),
        "minute_windows_path": str(minute_windows_path.resolve()),
        "calendar_path": str(calendar_path.resolve()),
        "output_csv": str(output_csv.resolve()),
        "rows": int(len(frame)),
        "usable_exec_rows": int(len(usable)),
        "dates": int(frame["datetime"].nunique()),
        "instruments": int(frame["instrument"].nunique()),
        "usable_dates": int(usable["datetime"].nunique()),
        "usable_instruments": int(usable["instrument"].nunique()),
        "target_missing_ratio": float(frame["target_missing"].mean()) if len(frame) else None,
        "target_snapshot_dates": int(target_snapshots["target_trade_date"].nunique()),
        "expanded_target_dates": int(targets["datetime"].nunique()) if len(targets) else 0,
        "execution_after_holding_ratio": float(
            (frame.loc[frame["trade_date"].notna(), "trade_date"] > frame.loc[frame["trade_date"].notna(), "datetime"]).mean()
        )
        if frame["trade_date"].notna().any()
        else None,
        "edge_mean": float(usable["held_t0_sell_open_buy_close_edge"].mean()) if len(usable) else None,
        "edge_positive_ratio": float((usable["held_t0_sell_open_buy_close_edge"] > 0).mean()) if len(usable) else None,
        "strong_hit_ratio": float((usable["held_t0_sell_open_buy_close_edge"] > 0.002).mean()) if len(usable) else None,
        "feature_cols": feature_cols,
        "label_cols": label_cols,
        "deployment_allowed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdings", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--minute-windows", default=str(DATA / "_canonical_topk_minute_windows_2025_20260605.csv"))
    parser.add_argument("--calendar", default=str(DATA / "cn_data_pool_inner_top80_100_150_union_20260605_v3" / "calendars" / "day.txt"))
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--buy-cost", type=float, default=0.001)
    parser.add_argument("--sell-cost", type=float, default=0.0025)
    parser.add_argument("--slippage", type=float, default=0.0005)
    args = parser.parse_args()
    report = build_context(
        Path(args.holdings).resolve(),
        Path(args.targets).resolve(),
        Path(args.minute_windows).resolve(),
        Path(args.calendar).resolve(),
        Path(args.output_csv).resolve(),
        Path(args.output_json).resolve(),
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        slippage=args.slippage,
    )
    print(
        "[held context] "
        f"rows={report['rows']} usable={report['usable_exec_rows']} "
        f"dates={report['usable_dates']} instruments={report['usable_instruments']} "
        f"edge_mean={report['edge_mean']} pos={report['edge_positive_ratio']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
