from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_experiments import CostScenario, STRESS_COST
from universe import load_price_panel


@dataclass(frozen=True)
class TransitionLimits:
    max_two_way_turnover: float
    max_estimated_cost_ratio: float
    min_cash_ratio: float = 0.20


BUFFERED_LIMITS = TransitionLimits(
    max_two_way_turnover=0.25,
    max_estimated_cost_ratio=0.0015,
)
INITIAL_LAUNCH_LIMITS = TransitionLimits(
    max_two_way_turnover=0.65,
    max_estimated_cost_ratio=0.0030,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def gm_to_local_symbol(value: object) -> str:
    text = str(value).strip().upper()
    if text.startswith("SHSE."):
        return "SH" + text.split(".", 1)[1]
    if text.startswith("SZSE."):
        return "SZ" + text.split(".", 1)[1]
    if text.startswith(("SH", "SZ")) and len(text) >= 8:
        return text[:2] + text[-6:]
    raise ValueError(f"cannot normalize target symbol: {value!r}")


def _single_date(frame: pd.DataFrame, column: str, label: str) -> pd.Timestamp:
    if column not in frame.columns:
        raise ValueError(f"{label} target missing {column}")
    values = pd.to_datetime(frame[column], errors="coerce").dropna().dt.normalize().unique()
    if len(values) != 1:
        raise ValueError(f"{label} target must contain one {column}: {list(values)}")
    return pd.Timestamp(values[0]).normalize()


def load_target(path: Path, label: str) -> tuple[pd.DataFrame, dict[str, str]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} target not found: {path}")
    frame = pd.read_csv(path, dtype={"symbol": str, "instrument": str})
    if "target_shares" not in frame.columns:
        raise ValueError(f"{label} target missing target_shares")
    if "instrument" in frame.columns:
        frame["instrument"] = frame["instrument"].map(gm_to_local_symbol)
    elif "symbol" in frame.columns:
        frame["instrument"] = frame["symbol"].map(gm_to_local_symbol)
    else:
        raise ValueError(f"{label} target must contain instrument or symbol")
    numeric_shares = pd.to_numeric(frame["target_shares"], errors="coerce")
    invalid_shares = numeric_shares.isna() | numeric_shares.le(0) | numeric_shares.mod(1).ne(0)
    if invalid_shares.any():
        raise ValueError(f"{label} target has invalid target_shares rows: {int(invalid_shares.sum())}")
    frame["target_shares"] = numeric_shares.astype(int)
    duplicates = frame["instrument"].duplicated(keep=False)
    if duplicates.any():
        values = sorted(frame.loc[duplicates, "instrument"].unique().tolist())
        raise ValueError(f"{label} target has duplicate instruments: {values}")
    trade_date = _single_date(frame, "trade_date", label)
    signal_date = _single_date(frame, "signal_date", label)
    return (
        frame.set_index("instrument").sort_index(),
        {
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "signal_date": signal_date.strftime("%Y-%m-%d"),
        },
    )


def load_prices(
    raw_daily_dir: Path,
    instruments: set[str],
    signal_date: str,
) -> tuple[dict[str, float], list[str]]:
    if not instruments:
        return {}, []
    panel = load_price_panel(
        raw_daily_dir.resolve(),
        sorted(instruments),
        start=signal_date,
        end=signal_date,
    )
    panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
    panel = panel[
        panel["datetime"].dt.normalize().eq(pd.Timestamp(signal_date).normalize())
        & np.isfinite(panel["close"])
        & panel["close"].gt(0)
    ].copy()
    prices = panel.drop_duplicates("instrument", keep="last").set_index("instrument")["close"]
    result = {str(key).upper(): float(value) for key, value in prices.items()}
    missing = sorted(instruments - set(result))
    return result, missing


def load_account_snapshot(
    path: Path,
    *,
    expected_account_id: str,
    as_of_date: str,
    max_age_days: int = 1,
    allowed_capture_dates: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"account snapshot not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("status") != "gm_paper_account_snapshot":
        raise ValueError("account snapshot has invalid status")
    if payload.get("passed") is not True or payload.get("failed_checks"):
        raise ValueError("account snapshot did not pass its capture checks")
    if payload.get("paper_only") is not True or payload.get("no_order") is not True:
        raise ValueError("account snapshot must be PAPER-only and no-order")
    if int(payload.get("orders_submitted", -1)) != 0:
        raise ValueError("account snapshot unexpectedly reports submitted orders")
    account_id = str(payload.get("account_id") or "").strip()
    if not expected_account_id or account_id != str(expected_account_id).strip():
        raise ValueError(
            f"account snapshot id mismatch: observed={account_id!r} "
            f"expected={expected_account_id!r}"
        )
    captured = pd.Timestamp(payload.get("captured_at")).normalize()
    as_of = pd.Timestamp(as_of_date).normalize()
    age_days = int((as_of - captured).days)
    normalized_allowed_dates = tuple(
        sorted(
            {
                pd.Timestamp(value).normalize().strftime("%Y-%m-%d")
                for value in (allowed_capture_dates or ())
            }
        )
    )
    captured_date = captured.strftime("%Y-%m-%d")
    if normalized_allowed_dates:
        if captured_date not in normalized_allowed_dates:
            raise ValueError(
                "account snapshot is not from an allowed release boundary: "
                f"captured={captured_date} allowed={list(normalized_allowed_dates)}"
            )
        freshness_policy = "signal_or_trade_date"
    else:
        if age_days < 0 or age_days > max_age_days:
            raise ValueError(
                "account snapshot is not fresh: "
                f"captured={captured.date()} as_of={as_of.date()} age_days={age_days} "
                f"allowed=0..{max_age_days}"
            )
        freshness_policy = "calendar_age"
    positions_meta = payload.get("positions", {})
    positions_path = Path(str(positions_meta.get("file") or ""))
    positions_path = (
        positions_path if positions_path.is_absolute() else path.parent / positions_path
    ).resolve()
    if not positions_path.is_file():
        raise FileNotFoundError(f"account positions file not found: {positions_path}")
    expected_positions_hash = str(positions_meta.get("sha256") or "").upper()
    if not expected_positions_hash or sha256_file(positions_path) != expected_positions_hash:
        raise ValueError("account positions hash mismatch")
    frame = pd.read_csv(positions_path, dtype={"symbol": str, "instrument": str})
    if not {"symbol", "volume"}.issubset(frame.columns):
        raise ValueError("account positions must contain symbol and volume")
    frame["instrument"] = frame["symbol"].map(gm_to_local_symbol)
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    invalid_volume = frame["volume"].isna() | frame["volume"].lt(0) | frame["volume"].mod(1).ne(0)
    if invalid_volume.any():
        raise ValueError("account positions contain invalid volumes")
    if "side" in frame.columns:
        side = pd.to_numeric(frame["side"], errors="coerce")
        unsupported = frame[frame["volume"].gt(0) & side.ne(1)]
        if not unsupported.empty:
            raise ValueError("account snapshot contains non-long positions")
    frame = frame[frame["volume"].gt(0)].copy()
    if int(payload.get("position_rows", -1)) != len(frame):
        raise ValueError("account snapshot position row count mismatch")
    shares = (
        frame.groupby("instrument", sort=True)["volume"].sum().astype(int).to_dict()
        if not frame.empty
        else {}
    )
    stock_universe_evidence = None
    stock_universe_meta = payload.get("stock_universe")
    if stock_universe_meta:
        stock_universe_path = Path(
            str(stock_universe_meta.get("file") or "")
        )
        stock_universe_path = (
            stock_universe_path
            if stock_universe_path.is_absolute()
            else path.parent / stock_universe_path
        ).resolve()
        if not stock_universe_path.is_file():
            raise FileNotFoundError(
                f"snapshot stock universe not found: {stock_universe_path}"
            )
        expected_stock_hash = str(
            stock_universe_meta.get("sha256") or ""
        ).upper()
        if (
            not expected_stock_hash
            or sha256_file(stock_universe_path) != expected_stock_hash
        ):
            raise ValueError("snapshot stock universe hash mismatch")
        stock_frame = pd.read_csv(stock_universe_path, dtype=str).fillna("")
        required_stock_columns = {
            "TS代码",
            "股票代码",
            "股票名称",
            "上市状态",
            "退市日期",
        }
        missing_stock_columns = required_stock_columns - set(stock_frame.columns)
        if missing_stock_columns:
            raise ValueError(
                "snapshot stock universe missing columns: "
                f"{sorted(missing_stock_columns)}"
            )
        if int(stock_universe_meta.get("rows", -1)) != len(stock_frame):
            raise ValueError("snapshot stock universe row count mismatch")
        stock_universe_evidence = {
            "path": str(stock_universe_path),
            "sha256": expected_stock_hash,
            "rows": int(len(stock_frame)),
            "source": str(stock_universe_meta.get("source") or ""),
            "st_name_rows": int(stock_universe_meta.get("st_name_rows", 0)),
            "delisted_rows": int(stock_universe_meta.get("delisted_rows", 0)),
        }
    cash = payload.get("cash", {})
    nav = float(cash.get("nav", 0.0))
    available = float(cash.get("available", -1.0))
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("account snapshot NAV must be positive")
    if not np.isfinite(available) or available < 0:
        raise ValueError("account snapshot available cash must be nonnegative")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "positions_path": str(positions_path),
        "positions_sha256": expected_positions_hash,
        "account_id": account_id,
        "captured_at": str(payload.get("captured_at")),
        "age_days": age_days,
        "freshness_policy": freshness_policy,
        "allowed_capture_dates": list(normalized_allowed_dates),
        "nav": nav,
        "available_cash": available,
        "position_rows": int(len(frame)),
        "position_symbols": int(len(shares)),
        "position_shares": int(sum(shares.values())),
        "stock_universe": stock_universe_evidence,
        "shares": shares,
    }


def _check(name: str, observed: Any, limit: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "limit": limit,
        "passed": bool(passed),
    }


def audit_target_transition(
    *,
    next_target: Path,
    raw_daily_dir: Path,
    output_path: Path,
    previous_target: Path | None,
    initial_launch: bool,
    initial_nav: float = 1_000_000.0,
    cost: CostScenario = STRESS_COST,
    limits: TransitionLimits | None = None,
    account_snapshot: Path | None = None,
    expected_account_id: str = "",
    as_of_date: str | None = None,
    max_account_snapshot_age_days: int = 1,
    account_snapshot_allowed_dates: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if initial_nav <= 0:
        raise ValueError("initial_nav must be positive")
    if initial_launch == (previous_target is not None):
        raise ValueError(
            "initial_launch requires no previous target; buffered mode requires one"
        )
    next_frame, next_dates = load_target(next_target, "next")
    if previous_target is None:
        previous_frame = pd.DataFrame(columns=["target_shares"])
        previous_frame.index.name = "instrument"
        previous_dates = {"trade_date": None, "signal_date": None}
        mode = "initial_launch"
    else:
        previous_frame, previous_dates = load_target(previous_target, "previous")
        if pd.Timestamp(previous_dates["signal_date"]) > pd.Timestamp(next_dates["signal_date"]):
            raise ValueError("previous target signal_date is after the next target")
        if pd.Timestamp(previous_dates["trade_date"]) > pd.Timestamp(next_dates["trade_date"]):
            raise ValueError("previous target trade_date is after the next target")
        mode = "buffered_previous_target"

    previous_shares = {
        str(index): int(value)
        for index, value in previous_frame.get("target_shares", pd.Series(dtype=int)).items()
    }
    next_shares = {
        str(index): int(value)
        for index, value in next_frame["target_shares"].items()
    }
    account_snapshot_evidence = None
    starting_shares = dict(previous_shares)
    effective_nav = float(initial_nav)
    snapshot_cash = None
    position_source = "previous_target"
    if account_snapshot is not None:
        if as_of_date is None:
            raise ValueError("as_of_date is required with an account snapshot")
        account_snapshot_evidence = load_account_snapshot(
            account_snapshot,
            expected_account_id=expected_account_id,
            as_of_date=as_of_date,
            max_age_days=max_account_snapshot_age_days,
            allowed_capture_dates=account_snapshot_allowed_dates,
        )
        starting_shares = dict(account_snapshot_evidence.pop("shares"))
        effective_nav = float(account_snapshot_evidence["nav"])
        snapshot_cash = float(account_snapshot_evidence["available_cash"])
        position_source = "account_snapshot"
        if initial_launch and starting_shares:
            raise ValueError("initial_launch account snapshot must contain no positions")
    instruments = set(starting_shares) | set(next_shares)
    prices, missing_prices = load_prices(
        raw_daily_dir,
        instruments,
        next_dates["signal_date"],
    )

    order_rows: list[dict[str, Any]] = []
    for instrument in sorted(instruments):
        before = int(starting_shares.get(instrument, 0))
        after = int(next_shares.get(instrument, 0))
        delta = after - before
        price = float(prices.get(instrument, 0.0))
        if delta == 0:
            side = "NONE"
        elif delta < 0:
            side = "SELL"
        else:
            side = "BUY"
        reference_notional = abs(delta) * price
        if side == "SELL":
            fill_price = price * (1.0 - cost.slippage)
            fill_notional = abs(delta) * fill_price
            fee = max(fill_notional * cost.sell_cost, cost.min_cost)
            cash_effect = fill_notional - fee
            estimated_cost = reference_notional - cash_effect
        elif side == "BUY":
            fill_price = price * (1.0 + cost.slippage)
            fill_notional = abs(delta) * fill_price
            fee = max(fill_notional * cost.buy_cost, cost.min_cost)
            cash_effect = -(fill_notional + fee)
            estimated_cost = -cash_effect - reference_notional
        else:
            fill_price = price
            fill_notional = 0.0
            fee = 0.0
            cash_effect = 0.0
            estimated_cost = 0.0
        order_rows.append(
            {
                "instrument": instrument,
                "previous_shares": before,
                "next_shares": after,
                "delta_shares": delta,
                "side": side,
                "reference_price": price,
                "reference_notional": reference_notional,
                "estimated_fill_price": fill_price,
                "estimated_fill_notional": fill_notional,
                "estimated_fee": fee,
                "estimated_cost": estimated_cost,
                "cash_effect": cash_effect,
            }
        )

    orders = pd.DataFrame(order_rows)
    actionable = orders[orders["side"].isin(["SELL", "BUY"])].copy()
    sells = actionable[actionable["side"].eq("SELL")]
    buys = actionable[actionable["side"].eq("BUY")]
    previous_market_value = float(
        sum(shares * prices.get(instrument, 0.0) for instrument, shares in starting_shares.items())
    )
    next_market_value = float(
        sum(shares * prices.get(instrument, 0.0) for instrument, shares in next_shares.items())
    )
    initial_cash = (
        float(snapshot_cash)
        if snapshot_cash is not None
        else float(effective_nav - previous_market_value)
    )
    cash = initial_cash
    cash_path = [cash]
    for frame in (sells, buys):
        for row in frame.sort_values("instrument").itertuples(index=False):
            cash += float(row.cash_effect)
            cash_path.append(cash)
    final_cash = float(cash)
    min_cash = float(min(cash_path))
    sell_notional = float(sells["reference_notional"].sum())
    buy_notional = float(buys["reference_notional"].sum())
    estimated_cost = float(actionable["estimated_cost"].sum())
    two_way_turnover = (sell_notional + buy_notional) / effective_nav
    max_leg_turnover = max(sell_notional, buy_notional) / effective_nav
    estimated_cost_ratio = estimated_cost / effective_nav
    min_cash_ratio = min_cash / effective_nav
    lot_violations = int(
        sum(shares <= 0 or shares % 100 != 0 for shares in next_shares.values())
    )
    retained = set(starting_shares) & set(next_shares)
    retained_changed = sum(
        starting_shares[instrument] != next_shares[instrument]
        for instrument in retained
    )
    extra_account_positions = sorted(set(starting_shares) - set(previous_shares))
    missing_account_positions = sorted(set(previous_shares) - set(starting_shares))
    account_share_mismatches = sorted(
        instrument
        for instrument in set(previous_shares) & set(starting_shares)
        if previous_shares[instrument] != starting_shares[instrument]
    )

    active_limits = limits or (INITIAL_LAUNCH_LIMITS if initial_launch else BUFFERED_LIMITS)
    checks = [
        _check("current_prices_complete", len(missing_prices), 0, not missing_prices),
        _check("target_board_lots", lot_violations, 0, lot_violations == 0),
        _check(
            "two_way_turnover",
            two_way_turnover,
            active_limits.max_two_way_turnover,
            two_way_turnover <= active_limits.max_two_way_turnover + 1e-12,
        ),
        _check(
            "estimated_cost_ratio",
            estimated_cost_ratio,
            active_limits.max_estimated_cost_ratio,
            estimated_cost_ratio <= active_limits.max_estimated_cost_ratio + 1e-12,
        ),
        _check(
            "sell_first_min_cash_ratio",
            min_cash_ratio,
            active_limits.min_cash_ratio,
            min_cash_ratio >= active_limits.min_cash_ratio - 1e-12,
        ),
        _check("negative_cash", min_cash, 0.0, min_cash >= -1e-6),
    ]
    failed_checks = [item["name"] for item in checks if not item["passed"]]

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    orders_path = output_path.with_suffix(".orders.csv")
    orders.to_csv(orders_path, index=False, encoding="utf-8-sig")
    report = {
        "status": "target_transition_audit",
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "mode": mode,
        "signal_date": next_dates["signal_date"],
        "trade_date": next_dates["trade_date"],
        "previous_target_dates": previous_dates,
        "initial_nav": effective_nav,
        "cost": asdict(cost),
        "limits": asdict(active_limits),
        "previous_target": (
            {
                "path": str(previous_target.resolve()),
                "sha256": sha256_file(previous_target.resolve()),
            }
            if previous_target is not None
            else None
        ),
        "position_source": position_source,
        "account_snapshot": account_snapshot_evidence,
        "next_target": {
            "path": str(next_target.resolve()),
            "sha256": sha256_file(next_target.resolve()),
        },
        "raw_daily_dir": str(raw_daily_dir.resolve()),
        "orders_csv": str(orders_path),
        "missing_prices": missing_prices,
        "counts": {
            "previous_names": len(starting_shares),
            "previous_target_names": len(previous_shares),
            "starting_position_names": len(starting_shares),
            "next_names": len(next_shares),
            "retained_names": len(retained),
            "retained_changed_shares": int(retained_changed),
            "full_exits": int(len(set(starting_shares) - set(next_shares))),
            "new_entries": int(len(set(next_shares) - set(starting_shares))),
            "sell_orders": int(len(sells)),
            "buy_orders": int(len(buys)),
            "actionable_orders": int(len(actionable)),
        },
        "previous_target_account_drift": {
            "extra_account_positions": extra_account_positions,
            "missing_account_positions": missing_account_positions,
            "share_mismatch_positions": account_share_mismatches,
            "drifted_symbols": len(
                set(extra_account_positions)
                | set(missing_account_positions)
                | set(account_share_mismatches)
            ),
        },
        "metrics": {
            "previous_market_value": previous_market_value,
            "next_market_value": next_market_value,
            "previous_exposure_ratio": previous_market_value / effective_nav,
            "next_exposure_ratio": next_market_value / effective_nav,
            "initial_cash": initial_cash,
            "sell_notional": sell_notional,
            "buy_notional": buy_notional,
            "two_way_turnover": two_way_turnover,
            "max_leg_turnover": max_leg_turnover,
            "estimated_cost": estimated_cost,
            "estimated_cost_ratio": estimated_cost_ratio,
            "cash_after_sell_first": final_cash,
            "cash_after_ratio": final_cash / effective_nav,
            "min_cash": min_cash,
            "min_cash_ratio": min_cash_ratio,
            "lot_violations": lot_violations,
        },
        "checks": checks,
        "paper_orders_allowed": False,
        "real_money_deployment_allowed": False,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit one sell-first transition between versioned target-share files."
    )
    parser.add_argument("--next-target", type=Path, required=True)
    parser.add_argument("--previous-target", type=Path)
    parser.add_argument("--initial-launch", action="store_true")
    parser.add_argument("--raw-daily-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-nav", type=float, default=1_000_000.0)
    parser.add_argument("--account-snapshot", type=Path)
    parser.add_argument("--expected-account-id", default="")
    parser.add_argument("--as-of-date")
    parser.add_argument("--account-snapshot-allowed-date", action="append", default=[])
    args = parser.parse_args()
    report = audit_target_transition(
        next_target=args.next_target,
        previous_target=args.previous_target,
        initial_launch=args.initial_launch,
        raw_daily_dir=args.raw_daily_dir,
        output_path=args.output,
        initial_nav=args.initial_nav,
        account_snapshot=args.account_snapshot,
        expected_account_id=args.expected_account_id,
        as_of_date=args.as_of_date,
        account_snapshot_allowed_dates=tuple(args.account_snapshot_allowed_date),
    )
    metrics = report["metrics"]
    print(
        "[target transition] "
        f"passed={report['passed']} mode={report['mode']} "
        f"orders={report['counts']['actionable_orders']} "
        f"turnover={metrics['two_way_turnover']:.2%} "
        f"cost={metrics['estimated_cost_ratio']:.4%} "
        f"min_cash={metrics['min_cash_ratio']:.2%}",
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
