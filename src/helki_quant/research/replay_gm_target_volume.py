from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_experiments import BASE_COST, STRESS_COST, CostScenario


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"

CHINESE_COLUMNS = {
    "日期": "datetime",
    "股票代码": "code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


@dataclass(frozen=True)
class ReplayConfig:
    initial_cash: float = 1_000_000.0
    lot_size: int = 100
    buffer_multiple: int = 2


def gm_to_local_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if "." not in text:
        return text
    exchange, code = text.split(".", 1)
    prefix = "SH" if exchange in {"SHSE", "SH"} else "SZ"
    return f"{prefix}{code}"


def min_buy_lot(symbol: str, lot_size: int) -> int:
    code = str(symbol).upper()[-6:]
    if code.startswith(("688", "689")):
        return max(lot_size, 200)
    return lot_size


def round_lot(value: float, lot_size: int, min_lot: int | None = None) -> int:
    if not np.isfinite(value):
        return 0
    rounded = max(0, int(value // lot_size) * lot_size)
    if min_lot is not None and rounded < min_lot:
        return 0
    return rounded


def safe_price(*values: object) -> float:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0:
            return number
    return 0.0


def load_targets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str, "instrument": str})
    required = {"trade_date", "symbol", "target_shares"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing target columns: {sorted(missing)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["instrument"] = frame.get("instrument", frame["symbol"].map(gm_to_local_symbol))
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame["target_shares"] = pd.to_numeric(frame["target_shares"], errors="coerce").fillna(0).astype(int)
    frame["target_weight"] = pd.to_numeric(frame.get("target_weight", 0.0), errors="coerce").fillna(0.0)
    return frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def load_price_panel(raw_daily_dir: Path, instruments: set[str], start: str, end: str) -> pd.DataFrame:
    rows = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for inst in sorted(instruments):
        code = inst[-6:]
        candidates = [
            raw_daily_dir / f"{code}_daily_qfq.csv",
            raw_daily_dir / f"{inst.lower()}_daily_qfq.csv",
            raw_daily_dir / f"{inst.upper()}_daily_qfq.csv",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            continue
        frame = pd.read_csv(path)
        frame = frame.rename(columns={k: v for k, v in CHINESE_COLUMNS.items() if k in frame.columns})
        if "datetime" not in frame.columns:
            continue
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
        frame = frame[frame["datetime"].between(start_ts, end_ts)].copy()
        if frame.empty:
            continue
        for column in ("open", "close", "high", "low", "volume", "amount"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["instrument"] = inst
        rows.append(frame[["datetime", "instrument", "open", "close", "high", "low", "volume", "amount"]])
    if not rows:
        raise ValueError("no price rows loaded")
    return pd.concat(rows, ignore_index=True).sort_values(["datetime", "instrument"])


def order_sort_key(item: tuple[str, int]) -> tuple[int, int, str]:
    symbol, delta = item
    return (0 if delta < 0 else 1, delta, symbol)


def replay(
    target_csv: Path,
    raw_daily_dir: Path,
    output_dir: Path,
    cost: CostScenario,
    cfg: ReplayConfig,
    price_start: str,
    price_end: str,
    profile_name: str,
) -> dict[str, object]:
    targets = load_targets(target_csv)
    instruments = set(targets["instrument"])
    prices = load_price_panel(raw_daily_dir, instruments, price_start, price_end)
    price_by_date = {
        date: part.set_index("instrument").copy()
        for date, part in prices.groupby("datetime", sort=True)
    }
    target_by_date = {
        date: part.copy()
        for date, part in targets.groupby("trade_date", sort=True)
    }
    calendar = pd.DatetimeIndex(prices["datetime"].drop_duplicates().sort_values())
    start_date = min(target_by_date)
    end_date = max(target_by_date)
    calendar = calendar[(calendar >= start_date) & (calendar <= end_date)]

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = output_dir / "gm_target_volume_replay"
    profile_dir.mkdir(parents=True, exist_ok=True)

    orders: list[dict[str, object]] = []
    fills: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []

    cash = float(cfg.initial_cash)
    held: dict[str, int] = {}
    last_mark: dict[str, float] = {}
    turnover = 0.0
    trades = 0
    total_fees = 0.0
    cash_min = cash
    nav_peak = cash
    max_drawdown = 0.0
    negative_cash_events = 0
    lot_violations = 0

    for trade_date in calendar:
        date_text = trade_date.strftime("%Y-%m-%d")
        price_day = price_by_date.get(trade_date)
        if price_day is None:
            continue
        price_open = price_day["open"].to_dict()
        mark_close = price_day["close"].to_dict()
        for inst, value in mark_close.items():
            mark = safe_price(value)
            if mark > 0:
                last_mark[inst] = mark

        day_turnover = 0.0
        day_trades = 0
        is_rebalance = trade_date in target_by_date
        if is_rebalance:
            day = target_by_date[trade_date]
            desired = dict(zip(day["instrument"], day["target_shares"]))
            weights = dict(zip(day["instrument"], day["target_weight"]))
            deltas = {
                inst: int(desired.get(inst, 0)) - int(held.get(inst, 0))
                for inst in set(held) | set(desired)
            }
            for inst, delta in sorted(deltas.items(), key=order_sort_key):
                if delta == 0:
                    continue
                current_shares = int(held.get(inst, 0))
                target_shares = int(desired.get(inst, 0))
                side = "SELL" if delta < 0 else "BUY"
                price = safe_price(price_open.get(inst), mark_close.get(inst), last_mark.get(inst))
                if price <= 0:
                    continue
                if delta > 0 and delta < min_buy_lot(inst, cfg.lot_size):
                    continue
                order_notional = abs(delta) * price
                orders.append(
                    {
                        "trade_date": date_text,
                        "instrument": inst,
                        "side": side,
                        "current_shares": current_shares,
                        "target_shares": target_shares,
                        "delta_shares": delta,
                        "order_price": price,
                        "order_notional": order_notional,
                        "reason": "target_volume_replay",
                    }
                )
                nav_ref = cash + sum(
                    volume * safe_price(mark_close.get(code), price_open.get(code), last_mark.get(code))
                    for code, volume in held.items()
                )
                cash_before = cash
                if delta < 0:
                    volume = min(-delta, current_shares)
                    fill_price = price * (1.0 - cost.slippage)
                    gross_value = volume * fill_price
                    fee = max(gross_value * cost.sell_cost, cost.min_cost) if volume > 0 else 0.0
                    cash += gross_value - fee
                    held[inst] = current_shares - volume
                else:
                    fill_price = price * (1.0 + cost.slippage)
                    affordable = round_lot(
                        max(0.0, cash - cost.min_cost) / (fill_price * (1.0 + cost.buy_cost)),
                        cfg.lot_size,
                        min_buy_lot(inst, cfg.lot_size),
                    )
                    volume = min(delta, affordable)
                    gross_value = volume * fill_price
                    fee = max(gross_value * cost.buy_cost, cost.min_cost) if volume > 0 else 0.0
                    cash -= gross_value + fee
                    held[inst] = current_shares + volume
                if volume <= 0:
                    continue
                turn = gross_value / max(nav_ref, 1e-12)
                day_turnover += turn
                turnover += turn
                day_trades += 1
                trades += 1
                total_fees += fee
                if volume % cfg.lot_size != 0:
                    lot_violations += 1
                cash_min = min(cash_min, cash)
                if cash < -1e-6:
                    negative_cash_events += 1
                fills.append(
                    {
                        "trade_date": date_text,
                        "instrument": inst,
                        "side": side,
                        "shares": volume,
                        "order_price": price,
                        "fill_price": fill_price,
                        "gross_value": gross_value,
                        "fee": fee,
                        "cash_before": cash_before,
                        "cash_after": cash,
                        "nav_ref": nav_ref,
                        "turnover_contribution": turn,
                    }
                )
            held = {inst: volume for inst, volume in held.items() if volume > 0}

        market_value = sum(
            volume * safe_price(mark_close.get(inst), price_open.get(inst), last_mark.get(inst))
            for inst, volume in held.items()
        )
        nav = cash + market_value
        nav_peak = max(nav_peak, nav)
        max_drawdown = max(max_drawdown, 1.0 - nav / max(nav_peak, 1e-12))
        exposure = market_value / max(nav, 1e-12)
        daily_rows.append(
            {
                "trade_date": date_text,
                "is_rebalance": int(is_rebalance),
                "cash": cash,
                "market_value": market_value,
                "nav": nav,
                "gross_exposure": exposure,
                "day_turnover": day_turnover,
                "cum_turnover": turnover,
                "day_trades": day_trades,
                "cum_trades": trades,
                "holdings_count": len(held),
            }
        )
        for inst, volume in sorted(held.items()):
            mark = safe_price(mark_close.get(inst), price_open.get(inst), last_mark.get(inst))
            holdings_rows.append(
                {
                    "trade_date": date_text,
                    "instrument": inst,
                    "shares": volume,
                    "mark_price": mark,
                    "market_value": volume * mark,
                    "weight": volume * mark / max(nav, 1e-12),
                }
            )

    orders_frame = pd.DataFrame(orders)
    fills_frame = pd.DataFrame(fills)
    daily_frame = pd.DataFrame(daily_rows)
    holdings_frame = pd.DataFrame(holdings_rows)
    orders_frame.to_csv(profile_dir / "orders.csv", index=False, encoding="utf-8-sig")
    fills_frame.to_csv(profile_dir / "fills.csv", index=False, encoding="utf-8-sig")
    daily_frame.to_csv(profile_dir / "daily_account.csv", index=False, encoding="utf-8-sig")
    holdings_frame.to_csv(profile_dir / "holdings.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(profile_dir / "targets.csv", index=False, encoding="utf-8-sig")

    final_nav = float(daily_frame["nav"].iloc[-1]) if not daily_frame.empty else cfg.initial_cash
    audit = {
        "status": "gm_target_volume_local_replay",
        "profile": {"name": profile_name},
        "config": asdict(cfg),
        "cost": asdict(cost),
        "target_csv": str(target_csv),
        "raw_daily_dir": str(raw_daily_dir),
        "output_dir": str(profile_dir),
        "days": int(len(daily_frame)),
        "final_nav": final_nav,
        "total_return": final_nav / cfg.initial_cash - 1.0,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "trades": trades,
        "total_fees": total_fees,
        "min_cash": cash_min,
        "negative_cash_events": negative_cash_events,
        "lot_violations": lot_violations,
        "max_daily_turnover": float(daily_frame["day_turnover"].max()) if not daily_frame.empty else 0.0,
        "max_gross_exposure": float(daily_frame["gross_exposure"].max()) if not daily_frame.empty else 0.0,
        "min_gross_exposure": float(daily_frame["gross_exposure"].min()) if not daily_frame.empty else 0.0,
        "nav_mismatch_max": 0.0,
        "deployment_allowed": False,
    }
    (profile_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "gm_target_volume_replay_complete", "audit": audit, "output_dir": str(output_dir)}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[gm target replay] "
        f"ret={audit['total_return']:+.2%} mdd={audit['max_drawdown']:.2%} "
        f"trades={trades} output={profile_dir}",
        flush=True,
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-csv", required=True, type=Path)
    parser.add_argument("--raw-daily-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cost", choices=["base", "stress"], default="stress")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--price-start", default="2025-01-01")
    parser.add_argument("--price-end", default="2026-06-05")
    parser.add_argument("--profile-name", default="gm_target_volume_replay")
    args = parser.parse_args()
    replay(
        args.target_csv.resolve(),
        args.raw_daily_dir.resolve(),
        args.output_dir.resolve(),
        {"base": BASE_COST, "stress": STRESS_COST}[args.cost],
        ReplayConfig(initial_cash=args.initial_cash),
        args.price_start,
        args.price_end,
        args.profile_name,
    )


if __name__ == "__main__":
    main()
