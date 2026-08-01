from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    return float((1.0 - nav / nav.cummax()).max())


def audit(report_path: Path, trades_path: Path, daily_path: Path, output_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    best = report.get("best_qualified") or report.get("best")
    if not best:
        raise ValueError(f"no best or best_qualified in {report_path}")
    trades = pd.read_csv(trades_path, parse_dates=["trade_date", "signal_date"])
    daily = pd.read_csv(daily_path, parse_dates=["trade_date"])
    if trades.empty:
        raise ValueError(f"empty trades file: {trades_path}")
    mask = (
        np.isclose(trades["threshold"].astype(float), float(best["threshold"]))
        & np.isclose(trades["trade_fraction"].astype(float), float(best["trade_fraction"]))
    )
    selected_trades = trades.loc[mask].copy()
    daily_mask = (
        np.isclose(daily["threshold"].astype(float), float(best["threshold"]))
        & np.isclose(daily["trade_fraction"].astype(float), float(best["trade_fraction"]))
    )
    selected_daily = daily.loc[daily_mask].copy()
    if selected_trades.empty or selected_daily.empty:
        raise ValueError("no rows matched best candidate setting")

    by_symbol = selected_trades.groupby("instrument")["pnl"].agg(["count", "sum"]).sort_values(
        "sum", ascending=False
    )
    by_month = selected_daily.assign(month=selected_daily["trade_date"].dt.to_period("M").astype(str)).groupby(
        "month"
    ).agg(
        day_pnl=("day_pnl", "sum"),
        trades=("day_trades", "sum"),
        max_day_turnover=("day_turnover", "max"),
    )
    by_quarter = selected_daily.assign(
        quarter=selected_daily["trade_date"].dt.to_period("Q").astype(str)
    ).groupby("quarter").agg(
        day_pnl=("day_pnl", "sum"),
        trades=("day_trades", "sum"),
        max_day_turnover=("day_turnover", "max"),
    )
    pnl_total = float(selected_trades["pnl"].sum())
    gross_positive = float(selected_trades.loc[selected_trades["pnl"] > 0, "pnl"].sum())
    gross_negative = float(selected_trades.loc[selected_trades["pnl"] < 0, "pnl"].sum())
    top_symbol_pnl = float(by_symbol["sum"].iloc[0]) if len(by_symbol) else 0.0
    top3_symbol_pnl = float(by_symbol["sum"].head(3).sum()) if len(by_symbol) else 0.0
    worst_trade = selected_trades.sort_values("pnl").head(5)
    best_trade = selected_trades.sort_values("pnl", ascending=False).head(5)
    losing_months = int((by_month["day_pnl"] < 0).sum()) if len(by_month) else 0
    active_months = int((by_month["trades"] > 0).sum()) if len(by_month) else 0

    result = {
        "status": "inner_t0_replay_audited",
        "report": str(report_path.resolve()),
        "trades": str(trades_path.resolve()),
        "daily": str(daily_path.resolve()),
        "selected_setting": {
            "threshold": float(best["threshold"]),
            "trade_fraction": float(best["trade_fraction"]),
        },
        "selected_metrics": best,
        "round_trips": int(len(selected_trades)),
        "orders": int(len(selected_trades) * 2),
        "pnl_total": pnl_total,
        "gross_positive_pnl": gross_positive,
        "gross_negative_pnl": gross_negative,
        "profit_factor": float(gross_positive / abs(gross_negative)) if gross_negative < 0 else None,
        "top_symbol_pnl_share": float(top_symbol_pnl / pnl_total) if pnl_total != 0 else None,
        "top3_symbol_pnl_share": float(top3_symbol_pnl / pnl_total) if pnl_total != 0 else None,
        "symbols_traded": int(by_symbol.shape[0]),
        "active_months": active_months,
        "losing_months": losing_months,
        "worst_month_pnl": float(by_month["day_pnl"].min()) if len(by_month) else 0.0,
        "best_month_pnl": float(by_month["day_pnl"].max()) if len(by_month) else 0.0,
        "max_overlay_drawdown": max_drawdown(selected_daily["overlay_nav"]),
        "top_symbols": by_symbol.head(10).reset_index().to_dict(orient="records"),
        "monthly": by_month.reset_index().to_dict(orient="records"),
        "quarterly": by_quarter.reset_index().to_dict(orient="records"),
        "worst_trades": worst_trade.to_dict(orient="records"),
        "best_trades": best_trade.to_dict(orient="records"),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--daily", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(
        Path(args.report).resolve(),
        Path(args.trades).resolve(),
        Path(args.daily).resolve(),
        Path(args.output).resolve(),
    )
    print(
        "[inner t0 audit] "
        f"round_trips={result['round_trips']} pnl={result['pnl_total']:.2f} "
        f"top3_share={result['top3_symbol_pnl_share']:.2%} "
        f"active_months={result['active_months']} losing_months={result['losing_months']}"
    )


if __name__ == "__main__":
    main()
