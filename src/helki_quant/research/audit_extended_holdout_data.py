from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_RAW = DATA / "A_Stock_daily_qfq" / "daily_qfq_6.5"
DEFAULT_STAGE = DATA / "_research_pit_daily_csv_20260605"
DEFAULT_OLD_STAGE = DATA / "_research_pit_daily_csv"
DEFAULT_PROVIDER = DATA / "cn_data_research_pit_20260605"
DEFAULT_PREDICTION = HERE / "outputs" / "oof" / "pit_holdout_20260605_de2_srfs_es" / "middle" / "fold_99.csv"
DEFAULT_OOF = HERE / "outputs" / "oof_combined" / "middle_oof_pit_de2_srfs_es.csv"
DEFAULT_OUTPUT = HERE / "outputs" / "extended_holdout_data_audit_20260605.json"


RAW_COLS = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}
PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def raw_path(raw_dir: Path, symbol: str) -> Path:
    return raw_dir / f"{symbol[-6:]}_daily_qfq.csv"


def stage_path(stage_dir: Path, symbol: str) -> Path:
    return stage_dir / f"{symbol.lower()}.csv"


def read_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda col: col in RAW_COLS).rename(columns=RAW_COLS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in PRICE_COLS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")


def read_stage(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    for col in PRICE_COLS + ["vwap"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")


def read_provider_instruments(provider: Path) -> list[str]:
    rows = []
    for line in (provider / "instruments" / "all.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(line.split("\t")[0].upper())
    return rows


def quantiles(values: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(len(clean)),
        "min": float(clean.min()),
        "p01": float(clean.quantile(0.01)),
        "p05": float(clean.quantile(0.05)),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def audit_raw_stage(raw_dir: Path, stage_dir: Path, symbols: list[str]) -> dict:
    rows = []
    ratio_values = []
    unadjusted_ratio_values = []
    mismatches = []
    for symbol in symbols:
        r_path = raw_path(raw_dir, symbol)
        s_path = stage_path(stage_dir, symbol)
        if not r_path.exists() or not s_path.exists():
            mismatches.append({"symbol": symbol, "reason": "missing_raw_or_stage"})
            continue
        raw = read_raw(r_path)
        stage = read_stage(s_path)
        merged = raw.merge(stage, on="date", suffixes=("_raw", "_stage"))
        max_abs = {}
        for col in PRICE_COLS:
            diff = (merged[f"{col}_raw"] - merged[f"{col}_stage"]).abs()
            max_abs[col] = float(diff.max()) if len(diff) else None
        bad_cols = [col for col, value in max_abs.items() if value is not None and value > 1e-8]
        if bad_cols:
            mismatches.append({"symbol": symbol, "reason": "raw_stage_value_mismatch", "max_abs": max_abs})
        ratio_values.append(stage["vwap"] / stage["close"].replace(0, np.nan))
        raw_unadjusted_vwap = stage["amount"] / (stage["volume"].replace(0, np.nan) * 100.0)
        unadjusted_ratio_values.append(raw_unadjusted_vwap / stage["close"].replace(0, np.nan))
        rows.append(
            {
                "symbol": symbol,
                "raw_rows": int(len(raw)),
                "stage_rows": int(len(stage)),
                "merged_rows": int(len(merged)),
                "first": str(stage["date"].min().date()) if len(stage) else None,
                "last": str(stage["date"].max().date()) if len(stage) else None,
                "max_abs": max_abs,
            }
        )
    ratio = pd.concat(ratio_values, ignore_index=True) if ratio_values else pd.Series(dtype=float)
    unadjusted_ratio = (
        pd.concat(unadjusted_ratio_values, ignore_index=True) if unadjusted_ratio_values else pd.Series(dtype=float)
    )
    return {
        "checked_symbols": int(len(rows)),
        "mismatch_count": int(len(mismatches)),
        "mismatch_sample": mismatches[:20],
        "row_count_min": int(min((row["stage_rows"] for row in rows), default=0)),
        "row_count_max": int(max((row["stage_rows"] for row in rows), default=0)),
        "last_date_counts": pd.Series([row["last"] for row in rows]).value_counts().head(20).to_dict(),
        "stage_vwap_div_close": quantiles(ratio),
        "amount_div_volume100_div_close": quantiles(unadjusted_ratio),
        "vwap_warning": (
            "stage vwap is amount/volume. Because raw OHLC is qfq-adjusted while amount/volume is not, "
            "vwap is not on the same scale as qfq prices."
        ),
    }


def audit_stage_overlap(old_stage: Path, new_stage: Path, symbols: list[str]) -> dict:
    rows = []
    changed = []
    for symbol in symbols:
        old_path = stage_path(old_stage, symbol)
        new_path = stage_path(new_stage, symbol)
        if not old_path.exists() or not new_path.exists():
            continue
        old = read_stage(old_path)
        new = read_stage(new_path)
        merged = old.merge(new, on="date", suffixes=("_old", "_new"))
        if merged.empty:
            continue
        max_abs = {}
        for col in PRICE_COLS + ["vwap"]:
            if f"{col}_old" not in merged or f"{col}_new" not in merged:
                continue
            diff = (merged[f"{col}_old"] - merged[f"{col}_new"]).abs()
            max_abs[col] = float(diff.max()) if len(diff) else None
        material = {
            col: value
            for col, value in max_abs.items()
            if value is not None and value > (1e-8 if col != "amount" else 1e-4)
        }
        if material:
            changed.append({"symbol": symbol, "max_abs": material})
        rows.append(
            {
                "symbol": symbol,
                "overlap_rows": int(len(merged)),
                "old_last": str(old["date"].max().date()) if len(old) else None,
                "new_last": str(new["date"].max().date()) if len(new) else None,
            }
        )
    return {
        "checked_symbols": int(len(rows)),
        "changed_symbol_count": int(len(changed)),
        "changed_sample": changed[:20],
        "overlap_rows_min": int(min((row["overlap_rows"] for row in rows), default=0)),
        "overlap_rows_max": int(max((row["overlap_rows"] for row in rows), default=0)),
    }


def prediction_stats(path: Path, name: str, start: str | None = None, end: str | None = None) -> dict:
    if not path.exists():
        return {"name": name, "exists": False}
    frame = pd.read_csv(path, parse_dates=["datetime"])
    if start:
        frame = frame[frame["datetime"] >= pd.Timestamp(start)]
    if end:
        frame = frame[frame["datetime"] <= pd.Timestamp(end)]
    stats = frame.groupby("datetime")["middle"].agg(["count", "mean", "std", "min", "max"])
    return {
        "name": name,
        "exists": True,
        "rows": int(len(frame)),
        "dates": int(frame["datetime"].nunique()) if len(frame) else 0,
        "start": str(frame["datetime"].min().date()) if len(frame) else None,
        "end": str(frame["datetime"].max().date()) if len(frame) else None,
        "overall": quantiles(frame["middle"]),
        "daily_count": quantiles(stats["count"]) if len(stats) else {"count": 0},
        "daily_mean": quantiles(stats["mean"]) if len(stats) else {"count": 0},
        "daily_std": quantiles(stats["std"]) if len(stats) else {"count": 0},
    }


def load_close_panel(raw_dir: Path, symbols: list[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        path = raw_path(raw_dir, symbol)
        if not path.exists():
            continue
        frame = read_raw(path)[["date", "close"]].copy()
        frame["instrument"] = symbol
        rows.append(frame.rename(columns={"date": "datetime"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["datetime", "close", "instrument"])


def label_diagnostic(raw_dir: Path, prediction_path: Path) -> dict:
    pred = pd.read_csv(prediction_path, parse_dates=["datetime"])
    symbols = sorted(pred["instrument"].astype(str).str.upper().unique().tolist())
    close = load_close_panel(raw_dir, symbols)
    close = close.sort_values(["instrument", "datetime"])
    grouped = close.groupby("instrument", sort=False)
    close["next_close"] = grouped["close"].shift(-1)
    close["fwd6_close"] = grouped["close"].shift(-6)
    close["label_5d_exec_like"] = close["fwd6_close"] / close["next_close"] - 1.0
    merged = pred.merge(close[["datetime", "instrument", "label_5d_exec_like"]], on=["datetime", "instrument"])
    merged = merged.dropna(subset=["middle", "label_5d_exec_like"])
    daily_rows = []
    for date, part in merged.groupby("datetime", sort=True):
        if len(part) < 20 or part["middle"].nunique() < 2:
            continue
        ic = spearmanr(part["middle"], part["label_5d_exec_like"], nan_policy="omit").correlation
        ranked = part.sort_values("middle", ascending=False)
        k = max(1, int(len(ranked) * 0.1))
        spread = ranked.head(k)["label_5d_exec_like"].mean() - ranked.tail(k)["label_5d_exec_like"].mean()
        daily_rows.append({"datetime": date, "ic": ic, "top_bottom_10pct_spread": spread, "count": len(part)})
    daily = pd.DataFrame(daily_rows)
    return {
        "label": "close[t+6] / close[t+1] - 1, available only where future close exists",
        "rows": int(len(merged)),
        "dates": int(merged["datetime"].nunique()) if len(merged) else 0,
        "daily_ic_mean": float(daily["ic"].mean()) if len(daily) else None,
        "daily_ic_median": float(daily["ic"].median()) if len(daily) else None,
        "positive_ic_ratio": float((daily["ic"] > 0).mean()) if len(daily) else None,
        "daily_spread_mean": float(daily["top_bottom_10pct_spread"].mean()) if len(daily) else None,
        "daily_spread_median": float(daily["top_bottom_10pct_spread"].median()) if len(daily) else None,
        "daily_sample": [
            {
                "datetime": str(row.datetime.date()),
                "ic": float(row.ic),
                "top_bottom_10pct_spread": float(row.top_bottom_10pct_spread),
                "count": int(row.count),
            }
            for row in daily.head(10).itertuples(index=False)
        ],
    }


def run(
    raw_dir: Path,
    stage_dir: Path,
    old_stage_dir: Path,
    provider_dir: Path,
    prediction_path: Path,
    oof_path: Path,
    output_path: Path,
    sample_limit: int,
) -> dict:
    provider_symbols = read_provider_instruments(provider_dir)
    symbols = provider_symbols[:sample_limit] if sample_limit > 0 else provider_symbols
    raw_stage = audit_raw_stage(raw_dir, stage_dir, symbols)
    overlap = audit_stage_overlap(old_stage_dir, stage_dir, symbols)
    pred_ext = prediction_stats(prediction_path, "extended_20260605")
    pred_oof = prediction_stats(oof_path, "oof_2025_2026", start="2025-01-03", end="2026-04-02")
    pred_micro = prediction_stats(oof_path, "oof_tail_before_holdout", start="2026-03-02", end="2026-04-02")
    label_diag = label_diagnostic(raw_dir, prediction_path)
    whitelist = json.loads(
        (HERE / "outputs" / "factor_reports" / "pit_holdout_de2_srfs_es" / "fold_99" / "feature_whitelist_middle_v2.json")
        .read_text(encoding="utf-8")
    )
    kept = set(whitelist.get("kept", []))
    report = {
        "status": "extended_holdout_data_audit_research_only",
        "raw_dir": str(raw_dir),
        "stage_dir": str(stage_dir),
        "old_stage_dir": str(old_stage_dir),
        "provider_dir": str(provider_dir),
        "prediction": str(prediction_path),
        "provider_instrument_count": len(provider_symbols),
        "sample_limit": sample_limit,
        "raw_stage": raw_stage,
        "old_new_stage_overlap": overlap,
        "prediction_stats": [pred_oof, pred_micro, pred_ext],
        "short_label_diagnostic": label_diag,
        "middle_whitelist_vwap_exposure": {
            "kept_feature_count": len(kept),
            "contains_vwap0": "VWAP0" in kept,
            "contains_any_vwap": any("VWAP" in name.upper() for name in kept),
            "contains_wvma": any("WVMA" in name.upper() for name in kept),
            "vwap_related_kept": sorted([name for name in kept if "VWAP" in name.upper() or "WVMA" in name.upper()]),
        },
        "preliminary_interpretation": (
            "Raw-to-stage OHLCV should match exactly. Stage vwap scale is expected to be incompatible with qfq prices "
            "because factor is fixed at 1.0; this is a data-design issue to remove or repair in the next feature build. "
            "The current middle whitelist does not keep VWAP0, but it may keep WVMA features."
        ),
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE))
    parser.add_argument("--old-stage-dir", default=str(DEFAULT_OLD_STAGE))
    parser.add_argument("--provider", default=str(DEFAULT_PROVIDER))
    parser.add_argument("--prediction", default=str(DEFAULT_PREDICTION))
    parser.add_argument("--oof", default=str(DEFAULT_OOF))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-limit", type=int, default=0, help="0 checks all provider instruments")
    args = parser.parse_args()
    report = run(
        Path(args.raw_dir).resolve(),
        Path(args.stage_dir).resolve(),
        Path(args.old_stage_dir).resolve(),
        Path(args.provider).resolve(),
        Path(args.prediction).resolve(),
        Path(args.oof).resolve(),
        Path(args.output).resolve(),
        args.sample_limit,
    )
    label = report["short_label_diagnostic"]
    vwap = report["raw_stage"]["stage_vwap_div_close"]
    print(
        "[extended data audit] "
        f"symbols={report['provider_instrument_count']} raw_stage_mismatch={report['raw_stage']['mismatch_count']} "
        f"old_new_changed={report['old_new_stage_overlap']['changed_symbol_count']} "
        f"vwap_close_median={vwap.get('median')} "
        f"label_ic_mean={label['daily_ic_mean']} spread_mean={label['daily_spread_mean']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
