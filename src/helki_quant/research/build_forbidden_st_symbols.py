from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_SOURCE = DATA / "股票列表.csv"
DEFAULT_OVERRIDES = DATA / "forbidden_st_manual_overrides.csv"
DEFAULT_OUTPUT = DATA / "forbidden_st_symbols_20260605.csv"
DEFAULT_REPORT = HERE / "outputs" / "forbidden_st_symbols_20260605_report.json"


def to_local_instrument(ts_code: object) -> str:
    text = str(ts_code).strip().upper()
    if "." in text:
        code, exchange = text.split(".", 1)
        prefix = "SH" if exchange in {"SH", "SHSE"} else "SZ"
        return f"{prefix}{code}"
    code = text[-6:]
    prefix = "SH" if code.startswith(("6", "9")) else "SZ"
    return f"{prefix}{code}"


def to_gm_symbol(local_instrument: object) -> str:
    text = str(local_instrument).strip().upper()
    code = text[-6:]
    if text.startswith("SH") or code.startswith(("6", "9")):
        return f"SHSE.{code}"
    return f"SZSE.{code}"


def _normalise_instruments(frame: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    legacy_required = {"TS代码", "股票代码", "股票名称", "上市状态", "退市日期"}
    if legacy_required.issubset(frame.columns):
        result = frame.copy()
        if "所属行业" not in result.columns:
            result["所属行业"] = ""
        result["_source_st"] = False
        return result

    rq_required = {
        "order_book_id",
        "trading_code",
        "symbol",
        "special_type",
        "status",
        "de_listed_date",
    }
    if not rq_required.issubset(frame.columns):
        missing = sorted(legacy_required - set(frame.columns))
        raise ValueError(f"{source_path} missing legacy or RQData columns: {missing}")

    order_book_id = frame["order_book_id"].astype(str).str.upper()
    exchange = order_book_id.str.rsplit(".", n=1).str[-1]
    ts_exchange = exchange.map({"XSHG": "SH", "XSHE": "SZ"}).fillna(exchange)
    result = pd.DataFrame(
        {
            "TS代码": frame["trading_code"].astype(str).str.zfill(6) + "." + ts_exchange,
            "股票代码": frame["trading_code"].astype(str).str.zfill(6),
            "股票名称": frame["symbol"].astype(str),
            "上市状态": frame["status"].astype(str).map(
                lambda value: "上市" if value.strip().lower() == "active" else value
            ),
            "退市日期": frame["de_listed_date"].astype(str).replace("0000-00-00", ""),
            "所属行业": frame.get("industry_name", pd.Series("", index=frame.index)).astype(str),
            "_source_st": frame["special_type"].astype(str).str.upper().isin(
                {"ST", "STARST"}
            ),
        }
    )
    return result


def build(
    source_path: Path,
    output_path: Path,
    report_path: Path,
    overrides_path: Path | None = DEFAULT_OVERRIDES,
    *,
    pit_market_state_path: Path | None = None,
    as_of_date: str | None = None,
) -> dict:
    raw = pd.read_csv(source_path, dtype=str).fillna("")
    frame = _normalise_instruments(raw, source_path).fillna("")
    required = {"TS代码", "股票代码", "股票名称", "上市状态", "退市日期"}
    override_rows = 0
    if overrides_path is not None and overrides_path.exists():
        overrides = _normalise_instruments(
            pd.read_csv(overrides_path, dtype=str).fillna(""), overrides_path
        ).fillna("")
        override_missing = required - set(overrides.columns)
        if override_missing:
            raise ValueError(f"{overrides_path} missing columns: {sorted(override_missing)}")
        override_rows = int(len(overrides))
        frame = pd.concat([frame, overrides], ignore_index=True, sort=False).fillna("")
        frame = frame.drop_duplicates(subset=["TS代码"], keep="last").reset_index(drop=True)

    name = (
        frame["股票名称"]
        .astype(str)
        .str.upper()
        .str.replace("＊", "*", regex=False)
        .str.replace(r"\s+", "", regex=True)
    )
    status = frame["上市状态"].astype(str)
    delist_date = frame["退市日期"].astype(str).str.strip()
    st_mask = name.str.startswith(("*ST", "ST", "S*ST", "SST", "PT")) | frame[
        "_source_st"
    ].astype(bool)
    delist_name_mask = frame["股票名称"].astype(str).str.contains("退市", regex=False)
    non_listed_mask = ~status.eq("上市")
    delisted_mask = delist_date.ne("")
    blocked = frame[st_mask | delist_name_mask | non_listed_mask | delisted_mask].copy()

    reasons = []
    for idx in blocked.index:
        row_reasons = []
        if bool(st_mask.loc[idx]):
            row_reasons.append("name_contains_ST")
        if bool(delist_name_mask.loc[idx]):
            row_reasons.append("name_contains_delist")
        if bool(non_listed_mask.loc[idx]):
            row_reasons.append(f"listing_status={status.loc[idx]}")
        if bool(delisted_mask.loc[idx]):
            row_reasons.append("delist_date_present")
        reasons.append(";".join(row_reasons))
    blocked["instrument"] = blocked["TS代码"].map(to_local_instrument)
    blocked["gm_symbol"] = blocked["instrument"].map(to_gm_symbol)
    blocked["reason"] = reasons

    pit_st_count = 0
    pit_snapshot_rows = 0
    if pit_market_state_path is not None:
        if as_of_date is None:
            raise ValueError("as_of_date is required with pit_market_state_path")
        pit = pd.read_csv(pit_market_state_path, dtype=str).fillna("")
        pit_required = {"date", "instrument", "is_st"}
        pit_missing = pit_required - set(pit.columns)
        if pit_missing:
            raise ValueError(
                f"{pit_market_state_path} missing columns: {sorted(pit_missing)}"
            )
        target_date = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
        snapshot = pit[pit["date"].astype(str).eq(target_date)].copy()
        if snapshot.empty:
            raise ValueError(
                f"PIT market-state snapshot is missing exact date {target_date}"
            )
        pit_snapshot_rows = int(len(snapshot))
        is_st = snapshot["is_st"].astype(str).str.lower().isin({"true", "1", "yes"})
        pit_instruments = {
            to_local_instrument(value) for value in snapshot.loc[is_st, "instrument"]
        }
        pit_st_count = len(pit_instruments)
        existing = set(blocked["instrument"].astype(str))
        additions = []
        indexed = frame.assign(
            instrument=frame["TS代码"].map(to_local_instrument)
        ).drop_duplicates("instrument", keep="last").set_index("instrument")
        for instrument in sorted(pit_instruments - existing):
            if instrument in indexed.index:
                row = indexed.loc[instrument]
                additions.append(
                    {
                        "instrument": instrument,
                        "gm_symbol": to_gm_symbol(instrument),
                        "TS代码": row["TS代码"],
                        "股票代码": row["股票代码"],
                        "股票名称": row["股票名称"],
                        "上市状态": row["上市状态"],
                        "退市日期": row["退市日期"],
                        "所属行业": row["所属行业"],
                        "reason": f"pit_is_st@{target_date}",
                    }
                )
            else:
                additions.append(
                    {
                        "instrument": instrument,
                        "gm_symbol": to_gm_symbol(instrument),
                        "TS代码": instrument[-6:] + (".SH" if instrument.startswith("SH") else ".SZ"),
                        "股票代码": instrument[-6:],
                        "股票名称": "",
                        "上市状态": "",
                        "退市日期": "",
                        "所属行业": "",
                        "reason": f"pit_is_st@{target_date}",
                    }
                )
        if additions:
            blocked = pd.concat([blocked, pd.DataFrame(additions)], ignore_index=True)
        pit_mask = blocked["instrument"].isin(pit_instruments)
        static_mask = ~blocked["reason"].astype(str).str.contains("pit_is_st", regex=False)
        blocked.loc[pit_mask & static_mask, "reason"] = (
            blocked.loc[pit_mask & static_mask, "reason"].astype(str)
            + f";pit_is_st@{target_date}"
        ).str.strip(";")
    out = blocked[
        [
            "instrument",
            "gm_symbol",
            "TS代码",
            "股票代码",
            "股票名称",
            "上市状态",
            "退市日期",
            "所属行业",
            "reason",
        ]
    ].sort_values(["instrument"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    report = {
        "status": "forbidden_st_symbols_built",
        "source": str(source_path),
        "overrides": str(overrides_path) if overrides_path is not None else None,
        "override_rows": override_rows,
        "output": str(output_path),
        "rows_source": int(len(frame)),
        "rows_forbidden": int(len(out)),
        "unique_instruments": int(out["instrument"].nunique()),
        "name_contains_ST": int(st_mask.sum()),
        "name_contains_delist": int(delist_name_mask.sum()),
        "non_listed": int(non_listed_mask.sum()),
        "delist_date_present": int(delisted_mask.sum()),
        "pit_market_state": str(pit_market_state_path) if pit_market_state_path else None,
        "pit_as_of_date": as_of_date,
        "pit_snapshot_rows": pit_snapshot_rows,
        "pit_st_symbols": pit_st_count,
        "policy": (
            "No ST, *ST, delisting-name, non-listed, or delist-date symbols may "
            "enter candidates, targets, or orders."
        ),
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[forbidden-st] rows={report['rows_forbidden']} output={output_path}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--overrides", default=str(DEFAULT_OVERRIDES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--pit-market-state")
    parser.add_argument("--as-of-date")
    args = parser.parse_args()
    build(
        Path(args.source).resolve(),
        Path(args.output).resolve(),
        Path(args.report).resolve(),
        Path(args.overrides).resolve() if args.overrides else None,
        pit_market_state_path=(
            Path(args.pit_market_state).resolve() if args.pit_market_state else None
        ),
        as_of_date=args.as_of_date,
    )


if __name__ == "__main__":
    main()
