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


def build(
    source_path: Path,
    output_path: Path,
    report_path: Path,
    overrides_path: Path | None = DEFAULT_OVERRIDES,
) -> dict:
    frame = pd.read_csv(source_path, dtype=str).fillna("")
    required = {"TS代码", "股票代码", "股票名称", "上市状态", "退市日期"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source_path} missing columns: {sorted(missing)}")
    override_rows = 0
    if overrides_path is not None and overrides_path.exists():
        overrides = pd.read_csv(overrides_path, dtype=str).fillna("")
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
    st_mask = name.str.startswith(("*ST", "ST", "S*ST", "SST", "PT"))
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
    args = parser.parse_args()
    build(
        Path(args.source).resolve(),
        Path(args.output).resolve(),
        Path(args.report).resolve(),
        Path(args.overrides).resolve() if args.overrides else None,
    )


if __name__ == "__main__":
    main()
