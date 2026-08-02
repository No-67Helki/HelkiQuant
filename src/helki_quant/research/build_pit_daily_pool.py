from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .qlib_provider_writer import build_qlib_provider
except ImportError:
    from qlib_provider_writer import build_qlib_provider


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RAW_DIR = REPO_ROOT / "data" / "A_Stock_daily_qfq"
DEFAULT_STAGE = REPO_ROOT / "data" / "_research_pit_daily_csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "cn_data_research_pit"
BOARD_PREFIXES = ("30", "68")

COL_MAP = {
    "日期": "date",
    "date": "date",
    "开盘": "open",
    "open": "open",
    "收盘": "close",
    "close": "close",
    "最高": "high",
    "high": "high",
    "最低": "low",
    "low": "low",
    "成交量": "volume",
    "volume": "volume",
    "成交额": "amount",
    "amount": "amount",
    "total_turnover": "amount",
}
REQUIRED = ["date", "open", "close", "high", "low", "volume", "amount"]


def symbol_for(code: str) -> str:
    return f"sh{code}" if code.startswith("68") else f"sz{code}"


def candidate_files(raw_dir: Path, limit: int | None = None) -> list[Path]:
    files = [
        path
        for path in raw_dir.glob("*_daily_qfq.csv")
        if path.stem.split("_")[0].startswith(BOARD_PREFIXES)
    ]
    files = sorted(files)
    return files[:limit] if limit else files


def read_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda col: col in COL_MAP).rename(columns=COL_MAP)
    if not all(column in frame for column in REQUIRED):
        raise ValueError(f"missing required columns: {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = (
        frame.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    for column in REQUIRED[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def audit(files: list[Path], output_path: Path) -> dict:
    rows = []
    for pos, path in enumerate(files, start=1):
        code = path.stem.split("_")[0]
        try:
            frame = read_source(path)
            rows.append(
                {
                    "code": code,
                    "symbol": symbol_for(code),
                    "rows": len(frame),
                    "first": str(frame["date"].min().date()) if len(frame) else None,
                    "last": str(frame["date"].max().date()) if len(frame) else None,
                    "status": "ready" if len(frame) else "empty",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "code": code,
                    "symbol": symbol_for(code),
                    "rows": 0,
                    "first": None,
                    "last": None,
                    "status": f"error: {error}",
                }
            )
        if pos % 250 == 0:
            print(f"[audit] {pos}/{len(files)}", flush=True)
    report = {
        "status": "point_in_time_source_audit",
        "raw_dir": str(files[0].parent) if files else None,
        "source_rule": (
            "All raw 30/68 board files are included without latest-date liquidity, "
            "suspension, listing-age, or minimum-history selection."
        ),
        "residual_warning": (
            "The raw file library itself may still omit historical delisted/inactive names."
        ),
        "candidates": len(files),
        "ready": sum(row["status"] == "ready" for row in rows),
        "errors": sum(row["status"].startswith("error") for row in rows),
        "instruments": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def add_vwap(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "amount_div_volume":
        return frame["amount"] / frame["volume"].replace(0, float("nan"))
    if mode == "amount_div_volume100":
        return frame["amount"] / (frame["volume"].replace(0, float("nan")) * 100.0)
    if mode == "close":
        return frame["close"]
    if mode == "ohlc4":
        return (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
    raise ValueError(f"unknown vwap_mode={mode!r}")


def build(
    files: list[Path],
    stage_dir: Path,
    output_dir: Path,
    max_workers: int,
    vwap_mode: str,
) -> None:
    if stage_dir.exists() or output_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing staging/output directory: "
            f"stage={stage_dir.exists()} output={output_dir.exists()}"
        )
    stage_dir.mkdir(parents=True)
    written = 0
    for pos, path in enumerate(files, start=1):
        code = path.stem.split("_")[0]
        frame = read_source(path)
        if frame.empty:
            continue
        frame["factor"] = 1.0
        frame["vwap"] = add_vwap(frame, vwap_mode)
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        frame[["date", "open", "high", "low", "close", "volume", "amount", "vwap", "factor"]].to_csv(
            stage_dir / f"{symbol_for(code)}.csv", index=False
        )
        written += 1
        if pos % 250 == 0:
            print(f"[stage] {pos}/{len(files)} written={written}", flush=True)

    print(f"[dump] instruments={written} vwap_mode={vwap_mode} -> {output_dir}", flush=True)
    build_qlib_provider(
        stage_dir,
        output_dir,
        frequency="day",
        date_field="date",
        symbol_field="symbol",
        max_workers=max_workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "build"], default="audit")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--audit-output", default=str(HERE / "outputs" / "pit_daily_source_audit.json")
    )
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--vwap-mode",
        choices=["amount_div_volume", "amount_div_volume100", "close", "ohlc4"],
        default="amount_div_volume",
        help=(
            "How to fill the qlib $vwap field. For qfq canonical builds, "
            "'close' avoids amount/volume scale mismatch with adjusted OHLC."
        ),
    )
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    files = candidate_files(raw_dir, args.limit)
    if args.mode == "audit":
        report = audit(files, Path(args.audit_output).resolve())
        print(
            f"[PIT audit] candidates={report['candidates']} ready={report['ready']} "
            f"errors={report['errors']}"
        )
    else:
        build(
            files,
            Path(args.stage_dir).resolve(),
            Path(args.output_dir).resolve(),
            args.max_workers,
            args.vwap_mode,
        )


if __name__ == "__main__":
    main()
