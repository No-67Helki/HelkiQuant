from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
RESEARCH_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
DEFAULT_CONFIG = RESEARCH_ROOT / "configs" / "rqdata_source.json"

DAILY_COLUMN_MAP = {
    "日期": "date",
    "date": "date",
    "开盘": "open",
    "open": "open",
    "最高": "high",
    "high": "high",
    "最低": "low",
    "low": "low",
    "收盘": "close",
    "close": "close",
    "成交量": "volume",
    "volume": "volume",
    "成交额": "amount",
    "total_turnover": "amount",
    "amount": "amount",
    "涨停价": "limit_up",
    "limit_up": "limit_up",
    "跌停价": "limit_down",
    "limit_down": "limit_down",
    "前收盘": "prev_close",
    "昨收": "prev_close",
    "prev_close": "prev_close",
}
MINUTE_COLUMN_MAP = {
    "时间": "date",
    "datetime": "date",
    "date": "date",
    "开盘价": "open",
    "open": "open",
    "最高价": "high",
    "high": "high",
    "最低价": "low",
    "low": "low",
    "收盘价": "close",
    "close": "close",
    "成交量": "volume",
    "volume": "volume",
    "成交额": "amount",
    "total_turnover": "amount",
    "amount": "amount",
    "涨停价": "limit_up",
    "limit_up": "limit_up",
    "跌停价": "limit_down",
    "limit_down": "limit_down",
    "前收盘": "prev_close",
    "昨收": "prev_close",
    "prev_close": "prev_close",
}
REQUIRED_PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]
OPTIONAL_PRICE_COLUMNS = ["limit_up", "limit_down", "prev_close"]


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("mode") != "rqdata_primary_local_fallback":
        raise ValueError(f"unsupported data-source mode in {path}")
    payload["_config_path"] = str(path)
    return payload


def resolve_repo_path(value: object) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def read_license(config: dict[str, Any], *, required: bool = True) -> str | None:
    credentials = config["credentials"]
    path = resolve_repo_path(credentials["license_file"])
    placeholder = str(credentials["placeholder"])
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"RQSDK license file not found: {path}")
        return None
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    values = [line for line in lines if line != placeholder]
    if not values:
        if required:
            raise ValueError(f"RQSDK license is not configured: {path}")
        return None
    if len(values) != 1:
        raise ValueError("RQSDK license file must contain exactly one non-empty line")
    return values[0]


def bridge_command(config: dict[str, Any], arguments: list[str]) -> list[str]:
    runtime = config["runtime"]
    python = resolve_repo_path(runtime["python"])
    bridge = resolve_repo_path(runtime["bridge"])
    if not python.is_file():
        raise FileNotFoundError(
            f"isolated RQSDK Python not found: {python}; run install_rqsdk_isolated.ps1"
        )
    if not bridge.is_file():
        raise FileNotFoundError(f"RQData bridge not found: {bridge}")
    return [str(python), str(bridge), *arguments]


def run_bridge(
    config: dict[str, Any],
    arguments: list[str],
    *,
    require_license: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    license_value = read_license(config, required=require_license)
    if license_value:
        environment["RQSDK_LICENSE_INPUT"] = license_value
    else:
        environment.pop("RQSDK_LICENSE_INPUT", None)
    command = bridge_command(config, arguments)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        check=False,
    )


def local_symbol(value: object) -> str:
    text = str(value).strip().lower()
    if text.endswith(".xshg"):
        return "sh" + text.split(".", 1)[0][-6:]
    if text.endswith(".xshe"):
        return "sz" + text.split(".", 1)[0][-6:]
    if text.startswith("shse."):
        return "sh" + text.split(".", 1)[1][-6:]
    if text.startswith("szse."):
        return "sz" + text.split(".", 1)[1][-6:]
    if text.startswith(("sh", "sz")) and len(text) >= 8:
        return text[:8]
    code = text[-6:]
    if not code.isdigit():
        raise ValueError(f"unsupported stock symbol: {value!r}")
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def read_price_csv(path: Path, *, frequency: str) -> pd.DataFrame:
    mapping = DAILY_COLUMN_MAP if frequency == "1d" else MINUTE_COLUMN_MAP
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="gbk")
    available = [column for column in frame.columns if column in mapping]
    out = frame[available].rename(columns=mapping)
    if len(set(REQUIRED_PRICE_COLUMNS) - set(out.columns)):
        missing = sorted(set(REQUIRED_PRICE_COLUMNS) - set(out.columns))
        raise ValueError(f"{path} missing price columns: {missing}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in [*REQUIRED_PRICE_COLUMNS[1:], *OPTIONAL_PRICE_COLUMNS]:
        if column not in out.columns:
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return (
        out.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def merge_price_sources(
    primary: pd.DataFrame | None,
    fallback: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge normalized bars, with RQData winning on equal timestamps."""
    parts: list[pd.DataFrame] = []
    if fallback is not None and not fallback.empty:
        local = fallback.copy()
        local["_source_priority"] = 0
        parts.append(local)
    if primary is not None and not primary.empty:
        rqdata = primary.copy()
        rqdata["_source_priority"] = 1
        parts.append(rqdata)
    if not parts:
        return pd.DataFrame(columns=REQUIRED_PRICE_COLUMNS)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "_source_priority"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .drop(columns=["_source_priority"])
        .reset_index(drop=True)
    )


def write_price_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output = output.dropna(subset=["date"])
    if output["date"].dt.normalize().equals(output["date"]):
        output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    else:
        output["date"] = output["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    pending = path.with_suffix(path.suffix + ".pending")
    output.to_csv(pending, index=False, encoding="utf-8-sig")
    pending.replace(path)


class MarketDataGateway:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.primary_daily = resolve_repo_path(config["primary"]["daily_root"])
        self.primary_minute = resolve_repo_path(config["primary"]["minute_root"])
        self.fallback_daily = resolve_repo_path(config["fallback"]["daily_root"])
        self.fallback_minute = resolve_repo_path(config["fallback"]["minute_root"])

    def daily_paths(self, symbol: object) -> tuple[Path, Path]:
        instrument = local_symbol(symbol)
        name = f"{instrument[2:]}_daily_qfq.csv"
        return self.primary_daily / name, self.fallback_daily / name

    def load_daily(self, symbol: object) -> tuple[pd.DataFrame, dict[str, Any]]:
        primary_path, fallback_path = self.daily_paths(symbol)
        sources: list[str] = []
        fallback: pd.DataFrame | None = None
        primary: pd.DataFrame | None = None
        if fallback_path.is_file():
            fallback = read_price_csv(fallback_path, frequency="1d")
            sources.append("local_fallback")
        if primary_path.is_file():
            primary = read_price_csv(primary_path, frequency="1d")
            sources.append("rqdata_primary")
        if primary is None and fallback is None:
            raise FileNotFoundError(
                f"no primary or fallback daily data for {local_symbol(symbol)}"
            )
        merged = merge_price_sources(primary, fallback)
        return merged, {
            "mode": self.config["mode"],
            "sources": sources,
            "primary_path": str(primary_path),
            "fallback_path": str(fallback_path),
            "rows": int(len(merged)),
            "first": merged["date"].min().strftime("%Y-%m-%d"),
            "last": merged["date"].max().strftime("%Y-%m-%d"),
        }

    def minute_primary_files(self, symbol: object) -> list[Path]:
        instrument = local_symbol(symbol)
        return sorted(self.primary_minute.glob(f"{instrument}_rqdata_*.csv"))


def write_symbol_file(path: Path, symbols: list[str]) -> Path:
    values: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        value = local_symbol(raw)
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError("at least one stock symbol is required")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")
    return path
