from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helki_quant.research.audit_rqdata_data_quality import (
    compare_price_frames,
)
from helki_quant.research.data_sources.rqdata_bridge import (
    local_to_rq,
    normalize_boolean_panel,
    normalize_price_frame,
    rq_to_local,
)
from helki_quant.research.data_sources.rqdata_source import (
    MarketDataGateway,
    local_symbol,
    merge_price_sources,
    read_license,
    read_price_csv,
)
from helki_quant.research.materialize_rqdata_canonical import (
    materialize_daily,
)


THRESHOLDS = {
    "min_daily_common_date_ratio": 0.98,
    "max_daily_return_abs_diff_p95": 0.005,
    "max_daily_normalized_ohlc_rel_diff_p95": 0.01,
    "min_daily_volume_log_correlation": 0.95,
    "min_daily_amount_log_correlation": 0.95,
    "min_minute_common_timestamp_ratio": 0.98,
    "max_minute_return_abs_diff_p95": 0.003,
    "max_minute_normalized_ohlc_rel_diff_p95": 0.005,
}


def price_frame(scale: float = 1.0, *, damaged: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", periods=12, freq="D")
    close = np.linspace(10.0, 11.1, len(dates)) * scale
    if damaged:
        close[4:8] *= 1.15
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.arange(1000, 2200, 100),
            "amount": np.arange(10000, 22000, 1000),
        }
    )


def test_symbol_conversions_cover_local_rq_and_gm_formats() -> None:
    assert local_symbol("000001.XSHE") == "sz000001"
    assert local_symbol("SHSE.600000") == "sh600000"
    assert local_to_rq("sz000001") == "000001.XSHE"
    assert rq_to_local("600000.XSHG") == "sh600000"


def test_placeholder_license_is_rejected(tmp_path: Path) -> None:
    secret = tmp_path / "license.txt"
    secret.write_text("PLACEHOLDER\n", encoding="utf-8")
    config = {
        "credentials": {
            "license_file": str(secret),
            "placeholder": "PLACEHOLDER",
        }
    }
    assert read_license(config, required=False) is None
    with pytest.raises(ValueError, match="not configured"):
        read_license(config, required=True)

    secret.write_text("PLACEHOLDER\nactual-license\n", encoding="utf-8")
    assert read_license(config, required=True) == "actual-license"


def test_primary_overrides_same_date_and_fallback_preserves_history() -> None:
    fallback = price_frame().iloc[:4].copy()
    primary = price_frame().iloc[3:6].copy()
    primary.loc[primary.index[0], "close"] = 99.0
    merged = merge_price_sources(primary, fallback)
    assert len(merged) == 6
    assert merged.loc[merged["date"] == primary.iloc[0]["date"], "close"].item() == 99.0
    assert merged.iloc[0]["close"] == fallback.iloc[0]["close"]


def test_quality_gate_accepts_front_adjustment_scale_only() -> None:
    result = compare_price_frames(
        price_frame(scale=2.5),
        price_frame(scale=1.0),
        frequency="1d",
        thresholds=THRESHOLDS,
    )
    assert result["passed"] is True
    assert result["normalized_ohlc_rel_diff_p95"] == pytest.approx(0.0)


def test_quality_gate_rejects_material_return_difference() -> None:
    result = compare_price_frames(
        price_frame(damaged=True),
        price_frame(),
        frequency="1d",
        thresholds=THRESHOLDS,
    )
    assert result["passed"] is False
    assert "return_abs_diff_p95" in result["failed_checks"]


def test_rqdata_multiindex_price_frame_normalizes() -> None:
    index = pd.MultiIndex.from_product(
        [["000001.XSHE"], pd.to_datetime(["2026-06-01", "2026-06-02"])],
        names=["order_book_id", "date"],
    )
    raw = pd.DataFrame(
        {
            "open": [10.0, 10.2],
            "high": [10.3, 10.4],
            "low": [9.9, 10.1],
            "close": [10.2, 10.3],
            "volume": [1000, 1100],
            "total_turnover": [10000, 11200],
        },
        index=index,
    )
    result = normalize_price_frame(raw, "1d")
    assert result["instrument"].tolist() == ["sz000001", "sz000001"]
    assert result["amount"].tolist() == [10000, 11200]


def test_rqdata_pit_boolean_panel_normalizes() -> None:
    raw = pd.DataFrame(
        {"000001.XSHE": [False, True], "600000.XSHG": [False, False]},
        index=pd.to_datetime(["2026-06-01", "2026-06-02"]),
    )
    result = normalize_boolean_panel(raw, "is_st")
    assert len(result) == 4
    assert set(result["instrument"]) == {"sz000001", "sh600000"}
    assert result.loc[
        (result["date"] == pd.Timestamp("2026-06-02"))
        & (result["instrument"] == "sz000001"),
        "is_st",
    ].item()


def test_daily_materialization_keeps_existing_history(tmp_path: Path) -> None:
    primary_root = tmp_path / "primary"
    fallback_root = tmp_path / "fallback"
    canonical_root = tmp_path / "canonical"
    for root in (primary_root, fallback_root, canonical_root):
        root.mkdir()
    config = {
        "mode": "rqdata_primary_local_fallback",
        "primary": {
            "daily_root": str(primary_root),
            "minute_root": str(tmp_path / "primary_minute"),
        },
        "fallback": {
            "daily_root": str(fallback_root),
            "minute_root": str(tmp_path / "fallback_minute"),
        },
    }
    gateway = MarketDataGateway(config)
    old = price_frame().iloc[:2]
    old.assign(date=old["date"].dt.strftime("%Y-%m-%d")).to_csv(
        canonical_root / "000001_daily_qfq.csv", index=False
    )
    incoming = price_frame().iloc[2:4]
    incoming.assign(date=incoming["date"].dt.strftime("%Y-%m-%d")).to_csv(
        primary_root / "000001_daily_qfq.csv", index=False
    )
    rows = materialize_daily(
        gateway,
        ["sz000001"],
        canonical_root,
        pd.Timestamp("2026-06-03"),
        pd.Timestamp("2026-06-04"),
    )
    result = read_price_csv(canonical_root / "000001_daily_qfq.csv", frequency="1d")
    assert len(rows) == 1
    assert result["date"].min() == pd.Timestamp("2026-06-01")
    assert result["date"].max() == pd.Timestamp("2026-06-04")
