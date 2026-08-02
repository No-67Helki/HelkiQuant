from __future__ import annotations

from pathlib import Path

import pandas as pd

from helki_quant.research.universe import load_price_panel


def test_load_price_panel_accepts_canonical_english_columns(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "date": ["2026-06-04", "2026-06-05", "2026-06-08"],
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [1000, 1100, 1200],
            "amount": [10100, 11220, 12360],
        }
    ).to_csv(tmp_path / "300001_daily_qfq.csv", index=False)

    result = load_price_panel(
        tmp_path,
        ["SZ300001"],
        start="2026-06-05",
        end="2026-06-08",
    )

    assert result["instrument"].unique().tolist() == ["SZ300001"]
    assert result["datetime"].tolist() == [
        pd.Timestamp("2026-06-05"),
        pd.Timestamp("2026-06-08"),
    ]
    assert result["close"].tolist() == [10.2, 10.3]
