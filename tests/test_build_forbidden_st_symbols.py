from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

from build_forbidden_st_symbols import build  # noqa: E402


def test_rqdata_schema_and_exact_pit_st_are_combined(tmp_path: Path) -> None:
    source = tmp_path / "instruments.csv"
    pd.DataFrame(
        {
            "order_book_id": ["000001.XSHE", "000002.XSHE", "600000.XSHG"],
            "trading_code": ["000001", "000002", "600000"],
            "symbol": ["平安银行", "万科A", "浦发银行"],
            "special_type": ["Normal", "ST", "Normal"],
            "status": ["Active", "Active", "Active"],
            "de_listed_date": ["0000-00-00"] * 3,
            "industry_name": ["银行", "地产", "银行"],
        }
    ).to_csv(source, index=False)
    pit = tmp_path / "pit.csv"
    pd.DataFrame(
        {
            "date": ["2026-07-31"] * 3,
            "instrument": ["sz000001", "sz000002", "sh600000"],
            "is_st": [True, True, False],
            "is_suspended": [False, False, False],
        }
    ).to_csv(pit, index=False)
    output = tmp_path / "forbidden.csv"
    report_path = tmp_path / "report.json"

    report = build(
        source,
        output,
        report_path,
        overrides_path=None,
        pit_market_state_path=pit,
        as_of_date="2026-07-31",
    )

    result = pd.read_csv(output, dtype=str)
    assert set(result["instrument"]) == {"SZ000001", "SZ000002"}
    assert report["pit_st_symbols"] == 2
    assert report["rows_forbidden"] == 2
    assert "pit_is_st@2026-07-31" in result.set_index("instrument").loc[
        "SZ000001", "reason"
    ]


def test_pit_snapshot_requires_exact_date(tmp_path: Path) -> None:
    source = tmp_path / "instruments.csv"
    pd.DataFrame(
        {
            "order_book_id": ["000001.XSHE"],
            "trading_code": ["000001"],
            "symbol": ["平安银行"],
            "special_type": ["Normal"],
            "status": ["Active"],
            "de_listed_date": ["0000-00-00"],
        }
    ).to_csv(source, index=False)
    pit = tmp_path / "pit.csv"
    pd.DataFrame(
        {"date": ["2026-07-30"], "instrument": ["sz000001"], "is_st": [False]}
    ).to_csv(pit, index=False)

    try:
        build(
            source,
            tmp_path / "forbidden.csv",
            tmp_path / "report.json",
            overrides_path=None,
            pit_market_state_path=pit,
            as_of_date="2026-07-31",
        )
    except ValueError as exc:
        assert "missing exact date" in str(exc)
    else:
        raise AssertionError("missing PIT date must fail closed")
