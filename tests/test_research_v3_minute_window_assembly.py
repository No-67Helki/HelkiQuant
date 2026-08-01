from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


RESEARCH = Path(__file__).resolve().parents[1] / "src" / "helki_quant" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from assemble_minute_windows import assemble  # noqa: E402


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_assemble_accepts_matching_overlap_and_fills_missing_value(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write(
        first,
        [
            {
                "trade_date": "2025-01-03",
                "instrument": "sz300001",
                "open_exec": 10.0,
                "close_exec": None,
                "mark_close": 10.2,
            }
        ],
    )
    _write(
        second,
        [
            {
                "trade_date": "2025-01-03",
                "instrument": "SZ300001",
                "open_exec": 10.0,
                "close_exec": 10.1,
                "mark_close": 10.2,
            }
        ],
    )

    report = assemble([first, second], tmp_path / "out.csv", tmp_path / "report.json")
    out = pd.read_csv(tmp_path / "out.csv")

    assert report["overlap_keys"] == 1
    assert len(out) == 1
    assert out.loc[0, "instrument"] == "SZ300001"
    assert out.loc[0, "close_exec"] == pytest.approx(10.1)


def test_assemble_rejects_conflicting_overlap(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    base = {
        "trade_date": "2025-01-03",
        "instrument": "SZ300001",
        "open_exec": 10.0,
        "close_exec": 10.1,
        "mark_close": 10.2,
    }
    _write(first, [base])
    _write(second, [{**base, "open_exec": 10.5}])

    with pytest.raises(ValueError, match="Conflicting minute windows"):
        assemble([first, second], tmp_path / "out.csv", tmp_path / "report.json")
