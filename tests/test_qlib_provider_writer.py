from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helki_quant.research.qlib_provider_writer import build_qlib_provider


def _write_source(path: Path, dates: list[str], closes: list[float]) -> None:
    pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * len(dates),
            "amount": [10000.0] * len(dates),
            "factor": [1.0] * len(dates),
        }
    ).to_csv(path, index=False)


def test_build_qlib_provider_writes_calendar_instruments_and_aligned_bins(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_source(
        source / "sz000001.csv",
        ["2026-06-01", "2026-06-03"],
        [10.0, 12.0],
    )
    _write_source(
        source / "sh600000.csv",
        ["2026-06-02", "2026-06-03"],
        [20.0, 21.0],
    )
    output = tmp_path / "provider"

    report = build_qlib_provider(source, output, max_workers=2)

    assert report["calendar_rows"] == 3
    assert report["instruments"] == 2
    assert (output / "calendars" / "day.txt").read_text().splitlines() == [
        "2026-06-01",
        "2026-06-02",
        "2026-06-03",
    ]
    assert (output / "instruments" / "all.txt").read_text().splitlines() == [
        "SH600000\t2026-06-02\t2026-06-03",
        "SZ000001\t2026-06-01\t2026-06-03",
    ]
    sz_close = np.fromfile(
        output / "features" / "sz000001" / "close.day.bin",
        dtype="<f4",
    )
    sh_close = np.fromfile(
        output / "features" / "sh600000" / "close.day.bin",
        dtype="<f4",
    )
    assert sz_close[:2].tolist() == [0.0, 10.0]
    assert np.isnan(sz_close[2])
    assert sz_close[3] == pytest.approx(12.0)
    assert sh_close.tolist() == pytest.approx([1.0, 20.0, 21.0])


def test_build_qlib_provider_refuses_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_source(source / "sz000001.csv", ["2026-06-01"], [10.0])
    output = tmp_path / "provider"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_qlib_provider(source, output)
