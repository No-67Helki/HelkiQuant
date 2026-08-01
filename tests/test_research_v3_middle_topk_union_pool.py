from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


RESEARCH = Path(__file__).resolve().parents[1] / "src" / "helki_quant" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from build_middle_topk_union_pool import build_pool  # noqa: E402


def test_build_pool_filters_forbidden_and_outputs_only_missing_windows(tmp_path: Path) -> None:
    middle = tmp_path / "middle.csv"
    pd.DataFrame(
        [
            {"datetime": "2024-10-10", "instrument": "SZ000001", "middle": 0.9},
            {"datetime": "2024-10-10", "instrument": "SZ000002", "middle": 0.8},
            {"datetime": "2024-10-10", "instrument": "SZ000003", "middle": 0.7},
            {"datetime": "2024-10-11", "instrument": "SZ000003", "middle": 0.95},
            {"datetime": "2024-10-11", "instrument": "SZ000004", "middle": 0.85},
            {"datetime": "2024-10-11", "instrument": "SZ000001", "middle": 0.75},
        ]
    ).to_csv(middle, index=False)
    forbidden = tmp_path / "forbidden.csv"
    pd.DataFrame([{"instrument": "SZ000002"}]).to_csv(forbidden, index=False)
    windows = tmp_path / "windows.csv"
    pd.DataFrame(
        [
            {
                "trade_date": "2024-10-10",
                "instrument": "SZ000001",
                "open_exec": 10.0,
                "close_exec": 10.1,
                "mark_close": 10.1,
            }
        ]
    ).to_csv(windows, index=False)

    output = tmp_path / "pool.txt"
    missing = tmp_path / "missing.txt"
    report_path = tmp_path / "report.json"
    report = build_pool(
        middle,
        output,
        missing,
        report_path,
        "2024-10-10",
        "2024-10-11",
        2,
        forbidden,
        windows,
    )

    assert set(output.read_text(encoding="utf-8").splitlines()) == {
        "sz000001",
        "sz000003",
        "sz000004",
    }
    assert set(missing.read_text(encoding="utf-8").splitlines()) == {
        "sz000003",
        "sz000004",
    }
    assert report["forbidden_prediction_rows_removed"] == 1
    assert report["union_instruments"] == 3
    assert report["union_missing_windows"] == 2
    assert json.loads(report_path.read_text(encoding="utf-8"))["deployment_allowed"] is False
