from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from helki_quant.research.audit_canonical_market_data import build_report


def write_inputs(tmp_path: Path, minute_symbols: list[str]) -> dict[str, Path]:
    manifest = {
        "frequencies": {
            "daily": {
                "requested_symbols": 3,
                "written_symbols": 3,
                "rqdata_symbols": 2,
            },
            "minute": {
                "source_precedence": ["rqdata_primary"],
                "files": [
                    {"instrument": symbol, "rows": 480}
                    for symbol in minute_symbols
                ],
            },
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pit_path = tmp_path / "pit.csv"
    pd.DataFrame(
        {
            "date": ["2026-06-08", "2026-06-08", "2026-06-09", "2026-06-09"],
            "instrument": ["sz000001", "sh600000", "sz000001", "sh600000"],
            "is_st": [False] * 4,
            "is_suspended": [False] * 4,
        }
    ).to_csv(pit_path, index=False)
    instruments_path = tmp_path / "instruments.csv"
    pd.DataFrame({"instrument": ["sz000001", "sh600000"]}).to_csv(
        instruments_path, index=False
    )
    targets_path = tmp_path / "targets.txt"
    targets_path.write_text("sz000001\nsh600000\nsz300344\n", encoding="utf-8")
    return {
        "manifest_path": manifest_path,
        "pit_state_path": pit_path,
        "active_instruments_path": instruments_path,
        "target_symbols_path": targets_path,
    }


def test_inactive_target_without_minute_rows_does_not_fail_integrity(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path, ["sz000001", "sh600000"])

    report = build_report(
        **paths,
        cutoff=pd.Timestamp("2026-06-05"),
        min_sessions=3,
        expected_bars_per_session=240,
    )

    assert report["data_integrity_passed"] is True
    assert report["passed"] is False
    assert report["holdout"]["remaining_sessions"] == 1
    assert report["minute"]["inactive_targets_without_rows"] == ["sz300344"]
    assert report["return_metrics_evaluated"] is False


def test_missing_active_target_fails_integrity(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path, ["sz000001"])

    report = build_report(
        **paths,
        cutoff=pd.Timestamp("2026-06-05"),
        min_sessions=2,
        expected_bars_per_session=240,
    )

    assert report["data_integrity_passed"] is False
    assert report["minute"]["missing_active_targets"] == ["sh600000"]
