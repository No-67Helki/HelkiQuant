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
                "files": [
                    {
                        "instrument": symbol,
                        "rqdata_rows": 2,
                        "fallback_rows": 2,
                        "scale_alignment_status": "aligned",
                    }
                    for symbol in ("sz000001", "sh600000")
                ],
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
    assert len(report["holdout"]["calendar_sha256"]) == 64
    assert len(report["canonical_manifest"]["sha256"]) == 64


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


def test_missing_daily_scale_alignment_evidence_fails_integrity(
    tmp_path: Path,
) -> None:
    paths = write_inputs(tmp_path, ["sz000001", "sh600000"])
    manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
    manifest["frequencies"]["daily"]["files"][0][
        "scale_alignment_status"
    ] = "missing_overlap"
    paths["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

    report = build_report(
        **paths,
        cutoff=pd.Timestamp("2026-06-05"),
        min_sessions=2,
        expected_bars_per_session=240,
    )

    assert report["data_integrity_passed"] is False
    assert report["daily"]["scale_alignment_failures"] == [
        {"instrument": "sz000001", "status": "missing_overlap"}
    ]


def test_pit_coverage_uses_listing_and_delisting_intervals(
    tmp_path: Path,
) -> None:
    paths = write_inputs(tmp_path, ["sz000001", "sh600000"])
    pd.DataFrame(
        {
            "instrument": ["sz000001", "sh600000"],
            "listed_date": ["1991-01-01", "2026-06-09"],
            "de_listed_date": ["2026-06-08", "0000-00-00"],
        }
    ).to_csv(paths["active_instruments_path"], index=False)
    pit = pd.read_csv(paths["pit_state_path"])
    pit = pit[
        ~(
            (pit["date"] == "2026-06-09")
            & (pit["instrument"] == "sz000001")
        )
    ]
    pit.to_csv(paths["pit_state_path"], index=False)

    report = build_report(
        **paths,
        cutoff=pd.Timestamp("2026-06-05"),
        min_sessions=2,
        expected_bars_per_session=240,
    )

    assert report["pit_market_state"]["passed"] is True
    assert report["pit_market_state"]["missing_symbol_dates"] == 0
    assert report["daily"]["active_symbols"] == 1


def test_missing_delisted_symbol_daily_primary_data_fails_integrity(
    tmp_path: Path,
) -> None:
    paths = write_inputs(tmp_path, ["sz000001", "sh600000"])
    historical = tmp_path / "historical_instruments.csv"
    pd.DataFrame(
        {
            "instrument": ["sz000001", "sh600000", "sz000004"],
            "listed_date": ["1991-01-01", "1999-11-10", "1991-01-01"],
            "de_listed_date": ["0000-00-00", "0000-00-00", "2026-06-08"],
        }
    ).to_csv(historical, index=False)
    pit = pd.read_csv(paths["pit_state_path"])
    pit = pd.concat(
        [
            pit,
            pd.DataFrame(
                {
                    "date": ["2026-06-08"],
                    "instrument": ["sz000004"],
                    "is_st": [True],
                    "is_suspended": [False],
                }
            ),
        ],
        ignore_index=True,
    )
    pit.to_csv(paths["pit_state_path"], index=False)

    report = build_report(
        **paths,
        pit_instruments_path=historical,
        cutoff=pd.Timestamp("2026-06-05"),
        min_sessions=2,
        expected_bars_per_session=240,
    )

    assert report["data_integrity_passed"] is False
    assert report["daily"]["missing_pit_primary_symbols"] == ["sz000004"]
