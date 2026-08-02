from __future__ import annotations

import sys
from pathlib import Path

from helki_quant.research.refresh_canonical_market_data import build_commands


def test_refresh_commands_preserve_overlap_and_primary_only_minute_policy(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    targets = tmp_path / "targets.txt"
    version_root = tmp_path / "v20260731_rqaligned"
    readiness = tmp_path / "readiness.json"
    config = {
        "primary": {
            "metadata_root": str(tmp_path / "metadata"),
        }
    }

    commands = build_commands(
        config_path=config_path,
        config=config,
        end_date="2026-07-31",
        target_symbols=targets,
        cutoff="2026-06-05",
        overlap_start="2026-05-01",
        history_start="1990-01-01",
        version_root=version_root,
        readiness_output=readiness,
        min_sessions=60,
        expected_bars_per_session=240,
    )

    by_name = {name: command for name, command in commands}
    assert list(by_name) == [
        "historical_universe_sync",
        "daily_overlap_sync",
        "pit_state_sync",
        "minute_holdout_sync",
        "daily_materialization",
        "minute_materialization",
        "canonical_readiness_audit",
    ]
    assert by_name["daily_overlap_sync"][:3] == [
        sys.executable,
        "-m",
        "helki_quant.research.sync_rqdata_market_data",
    ]
    assert by_name["daily_overlap_sync"][
        by_name["daily_overlap_sync"].index("--start-date") + 1
    ] == "2026-05-01"
    assert "--symbols-file" in by_name["daily_overlap_sync"]
    daily_symbols = by_name["daily_overlap_sync"][
        by_name["daily_overlap_sync"].index("--symbols-file") + 1
    ]
    assert daily_symbols.endswith("pit_universe_20260606_20260731.txt")
    assert by_name["pit_state_sync"][
        by_name["pit_state_sync"].index("--start-date") + 1
    ] == "2026-06-06"
    assert "--symbols-file" in by_name["pit_state_sync"]
    pit_symbols = by_name["pit_state_sync"][
        by_name["pit_state_sync"].index("--symbols-file") + 1
    ]
    assert pit_symbols.endswith("pit_universe_20260606_20260731.txt")
    assert by_name["minute_holdout_sync"][
        by_name["minute_holdout_sync"].index("--start-date") + 1
    ] == "2026-06-06"
    assert "--primary-only" in by_name["minute_materialization"]
    assert "--min-sessions" in by_name["canonical_readiness_audit"]
    assert "--pit-instruments" in by_name["canonical_readiness_audit"]
    audit = by_name["canonical_readiness_audit"]
    active_path = audit[audit.index("--active-instruments") + 1]
    pit_path = audit[audit.index("--pit-instruments") + 1]
    assert active_path == pit_path
