from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_replay_cli_starts_from_package_layout() -> None:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "helki_quant.research.export_c_baseline_production_logs",
            "--help",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--profile-specs" in result.stdout
    assert "--outer-risk-threshold" in result.stdout
