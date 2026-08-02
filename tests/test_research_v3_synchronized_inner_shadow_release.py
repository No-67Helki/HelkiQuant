from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from helki_quant.research import (  # noqa: E402
    build_synchronized_inner_shadow_release as sync_release,
)


def test_synchronized_release_cli_starts_from_package_layout() -> None:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(SRC) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "helki_quant.research.build_synchronized_inner_shadow_release",
            "--help",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--outer-package" in result.stdout
    assert "--model-manifest" in result.stdout


def _outer_package(
    tmp_path: Path,
    *,
    historical_smoke: bool,
    ready: bool,
) -> Path:
    package = tmp_path / "outer"
    package.mkdir()
    instruments = [f"SZ{index:06d}" for index in range(1, 21)]
    pd.DataFrame(
        {
            "trade_date": "2026-06-08",
            "signal_date": "2026-06-05",
            "instrument": instruments,
            "symbol": [f"SZSE.{value[-6:]}" for value in instruments],
            "target_shares": 100,
        }
    ).to_csv(package / "gm_c_baseline_targets.csv", index=False)
    pd.DataFrame({"instrument": ["SZ300001"]}).to_csv(
        package / "gm_c_forbidden_symbols.csv",
        index=False,
    )
    (package / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json").write_text(
        json.dumps({"paper_only": True}),
        encoding="utf-8",
    )
    (package / "RELEASE_PROVENANCE.json").write_text(
        json.dumps(
            {
                "signal_date": "2026-06-05",
                "trade_date": "2026-06-08",
                "historical_smoke": historical_smoke,
                "inner_t0_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    (package / "PAPER_READY_20260608.json").write_text(
        json.dumps({"passed": ready, "paper_orders_allowed": ready}),
        encoding="utf-8",
    )
    (package / "PREFLIGHT_20260608.json").write_text(
        json.dumps(
            {
                "passed": ready,
                "errors": (
                    []
                    if ready
                    else ["historical engineering smoke cannot authorize PAPER orders"]
                ),
            }
        ),
        encoding="utf-8",
    )
    return package


def _stub_inner_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build(
        model_manifest,
        target_path,
        forbidden_path,
        output_dir,
        output_manifest_path,
        *,
        as_of_date,
        max_signal_age_days,
    ):
        output_dir.mkdir(parents=True)
        output_manifest_path.write_text("{}", encoding="utf-8")
        return {"status": "stubbed_inner_build"}

    monkeypatch.setattr(sync_release, "build_inner_candidate", fake_build)
    monkeypatch.setattr(
        sync_release,
        "run_preflight",
        lambda *args, **kwargs: {
            "passed": True,
            "errors": [],
            "paper_orders_allowed": False,
            "main_py_integration_allowed": False,
            "deployment_allowed": False,
        },
    )


def test_historical_sync_builds_versioned_no_order_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _outer_package(tmp_path, historical_smoke=True, ready=False)
    model_manifest = tmp_path / "models.json"
    model_manifest.write_text("{}", encoding="utf-8")
    _stub_inner_build(monkeypatch)

    report = sync_release.build_synchronized_shadow(
        outer_package=package,
        output_dir=tmp_path / "inner",
        as_of_date=pd.Timestamp("2026-06-08"),
        model_manifest=model_manifest,
        historical_smoke=True,
    )

    assert report["inner_preflight_passed"] is True
    assert report["actual_submission_api_present"] is False
    assert report["paper_orders_allowed"] is False
    assert report["runnable_no_order_shadow"] is False
    assert report["next_action"] == "engineering_chain_verified_only"
    assert (tmp_path / "inner" / "SYNC_PROVENANCE.json").exists()


def test_real_sync_rejects_outer_package_that_is_not_paper_ready(
    tmp_path: Path,
) -> None:
    package = _outer_package(tmp_path, historical_smoke=False, ready=False)
    model_manifest = tmp_path / "models.json"
    model_manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="not PAPER_READY"):
        sync_release.build_synchronized_shadow(
            outer_package=package,
            output_dir=tmp_path / "inner",
            as_of_date=pd.Timestamp("2026-06-08"),
            model_manifest=model_manifest,
            historical_smoke=False,
        )
