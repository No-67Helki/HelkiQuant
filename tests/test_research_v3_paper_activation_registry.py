from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

from paper_activation_registry import (  # noqa: E402
    EVENT_FINALIZED,
    EVENT_READY,
    EVENT_STARTED,
    append_activation_event,
    build_activation_identity,
    read_activation_registry,
    resolve_latest_finalized_target,
    sha256_file,
)


ACCOUNT_ID = "paper-account"


def _package(root: Path, *, ready: bool = True) -> tuple[Path, Path, Path]:
    package = root / "package"
    package.mkdir()
    target = package / "gm_c_baseline_targets.csv"
    forbidden = package / "gm_c_forbidden_symbols.csv"
    pd.DataFrame(
        {
            "trade_date": ["2026-06-04"],
            "signal_date": ["2026-06-03"],
            "symbol": ["SZSE.000001"],
            "instrument": ["SZ000001"],
            "target_weight": [0.01],
            "target_shares": [1000],
        }
    ).to_csv(target, index=False)
    pd.DataFrame(
        {
            "instrument": ["SZ000002"],
            "gm_symbol": ["SZSE.000002"],
        }
    ).to_csv(forbidden, index=False)
    manifest = {
        "paper_only": True,
        "paper_account_id": ACCOUNT_ID,
        "target_provenance": {
            "signal_date": "2026-06-03",
            "trade_date": "2026-06-04",
            "sha256": {
                target.name: sha256_file(target),
                forbidden.name: sha256_file(forbidden),
            },
        },
    }
    (package / "PAPER_SIMULATION_CANDIDATE_MANIFEST.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    ready_payload = {
        "status": "paper_candidate_ready" if ready else "paper_candidate_blocked",
        "passed": ready,
        "paper_orders_allowed": ready,
        "paper_account_id": ACCOUNT_ID,
        "trade_date": "2026-06-04",
        "signal_date": "2026-06-03",
    }
    (package / "PAPER_READY_20260604.json").write_text(
        json.dumps(ready_payload),
        encoding="utf-8",
    )
    return package, target, forbidden


def _identity(root: Path, *, ready: bool = True) -> dict:
    package, target, forbidden = _package(root, ready=ready)
    return build_activation_identity(
        package_dir=package,
        target_path=target,
        forbidden_path=forbidden,
        account_id=ACCOUNT_ID,
        run_id="paper_run_001",
        strategy_id="outer-middle-paper",
        run_mode="LIVE",
        trading_env="PAPER",
    )


def _complete_registry(root: Path) -> tuple[Path, dict]:
    identity = _identity(root)
    registry = root / "PAPER_ACTIVATION_REGISTRY.jsonl"
    append_activation_event(
        registry,
        event=EVENT_STARTED,
        identity=identity,
        timestamp="2026-06-04 09:20:00",
    )
    append_activation_event(
        registry,
        event=EVENT_READY,
        identity=identity,
        timestamp="2026-06-04 09:21:00",
    )
    append_activation_event(
        registry,
        event=EVENT_FINALIZED,
        identity=identity,
        timestamp="2026-06-04 15:05:00",
        metrics={"submitted_orders": 0},
    )
    return registry, identity


def test_activation_registry_resolves_only_finalized_target(tmp_path: Path) -> None:
    registry, identity = _complete_registry(tmp_path)

    records, errors = read_activation_registry(registry)
    assert errors == []
    assert [record["event"] for record in records] == [
        EVENT_STARTED,
        EVENT_READY,
        EVENT_FINALIZED,
    ]
    resolved = resolve_latest_finalized_target(
        registry,
        expected_account_id=ACCOUNT_ID,
        before_trade_date="2026-06-08",
    )
    assert resolved["run_id"] == "paper_run_001"
    assert resolved["activation_id"] == identity["activation_id"]
    assert resolved["target_sha256"] == identity["target_sha256"]


def test_activation_registry_rejects_out_of_order_ready(tmp_path: Path) -> None:
    identity = _identity(tmp_path)
    with pytest.raises(ValueError, match="requires PAPER_RUN_STARTED"):
        append_activation_event(
            tmp_path / "registry.jsonl",
            event=EVENT_READY,
            identity=identity,
            timestamp="2026-06-04 09:21:00",
        )


def test_activation_registry_rejects_tampered_chain(tmp_path: Path) -> None:
    registry, _ = _complete_registry(tmp_path)
    text = registry.read_text(encoding="utf-8")
    registry.write_text(text.replace("paper_run_001", "paper_run_999", 1), encoding="utf-8")

    _, errors = read_activation_registry(registry)
    assert any("record hash mismatch" in error for error in errors)
    with pytest.raises(ValueError, match="integrity failed"):
        resolve_latest_finalized_target(
            registry,
            expected_account_id=ACCOUNT_ID,
            before_trade_date="2026-06-08",
        )


def test_activation_resolver_rejects_tampered_target(tmp_path: Path) -> None:
    registry, identity = _complete_registry(tmp_path)
    target = Path(identity["target_path"])
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target hash mismatch"):
        resolve_latest_finalized_target(
            registry,
            expected_account_id=ACCOUNT_ID,
            before_trade_date="2026-06-08",
        )


def test_activation_identity_requires_positive_paper_ready(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not authorize"):
        _identity(tmp_path, ready=False)


def test_activation_resolver_is_account_bound(tmp_path: Path) -> None:
    registry, _ = _complete_registry(tmp_path)
    with pytest.raises(ValueError, match="no finalized prior PAPER activation"):
        resolve_latest_finalized_target(
            registry,
            expected_account_id="another-account",
            before_trade_date="2026-06-08",
        )


def test_activation_registry_rejects_second_run_for_same_session(
    tmp_path: Path,
) -> None:
    registry, identity = _complete_registry(tmp_path)
    duplicate = {
        **identity,
        "activation_id": "duplicate-activation",
        "run_id": "paper_run_002",
    }

    with pytest.raises(ValueError, match="another PAPER run already owns"):
        append_activation_event(
            registry,
            event=EVENT_STARTED,
            identity=duplicate,
            timestamp="2026-06-04 10:00:00",
        )


def test_concurrent_registry_writes_preserve_every_hash_link(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.jsonl"

    def append_start(index: int) -> None:
        day = pd.Timestamp("2026-07-01") + pd.Timedelta(days=index)
        append_activation_event(
            registry,
            event=EVENT_STARTED,
            identity={
                "activation_schema_version": 1,
                "activation_id": f"activation-{index}",
                "run_id": f"run-{index}",
                "strategy_id": "outer-middle-paper",
                "account_id": ACCOUNT_ID,
                "run_mode": "LIVE",
                "trading_env": "PAPER",
                "signal_date": (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "trade_date": day.strftime("%Y-%m-%d"),
            },
            timestamp=day + pd.Timedelta(hours=9),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append_start, range(12)))

    records, errors = read_activation_registry(registry)
    assert errors == []
    assert len(records) == 12
    assert {record["run_id"] for record in records} == {
        f"run-{index}" for index in range(12)
    }
    assert not registry.with_suffix(registry.suffix + ".lock").exists()
    assert not registry.with_suffix(registry.suffix + ".pending").exists()
