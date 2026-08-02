from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

from validate_frozen_strategy_promotion import (  # noqa: E402
    build_canonical_binding,
    build_evidence,
    load_json,
    sha256_file,
    validate,
    verify_frozen_contract,
)


PROFILE_PARAMETERS = {
    "champion": {
        "top_k": 150,
        "rebalance_every": 20,
        "risk_budget": 0.6,
        "industry_cap": 0.3,
        "buffer_multiple": 2,
        "allocation_mode": "fixed_topk",
        "outer_risk_threshold": 0.5,
        "outer_risk_floor": 0.3,
        "pause_buys_on_sell_reject": True,
        "sell_first": True,
        "cost_name": "stress",
    },
    "capital": {
        "top_k": 150,
        "rebalance_every": 30,
        "risk_budget": 0.6,
        "industry_cap": 0.3,
        "buffer_multiple": 4,
        "allocation_mode": "capital_aware",
        "outer_risk_threshold": 0.5,
        "outer_risk_floor": 0.2,
        "pause_buys_on_sell_reject": True,
        "sell_first": True,
        "cost_name": "stress",
    },
    "alpha": {
        "top_k": 150,
        "rebalance_every": 30,
        "risk_budget": 0.6,
        "industry_cap": 0.3,
        "buffer_multiple": 4,
        "allocation_mode": "capital_aware",
        "outer_risk_threshold": 0.5,
        "outer_risk_floor": 0.2,
        "pause_buys_on_sell_reject": True,
        "sell_first": True,
        "cost_name": "stress",
    },
}


def test_repository_frozen_contract_is_self_contained() -> None:
    contract_path = ROOT / "configs" / "frozen_strategy_promotion_20260731.json"
    contract = load_json(contract_path)

    verified = verify_frozen_contract(contract)

    assert set(verified) == {
        "champion_outer_direct_stable",
        "challenger_capital_aware",
        "challenger_capital_aware_alpha_health",
        "challenger_capital_aware_alpha_health.generator",
    }
    public_payloads = [contract_path]
    public_payloads.extend(
        ROOT / profile["frozen_artifact"]["path"]
        for profile in contract["profiles"]
    )
    for path in public_payloads:
        text = path.read_text(encoding="utf-8")
        assert ":\\" not in text
        assert "1f822ada-4ce6-11f1-a506-00163e022aa6" not in text


def test_incomplete_canonical_window_fails_before_return_evaluation(
    tmp_path: Path,
) -> None:
    dates = pd.bdate_range("2026-06-08", periods=39)
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "passed": False,
                "data_integrity_passed": True,
                "promotion_window_ready": False,
                "profile_frozen": True,
                "return_metrics_evaluated": False,
                "holdout": {
                    "sessions": 39,
                    "required_sessions": 60,
                    "remaining_sessions": 21,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="canonical readiness has not passed.*promotion_window_ready",
    ):
        build_canonical_binding(readiness, dates)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _make_contract(tmp_path: Path) -> tuple[Path, Path]:
    frozen = {
        name: _write(tmp_path / f"{name}_frozen.json", json.dumps({"id": name}))
        for name in PROFILE_PARAMETERS
    }
    generator = _write(tmp_path / "alpha_generator.py", "POLICY = 'frozen'\n")
    policy = {
        "label_horizon_trading_days": 5,
        "rolling_ic_observations": 20,
        "min_observations": 10,
        "min_cross_section": 100,
        "health_threshold": 0.0,
        "trigger_value": 1.0,
        "causal_filter": "available_date <= decision_date",
    }
    profiles = []
    for profile_id, parameters in PROFILE_PARAMETERS.items():
        overlay: dict[str, object] = {"kind": "broad_loss5"}
        if profile_id == "alpha":
            overlay = {
                "kind": "broad_loss5_or_alpha_health",
                "generator": _artifact(generator),
                "policy": policy,
            }
        profiles.append(
            {
                "id": profile_id,
                "role": "champion" if profile_id == "champion" else "challenger",
                "frozen_artifact": _artifact(frozen[profile_id]),
                "parameters": parameters,
                "overlay": overlay,
            }
        )
    contract = {
        "schema_version": 1,
        "training_data_end": "2026-01-01",
        "profiles": profiles,
        "holdout_rules": {
            "min_sessions": 60,
            "min_total_return": 0.0,
            "min_sharpe": 0.5,
            "max_drawdown": 0.08,
            "max_daily_turnover": 0.8,
            "min_cash_ratio": 0.2,
            "max_gross_exposure": 0.75,
            "min_rebalance_count": 2,
            "min_trades": 50,
            "min_worst_month_return": -0.05,
            "min_positive_month_ratio": 0.5,
            "max_negative_cash_events": 0,
            "max_lot_violations": 0,
            "max_nav_mismatch": 1e-6,
        },
        "promotion_rules": {
            "min_return_delta": 0.0,
            "max_drawdown_increase": 0.005,
            "max_turnover_ratio": 1.5,
            "min_worst_month_delta": -0.005,
            "min_utility_delta": 0.005,
            "utility_weights": {
                "return": 1.0,
                "sharpe": 0.05,
                "drawdown": 2.0,
                "turnover": 0.01,
            },
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path, generator


def _make_log(
    tmp_path: Path,
    *,
    profile_id: str,
    dates: pd.DatetimeIndex,
    drift: float,
    outer_file: Path | None = None,
) -> tuple[Path, dict[str, Path]]:
    log_dir = tmp_path / f"log_{profile_id}"
    log_dir.mkdir()
    source_files = {
        "middle_prediction": _write(
            tmp_path / f"{profile_id}_middle.csv", "datetime,instrument,middle\n"
        ),
        "outer_prediction": outer_file
        or _write(tmp_path / f"{profile_id}_outer.csv", "datetime,instrument,outer\n"),
        "minute_windows": _write(
            tmp_path / f"{profile_id}_minute.csv", "trade_date,instrument,open\n"
        ),
        "group_metadata": _write(
            tmp_path / f"{profile_id}_group.csv", "instrument,industry\n"
        ),
        "forbidden_symbols": _write(
            tmp_path / f"{profile_id}_forbidden.csv", "instrument\n"
        ),
    }
    source_provenance = {
        name: _artifact(path) for name, path in source_files.items()
    }
    oscillation = 0.0015 * np.sin(np.arange(len(dates)) * 1.7)
    returns = drift + oscillation
    nav = 1_000_000.0 * np.cumprod(1.0 + returns)
    daily = pd.DataFrame(
        {
            "trade_date": dates,
            "is_rebalance": [1 if index in {0, 30, 60} else 0 for index in range(len(dates))],
            "cash": nav * 0.50,
            "market_value": nav * 0.50,
            "nav": nav,
            "gross_exposure": 0.50,
            "day_turnover": 0.01,
            "cum_turnover": np.arange(1, len(dates) + 1) * 0.01,
            "day_trades": 2,
            "cum_trades": np.arange(1, len(dates) + 1) * 2,
            "holdings_count": 80,
            "target_count": 80,
            "mapped_count": 80,
            "max_group_fraction": 0.25,
            "risk_budget": 0.6,
            "outer_risk_probability": 0.2,
        }
    )
    daily.to_csv(log_dir / "daily_account.csv", index=False)
    parameters = PROFILE_PARAMETERS[profile_id]
    audit = {
        "status": "production_log_export_research_only",
        "profile": {
            "name": profile_id,
            "top_k": parameters["top_k"],
            "rebalance_every": parameters["rebalance_every"],
            "risk_budget": parameters["risk_budget"],
            "industry_cap": parameters["industry_cap"],
        },
        "config": {
            "initial_cash": 1_000_000.0,
            "buffer_multiple": parameters["buffer_multiple"],
        },
        "cost": {"name": "stress"},
        "days": len(dates),
        "allocation_mode": parameters["allocation_mode"],
        "outer_risk_threshold": parameters["outer_risk_threshold"],
        "outer_risk_floor": parameters["outer_risk_floor"],
        "pause_buys_on_sell_reject": parameters["pause_buys_on_sell_reject"],
        "sell_first": parameters["sell_first"],
        "negative_cash_events": 0,
        "lot_violations": 0,
        "nav_mismatch_max": 0.0,
        "source_provenance": source_provenance,
    }
    (log_dir / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return log_dir, source_files


def _make_alpha_manifest(
    tmp_path: Path,
    *,
    generator: Path,
    combined_outer: Path,
    middle: Path,
    forbidden: Path,
) -> Path:
    broad = _write(tmp_path / "alpha_broad.csv", "datetime,instrument,outer\n")
    daily_health = _write(tmp_path / "alpha_daily_health.csv", "datetime,health\n")
    policy = {
        "label_horizon_trading_days": 5,
        "rolling_ic_observations": 20,
        "min_observations": 10,
        "min_cross_section": 100,
        "health_threshold": 0.0,
        "trigger_value": 1.0,
        "causal_filter": "available_date <= decision_date",
    }
    manifest = {
        "schema_version": 2,
        "status": "causal_middle_alpha_health_outer_research_only",
        "policy": policy,
        "source_provenance": {
            "generator": _artifact(generator),
            "middle_prediction": _artifact(middle),
            "broad_outer_prediction": _artifact(broad),
            "forbidden_symbols": _artifact(forbidden),
            "combined_outer": _artifact(combined_outer),
            "daily_health": _artifact(daily_health),
        },
    }
    path = tmp_path / "alpha_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _make_canonical_readiness(
    tmp_path: Path,
    dates: pd.DatetimeIndex,
) -> tuple[Path, Path]:
    manifest = _write(tmp_path / "canonical_manifest.json", "{\"version\": 1}\n")
    calendar_hash = hashlib.sha256(
        "\n".join(day.strftime("%Y-%m-%d") for day in dates).encode("ascii")
    ).hexdigest().upper()
    readiness = {
        "status": "canonical_market_data_readiness",
        "passed": True,
        "data_integrity_passed": True,
        "promotion_window_ready": True,
        "profile_frozen": True,
        "return_metrics_evaluated": False,
        "canonical_manifest": _artifact(manifest),
        "holdout": {
            "first_session": str(dates[0].date()),
            "last_session": str(dates[-1].date()),
            "sessions": len(dates),
            "required_sessions": 60,
            "remaining_sessions": 0,
            "calendar_sha256": calendar_hash,
        },
    }
    path = tmp_path / "canonical_readiness.json"
    path.write_text(json.dumps(readiness), encoding="utf-8")
    return path, manifest


def _complete_fixture(tmp_path: Path, start: str = "2026-01-02"):
    contract, generator = _make_contract(tmp_path)
    dates = pd.bdate_range(start, periods=65)
    champion, _ = _make_log(
        tmp_path,
        profile_id="champion",
        dates=dates,
        drift=0.0005,
    )
    capital, _ = _make_log(
        tmp_path,
        profile_id="capital",
        dates=dates,
        drift=0.0007,
    )
    alpha_outer = _write(
        tmp_path / "alpha_combined_outer.csv",
        "datetime,instrument,outer\n",
    )
    alpha, alpha_sources = _make_log(
        tmp_path,
        profile_id="alpha",
        dates=dates,
        drift=0.0010,
        outer_file=alpha_outer,
    )
    alpha_manifest = _make_alpha_manifest(
        tmp_path,
        generator=generator,
        combined_outer=alpha_outer,
        middle=alpha_sources["middle_prediction"],
        forbidden=alpha_sources["forbidden_symbols"],
    )
    readiness, manifest = _make_canonical_readiness(tmp_path, dates)
    return (
        contract,
        {"champion": champion, "capital": capital, "alpha": alpha},
        {"alpha": alpha_manifest},
        readiness,
        manifest,
    )


def test_complete_untouched_evidence_promotes_best_challenger(tmp_path):
    contract, logs, manifests, readiness, _ = _complete_fixture(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence = build_evidence(contract, logs, manifests, evidence_path, readiness)
    assert evidence["holdout"]["sessions"] == 65
    assert evidence["schema_version"] == 2

    report = validate(contract, evidence_path, tmp_path / "promotion.json")
    assert report["passed"] is True
    assert report["paper_candidate_promotion_allowed"] is True
    assert report["selected_profile_id"] == "alpha"
    assert "alpha" in report["challenger_promotions"]
    assert report["real_money_deployment_allowed"] is False


def test_holdout_must_start_after_frozen_training_boundary(tmp_path):
    contract, logs, manifests, readiness, _ = _complete_fixture(
        tmp_path, start="2025-12-01"
    )
    with pytest.raises(ValueError, match="overlaps frozen data boundary"):
        build_evidence(
            contract, logs, manifests, tmp_path / "evidence.json", readiness
        )


def test_replay_source_tampering_is_rejected(tmp_path):
    contract, logs, manifests, readiness, _ = _complete_fixture(tmp_path)
    audit = json.loads((logs["capital"] / "audit.json").read_text(encoding="utf-8"))
    middle_path = Path(audit["source_provenance"]["middle_prediction"]["path"])
    middle_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance hash mismatch"):
        build_evidence(
            contract, logs, manifests, tmp_path / "evidence.json", readiness
        )


def test_alpha_health_policy_change_is_rejected(tmp_path):
    contract, logs, manifests, readiness, _ = _complete_fixture(tmp_path)
    manifest = json.loads(manifests["alpha"].read_text(encoding="utf-8"))
    manifest["policy"]["health_threshold"] = -0.1
    manifests["alpha"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="alpha-health policy mismatch"):
        build_evidence(
            contract, logs, manifests, tmp_path / "evidence.json", readiness
        )


def test_contract_change_after_evaluation_invalidates_evidence(tmp_path):
    contract, logs, manifests, readiness, _ = _complete_fixture(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    build_evidence(contract, logs, manifests, evidence_path, readiness)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["description"] = "changed after evaluation"
    contract.write_text(json.dumps(payload), encoding="utf-8")

    report = validate(contract, evidence_path, tmp_path / "promotion.json")
    assert report["passed"] is False
    assert any(
        item["name"] == "evidence.contract_hash"
        for item in report["failed_checks"]
    )


def test_replay_calendar_must_match_canonical_readiness(tmp_path):
    contract, logs, manifests, readiness, _ = _complete_fixture(tmp_path)
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    payload["holdout"]["calendar_sha256"] = "0" * 64
    readiness.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match canonical readiness"):
        build_evidence(
            contract, logs, manifests, tmp_path / "evidence.json", readiness
        )


def test_canonical_manifest_tampering_invalidates_promotion(tmp_path):
    contract, logs, manifests, readiness, canonical_manifest = _complete_fixture(
        tmp_path
    )
    evidence_path = tmp_path / "evidence.json"
    build_evidence(contract, logs, manifests, evidence_path, readiness)
    canonical_manifest.write_text("tampered\n", encoding="utf-8")

    report = validate(contract, evidence_path, tmp_path / "promotion.json")

    assert report["passed"] is False
    assert "evidence.canonical_binding" in {
        item["name"] for item in report["failed_checks"]
    }


def test_legacy_unbound_evidence_schema_is_rejected(tmp_path):
    contract, logs, manifests, readiness, _ = _complete_fixture(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    build_evidence(contract, logs, manifests, evidence_path, readiness)
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload.pop("canonical_market_data")
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate(contract, evidence_path, tmp_path / "promotion.json")

    failed = {item["name"] for item in report["failed_checks"]}
    assert "evidence.schema_version" in failed
    assert "evidence.canonical_binding" in failed


def test_missing_future_evidence_fails_closed(tmp_path):
    contract, _ = _make_contract(tmp_path)
    report = validate(contract, None, tmp_path / "pending.json")
    assert report["passed"] is False
    assert report["paper_candidate_promotion_allowed"] is False
    assert report["selected_profile_id"] is None
    assert any(item["name"] == "evidence.exists" for item in report["failed_checks"])
