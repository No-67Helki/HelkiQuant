from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path


RESEARCH_V3 = Path(__file__).resolve().parents[1] / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH_V3))

from validate_c_baseline_paper_gate import validate_target_manifest


def _write_manifest(tmp_path: Path, signal_date: str, trade_date: str) -> Path:
    target = tmp_path / "targets.csv"
    target.write_text("symbol,target_weight\nSZSE.000001,0.3\n", encoding="utf-8")
    outer = tmp_path / "pred_outer.csv"
    outer.write_text("datetime,instrument,pred_outer\n", encoding="utf-8")
    manifest = {
        "target": str(target),
        "signal_date": signal_date,
        "trade_date": trade_date,
        "symbols": 80,
        "effective_exposure_ratio": 0.99,
        "forbidden_order_hits": 0,
        "rebalance_every": 30,
        "buffer_multiple": 4,
        "allocation": {"mode": "capital_aware", "max_name_weight": 0.01},
        "allocation_gate": {"passed": True},
        "outer_overlay": {"required": True, "prediction": str(outer)},
        "execution_risk_controls": {"pause_buys_on_sell_reject": True},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _thresholds() -> dict:
    return {
        "required": True,
        "min_symbols": 50,
        "min_effective_exposure_ratio": 0.90,
        "max_name_weight": 0.03,
        "require_allocation_gate": True,
        "expected_allocation_mode": "capital_aware",
        "expected_rebalance_every": 30,
        "expected_buffer_multiple": 4,
        "require_pause_buys_on_sell_reject": True,
        "max_forbidden_order_hits": 0,
        "require_outer_prediction": True,
        "max_signal_age_calendar_days": 7,
        "max_future_signal_days": 0,
        "max_target_age_calendar_days": 1,
        "max_future_target_days": 7,
    }


def test_strict_target_gate_accepts_fresh_complete_manifest(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, "2026-07-10", "2026-07-13")

    checks, _ = validate_target_manifest(path, _thresholds(), date(2026, 7, 13))

    assert checks
    assert all(check["passed"] for check in checks)


def test_strict_target_gate_rejects_stale_signal_and_trade_date(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, "2026-06-05", "2026-06-12")

    checks, _ = validate_target_manifest(path, _thresholds(), date(2026, 7, 13))
    failed = {check["name"] for check in checks if not check["passed"]}

    assert "target_manifest.signal_date_freshness" in failed
    assert "target_manifest.trade_date_freshness" in failed


def test_strict_target_gate_rejects_low_budget_utilization(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, "2026-07-10", "2026-07-13")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["effective_exposure_ratio"] = 0.44
    manifest["allocation_gate"]["passed"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")

    checks, _ = validate_target_manifest(path, _thresholds(), date(2026, 7, 13))
    failed = {check["name"] for check in checks if not check["passed"]}

    assert "target_manifest.min_effective_exposure_ratio" in failed
    assert "target_manifest.allocation_gate" in failed
