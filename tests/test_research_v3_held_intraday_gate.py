from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from validate_held_intraday_t0_gate import (  # noqa: E402
    select_replay_setting,
    validate,
)


def _metrics(threshold: float, *, pnl: float = 4000.0) -> dict:
    return {
        "threshold": threshold,
        "selection_mode": "daily_top_n",
        "daily_top_n": 2,
        "trade_fraction": 1.0,
        "sizing_mode": "one_lot",
        "trade_direction": "sell_first",
        "round_trips": 90,
        "cum_pnl": pnl,
        "incremental_return": 0.004,
        "symbols_traded": 30,
        "active_months": 6,
        "losing_months": 1,
        "profit_factor": 1.5 if pnl > 0 else 0.3,
        "top_symbol_positive_pnl_share": 0.2,
        "top3_positive_pnl_share": 0.4,
        "max_daily_turnover": 0.03,
        "max_overlay_drawdown": 0.02,
        "incremental_max_drawdown": 0.001,
        "fold_count": 6,
        "profitable_folds": 6,
        "worst_fold_return": 0.0001,
        "folds": [{"round_trips": 10} for _ in range(6)],
    }


def _write_replay(path: Path, results: list[dict], best: int = 0) -> None:
    path.write_text(
        json.dumps({"best": results[best], "results": results}),
        encoding="utf-8",
    )


def _validate(oof: Path, forward: Path, output: Path) -> dict:
    return validate(
        oof,
        output,
        min_cum_pnl=3000.0,
        min_incremental_return=0.003,
        min_round_trips=80,
        min_symbols_traded=20,
        min_active_months=4,
        min_profit_factor=1.4,
        max_top_symbol_positive_share=0.3,
        max_top3_positive_share=0.55,
        max_daily_turnover=0.1,
        max_overlay_drawdown=0.04,
        max_incremental_drawdown=0.003,
        min_fold_count=6,
        min_profitable_folds=6,
        min_worst_fold_return=0.0,
        min_round_trips_per_fold=3,
        gate_stage="research_oof",
        expected_threshold=0.001,
        expected_daily_top_n=2,
        expected_trade_fraction=1.0,
        expected_sizing_mode="one_lot",
        expected_trade_direction="sell_first",
        forward_replay_path=forward,
    )


def test_select_replay_setting_uses_frozen_profile_not_best() -> None:
    zero = _metrics(0.0)
    frozen = _metrics(0.001)
    selected = select_replay_setting(
        {"best": zero, "results": [zero, frozen]},
        expected_threshold=0.001,
        expected_daily_top_n=2,
        expected_trade_fraction=1.0,
        expected_sizing_mode="one_lot",
        expected_trade_direction="sell_first",
    )
    assert selected is frozen


def test_pair_gate_requires_forward_economic_evidence(tmp_path: Path) -> None:
    oof = tmp_path / "oof.json"
    forward = tmp_path / "forward.json"
    output = tmp_path / "gate.json"
    _write_replay(oof, [_metrics(0.0), _metrics(0.001)], best=0)
    _write_replay(forward, [_metrics(0.001, pnl=-100.0)])

    report = _validate(oof, forward, output)

    assert report["selected_setting"]["threshold"] == pytest.approx(0.001)
    assert not report["passed"]
    assert "forward_cum_pnl" in {row["name"] for row in report["failed_checks"]}
    assert report["decision"] == "keep_held_intraday_t0_research_only"


def test_pair_gate_passes_only_matching_positive_forward(tmp_path: Path) -> None:
    oof = tmp_path / "oof.json"
    forward = tmp_path / "forward.json"
    output = tmp_path / "gate.json"
    _write_replay(oof, [_metrics(0.0), _metrics(0.001)], best=0)
    _write_replay(forward, [_metrics(0.001)])

    report = _validate(oof, forward, output)

    assert report["passed"]
    assert report["selected_forward_metrics"]["threshold"] == pytest.approx(0.001)


def test_select_replay_setting_fails_closed_when_profile_is_missing() -> None:
    with pytest.raises(ValueError, match="found 0"):
        select_replay_setting(
            {"results": [_metrics(0.0)]},
            expected_threshold=0.001,
        )


def test_select_trigger_replay_uses_nested_frozen_profile() -> None:
    trigger = _metrics(0.005)
    trigger.pop("threshold")
    trigger.pop("daily_top_n")
    trigger.pop("trade_direction")
    trigger["profile"] = {
        "score_threshold": 0.005,
        "daily_top_n": 2,
        "direction": "sell_first",
    }

    selected = select_replay_setting(
        trigger,
        expected_threshold=0.005,
        expected_daily_top_n=2,
        expected_trade_direction="sell_first",
    )

    assert selected is trigger


def test_select_combination_replay_uses_top_level_direction() -> None:
    combined = _metrics(0.005)
    combined.pop("threshold")
    combined.pop("daily_top_n")

    selected = select_replay_setting(
        combined,
        expected_trade_direction="sell_first",
    )

    assert selected is combined


def test_nested_trigger_pair_is_validated_without_flat_profile_fields(
    tmp_path: Path,
) -> None:
    oof = tmp_path / "oof.json"
    forward = tmp_path / "forward.json"
    output = tmp_path / "gate.json"
    trigger = _metrics(0.005)
    for key in ("threshold", "daily_top_n", "trade_fraction", "sizing_mode", "trade_direction"):
        trigger.pop(key)
    trigger["profile"] = {
        "score_threshold": 0.005,
        "daily_top_n": 2,
        "direction": "sell_first",
    }
    oof.write_text(json.dumps(trigger), encoding="utf-8")
    forward.write_text(json.dumps(trigger), encoding="utf-8")

    report = validate(
        oof,
        output,
        min_cum_pnl=3000.0,
        min_incremental_return=0.003,
        min_round_trips=80,
        min_symbols_traded=20,
        min_active_months=4,
        min_profit_factor=1.4,
        max_top_symbol_positive_share=0.3,
        max_top3_positive_share=0.55,
        max_daily_turnover=0.1,
        max_overlay_drawdown=0.04,
        max_incremental_drawdown=0.003,
        min_fold_count=6,
        min_profitable_folds=6,
        min_worst_fold_return=0.0,
        min_round_trips_per_fold=3,
        gate_stage="research_oof",
        expected_threshold=0.005,
        expected_daily_top_n=2,
        expected_trade_direction="sell_first",
        forward_replay_path=forward,
    )

    assert report["selected_setting"]["threshold"] == pytest.approx(0.005)
    assert report["selected_setting"]["trade_direction"] == "sell_first"
    assert "forward_threshold_matches_oof" in {row["name"] for row in report["checks"]}
