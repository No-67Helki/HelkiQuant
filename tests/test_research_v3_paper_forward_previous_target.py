from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1] / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

from export_paper_forward_gm_targets import (  # noqa: E402
    _export_scheduled_carry,
    load_previous_selection,
)


def test_load_previous_selection_normalizes_gm_symbols_and_positive_shares(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.csv"
    pd.DataFrame(
        {
            "symbol": ["SZSE.300001", "SHSE.688001", "SZSE.300002"],
            "target_shares": [100, 200, 0],
        }
    ).to_csv(target, index=False)
    assert load_previous_selection(target) == {"SZ300001", "SH688001"}


def test_load_previous_selection_rejects_missing_symbol_columns(tmp_path: Path) -> None:
    target = tmp_path / "bad.csv"
    pd.DataFrame({"target_shares": [100]}).to_csv(target, index=False)
    try:
        load_previous_selection(target)
    except ValueError as exc:
        assert "instrument or symbol" in str(exc)
    else:
        raise AssertionError("missing symbol columns must fail closed")


def test_scheduled_carry_keeps_shares_and_only_forces_forbidden_exits(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "previous.csv"
    pd.DataFrame(
        {
            "trade_date": ["2026-06-10", "2026-06-10"],
            "signal_date": ["2026-06-09", "2026-06-09"],
            "symbol": ["SZSE.300001", "SZSE.300002"],
            "instrument": ["SZ300001", "SZ300002"],
            "rank": [1, 2],
            "middle": [0.8, 0.7],
            "target_weight": [0.004, 0.004],
            "target_shares": [300, 200],
            "nominal_target_weight": [0.004, 0.004],
            "effective_weight_ref": [0.003, 0.002],
            "target_notional_ref": [3000.0, 2000.0],
            "group": ["A", "B"],
            "effective_risk_budget": [0.6, 0.6],
        }
    ).to_csv(previous, index=False)

    manifest = _export_scheduled_carry(
        prediction_path=tmp_path / "middle.csv",
        previous_target_path=previous,
        forbidden_path=tmp_path / "forbidden.csv",
        output_dir=tmp_path / "out",
        forbidden={"SZ300002"},
        signal_ts=pd.Timestamp("2026-06-11"),
        trade_date="2026-06-12",
        top_k=150,
        risk_budget=0.6,
        industry_cap=0.3,
        min_avg_amount=100_000_000.0,
        initial_cash=1_000_000.0,
        outer_prediction_path=tmp_path / "outer.csv",
        require_outer=True,
        outer_stats={"probability": 0.9, "rows": 1, "minimum": 0.9, "maximum": 0.9},
        outer_risk_threshold=0.5,
        outer_risk_floor=0.3,
        allocation_mode="fixed_topk",
        min_effective_exposure_ratio=0.0,
        max_name_weight=0.03,
        rebalance_every=20,
        buffer_multiple=2,
        pause_buys_on_sell_reject=False,
        removed_forbidden_prediction_rows=1,
        middle_last_rebalance_signal_date="2026-06-09",
        trading_sessions_since_rebalance=2,
    )
    carried = pd.read_csv(manifest["target"])

    assert carried["instrument"].tolist() == ["SZ300001"]
    assert carried["target_shares"].tolist() == [300]
    assert carried["signal_date"].tolist() == ["2026-06-11"]
    assert manifest["selection_mode"] == "scheduled_carry"
    assert manifest["middle_rebalance"]["forced_forbidden_exits"] == ["SZ300002"]
    assert manifest["outer_overlay"]["applied_on_this_release"] is False
