from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from build_held_intraday_decision_dataset import (  # noqa: E402
    CONTEXT_FEATURE_COLUMNS,
    add_position_dependent_features,
    trigger_label_prefix,
)
from build_sequential_daily_meta_gate import build_daily_meta_frame  # noqa: E402
from freeze_sequential_daily_meta_gate import score_meta_features  # noqa: E402
from rebuild_held_intraday_from_cache import (  # noqa: E402
    partition_context,
    refresh_context_dependent_columns,
)
from replay_trigger_aligned_predictions import replay  # noqa: E402
from run_held_intraday_hurdle_oof import expected_hurdle_score  # noqa: E402


def test_frozen_daily_meta_score_reproduces_scaler_and_ridge() -> None:
    artifact = {
        "meta_features": ["first", "second"],
        "scaler": {"mean": [1.0, 2.0], "scale": [2.0, 4.0]},
        "ridge": {"coefficient": [3.0, -1.0], "intercept": 0.25},
    }

    score = score_meta_features({"first": 5.0, "second": 10.0}, artifact)

    assert score == pytest.approx(4.25)


def _context_row(date: str, instrument: str, shares: float) -> dict:
    row = {
        "datetime": date,
        "trade_date": str((pd.Timestamp(date) + pd.offsets.BDay(1)).date()),
        "instrument": instrument,
        "shares": shares,
        "mark_price": 10.0,
        "weight": 0.02,
        "target_weight": 0.025,
        "target_shares": shares + 100.0,
        "held_age_days": 3.0,
        "held_unrealized_ret_approx": 0.01,
        "held_prev_day_ret": 0.005,
        "held_weight_gap_to_target": 0.005,
        "held_abs_weight_gap_to_target": 0.005,
        "held_share_gap_to_target": 100.0,
        "held_target_share_ratio": 0.8,
        "target_missing": 0.0,
        "rank": 2.0,
        "middle": 0.7,
        "group": "software",
    }
    assert not [col for col in CONTEXT_FEATURE_COLUMNS if col not in row]
    return row


def test_partition_context_finds_shared_and_incremental_keys() -> None:
    old = pd.DataFrame(
        [
            _context_row("2024-01-02", "SZ000001", 1000.0),
            _context_row("2024-01-03", "SZ000002", 1000.0),
        ]
    )
    new = pd.DataFrame(
        [
            _context_row("2024-01-02", "SZ000001", 800.0),
            _context_row("2024-01-04", "SZ000003", 1000.0),
        ]
    )

    shared, new_only, old_only = partition_context(old, new)

    assert len(shared) == 1
    assert shared.iloc[0]["shares"] == 800.0
    assert new_only["instrument"].tolist() == ["SZ000003"]
    assert old_only["instrument"].tolist() == ["SZ000002"]


def test_position_dependent_features_recompute_fee_drag() -> None:
    frame = pd.DataFrame(
        {
            "shares": [1000.0],
            "mark_price": [10.0],
            "sell_price_decision": [10.0],
            "visible_volume": [5000.0],
            "visible_amount": [50000.0],
        }
    )

    result = add_position_dependent_features(frame, sell_cost=0.0025, buy_cost=0.001)

    assert result.iloc[0]["held_market_value"] == 10000.0
    assert result.iloc[0]["visible_volume_to_held"] == 5.0
    assert result.iloc[0]["visible_amount_to_position"] == 5.0
    assert result.iloc[0]["t0_volume_10pct"] == 100.0
    assert result.iloc[0]["t0_min_fee_drag_10pct"] == 0.01


def test_refresh_context_updates_position_labels_and_snapshots() -> None:
    old_context = _context_row("2024-01-02", "SZ000001", 1000.0)
    new_context = _context_row("2024-01-02", "SZ000001", 200.0)
    row = {
        **old_context,
        "decision_time": "1000",
        "decision_minute": 600,
        "sell_price_decision": 10.0,
        "label_trigger_window_low": 9.8,
        "label_trigger_window_high": 10.2,
        "label_trigger_window_minutes": 60,
        "visible_ret_from_open": 0.01,
        "visible_gap_vs_mark": 0.0,
        "visible_last_vs_mark": 0.01,
        "visible_range": 0.02,
        "visible_range_pos": 0.6,
        "visible_vwap_dev": 0.003,
        "visible_volume": 5000.0,
        "visible_amount": 50000.0,
        "buyback_1420_1430_price": 9.9,
        "buyback_1445_1450_price": 9.8,
        "buyback_1450_1455_price": 9.8,
    }

    result = refresh_context_dependent_columns(
        pd.DataFrame([row]),
        pd.DataFrame([new_context]),
        sell_cost=0.0025,
        buy_cost=0.001,
        slippage=0.0005,
    )

    assert result.iloc[0]["shares"] == 200.0
    assert result.iloc[0]["held_market_value"] == 2000.0
    assert result.iloc[0]["t0_exec_volume_one_lot_max50"] == 100.0
    assert result.iloc[0]["held_universe_size"] == 1.0
    assert result.iloc[0]["industry_held_count"] == 1.0


def test_trigger_replay_reports_base_plus_incremental_overlay_drawdown(
    tmp_path: Path,
) -> None:
    prefix = trigger_label_prefix("sell_first", 0.0075, 0.0, "1445_1450")
    dates = ["2024-01-03", "2024-01-04"]
    predictions = pd.DataFrame(
        {
            "datetime": ["2024-01-02", "2024-01-03"],
            "trade_date": dates,
            "instrument": ["SZ000001", "SZ000001"],
            "decision_time": ["1000", "1000"],
            "fold": [1, 1],
            "score": [-0.02, -0.02],
        }
    )
    dataset = predictions[["datetime", "trade_date", "instrument", "decision_time"]].copy()
    dataset["t0_exec_volume_one_lot_max50"] = 100.0
    dataset["buyback_1445_1450_price"] = 10.0
    dataset[f"{prefix}_entry_price"] = 10.075
    dataset[f"{prefix}_touched"] = 1.0
    dataset[f"{prefix}_realized_pnl"] = [100.0, -200.0]
    dataset[f"{prefix}_realized_edge"] = [0.01, -0.02]
    account = pd.DataFrame(
        {
            "trade_date": dates,
            "nav": [1000.0, 900.0],
            "cash": [500.0, 500.0],
        }
    )
    prediction_path = tmp_path / "predictions.csv"
    dataset_path = tmp_path / "dataset.csv"
    account_path = tmp_path / "account.csv"
    predictions.to_csv(prediction_path, index=False)
    dataset.to_csv(dataset_path, index=False)
    account.to_csv(account_path, index=False)

    report = replay(
        prediction_path,
        dataset_path,
        account_path,
        tmp_path / "report.json",
        tmp_path / "trades.csv",
        tmp_path / "daily.csv",
        direction="sell_first",
        trigger_distance=0.0075,
        touch_buffer=0.0,
        buyback_window="1445_1450",
        score_column="score",
        gate_score_column="score",
        score_threshold=None,
        daily_top_n=1,
        max_daily_turnover=3.0,
    )

    daily = pd.read_csv(tmp_path / "daily.csv")
    assert daily["overlay_nav"].tolist() == [1100.0, 800.0]
    assert report["profile"]["score_gate_enabled"] is False
    assert report["base_max_drawdown"] == pytest.approx(0.1)
    assert report["max_overlay_drawdown"] == pytest.approx(1.0 - 800.0 / 1100.0)
    assert report["overlay_drawdown_delta"] == pytest.approx(
        report["max_overlay_drawdown"] - 0.1
    )


def test_expected_hurdle_score_multiplies_touch_probability_and_net_edge() -> None:
    result = expected_hurdle_score(
        touch_probability=pd.Series([0.25, 0.8]).to_numpy(),
        conditional_edge=pd.Series([0.02, -0.01]).to_numpy(),
    )

    assert result.tolist() == pytest.approx([0.005, -0.008])


def test_daily_meta_frame_uses_only_ranked_top_two_for_target() -> None:
    predictions = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-02"]
            ),
            "instrument": ["SZ000001", "SZ000002", "SZ000003"],
            "raw_score": [0.8, 0.6, 0.1],
            "realized_edge": [0.01, -0.002, 0.5],
        }
    )

    daily = build_daily_meta_frame(
        predictions,
        ranking_col="raw_score",
        realized_edge_col="realized_edge",
        daily_top_n=2,
    )

    assert daily.iloc[0]["daily_realized_edge"] == pytest.approx(0.008)
    assert daily.iloc[0]["score_top2_mean"] == pytest.approx(0.7)
    assert daily.iloc[0]["held_count"] == 3.0
