from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from build_held_position_context import (  # noqa: E402
    build_context,
    expand_active_target_snapshots,
    load_targets,
)
from build_minute_staging import source_overlaps_date_range  # noqa: E402
from build_held_intraday_decision_dataset import (  # noqa: E402
    add_cross_sectional_features,
    add_trigger_aligned_labels,
    trigger_label_prefix,
)
from replay_held_intraday_t0 import run_replay  # noqa: E402
from run_held_intraday_anchored_oof import (  # noqa: E402
    classification_target,
    anchored_train_validation_calibration_dates,
    anchored_train_validation_dates,
    edge_col_for_label,
    time_decay_session_weights,
)
from run_held_intraday_two_stage_oof import (  # noqa: E402
    calibrate_probability,
    expected_value_from_win_probability,
)
from evaluate_held_intraday_decision_model import select_feature_cols  # noqa: E402
from run_held_intraday_catboost_densemble_oof import (  # noqa: E402
    FrameDatasetAdapter,
    qlib_table,
)
from blend_held_intraday_oof_predictions import blend  # noqa: E402


def _calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"])
    )


def test_minute_sources_are_pruned_before_parsing():
    start = pd.Timestamp("2025-01-07")
    end = pd.Timestamp("2026-04-03")

    assert source_overlaps_date_range(Path("2025_1min/sz300001_2025.csv"), start, end)
    assert not source_overlaps_date_range(Path("2024_1min/sz300001_2024.csv"), start, end)
    assert source_overlaps_date_range(Path("2026-03/20260331/sz300001.csv"), start, end)
    assert not source_overlaps_date_range(Path("2026-05/20260506/sz300001.csv"), start, end)
    assert source_overlaps_date_range(
        (Path("2026-03/20260331_1min.zip"), "sz300001.csv"), start, end
    )


def test_trigger_aligned_labels_match_limit_touch_and_inventory_rules():
    frame = pd.DataFrame(
        [
            {
                "instrument": "SZ300001",
                "shares": 200,
                "sell_price_decision": 10.0,
                "buyback_1445_1450_price": 10.2,
                "label_trigger_window_low": 9.93,
                "label_trigger_window_high": 10.09,
            },
            {
                "instrument": "SH688001",
                "shares": 200,
                "sell_price_decision": 10.0,
                "buyback_1445_1450_price": 9.8,
                "label_trigger_window_low": 9.90,
                "label_trigger_window_high": 10.09,
            },
        ]
    )
    out = add_trigger_aligned_labels(
        frame,
        sell_cost=0.0025,
        buy_cost=0.001,
        slippage=0.0005,
    )
    buy = trigger_label_prefix("buy_first", 0.006, 0.0)
    sell = trigger_label_prefix("sell_first", 0.0075, 0.0)

    assert out.loc[0, f"{buy}_touched"] == 1.0
    assert out.loc[0, f"{buy}_realized_pnl"] > 0
    assert out.loc[0, f"{sell}_touched"] == 1.0
    assert out.loc[0, f"{sell}_realized_pnl"] < 0
    # STAR requires a 200-share buy lot, which exceeds 50% of this holding.
    assert pd.isna(out.loc[1, f"{buy}_touched"])
    assert pd.isna(out.loc[1, f"{sell}_touched"])


def test_trigger_label_edge_inference():
    label = "trigger_buy_first_0060_touch0000_1445_1450_one_lot_max50_realized_hit"
    assert edge_col_for_label(label) == (
        "trigger_buy_first_0060_touch0000_1445_1450_one_lot_max50_realized_edge"
    )

    early_prefix = trigger_label_prefix(
        "sell_first",
        0.0075,
        0.001,
        "1420_1430",
    )
    assert early_prefix == (
        "trigger_sell_first_0075_touch0010_1420_1430_one_lot_max50"
    )


def test_two_stage_expected_value_preserves_cost_asymmetry():
    probability = np.array([0.0, 0.25, 1.0])
    expected = expected_value_from_win_probability(probability, 0.02, -0.01)

    assert np.allclose(expected, [-0.01, -0.0025, 0.02])


def test_two_stage_probability_calibration_uses_only_supplied_calibration_rows():
    raw = np.linspace(0.0, 1.0, 200)
    label = (raw >= 0.6).astype(int)
    test = np.array([0.2, 0.5, 0.8])

    calibrated, audit = calibrate_probability(
        raw,
        label,
        test,
        mode="isotonic",
        stage="unit",
    )

    assert audit["rows"] == 200
    assert np.all(np.diff(calibrated) >= 0.0)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_live_factor_modes_keep_limit_and_industry_ablations_disjoint():
    frame = pd.DataFrame(
        columns=[
            "visible_ret_from_open",
            "held_age_days",
            "visible_distance_to_limit_up",
            "industry_visible_ret_rel",
            "cs_visible_distance_to_limit_up_rank",
            "cs_industry_visible_ret_rel_rank",
        ]
    )

    core = set(select_feature_cols(frame, "live_core"))
    limit = set(select_feature_cols(frame, "live_limit"))
    industry = set(select_feature_cols(frame, "live_industry"))
    combined = set(select_feature_cols(frame, "live"))

    assert core == {"visible_ret_from_open"}
    assert "held_age_days" not in combined
    assert "visible_distance_to_limit_up" in limit
    assert "industry_visible_ret_rel" not in limit
    assert "industry_visible_ret_rel" in industry
    assert "visible_distance_to_limit_up" not in industry
    assert limit | industry == combined


def test_live_compact_modes_keep_only_reproducible_domain_features():
    frame = pd.DataFrame(
        columns=[
            "held_age_days",
            "held_unrealized_ret_approx",
            "visible_ret_from_open",
            "visible_recent_5m_ret",
            "visible_distance_to_limit_up",
            "industry_visible_ret_rel",
            "visible_price_vol_corr",
        ]
    )

    core = set(select_feature_cols(frame, "live_compact_core"))
    limit = set(select_feature_cols(frame, "live_compact_limit"))

    assert core == {
        "held_unrealized_ret_approx",
        "visible_ret_from_open",
        "visible_recent_5m_ret",
    }
    assert limit == core | {"visible_distance_to_limit_up"}
    assert "held_age_days" not in limit
    assert "industry_visible_ret_rel" not in limit
    assert "visible_price_vol_corr" not in limit


def test_classification_target_supports_explicit_net_edge_buffer():
    frame = pd.DataFrame(
        {
            "realized_hit": [0.0, 1.0, 1.0, 1.0],
            "realized_edge": [-0.01, 0.001, 0.0021, 0.01],
        }
    )

    legacy = classification_target(
        frame,
        label_col="realized_hit",
        edge_col="realized_edge",
        positive_edge_threshold=0.0,
    )
    buffered = classification_target(
        frame,
        label_col="realized_hit",
        edge_col="realized_edge",
        positive_edge_threshold=0.002,
    )

    assert legacy.tolist() == [0, 1, 1, 1]
    assert buffered.tolist() == [0, 0, 1, 1]


def test_time_decay_session_weights_are_causal_and_session_based():
    dates = pd.Series(
        pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-06"])
    )

    weights = time_decay_session_weights(dates, half_life_sessions=2.0)

    assert np.allclose(
        weights.to_numpy(),
        np.array([0.5, 0.5, 2 ** -0.5, 1.0]),
    )


def test_densemble_frame_adapter_preserves_feature_label_contract():
    frame = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
            "model_target": [0, 1],
        }
    )
    table = qlib_table(frame, ["feature_a", "feature_b"])
    dataset = FrameDatasetAdapter({"train": table, "valid": table.iloc[:1]})

    train, valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"])
    features = dataset.prepare("train", col_set="feature")

    assert train["label"].iloc[:, 0].tolist() == [0.0, 1.0]
    assert len(valid) == 1
    assert features.columns.tolist() == ["feature_a", "feature_b"]


def test_oof_blend_refuses_to_change_labels_and_averages_scores(tmp_path):
    label_col = "trigger_sell_realized_hit"
    edge_col = "trigger_sell_realized_edge"
    base = pd.DataFrame(
        {
            "datetime": ["2026-01-02", "2026-01-05"],
            "trade_date": ["2026-01-02", "2026-01-05"],
            "instrument": ["SZ300001", "SZ300002"],
            "decision_time": ["1000", "1000"],
            "fold": [1, 1],
            label_col: [1, 0],
            edge_col: [0.01, -0.01],
            "raw_score": [0.8, 0.2],
            "score": [0.9, 0.1],
        }
    )
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "blend.csv"
    report = tmp_path / "blend.json"
    base.to_csv(first, index=False)
    base.assign(raw_score=[0.6, 0.4], score=[0.7, 0.3]).to_csv(second, index=False)

    result = blend(
        [first, second],
        [1.0, 1.0],
        output,
        report,
        label_col=label_col,
        edge_col=edge_col,
    )
    got = pd.read_csv(output)

    assert result["weights"] == [0.5, 0.5]
    assert np.allclose(got["score"], [0.8, 0.2])
    assert got[edge_col].tolist() == [0.01, -0.01]
    assert got["consensus_max_rank"].tolist() == [1.0, 1.0]


def test_anchored_split_keeps_validation_and_purge_out_of_fit_and_test():
    dates = pd.bdate_range("2025-01-02", "2025-02-14")
    frame = pd.DataFrame({"trade_date": dates})
    test_start = pd.Timestamp("2025-02-03")

    fit, validation, purged = anchored_train_validation_dates(
        frame,
        test_start,
        validation_fraction=0.20,
        min_validation_sessions=5,
        purge_sessions=1,
    )

    assert len(validation) >= 5
    assert len(purged) == 1
    assert fit.max() < validation.min() < purged.min() < test_start
    assert not set(fit) & set(validation)
    assert not set(fit) & set(purged)
    assert not set(validation) & set(purged)


def test_anchored_calibration_split_is_strictly_before_purge_and_test():
    dates = pd.bdate_range("2025-01-02", "2025-04-30")
    frame = pd.DataFrame({"trade_date": dates})
    test_start = pd.Timestamp("2025-04-01")

    fit, validation, calibration, purged = anchored_train_validation_calibration_dates(
        frame,
        test_start,
        validation_fraction=0.15,
        min_validation_sessions=10,
        calibration_fraction=0.10,
        min_calibration_sessions=8,
        purge_sessions=1,
    )

    assert fit.max() < validation.min() < calibration.min() < purged.min() < test_start
    partitions = [set(part) for part in (fit, validation, calibration, purged)]
    for left, first in enumerate(partitions):
        for second in partitions[left + 1 :]:
            assert not first & second


def test_edge_column_is_tied_to_fixed_buyback_label():
    assert edge_col_for_label("t0_hit_1420_1430") == "t0_edge_1420_1430"
    assert edge_col_for_label("t0_best_hit") == "t0_best_edge"


def test_cross_sectional_features_are_computed_within_decision_snapshot():
    frame = pd.DataFrame(
        [
            {"trade_date": "2025-01-03", "decision_time": "1000", "instrument": "SZA", "visible_ret_from_open": 0.01},
            {"trade_date": "2025-01-03", "decision_time": "1000", "instrument": "SZB", "visible_ret_from_open": 0.03},
            {"trade_date": "2025-01-06", "decision_time": "1000", "instrument": "SZA", "visible_ret_from_open": -0.02},
        ]
    )

    out = add_cross_sectional_features(frame)

    first = out[out["trade_date"] == "2025-01-03"].sort_values("instrument")
    assert first["held_universe_size"].tolist() == [2.0, 2.0]
    assert first["cs_visible_ret_from_open_rank"].tolist() == [0.5, 1.0]
    assert first["cs_visible_ret_from_open_rel"].round(6).tolist() == [-0.01, 0.01]
    last = out[out["trade_date"] == "2025-01-06"].iloc[0]
    assert last["held_universe_size"] == 1.0
    assert last["cs_visible_ret_from_open_rank"] == 1.0
    assert last["cs_visible_ret_from_open_rel"] == 0.0


def test_daily_top_n_replay_preserves_fold_level_economics(tmp_path):
    prediction = tmp_path / "pred_1000_to_1420.csv"
    dataset = tmp_path / "dataset.csv"
    output_json = tmp_path / "replay.json"
    output_trades = tmp_path / "trades.csv"
    output_daily = tmp_path / "daily.csv"
    rows = []
    preds = []
    specifications = [
        ("2025-01-03", 1, "SZA", 0.9, 10.0, 9.0),
        ("2025-01-03", 1, "SZB", 0.8, 10.0, 10.5),
        ("2025-01-06", 2, "SZC", 0.9, 10.0, 11.0),
        ("2025-01-06", 2, "SZD", 0.8, 10.0, 9.5),
    ]
    for date, fold, instrument, score, sell, buy in specifications:
        common = {
            "datetime": date,
            "trade_date": date,
            "instrument": instrument,
            "decision_time": "1000",
        }
        preds.append({**common, "fold": fold, "score": score})
        rows.append(
            {
                **common,
                "shares": 1000,
                "sell_price_decision": sell,
                "buyback_1420_1430_price": buy,
            }
        )
    pd.DataFrame(preds).to_csv(prediction, index=False)
    pd.DataFrame(rows).to_csv(dataset, index=False)

    report = run_replay(
        prediction,
        dataset,
        output_json,
        output_trades,
        output_daily,
        thresholds=[0.0],
        trade_fractions=[1.0],
        daily_account_path=None,
        default_nav=1_000_000.0,
        lot_size=100,
        buy_cost=0.001,
        sell_cost=0.0025,
        slippage=0.0005,
        min_cost=5.0,
        max_daily_turnover=0.10,
        max_symbols_per_day=10,
        max_round_trips_per_symbol=0,
        selection_mode="daily_top_n",
        daily_top_ns=[1],
    )

    best = report["best"]
    assert best["round_trips"] == 2
    assert best["fold_count"] == 2
    assert best["profitable_folds"] == 1
    assert best["worst_fold_pnl"] < 0
    trades = pd.read_csv(output_trades)
    assert trades["instrument"].tolist() == ["SZA", "SZC"]


def test_replay_explicit_buyback_window_overrides_prediction_filename(tmp_path):
    prediction = tmp_path / "short_name.csv"
    dataset = tmp_path / "dataset.csv"
    output_json = tmp_path / "result.json"
    output_trades = tmp_path / "trades.csv"
    output_daily = tmp_path / "daily.csv"
    common = {
        "datetime": "2025-01-03",
        "trade_date": "2025-01-03",
        "instrument": "SZA",
        "decision_time": "1000",
    }
    pd.DataFrame([{**common, "fold": 1, "score": 0.9}]).to_csv(prediction, index=False)
    pd.DataFrame(
        [
            {
                **common,
                "shares": 1000,
                "sell_price_decision": 10.0,
                "buyback_1420_1430_price": 11.0,
                "buyback_1445_1450_price": 9.0,
            }
        ]
    ).to_csv(dataset, index=False)

    report = run_replay(
        prediction,
        dataset,
        output_json,
        output_trades,
        output_daily,
        thresholds=[0.0],
        trade_fractions=[1.0],
        daily_account_path=None,
        default_nav=1_000_000.0,
        lot_size=100,
        buy_cost=0.001,
        sell_cost=0.0025,
        slippage=0.0005,
        min_cost=5.0,
        max_daily_turnover=0.10,
        max_symbols_per_day=1,
        max_round_trips_per_symbol=0,
        selection_mode="daily_top_n",
        daily_top_ns=[1],
        buyback_window="1445_1450",
    )

    assert report["buyback_col"] == "buyback_1445_1450_price"
    assert report["buyback_window_explicit"] is True
    assert report["best"]["cum_pnl"] > 0


def test_gm_target_snapshot_uses_execution_date_and_carries_whole_portfolio(tmp_path):
    path = tmp_path / "gm_targets.csv"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-03", "signal_date": "2025-01-02", "instrument": "SZA", "rank": 1, "middle": 0.9, "target_weight": 0.5, "target_shares": 100, "group": "g"},
            {"trade_date": "2025-01-03", "signal_date": "2025-01-02", "instrument": "SZB", "rank": 2, "middle": 0.8, "target_weight": 0.5, "target_shares": 100, "group": "g"},
            {"trade_date": "2025-01-07", "signal_date": "2025-01-06", "instrument": "SZB", "rank": 1, "middle": 0.7, "target_weight": 0.5, "target_shares": 100, "group": "g"},
            {"trade_date": "2025-01-07", "signal_date": "2025-01-06", "instrument": "SZC", "rank": 2, "middle": 0.6, "target_weight": 0.5, "target_shares": 100, "group": "g"},
        ]
    ).to_csv(path, index=False)

    targets = load_targets(path, _calendar())
    expanded = expand_active_target_snapshots(
        targets,
        pd.DatetimeIndex(pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"])),
    )

    assert set(expanded.loc[expanded["datetime"] == pd.Timestamp("2025-01-06"), "instrument"]) == {"SZA", "SZB"}
    assert set(expanded.loc[expanded["datetime"] == pd.Timestamp("2025-01-07"), "instrument"]) == {"SZB", "SZC"}
    assert set(expanded.loc[expanded["datetime"] == pd.Timestamp("2025-01-08"), "instrument"]) == {"SZB", "SZC"}


def test_production_target_snapshot_moves_planning_date_to_next_session(tmp_path):
    path = tmp_path / "production_targets.csv"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-02", "instrument": "SZA", "rank": 1, "middle": 0.9, "target_weight": 0.5, "target_shares": 100, "group": "g"},
            {"trade_date": "2025-01-06", "instrument": "SZC", "rank": 1, "middle": 0.8, "target_weight": 0.5, "target_shares": 100, "group": "g"},
        ]
    ).to_csv(path, index=False)

    targets = load_targets(path, _calendar())

    assert targets["target_signal_date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-02", "2025-01-06"]
    assert targets["target_trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-03", "2025-01-07"]


def test_build_context_marks_only_real_stale_holding_as_target_missing(tmp_path):
    holdings_path = tmp_path / "holdings.csv"
    targets_path = tmp_path / "targets.csv"
    windows_path = tmp_path / "windows.csv"
    calendar_path = tmp_path / "calendar.txt"
    output_csv = tmp_path / "context.csv"
    output_json = tmp_path / "context.json"

    pd.Series(_calendar().strftime("%Y-%m-%d")).to_csv(calendar_path, index=False, header=False)
    pd.DataFrame(
        [
            {"trade_date": "2025-01-03", "instrument": "SZA", "shares": 100, "mark_price": 10.0, "market_value": 1000.0, "weight": 0.1},
            {"trade_date": "2025-01-06", "instrument": "SZA", "shares": 100, "mark_price": 10.1, "market_value": 1010.0, "weight": 0.1},
            {"trade_date": "2025-01-07", "instrument": "SZA", "shares": 100, "mark_price": 10.2, "market_value": 1020.0, "weight": 0.1},
            {"trade_date": "2025-01-07", "instrument": "SZC", "shares": 100, "mark_price": 20.0, "market_value": 2000.0, "weight": 0.2},
        ]
    ).to_csv(holdings_path, index=False)
    pd.DataFrame(
        [
            {"trade_date": "2025-01-03", "signal_date": "2025-01-02", "instrument": "SZA", "rank": 1, "middle": 0.9, "target_weight": 0.1, "target_shares": 100, "group": "g"},
            {"trade_date": "2025-01-07", "signal_date": "2025-01-06", "instrument": "SZC", "rank": 1, "middle": 0.8, "target_weight": 0.2, "target_shares": 100, "group": "g"},
        ]
    ).to_csv(targets_path, index=False)
    pd.DataFrame(
        [
            {"trade_date": "2025-01-06", "instrument": "SZA", "open_exec": 10.0, "close_exec": 10.1, "mark_close": 10.1},
            {"trade_date": "2025-01-07", "instrument": "SZA", "open_exec": 10.1, "close_exec": 10.2, "mark_close": 10.2},
            {"trade_date": "2025-01-08", "instrument": "SZA", "open_exec": 10.2, "close_exec": 10.3, "mark_close": 10.3},
            {"trade_date": "2025-01-08", "instrument": "SZC", "open_exec": 20.0, "close_exec": 20.1, "mark_close": 20.1},
        ]
    ).to_csv(windows_path, index=False)

    report = build_context(
        holdings_path,
        targets_path,
        windows_path,
        calendar_path,
        output_csv,
        output_json,
        buy_cost=0.001,
        sell_cost=0.0025,
        slippage=0.0005,
    )
    context = pd.read_csv(output_csv)

    assert report["target_snapshot_dates"] == 2
    assert report["expanded_target_dates"] == 3
    assert report["target_missing_ratio"] == 0.25
    assert report["execution_after_holding_ratio"] == 1.0
    assert (pd.to_datetime(context["trade_date"]) > pd.to_datetime(context["datetime"])).all()
    first = context[(context["datetime"] == "2025-01-03") & (context["instrument"] == "SZA")]
    assert first["trade_date"].tolist() == ["2025-01-06"]
    stale = context[(context["datetime"] == "2025-01-07") & (context["instrument"] == "SZA")]
    assert stale["target_missing"].tolist() == [1]
    assert context.loc[context["target_missing"] == 0, "target_shares"].tolist() == [100.0, 100.0, 100.0]
