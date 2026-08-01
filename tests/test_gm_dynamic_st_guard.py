from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = (
    ROOT
    / "src"
    / "helki_quant"
    / "deployment"
    / "gmquant"
    / "main.py"
)


def _load_main():
    spec = importlib.util.spec_from_file_location("gm_dynamic_st_guard_under_test", MAIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dynamic_risk_reason_covers_live_name_and_market_states():
    module = _load_main()

    assert "risk_name:ST派瑞" in module._dynamic_risk_reason("ST派瑞", 0, None, "2026-06-12")
    assert "risk_name:*ST样本" in module._dynamic_risk_reason("*ST样本", 0, None, "2026-06-12")
    assert "delisting_name:样本退" in module._dynamic_risk_reason("样本退", 0, None, "2026-06-12")
    assert module._dynamic_risk_reason("正常股份", 1, None, "2026-06-12") == "suspended"
    assert "delisted:2026-06-11" in module._dynamic_risk_reason(
        "正常股份", 0, "2026-06-11 00:00:00+08:00", "2026-06-12"
    )
    assert module._dynamic_risk_reason("正常股份", 0, "2027-01-01", "2026-06-12") == ""


def test_dynamic_refresh_removes_risk_targets_and_preserves_real_holdings(tmp_path, monkeypatch):
    module = _load_main()
    module.RUN_MODE = "LIVE"
    module.DYNAMIC_ST_CHECK = True
    module.DYNAMIC_ST_FAIL_CLOSED = True
    module.AUDIT_DIR = tmp_path
    module.AUDIT_RUN_ID = "dynamic_guard"
    module._targets_by_date = {
        "2026-06-12": {
            "SZSE.300831": {"target_weight": 0.004, "target_shares": 100},
            "SZSE.300290": {"target_weight": 0.004, "target_shares": 300},
            "SZSE.300001": {"target_weight": 0.004, "target_shares": 200},
        }
    }
    module._previous_symbols = {"SZSE.300831", "SZSE.300290", "SZSE.300001"}
    module._previous_volumes = {"SZSE.300831": 100, "SZSE.300290": 300, "SZSE.300001": 200}
    module._forbidden_local_symbols = set()
    module._forbidden_gm_symbols = set()
    module._dynamic_risk_events = []
    module._dynamic_risk_check_date = None
    module._forbidden_clear_pending = set()

    rows = [
        {"symbol": "SZSE.300831", "sec_name": "ST派瑞", "is_suspended": 0},
        {"symbol": "SZSE.300290", "sec_name": "ST荣科", "is_suspended": 0},
        {"symbol": "SZSE.300001", "sec_name": "正常股份", "is_suspended": 0},
    ]
    monkeypatch.setattr(module, "get_instruments", lambda **_: rows)
    context = type("Context", (), {"now": pd.Timestamp("2026-06-12 10:16:27")})()

    module._refresh_dynamic_market_risk(context, "test", force=True)

    assert set(module._targets_by_date["2026-06-12"]) == {"SZSE.300001"}
    assert {"SZSE.300831", "SZSE.300290"} <= module._forbidden_gm_symbols
    assert module._previous_volumes["SZSE.300831"] == 100
    assert module._previous_volumes["SZSE.300290"] == 300
    audit = pd.read_csv(tmp_path / "dynamic_guard" / "dynamic_market_risk.csv")
    assert set(audit["symbol"]) == {"SZSE.300831", "SZSE.300290"}


def test_forbidden_clear_is_deduplicated_without_faking_fills(monkeypatch):
    module = _load_main()
    module.RUN_MODE = "LIVE"
    module.LIVE_ALLOWED = True
    module.VERBOSE_ORDERS = False
    module._previous_symbols = {"SZSE.300831", "SZSE.300290"}
    module._previous_volumes = {"SZSE.300831": 100, "SZSE.300290": 300}
    module._forbidden_gm_symbols = {"SZSE.300831", "SZSE.300290"}
    module._forbidden_local_symbols = set()
    module._forbidden_clear_pending = set()
    submitted: list[tuple[str, int]] = []
    monkeypatch.setattr(module, "_record_submission", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_order_target_volume_safe",
        lambda symbol, shares: submitted.append((symbol, shares)),
    )
    context = type("Context", (), {"now": pd.Timestamp("2026-06-12 10:16:27")})()

    module._clear_forbidden_positions(context, "test")
    module._clear_forbidden_positions(context, "test_again")

    assert submitted == [("SZSE.300290", 0), ("SZSE.300831", 0)]
    assert module._previous_volumes == {"SZSE.300831": 100, "SZSE.300290": 300}
    assert module._previous_symbols == {"SZSE.300831", "SZSE.300290"}


def test_dynamic_refresh_fails_closed_when_live_query_fails(tmp_path, monkeypatch):
    module = _load_main()
    module.RUN_MODE = "LIVE"
    module.DYNAMIC_ST_CHECK = True
    module.DYNAMIC_ST_FAIL_CLOSED = True
    module.AUDIT_DIR = tmp_path
    module.AUDIT_RUN_ID = "query_failure"
    module._targets_by_date = {
        "2026-06-12": {"SZSE.300001": {"target_weight": 0.004, "target_shares": 200}}
    }
    module._previous_symbols = set()
    module._dynamic_risk_events = []
    module._dynamic_risk_check_date = None
    monkeypatch.setattr(module, "get_instruments", lambda **_: (_ for _ in ()).throw(RuntimeError("offline")))
    context = type("Context", (), {"now": pd.Timestamp("2026-06-12 10:16:27")})()

    with pytest.raises(RuntimeError, match="dynamic ST/market-state query failed"):
        module._refresh_dynamic_market_risk(context, "test", force=True)


def test_live_signal_context_accepts_completed_previous_session():
    module = _load_main()
    module.REQUIRE_SIGNAL_DATE = True
    module.MAX_LIVE_SIGNAL_AGE_DAYS = 4
    module.MAX_LIVE_SIGNAL_TO_TARGET_DAYS = 4
    module._signal_dates_by_target_date = {"2026-07-15": ("2026-07-14",)}

    signal = module._validate_live_signal_context(
        pd.Timestamp("2026-07-15 09:20:00"),
        pd.Timestamp("2026-07-15"),
    )

    assert signal == pd.Timestamp("2026-07-14")


def test_live_signal_context_rejects_trade_date_relabel_of_old_signal():
    module = _load_main()
    module.REQUIRE_SIGNAL_DATE = True
    module.MAX_LIVE_SIGNAL_AGE_DAYS = 4
    module.MAX_LIVE_SIGNAL_TO_TARGET_DAYS = 4
    module._signal_dates_by_target_date = {"2026-07-15": ("2026-06-05",)}

    with pytest.raises(RuntimeError, match="signal_date is stale"):
        module._validate_live_signal_context(
            pd.Timestamp("2026-07-15 09:20:00"),
            pd.Timestamp("2026-07-15"),
        )


def test_live_signal_context_requires_one_source_date_per_target_date():
    module = _load_main()
    module.REQUIRE_SIGNAL_DATE = True
    module._signal_dates_by_target_date = {
        "2026-07-15": ("2026-07-11", "2026-07-14")
    }

    with pytest.raises(RuntimeError, match="exactly one signal_date"):
        module._validate_live_signal_context(
            pd.Timestamp("2026-07-15 09:20:00"),
            pd.Timestamp("2026-07-15"),
        )


def test_live_signal_context_rejects_unfinished_same_day_signal():
    module = _load_main()
    module.REQUIRE_SIGNAL_DATE = True
    module.MAX_LIVE_SIGNAL_AGE_DAYS = 4
    module.MAX_LIVE_SIGNAL_TO_TARGET_DAYS = 4
    module._signal_dates_by_target_date = {"2026-07-16": ("2026-07-15",)}

    with pytest.raises(RuntimeError, match="session is not complete"):
        module._validate_live_signal_context(
            pd.Timestamp("2026-07-15 14:59:00"),
            pd.Timestamp("2026-07-16"),
        )

    assert module._validate_live_signal_context(
        pd.Timestamp("2026-07-15 15:10:00"),
        pd.Timestamp("2026-07-16"),
    ) == pd.Timestamp("2026-07-15")


def test_bound_paper_account_position_sync_fails_closed(monkeypatch):
    module = _load_main()
    module.RUN_MODE = "LIVE"
    module.REQUIRE_ACCOUNT_ID = True
    module.GM_ACCOUNT_ID = "paper-account"
    module.SYNC_EXISTING_POSITIONS = True
    monkeypatch.setattr(
        module,
        "get_position",
        lambda **_: (_ for _ in ()).throw(RuntimeError("account unavailable")),
    )
    context = type("Context", (), {})()

    with pytest.raises(RuntimeError, match="positions could not be read"):
        module._sync_existing_positions(context)

    assert module._position_sync_succeeded is False
    assert module._position_sync_source is None


def test_bound_paper_account_empty_position_snapshot_is_valid(monkeypatch):
    module = _load_main()
    module.RUN_MODE = "LIVE"
    module.REQUIRE_ACCOUNT_ID = True
    module.GM_ACCOUNT_ID = "paper-account"
    module.SYNC_EXISTING_POSITIONS = True
    monkeypatch.setattr(module, "get_position", lambda **_: [])
    context = type("Context", (), {})()

    module._sync_existing_positions(context)

    assert module._position_sync_succeeded is True
    assert module._position_sync_source == "get_position(account_id)"
    assert module._previous_symbols == set()
    assert module._previous_volumes == {}


def test_live_target_submission_waits_for_fill_and_deduplicates(monkeypatch):
    module = _load_main()
    module.RUN_MODE = "LIVE"
    module.VERBOSE_ORDERS = False
    module._pending_target_orders = {}
    module._previous_symbols = {"SZSE.300001"}
    module._previous_volumes = {"SZSE.300001": 100}
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(module, "_record_submission", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_write_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_order_target_volume_safe",
        lambda symbol, shares: calls.append((symbol, shares)),
    )

    first = module._submit_target_volume(
        "2026-07-15", "test", "SZSE.300001", 0.004, 200, 100
    )
    second = module._submit_target_volume(
        "2026-07-15", "test", "SZSE.300001", 0.004, 200, 100
    )

    assert first is True
    assert second is False
    assert calls == [("SZSE.300001", 200)]
    assert module._previous_volumes == {"SZSE.300001": 100}
    assert module._pending_target_orders["SZSE.300001"]["target_shares"] == 200


def test_tick_slices_advance_only_after_fill_confirmation(monkeypatch):
    module = _load_main()
    monkeypatch.setattr(module, "_write_audit", lambda *args, **kwargs: None)
    module.RUN_MODE = "LIVE"
    module.VERBOSE_ORDERS = False
    module.EXEC_SCAN_INTERVAL_SECONDS = 0
    module.EXEC_MAX_CHILD_FRACTION = 0.25
    module.EXEC_START = "09:31:00"
    module.EXEC_FORCE_TIME = "14:45:00"
    module.EXEC_END = "14:55:00"
    module._execution_date = "2026-07-15"
    module._execution_targets = {
        "SZSE.300001": {
            "target_shares": 300,
            "target_weight": 0.004,
            "price_ref_close": 10.0,
            "source": "test",
        }
    }
    module._previous_symbols = {"SZSE.300001"}
    module._previous_volumes = {"SZSE.300001": 100}
    module._pending_target_orders = {}
    module._last_exec_scan_ts = {}
    module._order_events = []
    module._forbidden_clear_pending = set()
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(module, "_record_submission", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_order_target_volume_safe",
        lambda symbol, shares: calls.append((symbol, shares)),
    )
    context = type("Context", (), {"now": pd.Timestamp("2026-07-15 10:00:00")})()
    tick = {"symbol": "SZSE.300001", "price": 10.0}

    module._process_tick_execution(context, "tick", tick)
    module._process_tick_execution(context, "tick", tick)

    assert calls == [("SZSE.300001", 200)]
    assert module._previous_volumes["SZSE.300001"] == 100
    module.on_order_status(
        context,
        {
            "status": 3,
            "symbol": "SZSE.300001",
            "side": 1,
            "target_volume": 200,
            "filled_volume": 100,
        },
    )
    assert module._previous_volumes["SZSE.300001"] == 200
    assert "SZSE.300001" not in module._pending_target_orders
    assert "SZSE.300001" in module._execution_targets

    context.now = pd.Timestamp("2026-07-15 10:01:00")
    module._process_tick_execution(context, "tick", tick)
    assert calls == [("SZSE.300001", 200), ("SZSE.300001", 300)]
    assert module._previous_volumes["SZSE.300001"] == 200

    module.on_order_status(
        context,
        {
            "status": 3,
            "symbol": "SZSE.300001",
            "side": 1,
            "target_volume": 300,
            "filled_volume": 100,
        },
    )
    assert module._previous_volumes["SZSE.300001"] == 300
    assert module._execution_targets == {}
    assert module._execution_date is None


def test_rejected_tick_target_stops_plan_and_resyncs_positions(monkeypatch):
    module = _load_main()
    module.RUN_MODE = "LIVE"
    module.PAUSE_BUYS_ON_SELL_REJECT = False
    module.VERBOSE_ORDERS = False
    module._previous_volumes = {"SZSE.300001": 300}
    module._previous_symbols = {"SZSE.300001"}
    module._pending_target_orders = {
        "SZSE.300001": {"target_shares": 200, "previous_shares": 300}
    }
    module._execution_targets = {
        "SZSE.300001": {"target_shares": 100, "target_weight": 0.004}
    }
    module._execution_date = "2026-07-15"
    module._forbidden_clear_pending = set()
    module._order_events = []
    sync_calls: list[bool] = []
    monkeypatch.setattr(module, "_write_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_sync_existing_positions",
        lambda context: sync_calls.append(True),
    )
    context = type("Context", (), {"now": pd.Timestamp("2026-07-15 10:00:00")})()

    module.on_order_status(
        context,
        {
            "status": 8,
            "symbol": "SZSE.300001",
            "side": 2,
            "target_volume": 200,
            "filled_volume": 0,
            "ord_rej_reason_detail": "limit down",
        },
    )

    assert sync_calls == [True]
    assert module._pending_target_orders == {}
    assert module._execution_targets == {}
    assert module._execution_date is None
    assert module._previous_volumes == {"SZSE.300001": 300}


@pytest.mark.parametrize(
    ("function_name", "api_name", "argument_name", "argument_value"),
    [
        ("_order_target_volume_safe", "order_target_volume", "volume", 100),
        ("_order_target_percent_safe", "order_target_percent", "percent", 0.01),
    ],
)
def test_required_account_never_falls_back_after_sdk_type_error(
    monkeypatch,
    function_name,
    api_name,
    argument_name,
    argument_value,
):
    module = _load_main()
    module.GM_ACCOUNT_ID = "paper-account"
    module.REQUIRE_ACCOUNT_ID = True
    calls: list[dict] = []

    def reject(**kwargs):
        calls.append(dict(kwargs))
        raise TypeError("unsupported keyword")

    monkeypatch.setattr(module, api_name, reject)

    with pytest.raises(RuntimeError, match="refusing terminal-default-account fallback"):
        getattr(module, function_name)("SZSE.300001", argument_value)

    assert len(calls) == 2
    assert all(call["account"] == "paper-account" for call in calls)
    assert calls[0][argument_name] == argument_value
    assert calls[1][argument_name] == argument_value


def test_tick_execution_waits_for_market_price_before_force_window(monkeypatch):
    module = _load_main()
    module.RUN_MODE = "LIVE"
    module.VERBOSE_ORDERS = False
    module.EXEC_SCAN_INTERVAL_SECONDS = 0
    module.EXEC_START = "09:31:00"
    module.EXEC_FORCE_TIME = "14:45:00"
    module.EXEC_END = "14:55:00"
    module._execution_date = "2026-07-15"
    module._execution_targets = {
        "SZSE.300001": {
            "target_shares": 200,
            "target_weight": 0.004,
            "price_ref_close": 10.0,
            "source": "test",
        }
    }
    module._previous_volumes = {"SZSE.300001": 100}
    module._pending_target_orders = {}
    module._last_exec_scan_ts = {}
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(module, "_record_submission", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_order_target_volume_safe",
        lambda symbol, shares: calls.append((symbol, shares)),
    )

    morning = type("Context", (), {"now": pd.Timestamp("2026-07-15 09:31:00")})()
    module._process_tick_execution(morning, "schedule", tick=None)
    assert calls == []

    force = type("Context", (), {"now": pd.Timestamp("2026-07-15 14:45:00")})()
    module._process_tick_execution(force, "force", tick=None)
    assert calls == [("SZSE.300001", 200)]


def test_tick_subscription_prioritizes_reductions_then_largest_buys():
    module = _load_main()
    module._all_target_dates = {"2026-07-15"}
    module._targets_by_date = {
        "2026-07-15": {
            "SZSE.300001": {"target_shares": 100, "price_ref_close": 10.0},
            "SZSE.300002": {"target_shares": 500, "price_ref_close": 20.0},
            "SZSE.300003": {"target_shares": 200, "price_ref_close": 5.0},
            "SZSE.300004": {"target_shares": 100, "price_ref_close": 8.0},
        }
    }
    module._previous_symbols = {"SZSE.300001", "SZSE.300003", "SZSE.300004"}
    module._previous_volumes = {
        "SZSE.300001": 300,
        "SZSE.300003": 100,
        "SZSE.300004": 100,
    }

    ordered = module._ordered_subscription_symbols()

    assert ordered == [
        "SZSE.300001",
        "SZSE.300002",
        "SZSE.300003",
        "SZSE.300004",
    ]


def test_live_audit_flush_is_atomic_and_preserves_runtime_state(tmp_path):
    module = _load_main()
    module.RUN_MODE = "LIVE"
    module.AUDIT_DIR = tmp_path
    module.AUDIT_RUN_ID = "paper_run"
    module._submission_events = [
        {
            "trade_date": "2026-07-15",
            "source": "test",
            "action": "target",
            "symbol": "SZSE.300001",
            "target_percent": 0.004,
            "target_volume": 100,
        }
    ]
    module._order_events = []
    module._rebalance_events = []
    module._execution_events = []
    module._pending_target_orders = {
        "SZSE.300001": {"target_shares": 100, "previous_shares": 0}
    }

    module._write_audit()

    run_dir = tmp_path / "paper_run"
    assert (run_dir / "submissions.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert not list(run_dir.glob("*.pending"))
    summary = pd.read_json(run_dir / "summary.json", typ="series")
    assert summary["submitted_orders"] == 1
    assert summary["pending_target_order_symbols_at_finish"] == ["SZSE.300001"]
