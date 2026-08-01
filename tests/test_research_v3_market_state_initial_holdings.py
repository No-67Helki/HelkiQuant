from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


RESEARCH = Path(__file__).resolve().parents[1] / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

import filter_gm_targets_market_state as market_filter  # noqa: E402


def test_market_filter_classifies_carried_target_as_hold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous = tmp_path / "previous.csv"
    target = tmp_path / "target.csv"
    output = tmp_path / "filtered.csv"
    base = {
        "symbol": ["SZSE.300001"],
        "instrument": ["SZ300001"],
        "rank": [1],
        "target_weight": [0.004],
        "target_shares": [300],
        "signal_date": ["2026-06-10"],
    }
    pd.DataFrame({**base, "trade_date": ["2026-06-11"]}).to_csv(
        previous, index=False
    )
    pd.DataFrame({**base, "trade_date": ["2026-06-12"]}).to_csv(
        target, index=False
    )
    monkeypatch.setattr(
        market_filter,
        "load_daily_state",
        lambda *args, **kwargs: ({}, []),
    )
    monkeypatch.setattr(
        market_filter,
        "load_gm_rejection_blocks",
        lambda *args, **kwargs: ({}, []),
    )

    report = market_filter.filter_targets(
        target,
        output,
        tmp_path,
        None,
        19.5,
        True,
        False,
        previous,
    )
    actions = pd.read_csv(output.with_suffix(".market_state_actions.csv"))
    filtered = pd.read_csv(output)

    assert report["initial_holding_count"] == 1
    assert actions["side"].tolist() == ["HOLD"]
    assert filtered["target_shares"].tolist() == [300]


def test_blocked_sell_is_carried_with_the_current_signal_date(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous = tmp_path / "previous.csv"
    target = tmp_path / "target.csv"
    output = tmp_path / "filtered.csv"
    pd.DataFrame(
        {
            "symbol": ["SZSE.300001"],
            "instrument": ["SZ300001"],
            "rank": [1],
            "target_weight": [0.004],
            "target_shares": [300],
            "signal_date": ["2026-06-10"],
            "trade_date": ["2026-06-11"],
        }
    ).to_csv(previous, index=False)
    pd.DataFrame(
        {
            "symbol": ["SZSE.300002"],
            "instrument": ["SZ300002"],
            "rank": [1],
            "target_weight": [0.004],
            "target_shares": [200],
            "signal_date": ["2026-06-11"],
            "trade_date": ["2026-06-12"],
        }
    ).to_csv(target, index=False)
    monkeypatch.setattr(
        market_filter,
        "load_daily_state",
        lambda *args, **kwargs: (
            {("2026-06-12", "SZSE.300001"): {"SELL"}},
            [],
        ),
    )
    monkeypatch.setattr(
        market_filter,
        "load_gm_rejection_blocks",
        lambda *args, **kwargs: ({}, []),
    )

    market_filter.filter_targets(
        target,
        output,
        tmp_path,
        None,
        19.5,
        True,
        False,
        previous,
    )
    filtered = pd.read_csv(output, dtype={"symbol": str})

    carried = filtered[filtered["symbol"].eq("SZSE.300001")].iloc[0]
    assert int(carried["target_shares"]) == 300
    assert carried["signal_date"] == "2026-06-11"
