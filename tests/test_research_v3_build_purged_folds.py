from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from build_purged_folds import build_purged_folds  # noqa: E402
from evaluate_outer_regime_oof import (  # noqa: E402
    discover_folds,
    load_fold_frame_from_daily_labels,
)
from training_frame_utils import collapse_identical_datetime_rows  # noqa: E402


def test_build_purged_folds_writes_reproducible_windows(tmp_path: Path) -> None:
    provider = tmp_path / "provider"
    calendar = provider / "calendars"
    calendar.mkdir(parents=True)
    dates = pd.bdate_range("2024-01-02", periods=120)
    (calendar / "day.txt").write_text(
        "\n".join(str(date.date()) for date in dates) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "folds.json"

    manifest = build_purged_folds(
        provider,
        output,
        start="2024-01-02",
        end=str(dates[-1].date()),
        min_train_days=20,
        valid_days=10,
        test_days=5,
        step_days=5,
        purge_days=2,
        embargo_days=1,
    )

    folds = json.loads(output.read_text(encoding="utf-8"))
    first = folds[0]
    assert manifest["fold_count"] == len(folds)
    assert first["train_start"] == "2024-01-02"
    assert pd.Timestamp(first["train_end"]) < pd.Timestamp(first["valid_start"])
    assert pd.Timestamp(first["valid_end"]) < pd.Timestamp(first["test_start"])
    assert output.with_name("folds.manifest.json").exists()


def test_build_purged_folds_fails_when_windows_do_not_fit(tmp_path: Path) -> None:
    provider = tmp_path / "provider"
    calendar = provider / "calendars"
    calendar.mkdir(parents=True)
    dates = pd.bdate_range("2024-01-02", periods=20)
    (calendar / "day.txt").write_text(
        "\n".join(str(date.date()) for date in dates) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no folds fit"):
        build_purged_folds(
            provider,
            tmp_path / "folds.json",
            start="2024-01-02",
            end=str(dates[-1].date()),
            min_train_days=20,
            valid_days=10,
            test_days=5,
            step_days=5,
            purge_days=2,
            embargo_days=1,
        )


def test_outer_evaluator_discovers_config_prediction_intersection(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for fold in (1, 2, 4):
        fold_dir = configs / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "simple.yaml").write_text("{}\n", encoding="utf-8")
    for fold in (1, 2, 3):
        (predictions / f"fold_{fold:02d}.csv").write_text("x\n", encoding="utf-8")

    assert discover_folds(configs, predictions, "simple.yaml") == [1, 2]


def _duplicated_outer_frame(different: bool = False) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["datetime", "instrument"],
    )
    features = [1.0, 1.0, 2.0, 2.0]
    if different:
        features[1] = 1.5
    return pd.DataFrame(
        {
            ("feature", "market_ret"): features,
            ("label", "adverse"): [0.0, 0.0, 1.0, 1.0],
        },
        index=index,
    )


def test_datetime_collapse_keeps_one_identical_market_row_per_day() -> None:
    collapsed = collapse_identical_datetime_rows(
        _duplicated_outer_frame(), context="test"
    )

    assert len(collapsed) == 2
    assert collapsed.index.name == "datetime"
    assert collapsed[("feature", "market_ret")].tolist() == [1.0, 2.0]


def test_datetime_collapse_rejects_stock_specific_features() -> None:
    with pytest.raises(ValueError, match="not identical"):
        collapse_identical_datetime_rows(
            _duplicated_outer_frame(different=True), context="test"
        )


def test_outer_fast_label_loader_aggregates_stock_predictions_by_day(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "fold_01.csv"
    predictions.write_text(
        "datetime,instrument,outer\n"
        "2024-01-02,A,0.2\n"
        "2024-01-02,B,0.4\n"
        "2024-01-03,A,0.7\n"
        "2024-01-03,B,0.9\n",
        encoding="utf-8",
    )
    labels = pd.Series(
        [0.0, 1.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    frame = load_fold_frame_from_daily_labels(predictions, labels)

    assert frame["score"].tolist() == pytest.approx([0.3, 0.8])
    assert frame["label"].tolist() == [0, 1]
