from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from ruamel.yaml import YAML

RESEARCH = Path(__file__).resolve().parents[1] / "src" / "helki_quant" / "research"
sys.path.insert(0, str(RESEARCH))

from build_paper_forward_config import build  # noqa: E402
from prepare_outer_middle_paper_candidate import validate_dates, validate_prediction  # noqa: E402


def test_forward_config_overrides_provider_and_segments(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    output = tmp_path / "forward.yaml"
    provider = tmp_path / "provider"
    provider.mkdir()
    base.write_text(
        """
qlib_init:
  provider_uri:
    day: old-provider
outer_model:
  dataset:
    kwargs:
      handler:
        kwargs: {}
      segments: {}
segments: {}
backtest: {}
""".strip(),
        encoding="utf-8",
    )
    build(
        base,
        output,
        "2026-04-01",
        "2026-04-02",
        "2026-05-29",
        "2026-06-05",
        provider,
    )
    config = YAML(typ="safe", pure=True).load(output.read_text(encoding="utf-8"))
    assert config["qlib_init"]["provider_uri"]["day"] == str(provider.resolve())
    assert config["segments"]["test"] == ["2026-06-05", "2026-06-05"]
    handler = config["outer_model"]["dataset"]["kwargs"]["handler"]["kwargs"]
    assert handler["fit_end_time"] == "2026-04-01"


def test_prepare_dates_fail_closed() -> None:
    validate_dates(
        "2026-04-01",
        "2026-04-02",
        "2026-05-29",
        "2026-06-05",
        "2026-06-12",
    )
    try:
        validate_dates(
            "2026-04-01",
            "2026-04-02",
            "2026-06-05",
            "2026-06-05",
            "2026-06-12",
        )
    except ValueError as exc:
        assert "required ordering" in str(exc)
    else:
        raise AssertionError("overlapping validation/test dates must fail")


def test_prediction_must_cover_exact_signal_date(tmp_path: Path) -> None:
    prediction = tmp_path / "middle.csv"
    pd.DataFrame(
        {
            "datetime": ["2026-06-05", "2026-06-05"],
            "middle": [0.1, 0.2],
        }
    ).to_csv(prediction, index=False)
    report = validate_prediction(prediction, "middle", "2026-06-05")
    assert report["finite_rows"] == 2
    try:
        validate_prediction(prediction, "middle", "2026-06-06")
    except ValueError as exc:
        assert "no finite middle prediction" in str(exc)
    else:
        raise AssertionError("missing signal-date prediction must fail")
