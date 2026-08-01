from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


RESEARCH = Path(__file__).resolve().parents[1] / "src" / "helki_quant" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from build_inner_pool_from_production_logs import read_ranked_candidates  # noqa: E402


def test_unmapped_selected_targets_are_kept_in_inner_pool(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "trade_date": "2026-05-22",
                "instrument": "SZ300001",
                "shares": 100,
                "mark_price": 10.0,
            }
        ]
    ).to_csv(tmp_path / "holdings.csv", index=False)
    pd.DataFrame(
        [
            {
                "trade_date": "2026-05-22",
                "instrument": "SZ300001",
                "rank": 1,
                "mapped": 1,
                "target_shares": 100,
            },
            {
                "trade_date": "2026-05-22",
                "instrument": "SZ300002",
                "rank": 2,
                "mapped": 0,
                "target_shares": 0,
            },
        ]
    ).to_csv(tmp_path / "targets.csv", index=False)

    candidates = read_ranked_candidates(tmp_path)

    assert set(candidates) == {"SZ300001", "SZ300002"}
