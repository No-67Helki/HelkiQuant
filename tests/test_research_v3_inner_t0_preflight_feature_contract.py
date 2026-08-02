from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "src" / "helki_quant" / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from preflight_inner_t0_bidirectional_dryrun_candidate import validate_models  # noqa: E402


def _write_candidate(candidate: Path, *, mismatch_sell: bool = False) -> None:
    features = [f"live_feature_{index}" for index in range(149)]
    models = {}
    for direction, threshold, top_n, trigger in (
        ("buy_first", 0.925, 2, 0.006),
        ("sell_first", 0.975, 1, 0.0075),
    ):
        model_name = f"{direction}.cbm"
        calibration_name = f"{direction}.npy"
        meta_name = f"{direction}.json"
        (candidate / model_name).write_bytes(b"model")
        np.save(candidate / calibration_name, np.linspace(0.0, 1.0, 100))
        meta_features = features[:-1] if mismatch_sell and direction == "sell_first" else features
        (candidate / meta_name).write_text(
            json.dumps(
                {
                    "deployment_allowed": False,
                    "feature_mode": "live",
                    "feature_cols": meta_features,
                    "score_threshold": threshold,
                    "daily_top_n": top_n,
                    "trigger_distance": trigger,
                }
            ),
            encoding="utf-8",
        )
        models[direction] = {
            "model_path": model_name,
            "calibration_path": calibration_name,
            "meta_path": meta_name,
            "feature_cols": features,
        }
    (candidate / "frozen_models_manifest.json").write_text(
        json.dumps(
            {
                "deployment_allowed": False,
                "runtime_intent_only": True,
                "models": models,
            }
        ),
        encoding="utf-8",
    )


def test_preflight_accepts_shared_149_feature_contract(tmp_path: Path) -> None:
    _write_candidate(tmp_path)
    report = validate_models(tmp_path)
    assert report["passed"] is True
    assert report["checks"]["shared_feature_contract"] is True


def test_preflight_rejects_manifest_meta_feature_mismatch(tmp_path: Path) -> None:
    _write_candidate(tmp_path, mismatch_sell=True)
    report = validate_models(tmp_path)
    assert report["passed"] is False
    assert report["details"]["sell_first"]["checks"]["feature_contract_matches_meta"] is False
