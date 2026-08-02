from __future__ import annotations

from helki_quant.catboost_densemble import CatBoostDEnsemble


def test_legacy_multiclass_state_restores_prediction_flags() -> None:
    model = CatBoostDEnsemble.__new__(CatBoostDEnsemble)
    model.__setstate__({"loss": "MultiClass", "ensemble": [object()]})

    assert model._is_multiclass is True
    assert model._is_binary is False
    assert model._is_cls is True


def test_legacy_regression_state_restores_prediction_flags() -> None:
    model = CatBoostDEnsemble.__new__(CatBoostDEnsemble)
    model.__setstate__({"loss": "RMSE", "ensemble": [object()]})

    assert model._is_multiclass is False
    assert model._is_binary is False
    assert model._is_cls is False
