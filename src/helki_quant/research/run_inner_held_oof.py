from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from ruamel.yaml import YAML

import qlib
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config


HERE = Path(__file__).resolve().parent
MULTI_LAYER = HERE.parent
INTRADAY = MULTI_LAYER.parent / "intraday_t"
for path in (MULTI_LAYER, INTRADAY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from realtime_output import log_step, setup_realtime_output


def load_yaml(path: Path) -> dict:
    return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))


def normalize_instrument(value: object) -> str:
    raw = str(value).upper()
    if "." in raw and raw.startswith(("SZSE.", "SHSE.")):
        code = raw.split(".", 1)[1]
        return ("SH" if raw.startswith("SHSE.") else "SZ") + code
    return raw


def load_held_context(holdings_path: Path, targets_path: Path | None) -> tuple[set[str], pd.DataFrame]:
    holdings = pd.read_csv(holdings_path, parse_dates=["trade_date"])
    holdings["datetime"] = holdings["trade_date"].dt.normalize()
    holdings["instrument"] = holdings["instrument"].map(normalize_instrument)
    holdings = holdings[holdings["shares"] > 0].copy()
    held_dates = holdings[["datetime", "instrument", "shares", "weight"]].rename(
        columns={"shares": "held_shares", "weight": "held_weight"}
    )
    universe = set(held_dates["instrument"].unique())

    if targets_path is not None and targets_path.exists():
        targets = pd.read_csv(targets_path)
        symbol_col = "instrument" if "instrument" in targets.columns else "symbol"
        targets["instrument"] = targets[symbol_col].map(normalize_instrument)
        universe.update(targets["instrument"].dropna().unique())

    held_dates = held_dates.drop_duplicates(["datetime", "instrument"], keep="last")
    return universe, held_dates


def flatten_prepared(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("expected DatasetH.prepare(..., col_set=['feature', 'label']) to return MultiIndex columns")
    feature = frame["feature"].copy()
    label = frame["label"]
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]
    label = label.rename("label")
    out = pd.concat([feature, label], axis=1).replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["label"])
    feature_cols = [col for col in out.columns if col != "label"]
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out[feature_cols], out["label"]


def load_extra_context(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path, parse_dates=["datetime"])
    frame["datetime"] = frame["datetime"].dt.normalize()
    frame["instrument"] = frame["instrument"].map(normalize_instrument)
    return frame.drop_duplicates(["datetime", "instrument"], keep="last")


def attach_held_context(
    feature: pd.DataFrame,
    label: pd.Series,
    held_universe: set[str],
    held_dates: pd.DataFrame,
    extra_context: pd.DataFrame | None,
    context_feature_cols: list[str],
    context_label_col: str | None,
) -> pd.DataFrame:
    frame = feature.copy()
    frame["label"] = label
    idx = frame.index.to_frame(index=False)
    idx["datetime"] = pd.to_datetime(idx["datetime"]).dt.normalize()
    idx["instrument"] = idx["instrument"].map(normalize_instrument)
    frame = frame.reset_index(drop=True)
    frame.insert(0, "datetime", idx["datetime"].to_numpy())
    frame.insert(1, "instrument", idx["instrument"].to_numpy())
    frame["held_universe"] = frame["instrument"].isin(held_universe)
    frame = frame.merge(held_dates, on=["datetime", "instrument"], how="left")
    frame["held_date"] = frame["held_shares"].fillna(0.0) > 0
    frame["held_weight"] = frame["held_weight"].fillna(0.0)
    frame["held_shares"] = frame["held_shares"].fillna(0.0)
    if extra_context is not None:
        keep_cols = ["datetime", "instrument"] + [
            col
            for col in context_feature_cols + ([context_label_col] if context_label_col else [])
            if col and col in extra_context.columns
        ]
        context = extra_context[keep_cols].copy()
        frame = frame.merge(context, on=["datetime", "instrument"], how="left")
        for col in context_feature_cols:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if context_label_col and context_label_col in frame.columns:
            frame["label"] = pd.to_numeric(frame[context_label_col], errors="coerce")
    return frame


def make_weights(
    frame: pd.DataFrame,
    mode: str,
    universe_weight: float,
    date_weight: float,
) -> pd.Series:
    weights = pd.Series(1.0, index=frame.index, dtype=float)
    if mode in {"held_weighted", "held_universe_only"}:
        weights.loc[frame["held_universe"]] += universe_weight
    if mode == "held_weighted":
        weights.loc[frame["held_date"]] += date_weight
    return weights.clip(lower=1.0)


def filter_mode(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "held_universe_only":
        return frame[frame["held_universe"]].copy()
    if mode == "held_only":
        return frame[frame["held_date"]].copy()
    return frame.copy()


def label_threshold(model_cfg: dict) -> float:
    thresholds = model_cfg["model"].get("kwargs", {}).get("thresholds") or [0.0]
    return float(thresholds[0])


def catboost_params(model_cfg: dict, fit_params: dict, verbose_eval: int) -> dict:
    kwargs = copy.deepcopy(model_cfg["model"].get("kwargs", {}))
    for key in ("num_classes", "thresholds"):
        kwargs.pop(key, None)
    kwargs["loss_function"] = "Logloss"
    kwargs.setdefault("eval_metric", "AUC")
    kwargs.setdefault("task_type", "CPU")
    kwargs.setdefault("allow_writing_files", False)
    kwargs["iterations"] = int(fit_params.get("num_boost_round", 220))
    kwargs["early_stopping_rounds"] = int(fit_params.get("early_stopping_rounds", 40))
    kwargs.setdefault("verbose", int(fit_params.get("verbose_eval", verbose_eval)))
    return kwargs


def restore_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[["datetime", "instrument"]], names=["datetime", "instrument"])


def train_fold(
    config_path: Path,
    fold: int,
    variant: str,
    output_dir: Path,
    holdings_path: Path,
    targets_path: Path | None,
    held_context_path: Path | None,
    context_feature_cols: list[str],
    context_label_col: str | None,
    label_threshold_override: float | None,
    mode: str,
    universe_weight: float,
    date_weight: float,
    min_train_rows: int,
    verbose_eval: int,
) -> Path:
    setup_realtime_output()
    started = time.time()
    cfg = load_yaml(config_path)
    log_step(f"[held-oof] init qlib fold={fold} config={config_path}")
    qlib.init(**cfg.get("qlib_init_inner", cfg["qlib_init"]))

    model_cfg = cfg["inner_model"]
    held_universe, held_dates = load_held_context(holdings_path, targets_path)
    extra_context = load_extra_context(held_context_path)
    dataset = init_instance_by_config(model_cfg["dataset"])
    log_step(f"[held-oof] dataset.prepare fold={fold}")
    df_train, df_valid, df_test = dataset.prepare(
        ["train", "valid", "test"],
        col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_L,
    )
    x_train, y_train = flatten_prepared(df_train)
    x_valid, y_valid = flatten_prepared(df_valid)
    x_test, y_test = flatten_prepared(df_test)

    train = attach_held_context(
        x_train, y_train, held_universe, held_dates, extra_context, context_feature_cols, context_label_col
    )
    valid = attach_held_context(
        x_valid, y_valid, held_universe, held_dates, extra_context, context_feature_cols, context_label_col
    )
    test_idx = x_test.index
    train_f = filter_mode(train, mode)
    valid_f = filter_mode(valid, mode)
    if len(train_f) < min_train_rows or len(valid_f) < max(100, min_train_rows // 5):
        raise ValueError(
            f"held-aware sample too small fold={fold} mode={mode} "
            f"train={len(train_f)} valid={len(valid_f)} min_train_rows={min_train_rows}"
        )

    if context_label_col:
        train_f = train_f.dropna(subset=["label"]).copy()
        valid_f = valid_f.dropna(subset=["label"]).copy()
        if len(train_f) < min_train_rows or len(valid_f) < max(100, min_train_rows // 5):
            raise ValueError(
                f"context-label sample too small fold={fold} mode={mode} label={context_label_col} "
                f"train={len(train_f)} valid={len(valid_f)} min_train_rows={min_train_rows}"
            )

    feature_cols = [
        col
        for col in train_f.columns
        if col
        not in {
            "datetime",
            "instrument",
            "label",
            "held_universe",
            "held_date",
            "held_shares",
            context_label_col,
        }
    ]
    threshold = label_threshold_override if label_threshold_override is not None else label_threshold(model_cfg)
    y_train_cls = (train_f["label"].to_numpy() > threshold).astype(int)
    y_valid_cls = (valid_f["label"].to_numpy() > threshold).astype(int)
    train_weight = make_weights(train_f, mode, universe_weight, date_weight)
    valid_weight = make_weights(valid_f, mode, universe_weight, date_weight)

    params = catboost_params(model_cfg, model_cfg.get("fit_params", {}), verbose_eval)
    log_step(
        "[held-oof] fit start "
        f"fold={fold} mode={mode} train={len(train_f)} valid={len(valid_f)} "
        f"held_universe={int(train_f['held_universe'].sum())} held_date={int(train_f['held_date'].sum())}"
    )
    model = CatBoostClassifier(**params)
    model.fit(
        Pool(train_f[feature_cols], label=y_train_cls, weight=train_weight),
        eval_set=Pool(valid_f[feature_cols], label=y_valid_cls, weight=valid_weight),
        use_best_model=True,
    )
    log_step(f"[held-oof] fit done fold={fold} best_iter={model.get_best_iteration()}")
    test = attach_held_context(
        x_test, y_test, held_universe, held_dates, extra_context, context_feature_cols, context_label_col
    )
    for col in feature_cols:
        if col not in test.columns:
            test[col] = 0.0
    test[feature_cols] = test[feature_cols].fillna(0.0)
    pred = model.predict_proba(test[feature_cols])[:, 1]
    prediction = pd.Series(pred, index=test_idx, name="inner")

    layer_dir = output_dir / "oof" / variant / "inner"
    layer_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = layer_dir / f"fold_{fold:02d}.csv"
    prediction.to_csv(prediction_path)
    model_path = layer_dir / f"fold_{fold:02d}_model.pkl"
    joblib.dump(model, model_path)

    meta = {
        "status": "held_position_aware_inner_oof_prediction",
        "fold": fold,
        "variant": variant,
        "config": str(config_path.resolve()),
        "holdings_path": str(holdings_path.resolve()),
        "targets_path": str(targets_path.resolve()) if targets_path else None,
        "held_context_path": str(held_context_path.resolve()) if held_context_path else None,
        "context_feature_cols": context_feature_cols,
        "context_label_col": context_label_col,
        "mode": mode,
        "universe_weight": float(universe_weight),
        "date_weight": float(date_weight),
        "held_universe_size": int(len(held_universe)),
        "train_rows": int(len(train_f)),
        "valid_rows": int(len(valid_f)),
        "test_rows": int(len(prediction)),
        "train_held_universe_rows": int(train_f["held_universe"].sum()),
        "train_held_date_rows": int(train_f["held_date"].sum()),
        "valid_held_universe_rows": int(valid_f["held_universe"].sum()),
        "valid_held_date_rows": int(valid_f["held_date"].sum()),
        "label_threshold": threshold,
        "prediction_start": str(prediction.index.get_level_values("datetime").min()),
        "prediction_end": str(prediction.index.get_level_values("datetime").max()),
        "elapsed_seconds": time.time() - started,
        "deployment_allowed": False,
        "research_only_reason": "Inner T+0 is held-position-aware research only and is not connected to main.py or GmQuant PAPER entrypoints.",
    }
    prediction_path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_step(f"[held-oof] saved prediction={prediction_path}")
    return prediction_path


def parse_folds(raw: str) -> list[int]:
    folds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            folds.extend(range(int(start), int(end) + 1))
        else:
            folds.append(int(part))
    folds = sorted(set(folds))
    bad = [fold for fold in folds if fold < 1 or fold > 6]
    if bad:
        raise ValueError(f"folds must be in 1..6, got {bad}")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--holdings", required=True)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--held-context", default=None)
    parser.add_argument(
        "--context-feature-cols",
        nargs="*",
        default=[],
        help="Optional held-position context columns to merge as extra model features.",
    )
    parser.add_argument(
        "--context-label-col",
        default=None,
        help="Optional held-position context column to use as the training label.",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=None,
        help="Optional classification threshold overriding the config threshold.",
    )
    parser.add_argument("--folds", default="1-6")
    parser.add_argument(
        "--mode",
        choices=["held_weighted", "held_universe_only", "held_only"],
        default="held_weighted",
    )
    parser.add_argument("--universe-weight", type=float, default=2.0)
    parser.add_argument("--date-weight", type=float, default=8.0)
    parser.add_argument("--min-train-rows", type=int, default=5000)
    parser.add_argument("--verbose-eval", type=int, default=50)
    parser.add_argument("--output-dir", default=str(HERE / "outputs"))
    args = parser.parse_args()

    config_dir = Path(args.config_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    holdings = Path(args.holdings).resolve()
    targets = Path(args.targets).resolve() if args.targets else None
    held_context = Path(args.held_context).resolve() if args.held_context else None
    for fold in parse_folds(args.folds):
        config_path = config_dir / f"fold_{fold:02d}" / "simple.yaml"
        train_fold(
            config_path=config_path,
            fold=fold,
            variant=args.variant,
            output_dir=output_dir,
            holdings_path=holdings,
            targets_path=targets,
            held_context_path=held_context,
            context_feature_cols=args.context_feature_cols,
            context_label_col=args.context_label_col,
            label_threshold_override=args.label_threshold,
            mode=args.mode,
            universe_weight=args.universe_weight,
            date_weight=args.date_weight,
            min_train_rows=args.min_train_rows,
            verbose_eval=args.verbose_eval,
        )


if __name__ == "__main__":
    main()
