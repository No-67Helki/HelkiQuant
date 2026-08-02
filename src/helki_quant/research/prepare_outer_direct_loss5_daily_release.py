from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from ruamel.yaml import YAML

from audit_target_transition import audit_target_transition
from audit_gm_target_csv import audit as audit_target
from build_forbidden_st_symbols import build as build_forbidden_symbols
from build_outer_middle_paper_launch_candidate import build_candidate, sha256_file
from build_paper_forward_config import build as build_forward_config
from build_synchronized_inner_shadow_release import (
    DEFAULT_MODEL_MANIFEST as DEFAULT_INNER_SHADOW_MODEL_MANIFEST,
    build_synchronized_shadow,
)
from export_paper_forward_gm_targets import export_targets
from filter_gm_targets_market_state import filter_targets


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
OUTPUTS = HERE / "outputs"

PROFILE = "c_outer_loss5_top150_rb20_risk0.60_floor0.30_cap0.30_nost"
CONTRACT: dict[str, Any] = {
    "middle_model": "canonical_densemble",
    "outer_model": "broad_adverse_loss5_20d",
    "outer_prediction_required": True,
    "top_k": 150,
    "rebalance_every": 20,
    "buffer_multiple": 2,
    "base_risk_budget": 0.60,
    "outer_risk_threshold": 0.50,
    "outer_risk_floor": 0.30,
    "industry_cap": 0.30,
    "allocation_mode": "fixed_topk",
    "initial_cash": 1_000_000.0,
}
DEFAULT_BASE_CANDIDATE = (
    HERE / "runtime_templates" / "outer_middle_paper_v5"
)
DEFAULT_MIDDLE_BASE = (
    OUTPUTS / "fold_configs_canonical_20260605" / "fold_06" / "densemble.yaml"
)
DEFAULT_MIDDLE_WHITELIST = (
    OUTPUTS
    / "factor_reports"
    / "canonical_20260605_densemble"
    / "fold_06"
    / "feature_whitelist_middle_v2.json"
)
DEFAULT_OUTER_BASE = (
    OUTPUTS
    / "outer_regime_fold_configs_broad_adverse_loss5_20d_v2_20260609"
    / "fold_06"
    / "simple.yaml"
)
DEFAULT_HISTORICAL_GATE = (
    OUTPUTS
    / "outer_direct_loss5_v2_shift1_market_filtered_target_replay_"
    "paper_gate_validation_1m_20260609.json"
)
DEFAULT_ACCOUNT_ID = os.environ.get("GM_ACCOUNT_ID", "").strip()
PURGE_DAYS = 21
EMBARGO_DAYS = 5
VALID_DAYS = 120


def log(message: str) -> None:
    print(f"[outer-direct daily release] {message}", flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _required_path(path: Path, label: str, *, directory: bool | None = None) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if directory is True and not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    if directory is False and not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def _resolve_artifact(raw: object, relative_to: Path) -> Path:
    path = Path(str(raw or ""))
    return (path if path.is_absolute() else relative_to / path).resolve()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path.resolve()),
        "bytes": int(path.stat().st_size),
    }


def validate_dates(
    *,
    signal_date: str,
    trade_date: str,
    as_of_date: str,
    train_end: str | None = None,
    valid_start: str | None = None,
    valid_end: str | None = None,
) -> None:
    signal = pd.Timestamp(signal_date).normalize()
    trade = pd.Timestamp(trade_date).normalize()
    as_of = pd.Timestamp(as_of_date).normalize()
    if not signal < trade:
        raise ValueError("signal_date must be earlier than trade_date")
    lag = int((trade - signal).days)
    if lag > 4:
        raise ValueError(f"signal-to-target lag exceeds 4 calendar days: {lag}")
    if as_of != trade:
        raise ValueError(
            "as_of_date must equal trade_date for a launch-specific versioned package"
        )
    training_dates = (train_end, valid_start, valid_end)
    if any(value is not None for value in training_dates):
        if not all(training_dates):
            raise ValueError("train_end, valid_start, and valid_end must be supplied together")
        train = pd.Timestamp(train_end).normalize()
        valid_from = pd.Timestamp(valid_start).normalize()
        valid_to = pd.Timestamp(valid_end).normalize()
        if not train < valid_from <= valid_to < signal:
            raise ValueError(
                "required training order: train_end < valid_start <= valid_end < signal_date"
            )


def shanghai_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Shanghai").normalize().tz_localize(None)


def validate_release_clock(
    *,
    signal_date: str,
    as_of_date: str,
    st_risk_refreshed_at: str,
    historical_smoke: bool,
    today: pd.Timestamp | None = None,
) -> None:
    if historical_smoke:
        return
    current = pd.Timestamp(today if today is not None else shanghai_today()).normalize()
    signal = pd.Timestamp(signal_date).normalize()
    as_of = pd.Timestamp(as_of_date).normalize()
    refresh = pd.Timestamp(st_risk_refreshed_at).normalize()
    forward_days = int((as_of - current).days)
    if forward_days < 0 or forward_days > 4:
        raise ValueError(
            f"live release target must be today or at most 4 days ahead: "
            f"today={current.date()} as_of={as_of.date()}"
        )
    signal_age = int((current - signal).days)
    if signal_age < 0 or signal_age > 4:
        raise ValueError(
            f"live release signal is not current: signal={signal.date()} today={current.date()}"
        )
    if refresh != current:
        raise ValueError(
            "live release must rebuild the static ST/delisting snapshot today: "
            f"refresh={refresh.date()} today={current.date()}"
        )


def prepare_group_metadata(
    source: Path,
    output: Path,
    signal_date: str,
) -> dict[str, Any]:
    source = _required_path(source, "group metadata source", directory=False)
    frame = pd.read_csv(source, dtype=str)
    required = {"instrument", "industry", "start_date", "end_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"group metadata missing columns: {sorted(missing)}")
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
    if frame[["start_date", "end_date"]].isna().any().any():
        raise ValueError("group metadata contains invalid interval dates")
    signal = pd.Timestamp(signal_date).normalize()
    latest_index = frame.sort_values(["instrument", "start_date"]).groupby(
        "instrument", sort=False
    ).tail(1).index
    extend_mask = frame.index.isin(latest_index) & frame["end_date"].lt(signal)
    extended = int(extend_mask.sum())
    frame.loc[extend_mask, "end_date"] = signal
    if "source" in frame.columns:
        frame.loc[extend_mask, "source"] = (
            frame.loc[extend_mask, "source"].astype(str)
            + "|risk_control_ffill_to_"
            + signal.strftime("%Y-%m-%d")
        )
    active = frame[
        frame["start_date"].le(signal) & frame["end_date"].ge(signal)
    ]
    if active["instrument"].nunique() < 1000:
        raise ValueError(
            f"group metadata coverage too small on {signal.date()}: "
            f"{active['instrument'].nunique()}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame["start_date"] = frame["start_date"].dt.strftime("%Y-%m-%d")
    frame["end_date"] = frame["end_date"].dt.strftime("%Y-%m-%d")
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return {
        "source": _artifact(source),
        "output": _artifact(output),
        "signal_date": signal.strftime("%Y-%m-%d"),
        "extended_latest_intervals": extended,
        "active_instruments": int(active["instrument"].nunique()),
        "policy": "forward-fill latest industry only for concentration/risk controls, never alpha",
    }


def prepare_forbidden_symbols(
    args: argparse.Namespace,
    stage: Path,
) -> tuple[Path, dict[str, Any]]:
    if args.historical_smoke:
        path = _required_path(
            args.forbidden_symbols,
            "historical forbidden symbols",
            directory=False,
        )
        return path, {
            "mode": "provided_historical_snapshot",
            "output": _artifact(path),
            "refreshed_at": args.st_risk_refreshed_at,
        }
    output = stage / "metadata" / f"forbidden_st_symbols_{args.signal_date.replace('-', '')}.csv"
    report_path = output.with_suffix(".report.json")
    stock_list = _required_path(args.stock_list, "stock list", directory=False)
    overrides = _required_path(
        args.forbidden_overrides, "forbidden overrides", directory=False
    )
    report = build_forbidden_symbols(
        stock_list,
        output,
        report_path,
        overrides,
        pit_market_state_path=(
            args.pit_market_state.resolve()
            if getattr(args, "pit_market_state", None)
            else None
        ),
        as_of_date=args.signal_date,
    )
    return output, {
        "mode": "rebuilt_for_release",
        "stock_list": _artifact(stock_list),
        "pit_market_state": (
            _artifact(args.pit_market_state.resolve())
            if getattr(args, "pit_market_state", None)
            else None
        ),
        "overrides": _artifact(overrides),
        "output": _artifact(output),
        "report": _artifact(report_path),
        "rows_forbidden": int(report["rows_forbidden"]),
        "refreshed_at": args.st_risk_refreshed_at,
    }


def validate_provider_calendar(provider: Path, signal_date: str, layer: str) -> dict[str, Any]:
    provider = _required_path(provider, f"{layer} provider", directory=True)
    calendar = _required_path(
        provider / "calendars" / "day.txt",
        f"{layer} provider day calendar",
        directory=False,
    )
    values = pd.to_datetime(
        [line.strip() for line in calendar.read_text(encoding="utf-8-sig").splitlines() if line.strip()],
        errors="coerce",
    )
    values = pd.DatetimeIndex(values).dropna().normalize()
    if values.empty:
        raise ValueError(f"{layer} provider calendar is empty: {calendar}")
    maximum = pd.Timestamp(values.max()).strftime("%Y-%m-%d")
    if maximum != pd.Timestamp(signal_date).strftime("%Y-%m-%d"):
        raise ValueError(
            f"{layer} provider max date {maximum} does not equal signal_date {signal_date}"
        )
    return {
        "provider": str(provider),
        "calendar": str(calendar),
        "sha256": sha256_file(calendar),
        "rows": int(len(values)),
        "min_date": pd.Timestamp(values.min()).strftime("%Y-%m-%d"),
        "max_date": maximum,
    }


def derive_forward_training_segments(
    provider: Path,
    signal_date: str,
    *,
    valid_days: int = VALID_DAYS,
    purge_days: int = PURGE_DAYS,
    embargo_days: int = EMBARGO_DAYS,
) -> dict[str, Any]:
    calendar_path = _required_path(
        provider.resolve() / "calendars" / "day.txt",
        "provider day calendar",
        directory=False,
    )
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            [
                line.strip()
                for line in calendar_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ],
            errors="coerce",
        )
    ).dropna().drop_duplicates().sort_values().normalize()
    signal = pd.Timestamp(signal_date).normalize()
    matches = calendar.get_indexer([signal])
    if len(matches) != 1 or int(matches[0]) < 0:
        raise ValueError(f"signal date is absent from provider calendar: {signal.date()}")
    test_pos = int(matches[0])
    valid_end_pos = test_pos - purge_days - embargo_days - 1
    valid_start_pos = valid_end_pos - valid_days + 1
    train_end_pos = valid_start_pos - purge_days - 1
    if train_end_pos < 0:
        raise ValueError("provider history is too short for frozen purged forward protocol")
    return {
        "train_end": calendar[train_end_pos].strftime("%Y-%m-%d"),
        "valid_start": calendar[valid_start_pos].strftime("%Y-%m-%d"),
        "valid_end": calendar[valid_end_pos].strftime("%Y-%m-%d"),
        "test_date": signal.strftime("%Y-%m-%d"),
        "valid_days": int(valid_days),
        "purge_days": int(purge_days),
        "embargo_days": int(embargo_days),
        "selection_policy": "calendar-derived; CLI dates cannot alter the frozen protocol",
    }


def derive_middle_rebalance_schedule(
    provider: Path,
    signal_date: str,
    *,
    previous_target: Path | None,
    initial_launch: bool,
    rebalance_every: int,
) -> dict[str, Any]:
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")
    calendar_path = _required_path(
        provider.resolve() / "calendars" / "day.txt",
        "provider day calendar",
        directory=False,
    )
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            [
                line.strip()
                for line in calendar_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ],
            errors="coerce",
        )
    ).dropna().drop_duplicates().sort_values().normalize()
    signal = pd.Timestamp(signal_date).normalize()
    signal_pos = int(calendar.get_indexer([signal])[0])
    if signal_pos < 0:
        raise ValueError(f"signal date is absent from provider calendar: {signal.date()}")

    if initial_launch:
        if previous_target is not None:
            raise ValueError("initial launch cannot use a previous target")
        previous_rebalance = None
        elapsed = 0
        due = True
        bootstrap = False
    else:
        if previous_target is None:
            raise ValueError("scheduled release requires a previous target")
        frame = pd.read_csv(
            previous_target.resolve(),
            dtype={"symbol": str, "instrument": str},
        )
        if frame.empty:
            raise ValueError("previous target is empty")
        if "middle_last_rebalance_signal_date" in frame.columns:
            values = (
                pd.to_datetime(
                    frame["middle_last_rebalance_signal_date"], errors="coerce"
                )
                .dropna()
                .dt.normalize()
                .unique()
            )
            if len(values) != 1:
                raise ValueError(
                    "previous target must contain one middle_last_rebalance_signal_date"
                )
            previous_rebalance = pd.Timestamp(values[0]).normalize()
            bootstrap = False
        else:
            if "signal_date" not in frame.columns:
                raise ValueError("legacy previous target is missing signal_date")
            values = (
                pd.to_datetime(frame["signal_date"], errors="coerce")
                .dropna()
                .dt.normalize()
                .unique()
            )
            if len(values) != 1:
                raise ValueError("legacy previous target must contain one signal_date")
            previous_rebalance = pd.Timestamp(values[0]).normalize()
            bootstrap = True
        previous_pos = int(calendar.get_indexer([previous_rebalance])[0])
        if previous_pos < 0:
            raise ValueError(
                "previous middle rebalance date is absent from provider calendar: "
                f"{previous_rebalance.date()}"
            )
        elapsed = signal_pos - previous_pos
        if elapsed <= 0:
            raise ValueError(
                "signal date must be after the previous middle rebalance date: "
                f"signal={signal.date()} previous={previous_rebalance.date()}"
            )
        due = elapsed >= rebalance_every

    effective_last = signal if due else previous_rebalance
    effective_pos = signal_pos if due else signal_pos - elapsed
    next_pos = effective_pos + rebalance_every
    next_date = calendar[next_pos].strftime("%Y-%m-%d") if next_pos < len(calendar) else None
    return {
        "policy": "provider_trading_sessions",
        "rebalance_every": int(rebalance_every),
        "due": bool(due),
        "signal_date": signal.strftime("%Y-%m-%d"),
        "previous_rebalance_signal_date": (
            previous_rebalance.strftime("%Y-%m-%d")
            if previous_rebalance is not None
            else None
        ),
        "last_rebalance_signal_date": effective_last.strftime("%Y-%m-%d"),
        "trading_sessions_elapsed_before_release": int(elapsed),
        "trading_sessions_since_rebalance": 0 if due else int(elapsed),
        "next_scheduled_rebalance_signal_date": next_date,
        "legacy_target_bootstrap": bool(bootstrap),
        "calendar": str(calendar_path.resolve()),
        "calendar_sha256": sha256_file(calendar_path),
    }


def validate_prediction(
    prediction: Path,
    *,
    layer: str,
    signal_date: str,
    provider: Path,
    expected_segments: dict[str, Any] | None = None,
    enforce_segments: bool = True,
) -> dict[str, Any]:
    prediction = _required_path(prediction, f"{layer} prediction", directory=False)
    frame = pd.read_csv(prediction)
    prediction_columns = {
        "middle": ("middle", "pred_middle"),
        "outer": ("outer", "pred_outer"),
    }
    if layer not in prediction_columns:
        raise ValueError(f"unsupported prediction layer: {layer}")
    if "datetime" not in frame.columns or "instrument" not in frame.columns:
        raise ValueError(f"{layer} prediction must contain datetime and instrument")
    column = next((name for name in prediction_columns[layer] if name in frame.columns), None)
    if column is None:
        raise ValueError(f"{layer} prediction column is missing")
    dates = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
    unique_dates = sorted(pd.Timestamp(value) for value in dates.dropna().unique())
    expected_date = pd.Timestamp(signal_date).normalize()
    if unique_dates != [expected_date]:
        raise ValueError(
            f"{layer} forward prediction must contain only {signal_date}: {unique_dates}"
        )
    values = pd.to_numeric(frame[column], errors="coerce")
    finite_rows = int(values.notna().sum())
    symbols = int(frame.loc[values.notna(), "instrument"].astype(str).nunique())
    if finite_rows < 20 or symbols < 20:
        raise ValueError(
            f"{layer} prediction coverage is too small: rows={finite_rows} symbols={symbols}"
        )

    metadata_path = _required_path(
        prediction.with_suffix(".json"),
        f"{layer} prediction metadata",
        directory=False,
    )
    metadata = _read_json(metadata_path)
    if metadata.get("layer") != layer:
        raise ValueError(f"{layer} prediction metadata layer mismatch")
    prediction_start = pd.to_datetime(
        metadata.get("test_start", metadata.get("prediction_start")), errors="coerce"
    )
    prediction_end = pd.to_datetime(
        metadata.get("test_end", metadata.get("prediction_end")), errors="coerce"
    )
    if (
        pd.isna(prediction_start)
        or pd.isna(prediction_end)
        or pd.Timestamp(prediction_start).normalize() != expected_date
        or pd.Timestamp(prediction_end).normalize() != expected_date
    ):
        raise ValueError(f"{layer} prediction metadata does not cover only {signal_date}")

    config = _required_path(
        _resolve_artifact(metadata.get("config"), metadata_path.parent),
        f"{layer} prediction config",
        directory=False,
    )
    config_payload = YAML(typ="safe", pure=True).load(config.read_text(encoding="utf-8"))
    provider_raw = config_payload.get("qlib_init", {}).get("provider_uri", {}).get("day")
    config_provider = _resolve_artifact(provider_raw, config.parent)
    if config_provider != provider.resolve():
        raise ValueError(
            f"{layer} prediction provider mismatch: config={config_provider} expected={provider.resolve()}"
        )
    model_class = (
        config_payload.get(f"{layer}_model", {}).get("model", {}).get("class")
    )
    expected_class = "CatBoostDEnsemble" if layer == "middle" else "CatBoostClsModel"
    if model_class != expected_class:
        raise ValueError(
            f"{layer} model class mismatch: observed={model_class!r} expected={expected_class!r}"
        )
    segments = config_payload.get("segments", {})
    observed_segments = {
        "train_end": str((segments.get("train") or [None, None])[-1]),
        "valid_start": str((segments.get("valid") or [None, None])[0]),
        "valid_end": str((segments.get("valid") or [None, None])[-1]),
        "test_date": str((segments.get("test") or [None, None])[0]),
    }
    segment_match = True
    if expected_segments is not None:
        segment_match = all(
            observed_segments.get(field) == str(expected_segments.get(field))
            for field in ("train_end", "valid_start", "valid_end", "test_date")
        )
        test_segment = segments.get("test") or []
        segment_match = bool(
            segment_match
            and len(test_segment) == 2
            and str(test_segment[0]) == str(test_segment[1]) == signal_date
        )
        if enforce_segments and not segment_match:
            raise ValueError(
                f"{layer} prediction config violates frozen purged protocol: "
                f"observed={observed_segments} expected={expected_segments}"
            )

    model_raw = metadata.get("model")
    model = (
        _resolve_artifact(model_raw, metadata_path.parent)
        if model_raw
        else prediction.with_name(f"{prediction.stem}_model.pkl").resolve()
    )
    model = _required_path(model, f"{layer} fitted model", directory=False)
    result = {
        "prediction": _artifact(prediction),
        "metadata": _artifact(metadata_path),
        "config": _artifact(config),
        "model": _artifact(model),
        "rows": int(len(frame)),
        "finite_rows": finite_rows,
        "symbols": symbols,
        "prediction_column": column,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "model_class": model_class,
        "config_segments": observed_segments,
        "frozen_protocol_match": segment_match,
    }
    whitelist_raw = metadata.get("whitelist_path") or metadata.get("whitelist")
    if whitelist_raw:
        whitelist = _required_path(
            _resolve_artifact(whitelist_raw, metadata_path.parent),
            f"{layer} whitelist",
            directory=False,
        )
        result["whitelist"] = _artifact(whitelist)
    return result


def validate_historical_gate(path: Path) -> dict[str, Any]:
    path = _required_path(path, "historical gate evidence", directory=False)
    gate = _read_json(path)
    if gate.get("passed") is not True or gate.get("failed_checks"):
        raise ValueError("historical gate evidence is not passed")
    if gate.get("profile") != PROFILE:
        raise ValueError(
            f"historical gate profile mismatch: {gate.get('profile')!r} expected={PROFILE!r}"
        )
    checks = gate.get("checks", [])
    if not checks or any(item.get("passed") is not True for item in checks):
        raise ValueError("historical gate contains a failed or missing check")
    local = gate.get("local_audit", {})
    gm = gate.get("gm", {})
    if float(local.get("total_return", -1)) <= 0:
        raise ValueError("historical local replay return is not positive")
    if float(local.get("max_drawdown", 1)) > 0.08:
        raise ValueError("historical local replay drawdown exceeds 8%")
    if int(gm.get("rejected_orders", 999999)) > 10:
        raise ValueError("historical GmQuant rejected orders exceed gate")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "profile": gate["profile"],
        "passed": True,
        "local_total_return": float(local["total_return"]),
        "local_max_drawdown": float(local["max_drawdown"]),
        "gm_total_return": float(gm.get("total_return", 0.0)),
        "gm_rejected_orders": int(gm.get("rejected_orders", 0)),
        "evidence_scope": "historical_replay_only",
        "future_holdout_proven": False,
    }


def run_command(command: list[str]) -> None:
    log("RUN " + subprocess.list2cmdline(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def train_predictions(args: argparse.Namespace, stage: Path) -> tuple[Path, Path]:
    if not args.train_end or not args.valid_start or not args.valid_end:
        raise ValueError(
            "--train-forward-models requires --train-end, --valid-start, and --valid-end"
        )
    config_dir = stage / "forward_configs"
    middle_config = config_dir / "middle_densemble.yaml"
    outer_config = config_dir / "outer_loss5_simple.yaml"
    build_forward_config(
        _required_path(args.middle_base_config, "middle base config", directory=False),
        middle_config,
        args.train_end,
        args.valid_start,
        args.valid_end,
        args.signal_date,
        args.middle_provider_day.resolve(),
    )
    build_forward_config(
        _required_path(args.outer_base_config, "outer base config", directory=False),
        outer_config,
        args.train_end,
        args.valid_start,
        args.valid_end,
        args.signal_date,
        args.outer_provider_day.resolve(),
    )
    stamp = args.signal_date.replace("-", "")
    middle_variant = f"outer_direct_loss5_release_{stamp}_middle_densemble"
    outer_variant = f"outer_direct_loss5_release_{stamp}_outer_loss5"
    run_command(
        [
            sys.executable,
            str(HERE / "run_oof.py"),
            "--config",
            str(middle_config),
            "--layer",
            "middle",
            "--fold",
            "99",
            "--variant",
            middle_variant,
            "--output-dir",
            str(stage),
            "--whitelist-path",
            str(_required_path(args.middle_whitelist, "middle whitelist", directory=False)),
        ]
    )
    run_command(
        [
            sys.executable,
            str(HERE / "run_oof.py"),
            "--config",
            str(outer_config),
            "--layer",
            "outer",
            "--fold",
            "99",
            "--variant",
            outer_variant,
            "--output-dir",
            str(stage),
        ]
    )
    return (
        stage / "oof" / middle_variant / "middle" / "fold_99.csv",
        stage / "oof" / outer_variant / "outer" / "fold_99.csv",
    )


def _validate_selection_contract(manifest: dict[str, Any]) -> None:
    outer = manifest.get("outer_overlay", {})
    allocation = manifest.get("allocation", {})
    observed_exact = {
        "top_k": manifest.get("top_k"),
        "rebalance_every": manifest.get("rebalance_every"),
        "buffer_multiple": manifest.get("buffer_multiple"),
        "allocation_mode": allocation.get("mode"),
        "outer_prediction_required": outer.get("required"),
    }
    expected_exact = {
        key: CONTRACT[key]
        for key in observed_exact
    }
    for field, expected in expected_exact.items():
        if observed_exact[field] != expected:
            raise ValueError(
                f"selection contract mismatch {field}: {observed_exact[field]!r} != {expected!r}"
            )
    observed_float = {
        "base_risk_budget": manifest.get("base_risk_budget"),
        "industry_cap": manifest.get("industry_cap"),
        "outer_risk_threshold": outer.get("threshold"),
        "outer_risk_floor": outer.get("risk_floor"),
        "initial_cash": manifest.get("initial_cash_for_share_estimate"),
    }
    for field, expected in observed_float.items():
        contract_value = float(CONTRACT[field])
        if abs(float(expected) - contract_value) > 1e-12:
            raise ValueError(
                f"selection contract mismatch {field}: {expected!r} != {contract_value!r}"
            )


def prepare_release(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "skip_inner_shadow", False)) and bool(
        getattr(args, "require_inner_shadow", False)
    ):
        raise ValueError(
            "--skip-inner-shadow and --require-inner-shadow cannot be combined"
        )
    validate_dates(
        signal_date=args.signal_date,
        trade_date=args.trade_date,
        as_of_date=args.as_of_date,
    )
    validate_release_clock(
        signal_date=args.signal_date,
        as_of_date=args.as_of_date,
        st_risk_refreshed_at=args.st_risk_refreshed_at,
        historical_smoke=args.historical_smoke,
    )
    if not args.initial_launch and args.previous_target is None:
        raise ValueError(
            "a previous target is required for buffered selection; use --initial-launch only once"
        )
    if not args.historical_smoke and args.account_snapshot is None:
        raise ValueError(
            "a fresh no-order GmQuant PAPER account snapshot is required for a real release"
        )
    if args.train_forward_models and (args.middle_prediction or args.outer_prediction):
        raise ValueError("do not combine --train-forward-models with supplied predictions")
    if not args.train_forward_models and not (
        args.middle_prediction and args.outer_prediction
    ):
        raise ValueError("supply both predictions or use --train-forward-models")

    previous_target = args.previous_target.resolve() if args.previous_target else None
    stage = args.stage_dir.resolve()
    if stage.exists():
        raise FileExistsError(f"versioned stage already exists: {stage}")
    stage.mkdir(parents=True)
    log(f"stage={stage}")
    middle_provider = _required_path(
        args.middle_provider_day, "middle provider", directory=True
    )
    outer_provider = _required_path(args.outer_provider_day, "outer provider", directory=True)
    provider_calendars = {
        "middle": validate_provider_calendar(middle_provider, args.signal_date, "middle"),
        "outer": validate_provider_calendar(outer_provider, args.signal_date, "outer"),
    }
    middle_rebalance = derive_middle_rebalance_schedule(
        middle_provider,
        args.signal_date,
        previous_target=previous_target,
        initial_launch=bool(args.initial_launch),
        rebalance_every=CONTRACT["rebalance_every"],
    )
    log(
        "middle rebalance schedule "
        f"due={middle_rebalance['due']} "
        f"last={middle_rebalance['last_rebalance_signal_date']} "
        f"elapsed={middle_rebalance['trading_sessions_elapsed_before_release']} "
        f"next={middle_rebalance['next_scheduled_rebalance_signal_date']}"
    )
    training_protocol = derive_forward_training_segments(
        middle_provider,
        args.signal_date,
    )
    outer_training_protocol = derive_forward_training_segments(
        outer_provider,
        args.signal_date,
    )
    if outer_training_protocol != training_protocol:
        raise ValueError("middle and outer providers derive different training protocols")
    supplied_training_dates = {
        "train_end": args.train_end,
        "valid_start": args.valid_start,
        "valid_end": args.valid_end,
    }
    if any(supplied_training_dates.values()):
        if not all(supplied_training_dates.values()):
            raise ValueError("supplied training dates must be complete")
        for field, observed in supplied_training_dates.items():
            if observed != training_protocol[field]:
                raise ValueError(
                    f"CLI {field} cannot alter frozen protocol: "
                    f"observed={observed} expected={training_protocol[field]}"
                )
    args.train_end = training_protocol["train_end"]
    args.valid_start = training_protocol["valid_start"]
    args.valid_end = training_protocol["valid_end"]
    validate_dates(
        signal_date=args.signal_date,
        trade_date=args.trade_date,
        as_of_date=args.as_of_date,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
    )
    log(
        "provider calendars verified "
        f"middle={provider_calendars['middle']['max_date']} "
        f"outer={provider_calendars['outer']['max_date']}"
    )
    log(
        "frozen purged protocol "
        f"train_end={args.train_end} valid={args.valid_start}..{args.valid_end} "
        f"purge={PURGE_DAYS} embargo={EMBARGO_DAYS}"
    )
    raw_daily = _required_path(args.raw_daily_dir, "raw daily directory", directory=True)
    group_metadata_source = _required_path(
        args.group_metadata, "group metadata source", directory=False
    )
    group_metadata = stage / "metadata" / f"industry_theme_pit_{args.signal_date.replace('-', '')}.csv"
    group_metadata_audit = prepare_group_metadata(
        group_metadata_source,
        group_metadata,
        args.signal_date,
    )
    forbidden, forbidden_audit = prepare_forbidden_symbols(args, stage)
    historical_gate = validate_historical_gate(args.historical_gate)
    log(
        "historical gate verified "
        f"return={historical_gate['local_total_return']:.2%} "
        f"mdd={historical_gate['local_max_drawdown']:.2%} scope=historical-only"
    )

    if args.train_forward_models:
        middle_prediction, outer_prediction = train_predictions(args, stage)
        prediction_mode = "trained"
    else:
        middle_prediction = args.middle_prediction.resolve()
        outer_prediction = args.outer_prediction.resolve()
        prediction_mode = "provided"
    prediction_audits = {
        "middle": validate_prediction(
            middle_prediction,
            layer="middle",
            signal_date=args.signal_date,
            provider=middle_provider,
            expected_segments=training_protocol,
            enforce_segments=not args.historical_smoke,
        ),
        "outer": validate_prediction(
            outer_prediction,
            layer="outer",
            signal_date=args.signal_date,
            provider=outer_provider,
            expected_segments=training_protocol,
            enforce_segments=not args.historical_smoke,
        ),
    }
    log(
        "predictions verified "
        f"middle={prediction_audits['middle']['symbols']} "
        f"outer={prediction_audits['outer']['symbols']}"
    )

    selection_dir = stage / "selection"
    selection = export_targets(
        middle_prediction,
        selection_dir,
        raw_daily,
        group_metadata,
        forbidden,
        args.signal_date,
        args.trade_date,
        CONTRACT["top_k"],
        CONTRACT["base_risk_budget"],
        CONTRACT["industry_cap"],
        100_000_000.0,
        CONTRACT["initial_cash"],
        outer_prediction,
        True,
        CONTRACT["outer_risk_threshold"],
        CONTRACT["outer_risk_floor"],
        CONTRACT["allocation_mode"],
        0.0,
        0.03,
        True,
        CONTRACT["rebalance_every"],
        CONTRACT["buffer_multiple"],
        False,
        previous_target,
        not args.initial_launch,
        middle_rebalance_due=bool(middle_rebalance["due"]),
        middle_last_rebalance_signal_date=middle_rebalance[
            "last_rebalance_signal_date"
        ],
        trading_sessions_since_rebalance=int(
            middle_rebalance["trading_sessions_since_rebalance"]
        ),
    )
    _validate_selection_contract(selection)
    selection_manifest = selection_dir / "manifest.json"
    raw_target = Path(selection["target"]).resolve()
    log(
        f"selection exported rows={selection['rows']} weight={selection['target_weight_sum']:.2%} "
        f"outer_p={selection['outer_overlay']['probability']:.6f}"
    )

    filtered_dir = stage / "market_filtered"
    filtered_target = filtered_dir / "gm_c_baseline_targets.csv"
    market_filter = filter_targets(
        raw_target,
        filtered_target,
        raw_daily,
        args.gm_audit_dir.resolve() if args.gm_audit_dir else None,
        19.5,
        True,
        False,
        previous_target,
    )
    filtered_manifest = filtered_target.with_suffix(".manifest.json")
    target_audit_path = filtered_dir / "TARGET_AUDIT.json"
    target_audit = audit_target(filtered_target, target_audit_path, forbidden)
    if target_audit.get("passed") is not True:
        raise RuntimeError(f"market-filtered target audit failed: {target_audit_path}")
    log(
        f"market filter complete rows={market_filter['output_rows']} "
        f"blocked={market_filter['blocked_actions']} audit=PASS"
    )

    transition_audit_path = stage / "transition" / "TARGET_TRANSITION_AUDIT.json"
    transition_audit = audit_target_transition(
        next_target=filtered_target,
        previous_target=previous_target,
        initial_launch=bool(args.initial_launch),
        raw_daily_dir=raw_daily,
        output_path=transition_audit_path,
        initial_nav=CONTRACT["initial_cash"],
        account_snapshot=(args.account_snapshot.resolve() if args.account_snapshot else None),
        expected_account_id=args.expected_account_id,
        as_of_date=args.as_of_date,
        account_snapshot_allowed_dates=(args.signal_date, args.trade_date),
    )
    if transition_audit.get("passed") is not True:
        raise RuntimeError(
            "target transition stability gate failed: "
            + ", ".join(transition_audit.get("failed_checks", []))
        )
    transition_metrics = transition_audit["metrics"]
    log(
        "transition audit PASS "
        f"mode={transition_audit['mode']} "
        f"orders={transition_audit['counts']['actionable_orders']} "
        f"turnover={transition_metrics['two_way_turnover']:.2%} "
        f"cost={transition_metrics['estimated_cost_ratio']:.4%} "
        f"min_cash={transition_metrics['min_cash_ratio']:.2%}"
    )

    release_provenance = {
        "status": "outer_direct_loss5_daily_release_provenance",
        "profile": PROFILE,
        "strategy_contract": CONTRACT,
        "signal_date": pd.Timestamp(args.signal_date).strftime("%Y-%m-%d"),
        "trade_date": pd.Timestamp(args.trade_date).strftime("%Y-%m-%d"),
        "as_of_date": pd.Timestamp(args.as_of_date).strftime("%Y-%m-%d"),
        "historical_smoke": bool(args.historical_smoke),
        "prediction_mode": prediction_mode,
        "training_protocol": training_protocol,
        "prediction_protocol_passed": bool(
            prediction_audits["middle"]["frozen_protocol_match"]
            and prediction_audits["outer"]["frozen_protocol_match"]
        ),
        "provider_calendars": provider_calendars,
        "middle_rebalance_schedule": middle_rebalance,
        "artifacts": {
            "middle_prediction": prediction_audits["middle"]["prediction"],
            "outer_prediction": prediction_audits["outer"]["prediction"],
            "middle_prediction_metadata": prediction_audits["middle"]["metadata"],
            "outer_prediction_metadata": prediction_audits["outer"]["metadata"],
            "middle_config": prediction_audits["middle"]["config"],
            "outer_config": prediction_audits["outer"]["config"],
            "middle_model": prediction_audits["middle"]["model"],
            "outer_model": prediction_audits["outer"]["model"],
            "forbidden_symbols": _artifact(forbidden),
            "group_metadata": _artifact(group_metadata),
            "selection_manifest": _artifact(selection_manifest),
            "raw_target": _artifact(raw_target),
            "market_filtered_target": _artifact(filtered_target),
            "market_filter_manifest": _artifact(filtered_manifest),
            "target_audit": _artifact(target_audit_path),
            "target_transition_audit": _artifact(transition_audit_path),
            "historical_gate": _artifact(args.historical_gate.resolve()),
        },
        "target_transition": transition_audit,
        "raw_daily_root": str(raw_daily),
        "raw_daily_file_count": int(sum(1 for path in raw_daily.iterdir() if path.is_file())),
        "group_metadata_preparation": group_metadata_audit,
        "forbidden_symbols_preparation": forbidden_audit,
        "historical_gate": historical_gate,
        "historical_evidence_scope": "historical_replay_only",
        "future_holdout_proven": False,
        "inner_t0_enabled": False,
        "real_money_deployment_allowed": False,
    }
    if args.account_snapshot is not None:
        account_snapshot_path = args.account_snapshot.resolve()
        account_positions_path = Path(
            str(transition_audit["account_snapshot"]["positions_path"])
        ).resolve()
        release_provenance["artifacts"]["account_snapshot"] = _artifact(
            account_snapshot_path
        )
        release_provenance["artifacts"]["account_positions"] = _artifact(
            account_positions_path
        )
    release_provenance_path = stage / "RELEASE_PROVENANCE.json"
    _write_json(release_provenance_path, release_provenance)

    package_dir = stage / "package"
    build_error = None
    try:
        package_report = build_candidate(
            base_candidate=_required_path(
                args.base_candidate, "base PAPER candidate", directory=True
            ),
            target_csv=filtered_target,
            target_manifest=filtered_manifest,
            selection_manifest=selection_manifest,
            forbidden_symbols=forbidden,
            output_dir=package_dir,
            as_of_date=pd.Timestamp(args.as_of_date),
            st_risk_refreshed_at=args.st_risk_refreshed_at,
            expected_account_id=args.expected_account_id,
            max_target_forward_days=0,
            max_signal_age_days=4,
            max_signal_to_target_days=4,
            max_metadata_age_days=7,
            release_provenance=release_provenance_path,
            historical_gate_evidence=args.historical_gate,
            transition_audit=transition_audit_path,
            account_snapshot=(args.account_snapshot.resolve() if args.account_snapshot else None),
            activation_registry_seed=(
                args.activation_registry_seed.resolve()
                if getattr(args, "activation_registry_seed", None)
                else None
            ),
        )
    except RuntimeError as exc:
        build_error = str(exc)
        ready_path = package_dir / f"PAPER_READY_{args.trade_date.replace('-', '')}.json"
        package_report = _read_json(ready_path) if ready_path.exists() else {
            "passed": False,
            "paper_orders_allowed": False,
            "errors": [build_error],
        }
        if not args.historical_smoke:
            raise

    inner_shadow_report = None
    inner_shadow_error = None
    inner_shadow_requested = not bool(getattr(args, "skip_inner_shadow", False))
    inner_shadow_dir = stage / "inner_shadow_no_order"
    if inner_shadow_requested:
        try:
            inner_shadow_report = build_synchronized_shadow(
                outer_package=package_dir,
                output_dir=inner_shadow_dir,
                as_of_date=pd.Timestamp(args.as_of_date),
                model_manifest=(
                    args.inner_shadow_model_manifest.resolve()
                    if getattr(args, "inner_shadow_model_manifest", None)
                    else DEFAULT_INNER_SHADOW_MODEL_MANIFEST
                ),
                historical_smoke=bool(args.historical_smoke),
                max_signal_age_days=4,
            )
            log(
                "synchronized inner shadow built "
                f"preflight={inner_shadow_report['inner_preflight_passed']} "
                f"runnable={inner_shadow_report['runnable_no_order_shadow']} "
                f"package={inner_shadow_dir}"
            )
        except Exception as exc:
            inner_shadow_error = f"{type(exc).__name__}: {exc}"
            log(f"synchronized inner shadow failed: {inner_shadow_error}")
            if bool(getattr(args, "require_inner_shadow", False)):
                raise

    result = {
        "status": (
            "historical_engineering_smoke_complete"
            if args.historical_smoke
            else "outer_direct_loss5_daily_release_complete"
        ),
        "profile": PROFILE,
        "signal_date": args.signal_date,
        "trade_date": args.trade_date,
        "stage_dir": str(stage),
        "package_dir": str(package_dir),
        "prediction_mode": prediction_mode,
        "middle_rebalance_due": bool(middle_rebalance["due"]),
        "middle_last_rebalance_signal_date": middle_rebalance[
            "last_rebalance_signal_date"
        ],
        "historical_smoke": bool(args.historical_smoke),
        "engineering_pipeline_complete": package_dir.exists(),
        "preflight_passed": bool(package_report.get("passed")),
        "paper_orders_allowed": bool(package_report.get("paper_orders_allowed")),
        "preflight_errors": list(package_report.get("errors", [])),
        "build_error": build_error,
        "inner_shadow_requested": inner_shadow_requested,
        "inner_shadow_package": (
            str(inner_shadow_dir) if inner_shadow_report is not None else None
        ),
        "inner_shadow_preflight_passed": bool(
            inner_shadow_report
            and inner_shadow_report.get("inner_preflight_passed")
        ),
        "inner_shadow_runnable_no_order": bool(
            inner_shadow_report
            and inner_shadow_report.get("runnable_no_order_shadow")
        ),
        "inner_shadow_error": inner_shadow_error,
        "future_holdout_proven": False,
        "inner_t0_enabled": False,
        "real_money_deployment_allowed": False,
    }
    _write_json(stage / "DAILY_RELEASE_RESULT.json", result)
    log(
        f"complete preflight={result['preflight_passed']} "
        f"paper_orders_allowed={result['paper_orders_allowed']} package={package_dir}"
    )
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build one versioned Top150/rb20 outer+middle PAPER release. "
            "The strategy contract is frozen in code and cannot be tuned by CLI."
        )
    )
    p.add_argument("--signal-date", required=True)
    p.add_argument("--trade-date", required=True)
    p.add_argument("--as-of-date", required=True)
    p.add_argument("--st-risk-refreshed-at", required=True)
    p.add_argument("--stage-dir", type=Path, required=True)
    p.add_argument("--base-candidate", type=Path, default=DEFAULT_BASE_CANDIDATE)
    p.add_argument(
        "--raw-daily-dir",
        type=Path,
        default=DATA / "A_Stock_daily_qfq" / "daily_qfq_6.5",
    )
    p.add_argument(
        "--middle-provider-day",
        type=Path,
        default=DATA / "cn_data_canonical_pit_20260605",
    )
    p.add_argument(
        "--outer-provider-day",
        type=Path,
        default=DATA / "cn_data_outer_regime_broad_20260605_v2",
    )
    p.add_argument(
        "--group-metadata",
        type=Path,
        default=DATA / "industry_theme_pit_ffill_20260605.csv",
    )
    p.add_argument(
        "--forbidden-symbols",
        type=Path,
        default=DATA / "forbidden_st_symbols_20260605.csv",
    )
    p.add_argument("--stock-list", type=Path, default=DATA / "股票列表.csv")
    p.add_argument("--pit-market-state", type=Path)
    p.add_argument(
        "--forbidden-overrides",
        type=Path,
        default=DATA / "forbidden_st_manual_overrides.csv",
    )
    p.add_argument("--historical-gate", type=Path, default=DEFAULT_HISTORICAL_GATE)
    p.add_argument("--gm-audit-dir", type=Path)
    p.add_argument("--previous-target", type=Path)
    p.add_argument("--account-snapshot", type=Path)
    p.add_argument("--activation-registry-seed", type=Path)
    p.add_argument("--initial-launch", action="store_true")
    p.add_argument("--train-forward-models", action="store_true")
    p.add_argument("--train-end", help="Optional assertion; derived from the provider calendar.")
    p.add_argument("--valid-start", help="Optional assertion; derived from the provider calendar.")
    p.add_argument("--valid-end", help="Optional assertion; derived from the provider calendar.")
    p.add_argument("--middle-base-config", type=Path, default=DEFAULT_MIDDLE_BASE)
    p.add_argument("--middle-whitelist", type=Path, default=DEFAULT_MIDDLE_WHITELIST)
    p.add_argument("--outer-base-config", type=Path, default=DEFAULT_OUTER_BASE)
    p.add_argument("--middle-prediction", type=Path)
    p.add_argument("--outer-prediction", type=Path)
    p.add_argument("--expected-account-id", default=DEFAULT_ACCOUNT_ID)
    p.add_argument(
        "--historical-smoke",
        action="store_true",
        help="Exercise the pipeline on an old date while forcing PAPER orders to remain blocked.",
    )
    p.add_argument(
        "--skip-inner-shadow",
        action="store_true",
        help="Skip the synchronized no-order inner observer package.",
    )
    p.add_argument(
        "--require-inner-shadow",
        action="store_true",
        help="Fail the release when the synchronized no-order inner package cannot be built.",
    )
    p.add_argument(
        "--inner-shadow-model-manifest",
        type=Path,
        help="Frozen no-order inner model manifest to observe alongside PAPER.",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = prepare_release(args)
    if not result["paper_orders_allowed"] and not args.historical_smoke:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
