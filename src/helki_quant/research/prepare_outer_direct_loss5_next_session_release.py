from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from audit_target_transition import (
    load_account_snapshot,
    load_target,
    sha256_file,
)
from prepare_outer_direct_loss5_daily_release import (
    DATA,
    DEFAULT_ACCOUNT_ID,
    DEFAULT_BASE_CANDIDATE,
    DEFAULT_HISTORICAL_GATE,
    DEFAULT_MIDDLE_BASE,
    DEFAULT_MIDDLE_WHITELIST,
    DEFAULT_OUTER_BASE,
    PROFILE,
    REPO_ROOT,
)
from paper_activation_registry import resolve_latest_finalized_target


HERE = Path(__file__).resolve().parent
DAILY_RELEASE = HERE / "prepare_outer_direct_loss5_daily_release.py"
DEFAULT_STAGE_ROOT = HERE / "outputs" / "outer_direct_loss5_live_releases"
DEFAULT_PLAN_ROOT = REPO_ROOT / "outputs" / "outer_direct_loss5_release_plans"


def _required_path(
    path: Path,
    label: str,
    *,
    directory: bool | None = None,
) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if directory is True and not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    if directory is False and not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")
    return resolved


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(pending, path)


def _calendar_evidence(provider: Path, layer: str) -> dict[str, Any]:
    provider = _required_path(provider, f"{layer} provider", directory=True)
    calendar = _required_path(
        provider / "calendars" / "day.txt",
        f"{layer} provider calendar",
        directory=False,
    )
    values = pd.DatetimeIndex(
        pd.to_datetime(
            [
                line.strip()
                for line in calendar.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if line.strip()
            ],
            errors="coerce",
        )
    ).dropna().drop_duplicates().sort_values().normalize()
    if values.empty:
        raise ValueError(f"{layer} provider calendar is empty")
    return {
        "provider": str(provider),
        "calendar": str(calendar),
        "sha256": sha256_file(calendar),
        "rows": int(len(values)),
        "min_date": values.min().strftime("%Y-%m-%d"),
        "max_date": values.max().strftime("%Y-%m-%d"),
    }


def _snapshot_release_dates(
    snapshot_path: Path,
    *,
    signal_date: str,
    expected_account_id: str,
) -> tuple[str, dict[str, Any]]:
    snapshot_path = _required_path(
        snapshot_path,
        "PAPER account snapshot",
        directory=False,
    )
    payload = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
    captured_date = pd.Timestamp(payload.get("captured_at")).strftime("%Y-%m-%d")
    if captured_date != signal_date:
        raise ValueError(
            "after-close account snapshot date must equal provider signal date: "
            f"captured={captured_date} signal={signal_date}"
        )
    calendar = payload.get("trading_calendar") or {}
    if calendar.get("source") != "gm.api.get_next_trading_date":
        raise ValueError("account snapshot lacks GmQuant trading-calendar evidence")
    if calendar.get("exchange") != "SHSE":
        raise ValueError("account snapshot trading-calendar exchange is not SHSE")
    trade = pd.Timestamp(calendar.get("next_trading_date")).normalize()
    signal = pd.Timestamp(signal_date).normalize()
    lag = int((trade - signal).days)
    if lag < 1 or lag > 4:
        raise ValueError(
            "snapshot next trading date is outside the frozen 1..4 day lag: "
            f"signal={signal.date()} trade={trade.date()} lag={lag}"
        )
    trade_date = trade.strftime("%Y-%m-%d")
    validated = load_account_snapshot(
        snapshot_path,
        expected_account_id=expected_account_id,
        as_of_date=trade_date,
        allowed_capture_dates=(signal_date, trade_date),
    )
    return trade_date, validated


def _validate_execution_window(
    *,
    now: pd.Timestamp,
    signal_date: str,
    trade_date: str,
) -> dict[str, Any]:
    current = pd.Timestamp(now)
    if current.tzinfo is not None:
        current = current.tz_convert("Asia/Shanghai").tz_localize(None)
    current_date = current.strftime("%Y-%m-%d")
    current_time = current.strftime("%H:%M:%S")
    if current_date == signal_date:
        passed = current_time >= "15:05:00"
        mode = "signal_day_after_close"
        requirement = ">=15:05:00"
    elif current_date == trade_date:
        passed = current_time <= "09:20:00"
        mode = "trade_day_preopen"
        requirement = "<=09:20:00"
    else:
        passed = False
        mode = "outside_release_boundary"
        requirement = f"date in {{{signal_date}, {trade_date}}}"
    if not passed:
        raise ValueError(
            "release planning is outside the fail-closed execution window: "
            f"now={current.isoformat()} mode={mode} required={requirement}"
        )
    return {
        "now": current.isoformat(),
        "mode": mode,
        "requirement": requirement,
        "passed": True,
    }


def build_release_plan(
    *,
    middle_provider: Path,
    outer_provider: Path,
    raw_daily_dir: Path,
    group_metadata: Path,
    forbidden_overrides: Path,
    historical_gate: Path,
    previous_target: Path | None,
    account_snapshot: Path,
    stage_root: Path,
    expected_account_id: str,
    now: pd.Timestamp,
    activation_registry: Path | None = None,
    base_candidate: Path = DEFAULT_BASE_CANDIDATE,
    middle_base_config: Path = DEFAULT_MIDDLE_BASE,
    middle_whitelist: Path = DEFAULT_MIDDLE_WHITELIST,
    outer_base_config: Path = DEFAULT_OUTER_BASE,
    minimum_stock_universe: int = 1000,
) -> dict[str, Any]:
    calendars = {
        "middle": _calendar_evidence(middle_provider, "middle"),
        "outer": _calendar_evidence(outer_provider, "outer"),
    }
    signal_dates = {
        calendars["middle"]["max_date"],
        calendars["outer"]["max_date"],
    }
    if len(signal_dates) != 1:
        raise ValueError(
            "middle and outer provider calendars end on different dates: "
            f"{sorted(signal_dates)}"
        )
    signal_date = next(iter(signal_dates))
    trade_date, snapshot = _snapshot_release_dates(
        account_snapshot,
        signal_date=signal_date,
        expected_account_id=expected_account_id,
    )
    stock_universe = snapshot.get("stock_universe")
    if not stock_universe:
        raise ValueError(
            "account snapshot lacks a hash-protected GmQuant stock-universe refresh"
        )
    if stock_universe.get("source") != "gm.api.get_instruments":
        raise ValueError("account snapshot stock-universe source is not GmQuant")
    if int(stock_universe.get("rows", 0)) < minimum_stock_universe:
        raise ValueError(
            "account snapshot stock-universe coverage is too small: "
            f"rows={stock_universe.get('rows')} required={minimum_stock_universe}"
        )
    release_window = _validate_execution_window(
        now=now,
        signal_date=signal_date,
        trade_date=trade_date,
    )

    if (previous_target is None) == (activation_registry is None):
        raise ValueError(
            "supply exactly one previous-target source: "
            "previous_target or activation_registry"
        )
    activation_evidence = None
    activation_registry_path = None
    if activation_registry is not None:
        activation_registry_path = _required_path(
            activation_registry,
            "PAPER activation registry",
            directory=False,
        )
        activation_evidence = resolve_latest_finalized_target(
            activation_registry_path,
            expected_account_id=expected_account_id,
            before_trade_date=trade_date,
        )
        previous_target = Path(activation_evidence["target_path"])
        previous_target_source = "finalized_paper_activation"
    else:
        previous_target_source = "explicit_legacy_override"
    previous_target = _required_path(
        Path(previous_target),
        "previous deployed target",
        directory=False,
    )
    previous_frame, previous_dates = load_target(previous_target, "previous")
    if pd.Timestamp(previous_dates["trade_date"]) >= pd.Timestamp(trade_date):
        raise ValueError(
            "previous target must precede the planned trade date: "
            f"previous={previous_dates['trade_date']} planned={trade_date}"
        )
    if pd.Timestamp(previous_dates["signal_date"]) > pd.Timestamp(signal_date):
        raise ValueError("previous target signal date is after the provider signal date")

    required_files = {
        "raw_daily_dir": _required_path(
            raw_daily_dir,
            "raw daily directory",
            directory=True,
        ),
        "group_metadata": _required_path(
            group_metadata,
            "group metadata",
            directory=False,
        ),
        "stock_list": _required_path(
            Path(str(stock_universe["path"])),
            "snapshot GmQuant stock universe",
            directory=False,
        ),
        "forbidden_overrides": _required_path(
            forbidden_overrides,
            "forbidden overrides",
            directory=False,
        ),
        "historical_gate": _required_path(
            historical_gate,
            "historical gate",
            directory=False,
        ),
        "base_candidate": _required_path(
            base_candidate,
            "base candidate",
            directory=True,
        ),
        "middle_base_config": _required_path(
            middle_base_config,
            "middle base config",
            directory=False,
        ),
        "middle_whitelist": _required_path(
            middle_whitelist,
            "middle whitelist",
            directory=False,
        ),
        "outer_base_config": _required_path(
            outer_base_config,
            "outer base config",
            directory=False,
        ),
    }
    stage_dir = (
        stage_root.resolve()
        / (
            "outer_direct_loss5_release_"
            f"{trade_date.replace('-', '')}_"
            f"{Path(str(snapshot['path'])).parent.name}"
        )
    )
    if stage_dir.exists():
        raise FileExistsError(f"versioned release stage already exists: {stage_dir}")

    command = [
        sys.executable,
        str(DAILY_RELEASE),
        "--signal-date",
        signal_date,
        "--trade-date",
        trade_date,
        "--as-of-date",
        trade_date,
        "--st-risk-refreshed-at",
        pd.Timestamp(now).strftime("%Y-%m-%d"),
        "--stage-dir",
        str(stage_dir),
        "--base-candidate",
        str(required_files["base_candidate"]),
        "--raw-daily-dir",
        str(required_files["raw_daily_dir"]),
        "--middle-provider-day",
        calendars["middle"]["provider"],
        "--outer-provider-day",
        calendars["outer"]["provider"],
        "--group-metadata",
        str(required_files["group_metadata"]),
        "--stock-list",
        str(required_files["stock_list"]),
        "--forbidden-overrides",
        str(required_files["forbidden_overrides"]),
        "--historical-gate",
        str(required_files["historical_gate"]),
        "--previous-target",
        str(previous_target),
        "--account-snapshot",
        str(Path(str(snapshot["path"])).resolve()),
        "--middle-base-config",
        str(required_files["middle_base_config"]),
        "--middle-whitelist",
        str(required_files["middle_whitelist"]),
        "--outer-base-config",
        str(required_files["outer_base_config"]),
        "--expected-account-id",
        expected_account_id,
        "--train-forward-models",
        "--require-inner-shadow",
    ]
    if activation_registry_path is not None:
        command.extend(
            [
                "--activation-registry-seed",
                str(activation_registry_path),
            ]
        )
    return {
        "status": "outer_direct_loss5_next_session_release_planned",
        "profile": PROFILE,
        "signal_date": signal_date,
        "trade_date": trade_date,
        "as_of_date": trade_date,
        "stage_dir": str(stage_dir),
        "release_window": release_window,
        "provider_calendars": calendars,
        "account_snapshot": snapshot,
        "stock_universe": stock_universe,
        "previous_target": {
            "path": str(previous_target),
            "sha256": sha256_file(previous_target),
            "rows": int(len(previous_frame)),
            "source": previous_target_source,
            "activation": activation_evidence,
            **previous_dates,
        },
        "activation_chain_seed": (
            {
                "path": str(activation_registry_path),
                "sha256": sha256_file(activation_registry_path),
                "source": "finalized_paper_activation",
            }
            if activation_registry_path is not None
            else None
        ),
        "command": command,
        "command_display": subprocess.list2cmdline(command),
        "training_mode": "frozen_purged_forward_models",
        "inner_mode": "synchronized_no_order_shadow_required",
        "paper_orders_allowed": False,
        "execution_started": False,
    }


def shanghai_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Shanghai")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Derive and optionally run the next-session frozen outer+middle "
            "release from one no-order PAPER snapshot."
        )
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
        "--raw-daily-dir",
        type=Path,
        default=DATA / "A_Stock_daily_qfq" / "daily_qfq_6.5",
    )
    p.add_argument(
        "--group-metadata",
        type=Path,
        default=DATA / "industry_theme_pit_ffill_20260605.csv",
    )
    p.add_argument(
        "--forbidden-overrides",
        type=Path,
        default=DATA / "forbidden_st_manual_overrides.csv",
    )
    p.add_argument("--historical-gate", type=Path, default=DEFAULT_HISTORICAL_GATE)
    p.add_argument("--base-candidate", type=Path, default=DEFAULT_BASE_CANDIDATE)
    p.add_argument("--middle-base-config", type=Path, default=DEFAULT_MIDDLE_BASE)
    p.add_argument(
        "--middle-whitelist",
        type=Path,
        default=DEFAULT_MIDDLE_WHITELIST,
    )
    p.add_argument("--outer-base-config", type=Path, default=DEFAULT_OUTER_BASE)
    previous_group = p.add_mutually_exclusive_group(required=True)
    previous_group.add_argument("--previous-target", type=Path)
    previous_group.add_argument("--activation-registry", type=Path)
    p.add_argument("--account-snapshot", type=Path, required=True)
    p.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    p.add_argument("--expected-account-id", default=DEFAULT_ACCOUNT_ID)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    plan = build_release_plan(
        middle_provider=args.middle_provider_day,
        outer_provider=args.outer_provider_day,
        raw_daily_dir=args.raw_daily_dir,
        group_metadata=args.group_metadata,
        forbidden_overrides=args.forbidden_overrides,
        historical_gate=args.historical_gate,
        previous_target=args.previous_target,
        activation_registry=args.activation_registry,
        account_snapshot=args.account_snapshot,
        stage_root=args.stage_root,
        expected_account_id=args.expected_account_id,
        now=shanghai_now(),
        base_candidate=args.base_candidate,
        middle_base_config=args.middle_base_config,
        middle_whitelist=args.middle_whitelist,
        outer_base_config=args.outer_base_config,
    )
    output = (
        args.output.resolve()
        if args.output
        else (
            DEFAULT_PLAN_ROOT
            / (
                "NEXT_SESSION_RELEASE_PLAN_"
                f"{plan['trade_date'].replace('-', '')}_"
                f"{args.account_snapshot.resolve().parent.name}.json"
            )
        ).resolve()
    )
    if output.exists():
        raise FileExistsError(f"versioned release plan already exists: {output}")
    _write_json_atomic(output, plan)
    print(
        "[next-session release] "
        f"planned signal={plan['signal_date']} trade={plan['trade_date']} "
        f"stage={plan['stage_dir']} plan={output}",
        flush=True,
    )
    print(f"[next-session release] RUN {plan['command_display']}", flush=True)
    if not args.execute:
        return
    try:
        subprocess.run(plan["command"], cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        plan["execution_started"] = True
        plan["execution_passed"] = False
        plan["execution_returncode"] = int(exc.returncode)
        _write_json_atomic(output, plan)
        raise
    plan["execution_started"] = True
    plan["execution_passed"] = True
    plan["execution_returncode"] = 0
    release_result = Path(plan["stage_dir"]) / "DAILY_RELEASE_RESULT.json"
    plan["daily_release_result"] = {
        "path": str(release_result),
        "sha256": sha256_file(release_result),
    }
    _write_json_atomic(output, plan)


if __name__ == "__main__":
    main()
