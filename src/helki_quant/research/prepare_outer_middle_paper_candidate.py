from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

try:
    from .audit_gm_target_csv import audit as audit_target
    from .build_paper_forward_config import build as build_forward_config
    from .export_paper_forward_gm_targets import export_targets
    from .validate_c_baseline_paper_gate import validate as validate_paper_gate
except ImportError:
    from audit_gm_target_csv import audit as audit_target
    from build_paper_forward_config import build as build_forward_config
    from export_paper_forward_gm_targets import export_targets
    from validate_c_baseline_paper_gate import validate as validate_paper_gate


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
OUTPUTS = HERE / "outputs"

DEFAULT_MIDDLE_BASE = (
    OUTPUTS / "fold_configs_canonical_20260605" / "fold_06" / "densemble.yaml"
)
DEFAULT_OUTER_BASE = (
    OUTPUTS
    / "outer_regime_fold_configs_broad_adverse_loss5_20d_v2_20260609"
    / "fold_06"
    / "simple.yaml"
)
DEFAULT_TEMPLATE = REPO_ROOT / "outputs" / "gmquant_outer_middle_ca_buf4_research_candidate"
DEFAULT_CONFIG = HERE / "configs" / "outer_direct_loss5_capital_aware_paper_gate.json"
DEFAULT_LOCAL_LOG = (
    OUTPUTS
    / "prod_outer_ca_buf4_floor20_sellfirst_1m"
    / "c_outer_ca_t150_r30_60_20_b4_stress"
)
DEFAULT_GM_COMPARE = OUTPUTS / "outer_middle_ca_buf4_floor20_frozen_target_gm_local_compare.json"
DEFAULT_HOLDOUT = OUTPUTS / "extended_daily_holdout_c_baseline_20260605_summary.csv"
DEFAULT_SELLBLOCK = OUTPUTS / "outer_ca_buf4_floor20_sellblock_stress_summary.csv"
DEFAULT_STALEEXIT = OUTPUTS / "outer_ca_buf4_floor20_staleexit_stress_summary.csv"


def log(message: str) -> None:
    print(f"[paper prepare] {message}", flush=True)


def run_command(command: list[str]) -> None:
    log("RUN " + subprocess.list2cmdline(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def require_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} path is required")
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def validate_dates(
    train_end: str,
    valid_start: str,
    valid_end: str,
    signal_date: str,
    trade_date: str,
) -> None:
    train_ts = pd.Timestamp(train_end)
    valid_start_ts = pd.Timestamp(valid_start)
    valid_end_ts = pd.Timestamp(valid_end)
    signal_ts = pd.Timestamp(signal_date)
    trade_ts = pd.Timestamp(trade_date)
    if not train_ts < valid_start_ts <= valid_end_ts < signal_ts < trade_ts:
        raise ValueError(
            "required ordering: train_end < valid_start <= valid_end < "
            "signal_date < trade_date"
        )


def validate_prediction(path: Path, layer: str, signal_date: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    if "datetime" not in frame.columns:
        raise ValueError(f"{path} has no datetime column")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
    day = frame[frame["datetime"].eq(pd.Timestamp(signal_date).normalize())].copy()
    prediction_columns = {
        "middle": ("middle", "pred_middle"),
        "outer": ("outer", "pred_outer"),
    }[layer]
    prediction_col = next((name for name in prediction_columns if name in day.columns), None)
    if prediction_col is None:
        raise ValueError(f"{path} has no {layer} prediction column")
    finite = pd.to_numeric(day[prediction_col], errors="coerce").dropna()
    if finite.empty:
        raise ValueError(f"{path} has no finite {layer} prediction on {signal_date}")
    return {
        "path": str(path),
        "layer": layer,
        "signal_date": signal_date,
        "rows": int(len(day)),
        "finite_rows": int(len(finite)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def rebuild_providers(args: argparse.Namespace, stage_dir: Path) -> None:
    if args.middle_provider_day.exists() or args.outer_provider_day.exists():
        raise FileExistsError(
            "provider rebuild is versioned and non-destructive; choose new, absent "
            "--middle-provider-day and --outer-provider-day paths"
        )
    provider_stage = stage_dir / "provider_stage"
    run_command(
        [
            sys.executable,
            str(HERE / "build_pit_daily_pool.py"),
            "--mode",
            "build",
            "--raw-dir",
            str(args.raw_daily_dir),
            "--stage-dir",
            str(provider_stage),
            "--output-dir",
            str(args.middle_provider_day),
            "--max-workers",
            str(args.max_workers),
            "--vwap-mode",
            "close",
        ]
    )
    validation_output = stage_dir / "provider_validation.json"
    command = [
        sys.executable,
        str(HERE / "validate_pit_daily_pool.py"),
        "--new-provider",
        str(args.middle_provider_day),
        "--output",
        str(validation_output),
    ]
    if args.old_middle_provider:
        command.extend(["--old-provider", str(args.old_middle_provider)])
    run_command(command)
    run_command(
        [
            sys.executable,
            str(HERE / "build_outer_regime_history_provider.py"),
            "--source-provider",
            str(args.middle_provider_day),
            "--raw-daily-dir",
            str(args.raw_daily_dir),
            "--output-provider",
            str(args.outer_provider_day),
            "--daily-output",
            str(stage_dir / "outer_regime_daily.csv"),
            "--report",
            str(stage_dir / "outer_regime_provider_report.json"),
            "--start",
            "2022-01-04",
            "--end",
            args.signal_date,
            "--horizons",
            "5,10,20",
            "--min-listing-days",
            "250",
            "--min-avg-amount",
            "100000000",
        ]
    )


def refresh_metadata(args: argparse.Namespace, stage_dir: Path) -> tuple[Path, Path]:
    metadata_dir = stage_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    forbidden = metadata_dir / f"forbidden_st_symbols_{args.signal_date.replace('-', '')}.csv"
    industry = metadata_dir / f"industry_theme_pit_ffill_{args.signal_date.replace('-', '')}.csv"
    run_command(
        [
            sys.executable,
            str(HERE / "build_forbidden_st_symbols.py"),
            "--source",
            str(args.stock_list),
            "--output",
            str(forbidden),
            "--report",
            str(metadata_dir / "forbidden_report.json"),
        ]
    )
    run_command(
        [
            sys.executable,
            str(HERE / "build_industry_pit_metadata.py"),
            "--raw-dir",
            str(args.industry_raw_dir),
            "--output",
            str(industry),
            "--report",
            str(metadata_dir / "industry_report.json"),
            "--forward-fill-to",
            args.signal_date,
        ]
    )
    return forbidden, industry


def train_predictions(args: argparse.Namespace, stage_dir: Path) -> tuple[Path, Path]:
    config_dir = stage_dir / "forward_configs"
    middle_config = config_dir / "middle_densemble.yaml"
    outer_config = config_dir / "outer_loss5_simple.yaml"
    build_forward_config(
        args.middle_base_config,
        middle_config,
        args.train_end,
        args.valid_start,
        args.valid_end,
        args.signal_date,
        args.middle_provider_day,
    )
    build_forward_config(
        args.outer_base_config,
        outer_config,
        args.train_end,
        args.valid_start,
        args.valid_end,
        args.signal_date,
        args.outer_provider_day,
    )
    stamp = args.signal_date.replace("-", "")
    middle_variant = f"paper_forward_{stamp}_middle_densemble"
    outer_variant = f"paper_forward_{stamp}_outer_loss5"
    for config, layer, variant in (
        (middle_config, "middle", middle_variant),
        (outer_config, "outer", outer_variant),
    ):
        run_command(
            [
                sys.executable,
                str(HERE / "run_oof.py"),
                "--config",
                str(config),
                "--layer",
                layer,
                "--fold",
                "99",
                "--variant",
                variant,
                "--output-dir",
                str(stage_dir),
            ]
        )
    return (
        stage_dir / "oof" / middle_variant / "middle" / "fold_99.csv",
        stage_dir / "oof" / outer_variant / "outer" / "fold_99.csv",
    )


def build_package(
    template_dir: Path,
    package_dir: Path,
    target_path: Path,
    target_manifest_path: Path,
    forbidden_path: Path,
    gate: dict,
) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    for name in ("main.py", "gm_outer_middle_ca_buf4_paper.py"):
        shutil.copy2(template_dir / name, package_dir / name)
    shutil.copy2(target_path, package_dir / "gm_c_baseline_targets.csv")
    shutil.copy2(target_manifest_path, package_dir / "gm_c_baseline_targets.manifest.json")
    shutil.copy2(forbidden_path, package_dir / "gm_c_forbidden_symbols.csv")
    readme = (
        "# Prepared Outer + Middle PAPER Candidate\n\n"
        f"Strict gate passed: `{gate['passed']}`.\n\n"
        "Run `gm_outer_middle_ca_buf4_paper.py` only when the gate passed. "
        "Never run `main.py` directly. Inner T+0 is not included.\n"
    )
    (package_dir / "README.md").write_text(readme, encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, object]:
    validate_dates(
        args.train_end,
        args.valid_start,
        args.valid_end,
        args.signal_date,
        args.trade_date,
    )
    if not args.previous_target and not args.initial_launch:
        raise ValueError(
            "provide --previous-target for Top600 retention, or explicitly use "
            "--initial-launch for a first portfolio"
        )
    stage_dir = args.stage_dir.resolve()
    if stage_dir.exists() and any(stage_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"staging directory is not empty: {stage_dir}; use a new versioned "
            "path or explicitly pass --resume"
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    log(f"stage={stage_dir}")

    if args.rebuild_providers:
        rebuild_providers(args, stage_dir)
    middle_provider = require_path(args.middle_provider_day, "middle provider")
    outer_provider = require_path(args.outer_provider_day, "outer provider")
    log(f"providers middle={middle_provider} outer={outer_provider}")

    if args.refresh_metadata:
        forbidden_path, industry_path = refresh_metadata(args, stage_dir)
    else:
        forbidden_path = require_path(args.forbidden_symbols, "forbidden symbols")
        industry_path = require_path(args.group_metadata, "industry metadata")

    if args.train_forward_models:
        middle_prediction, outer_prediction = train_predictions(args, stage_dir)
    else:
        middle_prediction = require_path(args.middle_prediction, "middle prediction")
        outer_prediction = require_path(args.outer_prediction, "outer prediction")
    middle_prediction = require_path(middle_prediction, "middle prediction")
    outer_prediction = require_path(outer_prediction, "outer prediction")
    prediction_audit = {
        "middle": validate_prediction(middle_prediction, "middle", args.signal_date),
        "outer": validate_prediction(outer_prediction, "outer", args.signal_date),
    }
    log(
        f"prediction rows middle={prediction_audit['middle']['finite_rows']} "
        f"outer={prediction_audit['outer']['finite_rows']}"
    )

    target_dir = stage_dir / "target"
    target_manifest = export_targets(
        middle_prediction,
        target_dir,
        require_path(args.raw_daily_dir, "raw daily directory"),
        industry_path,
        forbidden_path,
        args.signal_date,
        args.trade_date,
        150,
        0.60,
        0.30,
        100_000_000.0,
        1_000_000.0,
        outer_prediction,
        True,
        0.50,
        0.20,
        "capital_aware",
        0.90,
        0.03,
        True,
        30,
        4,
        True,
        args.previous_target.resolve() if args.previous_target else None,
        bool(args.previous_target),
    )
    target_path = Path(target_manifest["target"])
    target_manifest_path = target_dir / "manifest.json"
    target_audit_path = stage_dir / "target_audit.json"
    target_audit = audit_target(target_path, target_audit_path, forbidden_path)
    if not target_audit["passed"]:
        raise RuntimeError(f"target audit failed: {target_audit_path}")

    gate_path = stage_dir / "strict_gate.json"
    gate = validate_paper_gate(
        require_path(args.gate_config, "gate config"),
        require_path(args.local_log_dir, "local production log"),
        require_path(args.gm_compare, "GmQuant compare"),
        require_path(args.holdout_summary, "holdout summary"),
        require_path(args.sellblock_summary, "sell-block summary"),
        require_path(args.staleexit_summary, "stale-exit summary"),
        target_manifest_path,
        gate_path,
        date.fromisoformat(args.as_of_date),
    )
    package_dir = stage_dir / "package"
    build_package(
        require_path(args.candidate_template, "candidate template"),
        package_dir,
        target_path,
        target_manifest_path,
        forbidden_path,
        gate,
    )
    result: dict[str, object] = {
        "status": "outer_middle_paper_candidate_prepared",
        "paper_launch_allowed": bool(gate["passed"]),
        "published": False,
        "signal_date": args.signal_date,
        "trade_date": args.trade_date,
        "stage_dir": str(stage_dir),
        "package_dir": str(package_dir),
        "prediction_audit": prediction_audit,
        "target_manifest": str(target_manifest_path),
        "target_audit": str(target_audit_path),
        "strict_gate": str(gate_path),
        "failed_checks": gate["failed_checks"],
    }
    if args.publish_on_pass:
        if not gate["passed"]:
            raise RuntimeError("publish requested but strict gate did not pass")
        destination = args.candidate_template.resolve()
        for name in (
            "gm_c_baseline_targets.csv",
            "gm_c_baseline_targets.manifest.json",
            "gm_c_forbidden_symbols.csv",
        ):
            shutil.copy2(package_dir / name, destination / name)
        ready_path = destination / f"PAPER_READY_{args.trade_date.replace('-', '')}.json"
        ready_path.write_text(
            json.dumps(result | {"published": True}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["published"] = True
        result["paper_ready_manifest"] = str(ready_path)
    result_path = stage_dir / "prepare_manifest.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        f"gate_passed={gate['passed']} failed={len(gate['failed_checks'])} "
        f"package={package_dir}"
    )
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--signal-date", required=True)
    p.add_argument("--trade-date", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--valid-start", required=True)
    p.add_argument("--valid-end", required=True)
    p.add_argument("--stage-dir", type=Path, default=OUTPUTS / "paper_candidate_staging")
    p.add_argument("--raw-daily-dir", type=Path, default=DATA / "A_Stock_daily_qfq" / "daily_qfq_6.5")
    p.add_argument("--middle-provider-day", type=Path, default=DATA / "cn_data_canonical_pit_20260605")
    p.add_argument("--outer-provider-day", type=Path, default=DATA / "cn_data_outer_regime_broad_20260605_v2")
    p.add_argument("--old-middle-provider", type=Path)
    p.add_argument("--rebuild-providers", action="store_true")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--refresh-metadata", action="store_true")
    p.add_argument("--stock-list", type=Path, default=DATA / "股票列表.csv")
    p.add_argument("--industry-raw-dir", type=Path, default=DATA / "A_Stock_industry")
    p.add_argument("--forbidden-symbols", type=Path, default=DATA / "forbidden_st_symbols_20260605.csv")
    p.add_argument("--group-metadata", type=Path, default=DATA / "industry_theme_pit_ffill_20260605.csv")
    p.add_argument("--train-forward-models", action="store_true")
    p.add_argument("--middle-base-config", type=Path, default=DEFAULT_MIDDLE_BASE)
    p.add_argument("--outer-base-config", type=Path, default=DEFAULT_OUTER_BASE)
    p.add_argument("--middle-prediction", type=Path)
    p.add_argument("--outer-prediction", type=Path)
    p.add_argument("--previous-target", type=Path)
    p.add_argument("--initial-launch", action="store_true")
    p.add_argument("--candidate-template", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--gate-config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--local-log-dir", type=Path, default=DEFAULT_LOCAL_LOG)
    p.add_argument("--gm-compare", type=Path, default=DEFAULT_GM_COMPARE)
    p.add_argument("--holdout-summary", type=Path, default=DEFAULT_HOLDOUT)
    p.add_argument("--sellblock-summary", type=Path, default=DEFAULT_SELLBLOCK)
    p.add_argument("--staleexit-summary", type=Path, default=DEFAULT_STALEEXIT)
    p.add_argument("--as-of-date", default=date.today().isoformat())
    p.add_argument("--publish-on-pass", action="store_true")
    p.add_argument("--allow-gate-failure", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    result = prepare(args)
    if not result["paper_launch_allowed"] and not args.allow_gate_failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
