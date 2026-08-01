from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
MULTI_LAYER = HERE.parent
if str(MULTI_LAYER) not in sys.path:
    sys.path.insert(0, str(MULTI_LAYER))

from realtime_output import run_streaming, setup_realtime_output


def main() -> None:
    setup_realtime_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-dir", required=True)
    parser.add_argument("--config-name", required=True, help="For example simple.yaml or de2_srfs_es.yaml")
    parser.add_argument("--layer", choices=["outer", "middle", "inner"], required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--select-factors", action="store_true")
    parser.add_argument(
        "--whitelist-dir",
        default="",
        help=(
            "Reuse feature_whitelist_<layer>_v2.json from fold_XX under this directory. "
            "Mutually exclusive with --select-factors."
        ),
    )
    parser.add_argument(
        "--feature-blacklist",
        nargs="*",
        default=[],
        help="Feature names to exclude when --select-factors is used.",
    )
    parser.add_argument("--output-dir", default=str(HERE / "outputs"))
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()
    if args.select_factors and args.whitelist_dir:
        parser.error("use either --select-factors or --whitelist-dir, not both")

    configs_dir = Path(args.configs_dir).resolve()
    whitelist_dir = Path(args.whitelist_dir).resolve() if args.whitelist_dir else None
    log_dir = Path(args.log_dir).resolve() if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        config_path = configs_dir / f"fold_{fold:02d}" / args.config_name
        if not config_path.exists():
            raise FileNotFoundError(f"missing fold config: {config_path}")
        command = [
            sys.executable,
            str(HERE / "run_oof.py"),
            "--config",
            str(config_path),
            "--layer",
            args.layer,
            "--fold",
            str(fold),
            "--variant",
            args.variant,
            "--output-dir",
            str(Path(args.output_dir).resolve()),
        ]
        if args.select_factors:
            command.append("--select-factors")
            if args.feature_blacklist:
                command.append("--feature-blacklist")
                command.extend(args.feature_blacklist)
        elif whitelist_dir is not None:
            whitelist_path = (
                whitelist_dir
                / f"fold_{fold:02d}"
                / f"feature_whitelist_{args.layer}_v2.json"
            )
            if not whitelist_path.exists():
                raise FileNotFoundError(f"missing fold whitelist: {whitelist_path}")
            command.extend(["--whitelist-path", str(whitelist_path)])
        log_path = (
            log_dir / f"{args.variant}_{args.layer}_fold_{fold:02d}.log"
            if log_dir is not None
            else None
        )
        rc = run_streaming(
            command,
            cwd=HERE,
            prefix=f"[fold {fold:02d}] ",
            log_path=log_path,
        )
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
