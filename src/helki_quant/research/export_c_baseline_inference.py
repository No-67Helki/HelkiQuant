from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_c_baseline_production_logs import run


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "data"
DEFAULT_MIDDLE = HERE / "outputs" / "oof_combined" / "middle_oof_pit_de2_srfs_es.csv"
DEFAULT_GROUP_METADATA = DATA / "industry_theme_pit.csv"
DEFAULT_WINDOWS = DATA / "_research_topk_minute_windows_2025_2026.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Production-style C-baseline inference/export entry. This is a "
            "thin, explicit CLI around the validated C baseline accounting "
            "engine and is the path to compare against research logs."
        )
    )
    parser.add_argument("--middle", default=str(DEFAULT_MIDDLE))
    parser.add_argument("--minute-windows", default=str(DEFAULT_WINDOWS))
    parser.add_argument("--group-metadata", default=str(DEFAULT_GROUP_METADATA))
    parser.add_argument("--output-dir", default=str(HERE / "outputs" / "production_inference_c_baseline"))
    parser.add_argument("--start-signal", default="2025-01-03")
    parser.add_argument("--end-signal", default="2026-04-02")
    parser.add_argument("--cost", choices=["base", "stress"], default="stress")
    parser.add_argument("--initial-cash", type=float, default=500_000.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "c_baseline_inference_export_config",
        "middle": str(Path(args.middle).resolve()),
        "minute_windows": str(Path(args.minute_windows).resolve()),
        "group_metadata": str(Path(args.group_metadata).resolve()),
        "output_dir": str(output_dir),
        "start_signal": args.start_signal,
        "end_signal": args.end_signal,
        "cost": args.cost,
        "initial_cash": args.initial_cash,
        "deployment_allowed": False,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run(
        Path(args.middle).resolve(),
        Path(args.minute_windows).resolve(),
        Path(args.group_metadata).resolve(),
        output_dir,
        args.start_signal,
        args.end_signal,
        args.cost,
        args.initial_cash,
    )


if __name__ == "__main__":
    main()
