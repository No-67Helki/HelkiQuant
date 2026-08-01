from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA = REPO_ROOT / "data"
DEFAULT_SOURCE_PROVIDER = DATA / "cn_data_canonical_pit_20260605"


LABEL_SPECS = {
    "OUTER_TOP150_FWD_20D": "top150_fwd_20d",
    "OUTER_TOP150_MDD_20D": "top150_mdd_20d",
    "OUTER_TOP150_DOWNSIDE_VOL_20D": "top150_downside_vol_20d",
    "OUTER_TOP150_ADVERSE_MDD5_20D": "top150_adverse_mdd5_20d",
    "OUTER_TOP150_ADVERSE_LOSS5_20D": "top150_adverse_loss5_20d",
    "OUTER_TOP150_ADVERSE_COMBO_20D": "top150_adverse_combo_20d",
}


def read_calendar(provider: Path) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, int]]:
    calendar = pd.read_csv(
        provider / "calendars" / "day.txt",
        header=None,
        names=["date"],
        parse_dates=["date"],
    )["date"].tolist()
    return calendar, {date: idx for idx, date in enumerate(calendar)}


def read_instruments(provider: Path) -> list[str]:
    path = provider / "instruments" / "all.txt"
    return [
        line.split()[0].lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_qlib_bin(values: np.ndarray, start_idx: int, bin_path: Path) -> None:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    np.hstack([np.array([start_idx], dtype=np.float32), values.astype(np.float32)]).tofile(
        str(bin_path.resolve())
    )


def prepare_labels(daily_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(daily_path, parse_dates=["datetime"])
    frame["datetime"] = frame["datetime"].dt.normalize()
    frame = frame.sort_values("datetime").drop_duplicates("datetime", keep="last")
    if "top150_mdd_20d" not in frame or "top150_fwd_20d" not in frame:
        raise KeyError("daily labels must include top150_mdd_20d and top150_fwd_20d")
    frame["top150_adverse_mdd5_20d"] = (frame["top150_mdd_20d"] > 0.05).astype(float)
    frame["top150_adverse_loss5_20d"] = (frame["top150_fwd_20d"] < -0.05).astype(float)
    frame["top150_adverse_combo_20d"] = (
        (frame["top150_mdd_20d"] > 0.05) | (frame["top150_fwd_20d"] < -0.05)
    ).astype(float)
    return frame.set_index("datetime")


def copy_provider(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing provider: {output}")
    shutil.copytree(source, output)


def augment(
    source_provider: Path,
    output_provider: Path,
    daily_labels: Path,
    report_path: Path,
    skip_copy: bool,
) -> dict:
    if skip_copy:
        if not output_provider.exists():
            raise FileNotFoundError(f"--skip-copy requires existing output provider: {output_provider}")
    else:
        copy_provider(source_provider, output_provider)
    calendar, cal_index = read_calendar(output_provider)
    labels = prepare_labels(daily_labels)
    valid_dates = [date for date in labels.index if date in cal_index]
    labels = labels.loc[valid_dates]
    if labels.empty:
        raise ValueError("no label dates overlap provider calendar")
    start_idx = cal_index[labels.index.min()]
    end_idx = cal_index[labels.index.max()]
    aligned = labels.reindex(calendar[start_idx : end_idx + 1])
    instruments = read_instruments(output_provider)
    for pos, inst in enumerate(instruments, start=1):
        feature_dir = output_provider / "features" / inst
        for field, source_col in LABEL_SPECS.items():
            write_qlib_bin(
                aligned[source_col].to_numpy(dtype=np.float32),
                start_idx,
                feature_dir / f"{field.lower()}.day.bin",
            )
        if pos % 250 == 0:
            print(f"[outer regime labels] {pos}/{len(instruments)}", flush=True)
    report = {
        "status": "outer_regime_labels_augmented",
        "source_provider": str(source_provider.resolve()),
        "output_provider": str(output_provider.resolve()),
        "daily_labels": str(daily_labels.resolve()),
        "date_start": str(labels.index.min().date()),
        "date_end": str(labels.index.max().date()),
        "days": int(len(labels)),
        "instruments": int(len(instruments)),
        "fields": LABEL_SPECS,
        "base_rates": {
            field: float(aligned[source_col].mean())
            for field, source_col in LABEL_SPECS.items()
            if source_col.startswith("top150_adverse")
        },
        "deployment_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-provider", default=str(DEFAULT_SOURCE_PROVIDER))
    parser.add_argument("--output-provider", required=True)
    parser.add_argument("--daily-labels", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Write labels into an already-created output provider.",
    )
    args = parser.parse_args()
    report = augment(
        Path(args.source_provider).resolve(),
        Path(args.output_provider).resolve(),
        Path(args.daily_labels).resolve(),
        Path(args.report).resolve(),
        args.skip_copy,
    )
    print(
        "[outer regime labels] "
        f"instruments={report['instruments']} days={report['days']} "
        f"provider={report['output_provider']}"
    )


if __name__ == "__main__":
    main()
