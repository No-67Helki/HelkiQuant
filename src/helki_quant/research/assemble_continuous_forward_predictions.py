from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_prediction(path: Path, value_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime"])
    aliases = {"pred_middle": "middle", "pred_outer": "outer"}
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})
    required = {"datetime", "instrument", value_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame[["datetime", "instrument", value_col]].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    if frame[["datetime", "instrument", value_col]].isna().any().any():
        raise ValueError(f"{path} contains invalid prediction rows")
    duplicated = frame.duplicated(["datetime", "instrument"], keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, ["datetime", "instrument"]].head(5)
        raise ValueError(f"{path} contains duplicate keys: {sample.to_dict(orient='records')}")
    return frame.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _load_outer_folds(path: Path) -> pd.DataFrame:
    fold_paths = sorted(path.glob("fold_[0-9][0-9].csv"))
    if not fold_paths:
        raise FileNotFoundError(f"no strict OOF fold csv found under {path}")
    frames = [_load_prediction(fold_path, "outer") for fold_path in fold_paths]
    combined = pd.concat(frames, ignore_index=True)
    duplicated = combined.duplicated(["datetime", "instrument"], keep=False)
    if duplicated.any():
        sample = combined.loc[duplicated, ["datetime", "instrument"]].head(5)
        raise ValueError(f"outer OOF folds overlap: {sample.to_dict(orient='records')}")
    return combined.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _read_calendar(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    calendar = pd.read_csv(path, header=None, names=["datetime"], parse_dates=["datetime"])
    values = pd.DatetimeIndex(calendar["datetime"].dropna().dt.normalize().unique()).sort_values()
    return values[(values >= start) & (values <= end)]


def _date_text(values: pd.DatetimeIndex) -> list[str]:
    return [str(pd.Timestamp(value).date()) for value in values]


def assemble(
    baseline_middle_path: Path,
    baseline_outer_dir: Path,
    forward_middle_path: Path,
    forward_outer_path: Path,
    calendar_path: Path,
    output_dir: Path,
    start: str,
    baseline_end: str,
    forward_start: str,
    end: str,
) -> dict[str, object]:
    start_ts = pd.Timestamp(start).normalize()
    baseline_end_ts = pd.Timestamp(baseline_end).normalize()
    forward_start_ts = pd.Timestamp(forward_start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if not start_ts <= baseline_end_ts < forward_start_ts <= end_ts:
        raise ValueError("expected start <= baseline_end < forward_start <= end")

    calendar = _read_calendar(calendar_path, start_ts, end_ts)
    if len(calendar) == 0:
        raise ValueError("calendar contains no dates in requested interval")
    expected_forward_pos = int(calendar.searchsorted(baseline_end_ts, side="right"))
    if expected_forward_pos >= len(calendar) or calendar[expected_forward_pos] != forward_start_ts:
        expected = None if expected_forward_pos >= len(calendar) else str(calendar[expected_forward_pos].date())
        raise ValueError(
            f"forward_start={forward_start} is not the next calendar session after "
            f"baseline_end={baseline_end}; expected={expected}"
        )

    middle_oof = _load_prediction(baseline_middle_path, "middle")
    outer_oof = _load_outer_folds(baseline_outer_dir)
    middle_forward = _load_prediction(forward_middle_path, "middle")
    outer_forward = _load_prediction(forward_outer_path, "outer")

    middle_oof = middle_oof[middle_oof["datetime"].between(start_ts, baseline_end_ts)].copy()
    outer_oof = outer_oof[outer_oof["datetime"].between(start_ts, baseline_end_ts)].copy()
    middle_forward = middle_forward[middle_forward["datetime"].between(forward_start_ts, end_ts)].copy()
    outer_forward = outer_forward[outer_forward["datetime"].between(forward_start_ts, end_ts)].copy()

    middle = pd.concat([middle_oof, middle_forward], ignore_index=True).sort_values(
        ["datetime", "instrument"]
    )
    outer = pd.concat([outer_oof, outer_forward], ignore_index=True).sort_values(
        ["datetime", "instrument"]
    )
    for name, frame in (("middle", middle), ("outer", outer)):
        if frame.duplicated(["datetime", "instrument"]).any():
            raise ValueError(f"{name} has duplicate keys after assembly")

    middle_dates = pd.DatetimeIndex(middle["datetime"].unique()).sort_values()
    outer_dates = pd.DatetimeIndex(outer["datetime"].unique()).sort_values()
    missing_middle_dates = calendar.difference(middle_dates)
    missing_outer_dates = calendar.difference(outer_dates)
    if len(missing_middle_dates) or len(missing_outer_dates):
        raise ValueError(
            "prediction date gap: "
            f"middle={_date_text(missing_middle_dates)} outer={_date_text(missing_outer_dates)}"
        )

    key_check = middle[["datetime", "instrument"]].merge(
        outer[["datetime", "instrument"]],
        on=["datetime", "instrument"],
        how="outer",
        indicator=True,
    )
    missing_outer_keys = int((key_check["_merge"] == "left_only").sum())
    missing_middle_keys = int((key_check["_merge"] == "right_only").sum())
    if missing_outer_keys:
        sample = key_check[key_check["_merge"] == "left_only"].head(5)
        raise ValueError(
            f"outer is missing {missing_outer_keys} middle keys: "
            f"{sample.to_dict(orient='records')}"
        )

    outer_spread = outer.groupby("datetime")["outer"].agg(lambda values: float(values.max() - values.min()))
    max_outer_daily_spread = float(outer_spread.max()) if len(outer_spread) else np.nan
    if not np.isfinite(max_outer_daily_spread) or max_outer_daily_spread > 1e-10:
        raise ValueError(f"outer regime prediction is not daily-constant: max spread={max_outer_daily_spread}")

    output_dir.mkdir(parents=True, exist_ok=True)
    middle_path = output_dir / "middle_continuous.csv"
    outer_path = output_dir / "outer_continuous.csv"
    middle.to_csv(middle_path, index=False, encoding="utf-8")
    outer.to_csv(outer_path, index=False, encoding="utf-8")

    report: dict[str, object] = {
        "status": "continuous_oof_plus_frozen_forward_predictions_assembled",
        "window": {
            "start": str(start_ts.date()),
            "baseline_end": str(baseline_end_ts.date()),
            "forward_start": str(forward_start_ts.date()),
            "end": str(end_ts.date()),
        },
        "sources": {
            "baseline_middle": str(baseline_middle_path.resolve()),
            "baseline_outer_dir": str(baseline_outer_dir.resolve()),
            "forward_middle": str(forward_middle_path.resolve()),
            "forward_outer": str(forward_outer_path.resolve()),
            "calendar": str(calendar_path.resolve()),
        },
        "outputs": {
            "middle": str(middle_path.resolve()),
            "outer": str(outer_path.resolve()),
        },
        "calendar_dates": int(len(calendar)),
        "middle": {
            "rows": int(len(middle)),
            "dates": int(len(middle_dates)),
            "instruments": int(middle["instrument"].nunique()),
            "first_date": str(middle_dates.min().date()),
            "last_date": str(middle_dates.max().date()),
        },
        "outer": {
            "rows": int(len(outer)),
            "dates": int(len(outer_dates)),
            "instruments": int(outer["instrument"].nunique()),
            "first_date": str(outer_dates.min().date()),
            "last_date": str(outer_dates.max().date()),
            "max_daily_spread": max_outer_daily_spread,
        },
        "cross_layer": {
            "missing_outer_keys": missing_outer_keys,
            "outer_only_keys": missing_middle_keys,
        },
        "leakage_note": (
            "Rows through baseline_end are strict OOF predictions. Rows from forward_start are "
            "predictions from frozen models fitted and early-stopped before the forward interval."
        ),
        "deployment_allowed": False,
    }
    report_path = output_dir / "manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-middle", required=True)
    parser.add_argument("--baseline-outer-dir", required=True)
    parser.add_argument("--forward-middle", required=True)
    parser.add_argument("--forward-outer", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default="2025-01-03")
    parser.add_argument("--baseline-end", default="2026-04-02")
    parser.add_argument("--forward-start", default="2026-04-03")
    parser.add_argument("--end", default="2026-06-04")
    args = parser.parse_args()
    report = assemble(
        Path(args.baseline_middle).resolve(),
        Path(args.baseline_outer_dir).resolve(),
        Path(args.forward_middle).resolve(),
        Path(args.forward_outer).resolve(),
        Path(args.calendar).resolve(),
        Path(args.output_dir).resolve(),
        args.start,
        args.baseline_end,
        args.forward_start,
        args.end,
    )
    print(
        "[continuous predictions] "
        f"dates={report['calendar_dates']} middle_rows={report['middle']['rows']} "
        f"outer_rows={report['outer']['rows']} "
        f"missing_outer_keys={report['cross_layer']['missing_outer_keys']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
