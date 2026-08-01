from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


INSTRUMENT_COLS = ("instrument", "symbol", "ts_code", "code")
START_COLS = ("start_date", "effective_start", "from_date", "entry_date")
END_COLS = ("end_date", "effective_end", "to_date", "exit_date")


@dataclass(frozen=True)
class ConcentrationRules:
    group_col: str = "industry"
    max_group_fraction: float = 0.40


def normalize_instrument(value: object) -> str:
    text = str(value).strip().upper()
    if not text:
        return text
    if text.startswith(("SH", "SZ")) and len(text) >= 8:
        return text[:2] + text[-6:]
    code = text.replace(".", "")[-6:]
    if code.startswith("68"):
        return f"SH{code}"
    return f"SZ{code}"


def _first_existing(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def load_group_metadata(path: Path, group_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    inst_col = _first_existing(list(frame.columns), INSTRUMENT_COLS)
    if inst_col is None:
        raise ValueError(
            f"{path} must include one instrument column: {', '.join(INSTRUMENT_COLS)}"
        )
    actual_group_col = _first_existing(list(frame.columns), (group_col, "industry", "sector", "theme", "concept"))
    if actual_group_col is None:
        raise ValueError(f"{path} must include group column '{group_col}'")
    start_col = _first_existing(list(frame.columns), START_COLS)
    end_col = _first_existing(list(frame.columns), END_COLS)
    out = pd.DataFrame(
        {
            "instrument": frame[inst_col].map(normalize_instrument),
            "group": frame[actual_group_col].astype(str).str.strip(),
            "start_date": pd.Timestamp("1900-01-01"),
            "end_date": pd.Timestamp("2099-12-31"),
        }
    )
    if start_col is not None:
        out["start_date"] = pd.to_datetime(frame[start_col], errors="coerce").fillna(out["start_date"])
    if end_col is not None:
        out["end_date"] = pd.to_datetime(frame[end_col], errors="coerce").fillna(out["end_date"])
    out = out[(out["instrument"] != "") & (out["group"] != "")].drop_duplicates()
    out["is_pit"] = start_col is not None or end_col is not None
    return out


def groups_on_date(metadata: pd.DataFrame, trade_date: pd.Timestamp) -> dict[str, str]:
    date = pd.Timestamp(trade_date).normalize()
    active = metadata[
        (metadata["start_date"] <= date)
        & (metadata["end_date"] >= date)
    ].sort_values(["instrument", "start_date"])
    active = active.drop_duplicates("instrument", keep="last")
    return active.set_index("instrument")["group"].to_dict()


def select_with_group_cap(
    ranked: list[str],
    previous_selection: set[str],
    *,
    top_k: int,
    buffer_k: int,
    groups: dict[str, str] | None,
    rules: ConcentrationRules | None,
) -> list[str]:
    buffer = set(ranked[:buffer_k])
    retained = [inst for inst in ranked if inst in previous_selection and inst in buffer]
    candidates = retained + [inst for inst in ranked if inst not in retained]
    if not groups or rules is None:
        return candidates[:top_k]
    group_limit = max(1, int(np.floor(top_k * rules.max_group_fraction)))
    selected: list[str] = []
    group_counts: dict[str, int] = {}
    for inst in candidates:
        group = groups.get(inst, "__UNKNOWN__")
        if group_counts.get(group, 0) >= group_limit:
            continue
        selected.append(inst)
        group_counts[group] = group_counts.get(group, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def concentration_snapshot(instruments: list[str], groups: dict[str, str] | None) -> dict:
    if not groups:
        return {"available": False}
    values = [groups.get(inst, "__UNKNOWN__") for inst in instruments]
    counts = pd.Series(values).value_counts()
    return {
        "available": True,
        "max_group": str(counts.index[0]) if len(counts) else None,
        "max_group_count": int(counts.iloc[0]) if len(counts) else 0,
        "max_group_fraction": float(counts.iloc[0] / len(values)) if len(values) else 0.0,
        "groups": counts.to_dict(),
    }
