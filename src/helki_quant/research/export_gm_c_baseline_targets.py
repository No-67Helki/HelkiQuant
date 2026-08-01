from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "outputs" / "production_logs_c_baseline"
DEFAULT_OUTPUT = HERE / "outputs" / "gm_c_baseline_targets"
DEFAULT_FORBIDDEN = HERE.parents[2] / "data" / "forbidden_st_symbols_20260605.csv"


def to_gm_symbol(instrument: str) -> str:
    text = str(instrument).strip().upper()
    code = text[-6:]
    if text.startswith("SH") or code.startswith("68"):
        return f"SHSE.{code}"
    return f"SZSE.{code}"


def min_buy_shares(symbol: str) -> int:
    code = str(symbol).upper()[-6:]
    if code.startswith(("688", "689")):
        return 200
    return 100


def gm_to_local_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if "." not in text:
        return text
    exchange, code = text.split(".", 1)
    prefix = "SH" if exchange in {"SHSE", "SH"} else "SZ"
    return f"{prefix}{code}"


def load_forbidden(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None or not path.exists():
        return set(), set()
    frame = pd.read_csv(path, dtype=str).fillna("")
    local: set[str] = set()
    gm: set[str] = set()
    if "instrument" in frame.columns:
        local.update(frame["instrument"].astype(str).str.upper())
    if "gm_symbol" in frame.columns:
        gm.update(frame["gm_symbol"].astype(str).str.upper())
        local.update(frame["gm_symbol"].map(gm_to_local_symbol).astype(str).str.upper())
    return local, gm


def export_one(profile_dir: Path, output_dir: Path, forbidden_path: Path | None = DEFAULT_FORBIDDEN) -> dict:
    targets_path = profile_dir / "targets.csv"
    if not targets_path.exists():
        raise FileNotFoundError(targets_path)
    targets = pd.read_csv(targets_path, parse_dates=["trade_date"])
    targets = targets[
        (targets["mapped"].astype(int) == 1)
        & (pd.to_numeric(targets["target_weight"], errors="coerce") > 0)
    ].copy()
    if "target_shares" not in targets.columns:
        targets["target_shares"] = 0
    targets["target_shares"] = pd.to_numeric(targets["target_shares"], errors="coerce").fillna(0).astype(int)
    targets = targets[targets["target_shares"] > 0].copy()
    targets["symbol"] = targets["instrument"].map(to_gm_symbol)
    forbidden_local, forbidden_gm = load_forbidden(forbidden_path)
    before_forbidden = len(targets)
    if forbidden_local or forbidden_gm:
        targets = targets[
            ~targets["instrument"].astype(str).str.upper().isin(forbidden_local)
            & ~targets["symbol"].astype(str).str.upper().isin(forbidden_gm)
        ].copy()
    forbidden_rows_removed = before_forbidden - len(targets)
    out = targets[
        ["trade_date", "symbol", "instrument", "rank", "middle", "target_weight", "target_shares", "group"]
    ].copy()
    out["trade_date"] = out["trade_date"].dt.strftime("%Y-%m-%d")
    out = out.sort_values(["trade_date", "rank", "symbol"]).reset_index(drop=True)
    effective_rows = []
    previous: dict[str, int] = {}
    for _, row in out.iterrows():
        symbol = str(row["symbol"])
        raw_shares = int(row["target_shares"])
        prev_shares = previous.get(symbol, 0)
        if raw_shares > prev_shares and raw_shares - prev_shares < min_buy_shares(symbol):
            effective_shares = prev_shares
        else:
            effective_shares = raw_shares
        previous[symbol] = effective_shares
        if effective_shares <= 0:
            continue
        row = row.copy()
        row["target_shares"] = effective_shares
        effective_rows.append(row)
    out = pd.DataFrame(effective_rows, columns=out.columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{profile_dir.name}.csv"
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "profile": profile_dir.name,
        "output": str(output_path),
        "rows": int(len(out)),
        "rebalance_dates": int(out["trade_date"].nunique()),
        "symbols": int(out["symbol"].nunique()),
        "first_date": str(out["trade_date"].min()) if len(out) else None,
        "last_date": str(out["trade_date"].max()) if len(out) else None,
        "target_weight_sum_min": float(out.groupby("trade_date")["target_weight"].sum().min()) if len(out) else 0.0,
        "target_weight_sum_max": float(out.groupby("trade_date")["target_weight"].sum().max()) if len(out) else 0.0,
        "forbidden_rows_removed": int(forbidden_rows_removed),
        "forbidden_path": str(forbidden_path) if forbidden_path else None,
    }


def export_all(source_dir: Path, output_dir: Path, default_profile: str, forbidden_path: Path | None) -> dict:
    profiles = sorted(path for path in source_dir.iterdir() if path.is_dir())
    rows = [export_one(profile, output_dir, forbidden_path) for profile in profiles]
    manifest = {
        "status": "gm_c_baseline_targets_exported",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "profiles": rows,
        "default_profile": default_profile,
        "forbidden_path": str(forbidden_path) if forbidden_path else None,
        "forbidden_rows_removed_total": int(sum(row.get("forbidden_rows_removed", 0) for row in rows)),
        "deployment_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--default-profile", default="c_top150_rb45_risk0.80_cap0.30_stress")
    parser.add_argument("--forbidden-symbols", default=str(DEFAULT_FORBIDDEN))
    args = parser.parse_args()
    manifest = export_all(
        Path(args.source_dir).resolve(),
        Path(args.output_dir).resolve(),
        args.default_profile,
        Path(args.forbidden_symbols).resolve() if args.forbidden_symbols else None,
    )
    print(
        f"[gm targets] profiles={len(manifest['profiles'])} "
        f"output={manifest['output_dir']}",
        flush=True,
    )
    for row in manifest["profiles"]:
        print(
            f"  {row['profile']}: rows={row['rows']} dates={row['rebalance_dates']} "
            f"weight={row['target_weight_sum_min']:.2%}-{row['target_weight_sum_max']:.2%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
