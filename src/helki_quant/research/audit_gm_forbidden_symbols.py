from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _gm_to_local_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if "." not in text:
        return text
    exchange, code = text.split(".", 1)
    prefix = "SH" if exchange in {"SHSE", "SH"} else "SZ"
    return f"{prefix}{code}"


def _local_to_gm_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    code = text[-6:]
    if text.startswith("SH") or code.startswith(("6", "9")):
        return f"SHSE.{code}"
    return f"SZSE.{code}"


def _load_forbidden(path: Path) -> tuple[set[str], set[str]]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    local: set[str] = set()
    gm: set[str] = set()
    for column in ("instrument", "local_instrument"):
        if column in frame.columns:
            values = frame[column].astype(str).str.strip().str.upper()
            local.update(v for v in values if v)
            gm.update(_local_to_gm_symbol(v) for v in values if v)
    for column in ("gm_symbol", "symbol"):
        if column in frame.columns:
            values = frame[column].astype(str).str.strip().str.upper()
            gm.update(v for v in values if v)
            local.update(_gm_to_local_symbol(v) for v in values if v)
    return local, gm


def _audit_csv(path: Path, forbidden_gm: set[str]) -> dict[str, object]:
    if not path.exists():
        return {
            "file": str(path),
            "exists": False,
            "rows": 0,
            "forbidden_hits": 0,
            "hit_symbols": [],
        }
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "symbol" not in frame.columns:
        return {
            "file": str(path),
            "exists": True,
            "rows": int(len(frame)),
            "forbidden_hits": 0,
            "hit_symbols": [],
            "warning": "symbol column not found",
        }
    symbols = frame["symbol"].astype(str).str.strip().str.upper()
    mask = symbols.isin(forbidden_gm)
    hits = frame.loc[mask].copy()
    hit_symbols = sorted(set(hits["symbol"].astype(str).str.upper()))
    sample_columns = [c for c in ("trade_date", "event_date", "symbol", "action", "side", "status_name") if c in hits.columns]
    sample = hits[sample_columns].head(20).to_dict(orient="records") if not hits.empty else []
    return {
        "file": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "forbidden_hits": int(mask.sum()),
        "hit_symbols": hit_symbols,
        "sample": sample,
    }


def audit(forbidden_symbols: Path, audit_dirs: list[Path]) -> dict[str, object]:
    forbidden_local, forbidden_gm = _load_forbidden(forbidden_symbols)
    audits = []
    total_hits = 0
    for audit_dir in audit_dirs:
        files = [audit_dir / "submissions.csv", audit_dir / "order_status.csv"]
        file_results = [_audit_csv(path, forbidden_gm) for path in files]
        hits = sum(int(item["forbidden_hits"]) for item in file_results)
        total_hits += hits
        audits.append(
            {
                "audit_dir": str(audit_dir),
                "exists": audit_dir.exists(),
                "forbidden_hits": hits,
                "files": file_results,
            }
        )
    return {
        "status": "gm_forbidden_symbols_audit",
        "passed": total_hits == 0,
        "forbidden_symbols": str(forbidden_symbols),
        "forbidden_local_count": len(forbidden_local),
        "forbidden_gm_count": len(forbidden_gm),
        "total_forbidden_hits": total_hits,
        "audits": audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forbidden-symbols", required=True, type=Path)
    parser.add_argument("--audit-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = audit(args.forbidden_symbols.resolve(), [p.resolve() for p in args.audit_dir])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[gm forbidden audit] "
        f"passed={result['passed']} total_forbidden_hits={result['total_forbidden_hits']} "
        f"output={args.output}",
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
