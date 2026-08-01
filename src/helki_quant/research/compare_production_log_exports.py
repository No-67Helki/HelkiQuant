from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = (
    "targets.csv",
    "orders.csv",
    "fills.csv",
    "daily_account.csv",
    "holdings.csv",
    "audit.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def canonical_json_hash(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data.pop("output_dir", None)
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_dirs(reference_dir: Path, candidate_dir: Path, output_path: Path) -> dict:
    profile_dirs = sorted(path.name for path in reference_dir.iterdir() if path.is_dir())
    rows = []
    all_match = True
    for profile in profile_dirs:
        ref_profile = reference_dir / profile
        cand_profile = candidate_dir / profile
        for name in FILES:
            ref = ref_profile / name
            cand = cand_profile / name
            exists = ref.exists() and cand.exists()
            if name.endswith(".json"):
                ref_hash = canonical_json_hash(ref) if ref.exists() else None
                cand_hash = canonical_json_hash(cand) if cand.exists() else None
            else:
                ref_hash = sha256(ref) if ref.exists() else None
                cand_hash = sha256(cand) if cand.exists() else None
            match = bool(exists and ref_hash == cand_hash)
            all_match = all_match and match
            rows.append(
                {
                    "profile": profile,
                    "file": name,
                    "reference_exists": ref.exists(),
                    "candidate_exists": cand.exists(),
                    "reference_rows": count_rows(ref) if ref.exists() else None,
                    "candidate_rows": count_rows(cand) if cand.exists() else None,
                    "reference_sha256": ref_hash,
                    "candidate_sha256": cand_hash,
                    "match": match,
                }
            )
    report = {
        "status": "production_export_comparison",
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "profiles": profile_dirs,
        "files_checked": list(FILES),
        "all_match": all_match,
        "comparisons": rows,
        "deployment_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[compare exports] profiles={len(profile_dirs)} files={len(rows)} "
        f"all_match={all_match} output={output_path}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare_dirs(
        Path(args.reference_dir).resolve(),
        Path(args.candidate_dir).resolve(),
        Path(args.output).resolve(),
    )


if __name__ == "__main__":
    main()
