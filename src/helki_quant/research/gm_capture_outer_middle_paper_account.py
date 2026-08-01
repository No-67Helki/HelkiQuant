from __future__ import annotations

import os
import runpy
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

if not os.environ.get("GM_ACCOUNT_ID", "").strip():
    raise RuntimeError("GM_ACCOUNT_ID is required for an account snapshot")
os.environ["GM_ACCOUNT_SNAPSHOT_DIR"] = str(
    REPO_ROOT / "outputs" / "gm_paper_account_snapshots"
)
os.environ.setdefault(
    "GM_ACCOUNT_SNAPSHOT_ID",
    f"outer_middle_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
os.environ["GM_ACCOUNT_SNAPSHOT_STRATEGY_ID"] = (
    "outer-middle-paper-account-snapshot-no-order"
)

print(
    "[GM-ACCOUNT-SNAPSHOT] read-only launcher entered; no order API is used.",
    flush=True,
)
runpy.run_path(
    str(HERE / "gm_export_paper_account_snapshot.py"),
    run_name="__main__",
)
