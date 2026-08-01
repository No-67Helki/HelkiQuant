# coding=utf-8
"""
GmQuant PAPER-simulation wrapper for the outer-direct-loss5 market-filtered
candidate.

This wrapper is intentionally explicit: it does not replace the repository
default target and it does not enable inner T+0. The loaded target CSV must
contain a non-stale target date accepted by main.py's LIVE/PAPER guard.
It must also carry one completed, recent signal_date per target date; changing
only trade_date cannot make an old signal deployable.
"""
from __future__ import annotations

import os
import runpy
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent

os.environ["GM_MODE"] = "LIVE"
os.environ["GM_C_TRADING_ENV"] = "PAPER"
os.environ["GM_C_ALLOW_LIVE"] = "C_BASELINE_APPROVED_FOR_PAPER_TRADING"
os.environ["GM_STRATEGY_ID"] = "outer-direct-loss5-mf-paper"
if not os.environ.get("GM_ACCOUNT_ID", "").strip():
    raise RuntimeError("GM_ACCOUNT_ID must identify the selected simulation account")
os.environ["GM_C_REQUIRE_ACCOUNT_ID"] = "1"
os.environ["GM_C_VERBOSE_ORDERS"] = "1"
os.environ["GM_C_TARGETS"] = str(ROOT / "gm_c_baseline_targets.csv")
os.environ["GM_C_FORBIDDEN_SYMBOLS"] = str(ROOT / "gm_c_forbidden_symbols.csv")
os.environ["GM_C_DYNAMIC_ST_CHECK"] = "1"
os.environ["GM_C_DYNAMIC_ST_FAIL_CLOSED"] = "1"
os.environ["GM_C_AUDIT_DIR"] = str(ROOT / "outputs" / "gm_audit_paper_candidate")
os.environ["GM_C_ACTIVATION_REGISTRY"] = str(
    ROOT / "outputs" / "gm_audit_paper_candidate" / "PAPER_ACTIVATION_REGISTRY.jsonl"
)
os.environ["GM_C_REQUIRE_ACTIVATION_AUDIT"] = "1"
os.environ.setdefault(
    "GM_C_AUDIT_RUN_ID",
    f"paper_outer_direct_loss5_mf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
os.environ["GM_C_INITIAL_CASH"] = "1000000"
os.environ["GM_C_ORDER_STYLE"] = "VOLUME"
os.environ["GM_C_REBALANCE_PHASE"] = "AT_ONCE"
os.environ["GM_C_EXECUTION_MODE"] = "TICK_EXEC"
os.environ["GM_C_SUBSCRIBE_TARGETS"] = "1"
os.environ["GM_C_REBALANCE_ON_INIT"] = "1"
os.environ["GM_C_SYNC_EXISTING_POSITIONS"] = "1"
os.environ["GM_C_MAX_TICK_SUBSCRIPTIONS"] = "45"
os.environ["GM_C_MAX_LIVE_TARGET_FORWARD_DAYS"] = "7"
os.environ["GM_C_REQUIRE_SIGNAL_DATE"] = "1"
os.environ["GM_C_MAX_LIVE_SIGNAL_AGE_DAYS"] = "4"
os.environ["GM_C_MAX_LIVE_SIGNAL_TO_TARGET_DAYS"] = "4"

runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
