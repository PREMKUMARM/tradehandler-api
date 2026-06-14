"""
Sensex wizard constants — Zerodha product rules for BFO options.
"""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# BSE cash/F&O session (IST)
SENSEX_SESSION_OPEN_MINUTES = 9 * 60 + 15
SENSEX_SESSION_CLOSE_MINUTES = 15 * 60 + 30

# GTT on NFO/BFO is not supported for MIS (intraday). Entry and GTT legs must use NRML.
SENSEX_BFO_PRODUCT = os.getenv("SENSEX_BFO_PRODUCT", "NRML").upper()
if SENSEX_BFO_PRODUCT not in ("NRML",):
    SENSEX_BFO_PRODUCT = "NRML"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return default


def sensex_entry_cutoff_minutes() -> int:
    """Last minute we allow new 20rupees entries (default 15:00 IST — 30m before close)."""
    hour = _env_int("SENSEX_ENTRY_CUTOFF_HOUR", 15)
    minute = _env_int("SENSEX_ENTRY_CUTOFF_MINUTE", 0)
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return hour * 60 + minute


def sensex_entry_cutoff_label() -> str:
    mins = sensex_entry_cutoff_minutes()
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _ist_minutes_now() -> tuple[int, bool]:
    now = datetime.now(IST)
    return now.hour * 60 + now.minute, 1 <= now.weekday() <= 5


def is_past_sensex_entry_cutoff() -> bool:
    minutes, is_weekday = _ist_minutes_now()
    if not is_weekday:
        return True
    return minutes >= sensex_entry_cutoff_minutes()


def is_sensex_new_entry_allowed() -> bool:
    """New Sensex 20rupees entries / autonomous placement during the session."""
    minutes, is_weekday = _ist_minutes_now()
    if not is_weekday:
        return False
    if minutes < SENSEX_SESSION_OPEN_MINUTES:
        return False
    return minutes < sensex_entry_cutoff_minutes()


def sensex_entry_cutoff_message() -> str:
    return (
        f"No new Sensex entries after {sensex_entry_cutoff_label()} IST "
        f"(avoid last-minute gamma and settlement decay)"
    )


def sensex_max_lots_per_trade() -> int:
    """Upper cap for 20rupees risk-based sizing (matches BFO wizard schema max)."""
    return max(1, min(50, _env_int("SENSEX_MAX_LOTS", 50)))


def resolve_sensex_bfo_product(plan: dict | None = None) -> str:
    """Product for V2 entry + GTT exit. Always NRML when exit is GTT_OCO."""
    if not plan:
        return SENSEX_BFO_PRODUCT
    if str(plan.get("exit_order_type") or "GTT_OCO").upper() in ("GTT_OCO", "GTT"):
        return SENSEX_BFO_PRODUCT
    p = str(plan.get("product") or SENSEX_BFO_PRODUCT).upper()
    return SENSEX_BFO_PRODUCT if p == "MIS" else (p if p in ("NRML", "MIS", "CNC") else SENSEX_BFO_PRODUCT)
