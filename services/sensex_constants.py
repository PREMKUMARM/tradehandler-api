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


def sensex_backtest_max_trades_per_contract_per_day() -> int:
    """Max entry/exit round-trips per tradingsymbol within one backtest session day."""
    return max(1, min(20, _env_int("SENSEX_BACKTEST_MAX_TRADES_PER_CONTRACT_PER_DAY", 5)))


def sensex_gap_pct(day_open: float, prev_close: float) -> float:
    if prev_close <= 0:
        return 0.0
    return (day_open - prev_close) / prev_close * 100.0


def sensex_is_gap_up_session(day_open: float, prev_close: float) -> bool:
    """True when index opens above prior close."""
    return prev_close > 0 and day_open > prev_close


def sensex_is_gap_down_session(day_open: float, prev_close: float) -> bool:
    """True when index opens below prior close."""
    return prev_close > 0 and day_open < prev_close


def sensex_gap_direction_kind(day_open: float, prev_close: float) -> str:
    """AUTO leg: gap-up/flat → CE, gap-down → PE."""
    if prev_close > 0 and day_open < prev_close:
        return "PE"
    return "CE"


def sensex_is_bad_option_bar(open_p: float, high: float, close: float) -> bool:
    """Reject corrupt option ticks where open/high spiked vs bar close."""
    if close <= 0:
        return False
    return open_p > 3.0 * close or high > 3.0 * close


def sensex_entry_scan_start_minutes() -> int:
    """First bar to scan for entry (default 14:00 — afternoon close-in-band, skip morning chop)."""
    hour = _env_int("SENSEX_ENTRY_START_HOUR", 14)
    minute = _env_int("SENSEX_ENTRY_START_MINUTE", 0)
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return hour * 60 + minute


def sensex_default_min_target_inr() -> float:
    """Trail activates at entry + this (default ₹10 = 1R, same as SL / 1:1 target)."""
    try:
        return max(5.0, float(os.getenv("SENSEX_MIN_TARGET_INR", "10").strip()))
    except (TypeError, ValueError):
        return 10.0


def sensex_premium_in_band(px: float, band_low: float = 17.0, band_high: float = 23.0) -> bool:
    return band_low <= px <= band_high


def sensex_atm_near_steps() -> int:
    """Rolling offsets each side of ATM to monitor (default 5 → ATM-5…ATM+5 = 11 strikes)."""
    return max(0, min(5, _env_int("SENSEX_ATM_NEAR_STEPS", 5)))


def sensex_atm_near_offsets() -> list[str]:
    """Dhan rolling offsets monitored around ATM (11 per leg by default)."""
    steps = sensex_atm_near_steps()
    if steps <= 0:
        return ["ATM"]
    minus = [f"ATM-{i}" for i in range(steps, 0, -1)]
    plus = [f"ATM+{i}" for i in range(1, steps + 1)]
    return minus + ["ATM"] + plus


def normalize_rolling_offset(offset: str) -> str:
    """
    Normalize Dhan rolling offset labels from query strings and UI.

    URL query decoding turns ``ATM+2`` into ``ATM 2`` (``+`` → space).
    """
    import re

    off = (offset or "ATM").strip().upper()
    allowed = sensex_atm_near_offsets()
    if off in allowed:
        return off
    compact = re.sub(r"\s+", "", off)
    if compact in allowed:
        return compact
    m = re.match(r"^ATM(\d+)$", compact)
    if m:
        candidate = f"ATM+{m.group(1)}"
        if candidate in allowed:
            return candidate
    m = re.match(r"^ATM\s+(\d+)$", off)
    if m:
        candidate = f"ATM+{m.group(1)}"
        if candidate in allowed:
            return candidate
    m = re.match(r"^ATM\s+-(\d+)$", off)
    if m:
        candidate = f"ATM-{m.group(1)}"
        if candidate in allowed:
            return candidate
    return off


def sensex_atm_near_strike_points() -> int:
    """Absolute strike distance from ATM included in live chain monitoring."""
    return sensex_atm_near_steps() * 100


def resolve_sensex_bfo_product(plan: dict | None = None) -> str:
    """Product for V2 entry + GTT exit. Always NRML when exit is GTT_OCO."""
    if not plan:
        return SENSEX_BFO_PRODUCT
    if str(plan.get("exit_order_type") or "GTT_OCO").upper() in ("GTT_OCO", "GTT"):
        return SENSEX_BFO_PRODUCT
    p = str(plan.get("product") or SENSEX_BFO_PRODUCT).upper()
    return SENSEX_BFO_PRODUCT if p == "MIS" else (p if p in ("NRML", "MIS", "CNC") else SENSEX_BFO_PRODUCT)
