"""MCX commodity wizard constants (product-aware via commodity_product_context)."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from services.commodity_product_context import get_active_product

IST = ZoneInfo("Asia/Kolkata")

EXCHANGE = "MCX"
COMMODITY_PRODUCT = "NRML"
DEFAULT_NUM_LOTS = 1

# MCX exchange hours (IST) — quotes / exchange technically open until 23:30
MCX_OPEN_MINUTES = 9 * 60
MCX_CLOSE_MINUTES = 23 * 60 + 30
OR_START_HOUR = 9
OR_START_MINUTE = 15
OR_END_HOUR = 9
OR_END_MINUTE = 30


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return default


def commodity_trading_cutoff_minutes() -> int:
    """Last minute we allow new commodity entries (default 23:15 IST)."""
    hour = _env_int("COMMODITY_TRADING_CUTOFF_HOUR", 23)
    minute = _env_int("COMMODITY_TRADING_CUTOFF_MINUTE", 15)
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return hour * 60 + minute


def commodity_trading_cutoff_label() -> str:
    mins = commodity_trading_cutoff_minutes()
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _ist_minutes_now() -> tuple[int, bool]:
    now = datetime.now(IST)
    return now.hour * 60 + now.minute, now.weekday() <= 4


def is_past_commodity_trading_cutoff() -> bool:
    minutes, is_weekday = _ist_minutes_now()
    if not is_weekday:
        return True
    return minutes >= commodity_trading_cutoff_minutes()


def is_commodity_new_trading_allowed() -> bool:
    """New commodity entries / autonomous watch placement."""
    minutes, is_weekday = _ist_minutes_now()
    if not is_weekday:
        return False
    if minutes < MCX_OPEN_MINUTES:
        return False
    return minutes < commodity_trading_cutoff_minutes()


def future_symbol() -> str:
    return get_active_product().future_symbol


def option_prefix() -> str:
    return get_active_product().option_prefix


def strike_step() -> float:
    return get_active_product().strike_step


def units_per_lot() -> int:
    return get_active_product().units_per_lot


def product_label() -> str:
    return get_active_product().label


# Legacy module-level defaults (CRUDEOILM only)
FUTURE_SYMBOL = os.getenv("COMMODITY_FUTURE_SYMBOL", "CRUDEOILM26JUNFUT").strip() or "CRUDEOILM26JUNFUT"
OPTION_PREFIX = os.getenv("COMMODITY_OPTION_PREFIX", "CRUDEOILM26JUN").strip() or "CRUDEOILM26JUN"
STRIKE_STEP = float(os.getenv("COMMODITY_STRIKE_STEP", "50") or 50)
DEFAULT_LOT_SIZE = int(os.getenv("COMMODITY_UNITS_PER_LOT", "10") or 10)


def is_mcx_session_open() -> bool:
    """Exchange session for quotes (09:00–23:30 IST Mon–Fri)."""
    minutes, is_weekday = _ist_minutes_now()
    if not is_weekday:
        return False
    return MCX_OPEN_MINUTES <= minutes < MCX_CLOSE_MINUTES


def allow_offhours_commodity_place() -> bool:
    return os.getenv("COMMODITY_ALLOW_OFFHOURS_PLACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def live_fixed_order_qty() -> Optional[int]:
    """
    Fixed Kite qty for commodity live mode (paper mode ignores this).
    Set COMMODITY_LIVE_FIXED_QTY=0 to restore auto sizing from risk % / sizing fund.
    Default 1 for live testing.
    """
    raw = os.getenv("COMMODITY_LIVE_FIXED_QTY", "1").strip().lower()
    if raw in ("0", "false", "no", "off", "none", ""):
        return None
    try:
        return max(1, min(50, int(float(raw))))
    except ValueError:
        return 1


def resolve_commodity_product(plan: dict | None = None) -> str:
    return COMMODITY_PRODUCT
