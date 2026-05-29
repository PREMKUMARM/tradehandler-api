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

# MCX session (IST)
MCX_OPEN_MINUTES = 9 * 60
MCX_CLOSE_MINUTES = 23 * 60 + 30
OR_START_HOUR = 9
OR_START_MINUTE = 15
OR_END_HOUR = 9
OR_END_MINUTE = 30


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
    now = datetime.now(IST)
    if now.weekday() > 4:
        return False
    minutes = now.hour * 60 + now.minute
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
