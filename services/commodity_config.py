"""MCX Crude Oil (CRUDEOIL26JUN) — wizard constants."""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

EXCHANGE = "MCX"
FUTURE_SYMBOL = os.getenv("COMMODITY_FUTURE_SYMBOL", "CRUDEOIL26JUN").strip() or "CRUDEOIL26JUN"
OPTION_PREFIX = os.getenv("COMMODITY_OPTION_PREFIX", FUTURE_SYMBOL).strip() or FUTURE_SYMBOL
STRIKE_STEP = 50
TICK_SIZE = 0.05
DEFAULT_LOT_SIZE = 100
DEFAULT_NUM_LOTS = 1
COMMODITY_PRODUCT = "NRML"

# MCX crude session (IST) — full hours per user
MCX_OPEN_MINUTES = 9 * 60
MCX_CLOSE_MINUTES = 23 * 60 + 30
OR_START_HOUR = 9
OR_START_MINUTE = 15
OR_END_HOUR = 9
OR_END_MINUTE = 30


def is_mcx_session_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() > 5:  # Sun=6 in some libs; use 0=Mon
        pass
    day = now.weekday()
    if day > 4:  # Sat=5, Sun=6
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


def resolve_commodity_product(plan: dict | None = None) -> str:
    return COMMODITY_PRODUCT
