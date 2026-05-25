"""MCX Crude Oil Mini (CRUDEOILM) — sole commodity product for /commodity."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Root name on Kite instruments master (name field).
CRUDEOILM_ROOT = "CRUDEOILM"
DEFAULT_FUTURE_SYMBOL = "CRUDEOILM26JUNFUT"
DEFAULT_OPTION_PREFIX = "CRUDEOILM26JUN"


@dataclass(frozen=True)
class McxProduct:
    future_symbol: str
    option_prefix: str
    label: str
    units_per_lot: int
    strike_step: float
    min_premium: float = 0.05


CRUDEOILM_PRODUCT = McxProduct(
    future_symbol=DEFAULT_FUTURE_SYMBOL,
    option_prefix=DEFAULT_OPTION_PREFIX,
    label="Crude Oil Mini (10 bbl)",
    units_per_lot=10,
    strike_step=50,
)

# App supports only CRUDEOILM; map legacy env aliases to mini.
_LEGACY_ALIASES = {
    "CRUDEOIL26JUN",
    "CRUDEOIL26JUNFUT",
    "CRUDEOILM",
    "CRUDEOILM26JUN",
}


def resolve_product(future_symbol: Optional[str] = None) -> McxProduct:
    """Always CRUDEOILM — optional symbol selects nearest listed FUT month only."""
    sym = (future_symbol or os.getenv("COMMODITY_FUTURE_SYMBOL", DEFAULT_FUTURE_SYMBOL)).strip().upper()
    if sym and sym not in _LEGACY_ALIASES and not sym.startswith("CRUDEOILM"):
        sym = DEFAULT_FUTURE_SYMBOL
    prefix = (
        os.getenv("COMMODITY_OPTION_PREFIX", "").strip()
        or DEFAULT_OPTION_PREFIX
    )
    units = int(os.getenv("COMMODITY_UNITS_PER_LOT", "10") or 10)
    step = float(os.getenv("COMMODITY_STRIKE_STEP", "50") or 50)
    fut = sym if sym.endswith("FUT") and sym.startswith("CRUDEOILM") else DEFAULT_FUTURE_SYMBOL
    if fut != DEFAULT_FUTURE_SYMBOL:
        prefix = fut.replace("FUT", "")
    return McxProduct(
        future_symbol=fut,
        option_prefix=prefix,
        label=CRUDEOILM_PRODUCT.label,
        units_per_lot=units,
        strike_step=step,
    )


def strike_filter_band(spot: float) -> Tuple[float, float, float]:
    """(low, high, max_dist) strike band for crude mini options."""
    if spot <= 0:
        return 0, 10**9, 10**9
    return max(1000, spot * 0.5), spot * 1.5, max(800, spot * 0.12)


def nearest_listed_future(rows: list) -> Optional[str]:
    """Nearest CRUDEOILM FUT from Kite instruments."""
    futs = [
        r
        for r in rows
        if str(r.get("name") or "") == CRUDEOILM_ROOT
        and str(r.get("instrument_type") or "").upper() == "FUT"
    ]
    if not futs:
        return None
    futs.sort(key=lambda r: str(r.get("expiry") or ""))
    return str(futs[0]["tradingsymbol"])
