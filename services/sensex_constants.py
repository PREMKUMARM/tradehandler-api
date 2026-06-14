"""
Sensex wizard constants — Zerodha product rules for BFO options.
"""
from __future__ import annotations

import os

# GTT on NFO/BFO is not supported for MIS (intraday). Entry and GTT legs must use NRML.
SENSEX_BFO_PRODUCT = os.getenv("SENSEX_BFO_PRODUCT", "NRML").upper()
if SENSEX_BFO_PRODUCT not in ("NRML",):
    SENSEX_BFO_PRODUCT = "NRML"


def resolve_sensex_bfo_product(plan: dict | None = None) -> str:
    """Product for V2 entry + GTT exit. Always NRML when exit is GTT_OCO."""
    if not plan:
        return SENSEX_BFO_PRODUCT
    if str(plan.get("exit_order_type") or "GTT_OCO").upper() in ("GTT_OCO", "GTT"):
        return SENSEX_BFO_PRODUCT
    p = str(plan.get("product") or SENSEX_BFO_PRODUCT).upper()
    return SENSEX_BFO_PRODUCT if p == "MIS" else (p if p in ("NRML", "MIS", "CNC") else SENSEX_BFO_PRODUCT)
