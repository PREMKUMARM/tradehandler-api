"""
V2 wizard constants — Zerodha product rules for NFO options.
"""
from __future__ import annotations

import os

# GTT on NFO/BFO is not supported for MIS (intraday). Entry and GTT legs must use NRML.
V2_NFO_PRODUCT = os.getenv("V2_NFO_PRODUCT", "NRML").upper()
if V2_NFO_PRODUCT not in ("NRML",):
    V2_NFO_PRODUCT = "NRML"


def resolve_v2_nfo_product(plan: dict | None = None) -> str:
    """Product for V2 entry + GTT exit. Always NRML when exit is GTT_OCO."""
    if not plan:
        return V2_NFO_PRODUCT
    if str(plan.get("exit_order_type") or "SL_STEPPED").upper() in ("GTT_OCO", "GTT", "SL_STEPPED"):
        return V2_NFO_PRODUCT
    p = str(plan.get("product") or V2_NFO_PRODUCT).upper()
    return V2_NFO_PRODUCT if p == "MIS" else (p if p in ("NRML", "MIS", "CNC") else V2_NFO_PRODUCT)
