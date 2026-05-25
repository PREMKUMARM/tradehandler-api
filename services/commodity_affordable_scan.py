"""CRUDEOILM margin check vs available commodity funds."""
from __future__ import annotations

from typing import Any, Dict, Optional

from services.commodity_indicator_plan import live_option_quote
from services.commodity_instruments import resolve_commodity_contract
from services.commodity_product_context import use_commodity_product
from services.commodity_products import (
    CRUDEOILM_PRODUCT,
    DEFAULT_FUTURE_SYMBOL,
    nearest_listed_future,
    resolve_product,
)
from services.commodity_trade_service import _check_kite_and_margin, _validate_trade_plan
from services.commodity_instruments import resolve_future
from utils.kite_utils import get_kite_instance


def _active_product():
    rows = get_kite_instance().instruments("MCX")
    listed = nearest_listed_future(rows)
    prod = resolve_product(listed or DEFAULT_FUTURE_SYMBOL)
    if listed and listed != prod.future_symbol:
        from services.commodity_products import McxProduct

        return McxProduct(
            future_symbol=listed,
            option_prefix=listed.replace("FUT", ""),
            label=CRUDEOILM_PRODUCT.label,
            units_per_lot=CRUDEOILM_PRODUCT.units_per_lot,
            strike_step=CRUDEOILM_PRODUCT.strike_step,
        )
    return prod


def _scan_crude_mini(margin: float, direction: str = "AUTO") -> Optional[Dict[str, Any]]:
    prod = _active_product()
    with use_commodity_product(prod.future_symbol):
        kite = get_kite_instance()
        row = resolve_future()
        key = f"MCX:{row['tradingsymbol']}"
        spot = float((kite.quote([key]) or {}).get(key, {}).get("last_price") or 0)
        if spot <= 0:
            return None
        d = (direction or "AUTO").upper()
        kind = "PE" if d == "PE" else "CE" if d == "CE" else "PE"
        contract = resolve_commodity_contract(spot=spot, kind=kind, moneyness="ATM")
        if not contract:
            contract = resolve_commodity_contract(spot=spot, kind=kind, moneyness="OTM1")
        if not contract:
            return None
        quote = live_option_quote(contract.tradingsymbol)
        ltp = float(quote.get("ltp") or 0)
        if ltp <= 0:
            return None
        units = prod.units_per_lot
        premium_cost = ltp * units
        plan_stub = {
            "entry_premium": ltp,
            "stop_loss_premium": max(0.05, ltp * 0.92),
            "target_premium": ltp * 1.15,
            "quantity": 1,
            "num_lots": 1,
            "lot_size": units,
            "tradingsymbol": contract.tradingsymbol,
            "option_type": kind,
        }
        validation = _validate_trade_plan(
            plan_stub, margin, 1.0, 2.0, available_margin=margin
        )
        return {
            "future_symbol": prod.future_symbol,
            "label": prod.label,
            "tradingsymbol": contract.tradingsymbol,
            "option_type": kind,
            "strike": contract.strike,
            "spot": round(spot, 2),
            "ltp": round(ltp, 2),
            "units_per_lot": units,
            "premium_cost_inr": round(premium_cost, 2),
            "affordable": premium_cost <= margin,
            "validation": validation,
            "can_place": bool(validation.get("is_good_trade")),
        }


def scan_affordable_commodities(
    direction: str = "AUTO",
    margin_cap: Optional[float] = None,
) -> Dict[str, Any]:
    ok, margin, msg = _check_kite_and_margin()
    if not ok:
        return {"connected": False, "message": msg, "candidates": [], "best": None}
    cap = float(margin_cap or margin)
    try:
        hit = _scan_crude_mini(cap, direction)
    except Exception as exc:
        return {
            "connected": True,
            "available_margin": round(margin, 2),
            "margin_cap": round(cap, 2),
            "candidates": [],
            "affordable": [],
            "best": None,
            "recommendation": None,
            "error": str(exc),
        }
    candidates = [hit] if hit else []
    affordable = [c for c in candidates if c.get("affordable")]
    best = affordable[0] if affordable else (candidates[0] if candidates else None)
    return {
        "connected": True,
        "available_margin": round(margin, 2),
        "margin_cap": round(cap, 2),
        "product": "CRUDEOILM",
        "candidates": candidates,
        "affordable": affordable,
        "best": best,
        "recommendation": best["future_symbol"] if best else DEFAULT_FUTURE_SYMBOL,
    }
