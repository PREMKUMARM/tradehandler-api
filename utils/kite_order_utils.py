"""
Kite place_order helpers — market protection (SEBI/API) and LIMIT fallback.
"""
from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Optional

from kiteconnect.exceptions import KiteException

from utils.logger import log_info, log_warning


def market_protection_value() -> int:
    """
    -1 = automatic (Zerodha recommended for API MARKET orders).
    0-100 = custom % band. Env: KITE_MARKET_PROTECTION (default -1).
    """
    raw = os.getenv("KITE_MARKET_PROTECTION", "-1").strip()
    try:
        return int(raw)
    except ValueError:
        return -1


def round_to_tick(price: float, tick: float = 0.05) -> float:
    if price <= 0:
        return tick
    return round(round(price / tick) * tick, 2)


def circuit_limits_from_quote_row(row: Dict[str, Any]) -> Dict[str, float]:
    """Extract circuit band from a Kite /quote row."""
    if not isinstance(row, dict):
        return {"lower_circuit_limit": 0.0, "upper_circuit_limit": 0.0}
    return {
        "lower_circuit_limit": float(row.get("lower_circuit_limit") or 0),
        "upper_circuit_limit": float(row.get("upper_circuit_limit") or 0),
    }


def merge_quote_with_circuit(quote: Dict[str, float], row: Dict[str, Any]) -> Dict[str, float]:
    out = dict(quote or {})
    out.update(circuit_limits_from_quote_row(row))
    return out


def validate_buy_limit_price(
    limit: float,
    *,
    quote: Optional[Dict[str, float]] = None,
    stale_vs_ltp_pct: float = 12.0,
) -> tuple[float, bool, str]:
    """
    Ensure a BUY LIMIT is inside exchange circuit band.

    Returns (adjusted_limit, ok, message).
    If limit is far below lower circuit (stale patient quote), ok=False so we skip
    rather than chase at the circuit floor.
    """
    q = quote or {}
    lower = float(q.get("lower_circuit_limit") or 0)
    upper = float(q.get("upper_circuit_limit") or 0)
    ltp = float(q.get("ltp") or q.get("last_price") or 0)
    px = float(limit or 0)
    if px <= 0:
        return px, False, "Entry limit price missing"

    if upper > 0 and px > upper:
        return px, False, f"Limit ₹{px:.2f} above upper circuit ₹{upper:.2f}"

    if lower > 0 and px < lower:
        ref = ltp if ltp > 0 else lower
        gap_pct = ((lower - px) / ref * 100.0) if ref > 0 else 100.0
        if gap_pct > stale_vs_ltp_pct or (ltp > 0 and px < ltp * (1.0 - stale_vs_ltp_pct / 100.0)):
            return (
                px,
                False,
                f"Limit ₹{px:.2f} below circuit floor ₹{lower:.2f} (LTP ₹{ltp:.2f}) — refresh live preview",
            )
        adjusted = round_to_tick(lower)
        return adjusted, True, f"Limit raised to circuit floor ₹{adjusted:.2f}"

    if ltp >= 2.0 and px < max(1.0, ltp * 0.5):
        return (
            px,
            False,
            f"Limit ₹{px:.2f} too far below LTP ₹{ltp:.2f} — wait for live quotes",
        )

    return round_to_tick(px), True, ""


def aggressive_limit_price(
    transaction_type: str,
    reference_price: float,
    *,
    buffer_pct: float = 1.0,
) -> float:
    """LIMIT price biased to fill: BUY slightly above ref, SELL slightly below."""
    ref = max(0.05, float(reference_price))
    buf = max(0.0, float(buffer_pct)) / 100.0
    tx = (transaction_type or "BUY").upper()
    if tx == "BUY":
        px = ref * (1 + buf)
    else:
        px = ref * (1 - buf)
    return round_to_tick(px)


def _attach_market_protection(order_params: Dict[str, Any]) -> Dict[str, Any]:
    ot = (order_params.get("order_type") or "").upper()
    if ot in ("MARKET", "SL-M"):
        order_params["market_protection"] = market_protection_value()
    return order_params


def place_kite_order(kite, order_params: Dict[str, Any]) -> Any:
    """
    Place order with market_protection on MARKET/SL-M.
    On SDK without market_protection param, retries as LIMIT if price available.
    """
    params = dict(order_params)
    params = _attach_market_protection(params)

    sig = inspect.signature(kite.place_order)
    if "market_protection" not in sig.parameters:
        params.pop("market_protection", None)
        ot = (params.get("order_type") or "").upper()
        if ot == "MARKET" and params.get("price"):
            params["order_type"] = kite.ORDER_TYPE_LIMIT
            log_warning("[Kite] SDK lacks market_protection — using LIMIT from price")
        elif ot == "MARKET":
            log_warning("[Kite] SDK lacks market_protection — caller should pass LIMIT + price")

    try:
        return kite.place_order(**params)
    except TypeError as exc:
        if "market_protection" not in str(exc):
            raise
        params.pop("market_protection", None)
        return kite.place_order(**params)
    except KiteException as exc:
        err = str(exc).lower()
        if "market protection" not in err:
            raise
        price = params.get("price")
        if price and (params.get("order_type") or "").upper() == "MARKET":
            params["order_type"] = kite.ORDER_TYPE_LIMIT
            params.pop("market_protection", None)
            log_info(f"[Kite] MARKET rejected — retry LIMIT @ {price}")
            return kite.place_order(**params)
        raise
