"""Binance USDT-M futures signed REST (orders, leverage, balance)."""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import httpx

from core.config import get_settings
from utils.logger import log_error, log_info, log_warning

BASE_URL = os.getenv("BINANCE_FUTURES_API", "https://fapi.binance.com").rstrip("/")
RECV_WINDOW = 5000


def get_binance_credentials() -> Tuple[str, str]:
    settings = get_settings()
    key = (settings.binance_api_key or os.getenv("BINANCE_API_KEY") or "").strip()
    secret = (settings.binance_api_secret or os.getenv("BINANCE_API_SECRET") or "").strip()
    if not key or not secret:
        raise ValueError(
            "Binance API not configured — set BINANCE_API_KEY and BINANCE_API_SECRET in .env"
        )
    return key, secret


def _sign(query_string: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()


def signed_request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> Any:
    key, secret = api_key or "", api_secret or ""
    if not key or not secret:
        key, secret = get_binance_credentials()

    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    p["recvWindow"] = RECV_WINDOW
    query = urlencode(p, doseq=True)
    sig = _sign(query, secret)
    url = f"{BASE_URL}{path}?{query}&signature={sig}"
    headers = {"X-MBX-APIKEY": key}

    with httpx.Client(timeout=20.0) as client:
        resp = client.request(method.upper(), url, headers=headers)
        if resp.status_code >= 400:
            try:
                body = resp.json()
                msg = body.get("msg") or body
            except Exception:
                msg = resp.text
            raise RuntimeError(f"Binance API error: {msg}")
        return resp.json()


def public_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(f"{BASE_URL}{path}", params=params or {})
        resp.raise_for_status()
        return resp.json()


def get_symbol_price(symbol: str) -> float:
    data = public_get("/fapi/v1/ticker/price", {"symbol": symbol.upper()})
    return float(data.get("price") or 0)


def get_exchange_info_symbol(symbol: str) -> Dict[str, Any]:
    info = public_get("/fapi/v1/exchangeInfo")
    sym = symbol.upper()
    for s in info.get("symbols") or []:
        if s.get("symbol") == sym:
            return s
    return {}


def round_quantity(symbol: str, qty: float) -> float:
    """Round qty to Binance LOT_SIZE step."""
    info = get_exchange_info_symbol(symbol)
    for f in info.get("filters") or []:
        if f.get("filterType") == "LOT_SIZE":
            step = float(f.get("stepSize") or 0.001)
            if step > 0:
                prec = max(0, len(str(step).rstrip("0").split(".")[-1]))
                n = int(qty / step)
                return round(n * step, prec)
    return round(max(qty, 0.001), 3)


def round_price(symbol: str, price: float) -> float:
    info = get_exchange_info_symbol(symbol)
    for f in info.get("filters") or []:
        if f.get("filterType") == "PRICE_FILTER":
            tick = float(f.get("tickSize") or 0.1)
            if tick > 0:
                n = int(price / tick)
                return round(n * tick, 8)
    return round(price, 2)


def set_leverage(symbol: str, leverage: int) -> Dict[str, Any]:
    lev = max(1, min(125, int(leverage)))
    out = signed_request(
        "POST",
        "/fapi/v1/leverage",
        {"symbol": symbol.upper(), "leverage": lev},
    )
    log_info(f"[Binance] Leverage {symbol} → {lev}x")
    return out if isinstance(out, dict) else {"ok": True}


def set_margin_type(symbol: str, margin_type: str = "ISOLATED") -> Dict[str, Any]:
    try:
        return signed_request(
            "POST",
            "/fapi/v1/marginType",
            {"symbol": symbol.upper(), "marginType": margin_type.upper()},
        )
    except RuntimeError as exc:
        if "No need to change" in str(exc) or "-4046" in str(exc):
            return {"ok": True, "note": "margin type unchanged"}
        raise


def get_usdt_balance() -> float:
    rows = signed_request("GET", "/fapi/v2/balance")
    if not isinstance(rows, list):
        return 0.0
    for row in rows:
        if row.get("asset") == "USDT":
            return float(row.get("availableBalance") or row.get("balance") or 0)
    return 0.0


def place_limit_order(
    *,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    reduce_only: bool = False,
    leverage: Optional[int] = None,
) -> Dict[str, Any]:
    sym = symbol.upper()
    if leverage is not None:
        set_leverage(sym, leverage)
    qty = round_quantity(sym, quantity)
    px = round_price(sym, price)
    params: Dict[str, Any] = {
        "symbol": sym,
        "side": side.upper(),
        "type": "LIMIT",
        "timeInForce": "GTC",
        "quantity": qty,
        "price": px,
    }
    if reduce_only:
        params["reduceOnly"] = "true"
    try:
        order = signed_request("POST", "/fapi/v1/order", params)
        oid = str(order.get("orderId") or "")
        log_info(f"[Binance] LIMIT {side} {sym} qty={qty} @ {px} id={oid}")
        return {"ok": True, "order_id": oid, "order": order, "price": px, "quantity": qty}
    except Exception as exc:
        log_error(f"[Binance] place_limit_order failed: {exc}")
        return {"ok": False, "error": str(exc)}


def _fmt_param(val: float) -> str:
    """Binance DECIMAL params as plain strings (no scientific notation)."""
    s = f"{val:.8f}".rstrip("0").rstrip(".")
    return s if s else "0"


def place_algo_order(
    *,
    symbol: str,
    side: str,
    order_type: str,
    quantity: Optional[float] = None,
    trigger_price: Optional[float] = None,
    price: Optional[float] = None,
    reduce_only: bool = False,
    close_position: bool = False,
    working_type: str = "CONTRACT_PRICE",
) -> Dict[str, Any]:
    """Place conditional algo order (STOP_MARKET, TAKE_PROFIT_MARKET, etc.)."""
    sym = symbol.upper()
    ot = order_type.upper()
    params: Dict[str, Any] = {
        "algoType": "CONDITIONAL",
        "symbol": sym,
        "side": side.upper(),
        "type": ot,
        "workingType": working_type,
    }
    if close_position:
        params["closePosition"] = "true"
    else:
        if quantity is not None:
            params["quantity"] = _fmt_param(round_quantity(sym, quantity))
        if reduce_only:
            params["reduceOnly"] = "true"
    if trigger_price is not None:
        params["triggerPrice"] = _fmt_param(round_price(sym, trigger_price))
    if price is not None:
        params["price"] = _fmt_param(round_price(sym, price))
    try:
        order = signed_request("POST", "/fapi/v1/algoOrder", params)
        oid = str(order.get("algoId") or order.get("clientAlgoId") or "")
        log_info(
            f"[Binance] ALGO {ot} {side} {sym} qty={params.get('quantity')} "
            f"trigger={params.get('triggerPrice')} algoId={oid}"
        )
        return {"ok": True, "order_id": oid, "algo_id": oid, "order": order}
    except Exception as exc:
        log_error(f"[Binance] place_algo_order {ot} failed: {exc}")
        return {"ok": False, "error": str(exc)}


def get_open_algo_orders(symbol: str) -> list:
    try:
        data = signed_request("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol.upper()})
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.get("orders") or data.get("data") or [])
    except Exception as exc:
        log_warning(f"[Binance] openAlgoOrders failed: {exc}")
    return []


def cancel_algo_order(symbol: str, algo_id: str) -> Dict[str, Any]:
    try:
        signed_request(
            "DELETE",
            "/fapi/v1/algoOrder",
            {"symbol": symbol.upper(), "algoId": int(algo_id)},
        )
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def place_stop_market(
    *,
    symbol: str,
    side: str,
    quantity: float,
    stop_price: float,
    reduce_only: bool = True,
) -> Dict[str, Any]:
    """Stop-loss via Binance Algo Order API (required since 2025-12 for STOP_MARKET)."""
    return place_algo_order(
        symbol=symbol,
        side=side,
        order_type="STOP_MARKET",
        quantity=quantity,
        trigger_price=stop_price,
        reduce_only=reduce_only,
    )


def place_take_profit_market(
    *,
    symbol: str,
    side: str,
    quantity: float,
    trigger_price: float,
    reduce_only: bool = True,
) -> Dict[str, Any]:
    """Take-profit market exit via Algo Order API."""
    return place_algo_order(
        symbol=symbol,
        side=side,
        order_type="TAKE_PROFIT_MARKET",
        quantity=quantity,
        trigger_price=trigger_price,
        reduce_only=reduce_only,
    )


def cancel_order(symbol: str, order_id: str) -> Dict[str, Any]:
    try:
        signed_request(
            "DELETE",
            "/fapi/v1/order",
            {"symbol": symbol.upper(), "orderId": int(order_id)},
        )
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def get_order_status(symbol: str, order_id: str) -> Optional[str]:
    try:
        o = signed_request(
            "GET",
            "/fapi/v1/order",
            {"symbol": symbol.upper(), "orderId": int(order_id)},
        )
        return str(o.get("status") or "").upper() or None
    except Exception:
        return None


def get_order_avg_fill_price(symbol: str, order_id: str) -> Optional[float]:
    try:
        o = signed_request(
            "GET",
            "/fapi/v1/order",
            {"symbol": symbol.upper(), "orderId": int(order_id)},
        )
        avg = float(o.get("avgPrice") or 0)
        if avg > 0:
            return avg
        px = float(o.get("price") or 0)
        return px if px > 0 else None
    except Exception:
        return None
