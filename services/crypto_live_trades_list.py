"""Live crypto journal rows from Binance USDT-M (positions + open orders)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from services.crypto_config import DEFAULT_LEVERAGE, SYMBOL
from services.paper_funds import split_profit_loss


def _ms_to_iso(ms: Any) -> Optional[str]:
    try:
        ts = int(ms) / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat()
    except (TypeError, ValueError):
        return None


def _exit_levels_from_open_orders(
    open_orders: List[Dict[str, Any]],
    algo_orders: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    sl_px: Optional[float] = None
    tp_px: Optional[float] = None
    sl_id: Optional[str] = None
    tp_id: Optional[str] = None

    for o in algo_orders or []:
        ot = str(o.get("orderType") or o.get("type") or "").upper()
        oid = str(o.get("algoId") or o.get("clientAlgoId") or "")
        try:
            trig = float(o.get("triggerPrice") or 0) or None
        except (TypeError, ValueError):
            trig = None
        if ot in ("STOP", "STOP_MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT"):
            sl_px = trig
            sl_id = oid or None
        elif ot in ("TAKE_PROFIT", "TAKE_PROFIT_MARKET"):
            tp_px = trig
            tp_id = oid or None

    for o in open_orders or []:
        ot = str(o.get("type") or "").upper()
        oid = str(o.get("orderId") or "")
        if ot in ("STOP", "STOP_MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"):
            try:
                sl_px = float(o.get("stopPrice") or o.get("price") or 0) or None
            except (TypeError, ValueError):
                sl_px = None
            sl_id = oid or None
        elif ot == "LIMIT" and str(o.get("reduceOnly", "")).lower() in ("true", "1"):
            try:
                tp_px = float(o.get("price") or 0) or None
            except (TypeError, ValueError):
                tp_px = None
            tp_id = oid or None
    return sl_px, tp_px, sl_id, tp_id


def _entry_from_orders(
    all_orders: List[Dict[str, Any]],
    *,
    side: str,
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Most recent filled entry order for long/short."""
    want = "BUY" if side == "BUY" else "SELL"
    for o in reversed(all_orders or []):
        if str(o.get("status") or "").upper() != "FILLED":
            continue
        if str(o.get("side") or "").upper() != want:
            continue
        if str(o.get("reduceOnly", "")).lower() in ("true", "1"):
            continue
        oid = str(o.get("orderId") or "")
        t = _ms_to_iso(o.get("updateTime") or o.get("time"))
        try:
            px = float(o.get("avgPrice") or o.get("price") or 0) or None
        except (TypeError, ValueError):
            px = None
        return oid or None, t, px
    return None, None, None


def list_binance_crypto_trades(limit: int = 50) -> Dict[str, Any]:
    from utils.binance_order_utils import get_symbol_price, signed_request

    sym = SYMBOL
    meta: Dict[str, Any] = {"quotes_ok": True, "quote_error": None, "source": "binance"}
    trades: List[Dict[str, Any]] = []

    try:
        positions = signed_request("GET", "/fapi/v2/positionRisk", {"symbol": sym})
        open_orders = signed_request("GET", "/fapi/v1/openOrders", {"symbol": sym})
        from utils.binance_order_utils import get_open_algo_orders

        algo_orders = get_open_algo_orders(sym)
        all_orders = signed_request("GET", "/fapi/v1/allOrders", {"symbol": sym, "limit": 50})
        ltp = float(get_symbol_price(sym) or 0)
    except Exception as exc:
        return {
            "data": [],
            "meta": {**meta, "quotes_ok": False, "quote_error": str(exc)},
            "segment": "crypto",
        }

    sl_px, tp_px, sl_id, tp_id = _exit_levels_from_open_orders(
        open_orders if isinstance(open_orders, list) else [],
        algo_orders if isinstance(algo_orders, list) else [],
    )
    orders_list = all_orders if isinstance(all_orders, list) else []

    for p in positions or []:
        if not isinstance(p, dict):
            continue
        try:
            amt = float(p.get("positionAmt") or 0)
        except (TypeError, ValueError):
            continue
        if abs(amt) < 1e-12:
            continue

        side = "BUY" if amt > 0 else "SELL"
        qty = abs(amt)
        try:
            entry_px = float(p.get("entryPrice") or 0)
        except (TypeError, ValueError):
            entry_px = 0.0
        try:
            upnl = float(p.get("unRealizedProfit") or 0)
        except (TypeError, ValueError):
            upnl = 0.0

        entry_oid, entry_time, fill_px = _entry_from_orders(orders_list, side=side)
        if fill_px and fill_px > 0:
            entry_px = fill_px

        est_p, est_l = split_profit_loss(upnl if upnl else None)
        margin = (qty * entry_px / max(1, DEFAULT_LEVERAGE)) if entry_px > 0 else None

        trades.append(
            {
                "id": entry_oid,
                "order_id": entry_oid,
                "segment": "crypto",
                "symbol": sym,
                "exchange": "BINANCE",
                "side": side,
                "quantity": qty,
                "entry_price": round(entry_px, 2) if entry_px else None,
                "entry_cost": round(margin, 2) if margin else None,
                "entry_time": entry_time,
                "stoploss": sl_px,
                "target": tp_px,
                "trailing_stoploss": None,
                "exit_reason": None,
                "exit_price": None,
                "exit_time": None,
                "ltp": round(ltp, 2) if ltp > 0 else None,
                "pnl": round(upnl, 2) if upnl else None,
                "pnl_type": "unrealized" if upnl else "none",
                "estimated_profit": est_p,
                "estimated_loss": est_l,
                "actual_profit": None,
                "actual_loss": None,
                "status": "open",
                "product": "USDT-M",
                "leverage": DEFAULT_LEVERAGE,
                "entry_order_id": entry_oid,
                "sl_order_id": sl_id,
                "tp_order_id": tp_id,
                "entry_status": "FILLED",
            }
        )

    trades.sort(key=lambda t: t.get("entry_time") or "", reverse=True)
    if limit > 0:
        trades = trades[:limit]

    return {"data": trades, "meta": meta, "segment": "crypto"}
