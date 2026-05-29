"""
Live trades journal — Kite executed orders grouped as entry + exit rows (per segment).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from services.paper_funds import mcx_notional_multiplier, split_profit_loss
from services.paper_trading import infer_segment_from_order, normalize_segment


def _parse_ts(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(s[:26], fmt)
        except ValueError:
            continue
    return None


def _lot_multiplier(exchange: str, symbol: str) -> int:
    return mcx_notional_multiplier({"exchange": exchange, "tradingsymbol": symbol})


def _premium_pnl(
    entry: float,
    mark: float,
    qty: int,
    exchange: str,
    symbol: str,
    *,
    buy: bool = True,
) -> float:
    mult = _lot_multiplier(exchange, symbol)
    delta = (mark - entry) if buy else (entry - mark)
    return round(delta * qty * mult, 2)


def _load_audit_sl_tp() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Map entry order_id → (stoploss, target) from execution audit when available."""
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    try:
        from database.connection import get_database

        db = get_database()
        conn = db.get_connection()
        cur = conn.execute(
            """
            SELECT payload, result FROM execution_audit_log
            WHERE paper = 0 AND action IN ('PLACE_ORDER', 'REST_PLACE_ORDER', 'BASKET_LEG')
            ORDER BY id DESC
            LIMIT 500
            """
        )
        for row in cur.fetchall():
            payload_raw = row["payload"]
            result_raw = row["result"]
            try:
                payload = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})
            except Exception:
                payload = {}
            try:
                result = json.loads(result_raw) if isinstance(result_raw, str) else (result_raw or {})
            except Exception:
                result = {}
            oid = str(result.get("order_id") or "")
            if not oid or oid in out:
                continue
            sl = payload.get("stoploss")
            tp = payload.get("target")
            try:
                sl_f = float(sl) if sl is not None else None
            except (TypeError, ValueError):
                sl_f = None
            try:
                tp_f = float(tp) if tp is not None else None
            except (TypeError, ValueError):
                tp_f = None
            if sl_f or tp_f:
                out[oid] = (sl_f, tp_f)
    except Exception:
        pass
    return out


def _gtt_levels_by_symbol() -> Dict[str, Tuple[Optional[float], Optional[float], Optional[str]]]:
    """Active GTT per tradingsymbol → (sl, tp, gtt_id)."""
    out: Dict[str, Tuple[Optional[float], Optional[float], Optional[str]]] = {}
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance(skip_validation=True)
        for g in kite.get_gtts() or []:
            if str(g.get("status") or "").lower() not in ("active", "enabled", ""):
                continue
            cond = g.get("condition") or {}
            sym = str(cond.get("tradingsymbol") or g.get("tradingsymbol") or "")
            if not sym:
                continue
            legs = g.get("orders") or []
            prices = sorted(
                [
                    float(leg.get("price"))
                    for leg in legs
                    if leg.get("price") is not None
                ]
            )
            sl = prices[0] if len(prices) >= 1 else None
            tp = prices[-1] if len(prices) >= 2 else None
            out[sym] = (sl, tp, str(g.get("id") or ""))
    except Exception:
        pass
    return out


def _position_map() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance(skip_validation=True)
        for p in kite.positions().get("net", []) or []:
            sym = str(p.get("tradingsymbol") or "")
            if sym:
                out[sym] = p
    except Exception:
        pass
    return out


def _fetch_quotes(symbols: List[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance(skip_validation=True)
        keys = []
        key_by_sym: Dict[str, str] = {}
        for sym, ex in symbols:
            k = f"{ex}:{sym}"
            keys.append(k)
            key_by_sym[sym] = k
        q = kite.quote(keys) or {}
        out: Dict[str, float] = {}
        for sym, k in key_by_sym.items():
            row = q.get(k) or {}
            lp = row.get("last_price")
            if lp is not None:
                try:
                    out[sym] = float(lp)
                except (TypeError, ValueError):
                    pass
        return out
    except Exception:
        return {}


_OPEN_ENTRY_STATUSES = frozenset(
    {
        "OPEN",
        "TRIGGER PENDING",
        "AMO REQ RECEIVED",
        "PUT ORDER REQ RECEIVED",
        "OPEN PENDING",
        "OPEN QUEUED",
        "VALIDATION PENDING",
    }
)


def _entry_price_from_order(o: Dict[str, Any], *, filled: int, avg_f: Optional[float]) -> Optional[float]:
    if filled > 0 and avg_f is not None:
        return avg_f
    try:
        px = float(o.get("price") or 0)
    except (TypeError, ValueError):
        return None
    return px if px > 0 else None


def _exit_reason(entry_px: float, exit_px: float, sl: Optional[float], tp: Optional[float]) -> str:
    if sl is not None and exit_px <= sl + 0.01:
        return "Stop loss hit"
    if tp is not None and exit_px >= tp - 0.01:
        return "Target hit"
    if exit_px > entry_px:
        return "Closed (profit)"
    if exit_px < entry_px:
        return "Closed (loss)"
    return "Closed"


def _group_live_trades(
    orders: List[Dict[str, Any]],
    *,
    seg_filter: Optional[str],
    audit_sl_tp: Dict[str, Tuple[Optional[float], Optional[float]]],
    gtt_by_sym: Dict[str, Tuple[Optional[float], Optional[float], Optional[str]]],
    positions: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pair BUY entries with SELL exits per symbol+product (FIFO)."""
    relevant: List[Dict[str, Any]] = []
    for o in orders:
        ex = str(o.get("exchange") or "").upper()
        sym = str(o.get("tradingsymbol") or "")
        seg = infer_segment_from_order(ex, sym)
        if seg == "crypto":
            continue
        if seg_filter and seg != seg_filter:
            continue
        if ex not in ("NFO", "MCX", "BFO"):
            continue
        relevant.append(o)

    relevant.sort(
        key=lambda x: _parse_ts(x.get("exchange_timestamp") or x.get("order_timestamp"))
        or datetime.min
    )

    open_entries: Dict[str, List[Dict[str, Any]]] = {}
    trades: List[Dict[str, Any]] = []

    for o in relevant:
        sym = str(o.get("tradingsymbol") or "")
        product = str(o.get("product") or "")
        key = f"{sym}|{product}"
        side = str(o.get("transaction_type") or "").upper()
        status = str(o.get("status") or "").upper()
        filled = int(o.get("filled_quantity") or 0)
        qty = int(o.get("quantity") or 0)
        avg = o.get("average_price")
        try:
            avg_f = float(avg) if avg is not None else None
        except (TypeError, ValueError):
            avg_f = None

        if side == "BUY" and (
            (status == "COMPLETE" and filled > 0)
            or (status in _OPEN_ENTRY_STATUSES and filled >= 0)
        ):
            entry_px = _entry_price_from_order(o, filled=filled, avg_f=avg_f)
            if entry_px is None:
                continue
            remaining = filled if filled > 0 else qty
            if remaining <= 0:
                continue
            entry = {
                "order": o,
                "remaining": remaining,
                "entry_px": entry_px,
                "entry_time": o.get("exchange_timestamp") or o.get("order_timestamp"),
                "entry_id": str(o.get("order_id") or ""),
            }
            open_entries.setdefault(key, []).append(entry)
            continue

        if side != "SELL" or status != "COMPLETE" or filled <= 0 or avg_f is None:
            continue

        remaining_exit = filled
        queue = open_entries.get(key) or []
        while remaining_exit > 0 and queue:
            entry = queue[0]
            take = min(remaining_exit, entry["remaining"])
            entry_px = entry["entry_px"]
            if entry_px is None:
                entry["remaining"] -= take
                if entry["remaining"] <= 0:
                    queue.pop(0)
                continue

            ex = str(entry["order"].get("exchange") or "")
            sym_only = sym
            entry_id = entry["entry_id"]
            sl, tp = audit_sl_tp.get(entry_id, (None, None))
            gtt_sl, gtt_tp, gtt_id = gtt_by_sym.get(sym_only, (None, None, None))
            if sl is None:
                sl = gtt_sl
            if tp is None:
                tp = gtt_tp

            pnl = _premium_pnl(entry_px, avg_f, take, ex, sym_only, buy=True)
            reason = _exit_reason(entry_px, avg_f, sl, tp)
            act_p, act_l = split_profit_loss(pnl)

            trades.append(
                {
                    "order_id": entry_id,
                    "exit_order_id": str(o.get("order_id") or ""),
                    "gtt_id": gtt_id,
                    "segment": infer_segment_from_order(ex, sym_only),
                    "symbol": sym_only,
                    "exchange": ex,
                    "side": "BUY",
                    "quantity": take,
                    "entry_price": entry_px,
                    "entry_time": entry["entry_time"],
                    "stoploss": sl,
                    "target": tp,
                    "exit_reason": reason,
                    "exit_price": avg_f,
                    "exit_time": o.get("exchange_timestamp") or o.get("order_timestamp"),
                    "ltp": None,
                    "pnl": pnl,
                    "pnl_type": "realized",
                    "estimated_profit": None,
                    "estimated_loss": None,
                    "actual_profit": act_p,
                    "actual_loss": act_l,
                    "status": "closed",
                    "product": product,
                    "entry_status": str(entry["order"].get("status") or ""),
                }
            )

            entry["remaining"] -= take
            remaining_exit -= take
            if entry["remaining"] <= 0:
                queue.pop(0)

    # Open / pending entries still in queue
    for key, queue in open_entries.items():
        sym, product = key.split("|", 1)
        for entry in queue:
            if entry["remaining"] <= 0:
                continue
            o = entry["order"]
            ex = str(o.get("exchange") or "")
            sym_only = sym
            entry_id = entry["entry_id"]
            entry_px = entry["entry_px"]
            sl, tp = audit_sl_tp.get(entry_id, (None, None))
            gtt_sl, gtt_tp, gtt_id = gtt_by_sym.get(sym_only, (None, None, None))
            if sl is None:
                sl = gtt_sl
            if tp is None:
                tp = gtt_tp

            pos = positions.get(sym_only) or {}
            pos_qty = int(pos.get("quantity") or 0)
            ltp = pos.get("last_price")
            try:
                ltp_f = float(ltp) if ltp is not None else None
            except (TypeError, ValueError):
                ltp_f = None

            pnl = None
            pnl_type = "none"
            est_p, est_l = (None, None)
            if pos_qty != 0 and entry_px is not None and pos.get("pnl") is not None:
                try:
                    pnl = float(pos["pnl"])
                    pnl_type = "unrealized"
                    est_p, est_l = split_profit_loss(pnl)
                except (TypeError, ValueError):
                    pass
            elif entry_px is not None and ltp_f is not None:
                pnl = _premium_pnl(
                    entry_px, ltp_f, entry["remaining"], ex, sym_only, buy=True
                )
                pnl_type = "unrealized"
                est_p, est_l = split_profit_loss(pnl)

            st = str(o.get("status") or "").upper()
            order_filled = int(o.get("filled_quantity") or 0)
            if st == "COMPLETE" and pos_qty == 0 and entry["remaining"] > 0:
                continue
            if order_filled == 0 and st in _OPEN_ENTRY_STATUSES:
                row_status = "pending"
            elif st == "COMPLETE" and pos_qty != 0:
                row_status = "open"
            else:
                row_status = "pending"

            trades.append(
                {
                    "order_id": entry_id,
                    "exit_order_id": None,
                    "gtt_id": gtt_id,
                    "segment": infer_segment_from_order(ex, sym_only),
                    "symbol": sym_only,
                    "exchange": ex,
                    "side": "BUY",
                    "quantity": entry["remaining"],
                    "entry_price": entry_px,
                    "entry_time": entry["entry_time"],
                    "stoploss": sl,
                    "target": tp,
                    "exit_reason": None,
                    "exit_price": None,
                    "exit_time": None,
                    "ltp": ltp_f,
                    "pnl": pnl,
                    "pnl_type": pnl_type,
                    "estimated_profit": est_p,
                    "estimated_loss": est_l,
                    "actual_profit": None,
                    "actual_loss": None,
                    "status": row_status,
                    "product": product,
                    "entry_status": st,
                }
            )

    trades.sort(
        key=lambda t: _parse_ts(t.get("entry_time")) or datetime.min,
        reverse=True,
    )
    return trades


def list_live_trades(
    segment: Optional[str] = None,
    limit: int = 200,
    enrich: bool = True,
) -> Dict[str, Any]:
    """Today's Kite orders grouped as live trade rows."""
    seg_filter = normalize_segment(segment) if segment else None
    if seg_filter == "crypto":
        return {
            "data": [],
            "meta": {"quotes_ok": True, "quote_error": None, "source": "kite"},
            "segment": seg_filter,
            "message": "Live crypto trades are on Binance — not shown here.",
        }

    meta: Dict[str, Any] = {"quotes_ok": True, "quote_error": None, "source": "kite"}
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance()
        orders = list(kite.orders() or [])
    except Exception as e:
        meta["quotes_ok"] = False
        meta["quote_error"] = str(e)
        return {"data": [], "meta": meta, "segment": seg_filter or "all"}

    audit_sl_tp = _load_audit_sl_tp()
    gtt_by_sym = _gtt_levels_by_symbol()
    positions = _position_map() if enrich else {}

    trades = _group_live_trades(
        orders,
        seg_filter=seg_filter,
        audit_sl_tp=audit_sl_tp,
        gtt_by_sym=gtt_by_sym,
        positions=positions,
    )

    if enrich:
        need_quotes = [
            (t["symbol"], t["exchange"])
            for t in trades
            if t.get("status") in ("open", "pending") and t.get("ltp") is None
        ]
        quotes = _fetch_quotes(need_quotes)
        for t in trades:
            if t.get("status") not in ("open", "pending"):
                continue
            sym = t.get("symbol") or ""
            if t.get("ltp") is None and sym in quotes:
                t["ltp"] = quotes[sym]
            if (
                t.get("pnl") is None
                and t.get("entry_price") is not None
                and t.get("ltp") is not None
            ):
                pnl = _premium_pnl(
                    float(t["entry_price"]),
                    float(t["ltp"]),
                    int(t.get("quantity") or 1),
                    str(t.get("exchange") or ""),
                    sym,
                    buy=True,
                )
                t["pnl"] = pnl
                t["pnl_type"] = "unrealized"
                est_p, est_l = split_profit_loss(pnl)
                t["estimated_profit"] = est_p
                t["estimated_loss"] = est_l

    if limit > 0:
        trades = trades[:limit]

    return {
        "data": trades,
        "meta": meta,
        "segment": seg_filter or "all",
    }
