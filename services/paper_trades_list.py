"""
Paper trades listing — entry rows only, grouped for FnO UI (per segment).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from services.paper_funds import get_fund_snapshot, split_profit_loss
from services.paper_trading import (
    enrich_paper_orders_with_quotes,
    infer_segment_from_order,
    normalize_segment,
)


def _parse_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def _row_segment(row: Dict[str, Any], payload: Dict[str, Any]) -> str:
    seg = payload.get("segment")
    if seg:
        return normalize_segment(str(seg))
    return infer_segment_from_order(
        str(payload.get("exchange") or ""),
        str(payload.get("tradingsymbol") or ""),
    )


def _pnl_for_trade(row: Dict[str, Any]) -> tuple[Optional[float], str]:
    if row.get("realized_pnl") is not None:
        return float(row["realized_pnl"]), "realized"
    if row.get("unrealized_pnl") is not None:
        return float(row["unrealized_pnl"]), "unrealized"
    return None, "none"


def list_paper_trades(
    segment: Optional[str] = None,
    limit: int = 200,
    enrich: bool = True,
) -> Dict[str, Any]:
    from database.connection import get_database

    seg_filter = normalize_segment(segment) if segment else None
    db = get_database()
    conn = db.get_connection()
    cur = conn.execute(
        """
        SELECT id, created_at, order_id, payload, status,
               stoploss, target, trailing_stoploss,
               exit_reason, exit_price, exit_at, exit_order_id
        FROM paper_orders
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit * 3 if seg_filter else limit,),
    )
    rows: List[Dict[str, Any]] = []
    for r in cur.fetchall():
        d = {k: r[k] for k in r.keys()}
        raw = d.get("payload")
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str) and raw.strip():
            try:
                d["payload"] = json.loads(raw)
            except Exception:
                pass
        rows.append(d)

    entries: List[Dict[str, Any]] = []
    for row in rows:
        payload = _parse_payload(row.get("payload"))
        if payload.get("paper_exit_leg"):
            continue
        row_seg = _row_segment(row, payload)
        if seg_filter and row_seg != seg_filter:
            continue
        entries.append(row)
        if len(entries) >= limit:
            break

    meta = enrich_paper_orders_with_quotes(entries, fetch_quotes=enrich)
    trades: List[Dict[str, Any]] = []
    for row in entries:
        payload = _parse_payload(row.get("payload"))
        pnl, pnl_type = _pnl_for_trade(row)
        entry_px = row.get("entry_price")
        if entry_px is None:
            for k in ("paper_fill_price", "price"):
                v = payload.get(k)
                if v is not None:
                    try:
                        entry_px = float(v)
                        break
                    except (TypeError, ValueError):
                        pass
        qty = row.get("quantity")
        if qty is None and payload.get("quantity") is not None:
            from services.paper_funds import _quantity_from_payload

            qty = _quantity_from_payload(payload)
            if qty <= 0:
                qty = None

        exit_reason = row.get("exit_reason")
        status = "closed" if exit_reason else "open"
        est_p, est_l = (None, None)
        act_p, act_l = (None, None)
        if status == "open":
            est_p, est_l = split_profit_loss(pnl if pnl_type == "unrealized" else None)
        else:
            act_p, act_l = split_profit_loss(pnl if pnl_type == "realized" else None)

        trades.append(
            {
                "id": row.get("id"),
                "order_id": row.get("order_id"),
                "segment": _row_segment(row, payload),
                "symbol": payload.get("tradingsymbol") or "",
                "exchange": payload.get("exchange") or "",
                "side": (row.get("transaction_type") or payload.get("transaction_type") or "").upper(),
                "quantity": qty,
                "entry_price": entry_px,
                "entry_cost": payload.get("paper_entry_cost"),
                "entry_time": row.get("created_at"),
                "stoploss": row.get("stoploss"),
                "target": row.get("target"),
                "trailing_stoploss": row.get("trailing_stoploss"),
                "exit_reason": exit_reason,
                "exit_price": row.get("exit_price"),
                "exit_time": row.get("exit_at"),
                "ltp": row.get("ltp"),
                "pnl": pnl,
                "pnl_type": pnl_type,
                "estimated_profit": est_p,
                "estimated_loss": est_l,
                "actual_profit": act_p,
                "actual_loss": act_l,
                "status": status,
                "product": payload.get("product"),
            }
        )

    funds = None
    if seg_filter:
        try:
            funds = get_fund_snapshot(seg_filter, fetch_quotes=enrich)
        except Exception:
            funds = None

    return {
        "data": trades,
        "meta": meta,
        "segment": seg_filter or "all",
        "funds": funds,
    }
