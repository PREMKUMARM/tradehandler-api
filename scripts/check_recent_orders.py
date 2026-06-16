#!/usr/bin/env python3
"""Inspect today's Kite orders, GTTs, positions, watch events."""
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IST = ZoneInfo("Asia/Kolkata")


def main() -> None:
    from utils.kite_utils import get_kite_instance
    from database.connection import get_database

    today = datetime.now(IST).date()
    kite = get_kite_instance()
    orders = kite.orders() or []

    today_orders = []
    for o in orders:
        ts = o.get("order_timestamp") or o.get("exchange_timestamp") or ""
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=IST)
            od = dt.astimezone(IST).date()
        except Exception:
            od = today
        if od == today:
            today_orders.append(o)

    today_orders.sort(key=lambda x: str(x.get("order_timestamp") or ""))

    print("=== TODAY ORDERS (IST) ===")
    for o in today_orders:
        print(
            o.get("order_timestamp"),
            o.get("order_id"),
            o.get("tradingsymbol"),
            o.get("transaction_type"),
            o.get("status"),
            "qty=%s" % o.get("quantity"),
            "price=%s" % o.get("price"),
            "avg=%s" % o.get("average_price"),
            (o.get("status_message") or "")[:60],
        )

    print("\n=== GTTs ===")
    for g in kite.get_gtts() or []:
        cond = g.get("condition") or {}
        legs = g.get("orders") or []
        print(json.dumps({
            "id": g.get("id"),
            "status": g.get("status"),
            "symbol": cond.get("tradingsymbol"),
            "triggers": cond.get("trigger_values"),
            "last_price": cond.get("last_price"),
            "sl_limit": legs[0].get("price") if legs else None,
            "tp_limit": legs[1].get("price") if len(legs) > 1 else None,
            "created": g.get("created_at"),
            "updated": g.get("updated_at"),
        }, default=str))

    print("\n=== POSITIONS WITH DAY ACTIVITY ===")
    for p in (kite.positions() or {}).get("net") or []:
        bq = int(p.get("buy_quantity") or 0)
        sq = int(p.get("sell_quantity") or 0)
        if bq or sq:
            print(json.dumps({
                "symbol": p.get("tradingsymbol"),
                "net_qty": p.get("quantity"),
                "buy_qty": bq,
                "sell_qty": sq,
                "buy_price": p.get("buy_price"),
                "sell_price": p.get("sell_price"),
                "last_price": p.get("last_price"),
                "pnl": p.get("pnl"),
                "realised": p.get("realised"),
            }, default=str))

    path = "data/v2_strategy_watch.json"
    if os.path.exists(path):
        with open(path) as f:
            w = json.load(f)
        print("\n=== V2 WATCH (recent order events) ===")
        print("armed:", w.get("armed"), "placed_today:", w.get("placed_symbol_today"))
        print("pending_gtt:", w.get("pending_gtt_trigger_id"))
        for ev in (w.get("events") or [])[:10]:
            k = ev.get("kind") or ""
            if any(x in k for x in ("placed", "gtt", "cancel", "skip")):
                print(ev.get("at"), k, (ev.get("message") or "")[:140])

    db = get_database()
    conn = db.get_connection()
    print("\n=== EXIT TRAILS (last 5) ===")
    try:
        for r in conn.execute(
            "SELECT id,tradingsymbol,entry_order_id,gtt_trigger_id,entry_price,"
            "stop_loss,target,peak_ltp,trail_active,status,created_at,updated_at "
            "FROM exit_trails ORDER BY id DESC LIMIT 5"
        ):
            print(dict(r))
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
