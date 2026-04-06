"""Multi-leg and batch execution helpers (same gates as single REST place)."""
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Query, Request

from core.exceptions import ValidationError, AlgoFeastException
from core.user_context import get_user_id_from_request
from kiteconnect.exceptions import KiteException
from schemas.support_ops import BasketPlaceRequest
from services.execution_audit import log_execution_audit
from services.strategy_run_fills import record_strategy_fill_if_run
from services.paper_trading import enrich_paper_orders_with_quotes, is_paper_mode, paper_place_order
from services.risk_gate import check_order_allowed, record_order_placed
from utils.kite_utils import get_kite_instance
from utils.logger import log_error, log_info

router = APIRouter(prefix="/execution", tags=["Execution"])


@router.get("/audit-log")
def list_execution_audit_log(limit: int = Query(100, ge=1, le=500)):
    """P2: recent execution audit rows for operations / compliance review."""
    import json as _json

    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    cur = conn.execute(
        """
        SELECT id, created_at, actor, action, exchange, tradingsymbol, payload, result, paper
        FROM execution_audit_log
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = []
    for r in cur.fetchall():
        d = {k: r[k] for k in r.keys()}
        for key in ("payload", "result"):
            raw = d.get(key)
            if isinstance(raw, str) and raw:
                try:
                    d[key] = _json.loads(raw)
                except Exception:
                    pass
        rows.append(d)
    return {"data": rows}


@router.get("/paper-orders")
def list_paper_orders(
    limit: int = Query(200, ge=1, le=1000),
    enrich: bool = Query(
        True,
        description="If true, fetch live LTP from Kite and compute unrealized PnL per row (needs valid session).",
    ),
):
    """Recent synthetic orders with optional live quotes for paper-trading P&L view."""
    import json as _json

    from database.connection import get_database

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
        (limit,),
    )
    rows = []
    for r in cur.fetchall():
        d = {k: r[k] for k in r.keys()}
        raw = d.get("payload")
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str) and raw.strip():
            try:
                d["payload"] = _json.loads(raw)
            except Exception:
                pass
        elif isinstance(raw, dict):
            d["payload"] = raw
        rows.append(d)
    meta = enrich_paper_orders_with_quotes(rows, fetch_quotes=enrich)
    return {"data": rows, "meta": meta}


@router.delete("/paper-orders/{order_id:path}")
def delete_paper_order(order_id: str):
    """Delete one paper order row by `order_id` (e.g. PAPER-ABC123)."""
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    cur = conn.execute("DELETE FROM paper_orders WHERE order_id = ?", (order_id,))
    conn.commit()
    if cur.rowcount == 0:
        raise ValidationError(message=f"No paper order with id {order_id!r}", field="order_id")
    return {"data": {"deleted": True, "order_id": order_id}}


@router.delete("/paper-orders")
def delete_all_paper_orders(confirm: bool = Query(False)):
    """Clear all paper orders. Pass confirm=true."""
    if not confirm:
        raise ValidationError(
            message="Refusing to delete all paper orders without confirm=true",
            field="confirm",
        )
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    cur = conn.execute("DELETE FROM paper_orders")
    conn.commit()
    return {"data": {"deleted": cur.rowcount}}


@router.post("/basket")
async def place_basket(request: Request, body: BasketPlaceRequest) -> Dict[str, Any]:
    """Place multiple legs sequentially; each leg passes the risk gate."""
    user_id = "default"
    try:
        user_id = get_user_id_from_request(request)
    except Exception:
        pass

    skip_sess = os.getenv("SKIP_SESSION_CHECK_ON_REST", "").lower() in ("1", "true", "yes")
    results: List[Dict[str, Any]] = []
    kite = None
    if not is_paper_mode():
        kite = get_kite_instance(user_id=user_id)

    for i, leg in enumerate(body.legs):
        ex = leg.exchange.upper()
        invest = float(leg.price or 0) * int(leg.quantity) if leg.price else 0.0
        ok_r, msg_r = check_order_allowed(
            ex,
            leg.tradingsymbol,
            leg.quantity,
            leg.transaction_type,
            invest,
            skip_session_check=skip_sess,
        )
        if not ok_r:
            raise ValidationError(message=f"Leg {i}: {msg_r}", field="risk_gate")

        if is_paper_mode():
            oid = paper_place_order(
                {
                    "basket_leg": i,
                    "strategy_run_id": leg.strategy_run_id,
                    "tradingsymbol": leg.tradingsymbol,
                    "exchange": ex,
                    "transaction_type": leg.transaction_type,
                    "quantity": leg.quantity,
                    "order_type": leg.order_type,
                    "product": leg.product,
                    "price": leg.price,
                    "trigger_price": leg.trigger_price,
                    "stoploss": leg.stoploss,
                    "target": leg.target,
                    "trailing_stoploss": leg.trailing_stoploss,
                }
            )
            record_order_placed(invest)
            log_execution_audit(
                "BASKET_LEG",
                actor=user_id,
                exchange=ex,
                tradingsymbol=leg.tradingsymbol,
                payload={"leg": i, "paper": True},
                result={"order_id": oid},
                paper=True,
            )
            record_strategy_fill_if_run(
                leg.strategy_run_id,
                oid,
                leg.tradingsymbol,
                leg.transaction_type,
                leg.quantity,
                leg.price,
            )
            results.append({"leg": i, "order_id": oid, "status": "paper"})
            continue

        assert kite is not None
        params: Dict[str, Any] = {
            "variety": kite.VARIETY_REGULAR,
            "exchange": ex,
            "tradingsymbol": leg.tradingsymbol,
            "transaction_type": leg.transaction_type,
            "quantity": leg.quantity,
            "product": leg.product,
            "order_type": leg.order_type,
            "validity": kite.VALIDITY_DAY,
            "tag": "basket",
        }
        if leg.price is not None:
            params["price"] = leg.price
        if leg.trigger_price is not None:
            params["trigger_price"] = leg.trigger_price
        try:
            order_id = kite.place_order(**params)
        except KiteException as e:
            log_error(f"Basket leg {i} Kite error: {e}")
            raise AlgoFeastException(
                message=f"Leg {i} failed: {e}",
                status_code=502,
                error_code="KITE_ERROR",
            )
        record_order_placed(invest)
        log_execution_audit(
            "BASKET_LEG",
            actor=user_id,
            exchange=ex,
            tradingsymbol=leg.tradingsymbol,
            result={"order_id": str(order_id)},
            paper=False,
        )
        record_strategy_fill_if_run(
            leg.strategy_run_id,
            str(order_id),
            leg.tradingsymbol,
            leg.transaction_type,
            leg.quantity,
            leg.price,
        )
        results.append({"leg": i, "order_id": str(order_id), "status": "live"})

    log_info(f"Basket placed {len(results)} legs for user={user_id}")
    return {"data": {"results": results}}
