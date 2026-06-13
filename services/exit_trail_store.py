"""Persist open exit trails for momentum trailing (paper + live GTT)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from services.paper_trading import infer_segment_from_order, normalize_segment


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()


def register_exit_trail(
    *,
    segment: str,
    entry_order_id: str,
    tradingsymbol: str,
    exchange: str,
    product: str,
    quantity: int,
    entry_price: float,
    stop_loss: float,
    target: float,
    gtt_trigger_id: Optional[str] = None,
    paper: bool = False,
    paper_order_id: Optional[str] = None,
) -> None:
    from services.momentum_trail import get_momentum_trail_config

    if not get_momentum_trail_config().enabled:
        return
    if stop_loss <= 0 or target <= 0 or entry_price <= 0:
        return

    risk_unit = max(0.05, float(entry_price) - float(stop_loss))
    initial_target = float(entry_price) + risk_unit

    seg = normalize_segment(segment or infer_segment_from_order(exchange, tradingsymbol))
    oid = (entry_order_id or paper_order_id or "").strip()
    if not oid:
        return

    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    now = _now()
    conn.execute(
        """
        INSERT INTO exit_trails (
            created_at, updated_at, segment, entry_order_id, gtt_trigger_id,
            tradingsymbol, exchange, product, quantity, entry_price,
            stop_loss, target, initial_target, peak_ltp, trail_active, paper, paper_order_id, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 'open')
        ON CONFLICT(entry_order_id) DO UPDATE SET
            updated_at = excluded.updated_at,
            gtt_trigger_id = COALESCE(excluded.gtt_trigger_id, exit_trails.gtt_trigger_id),
            stop_loss = excluded.stop_loss,
            target = excluded.target,
            initial_target = COALESCE(excluded.initial_target, exit_trails.initial_target),
            status = 'open'
        """,
        (
            now,
            now,
            seg,
            oid,
            gtt_trigger_id,
            tradingsymbol,
            exchange.upper(),
            product,
            int(quantity),
            float(entry_price),
            float(stop_loss),
            float(target),
            float(initial_target),
            float(entry_price),
            1 if paper else 0,
            paper_order_id or oid,
        ),
    )
    conn.commit()


def list_open_exit_trails() -> List[Dict[str, Any]]:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    cur = conn.execute(
        """
        SELECT id, created_at, segment, entry_order_id, gtt_trigger_id, tradingsymbol, exchange,
               product, quantity, entry_price, stop_loss, target, initial_target, peak_ltp,
               trail_active, paper, paper_order_id, status, updated_at
        FROM exit_trails
        WHERE status = 'open'
        ORDER BY id ASC
        """
    )
    return [{k: r[k] for k in r.keys()} for r in cur.fetchall()]


def update_exit_trail_levels(
    trail_id: int,
    *,
    stop_loss: float,
    target: float,
    peak_ltp: float,
    trail_active: bool,
) -> None:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    conn.execute(
        """
        UPDATE exit_trails
        SET updated_at = ?, stop_loss = ?, target = ?, peak_ltp = ?, trail_active = ?
        WHERE id = ?
        """,
        (_now(), float(stop_loss), float(target), float(peak_ltp), 1 if trail_active else 0, trail_id),
    )
    conn.commit()


def update_exit_trail_gtt(trail_id: int, gtt_trigger_id: str) -> None:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    conn.execute(
        "UPDATE exit_trails SET updated_at = ?, gtt_trigger_id = ? WHERE id = ?",
        (_now(), str(gtt_trigger_id), trail_id),
    )
    conn.commit()


def close_exit_trail(trail_id: int, *, reason: str = "closed") -> None:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    conn.execute(
        "UPDATE exit_trails SET status = ?, updated_at = ? WHERE id = ?",
        (reason, _now(), trail_id),
    )
    conn.commit()


def sync_paper_order_levels(paper_order_id: str, stop_loss: float, target: float) -> None:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    conn.execute(
        "UPDATE paper_orders SET stoploss = ?, target = ? WHERE order_id = ?",
        (float(stop_loss), float(target), paper_order_id),
    )
    conn.commit()
