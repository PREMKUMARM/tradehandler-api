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
    strategy_id: Optional[str] = None,
) -> None:
    from services.momentum_trail import get_momentum_trail_config

    if not get_momentum_trail_config().enabled:
        return
    if stop_loss <= 0 or target <= 0 or entry_price <= 0:
        return

    risk_unit = max(0.05, float(entry_price) - float(stop_loss))
    initial_target = float(entry_price) + risk_unit
    qty = int(quantity)

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
            stop_loss, target, initial_target, peak_ltp, trail_active, paper,
            paper_order_id, status, strategy_id, target_touch_since, partial_exit_done,
            initial_quantity, gtt_sync_fail_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 'open', ?, NULL, 0, ?, 0)
        ON CONFLICT(entry_order_id) DO UPDATE SET
            updated_at = excluded.updated_at,
            gtt_trigger_id = COALESCE(excluded.gtt_trigger_id, exit_trails.gtt_trigger_id),
            stop_loss = excluded.stop_loss,
            target = excluded.target,
            initial_target = COALESCE(excluded.initial_target, exit_trails.initial_target),
            strategy_id = COALESCE(excluded.strategy_id, exit_trails.strategy_id),
            initial_quantity = COALESCE(excluded.initial_quantity, exit_trails.initial_quantity),
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
            qty,
            float(entry_price),
            float(stop_loss),
            float(target),
            float(initial_target),
            float(entry_price),
            1 if paper else 0,
            paper_order_id or oid,
            (strategy_id or "").strip() or None,
            qty,
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
               trail_active, paper, paper_order_id, status, updated_at, strategy_id,
               target_touch_since, partial_exit_done, initial_quantity, gtt_sync_fail_count,
               last_alert_at
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
    quantity: Optional[int] = None,
) -> None:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    if quantity is not None:
        conn.execute(
            """
            UPDATE exit_trails
            SET updated_at = ?, stop_loss = ?, target = ?, peak_ltp = ?, trail_active = ?,
                quantity = ?, gtt_sync_fail_count = 0
            WHERE id = ?
            """,
            (
                _now(),
                float(stop_loss),
                float(target),
                float(peak_ltp),
                1 if trail_active else 0,
                int(quantity),
                trail_id,
            ),
        )
    else:
        conn.execute(
            """
            UPDATE exit_trails
            SET updated_at = ?, stop_loss = ?, target = ?, peak_ltp = ?, trail_active = ?,
                gtt_sync_fail_count = 0
            WHERE id = ?
            """,
            (
                _now(),
                float(stop_loss),
                float(target),
                float(peak_ltp),
                1 if trail_active else 0,
                trail_id,
            ),
        )
    conn.commit()


def set_target_touch_since(trail_id: int, touch_since: str) -> None:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    conn.execute(
        "UPDATE exit_trails SET target_touch_since = ?, updated_at = ? WHERE id = ?",
        (touch_since, _now(), trail_id),
    )
    conn.commit()


def mark_partial_exit_done(trail_id: int, remaining_qty: int) -> None:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    conn.execute(
        """
        UPDATE exit_trails
        SET partial_exit_done = 1, quantity = ?, updated_at = ?, gtt_sync_fail_count = 0
        WHERE id = ?
        """,
        (int(remaining_qty), _now(), trail_id),
    )
    conn.commit()


def increment_gtt_sync_fail(trail_id: int) -> int:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    conn.execute(
        """
        UPDATE exit_trails
        SET gtt_sync_fail_count = COALESCE(gtt_sync_fail_count, 0) + 1, updated_at = ?
        WHERE id = ?
        """,
        (_now(), trail_id),
    )
    conn.commit()
    cur = conn.execute(
        "SELECT gtt_sync_fail_count FROM exit_trails WHERE id = ?",
        (trail_id,),
    )
    row = cur.fetchone()
    return int(row[0] or 0) if row else 0


def mark_trail_alert_sent(trail_id: int) -> None:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    conn.execute(
        "UPDATE exit_trails SET last_alert_at = ?, updated_at = ? WHERE id = ?",
        (_now(), _now(), trail_id),
    )
    conn.commit()


def get_trail_last_alert_at(trail_id: int) -> Optional[str]:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    cur = conn.execute("SELECT last_alert_at FROM exit_trails WHERE id = ?", (trail_id,))
    row = cur.fetchone()
    return str(row[0]) if row and row[0] else None


def update_exit_trail_gtt(trail_id: int, gtt_trigger_id: str) -> None:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    conn.execute(
        "UPDATE exit_trails SET updated_at = ?, gtt_trigger_id = ?, gtt_sync_fail_count = 0 WHERE id = ?",
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
