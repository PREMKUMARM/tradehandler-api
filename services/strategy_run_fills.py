"""Attach broker orders to strategy_runs.strategy_fills when run_id is supplied (P1 traceability)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from zoneinfo import ZoneInfo

from utils.logger import log_warning


def record_strategy_fill_if_run(
    run_id: Optional[str],
    broker_order_id: str,
    tradingsymbol: str,
    side: str,
    quantity: int,
    price: Optional[float] = None,
) -> None:
    if not run_id or not str(run_id).strip():
        return
    rid = str(run_id).strip()
    try:
        from database.connection import get_database

        db = get_database()
        conn = db.get_connection()
        cur = conn.execute("SELECT id FROM strategy_runs WHERE id = ?", (rid,))
        if not cur.fetchone():
            log_warning(f"[StrategyFill] run_id={rid} not found; skip fill record")
            return
        now = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
        conn.execute(
            """
            INSERT INTO strategy_fills (run_id, broker_order_id, tradingsymbol, side, quantity, price, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (rid, str(broker_order_id), tradingsymbol, side.upper(), int(quantity), price, now),
        )
        conn.commit()
    except Exception as e:
        log_warning(f"[StrategyFill] insert failed: {e}")
