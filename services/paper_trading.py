"""
P2: Paper trading — when PAPER_TRADING_MODE is enabled, orders are logged locally instead of Kite.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict

from zoneinfo import ZoneInfo


def is_paper_mode() -> bool:
    v = os.getenv("PAPER_TRADING_MODE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def paper_place_order(payload: Dict[str, Any]) -> str:
    """Persist synthetic order; returns paper order id."""
    from database.connection import get_database

    oid = f"PAPER-{uuid.uuid4().hex[:12].upper()}"
    db = get_database()
    conn = db.get_connection()
    conn.execute(
        """
        INSERT INTO paper_orders (created_at, order_id, payload, status)
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            oid,
            json.dumps(payload, default=str),
            "COMPLETE",
        ),
    )
    conn.commit()
    return oid
