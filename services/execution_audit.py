"""
P2: Immutable-style execution audit trail for compliance and debugging.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from zoneinfo import ZoneInfo


def log_execution_audit(
    action: str,
    *,
    actor: str = "system",
    exchange: Optional[str] = None,
    tradingsymbol: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    paper: bool = False,
) -> None:
    try:
        from database.connection import get_database

        db = get_database()
        conn = db.get_connection()
        conn.execute(
            """
            INSERT INTO execution_audit_log (created_at, actor, action, exchange, tradingsymbol, payload, result, paper)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                actor,
                action,
                exchange,
                tradingsymbol,
                json.dumps(payload or {}, default=str),
                json.dumps(result or {}, default=str),
                1 if paper else 0,
            ),
        )
        conn.commit()
    except Exception:
        pass
