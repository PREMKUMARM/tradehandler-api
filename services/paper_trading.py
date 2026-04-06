"""
P2: Paper trading — when enabled, orders are logged locally instead of Kite.

Precedence:
- PAPER_TRADING_MODE=true in environment → always paper (deploy default).
- PAPER_TRADING_MODE=false in environment → always live (UI file ignored).
- Otherwise → persisted flag in data/paper_trading.json (Operations UI toggle).
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from zoneinfo import ZoneInfo

from utils.logger import log_info, log_warning

PAPER_TRADING_STATE_PATH = Path(os.getenv("PAPER_TRADING_STATE_FILE", "data/paper_trading.json"))


def _env_forces_paper() -> bool:
    v = os.getenv("PAPER_TRADING_MODE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_forces_live() -> bool:
    v = os.getenv("PAPER_TRADING_MODE", "").strip().lower()
    return v in ("0", "false", "no", "off")


def is_paper_mode() -> bool:
    if _env_forces_paper():
        return True
    if _env_forces_live():
        return False
    try:
        if PAPER_TRADING_STATE_PATH.exists():
            data = json.loads(PAPER_TRADING_STATE_PATH.read_text(encoding="utf-8"))
            return bool(data.get("active"))
    except Exception as e:
        log_warning(f"[PaperTrading] State file read failed: {e}")
    return False


def set_paper_trading_active(active: bool) -> None:
    """Persist UI-controlled paper mode (ignored when PAPER_TRADING_MODE env is set to true/false)."""
    if _env_forces_paper() or _env_forces_live():
        raise ValueError(
            "Paper mode is controlled by PAPER_TRADING_MODE in the environment. "
            "Unset or comment it out in .env to use the Operations UI toggle."
        )
    PAPER_TRADING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TRADING_STATE_PATH.write_text(
        json.dumps({"active": bool(active)}, indent=2),
        encoding="utf-8",
    )
    log_info(f"[PaperTrading] State file set to active={active}")


def paper_trading_env_locks_ui() -> bool:
    """True when .env explicitly sets paper on/off so the UI cannot override."""
    return _env_forces_paper() or _env_forces_live()


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
