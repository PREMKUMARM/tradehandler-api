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
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

from utils.logger import log_info, log_warning


def _float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _resolve_paper_fill_price(payload: Dict[str, Any]) -> Optional[float]:
    """Best-effort simulated fill: limit/trigger price, else live LTP from Kite."""
    pre = _float_or_none(payload.get("paper_fill_price"))
    if pre is not None:
        return pre
    ot = str(payload.get("order_type") or "").upper()
    for key in ("price",):
        v = payload.get(key)
        if v is not None:
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    if ot in ("SL", "SL-M", "SL-MKT"):
        v = payload.get("trigger_price")
        if v is not None:
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    ex = str(payload.get("exchange") or "NFO").upper()
    sym = payload.get("tradingsymbol")
    if not sym:
        return None
    key = f"{ex}:{sym}"
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance(skip_validation=True)
        q = kite.quote([key])
        if key in q:
            lp = q[key].get("last_price")
            if lp is not None:
                return float(lp)
    except Exception as e:
        log_warning(f"[PaperTrading] LTP for fill price {key}: {e}")
    return None


def _entry_price_from_payload(p: Dict[str, Any]) -> Optional[float]:
    for k in ("paper_fill_price", "price"):
        v = p.get(k)
        if v is not None:
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    return None


def enrich_paper_orders_with_quotes(
    rows: List[Dict[str, Any]], *, fetch_quotes: bool = True
) -> Dict[str, Any]:
    """
    Add ltp, entry_price, quantity, transaction_type, unrealized_pnl per row.
    When fetch_quotes is True, batches Kite quote() for unique instruments.
    """
    quote_keys: List[str] = []
    seen: set = set()
    for row in rows:
        p = row.get("payload")
        if not isinstance(p, dict):
            continue
        ex = str(p.get("exchange") or "NFO").upper()
        sym = p.get("tradingsymbol")
        if not sym:
            continue
        qk = f"{ex}:{sym}"
        row["_qk"] = qk
        if qk not in seen:
            seen.add(qk)
            quote_keys.append(qk)

    quotes: Dict[str, Any] = {}
    quote_error: Optional[str] = None
    if fetch_quotes and quote_keys:
        try:
            from utils.kite_utils import get_kite_instance

            kite = get_kite_instance(skip_validation=True)
            chunk = 400
            for i in range(0, len(quote_keys), chunk):
                part = quote_keys[i : i + chunk]
                quotes.update(kite.quote(part))
        except Exception as e:
            quote_error = str(e)
            log_warning(f"[PaperOrders] Batch quote failed: {e}")

    for row in rows:
        p = row.get("payload")
        if not isinstance(p, dict):
            row["ltp"] = None
            row["entry_price"] = None
            row["quantity"] = None
            row["transaction_type"] = None
            row["unrealized_pnl"] = None
            row["realized_pnl"] = None
            row.pop("_qk", None)
            continue

        qk = row.pop("_qk", None)
        ltp: Optional[float] = None
        if qk and qk in quotes:
            lp = quotes[qk].get("last_price")
            if lp is not None:
                ltp = float(lp)
        row["ltp"] = ltp

        qty_raw = p.get("quantity")
        qty_i: Optional[int] = None
        if qty_raw is not None:
            try:
                qty_i = int(qty_raw)
            except (TypeError, ValueError):
                pass
        row["quantity"] = qty_i

        tt = str(p.get("transaction_type") or "").upper() or None
        row["transaction_type"] = tt

        entry = _entry_price_from_payload(p)
        row["entry_price"] = entry

        if p.get("paper_exit_leg"):
            row["unrealized_pnl"] = None
            row["realized_pnl"] = None
        elif row.get("exit_reason") and row.get("exit_price") is not None and entry is not None and qty_i and tt in (
            "BUY",
            "SELL",
        ):
            ep = float(row["exit_price"])
            if tt == "BUY":
                row["realized_pnl"] = round((ep - entry) * qty_i, 2)
            else:
                row["realized_pnl"] = round((entry - ep) * qty_i, 2)
            row["unrealized_pnl"] = None
        elif ltp is not None and entry is not None and qty_i and tt in ("BUY", "SELL"):
            row["realized_pnl"] = None
            if tt == "BUY":
                row["unrealized_pnl"] = round((ltp - entry) * qty_i, 2)
            else:
                row["unrealized_pnl"] = round((entry - ltp) * qty_i, 2)
        else:
            row["unrealized_pnl"] = None
            row["realized_pnl"] = None

    return {"quotes_ok": quote_error is None, "quote_error": quote_error}

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

    to_store = dict(payload)
    fp = _resolve_paper_fill_price(to_store)
    if fp is not None:
        to_store["paper_fill_price"] = fp

    sl = _float_or_none(to_store.get("stoploss"))
    tgt = _float_or_none(to_store.get("target"))
    trail = _float_or_none(to_store.get("trailing_stoploss"))
    if to_store.get("paper_exit_leg"):
        sl, tgt, trail = None, None, None

    is_exit = bool(to_store.get("paper_exit_leg"))
    status_row = "EXIT" if is_exit else "COMPLETE"

    oid = f"PAPER-{uuid.uuid4().hex[:12].upper()}"
    db = get_database()
    conn = db.get_connection()
    conn.execute(
        """
        INSERT INTO paper_orders (
            created_at, order_id, payload, status,
            stoploss, target, trailing_stoploss
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            oid,
            json.dumps(to_store, default=str),
            status_row,
            sl,
            tgt,
            trail,
        ),
    )
    conn.commit()
    return oid
