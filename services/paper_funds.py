"""
Per-segment paper trading capital: configured allocation, available balance, reset.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.paper_trading import infer_segment_from_order, normalize_segment
from utils.logger import log_info, log_warning

PAPER_FUNDS_PATH = Path(os.getenv("PAPER_TRADING_FUNDS_FILE", "data/paper_trading_funds.json"))

DEFAULT_FUNDS: Dict[str, Dict[str, Any]] = {
    "nifty50": {"allocated": 500_000.0, "currency": "INR"},
    "commodity": {"allocated": 200_000.0, "currency": "INR"},
    "crypto": {"allocated": 10_000.0, "currency": "USDT"},
}


def _read_funds_file() -> Dict[str, Dict[str, Any]]:
    out = {k: dict(v) for k, v in DEFAULT_FUNDS.items()}
    try:
        if PAPER_FUNDS_PATH.exists():
            raw = json.loads(PAPER_FUNDS_PATH.read_text(encoding="utf-8"))
            for seg in ("nifty50", "commodity", "crypto"):
                if seg in raw and isinstance(raw[seg], dict):
                    if "allocated" in raw[seg]:
                        out[seg]["allocated"] = float(raw[seg]["allocated"])
                    if raw[seg].get("currency"):
                        out[seg]["currency"] = str(raw[seg]["currency"]).upper()
    except Exception as e:
        log_warning(f"[PaperFunds] read failed: {e}")
    return out


def _write_funds_file(data: Dict[str, Dict[str, Any]]) -> None:
    PAPER_FUNDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_FUNDS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_allocated_amount(segment: str) -> float:
    seg = normalize_segment(segment)
    return float(_read_funds_file().get(seg, DEFAULT_FUNDS["nifty50"])["allocated"])


def get_segment_currency(segment: str) -> str:
    seg = normalize_segment(segment)
    return str(_read_funds_file().get(seg, DEFAULT_FUNDS["nifty50"]).get("currency") or "INR")


def set_segment_allocated(segment: str, allocated: float) -> Dict[str, Any]:
    seg = normalize_segment(segment)
    amt = float(allocated)
    if amt < 1000:
        raise ValueError("Paper fund amount must be at least 1000")
    data = _read_funds_file()
    cur = data.get(seg, dict(DEFAULT_FUNDS.get(seg, DEFAULT_FUNDS["nifty50"])))
    cur["allocated"] = amt
    data[seg] = cur
    _write_funds_file(data)
    log_info(f"[PaperFunds] {seg} allocated={amt:,.2f} {cur.get('currency')}")
    return get_fund_snapshot(seg)


def get_all_fund_snapshots() -> Dict[str, Dict[str, Any]]:
    return {seg: get_fund_snapshot(seg) for seg in ("nifty50", "commodity", "crypto")}


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


def _entry_price_from_row(row: Dict[str, Any], payload: Dict[str, Any]) -> Optional[float]:
    for k in ("entry_price",):
        v = row.get(k)
        if v is not None:
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    for k in ("paper_fill_price", "price"):
        v = payload.get(k)
        if v is not None:
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    return None


def _open_position_cost(row: Dict[str, Any], payload: Dict[str, Any]) -> float:
    """Capital locked by an open entry (prefer stored paper_entry_cost)."""
    v = payload.get("paper_entry_cost")
    if v is not None:
        try:
            c = float(v)
            if c > 0:
                return c
        except (TypeError, ValueError):
            pass
    entry = _entry_price_from_row(row, payload)
    if entry is None:
        return 0.0
    qty_raw = payload.get("quantity")
    try:
        qty = int(qty_raw) if qty_raw is not None else 0
    except (TypeError, ValueError):
        qty = 0
    if qty <= 0:
        return 0.0
    return max(0.0, entry * qty)


def _row_segment(row: Dict[str, Any], payload: Dict[str, Any]) -> str:
    if payload.get("segment"):
        return normalize_segment(str(payload["segment"]))
    return infer_segment_from_order(
        str(payload.get("exchange") or ""),
        str(payload.get("tradingsymbol") or ""),
    )


def _load_segment_order_rows(segment: str) -> List[Dict[str, Any]]:
    from database.connection import get_database

    seg = normalize_segment(segment)
    db = get_database()
    conn = db.get_connection()
    cur = conn.execute(
        """
        SELECT id, order_id, payload, exit_reason, exit_price, exit_at
        FROM paper_orders
        ORDER BY id ASC
        """
    )
    rows: List[Dict[str, Any]] = []
    for r in cur.fetchall():
        d = {k: r[k] for k in r.keys()}
        payload = _parse_payload(d.get("payload"))
        if payload.get("paper_exit_leg"):
            continue
        if _row_segment(d, payload) != seg:
            continue
        d["payload"] = payload
        rows.append(d)
    return rows


def _sum_open_and_realized(segment: str) -> Tuple[float, int, float, int]:
    """Returns (open_locked, open_count, realized_pnl, closed_count)."""
    open_locked = 0.0
    open_count = 0
    realized = 0.0
    closed_count = 0
    for row in _load_segment_order_rows(segment):
        payload = row.get("payload") or {}
        entry = _entry_price_from_row(row, payload)
        qty_raw = payload.get("quantity")
        try:
            qty = int(qty_raw) if qty_raw is not None else 0
        except (TypeError, ValueError):
            qty = 0
        tt = str(payload.get("transaction_type") or "BUY").upper()
        if row.get("exit_reason"):
            closed_count += 1
            if entry is None:
                continue
            exit_px = float(row.get("exit_price") or entry)
            if tt == "BUY":
                realized += (exit_px - entry) * qty
            else:
                realized += (entry - exit_px) * qty
        else:
            cost = _open_position_cost(row, payload)
            if cost <= 0:
                continue
            open_count += 1
            open_locked += cost
    return open_locked, open_count, realized, closed_count


def get_available_balance(segment: str) -> float:
    """Capital available for the next paper entry (allocation + realized − open cost)."""
    seg = normalize_segment(segment)
    allocated = get_allocated_amount(seg)
    open_locked, _, realized, _ = _sum_open_and_realized(seg)
    return max(0.0, allocated + realized - open_locked)


def get_fund_snapshot(segment: str) -> Dict[str, Any]:
    seg = normalize_segment(segment)
    cfg = _read_funds_file().get(seg, DEFAULT_FUNDS["nifty50"])
    allocated = float(cfg["allocated"])
    currency = str(cfg.get("currency") or "INR")
    open_locked, open_count, realized, closed_count = _sum_open_and_realized(seg)
    available = max(0.0, allocated + realized - open_locked)
    return {
        "segment": seg,
        "allocated": round(allocated, 2),
        "currency": currency,
        "available": round(available, 2),
        "open_positions_cost": round(open_locked, 2),
        "open_positions": open_count,
        "realized_pnl": round(realized, 2),
        "closed_trades": closed_count,
    }


def resolve_capital_for_segment(
    segment: str,
    *,
    margin_fallback: float = 0.0,
    cfg_capital: float = 100_000.0,
) -> float:
    """Trading capital for sizing: paper available balance, else live margin."""
    from services.paper_trading import is_paper_mode_for_segment

    if is_paper_mode_for_segment(segment):
        avail = get_available_balance(segment)
        if avail > 0:
            return avail
        return get_allocated_amount(segment)
    if margin_fallback > 0:
        return margin_fallback
    return cfg_capital


def entry_cost_from_payload(payload: Dict[str, Any], fill_price: Optional[float] = None) -> float:
    px = fill_price
    if px is None:
        for k in ("paper_fill_price", "price"):
            v = payload.get(k)
            if v is not None:
                try:
                    px = float(v)
                    break
                except (TypeError, ValueError):
                    pass
    if px is None or px <= 0:
        return 0.0
    try:
        qty = int(payload.get("quantity") or 0)
    except (TypeError, ValueError):
        qty = 0
    return max(0.0, px * qty)


def assert_can_allocate(segment: str, entry_cost: float) -> Tuple[bool, str]:
    if entry_cost <= 0:
        return False, "Invalid entry cost for paper order"
    avail = get_available_balance(segment)
    if entry_cost > avail + 0.01:
        cur = get_segment_currency(segment)
        sym = "₹" if cur == "INR" else "$"
        return (
            False,
            f"Insufficient paper funds: need {sym}{entry_cost:,.2f}, available {sym}{avail:,.2f}",
        )
    return True, "ok"


def reset_segment_funds(segment: str) -> Dict[str, Any]:
    """Delete all paper orders for segment and restore available balance to allocated."""
    from database.connection import get_database

    seg = normalize_segment(segment)
    rows = _load_segment_order_rows(seg)
    order_ids = [str(r["order_id"]) for r in rows if r.get("order_id")]
    db = get_database()
    conn = db.get_connection()
    deleted = 0
    for oid in order_ids:
        cur = conn.execute("DELETE FROM paper_orders WHERE order_id = ?", (oid,))
        deleted += cur.rowcount
    # exit legs may reference parent — remove any row whose payload segment matches
    cur = conn.execute("SELECT id, order_id, payload FROM paper_orders")
    for r in cur.fetchall():
        payload = _parse_payload(r["payload"])
        row_seg = infer_segment_from_order(
            str(payload.get("exchange") or ""),
            str(payload.get("tradingsymbol") or ""),
        )
        if payload.get("segment"):
            row_seg = normalize_segment(str(payload["segment"]))
        if row_seg == seg:
            conn.execute("DELETE FROM paper_orders WHERE order_id = ?", (r["order_id"],))
            deleted += 1
    conn.commit()
    log_info(f"[PaperFunds] reset {seg}: deleted {deleted} paper row(s)")
    snap = get_fund_snapshot(seg)
    snap["deleted_orders"] = deleted
    return snap
