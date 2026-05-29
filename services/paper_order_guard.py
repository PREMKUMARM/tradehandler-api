"""Guards for autonomous paper placement (cooldown, open positions, pending reconcile)."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _parse_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def _row_segment(payload: Dict[str, Any]) -> str:
    from services.paper_trading import infer_segment_from_order, normalize_segment

    seg = payload.get("segment")
    if seg:
        return normalize_segment(str(seg))
    return infer_segment_from_order(
        str(payload.get("exchange") or ""),
        str(payload.get("tradingsymbol") or ""),
    )


def _open_entry_rows(segment: Optional[str] = None) -> List[Dict[str, Any]]:
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    cur = conn.execute(
        """
        SELECT id, order_id, created_at, payload, exit_reason
        FROM paper_orders
        WHERE exit_reason IS NULL
        ORDER BY id DESC
        """
    )
    rows: List[Dict[str, Any]] = []
    for r in cur.fetchall():
        row = dict(r)
        payload = _parse_payload(row.get("payload"))
        if payload.get("paper_exit_leg"):
            continue
        rows.append(row)
    if not segment:
        return rows
    seg = segment.strip().lower()
    return [row for row in rows if _row_segment(_parse_payload(row.get("payload"))) == seg]


def is_paper_position_open(order_id: str) -> bool:
    oid = (order_id or "").strip().upper()
    if not oid:
        return False
    from database.connection import get_database

    db = get_database()
    conn = db.get_connection()
    cur = conn.execute(
        """
        SELECT exit_reason
        FROM paper_orders
        WHERE UPPER(order_id) = ?
        LIMIT 1
        """,
        (oid,),
    )
    row = cur.fetchone()
    if not row:
        return False
    return row["exit_reason"] is None


def count_open_paper_entries(segment: str) -> int:
    return len(_open_entry_rows(segment))


def has_open_paper_symbol(segment: str, tradingsymbol: str) -> Tuple[bool, str]:
    sym = (tradingsymbol or "").strip().upper()
    if not sym:
        return False, ""
    for row in _open_entry_rows(segment):
        payload = _parse_payload(row.get("payload"))
        if str(payload.get("tradingsymbol") or "").upper() == sym:
            return True, f"Open paper position on {sym} — wait for exit"
    return False, ""


def _latest_entry_time(segment: str) -> Optional[datetime]:
    from database.connection import get_database

    seg = segment.strip().lower()
    db = get_database()
    conn = db.get_connection()
    cur = conn.execute(
        """
        SELECT created_at, payload
        FROM paper_orders
        ORDER BY id DESC
        LIMIT 200
        """
    )
    latest: Optional[datetime] = None
    for r in cur.fetchall():
        payload = _parse_payload(r["payload"])
        if payload.get("paper_exit_leg"):
            continue
        if _row_segment(payload) != seg:
            continue
        raw = r["created_at"]
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=IST)
            else:
                dt = dt.astimezone(IST)
        except Exception:
            continue
        if latest is None or dt > latest:
            latest = dt
    return latest


def seconds_since_last_paper_entry(segment: str) -> Optional[float]:
    latest = _latest_entry_time(segment)
    if latest is None:
        return None
    return (datetime.now(IST) - latest).total_seconds()


def paper_place_cooldown_ok(segment: str) -> Tuple[bool, str]:
    cooldown = max(0, _env_int("PAPER_AUTO_COOLDOWN_SEC", 45))
    if cooldown <= 0:
        return True, ""
    age = seconds_since_last_paper_entry(segment)
    if age is None:
        return True, ""
    if age < cooldown:
        wait = int(cooldown - age) + 1
        return False, f"Paper cooldown: wait {wait}s before next {segment} entry"
    return True, ""


def widen_paper_exits(
    entry_premium: float,
    sl_prem: float,
    tgt_prem: float,
) -> Tuple[float, float]:
    """Widen SL/target for paper so monitor does not exit on tick noise."""
    from services.commodity_indicator_plan import round_to_tick

    entry = float(entry_premium)
    sl = float(sl_prem)
    tp = float(tgt_prem)
    pct = max(0.5, min(10.0, _env_float("PAPER_MIN_EXIT_GAP_PCT", 2.5))) / 100.0
    gap = max(0.25, entry * pct)
    if sl >= entry - gap * 0.5:
        sl = max(0.05, entry - gap)
    if tp <= entry + gap * 0.5:
        tp = entry + gap
    if sl >= tp:
        sl = max(0.05, entry - gap)
        tp = entry + gap
    return round_to_tick(sl), round_to_tick(tp)


def paper_autonomous_place_allowed(
    plan: Dict[str, Any],
    *,
    placed_today: bool,
    segment: str,
    entry_quality_check=None,
) -> Tuple[bool, str]:
    """
    Extra checks when segment is in paper mode (called from segment order guards).
    entry_quality_check: optional callable(plan) -> (bool, str) e.g. entry_quality_for_autonomous.
    """
    if placed_today:
        return False, "Max autonomous paper trades per day reached"
    if not plan or not plan.get("tradingsymbol"):
        return False, "No trade plan from checklist/strategy"

    seg = segment.strip().lower()
    sym = str(plan.get("tradingsymbol") or "")

    if _env_bool("PAPER_AUTO_STRICT_ENTRY", True) and entry_quality_check:
        ok, msg = entry_quality_check(plan)
        if not ok:
            return False, msg

    if _env_bool("PAPER_AUTO_ONE_OPEN_PER_SEGMENT", True):
        n = count_open_paper_entries(seg)
        if n > 0:
            return False, f"{n} open paper position(s) on {seg} — exit or wait before new entry"

    open_sym, sym_msg = has_open_paper_symbol(seg, sym)
    if open_sym:
        return False, sym_msg

    ok_cd, cd_msg = paper_place_cooldown_ok(seg)
    if not ok_cd:
        return False, cd_msg

    return True, "Paper autonomous OK"
