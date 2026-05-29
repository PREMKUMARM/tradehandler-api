"""Duplicate-order and entry-quality guards for Nifty V2 autonomous place."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import log_warning

_MIN_ENTRY_SCORE = int(os.getenv("NIFTY_AUTO_MIN_ENTRY_SCORE", "65") or 65)
_MAX_SPREAD_PCT = float(os.getenv("NIFTY_AUTO_MAX_SPREAD_PCT", "2.5") or 2.5)

_OPEN_ORDER_STATUSES = frozenset(
    {
        "OPEN",
        "TRIGGER PENDING",
        "AMO REQ RECEIVED",
        "VALIDATION PENDING",
        "PUT ORDER REQ RECEIVED",
        "OPEN PENDING",
        "OPEN QUEUED",
    }
)


def min_entry_confirmation_score() -> int:
    return max(40, min(100, _MIN_ENTRY_SCORE))


def entry_quality_for_autonomous(plan: Dict[str, Any]) -> Tuple[bool, str]:
    """Strict entry gate — autonomous must not fire on missing or weak confirmation."""
    if not plan:
        return False, "No trade plan"
    if plan.get("entry_ready") is not True:
        reason = plan.get("entry_block_reason") or "Entry not confirmed by indicators"
        return False, reason
    block = plan.get("entry_block_reason")
    if block:
        return False, str(block)
    score = int(plan.get("entry_confirmation_score") or 0)
    if score < min_entry_confirmation_score():
        return False, (
            f"Confirmation score {score} below minimum {min_entry_confirmation_score()} "
            f"for autonomous entry"
        )
    style = str(plan.get("entry_style") or "")
    if "blocked" in style.lower() or style == "blocked_wait":
        return False, "Entry style blocked — wait for setup"
    limit_px = float(plan.get("entry_limit_price") or 0)
    fair = float(plan.get("entry_fair_premium") or limit_px)
    if limit_px <= 0:
        return False, "Invalid entry limit price"
    if fair > 0 and limit_px > fair * 1.025:
        return False, f"Limit ₹{limit_px} chases above fair ₹{fair:.2f} — no autonomous chase"
    ind = plan.get("indicators") or {}
    bid = float(ind.get("option_bid") or 0)
    ask = float(ind.get("option_ask") or 0)
    ltp = float(ind.get("option_ltp") or plan.get("entry_premium") or 0)
    if bid > 0 and ask > bid and ltp > 0:
        spread = (ask - bid) / ltp * 100.0
        if spread > _MAX_SPREAD_PCT:
            return False, f"Option spread {spread:.1f}% too wide for autonomous entry"
    return True, "Entry confirmed"


def has_pending_nfo_order(tradingsymbol: str) -> Tuple[bool, str]:
    """True if Kite already has an open BUY order for this NFO symbol."""
    sym = (tradingsymbol or "").strip().upper()
    if not sym:
        return False, ""
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance()
        orders: List[Dict[str, Any]] = list(kite.orders() or [])
        for o in orders:
            if str(o.get("exchange") or "").upper() != "NFO":
                continue
            if str(o.get("tradingsymbol") or "").upper() != sym:
                continue
            status = str(o.get("status") or "").upper()
            if status not in _OPEN_ORDER_STATUSES:
                continue
            if str(o.get("transaction_type") or "").upper() == "BUY":
                oid = o.get("order_id")
                return True, f"Open BUY order {oid} already exists for {sym}"
        return False, ""
    except Exception as exc:
        log_warning(f"[V2Guard] orders check failed: {exc}")
        return True, f"Could not verify open orders — blocked: {exc}"


def has_nfo_position(tradingsymbol: str) -> Tuple[bool, str]:
    """True if net long position already open on this option."""
    sym = (tradingsymbol or "").strip().upper()
    if not sym:
        return False, ""
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance()
        positions = kite.positions() or {}
        for bucket in ("net", "day"):
            for p in positions.get(bucket) or []:
                if str(p.get("exchange") or "").upper() != "NFO":
                    continue
                if str(p.get("tradingsymbol") or "").upper() != sym:
                    continue
                qty = int(p.get("quantity") or 0)
                if qty > 0:
                    return True, f"Open position on {sym} (qty={qty}) — exit before re-entry"
        return False, ""
    except Exception as exc:
        log_warning(f"[V2Guard] positions check failed: {exc}")
        return True, f"Could not verify positions — blocked: {exc}"


def autonomous_place_allowed(
    plan: Dict[str, Any],
    *,
    placed_today: bool,
    segment: str = "nifty50",
) -> Tuple[bool, str]:
    if placed_today:
        return False, "Max autonomous Nifty trades per day reached"
    try:
        from services.paper_trading import is_paper_mode_for_segment

        if is_paper_mode_for_segment(segment):
            from services.paper_order_guard import paper_autonomous_place_allowed

            return paper_autonomous_place_allowed(
                plan,
                placed_today=placed_today,
                segment=segment,
            )
    except Exception:
        pass
    sym = str(plan.get("tradingsymbol") or "")
    ok, msg = entry_quality_for_autonomous(plan)
    if not ok:
        return False, msg
    pending, pend_msg = has_pending_nfo_order(sym)
    if pending:
        return False, pend_msg
    pos, pos_msg = has_nfo_position(sym)
    if pos:
        return False, pos_msg
    return True, "OK"
