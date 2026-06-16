"""ReconcileAgent — sync persisted watch pending state with live Kite orders / GTTs."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set


def _gtt_ids_on_broker() -> Set[str]:
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance()
        gtts = kite.get_gtts() or []
        out: Set[str] = set()
        for g in gtts:
            if not isinstance(g, dict):
                continue
            st = str(g.get("status") or "").lower()
            if st in ("triggered", "disabled", "cancelled", "expired", "deleted"):
                continue
            tid = g.get("id") or g.get("trigger_id")
            if tid is not None:
                out.add(str(tid))
        return out
    except Exception:
        return set()


def gtt_exists_on_broker(trigger_id: str) -> bool:
    tid = (trigger_id or "").strip()
    if not tid:
        return False
    return tid in _gtt_ids_on_broker()


def reconcile_pending_watch(
    *,
    entry_order_id: Optional[str],
    gtt_trigger_id: Optional[str],
    pending_trade_plan: Optional[Dict[str, Any]],
    order_status: Callable[[str], Optional[str]],
) -> Dict[str, Any]:
    """
    Returns suggested actions for the watch loop:
    - clear_entry: broker has no / terminal entry
    - clear_gtt: GTT id stale on broker
    - attach_gtt: entry filled but deferred plan still waiting for GTT
    """
    actions: List[str] = []
    clear_entry = False
    clear_gtt = False
    attach_gtt = False

    entry_id = (entry_order_id or "").strip()
    gtt_id = (gtt_trigger_id or "").strip()

    if entry_id:
        st = (order_status(entry_id) or "").upper()
        if st in ("CANCELLED", "REJECTED"):
            actions.append("clear_entry")
            clear_entry = True
        elif st in ("COMPLETE", "EXECUTED"):
            if pending_trade_plan and not gtt_id:
                actions.append("attach_gtt")
                attach_gtt = True

    if gtt_id and not gtt_exists_on_broker(gtt_id):
        actions.append("clear_stale_gtt")
        clear_gtt = True

    return {
        "actions": actions,
        "clear_entry": clear_entry,
        "clear_gtt": clear_gtt,
        "attach_gtt": attach_gtt,
    }
