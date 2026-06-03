"""Multi-entry pending state for commodity LIMIT+deferred-GTT flow."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple


def migrate_pending_entries(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = data.get("pending_entries")
    if isinstance(raw, dict) and raw:
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in raw.items():
            if isinstance(v, dict) and v.get("order_id"):
                out[str(k)] = dict(v)
        return out
    oid = (data.get("pending_entry_order_id") or "").strip()
    if not oid:
        return {}
    return {
        oid: {
            "order_id": oid,
            "symbol": data.get("pending_symbol"),
            "placed_at": data.get("pending_entry_placed_at"),
            "trade_plan": data.get("pending_trade_plan"),
            "gtt_trigger_id": data.get("pending_gtt_trigger_id"),
        }
    }


def migrate_trade_plans_by_symbol(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = data.get("trade_plans_by_symbol")
    if isinstance(raw, dict):
        return {str(k).upper(): dict(v) for k, v in raw.items() if isinstance(v, dict)}
    plan = data.get("pending_trade_plan")
    sym = (data.get("pending_symbol") or "").strip().upper()
    if sym and isinstance(plan, dict):
        return {sym: dict(plan)}
    return {}


def register_pending_entry(
    pending_entries: Dict[str, Dict[str, Any]],
    *,
    order_id: str,
    symbol: str,
    placed_at: str,
    trade_plan: Optional[Dict[str, Any]],
    gtt_trigger_id: Optional[str] = None,
) -> None:
    oid = (order_id or "").strip()
    if not oid:
        return
    pending_entries[oid] = {
        "order_id": oid,
        "symbol": (symbol or "").strip().upper() or None,
        "placed_at": placed_at,
        "trade_plan": copy.deepcopy(trade_plan) if trade_plan else None,
        "gtt_trigger_id": (str(gtt_trigger_id).strip() or None) if gtt_trigger_id else None,
    }


def cache_trade_plan(
    plans_by_symbol: Dict[str, Dict[str, Any]],
    *,
    symbol: str,
    trade_plan: Dict[str, Any],
) -> None:
    sym = (symbol or "").strip().upper()
    if sym and trade_plan:
        plans_by_symbol[sym] = copy.deepcopy(trade_plan)


def pending_needing_gtt(pending_entries: Dict[str, Dict[str, Any]]) -> bool:
    for pe in pending_entries.values():
        if pe.get("trade_plan") and not pe.get("gtt_trigger_id"):
            return True
    return False


def get_pending_entry(
    pending_entries: Dict[str, Dict[str, Any]],
    order_id: str,
) -> Optional[Dict[str, Any]]:
    return pending_entries.get((order_id or "").strip())


def update_pending_gtt(
    pending_entries: Dict[str, Dict[str, Any]],
    order_id: str,
    gtt_trigger_id: Optional[str],
) -> None:
    pe = get_pending_entry(pending_entries, order_id)
    if pe is not None:
        pe["gtt_trigger_id"] = (str(gtt_trigger_id).strip() or None) if gtt_trigger_id else None


def remove_pending_entry(
    pending_entries: Dict[str, Dict[str, Any]],
    order_id: str,
) -> Optional[Dict[str, Any]]:
    return pending_entries.pop((order_id or "").strip(), None)


def sync_legacy_pending_fields(
    pending_entries: Dict[str, Dict[str, Any]],
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[Dict[str, Any]],
]:
    """
    Legacy single-slot fields track the oldest entry still awaiting GTT attach,
    else the newest entry still open.
    """
    if not pending_entries:
        return None, None, None, None, None

    def _sort_key(pe: Dict[str, Any]) -> str:
        return str(pe.get("placed_at") or "")

    needing = [
        pe
        for pe in pending_entries.values()
        if pe.get("trade_plan") and not pe.get("gtt_trigger_id")
    ]
    if needing:
        pe = sorted(needing, key=_sort_key)[0]
    else:
        pe = sorted(pending_entries.values(), key=_sort_key)[-1]

    return (
        pe.get("order_id"),
        pe.get("placed_at"),
        pe.get("gtt_trigger_id"),
        pe.get("symbol"),
        pe.get("trade_plan"),
    )


def list_mcx_long_positions() -> List[Dict[str, Any]]:
    """Open long MCX option/future positions (qty > 0)."""
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance()
        positions = kite.positions() or {}
        seen: set[str] = set()
        out: List[Dict[str, Any]] = []
        for bucket in ("net", "day"):
            for p in positions.get(bucket) or []:
                if str(p.get("exchange") or "").upper() != "MCX":
                    continue
                sym = str(p.get("tradingsymbol") or "").upper()
                qty = int(p.get("quantity") or 0)
                if qty <= 0 or not sym or sym in seen:
                    continue
                seen.add(sym)
                out.append(
                    {
                        "tradingsymbol": sym,
                        "quantity": qty,
                        "average_price": float(p.get("average_price") or 0),
                    }
                )
        return out
    except Exception:
        return []


def symbols_with_exit_trail() -> Dict[str, str]:
    """tradingsymbol -> gtt_trigger_id for open commodity trails."""
    try:
        from services.exit_trail_store import list_open_exit_trails

        out: Dict[str, str] = {}
        for t in list_open_exit_trails():
            if str(t.get("segment") or "").lower() != "commodity":
                continue
            sym = str(t.get("tradingsymbol") or "").upper()
            if sym:
                out[sym] = str(t.get("gtt_trigger_id") or "").strip()
        return out
    except Exception:
        return {}
