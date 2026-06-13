"""Public API helpers for open momentum exit trails."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from services.exit_trail_store import list_open_exit_trails
from services.paper_trading import normalize_segment


def _risk_unit(trail: Dict[str, Any]) -> float:
    entry = float(trail.get("entry_price") or 0)
    initial_target = float(trail.get("initial_target") or 0)
    sl = float(trail.get("stop_loss") or 0)
    if initial_target > entry:
        return max(0.05, initial_target - entry)
    if sl < entry:
        return max(0.05, entry - sl)
    return 0.05


def _step_r(trail: Dict[str, Any], R: float) -> int:
    entry = float(trail.get("entry_price") or 0)
    peak = float(trail.get("peak_ltp") or entry)
    if R <= 0 or peak <= entry:
        return 0
    return max(0, int((peak - entry) / R))


def serialize_exit_trail(trail: Dict[str, Any]) -> Dict[str, Any]:
    entry = float(trail.get("entry_price") or 0)
    sl = float(trail.get("stop_loss") or 0)
    tp = float(trail.get("target") or 0)
    initial_target = float(trail.get("initial_target") or tp)
    peak = float(trail.get("peak_ltp") or entry)
    qty = int(trail.get("quantity") or 1)
    R = _risk_unit(trail)
    step = _step_r(trail, R)
    trail_active = bool(trail.get("trail_active"))
    partial_done = bool(trail.get("partial_exit_done"))

    if trail_active:
        phase = f"Trailing — step {max(1, step)}R"
    elif peak >= initial_target and initial_target > entry:
        phase = "At 1R — confirming hold"
    else:
        phase = "Waiting for 1R target"

    if qty <= 1 and not partial_done:
        partial_note = "Single lot — breakeven trail only (no partial exit)"
    elif partial_done:
        partial_note = f"Partial exit done · {qty} lot(s) trailing"
    else:
        from services.momentum_trail import get_momentum_trail_config

        pct = int(get_momentum_trail_config().partial_exit_pct * 100)
        partial_note = f"Partial {pct}% at 1R pending"

    return {
        "id": int(trail.get("id") or 0),
        "segment": str(trail.get("segment") or ""),
        "tradingsymbol": str(trail.get("tradingsymbol") or ""),
        "exchange": str(trail.get("exchange") or ""),
        "entry_order_id": str(trail.get("entry_order_id") or ""),
        "gtt_trigger_id": trail.get("gtt_trigger_id"),
        "strategy_id": trail.get("strategy_id"),
        "entry_price": round(entry, 2),
        "stop_loss": round(sl, 2),
        "target": round(tp, 2),
        "initial_target": round(initial_target, 2),
        "peak_ltp": round(peak, 2),
        "risk_unit": round(R, 2),
        "step_r": step,
        "next_target": round(entry + (max(1, step) + 1) * R, 2) if R > 0 else round(tp, 2),
        "trail_active": trail_active,
        "partial_exit_done": partial_done,
        "quantity": qty,
        "initial_quantity": int(trail.get("initial_quantity") or qty),
        "paper": bool(trail.get("paper")),
        "gtt_sync_fail_count": int(trail.get("gtt_sync_fail_count") or 0),
        "target_touch_since": trail.get("target_touch_since"),
        "phase": phase,
        "partial_note": partial_note,
        "updated_at": trail.get("updated_at"),
        "created_at": trail.get("created_at"),
    }


def list_exit_trails_for_api(
    *,
    segment: Optional[str] = None,
    tradingsymbol: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    seg_filter = normalize_segment(segment) if segment else None
    sym_filter = (tradingsymbol or "").strip().upper()
    trails = list_open_exit_trails()
    out: List[Dict[str, Any]] = []
    for t in trails:
        if seg_filter and normalize_segment(str(t.get("segment") or "")) != seg_filter:
            continue
        if sym_filter and str(t.get("tradingsymbol") or "").upper() != sym_filter:
            continue
        out.append(serialize_exit_trail(t))
        if len(out) >= limit:
            break
    return out
