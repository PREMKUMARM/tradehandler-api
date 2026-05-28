"""Live readiness gates and checklist summary for autonomous watch status API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _trade_plan_preview(plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not plan or not plan.get("tradingsymbol"):
        return None
    ind = plan.get("indicators") or {}
    return {
        "tradingsymbol": plan.get("tradingsymbol"),
        "strategy_name": plan.get("strategy_name"),
        "entry_limit_price": plan.get("entry_limit_price") or plan.get("entry_premium"),
        "stop_loss_premium": plan.get("stop_loss_premium"),
        "target_premium": plan.get("target_premium"),
        "spot_stop_loss": plan.get("spot_stop_loss"),
        "spot_target": plan.get("spot_target"),
        "entry_ready": bool(plan.get("entry_ready")),
        "entry_block_reason": plan.get("entry_block_reason"),
        "spot": ind.get("nifty_spot") or ind.get("crude_spot") or ind.get("spot"),
        "or_low": ind.get("or_low"),
        "or_high": ind.get("or_high"),
        "pdh": ind.get("pdh"),
        "pdl": ind.get("pdl"),
        "ema9": ind.get("ema9"),
    }


def _step_live_summary(step_statuses: Optional[List[Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for st in step_statuses or []:
        if isinstance(st, dict):
            row = st
        else:
            row = {
                "index": getattr(st, "index", None),
                "server_ok": getattr(st, "server_ok", False),
                "completed": getattr(st, "completed", False),
                "message": getattr(st, "message", ""),
                "output": getattr(st, "output", ""),
            }
        idx = row.get("index")
        if idx is None:
            continue
        out.append(
            {
                "index": int(idx),
                "server_ok": bool(row.get("server_ok")),
                "completed": bool(row.get("completed")),
                "message": str(row.get("message") or "")[:160],
                "output": str(row.get("output") or "")[:200],
            }
        )
    return sorted(out, key=lambda x: x["index"])


def build_readiness_payload(
    *,
    armed: bool,
    autonomous_mode: bool,
    plan: Optional[Dict[str, Any]],
    checklist_ready: bool,
    entry_ready: Optional[bool],
    can_place: bool,
    can_execute: bool,
    autonomous_eligible: bool,
    kill_switch_active: bool,
    market_open: bool,
    paper_trading_mode: bool,
    kite_connected: bool,
    guard_message: Optional[str],
    min_entry_score: int,
    entry_confirmation_score: Optional[int],
    pending_entry_order_id: Optional[str] = None,
    step_statuses: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    plan = plan or {}
    score = int(entry_confirmation_score or plan.get("entry_confirmation_score") or 0)
    min_score = max(40, min(100, int(min_entry_score or 65)))
    score_ok = score >= min_score
    entry_ok = entry_ready is True

    guards_ok = not (guard_message or "").strip()
    can_auto_place = (
        armed
        and autonomous_mode
        and checklist_ready
        and entry_ok
        and score_ok
        and can_execute
        and guards_ok
        and not kill_switch_active
    )

    gates: List[Dict[str, Any]] = [
        {
            "id": "armed",
            "label": "Watch armed",
            "ok": armed,
            "detail": "Autonomous loop polling live Kite data"
            if armed
            else "Start autonomous to evaluate live market",
        },
        {
            "id": "kite",
            "label": "Kite connected",
            "ok": kite_connected,
            "detail": "Live quotes and checklist" if kite_connected else "Connect Kite token",
        },
        {
            "id": "session",
            "label": "Market session",
            "ok": market_open or paper_trading_mode,
            "detail": "Exchange open for live orders"
            if market_open
            else ("Paper mode" if paper_trading_mode else "Closed — preview only until session"),
        },
        {
            "id": "checklist",
            "label": "Live checklist",
            "ok": checklist_ready,
            "detail": "All live steps pass on server"
            if checklist_ready
            else "Waiting for live step validation",
        },
        {
            "id": "entry",
            "label": "Entry setup",
            "ok": entry_ok,
            "detail": plan.get("entry_block_reason") or "Indicators confirm direction",
        },
        {
            "id": "score",
            "label": f"Confirmation score ≥ {min_score}",
            "ok": score_ok,
            "detail": f"Score {score} / {min_score}",
        },
        {
            "id": "execute",
            "label": "Margin & validation",
            "ok": can_execute,
            "detail": "can_execute" if can_execute else "Risk/margin or session block",
        },
        {
            "id": "guards",
            "label": "Position & duplicate guards",
            "ok": guards_ok,
            "detail": guard_message or "No block",
        },
    ]

    steps = _step_live_summary(step_statuses)
    passed = sum(1 for s in steps if s.get("server_ok"))
    total = len(steps) if steps else 12

    return {
        "readiness_gates": gates,
        "can_autonomous_place": can_auto_place,
        "live_checklist_passed": passed,
        "live_checklist_total": total,
        "step_live": steps,
        "trade_plan_preview": _trade_plan_preview(plan),
        "market_open": market_open,
        "paper_trading_mode": paper_trading_mode,
        "kite_connected": kite_connected,
        "pending_entry_order_id": pending_entry_order_id,
        "live_data_stale": not armed,
    }
