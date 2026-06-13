"""Live readiness gates and checklist summary for autonomous watch status API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _gate(
    *,
    id: str,
    label: str,
    ok: bool,
    detail: str,
    value: Optional[Any] = None,
    threshold: Optional[Any] = None,
    action_hint: Optional[str] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": id,
        "label": label,
        "ok": ok,
        "detail": detail,
    }
    if value is not None:
        row["value"] = value
    if threshold is not None:
        row["threshold"] = threshold
    if action_hint and not ok:
        row["action_hint"] = action_hint
    return row


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
        "spot": ind.get("btc_spot") or ind.get("nifty_spot") or ind.get("crude_spot") or ind.get("spot"),
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


def _risk_limits_status() -> Dict[str, Any]:
    try:
        from utils.trade_limits import trade_limits

        return trade_limits.get_limits_status()
    except Exception:
        return {}


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
    segment: Optional[str] = None,
    validation: Optional[Dict[str, Any]] = None,
    missing_steps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    plan = plan or {}
    score = int(entry_confirmation_score or plan.get("entry_confirmation_score") or 0)
    min_score = max(40, min(100, int(min_entry_score or 65)))
    score_ok = score >= min_score
    entry_ok = entry_ready is True

    guards_ok = not (guard_message or "").strip()
    risk = _risk_limits_status()
    can_trade_limits = bool(risk.get("can_trade", True))
    pnl_inr = float(risk.get("pnl_inr_today") or 0)
    max_loss_inr = float(risk.get("max_loss_inr_per_day") or 0)
    premium_spent = float(risk.get("premium_spent_inr_today") or 0)
    max_premium_inr = float(risk.get("max_premium_inr_per_day") or 0)
    loss_cap_hit = max_loss_inr > 0 and pnl_inr <= -abs(max_loss_inr)
    premium_cap_hit = max_premium_inr > 0 and premium_spent >= max_premium_inr

    # Paper: place when live checklist + strategy plan are ready (no live broker entry gate).
    if paper_trading_mode:
        entry_gate_ok = checklist_ready and can_execute
        score_gate_ok = True
    else:
        entry_gate_ok = entry_ok
        score_gate_ok = score_ok

    can_auto_place = (
        armed
        and autonomous_mode
        and checklist_ready
        and entry_gate_ok
        and score_gate_ok
        and can_execute
        and guards_ok
        and not kill_switch_active
        and can_trade_limits
        and not loss_cap_hit
        and not premium_cap_hit
    )

    seg = (segment or "nifty").strip().lower()
    is_crypto = seg in ("crypto", "binance", "btc")
    from services.watch_skip_utils import execute_gate_detail

    exec_preview = {
        "checklist_ready": checklist_ready,
        "can_place": can_place,
        "market_open": market_open,
        "paper_trading_mode": paper_trading_mode,
        "validation": validation,
        "missing_steps": missing_steps or [],
    }
    execute_detail = execute_gate_detail(
        exec_preview,
        can_execute=can_execute,
        plan=plan,
        segment=segment,
    )
    broker_label = "Binance connected" if is_crypto else "Kite connected"
    broker_detail_ok = (
        "Kite token valid · margin available"
        if kite_connected and not is_crypto
        else ("Live quotes and checklist from Binance" if is_crypto else "Live quotes and checklist")
    )
    broker_detail_fail = "Connect Binance API keys in .env" if is_crypto else "Connect Kite token"
    broker_hint = "Settings → Binance API keys on server" if is_crypto else "Settings → connect Zerodha / refresh token"
    armed_detail = (
        "Autonomous loop polling live Binance data"
        if armed and is_crypto
        else ("Autonomous loop polling live Kite data" if armed else "Start autonomous to evaluate live market")
    )

    gates: List[Dict[str, Any]] = [
        _gate(
            id="armed",
            label="Watch armed",
            ok=armed,
            detail=armed_detail if armed else "Start autonomous to evaluate live market",
            action_hint="Press Start autonomous",
        ),
        _gate(
            id="kite" if not is_crypto else "binance",
            label=broker_label,
            ok=kite_connected,
            detail=broker_detail_ok if kite_connected else broker_detail_fail,
            action_hint=broker_hint,
        ),
        _gate(
            id="session",
            label="Market session",
            ok=market_open or paper_trading_mode,
            detail="Exchange open for live orders"
            if market_open
            else ("Paper mode — simulated orders" if paper_trading_mode else "Closed — preview only until session"),
            action_hint="Wait for market open or enable paper mode in Settings",
        ),
        _gate(
            id="checklist",
            label="Live checklist",
            ok=checklist_ready,
            detail="All live steps pass on server"
            if checklist_ready
            else "Waiting for live step validation",
            action_hint="Complete wizard steps or wait for auto-execute",
        ),
        _gate(
            id="entry",
            label="Entry setup",
            ok=entry_ok,
            detail=plan.get("entry_block_reason") or "Indicators confirm direction",
            action_hint="Wait for OR/EMA/PDH alignment or adjust direction",
        ),
        _gate(
            id="score",
            label=f"Confirmation score ≥ {min_score}",
            ok=score_ok,
            detail=f"Score {score} / {min_score}",
            value=score,
            threshold=min_score,
            action_hint=f"Need {max(0, min_score - score)} more points — wait for stronger setup",
        ),
        _gate(
            id="execute",
            label="Margin & validation",
            ok=can_execute,
            detail=execute_detail,
            action_hint="Check buying power and validation errors in preview",
        ),
        _gate(
            id="guards",
            label="Position & duplicate guards",
            ok=guards_ok,
            detail=guard_message or "No block",
            action_hint="Close duplicate position or wait for pending order to clear",
        ),
    ]

    if kill_switch_active:
        gates.append(
            _gate(
                id="kill_switch",
                label="Kill switch",
                ok=False,
                detail="Execution kill switch is ON",
                action_hint="Operations → Risk Control: turn off kill switch",
            )
        )

    if max_loss_inr > 0:
        gates.append(
            _gate(
                id="loss_cap_inr",
                label="Daily loss cap (INR)",
                ok=not loss_cap_hit,
                detail=f"P&L ₹{pnl_inr:.0f} · cap -₹{abs(max_loss_inr):.0f}",
                value=round(pnl_inr, 2),
                threshold=-abs(max_loss_inr),
                action_hint="Stop trading for today or raise MAX_LOSS_INR_PER_DAY",
            )
        )

    if max_premium_inr > 0:
        gates.append(
            _gate(
                id="premium_cap_inr",
                label="Daily premium cap (INR)",
                ok=not premium_cap_hit,
                detail=f"Spent ₹{premium_spent:.0f} · cap ₹{max_premium_inr:.0f}",
                value=round(premium_spent, 2),
                threshold=max_premium_inr,
                action_hint="Reduce size or wait until tomorrow",
            )
        )

    steps = _step_live_summary(step_statuses)
    passed = sum(1 for s in steps if s.get("server_ok"))
    total = len(steps) if steps else 12

    from services.trail_ops import get_exit_policy_summary

    exit_policy = get_exit_policy_summary(
        (plan or {}).get("strategy_id"),
        quantity=int((plan or {}).get("quantity") or 0) or None,
    )

    return {
        "readiness_gates": gates,
        "can_autonomous_place": can_auto_place,
        "live_checklist_passed": passed,
        "live_checklist_total": total,
        "step_live": steps,
        "trade_plan_preview": _trade_plan_preview(plan),
        "exit_policy": exit_policy,
        "market_open": market_open,
        "paper_trading_mode": paper_trading_mode,
        "kite_connected": kite_connected,
        "pending_entry_order_id": pending_entry_order_id,
        "live_data_stale": not armed,
        "risk_limits_status": risk,
    }
