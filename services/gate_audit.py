"""
Daily gate audit — rate-limited EC2 logs for blocked vs allowed strategy signals.

Log prefix: [GateAudit:<segment>]
Env:
  GATE_AUDIT_DISABLE=1          — turn off audit lines
  GATE_AUDIT_LOG_SECONDS=120    — min seconds between duplicate lines
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from utils.logger import log_info

IST = ZoneInfo("Asia/Kolkata")

_trackers: Dict[str, "GateAuditState"] = {}


@dataclass
class GateAuditState:
    day: date
    evals: int = 0
    blocked_gate: int = 0
    blocked_execute: int = 0
    allowed_ready: int = 0
    autonomous_armed: int = 0
    placed: int = 0
    block_reasons: Counter = field(default_factory=Counter)
    last_log_key: Optional[str] = None
    last_log_at: Optional[datetime] = None


def _log_interval_sec() -> float:
    try:
        return max(30.0, min(600.0, float(os.getenv("GATE_AUDIT_LOG_SECONDS", "120"))))
    except (TypeError, ValueError):
        return 120.0


def _audit_disabled() -> bool:
    return os.getenv("GATE_AUDIT_DISABLE", "").strip().lower() in ("1", "true", "yes", "on")


def _state(segment: str) -> GateAuditState:
    seg = (segment or "unknown").lower()
    today = datetime.now(IST).date()
    st = _trackers.get(seg)
    if st is None or st.day != today:
        if st is not None and st.evals > 0:
            emit_gate_audit_day_summary(seg)
        st = GateAuditState(day=today)
        _trackers[seg] = st
    return st


def _classify_verdict(
    *,
    market_open: bool,
    entry_ready: bool,
    can_execute: bool,
    try_autonomous: bool,
    block_reason: str,
) -> tuple[str, str]:
    if not market_open:
        return "SESSION_CLOSED", ""
    block = (block_reason or "").strip()
    if entry_ready and try_autonomous:
        return "ALLOW_AUTONOMOUS", block
    if entry_ready and can_execute:
        return "ALLOW_READY", block
    if entry_ready:
        detail = block or "can_execute=false"
        return "BLOCKED_EXECUTE", detail
    if block:
        return "BLOCKED_GATE", block
    return "WAITING", block


def record_gate_audit(
    segment: str,
    plan: Dict[str, Any],
    preview: Dict[str, Any],
    *,
    try_autonomous: bool = False,
    market_open: bool = True,
    can_execute: Optional[bool] = None,
    entry_ready: Optional[bool] = None,
) -> None:
    """Log one rate-limited audit line per evaluation cycle."""
    if _audit_disabled() or not plan:
        return

    ready = entry_ready if entry_ready is not None else plan.get("entry_ready") is True
    exec_ok = (
        can_execute
        if can_execute is not None
        else bool(preview.get("can_execute") or preview.get("can_place"))
    )
    block = str(plan.get("entry_block_reason") or "")
    verdict, detail = _classify_verdict(
        market_open=market_open,
        entry_ready=ready,
        can_execute=exec_ok,
        try_autonomous=try_autonomous,
        block_reason=block,
    )

    st = _state(segment)
    st.evals += 1
    if verdict == "BLOCKED_GATE":
        st.blocked_gate += 1
        if detail:
            st.block_reasons[detail[:140]] += 1
    elif verdict == "BLOCKED_EXECUTE":
        st.blocked_execute += 1
    elif verdict in ("ALLOW_AUTONOMOUS", "ALLOW_READY"):
        st.allowed_ready += 1
        if verdict == "ALLOW_AUTONOMOUS":
            st.autonomous_armed += 1

    if verdict in ("SESSION_CLOSED", "WAITING"):
        return

    sym = str(plan.get("tradingsymbol") or "—")
    kind = str(plan.get("option_type") or "?")
    score = plan.get("entry_confirmation_score")
    strat = plan.get("strategy_name") or (
        (preview.get("strategy_analysis") or {}).get("selected_name") or "?"
    )
    log_key = f"{verdict}|{sym}|{detail[:96]}"
    now = datetime.now(IST)
    force = verdict.startswith("ALLOW") or verdict == "BLOCKED_GATE"
    should_log = force or st.last_log_key != log_key or st.last_log_at is None
    if should_log and st.last_log_at and not force:
        should_log = (now - st.last_log_at).total_seconds() >= _log_interval_sec()
    if not should_log:
        return

    st.last_log_key = log_key
    st.last_log_at = now
    reason_part = f' reason="{detail}"' if detail else ""
    log_info(
        f"[GateAudit:{segment.lower()}] {verdict} {kind} {sym} "
        f"score={score} strategy={strat}{reason_part}"
    )


def record_gate_audit_placed(segment: str, tradingsymbol: str) -> None:
    if _audit_disabled():
        return
    st = _state(segment)
    st.placed += 1
    log_info(f"[GateAudit:{segment.lower()}] PLACED {tradingsymbol}")


def emit_gate_audit_day_summary(segment: str) -> None:
    if _audit_disabled():
        return
    seg = (segment or "unknown").lower()
    st = _trackers.get(seg)
    if not st or st.evals <= 0:
        return
    top = st.block_reasons.most_common(4)
    top_s = ", ".join(f'"{reason}"×{count}' for reason, count in top) if top else "none"
    log_info(
        f"[GateAudit:{seg}] day_summary date={st.day} evals={st.evals} "
        f"blocked_gate={st.blocked_gate} blocked_execute={st.blocked_execute} "
        f"allowed_ready={st.allowed_ready} autonomous_armed={st.autonomous_armed} "
        f"placed={st.placed} top_blocks={top_s}"
    )
