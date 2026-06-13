"""Shared helpers for autonomous watch skip messages."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def normalize_skip_message(
    message: Optional[str],
    *,
    fallback: str = "Autonomous placement blocked",
) -> str:
    msg = (message or "").strip()
    return (msg[:240] if msg else fallback)


def validation_skip_message(
    preview: Optional[Dict[str, Any]],
    *,
    can_place: Optional[bool] = None,
) -> str:
    """Build a human-readable skip reason from preview validation (risk/reward, margin)."""
    if not isinstance(preview, dict):
        return "Waiting for margin/validation (no preview)"

    val = preview.get("validation")
    if isinstance(val, dict):
        reasons = val.get("failure_reasons") or []
        if reasons:
            return "; ".join(str(r) for r in reasons[:2])
        err = val.get("error")
        if err:
            return str(err)
        summary = val.get("summary")
        if summary and val.get("is_good_trade") is False:
            return str(summary)
        if val.get("is_good_trade") is False:
            return "Risk/reward validation failed — adjust size or levels"

    cp = can_place if can_place is not None else preview.get("can_place")
    if cp is False:
        return "Waiting for margin/validation (can_place=false)"
    return "Waiting for margin/validation or session (can_execute=false)"


def can_execute_block_errors(
    preview: Optional[Dict[str, Any]],
    plan: Optional[Dict[str, Any]] = None,
    *,
    segment: Optional[str] = None,
) -> List[str]:
    """Specific reasons place_trade blocked on can_execute (for UI, events, and journal)."""
    preview = preview or {}
    plan = plan or preview.get("trade_plan") or {}
    errors: List[str] = []

    if not preview.get("checklist_ready"):
        missing = preview.get("missing_steps") or []
        if missing:
            steps = ", ".join(str(int(i) + 1) for i in missing[:8])
            errors.append(f"Checklist incomplete (steps {steps})")
        else:
            errors.append("Checklist not ready")

    if plan.get("entry_ready") is not True:
        errors.append(
            str(plan.get("entry_block_reason") or "Entry not confirmed (entry_ready=false)")
        )

    val = preview.get("validation") if isinstance(preview.get("validation"), dict) else None
    if val and val.get("is_good_trade") is False:
        reasons = [str(r) for r in (val.get("failure_reasons") or []) if r]
        if reasons:
            for reason in reasons[:4]:
                if reason not in errors:
                    errors.append(reason)
        elif val.get("error"):
            errors.append(str(val["error"]))
        else:
            summary = val.get("summary")
            if summary and str(summary) != "Validation OK":
                errors.append(str(summary))
            else:
                errors.append("Risk/reward validation failed — adjust size or levels")

    paper = bool(preview.get("paper_trading_mode"))
    seg = (segment or "").strip().lower()
    if not paper and seg:
        try:
            from services.paper_trading import is_paper_mode_for_segment

            paper = is_paper_mode_for_segment(seg)
        except Exception:
            pass

    if paper and plan.get("tradingsymbol"):
        try:
            from services.paper_order_guard import paper_entry_levels_valid

            entry = float(plan.get("entry_limit_price") or plan.get("entry_premium") or 0)
            sl = float(plan.get("stop_loss_premium") or 0)
            tp = float(plan.get("target_premium") or 0)
            ok, msg = paper_entry_levels_valid(entry, sl, tp)
            if not ok and msg not in errors:
                errors.append(msg)
        except Exception:
            pass

    if (
        preview.get("can_place") is False
        and not preview.get("market_open")
        and not paper
        and not any("Market" in e for e in errors)
    ):
        errors.append("Market closed — live orders require session open")

    if not errors:
        errors.append(validation_skip_message(preview, can_place=preview.get("can_place")))
    return errors


def execute_gate_detail(
    preview: Optional[Dict[str, Any]],
    *,
    can_execute: bool,
    plan: Optional[Dict[str, Any]] = None,
    segment: Optional[str] = None,
) -> str:
    if can_execute:
        return "Margin and validation OK"
    reasons = can_execute_block_errors(preview, plan, segment=segment)
    return "; ".join(reasons[:2])[:200]


def format_place_skip_message(errors: List[str]) -> str:
    """Join place_trade errors for watch events (max 240 chars)."""
    parts = [str(e).strip() for e in errors if e and str(e).strip()]
    if not parts:
        return "Autonomous skipped: unknown"
    return f"Autonomous skipped: {'; '.join(parts)}"[:240]


def dump_checklist_steps(raw_list: List[Any], model_cls: Any) -> List[Dict[str, Any]]:
    """Parse checklist steps and return JSON-safe dicts (null bools → false)."""
    from services.checklist_step_utils import parse_checklist_steps

    parsed = parse_checklist_steps(raw_list or [], model_cls)
    out: List[Dict[str, Any]] = []
    for step in parsed:
        if hasattr(step, "model_dump"):
            out.append(step.model_dump())
        elif hasattr(step, "dict"):
            out.append(step.dict())
        else:
            out.append(dict(step))
    return out
