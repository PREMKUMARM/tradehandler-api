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
