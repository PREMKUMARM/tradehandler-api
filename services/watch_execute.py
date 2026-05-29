"""Shared execute gates for V2 / commodity watch loops and place_trade."""
from __future__ import annotations

from typing import Any, Dict, Optional


def is_paper_trading(preview: Optional[Dict[str, Any]] = None) -> bool:
    if preview and preview.get("paper_trading_mode"):
        return True
    try:
        from services.paper_trading import is_paper_mode

        return is_paper_mode()
    except Exception:
        return False


def resolve_can_execute(
    preview: Dict[str, Any],
    plan: Optional[Dict[str, Any]] = None,
    *,
    offhours_allowed: bool = False,
) -> bool:
    """
    True when place_trade may run:
    - Live: preview.can_place (validation + session, paper excluded from can_place)
    - Paper: all checklist steps complete with a trade plan
    - Off-hours test bypass when enabled
    """
    if not plan:
        plan = preview.get("trade_plan")
    if not plan:
        return False

    checklist_ready = bool(preview.get("checklist_ready"))
    entry_ready = plan.get("entry_ready") is True
    if bool(preview.get("can_place")):
        return entry_ready
    if is_paper_trading(preview) and checklist_ready:
        return entry_ready
    if offhours_allowed:
        return True
    return False
