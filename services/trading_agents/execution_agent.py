"""ExecutionAgent — shared paper vs live place gates for segment watch loops."""
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


def _paper_plan_executable(preview: Dict[str, Any], plan: Dict[str, Any]) -> bool:
    """Paper watch must still pass risk/reward validation and min premium guards."""
    if plan.get("entry_ready") is not True:
        return False
    val = preview.get("validation")
    if isinstance(val, dict) and val.get("is_good_trade") is False:
        return False
    try:
        from services.paper_order_guard import paper_entry_levels_valid

        entry = float(plan.get("entry_limit_price") or plan.get("entry_premium") or 0)
        sl = float(plan.get("stop_loss_premium") or 0)
        tp = float(plan.get("target_premium") or 0)
        ok, _ = paper_entry_levels_valid(entry, sl, tp)
        return ok
    except Exception:
        return True


def resolve_can_execute(
    preview: Dict[str, Any],
    plan: Optional[Dict[str, Any]] = None,
    *,
    offhours_allowed: bool = False,
) -> bool:
    """
    True when place_trade may run:
    - Live: preview.can_place (validation + session, paper excluded from can_place)
    - Paper: checklist ready, entry confirmed, validation + premium levels OK
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
        return _paper_plan_executable(preview, plan)
    if offhours_allowed:
        return entry_ready
    return False
