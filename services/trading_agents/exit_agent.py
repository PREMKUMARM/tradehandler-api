"""ExitAgent — minimum premium exits and momentum trailing helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from services.momentum_trail import (
    compute_trailed_levels,
    get_momentum_trail_config,
    gtt_tp_cap_for_trail,
    should_activate_trail,
)
from services.premium_exit_policy import enforce_min_premium_exits, enforce_plan_exits


def enforce_exits_on_plan(plan: Dict[str, Any], *, entry: float) -> Dict[str, Any]:
    """Apply minimum premium SL/TP spacing to a trade plan."""
    return enforce_plan_exits(plan, entry=entry)


def min_premium_exits(
    entry: float,
    sl: float,
    tp: float,
    *,
    risk_pct: float = 1.0,
    reward_pct: float = 2.0,
) -> Tuple[float, float]:
    return enforce_min_premium_exits(entry, sl, tp, risk_pct=risk_pct, reward_pct=reward_pct)


def trail_after_target(
    *,
    entry: float,
    ltp: float,
    peak: float,
    current_sl: float,
    current_tp: float,
    trail_active: bool,
) -> Tuple[float, float, float, bool, str]:
    """Delegate to momentum trail engine (profitable long only)."""
    cfg = get_momentum_trail_config()
    if not cfg.enabled:
        return current_sl, current_tp, peak, trail_active, ""
    return compute_trailed_levels(
        entry=entry,
        peak=peak,
        ltp=ltp,
        current_sl=current_sl,
        current_tp=current_tp,
        trail_active=trail_active,
        cfg=cfg,
    )


def broker_tp_for_trail(plan: Dict[str, Any]) -> Optional[float]:
    """Wide GTT TP when software manages first target."""
    entry = float(plan.get("entry_limit_price") or plan.get("entry_premium") or 0)
    tp = float(plan.get("target_premium") or 0)
    if entry <= 0:
        return None
    return gtt_tp_cap_for_trail(entry, tp)


__all__ = [
    "broker_tp_for_trail",
    "enforce_exits_on_plan",
    "min_premium_exits",
    "should_activate_trail",
    "trail_after_target",
]
