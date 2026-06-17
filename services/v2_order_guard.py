"""Duplicate-order and entry-quality guards for Nifty V2 autonomous place."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from services.trading_agents.guard_agent import (
    NIFTY_GUARD,
    autonomous_place_allowed as _autonomous_place_allowed,
    entry_quality_for_autonomous as _entry_quality,
    has_exchange_position,
    has_pending_exchange_order,
    min_entry_confirmation_score as _min_score,
)


def min_entry_confirmation_score() -> int:
    return _min_score(NIFTY_GUARD)


def entry_quality_for_autonomous(plan: Dict[str, Any]) -> Tuple[bool, str]:
    return _entry_quality(plan, config=NIFTY_GUARD)


def has_pending_nfo_order(tradingsymbol: str) -> Tuple[bool, str]:
    return has_pending_exchange_order(
        tradingsymbol, exchange="NFO", log_prefix=NIFTY_GUARD.log_prefix
    )


def has_nfo_position(tradingsymbol: str) -> Tuple[bool, str]:
    return has_exchange_position(
        tradingsymbol, exchange="NFO", log_prefix=NIFTY_GUARD.log_prefix
    )


def autonomous_place_allowed(
    plan: Dict[str, Any],
    *,
    placed_today: bool,
    segment: str = "nifty50",
    last_trade_kind: Optional[str] = None,
    seconds_since_last_trade: Optional[float] = None,
) -> Tuple[bool, str]:
    ok, msg = _autonomous_place_allowed(
        plan, config=NIFTY_GUARD, placed_today=placed_today
    )
    if not ok:
        return ok, msg
    from services.nifty_regime_guard import (
        nifty_autonomous_regime_allowed,
        opposite_direction_blocked,
    )

    kind = str(plan.get("option_type") or "CE")
    flip = opposite_direction_blocked(
        kind, last_trade_kind, seconds_since_last=seconds_since_last_trade
    )
    if flip:
        return False, flip
    return nifty_autonomous_regime_allowed(plan)
