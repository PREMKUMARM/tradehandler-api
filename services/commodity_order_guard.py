"""Duplicate-order and entry-quality guards for commodity autonomous place."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from services.trading_agents.guard_agent import (
    COMMODITY_GUARD,
    autonomous_place_allowed as _autonomous_place_allowed,
    entry_quality_for_autonomous as _entry_quality,
    has_exchange_position,
    has_pending_exchange_order,
    min_entry_confirmation_score as _min_score,
)


def min_entry_confirmation_score() -> int:
    return _min_score(COMMODITY_GUARD)


def entry_quality_for_autonomous(plan: Dict[str, Any]) -> Tuple[bool, str]:
    return _entry_quality(plan, config=COMMODITY_GUARD)


def has_pending_mcx_order(tradingsymbol: str) -> Tuple[bool, str]:
    return has_pending_exchange_order(
        tradingsymbol, exchange="MCX", log_prefix=COMMODITY_GUARD.log_prefix
    )


def has_mcx_position(tradingsymbol: str) -> Tuple[bool, str]:
    return has_exchange_position(
        tradingsymbol, exchange="MCX", log_prefix=COMMODITY_GUARD.log_prefix
    )


def autonomous_place_allowed(
    plan: Dict[str, Any],
    *,
    placed_today: bool,
    segment: str = "commodity",
) -> Tuple[bool, str]:
    return _autonomous_place_allowed(
        plan, config=COMMODITY_GUARD, placed_today=placed_today
    )
