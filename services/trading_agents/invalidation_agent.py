"""InvalidationAgent — cancel pending LIMIT when live setup no longer matches."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

from services.trading_agents.types import is_filled_order_status, is_open_order_status


def _float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def spot_from_plan(plan: Optional[Dict[str, Any]]) -> Optional[float]:
    if not plan:
        return None
    for key in ("nifty_spot", "underlying_spot", "crude_spot", "spot"):
        v = _float(plan.get(key))
        if v is not None:
            return v
    ind = plan.get("indicators") or {}
    for key in ("underlying_spot", "nifty_spot", "crude_spot", "spot"):
        v = _float(ind.get(key))
        if v is not None:
            return v
    return None


def _strike_from_symbol(sym: str) -> Optional[int]:
    m = re.search(r"(\d{4,5})(?:PE|CE)$", (sym or "").upper())
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _cancel_on_symbol_drift() -> bool:
    return os.getenv("COMMODITY_WATCH_CANCEL_ON_SYMBOL_DRIFT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _symbol_drift_min_strike_steps() -> int:
    try:
        return max(1, int(os.getenv("COMMODITY_WATCH_SYMBOL_DRIFT_MIN_STRIKES", "2") or 2))
    except (TypeError, ValueError):
        return 2


def pending_entry_invalidated(
    *,
    pending_plan: Optional[Dict[str, Any]],
    pending_symbol: Optional[str],
    current_plan: Optional[Dict[str, Any]],
    min_score: int,
    paper_mode: bool = False,
) -> Tuple[bool, str]:
    """
    Return (True, reason) when an unfilled or just-filled entry should not proceed.

    Uses the plan snapshot from placement plus the live preview plan for spot/score.
    """
    if paper_mode:
        return False, ""

    pending = pending_plan or {}
    current = current_plan or {}

    if current and current.get("entry_ready") is not True:
        return True, str(current.get("entry_block_reason") or "Entry no longer confirmed")

    if current:
        score = int(current.get("entry_confirmation_score") or 0)
        if score < min_score:
            return True, f"Score dropped to {score} (<{min_score})"
        block = current.get("entry_block_reason")
        if block:
            return True, str(block)

    if not pending:
        if not current:
            return True, "No plan (setup invalidated)"
        return False, ""

    pend_sym = (pending_symbol or pending.get("tradingsymbol") or "").upper()
    curr_sym = (current.get("tradingsymbol") or "").upper()
    if pend_sym and curr_sym and pend_sym != curr_sym:
        if _cancel_on_symbol_drift():
            pend_kind = str(pending.get("option_type") or ("PE" if "PE" in pend_sym else "CE")).upper()
            curr_kind = str(current.get("option_type") or ("PE" if "PE" in curr_sym else "CE")).upper()
            if pend_kind != curr_kind:
                return True, f"Direction changed ({pend_kind}→{curr_kind}) — canceling {pend_sym}"
            pend_strike = _strike_from_symbol(pend_sym)
            curr_strike = _strike_from_symbol(curr_sym)
            step = 50  # CRUDEOILM strike step
            min_steps = _symbol_drift_min_strike_steps()
            if (
                pend_strike is not None
                and curr_strike is not None
                and abs(curr_strike - pend_strike) >= step * min_steps
            ):
                return True, (
                    f"Setup moved to {curr_sym} (≥{min_steps} strikes away) — canceling {pend_sym}"
                )
        # Adjacent-strike flicker (8200↔8250↔8300) — keep pending LIMIT; real invalidation below.

    spot = spot_from_plan(current) or spot_from_plan(pending)
    if spot is None:
        return False, ""

    kind = str(pending.get("option_type") or "CE").upper()
    spot_sl = _float(pending.get("spot_stop_loss"))
    if spot_sl is not None:
        if kind == "PE" and spot >= spot_sl:
            return (
                True,
                f"Spot {spot:.0f} at/above setup SL {spot_sl:.0f} — pending entry no longer valid",
            )
        if kind == "CE" and spot <= spot_sl:
            return (
                True,
                f"Spot {spot:.0f} at/below setup SL {spot_sl:.0f} — pending entry no longer valid",
            )

    strategy_id = str(pending.get("strategy_id") or "").lower()
    strat_name = str(pending.get("strategy_name") or "").lower()
    ind = (current.get("indicators") or pending.get("indicators") or {})

    if "orb" in strategy_id or "orb" in strat_name or "opening range" in strat_name:
        or_l, or_h = ind.get("or_low"), ind.get("or_high")
        if kind == "PE" and or_l is not None:
            trigger = float(or_l)
            if spot >= trigger:
                return (
                    True,
                    f"ORB PE setup reset — spot {spot:.0f} back above OR low {trigger:.0f}",
                )
        if kind == "CE" and or_h is not None:
            trigger = float(or_h)
            if spot <= trigger:
                return (
                    True,
                    f"ORB CE setup reset — spot {spot:.0f} back below OR high {trigger:.0f}",
                )

    trig = _float(pending.get("entry_spot_trigger"))
    if trig is not None:
        if kind == "PE" and spot >= trig:
            return (
                True,
                f"Breakdown invalid — spot {spot:.0f} reclaimed entry trigger {trig:.0f}",
            )
        if kind == "CE" and spot <= trig:
            return (
                True,
                f"Breakout invalid — spot {spot:.0f} below entry trigger {trig:.0f}",
            )

    if "pdh" in strategy_id or "pdl" in strategy_id or "pdh" in strat_name or "pdl" in strat_name:
        if kind == "PE":
            pdl = _float(ind.get("pdl"))
            if pdl is not None and spot >= pdl:
                return (
                    True,
                    f"PDL break failed — spot {spot:.0f} back above PDL {pdl:.0f}",
                )
        if kind == "CE":
            pdh = _float(ind.get("pdh"))
            if pdh is not None and spot <= pdh:
                return (
                    True,
                    f"PDH break failed — spot {spot:.0f} back below PDH {pdh:.0f}",
                )

    return False, ""


__all__ = [
    "pending_entry_invalidated",
    "spot_from_plan",
    "is_open_order_status",
    "is_filled_order_status",
]
