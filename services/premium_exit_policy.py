"""
Minimum premium SL/TP spacing for long option exits (MCX/NFO).

Prevents GTT OCO targets a few ticks above entry when underlying mapping collapses.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

from utils.kite_order_utils import round_to_tick


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.getenv(name, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def entry_initial_rr() -> float:
    """Initial GTT target R-multiple (1:1 default). Upside handled by momentum trail."""
    return max(1.0, min(5.0, _env_float("ENTRY_INITIAL_RR", 1.0)))


def entry_validation_skips_reward() -> bool:
    """When true, entry validation only checks risk cap — not reward ratio."""
    return _env_bool("ENTRY_VALIDATION_SKIP_REWARD", True)


def enforce_min_premium_exits(
    entry: float,
    sl: float,
    tp: float,
    *,
    risk_pct: float = 1.0,
    reward_pct: float = 2.0,
    min_rr: Optional[float] = None,
    min_tp_pct: Optional[float] = None,
    min_tp_inr: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Long premium BUY exits: SL below entry, TP above entry with minimum R:R.

    Uses the wider of policy % (risk/reward) and actual SL distance × R:R.
    """
    entry = max(0.05, float(entry))
    sl = float(sl)
    tp = float(tp)
    risk_pct = max(0.1, float(risk_pct or 1.0))
    reward_pct = max(0.1, float(reward_pct or 2.0))
    rr = float(min_rr) if min_rr is not None else entry_initial_rr()
    rr = max(1.0, min(5.0, rr))

    min_risk_inr = max(
        entry * risk_pct / 100.0,
        _env_float("PREMIUM_EXIT_MIN_RISK_INR", 3.0),
        entry * _env_float("PREMIUM_EXIT_MIN_RISK_PCT", 0.008),
    )
    min_tp_pct_val = min_tp_pct if min_tp_pct is not None else _env_float(
        "PREMIUM_EXIT_MIN_TP_PCT", 0.015
    )
    min_tp_inr_val = min_tp_inr if min_tp_inr is not None else _env_float(
        "PREMIUM_EXIT_MIN_TP_INR", 5.0
    )

    prem_risk_actual = max(0.05, entry - sl) if sl < entry else 0.0
    prem_risk = max(prem_risk_actual, min_risk_inr)

    if sl >= entry - 0.05:
        sl = round_to_tick(max(0.05, entry - min_risk_inr))
        prem_risk = max(0.05, entry - sl)

    min_reward = max(
        prem_risk * rr,
        entry * reward_pct / 100.0,
        entry * min_tp_pct_val,
        min_tp_inr_val,
    )

    if tp < entry + min_reward:
        tp = entry + min_reward

    sl = round_to_tick(sl)
    tp = round_to_tick(tp)

    if sl >= tp:
        sl = round_to_tick(max(0.05, entry - min_risk_inr))
        tp = round_to_tick(entry + min_reward)
    if tp <= entry:
        tp = round_to_tick(entry + min_reward)
    return sl, tp


def policy_from_plan(plan: dict) -> Tuple[float, float, Optional[float]]:
    ind = plan.get("indicators") or {}
    risk_pct = float(ind.get("risk_pct") or plan.get("risk_pct") or 1.0)
    reward_pct = float(ind.get("reward_pct") or plan.get("reward_pct") or 2.0)
    rr = plan.get("reward_ratio")
    try:
        rr_f = float(rr) if rr is not None else None
    except (TypeError, ValueError):
        rr_f = None
    return risk_pct, reward_pct, rr_f


def enforce_plan_exits(plan: dict, *, entry: Optional[float] = None) -> dict:
    """Apply minimum exit spacing on a trade plan dict (mutates copy)."""
    out = dict(plan)
    ep = float(
        entry
        or out.get("entry_limit_price")
        or out.get("entry_premium")
        or 0
    )
    if ep <= 0:
        return out
    sl = float(out.get("stop_loss_premium") or 0)
    tp = float(out.get("target_premium") or 0)
    risk_pct, reward_pct, rr = policy_from_plan(out)
    sl, tp = enforce_min_premium_exits(
        ep, sl, tp, risk_pct=risk_pct, reward_pct=reward_pct, min_rr=rr
    )
    out["stop_loss_premium"] = sl
    out["target_premium"] = tp
    out["entry_limit_price"] = ep
    out["entry_premium"] = ep
    return out
