"""
Nifty intraday regime gates — reduce whipsaw flip-flops (gap fade → bounce → PE chase).

Used by entry pricing (preview) and autonomous guards (place).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from services.kite_live_indicators import (
    bb_mean_reversion_index_gate,
    bb_session_bias_gate,
    bollinger_zone,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def nifty_intraday_regime_block(
    kind: str,
    *,
    spot: float,
    prev_close: float,
    day_low: float,
    day_high: float,
    index_bb_lower: Optional[float] = None,
    index_bb_middle: Optional[float] = None,
    index_bb_upper: Optional[float] = None,
    contract_zone: Optional[str] = None,
) -> Optional[str]:
    """
    Return block reason when index regime makes this option direction a poor fade.

    - PE into recovery after a lower-BB washout (today's 09:25–09:42 pattern)
    - CE into fade after an upper-BB extension
    - Mid-range chop when day range is wide
    """
    k = (kind or "CE").upper()
    if spot <= 0:
        return None

    bounce_recovery = _env_float("NIFTY_BOUNCE_RECOVERY_PTS", 30.0)
    fade_recovery = _env_float("NIFTY_FADE_RECOVERY_PTS", 30.0)
    bb_touch_buf = _env_float("NIFTY_BB_TOUCH_BUFFER_PTS", 15.0)
    chop_range = _env_float("NIFTY_CHOP_RANGE_PTS", 70.0)
    chop_lo = _env_float("NIFTY_CHOP_MID_LOW", 0.35)
    chop_hi = _env_float("NIFTY_CHOP_MID_HIGH", 0.65)

    dl = day_low if day_low > 0 else spot
    dh = day_high if day_high > 0 else spot

    if k == "PE" and dl > 0 and index_bb_lower is not None:
        low_at_band = dl <= float(index_bb_lower) + bb_touch_buf
        recovered = spot - dl
        if low_at_band and recovered >= bounce_recovery:
            return (
                f"PE blocked: Nifty bounced {recovered:.0f} pts from day low {dl:.0f} "
                f"(≥{bounce_recovery:.0f}) after lower-band washout — no PE into recovery"
            )

    if k == "CE" and dh > 0 and index_bb_upper is not None:
        high_at_band = dh >= float(index_bb_upper) - bb_touch_buf
        faded = dh - spot
        if high_at_band and faded >= fade_recovery:
            return (
                f"CE blocked: Nifty faded {faded:.0f} pts from day high {dh:.0f} "
                f"(≥{fade_recovery:.0f}) after upper-band extension — no CE into fade"
            )

    if dh > dl and (dh - dl) >= chop_range:
        pos = (spot - dl) / (dh - dl)
        zone = (contract_zone or "").lower()
        at_band = zone in ("lower", "upper", "middle")
        if chop_lo <= pos <= chop_hi and not at_band:
            return (
                f"Chop: Nifty mid-range ({pos:.0%} of {dh - dl:.0f} pt day range) — "
                "wait for index/contract band touch"
            )

    if prev_close > 0 and index_bb_lower and index_bb_middle and index_bb_upper:
        idx_block = bb_mean_reversion_index_gate(
            k,
            spot=spot,
            lower=float(index_bb_lower),
            middle=float(index_bb_middle),
            upper=float(index_bb_upper),
        )
        if idx_block:
            return idx_block

    if prev_close > 0 and contract_zone:
        sess = bb_session_bias_gate(
            k,
            spot=spot,
            prev_close=prev_close,
            contract_zone=contract_zone,
        )
        if sess:
            return sess

    return None


def nifty_bb_zone_from_plan(plan: Dict[str, Any]) -> Optional[str]:
    ind = plan.get("indicators") or {}
    kind = str(plan.get("option_type") or "CE").upper()
    px = _f(ind.get("option_ltp") or ind.get("contract_ltp") or plan.get("entry_premium"))
    lo = _f(ind.get("bb_lower"))
    mid = _f(ind.get("bb_middle"))
    hi = _f(ind.get("bb_upper"))
    if px is None or lo is None or mid is None or hi is None:
        return ind.get("bb_zone")
    return bollinger_zone(px, mid, hi, lo, kind).get("zone")


def nifty_regime_block_for_plan(plan: Dict[str, Any]) -> Optional[str]:
    """Regime gate from a trade plan snapshot (preview or place)."""
    if not plan:
        return "No trade plan"
    sid = str(plan.get("strategy_id") or "").lower()
    if sid and sid != "bb_5m_mean_reversion" and "bb" not in sid:
        return None

    ind = plan.get("indicators") or {}
    spot = _f(plan.get("nifty_spot") or ind.get("nifty_spot") or ind.get("underlying_spot"))
    if spot is None:
        return None

    prev_close = _f(ind.get("prev_close") or plan.get("prev_close")) or 0.0
    day_low = _f(ind.get("day_low") or plan.get("day_low")) or spot
    day_high = _f(ind.get("day_high") or plan.get("day_high")) or spot
    kind = str(plan.get("option_type") or "CE").upper()
    zone = nifty_bb_zone_from_plan(plan) or ind.get("bb_zone")

    return nifty_intraday_regime_block(
        kind,
        spot=spot,
        prev_close=prev_close,
        day_low=day_low,
        day_high=day_high,
        index_bb_lower=_f(ind.get("index_bb_lower")),
        index_bb_middle=_f(ind.get("index_bb_middle")),
        index_bb_upper=_f(ind.get("index_bb_upper")),
        contract_zone=str(zone) if zone else None,
    )


def nifty_autonomous_regime_allowed(plan: Dict[str, Any]) -> Tuple[bool, str]:
    """Extra autonomous-only checks (lot cap, regime, preferred BB zone)."""
    max_lots = _env_int("V2_AUTO_MAX_LOTS", 5)
    lot_size = int(plan.get("lot_size") or 75)
    qty = int(plan.get("quantity") or 0)
    num_lots = int(plan.get("num_lots") or 0)
    if num_lots <= 0 and qty > 0 and lot_size > 0:
        num_lots = qty // lot_size
    if num_lots > max_lots:
        return (
            False,
            f"Autonomous lot cap: {num_lots} lots exceeds max {max_lots} "
            f"(set V2_AUTO_MAX_LOTS to change)",
        )

    block = nifty_regime_block_for_plan(plan)
    if block:
        return False, block

    kind = str(plan.get("option_type") or "CE").upper()
    zone = (nifty_bb_zone_from_plan(plan) or "").lower()
    require_preferred = os.getenv("V2_AUTO_REQUIRE_PREFERRED_BB", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if require_preferred and zone:
        if kind == "CE" and zone not in ("lower", "middle"):
            return False, f"Autonomous CE requires contract lower/middle BB touch (now {zone})"
        if kind == "PE" and zone not in ("upper", "middle"):
            return False, f"Autonomous PE requires contract upper/middle BB touch (now {zone})"

    return True, "OK"


def opposite_direction_blocked(
    new_kind: str,
    last_kind: Optional[str],
    *,
    seconds_since_last: Optional[float],
) -> Optional[str]:
    """Block CE↔PE flips within cooldown (whipsaw churn)."""
    if not last_kind or not new_kind:
        return None
    nk = new_kind.upper()
    lk = last_kind.upper()
    if nk == lk:
        return None
    cooldown = _env_float("NIFTY_OPPOSITE_DIRECTION_COOLDOWN_SEC", 600.0)
    if cooldown <= 0 or seconds_since_last is None:
        return None
    if seconds_since_last < cooldown:
        wait = int(cooldown - seconds_since_last) + 1
        return (
            f"Direction flip {lk}→{nk} blocked — wait {wait}s "
            f"(NIFTY_OPPOSITE_DIRECTION_COOLDOWN_SEC={int(cooldown)})"
        )
    return None
