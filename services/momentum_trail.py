"""
Momentum trailing exit — after initial target is reached (profitable long), extend TP
and ratchet SL toward breakeven / locked profit instead of exiting at first target.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class MomentumTrailConfig:
    enabled: bool
    breakeven_buffer: float
    extend_gain_ratio: float
    lock_gain_ratio: float
    min_level_update: float


def get_momentum_trail_config() -> MomentumTrailConfig:
    return MomentumTrailConfig(
        enabled=_env_bool("MOMENTUM_TRAIL_ENABLED", True),
        breakeven_buffer=_env_float("MOMENTUM_TRAIL_BREAKEVEN_BUFFER", 0.05),
        extend_gain_ratio=_env_float("MOMENTUM_TRAIL_EXTEND_GAIN_RATIO", 0.5),
        lock_gain_ratio=_env_float("MOMENTUM_TRAIL_LOCK_GAIN_RATIO", 0.35),
        min_level_update=_env_float("MOMENTUM_TRAIL_MIN_UPDATE", 0.10),
    )


def is_profitable_long(entry: float, ltp: float) -> bool:
    return entry > 0 and ltp > entry


def should_activate_trail(
    ltp: float,
    entry: float,
    target: float,
    *,
    trail_active: bool,
) -> bool:
    """Activate once LTP reaches initial target while still in profit."""
    if trail_active:
        return True
    if target <= 0 or entry <= 0:
        return False
    return ltp >= target and ltp > entry


def compute_trailed_levels(
    *,
    entry: float,
    peak: float,
    ltp: float,
    current_sl: float,
    current_tp: float,
    trail_active: bool,
    cfg: Optional[MomentumTrailConfig] = None,
) -> Tuple[float, float, float, bool, str]:
    """
    Returns (new_sl, new_tp, new_peak, activated, note).
    Only ratchets SL/TP upward for long premium exits.
    """
    cfg = cfg or get_momentum_trail_config()
    peak = max(peak, ltp, entry)
    gain = max(0.0, peak - entry)

    activate = should_activate_trail(ltp, entry, current_tp, trail_active=trail_active)
    if not activate:
        return current_sl, current_tp, peak, False, ""

    if not trail_active:
        new_sl = round_to_tick(entry + cfg.breakeven_buffer)
        extension = gain * cfg.extend_gain_ratio
        new_tp = round_to_tick(max(current_tp, peak + extension))
        note = f"Target reached @ {ltp:.2f} — SL→breakeven ₹{new_sl:.2f}, TP extended→₹{new_tp:.2f}"
    else:
        locked_sl = peak - (gain * cfg.lock_gain_ratio)
        new_sl = round_to_tick(max(current_sl, entry + cfg.breakeven_buffer, locked_sl))
        extension = gain * cfg.extend_gain_ratio
        new_tp = round_to_tick(max(current_tp, peak + extension))
        note = f"Trail peak ₹{peak:.2f} — SL ₹{new_sl:.2f} TP ₹{new_tp:.2f}"

    new_sl = min(new_sl, ltp - 0.05) if ltp > new_sl + 0.05 else new_sl
    new_tp = max(new_tp, ltp + 0.05)
    return new_sl, new_tp, peak, True, note


def levels_changed_enough(
    old_sl: float,
    old_tp: float,
    new_sl: float,
    new_tp: float,
    *,
    cfg: Optional[MomentumTrailConfig] = None,
) -> bool:
    cfg = cfg or get_momentum_trail_config()
    return (
        abs(new_sl - old_sl) >= cfg.min_level_update
        or abs(new_tp - old_tp) >= cfg.min_level_update
    )


def gtt_triggers_for_levels(
    entry_ref: float,
    sl_prem: float,
    tgt_prem: float,
    last_price: float,
) -> Tuple[float, float, float]:
    """OCO trigger prices for GTT modify (same rules as commodity plan helper)."""
    last_price = float(last_price or entry_ref or sl_prem)
    sl_prem = float(sl_prem)
    tgt_prem = float(tgt_prem)
    min_gap = max(0.05, last_price * 0.0026) if last_price > 0 else 0.05

    sl_trigger = round_to_tick(sl_prem * 1.002)
    tp_trigger = round_to_tick(tgt_prem * 0.998)

    if last_price > 0:
        if last_price - sl_trigger < min_gap:
            sl_trigger = round_to_tick(max(0.05, last_price - min_gap))
        if tp_trigger - last_price < min_gap:
            tp_trigger = round_to_tick(last_price + min_gap)
        if sl_trigger >= last_price:
            sl_trigger = round_to_tick(max(0.05, last_price - min_gap))
        if tp_trigger <= last_price:
            tp_trigger = round_to_tick(last_price + min_gap)
    return sl_trigger, tp_trigger, last_price
