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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MomentumTrailConfig:
    enabled: bool
    breakeven_buffer: float
    breakeven_pct: float
    breakeven_r_fraction: float
    extend_gain_ratio: float
    lock_gain_ratio: float
    min_level_update: float
    stepped_rr: bool
    partial_exit_enabled: bool
    partial_exit_pct: float
    activation_hold_sec: int
    require_5m_close: bool
    time_stop_enabled: bool
    time_stop_minutes: int
    time_stop_before_ist: str
    time_stop_only_without_1r: bool
    trail_stale_alert_min: int
    gtt_fail_alert_threshold: int


def get_momentum_trail_config() -> MomentumTrailConfig:
    return MomentumTrailConfig(
        enabled=_env_bool("MOMENTUM_TRAIL_ENABLED", True),
        breakeven_buffer=_env_float("MOMENTUM_TRAIL_BREAKEVEN_BUFFER", 0.05),
        breakeven_pct=_env_float("MOMENTUM_TRAIL_BREAKEVEN_PCT", 0.004),
        breakeven_r_fraction=_env_float("MOMENTUM_TRAIL_BREAKEVEN_R_FRACTION", 0.25),
        extend_gain_ratio=_env_float("MOMENTUM_TRAIL_EXTEND_GAIN_RATIO", 0.5),
        lock_gain_ratio=_env_float("MOMENTUM_TRAIL_LOCK_GAIN_RATIO", 0.35),
        min_level_update=_env_float("MOMENTUM_TRAIL_MIN_UPDATE", 0.10),
        stepped_rr=_env_bool("STEPPED_RR_TRAIL", True),
        partial_exit_enabled=_env_bool("TRAIL_PARTIAL_EXIT_ENABLED", True),
        partial_exit_pct=_env_float("TRAIL_PARTIAL_EXIT_PCT", 0.40),
        activation_hold_sec=_env_int("TRAIL_ACTIVATION_HOLD_SEC", 30),
        require_5m_close=_env_bool("TRAIL_REQUIRE_5M_CLOSE", False),
        time_stop_enabled=_env_bool("TRAIL_TIME_STOP_ENABLED", True),
        time_stop_minutes=_env_int("TRAIL_TIME_STOP_MINUTES", 90),
        time_stop_before_ist=os.getenv("TRAIL_TIME_STOP_BEFORE_IST", "15:10").strip() or "15:10",
        time_stop_only_without_1r=_env_bool("TRAIL_TIME_STOP_ONLY_WITHOUT_1R", True),
        trail_stale_alert_min=_env_int("TRAIL_STALE_ALERT_MIN", 10),
        gtt_fail_alert_threshold=_env_int("TRAIL_GTT_FAIL_ALERT_THRESHOLD", 2),
    )


def get_regime_for_strategy(strategy_id: Optional[str]) -> str:
    sid = (strategy_id or "").strip().lower()
    if sid in {"bb_5m_mean_reversion"}:
        return "mean_reversion"
    if sid in {
        "orb_15m_breakout",
        "pdh_pdl_breakout",
        "ema_pullback_continuation",
        "long_atm_directional",
        "green_bar_sentinel_2nd_oi",
    }:
        return "trend"
    return "default"


def breakeven_stop(entry: float, risk_unit: float, cfg: MomentumTrailConfig) -> float:
    """SL after 1R: max(fixed tick, % of premium, fraction of R)."""
    entry = float(entry or 0)
    R = max(0.05, float(risk_unit or 0))
    pct_buf = entry * max(0.0, cfg.breakeven_pct)
    r_buf = R * max(0.0, cfg.breakeven_r_fraction)
    buf = max(cfg.breakeven_buffer, pct_buf, r_buf)
    return round_to_tick(entry + buf)


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


def gtt_tp_cap_for_trail(entry: float, target: float) -> float:
    """
    Wide broker TP so the OCO does not exit at the first target.
    Momentum trail monitor exits / extends via SL ratchet instead.
    """
    entry = float(entry or 0)
    target = float(target or 0)
    if entry <= 0 or target <= 0:
        return target
    gain = max(0.0, target - entry)
    mult = _env_float("MOMENTUM_TRAIL_GTT_TP_MULT", 1.35)
    cap = max(target * mult, target + gain * 1.5, entry + gain * 3.0)
    return round_to_tick(cap)


def compute_trailed_levels(
    *,
    entry: float,
    peak: float,
    ltp: float,
    current_sl: float,
    current_tp: float,
    trail_active: bool,
    cfg: Optional[MomentumTrailConfig] = None,
    initial_risk_unit: Optional[float] = None,
    initial_target: Optional[float] = None,
) -> Tuple[float, float, float, bool, str]:
    """
    Returns (new_sl, new_tp, new_peak, activated, note).
    Stepped mode: 1:1 initial target, SL→breakeven, TP→next R step.
    """
    cfg = cfg or get_momentum_trail_config()
    peak = max(peak, ltp, entry)

    R = float(initial_risk_unit or 0)
    if R <= 0 and initial_target and initial_target > entry:
        R = max(0.05, initial_target - entry)
    if R <= 0 and current_tp > entry:
        R = max(0.05, current_tp - entry)
    if R <= 0 and current_sl < entry:
        R = max(0.05, entry - current_sl)

    first_target = entry + R if R > 0 else current_tp
    activate = trail_active or should_activate_trail(
        ltp, entry, first_target, trail_active=trail_active
    )
    if not activate:
        return current_sl, current_tp, peak, False, ""

    be = breakeven_stop(entry, R, cfg)

    if cfg.stepped_rr and R > 0:
        step = max(1, int((peak - entry) / R))
        if not trail_active:
            new_sl = be
            new_tp = round_to_tick(entry + (step + 1) * R)
            note = (
                f"1:1 target ₹{first_target:.2f} reached @ {ltp:.2f} — "
                f"SL→breakeven ₹{new_sl:.2f}, next TP ₹{new_tp:.2f}"
            )
        else:
            locked_step = max(0, step - 1)
            locked_sl = round_to_tick(entry + locked_step * R) if locked_step >= 1 else be
            new_sl = round_to_tick(max(current_sl, be, locked_sl))
            new_tp = round_to_tick(entry + (step + 1) * R)
            note = (
                f"Step {step}R peak ₹{peak:.2f} — SL ₹{new_sl:.2f}, next TP ₹{new_tp:.2f}"
            )
    else:
        gain = max(0.0, peak - entry)
        if not trail_active:
            new_sl = be
            extension = gain * cfg.extend_gain_ratio
            new_tp = round_to_tick(max(current_tp, peak + extension))
            note = f"Target reached @ {ltp:.2f} — SL→breakeven ₹{new_sl:.2f}, TP extended→₹{new_tp:.2f}"
        else:
            locked_sl = peak - (gain * cfg.lock_gain_ratio)
            new_sl = round_to_tick(max(current_sl, be, locked_sl))
            extension = gain * cfg.extend_gain_ratio
            new_tp = round_to_tick(max(current_tp, peak + extension))
            note = f"Trail peak ₹{peak:.2f} — SL ₹{new_sl:.2f} TP ₹{new_tp:.2f}"

    new_sl = min(new_sl, ltp - 0.05) if ltp > 0.05 else new_sl
    if ltp > 0 and new_sl >= ltp:
        new_sl = round_to_tick(max(entry, ltp - max(0.10, ltp * 0.002)))
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
