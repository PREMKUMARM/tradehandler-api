"""Trailing stop-loss for BTCUSDT perp — ratchet SL after min R:R profit zone."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from utils.binance_order_utils import round_price


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.getenv(name, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class CryptoTrailConfig:
    enabled: bool
    activation_r: float
    lock_gain_ratio: float
    breakeven_buffer: float
    min_sl_update: float


def get_crypto_trail_config() -> CryptoTrailConfig:
    return CryptoTrailConfig(
        enabled=_env_bool("CRYPTO_TRAIL_ENABLED", True),
        activation_r=_env_float("CRYPTO_TRAIL_ACTIVATION_R", 0.75),
        lock_gain_ratio=_env_float("CRYPTO_TRAIL_LOCK_GAIN", 0.4),
        breakeven_buffer=_env_float("CRYPTO_TRAIL_BREAKEVEN_BUFFER", 5.0),
        min_sl_update=_env_float("CRYPTO_TRAIL_MIN_UPDATE", 8.0),
    )


def risk_points(entry: float, sl: float, side: str) -> float:
    side_u = str(side or "").upper()
    if side_u == "LONG":
        return max(0.0, entry - sl)
    return max(0.0, sl - entry)


def reward_points(entry: float, tp: float, side: str) -> float:
    side_u = str(side or "").upper()
    if side_u == "LONG":
        return max(0.0, tp - entry)
    return max(0.0, entry - tp)


def reward_risk_ratio(entry: float, sl: float, tp: float, side: str) -> float:
    r = risk_points(entry, sl, side)
    if r <= 0:
        return 0.0
    return reward_points(entry, tp, side) / r


def ensure_min_rr_tp(
    entry: float,
    sl: float,
    tp: float,
    side: str,
    *,
    min_rr: float,
    symbol: str = "BTCUSDT",
) -> float:
    """Extend TP if needed so reward ≥ min_rr × risk (e.g. 1.5)."""
    side_u = str(side or "").upper()
    r = risk_points(entry, sl, side_u)
    if r <= 0:
        return tp
    need = r * min_rr
    if side_u == "LONG":
        if reward_points(entry, tp, side_u) + 1e-9 >= need:
            return round_price(symbol, tp)
        return round_price(symbol, entry + need)
    if reward_points(entry, tp, side_u) + 1e-9 >= need:
        return round_price(symbol, tp)
    return round_price(symbol, entry - need)


def should_activate_trail(
    *,
    side: str,
    entry: float,
    initial_sl: float,
    ltp: float,
    min_tp: float,
    trail_active: bool,
    cfg: Optional[CryptoTrailConfig] = None,
) -> bool:
    if trail_active:
        return True
    cfg = cfg or get_crypto_trail_config()
    r = risk_points(entry, initial_sl, side)
    if r <= 0:
        return False
    side_u = str(side or "").upper()
    if side_u == "LONG":
        profit = ltp - entry
        hit_min_tp = ltp >= min_tp
    else:
        profit = entry - ltp
        hit_min_tp = ltp <= min_tp
    return profit >= r * cfg.activation_r or hit_min_tp


def compute_crypto_trail_sl(
    *,
    side: str,
    entry: float,
    initial_sl: float,
    min_tp: float,
    peak: float,
    ltp: float,
    current_sl: float,
    trail_active: bool,
    symbol: str = "BTCUSDT",
    cfg: Optional[CryptoTrailConfig] = None,
) -> Tuple[float, float, bool, str]:
    """
    Returns (new_sl, new_peak, activated, note).
    Ratchets SL only in favorable direction after activation.
    """
    cfg = cfg or get_crypto_trail_config()
    side_u = str(side or "").upper()
    if side_u == "LONG":
        peak = max(peak, ltp, entry)
    else:
        peak = min(peak, ltp, entry) if peak > 0 else min(ltp, entry)

    activate = should_activate_trail(
        side=side_u,
        entry=entry,
        initial_sl=initial_sl,
        ltp=ltp,
        min_tp=min_tp,
        trail_active=trail_active,
        cfg=cfg,
    )
    if not activate:
        return current_sl, peak, False, ""

    gain = (peak - entry) if side_u == "LONG" else (entry - peak)
    if not trail_active:
        if side_u == "LONG":
            new_sl = round_price(symbol, entry + cfg.breakeven_buffer)
        else:
            new_sl = round_price(symbol, entry - cfg.breakeven_buffer)
        note = f"Trail ON @ ${ltp:,.0f} — SL→breakeven ${new_sl:,.0f}"
    else:
        locked = gain * cfg.lock_gain_ratio
        if side_u == "LONG":
            new_sl = round_price(
                symbol,
                max(current_sl, entry + cfg.breakeven_buffer, peak - locked),
            )
            new_sl = min(new_sl, ltp - cfg.breakeven_buffer)
        else:
            new_sl = round_price(
                symbol,
                min(current_sl, entry - cfg.breakeven_buffer, peak + locked),
            )
            new_sl = max(new_sl, ltp + cfg.breakeven_buffer)
        note = f"Trail peak ${peak:,.0f} — SL ${new_sl:,.0f}"

    if side_u == "LONG":
        new_sl = max(new_sl, initial_sl)
    else:
        new_sl = min(new_sl, initial_sl)

    return new_sl, peak, True, note


def sl_changed_enough(old_sl: float, new_sl: float, cfg: Optional[CryptoTrailConfig] = None) -> bool:
    cfg = cfg or get_crypto_trail_config()
    return abs(new_sl - old_sl) >= cfg.min_sl_update
