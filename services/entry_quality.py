"""Shared entry filters — backtest + live 20rupees-style band entries."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Nifty50 point thresholds in .env are tuned at ~this index level; Sensex scales from it.
NIFTY_INDEX_REF_OPEN = 24000.0


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def entry_momentum_required() -> bool:
    """Require bullish option 5m bar (close > open) — filters weak band closes."""
    return _env_bool("ENTRY_REQUIRE_MOMENTUM_BAR", True)


def entry_index_momentum_required() -> bool:
    """CE when index spot rose vs prior bar; PE when index spot fell."""
    return _env_bool("ENTRY_REQUIRE_INDEX_MOMENTUM", True)


def entry_direction_aligned_required() -> bool:
    """CE only when spot >= prev close; PE only when spot < prev close."""
    return _env_bool("ENTRY_REQUIRE_DIRECTION_ALIGNED", False)


def entry_t1_close_confirm() -> bool:
    """Ratchet SL at T1 only when bar closes at/above T1 (not wick touch)."""
    return _env_bool("EXIT_T1_CLOSE_CONFIRM", False)


def block_reentry_after_loss() -> bool:
    return _env_bool("ENTRY_BLOCK_REENTRY_AFTER_LOSS", True)


def block_reentry_after_breakeven() -> bool:
    return _env_bool("ENTRY_BLOCK_REENTRY_AFTER_BREAKEVEN", True)


def max_trades_per_contract_per_day() -> int:
    try:
        return max(1, int(os.getenv("MAX_TRADES_PER_CONTRACT_PER_DAY", "1") or 1))
    except (TypeError, ValueError):
        return 1


def max_trades_per_session_day() -> int:
    """Cap total round-trips per index session day (all contracts)."""
    try:
        return max(1, int(os.getenv("MAX_TRADES_PER_SESSION_DAY", "1") or 1))
    except (TypeError, ValueError):
        return 1


def entry_band_limits() -> tuple[float, float]:
    try:
        lo = float(os.getenv("ENTRY_BAND_LOW", os.getenv("SENSEX_ENTRY_BAND_LOW", "17")) or 17)
        hi = float(os.getenv("ENTRY_BAND_HIGH", os.getenv("SENSEX_ENTRY_BAND_HIGH", "23")) or 23)
        return lo, hi
    except (TypeError, ValueError):
        return 17.0, 23.0


def entry_day_aligned_required() -> bool:
    """CE when index up from open; PE when index down from open (intraday trend)."""
    return _env_bool("ENTRY_REQUIRE_DAY_ALIGNED", True)


def exit_model() -> str:
    """stepped (T1→entry, T2→T1, …) or t1_scalp (full exit at 1R — fewer breakeven traps)."""
    return (os.getenv("EXIT_MODEL") or "t1_scalp").strip().lower()


def entry_mid_band_only() -> bool:
    """Require confirm close in inner 60% of band (filters band edges)."""
    return _env_bool("ENTRY_MID_BAND_ONLY", False)


def entry_max_close_position_in_bar() -> float:
    """
    Reject entries when bar close sits in the top of the 5m range (chase-the-wick).
    0 disables; 0.75 = close must be in bottom 75% of bar (low→high).
    """
    try:
        raw = float(os.getenv("ENTRY_MAX_CLOSE_POSITION_IN_BAR", "0.75") or 0.75)
    except (TypeError, ValueError):
        raw = 0.75
    if raw <= 0 or raw >= 1:
        return 0.0
    return raw


def entry_mid_band_limits(band_low: float, band_high: float) -> tuple[float, float]:
    """Inner band for mid-band filter."""
    width = band_high - band_low
    pad = width * 0.2
    return band_low + pad, band_high - pad


def day_direction_kind(index_open: float, spot: float) -> Optional[str]:
    """CE when index is above day open, PE when below; None when flat (skip entry)."""
    if index_open <= 0 or spot <= 0:
        return None
    if spot > index_open:
        return "CE"
    if spot < index_open:
        return "PE"
    return None


def auto_entry_kind(index_open: float, spot: float, prev_close: float = 0.0) -> Optional[str]:
    """AUTO leg: index day direction when enabled, else opening-gap direction."""
    if entry_day_aligned_required():
        return day_direction_kind(index_open, spot)
    from services.sensex_constants import sensex_gap_direction_kind

    return sensex_gap_direction_kind(index_open or spot, prev_close)


def entry_day_aligned_ok(*, kind: str, index_open: float, spot: float) -> bool:
    if not entry_day_aligned_required() or index_open <= 0 or spot <= 0:
        return True
    expected = day_direction_kind(index_open, spot)
    if expected is None:
        return False
    return (kind or "CE").upper() == expected


def confirmation_candle_required() -> bool:
    """Two-candle entry: setup bar touches band, next bar confirms direction."""
    return _env_bool("ENTRY_REQUIRE_CONFIRMATION_CANDLE", False)


def band_setup_bar_ok(
    *,
    kind: str,
    setup_open: float,
    setup_high: float,
    setup_low: float,
    setup_close: float,
    band_low: float,
    band_high: float,
) -> bool:
    """
  Setup bar approached premium band (pullback entry model).
  CE: wick dipped into band from above; PE: wick rallied into band from below.
  Also accept close inside band on setup bar.
    """
    if band_bar_touched_band(setup_low, setup_high, setup_close, band_low, band_high):
        return True
    k = (kind or "CE").upper()
    if k == "CE":
        return setup_low <= band_high and setup_close >= band_low * 0.85
    return setup_high >= band_low and setup_close <= band_high * 1.15


def band_bar_touched_band(
    bar_low: float,
    bar_high: float,
    bar_close: float,
    band_low: float,
    band_high: float,
) -> bool:
    """Bar interacted with premium band (close in band or wick overlap)."""
    if bar_close > 0 and band_low <= bar_close <= band_high:
        return True
    return bar_low <= band_high and bar_high >= band_low


def confirmation_candle_entry_ok(
    *,
    kind: str,
    setup_open: float,
    setup_high: float,
    setup_low: float,
    setup_close: float,
    confirm_open: float,
    confirm_high: float,
    confirm_low: float,
    confirm_close: float,
    band_low: float,
    band_high: float,
) -> tuple[bool, str]:
    """
    20rupees two-candle entry:
      1) Setup bar (prior 5m) touched or closed in premium band
      2) Confirmation bar closes in band with directional body and breaks setup extreme
    """
    if not confirmation_candle_required():
        return True, ""

    if not band_setup_bar_ok(
        kind=kind,
        setup_open=setup_open,
        setup_high=setup_high,
        setup_low=setup_low,
        setup_close=setup_close,
        band_low=band_low,
        band_high=band_high,
    ):
        return False, "no_setup"

    if confirm_close <= 0 or not (band_low <= confirm_close <= band_high):
        return False, "confirm_not_in_band"

    if entry_mid_band_only():
        mid_lo, mid_hi = entry_mid_band_limits(band_low, band_high)
        if not (mid_lo <= confirm_close <= mid_hi):
            return False, "confirm_not_mid_band"

    k = (kind or "CE").upper()
    if k == "CE":
        if confirm_close <= confirm_open:
            return False, "confirm_not_bullish"
        if setup_close > 0 and confirm_close <= setup_close:
            return False, "confirm_no_progress"
    else:
        if confirm_close >= confirm_open:
            return False, "confirm_not_bearish"
        if setup_close > 0 and confirm_close >= setup_close:
            return False, "confirm_no_progress"

    return True, ""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def entry_scan_warmup_minutes() -> int:
    """Skip first N minutes of entry scan (avoids 14:00 open-bar noise)."""
    try:
        return max(0, int(os.getenv("ENTRY_SCAN_WARMUP_MIN", "5") or 5))
    except (TypeError, ValueError):
        return 5


def entry_index_segment(segment: str) -> str:
    """Normalize backtest/live segment id."""
    s = (segment or "sensex").strip().lower().replace("-", "").replace("_", "")
    if s in ("nifty50", "nifty"):
        return "nifty50"
    return "sensex"


def entry_pt_scale(segment: str, index_open: float) -> float:
    """
    Scale Nifty-tuned point thresholds for higher-index instruments (Sensex ~3×).
    Nifty50 always uses scale 1.0 (absolute .env points).
    """
    if entry_index_segment(segment) == "nifty50":
        return 1.0
    if index_open > 0:
        ref = max(1.0, _env_float("NIFTY_INDEX_REF_OPEN", NIFTY_INDEX_REF_OPEN))
        return index_open / ref
    return max(1.0, _env_float("SENSEX_INDEX_PT_SCALE", 74800.0 / NIFTY_INDEX_REF_OPEN))


def _segment_pts(
    base_env: str,
    sensex_env: str,
    default: float,
    *,
    segment: str,
    index_open: float,
) -> float:
    seg = entry_index_segment(segment)
    if seg == "sensex":
        raw = (os.getenv(sensex_env) or "").strip()
        if raw:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                pass
    return max(0.0, _env_float(base_env, default)) * entry_pt_scale(segment, index_open)


def entry_min_day_move_pts(*, segment: str = "nifty50", index_open: float = 0.0) -> float:
    """Index must be at least this many pts from day open in trade direction."""
    return _segment_pts(
        "ENTRY_MIN_DAY_MOVE_PTS",
        "SENSEX_ENTRY_MIN_DAY_MOVE_PTS",
        30.0,
        segment=segment,
        index_open=index_open,
    )


def entry_chase_day_pts(*, segment: str = "nifty50", index_open: float = 0.0) -> float:
    """When day move exceeds this, require fresh index bar momentum or skip."""
    return _segment_pts(
        "ENTRY_CHASE_DAY_PTS",
        "SENSEX_ENTRY_CHASE_DAY_PTS",
        250.0,
        segment=segment,
        index_open=index_open,
    )


def entry_chase_max_bar_move_pts(*, segment: str = "nifty50", index_open: float = 0.0) -> float:
    """Max index pts/bar allowed when day is already extended (chase filter)."""
    return _segment_pts(
        "ENTRY_CHASE_MAX_BAR_MOVE_PTS",
        "SENSEX_ENTRY_CHASE_MAX_BAR_MOVE_PTS",
        10.0,
        segment=segment,
        index_open=index_open,
    )


def entry_chase_min_bar_move_pts(*, segment: str = "nifty50", index_open: float = 0.0) -> float:
    """On extended days, block violent capitulation bars (likely exhaustion bounce)."""
    return _segment_pts(
        "ENTRY_CHASE_MIN_BAR_MOVE_PTS",
        "SENSEX_ENTRY_CHASE_MIN_BAR_MOVE_PTS",
        25.0,
        segment=segment,
        index_open=index_open,
    )


def entry_pe_bounce_recovery_max_pct() -> float:
    """For PE: skip when spot has recovered too far from session low (shorting a bounce)."""
    return max(0.0, _env_float("ENTRY_PE_BOUNCE_RECOVERY_MAX_PCT", 50.0))


@dataclass(frozen=True)
class EntryIntradayThresholds:
    min_day_pts: float
    chase_day_pts: float
    chase_bar_pts: float
    chase_min_bar_pts: float
    bounce_recovery_max_pct: float


def entry_intraday_thresholds(segment: str, index_open: float) -> EntryIntradayThresholds:
    return EntryIntradayThresholds(
        min_day_pts=entry_min_day_move_pts(segment=segment, index_open=index_open),
        chase_day_pts=entry_chase_day_pts(segment=segment, index_open=index_open),
        chase_bar_pts=entry_chase_max_bar_move_pts(segment=segment, index_open=index_open),
        chase_min_bar_pts=entry_chase_min_bar_move_pts(segment=segment, index_open=index_open),
        bounce_recovery_max_pct=entry_pe_bounce_recovery_max_pct(),
    )


def entry_intraday_context_ok(
    *,
    kind: str,
    index_open: float,
    spot: float,
    spot_prev: float,
    bar_minutes: int,
    scan_start_minutes: int,
    session_low_so_far: float = 0.0,
    session_high_so_far: float = 0.0,
    segment: str = "nifty50",
) -> tuple[bool, str]:
    """Filters for marginal day trend, scan warmup, and late-day chase entries."""
    warmup = entry_scan_warmup_minutes()
    if warmup > 0 and bar_minutes < scan_start_minutes + warmup:
        return False, "scan_warmup"

    thr = entry_intraday_thresholds(segment, index_open)
    min_day = thr.min_day_pts
    if min_day > 0 and index_open > 0 and spot > 0:
        day_pts = spot - index_open
        k = (kind or "CE").upper()
        if k == "CE" and day_pts < min_day:
            return False, "weak_day_trend"
        if k == "PE" and day_pts > -min_day:
            return False, "weak_day_trend"

    chase_day = thr.chase_day_pts
    chase_bar = thr.chase_bar_pts
    if chase_day > 0 and chase_bar > 0 and index_open > 0 and spot > 0 and spot_prev > 0:
        day_pts = spot - index_open
        bar_move = spot - spot_prev
        k = (kind or "CE").upper()
        if k == "CE" and day_pts > chase_day and bar_move < chase_bar:
            return False, "chase_exhaustion"
        if k == "PE" and day_pts < -chase_day and bar_move > -chase_bar:
            return False, "chase_exhaustion"

    chase_min = thr.chase_min_bar_pts
    if chase_min > 0 and chase_day > 0 and index_open > 0 and spot > 0 and spot_prev > 0:
        day_pts = spot - index_open
        bar_move = spot - spot_prev
        k = (kind or "CE").upper()
        if k == "PE" and day_pts < -chase_day and bar_move < -chase_min:
            return False, "capitulation_bar"
        if k == "CE" and day_pts > chase_day and bar_move > chase_min:
            return False, "capitulation_bar"

    bounce_max = thr.bounce_recovery_max_pct
    if bounce_max > 0 and session_low_so_far > 0 and session_high_so_far > 0 and spot > 0:
        k = (kind or "CE").upper()
        if k == "PE":
            day_range = session_high_so_far - session_low_so_far
            if day_range > 0:
                recovery_pct = (spot - session_low_so_far) / day_range * 100.0
                if recovery_pct > bounce_max:
                    return False, "bounce_from_low"

    return True, ""


def bar_close_position(bar_low: float, bar_high: float, bar_close: float) -> float:
    """Fraction of bar range where close sits (0 = at low, 1 = at high)."""
    rng = max(float(bar_high) - float(bar_low), 0.01)
    return (float(bar_close) - float(bar_low)) / rng


def entry_bar_quality_ok(
    *,
    kind: str,
    bar_open: float,
    bar_close: float,
    spot: float,
    prev_close: float,
    spot_prev: float = 0.0,
    bar_high: float = 0.0,
    bar_low: float = 0.0,
) -> tuple[bool, str]:
    """Return (ok, reason) for a candidate entry bar."""
    k = (kind or "").upper()
    if entry_momentum_required() and bar_close > 0 and bar_open > 0:
        if bar_close <= bar_open:
            return False, "momentum"

    max_pos = entry_max_close_position_in_bar()
    if max_pos > 0 and bar_high > bar_low > 0 and bar_close > 0:
        if bar_close_position(bar_low, bar_high, bar_close) > max_pos:
            return False, "close_near_high"

    if entry_index_momentum_required() and spot > 0 and spot_prev > 0:
        if k == "CE" and spot <= spot_prev:
            return False, "index_momentum"
        if k == "PE" and spot >= spot_prev:
            return False, "index_momentum"

    if entry_direction_aligned_required() and prev_close > 0 and spot > 0:
        if k == "CE" and spot < prev_close:
            return False, "direction"
        if k == "PE" and spot >= prev_close:
            return False, "direction"

    return True, ""


def exit_t1_trigger_price(t1: float) -> float:
    """Live LTP threshold for T1 ratchet (optional buffer when close-confirm mode is on)."""
    t1 = float(t1)
    if t1 <= 0:
        return t1
    if not entry_t1_close_confirm():
        return t1
    try:
        buf = float(os.getenv("EXIT_T1_LTP_BUFFER_PCT", "0.01") or 0.01)
    except (TypeError, ValueError):
        buf = 0.01
    return round(t1 * (1.0 + max(0.0, buf)), 2)


def contract_may_reenter(contract: str, last_exit_reason: str | None) -> bool:
    if not last_exit_reason:
        return True
    reason = (last_exit_reason or "").lower()
    if block_reentry_after_loss() and reason == "stop_loss":
        return False
    if block_reentry_after_breakeven() and (
        reason in ("trail_t1", "breakeven") or reason.startswith("trail_t")
    ):
        return False
    return True
