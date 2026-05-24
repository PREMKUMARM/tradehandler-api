"""
NIFTY 50 — 15-minute Opening Range Breakout (ORB) signal, push-only.

Strategy (deterministic, rule-based):
  1. Build OR from the first 15 min of trading (09:15-09:30 IST).
  2. Skip the day if OR-range is outside [MIN, MAX] points (too noisy / too quiet).
  3. From OR-end until cutoff (default 13:30 IST), watch closing 5m candles.
     LONG  signal: first 5m close > OR-high.
     SHORT signal: first 5m close < OR-low.
  4. Once a signal fires, no more signals that day.
  5. Compose entry / SL / target on NIFTY spot using OR-range and the RR ratio.
  6. Recommend an ATM weekly option leg (CE for LONG, PE for SHORT) and estimate
     option premium move and ₹ P&L using a delta heuristic (default 0.5).
  7. Send an FCM push to every registered device. NO ORDER IS PLACED.

This module hooks into the existing Kite ticker via `register_tick_callback(...)`,
same pattern as `services/push/nifty_ticker_candle_alerts.py`. Tick callbacks run
in the KiteTicker background thread, so all mutable state is guarded by a lock
and the actual push send is scheduled onto the main asyncio loop.
"""
from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

from database.repositories import get_push_device_repository
from services.push.option_contract_resolver import (
    OptionLegEstimate,
    estimate_option_leg,
)
from services.push.push_service import push_service
from services.push.strategy_alert_logger import augment_payload_with_order, save_strategy_alert
from services.strategy_auto_trader import place_strategy_order
from utils.kite_utils import get_kite_instance
from utils.logger import log_error, log_info, log_warning


IST = ZoneInfo("Asia/Kolkata")


# ----------------------------- env helpers -----------------------------


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        x = int(os.getenv(name, str(default)))
        return max(lo, min(hi, x))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ----------------------------- config -----------------------------


@dataclass(frozen=True)
class OrbConfig:
    enabled: bool
    or_start: time
    or_end: time
    cutoff: time
    min_range_pts: float
    max_range_pts: float
    rr_ratio: float
    sl_buffer_pts: float
    atm_step: int
    atm_delta: float
    lot_size: int
    num_lots: int
    instrument_token: int
    instrument_symbol: str
    user_id_override: Optional[str]

    def or_start_dt(self, d: date) -> datetime:
        return datetime.combine(d, self.or_start, tzinfo=IST)

    def or_end_dt(self, d: date) -> datetime:
        return datetime.combine(d, self.or_end, tzinfo=IST)

    def cutoff_dt(self, d: date) -> datetime:
        return datetime.combine(d, self.cutoff, tzinfo=IST)


def _load_config() -> OrbConfig:
    return OrbConfig(
        enabled=_env_bool("NIFTY_ORB_SIGNAL_ENABLED", True),
        or_start=time(
            hour=_env_int("NIFTY_ORB_OR_START_HOUR", 9, 0, 23),
            minute=_env_int("NIFTY_ORB_OR_START_MINUTE", 15, 0, 59),
        ),
        or_end=time(
            hour=_env_int("NIFTY_ORB_OR_END_HOUR", 9, 0, 23),
            minute=_env_int("NIFTY_ORB_OR_END_MINUTE", 30, 0, 59),
        ),
        cutoff=time(
            hour=_env_int("NIFTY_ORB_CUTOFF_HOUR", 13, 0, 23),
            minute=_env_int("NIFTY_ORB_CUTOFF_MINUTE", 30, 0, 59),
        ),
        min_range_pts=_env_float("NIFTY_ORB_MIN_RANGE_POINTS", 25.0),
        max_range_pts=_env_float("NIFTY_ORB_MAX_RANGE_POINTS", 180.0),
        rr_ratio=max(0.5, _env_float("NIFTY_ORB_RR_RATIO", 1.5)),
        sl_buffer_pts=max(0.0, _env_float("NIFTY_ORB_SL_BUFFER_POINTS", 2.0)),
        atm_step=_env_int("NIFTY_ORB_ATM_STEP", 50, 5, 500),
        atm_delta=min(0.99, max(0.05, _env_float("NIFTY_ORB_ATM_DELTA", 0.5))),
        lot_size=_env_int("NIFTY_ORB_LOT_SIZE", 75, 1, 10000),
        num_lots=_env_int("NIFTY_ORB_NUM_LOTS", 1, 1, 1000),
        instrument_token=_env_int("NIFTY_ORB_INSTRUMENT_TOKEN", 256265, 1, 2_000_000_000),
        instrument_symbol=(
            os.getenv("NIFTY_ORB_INSTRUMENT_SYMBOL", "NSE:NIFTY 50").strip()
            or "NSE:NIFTY 50"
        ),
        user_id_override=(os.getenv("NIFTY_ORB_USER_ID") or "").strip() or None,
    )


# ----------------------------- formatting helpers -----------------------------


def _fmt_level(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.2f}"


def _fmt_inr(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}₹{abs(x):,.0f}"


def _floor_5m(dt: datetime) -> datetime:
    dt = dt.astimezone(IST)
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def _tick_time(t: Dict[str, Any]) -> Optional[datetime]:
    ts = t.get("exchange_timestamp") or t.get("last_trade_time")
    if hasattr(ts, "astimezone"):
        return ts.astimezone(IST)
    return None


# ----------------------------- core state -----------------------------


@dataclass
class Candle5m:
    start: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class DayState:
    day: Optional[date] = None
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    or_range: Optional[float] = None
    or_built: bool = False
    signal_fired: bool = False
    signal_direction: Optional[str] = None  # 'LONG' | 'SHORT'
    or_skip_reason: Optional[str] = None
    last_price: Optional[float] = None
    cur_candle: Optional[Candle5m] = None
    closed_candles_today: List[Candle5m] = field(default_factory=list)


@dataclass
class OrbSignal:
    direction: str  # 'LONG' | 'SHORT'
    entry: float
    stop_loss: float
    target: float
    risk_points: float
    reward_points: float
    rr_ratio: float
    or_high: float
    or_low: float
    or_range: float
    breakout_candle_start: datetime
    breakout_candle_close: float
    confidence: int
    atm_strike: int
    option_kind: str          # 'CE' | 'PE'
    option_leg: OptionLegEstimate
    lot_size: int
    num_lots: int


_state = DayState()
_state_lock = threading.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None
_registered = False


# ----------------------------- helpers -----------------------------


def _ensure_day_state(now_ist: datetime, cfg: OrbConfig) -> None:
    """Reset state at the start of a new trading day."""
    today = now_ist.astimezone(IST).date()
    if _state.day != today:
        _state.day = today
        _state.or_high = None
        _state.or_low = None
        _state.or_range = None
        _state.or_built = False
        _state.signal_fired = False
        _state.signal_direction = None
        _state.or_skip_reason = None
        _state.last_price = None
        _state.cur_candle = None
        _state.closed_candles_today = []


def _confidence_score(
    *,
    or_range: float,
    breakout_time_ist: datetime,
    breakout_candle: Candle5m,
    direction: str,
) -> int:
    """Heuristic 'AI confidence' 1..10."""
    score = 2  # base

    # OR-range sweet spot
    if 35 <= or_range <= 90:
        score += 3
    elif 25 <= or_range <= 150:
        score += 1

    # Time of breakout (earlier = stronger trend day potential)
    hh = breakout_time_ist.hour
    mm = breakout_time_ist.minute
    minutes = hh * 60 + mm
    if minutes <= 11 * 60:  # by 11:00
        score += 3
    elif minutes <= 12 * 60 + 30:  # by 12:30
        score += 2
    else:
        score += 1

    # Body strength of the breakout candle (close - open) vs range
    rng = max(1e-6, breakout_candle.high - breakout_candle.low)
    body = abs(breakout_candle.close - breakout_candle.open)
    body_ratio = body / rng
    if body_ratio >= 0.6:
        score += 2
    elif body_ratio >= 0.35:
        score += 1

    # Direction-consistent close (close on the right side of mid)
    mid = (breakout_candle.high + breakout_candle.low) / 2.0
    if direction == "LONG" and breakout_candle.close >= mid:
        score += 1
    elif direction == "SHORT" and breakout_candle.close <= mid:
        score += 1

    return max(1, min(10, int(score)))


def _round_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)


def _build_signal(
    *,
    cfg: OrbConfig,
    direction: str,
    or_high: float,
    or_low: float,
    breakout_candle: Candle5m,
) -> OrbSignal:
    entry = breakout_candle.close
    or_range = or_high - or_low

    if direction == "LONG":
        stop_loss = or_low - cfg.sl_buffer_pts
        risk_pts = max(1e-6, entry - stop_loss)
        target = entry + cfg.rr_ratio * risk_pts
        option_kind = "CE"
    else:
        stop_loss = or_high + cfg.sl_buffer_pts
        risk_pts = max(1e-6, stop_loss - entry)
        target = entry - cfg.rr_ratio * risk_pts
        option_kind = "PE"

    reward_pts = abs(target - entry)

    atm_strike = _round_to_step(entry, cfg.atm_step)

    # Resolve the actual ATM CE/PE for this signal and project its SL/Target premium.
    option_leg = estimate_option_leg(
        spot_entry=entry,
        spot_stop_loss=stop_loss,
        spot_target=target,
        kind=option_kind,
        delta=cfg.atm_delta,
        lot_size=cfg.lot_size,
        num_lots=cfg.num_lots,
        atm_step=cfg.atm_step,
        name="NIFTY",
    )

    confidence = _confidence_score(
        or_range=or_range,
        breakout_time_ist=breakout_candle.start.astimezone(IST),
        breakout_candle=breakout_candle,
        direction=direction,
    )

    return OrbSignal(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        target=target,
        risk_points=risk_pts,
        reward_points=reward_pts,
        rr_ratio=cfg.rr_ratio,
        or_high=or_high,
        or_low=or_low,
        or_range=or_range,
        breakout_candle_start=breakout_candle.start,
        breakout_candle_close=breakout_candle.close,
        confidence=confidence,
        atm_strike=atm_strike,
        option_kind=option_kind,
        option_leg=option_leg,
        lot_size=cfg.lot_size,
        num_lots=cfg.num_lots,
    )


def _compose_push(sig: OrbSignal) -> Dict[str, Any]:
    arrow = "▲" if sig.direction == "LONG" else "▼"
    title = (
        f"NIFTY 15m ORB {sig.direction} {arrow} · "
        f"conf {sig.confidence}/10 · R:R 1:{sig.rr_ratio:g}"
    )
    leg = sig.option_leg
    qty = sig.lot_size * sig.num_lots
    option_label = (
        leg.contract.tradingsymbol
        if leg.contract
        else f"NIFTY {sig.atm_strike} {sig.option_kind} (ATM)"
    )
    est_tag = " (est.)" if leg.estimated else ""
    risk_inr = leg.risk_inr if leg.risk_inr is not None else 0.0
    reward_inr = leg.reward_inr if leg.reward_inr is not None else 0.0

    lines = [
        f"Spot: Entry {_fmt_level(sig.entry)} · SL {_fmt_level(sig.stop_loss)} · Tgt {_fmt_level(sig.target)}",
        f"Spot risk {sig.risk_points:.0f} pts | reward {sig.reward_points:.0f} pts",
        f"OR {_fmt_level(sig.or_low)}–{_fmt_level(sig.or_high)} ({sig.or_range:.0f} pts)",
        f"Option leg (not placed): {option_label}",
        (
            f"Premium{est_tag}: Entry ₹{leg.entry_premium:.1f} · "
            f"SL ₹{leg.sl_premium:.1f} · Tgt ₹{leg.target_premium:.1f}"
        ),
        (
            f"{sig.num_lots} lot × {sig.lot_size} = {qty} qty → "
            f"Risk {_fmt_inr(risk_inr)} | Reward {_fmt_inr(reward_inr)}"
        ),
        "Signal only — no order placed.",
    ]
    body = "\n".join(lines)

    data: Dict[str, str] = {
        "type": "nifty_orb_signal",
        "source": "vibefno_backend",
        "strategy": "nifty_15m_orb",
        "direction": sig.direction,
        "confidence": str(sig.confidence),
        "rr_ratio": f"{sig.rr_ratio:.2f}",
        "entry": f"{sig.entry:.2f}",
        "stop_loss": f"{sig.stop_loss:.2f}",
        "target": f"{sig.target:.2f}",
        "risk_points": f"{sig.risk_points:.2f}",
        "reward_points": f"{sig.reward_points:.2f}",
        "or_high": f"{sig.or_high:.2f}",
        "or_low": f"{sig.or_low:.2f}",
        "or_range": f"{sig.or_range:.2f}",
        "breakout_close": f"{sig.breakout_candle_close:.2f}",
        "breakout_candle_start": sig.breakout_candle_start.astimezone(IST).isoformat(),
        "atm_strike": str(sig.atm_strike),
        "option_kind": sig.option_kind,
        "lot_size": str(sig.lot_size),
        "num_lots": str(sig.num_lots),
    }
    data.update(leg.to_payload())
    return {"title": title, "body": body, "data": data, "signal": sig}


# ----------------------------- tick processing -----------------------------


def _process_closed_candle(closed: Candle5m, cfg: OrbConfig) -> Optional[Dict[str, Any]]:
    """
    Called whenever a 5m candle has just *closed*. Updates OR if we're still in
    OR-window, otherwise checks for breakout against an already-built OR.
    Returns a composed-push dict on signal, else None.
    """
    cs = closed.start.astimezone(IST)
    today = cs.date()
    or_start = cfg.or_start_dt(today)
    or_end = cfg.or_end_dt(today)
    cutoff = cfg.cutoff_dt(today)

    # During OR window (candles whose start time is in [or_start, or_end))
    if or_start <= cs < or_end:
        _state.closed_candles_today.append(closed)
        return None

    # First call AFTER the OR window — build OR if not yet built
    if not _state.or_built and cs >= or_end:
        bars = [c for c in _state.closed_candles_today if or_start <= c.start.astimezone(IST) < or_end]
        if bars:
            _state.or_high = max(b.high for b in bars)
            _state.or_low = min(b.low for b in bars)
            _state.or_range = _state.or_high - _state.or_low
            _state.or_built = True
            if _state.or_range < cfg.min_range_pts:
                _state.or_skip_reason = (
                    f"OR range {_state.or_range:.1f} < min {cfg.min_range_pts:.1f} — skipping day"
                )
                log_info(f"[ORB] {_state.or_skip_reason}")
            elif _state.or_range > cfg.max_range_pts:
                _state.or_skip_reason = (
                    f"OR range {_state.or_range:.1f} > max {cfg.max_range_pts:.1f} — skipping day"
                )
                log_info(f"[ORB] {_state.or_skip_reason}")
            else:
                log_info(
                    f"[ORB] OR built for {today}: high={_state.or_high:.2f} "
                    f"low={_state.or_low:.2f} range={_state.or_range:.2f}"
                )

    # If OR not built or skipped, no further action this candle
    if not _state.or_built or _state.or_skip_reason or _state.signal_fired:
        return None

    # Past cutoff?
    if cs >= cutoff:
        return None

    # Look for a breakout on this just-closed candle
    direction: Optional[str] = None
    if _state.or_high is not None and closed.close > _state.or_high:
        direction = "LONG"
    elif _state.or_low is not None and closed.close < _state.or_low:
        direction = "SHORT"

    if direction is None:
        return None

    sig = _build_signal(
        cfg=cfg,
        direction=direction,
        or_high=float(_state.or_high or 0.0),
        or_low=float(_state.or_low or 0.0),
        breakout_candle=closed,
    )
    _state.signal_fired = True
    _state.signal_direction = direction
    return _compose_push(sig)


def _update_candle(tick: Dict[str, Any], cfg: OrbConfig) -> Optional[Candle5m]:
    """Aggregate ticks into 5m candles. Returns the *just-closed* candle if any."""
    ts = _tick_time(tick)
    lp = tick.get("last_price")
    if ts is None or lp is None:
        return None
    try:
        price = float(lp)
    except (TypeError, ValueError):
        return None

    _state.last_price = price

    start = _floor_5m(ts)
    if _state.cur_candle is None:
        _state.cur_candle = Candle5m(start=start, open=price, high=price, low=price, close=price)
        return None

    if start == _state.cur_candle.start:
        c = _state.cur_candle
        c.high = max(c.high, price)
        c.low = min(c.low, price)
        c.close = price
        return None

    closed = _state.cur_candle
    _state.cur_candle = Candle5m(start=start, open=price, high=price, low=price, close=price)
    return closed


def _on_ticks_batch(ticks: List[Dict]) -> None:
    """
    Runs in the KiteTicker thread — must not block. We update state under a lock,
    then schedule the push send onto the main asyncio loop.
    """
    if _loop is None:
        return

    cfg = _load_config()
    if not cfg.enabled:
        return

    pushes: List[Dict[str, Any]] = []
    with _state_lock:
        now_ist = datetime.now(IST)
        _ensure_day_state(now_ist, cfg)

        for t in ticks:
            try:
                if int(t.get("instrument_token") or 0) != cfg.instrument_token:
                    continue
            except (TypeError, ValueError):
                continue

            closed = _update_candle(t, cfg)
            if closed is None:
                continue
            push = _process_closed_candle(closed, cfg)
            if push is not None:
                pushes.append(push)

    for p in pushes:
        try:
            fut = asyncio.run_coroutine_threadsafe(_dispatch_push(p), _loop)
            fut.add_done_callback(lambda f: f.exception())
        except Exception as e:  # noqa: BLE001
            log_error(f"[ORB] failed to dispatch push: {e}")


# ----------------------------- push dispatch -----------------------------


async def _dispatch_push(p: Dict[str, Any], *, is_test: bool = False) -> Dict[str, Any]:
    # Place the auto-trade order first (if enabled) so the push body and the
    # persisted audit row can both include the broker order id.
    order_result = await place_strategy_order(p["data"], is_test=is_test)
    p = augment_payload_with_order(p, order_result)

    if not push_service.configured():
        log_warning("[ORB] FCM not configured; skipping send.")
        result = {"sent": 0, "failed": 0, "reason": "fcm_not_configured", "order": order_result}
        alert_id = save_strategy_alert(p, result, is_test=is_test, order_result=order_result)
        if alert_id is not None:
            result["alert_id"] = alert_id
        return result

    cfg = _load_config()
    if cfg.user_id_override:
        user_ids = [cfg.user_id_override]
    else:
        repo = get_push_device_repository()
        user_ids = repo.list_distinct_user_ids() or []

    if not user_ids:
        log_info("[ORB] No registered devices; nothing to send.")
        result = {
            "sent": 0,
            "failed": 0,
            "reason": "no_devices",
            "payload": p["data"],
            "order": order_result,
        }
        alert_id = save_strategy_alert(p, result, is_test=is_test, order_result=order_result)
        if alert_id is not None:
            result["alert_id"] = alert_id
        return result

    title = p["title"]
    body = p["body"]
    data = p["data"]

    totals: Dict[str, Any] = {
        "sent": 0,
        "failed": 0,
        "per_user": {},
        "payload": data,
        "order": order_result,
    }
    for uid in user_ids:
        try:
            r = await push_service.send_to_user(user_id=uid, title=title, body=body, data=data)
            totals["per_user"][uid] = r
            totals["sent"] += int(r.get("sent", 0))
            totals["failed"] += int(r.get("failed", 0))
        except Exception as e:  # noqa: BLE001
            log_error(f"[ORB] send failed for user_id={uid!r}: {e}")
            totals["failed"] += 1
            totals["per_user"][uid] = {"sent": 0, "failed": 1, "errors": [str(e)]}

    log_info(
        f"[ORB] dispatched: users={len(user_ids)} sent={totals['sent']} "
        f"failed={totals['failed']} title={title!r}"
    )
    alert_id = save_strategy_alert(p, totals, is_test=is_test, order_result=order_result)
    if alert_id is not None:
        totals["alert_id"] = alert_id
    return totals


# ----------------------------- public API -----------------------------


def register_nifty_orb_signal(loop: asyncio.AbstractEventLoop) -> None:
    """Register the tick callback into the existing Kite ticker system."""
    global _registered, _loop
    if _registered:
        return
    cfg = _load_config()
    if not cfg.enabled:
        log_info("[ORB] disabled via NIFTY_ORB_SIGNAL_ENABLED=0")
        _registered = True
        return

    _loop = loop
    from utils.kite_websocket_ticker import register_tick_callback

    register_tick_callback(_on_ticks_batch)
    _registered = True
    log_info(
        f"[ORB] registered tick handler for token={cfg.instrument_token} "
        f"OR={cfg.or_start.isoformat()}–{cfg.or_end.isoformat()} cutoff={cfg.cutoff.isoformat()}"
    )


def get_state_snapshot() -> Dict[str, Any]:
    """JSON-friendly snapshot for the /info endpoint and UI."""
    cfg = _load_config()
    with _state_lock:
        s = _state
        return {
            "enabled": cfg.enabled,
            "config": {
                "or_start": cfg.or_start.isoformat(),
                "or_end": cfg.or_end.isoformat(),
                "cutoff": cfg.cutoff.isoformat(),
                "min_range_pts": cfg.min_range_pts,
                "max_range_pts": cfg.max_range_pts,
                "rr_ratio": cfg.rr_ratio,
                "sl_buffer_pts": cfg.sl_buffer_pts,
                "atm_step": cfg.atm_step,
                "atm_delta": cfg.atm_delta,
                "lot_size": cfg.lot_size,
                "num_lots": cfg.num_lots,
                "instrument_token": cfg.instrument_token,
                "instrument_symbol": cfg.instrument_symbol,
            },
            "day": s.day.isoformat() if s.day else None,
            "or_built": s.or_built,
            "or_high": s.or_high,
            "or_low": s.or_low,
            "or_range": s.or_range,
            "or_skip_reason": s.or_skip_reason,
            "signal_fired": s.signal_fired,
            "signal_direction": s.signal_direction,
            "last_price": s.last_price,
            "closed_candles_today": len(s.closed_candles_today),
            "fcm_configured": push_service.configured(),
        }


def _fetch_current_spot() -> Optional[float]:
    """Best-effort spot price via kite.quote() — for preview/force-test outside ticks."""
    cfg = _load_config()
    try:
        kite = get_kite_instance()
        quotes = kite.quote(cfg.instrument_symbol) or {}
        q = quotes.get(cfg.instrument_symbol) or {}
        lp = q.get("last_price")
        if lp is None:
            ohlc = q.get("ohlc") or {}
            lp = ohlc.get("open") or ohlc.get("close")
        return float(lp) if lp is not None else None
    except Exception as e:  # noqa: BLE001
        log_warning(f"[ORB] _fetch_current_spot failed: {e}")
        return None


def preview_signal(*, direction_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a *would-be* signal payload using current state for inspection only.
    Useful from the UI / API to verify formatting without waiting for a breakout.
    Direction: 'LONG' (default) or 'SHORT'.
    """
    cfg = _load_config()

    with _state_lock:
        or_high = _state.or_high
        or_low = _state.or_low
        last_price = _state.last_price

    if last_price is None:
        last_price = _fetch_current_spot()
    if last_price is None:
        return {"available": False, "reason": "spot price unavailable"}

    if or_high is None or or_low is None:
        # Synthesize a plausible OR around current spot for the preview
        synth_range = max(cfg.min_range_pts + 10.0, min(60.0, cfg.max_range_pts - 10.0))
        or_high = last_price + synth_range / 2.0
        or_low = last_price - synth_range / 2.0
        synthesized = True
    else:
        synthesized = False

    direction = (direction_override or "LONG").upper()
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"

    # Place breakout 5 pts beyond OR in the direction asked
    if direction == "LONG":
        close = max(last_price, or_high + 5.0)
    else:
        close = min(last_price, or_low - 5.0)
    open_ = close - 4.0 if direction == "LONG" else close + 4.0
    high = max(open_, close) + 2.0
    low = min(open_, close) - 2.0
    breakout_candle = Candle5m(
        start=_floor_5m(datetime.now(IST)),
        open=open_,
        high=high,
        low=low,
        close=close,
    )

    sig = _build_signal(
        cfg=cfg,
        direction=direction,
        or_high=or_high,
        or_low=or_low,
        breakout_candle=breakout_candle,
    )
    p = _compose_push(sig)
    return {
        "available": True,
        "synthesized_or": synthesized,
        "title": p["title"],
        "body": p["body"],
        "payload": p["data"],
    }


async def force_test_send(*, direction: str = "LONG") -> Dict[str, Any]:
    """
    Compose a signal using `preview_signal()` and dispatch it as a real push.
    """
    if not push_service.configured():
        return {"sent": 0, "failed": 0, "reason": "fcm_not_configured"}

    prev = preview_signal(direction_override=direction)
    if not prev.get("available"):
        return {"sent": 0, "failed": 0, "reason": prev.get("reason", "preview_unavailable")}

    return await _dispatch_push(
        {"title": prev["title"], "body": prev["body"], "data": prev["payload"]},
        is_test=True,
    )


def reset_day_state() -> Dict[str, Any]:
    """Manually reset today's state (useful for re-testing)."""
    with _state_lock:
        _state.day = None
        _state.or_high = None
        _state.or_low = None
        _state.or_range = None
        _state.or_built = False
        _state.signal_fired = False
        _state.signal_direction = None
        _state.or_skip_reason = None
        _state.last_price = None
        _state.cur_candle = None
        _state.closed_candles_today = []
    log_info("[ORB] day state reset")
    return {"ok": True}
