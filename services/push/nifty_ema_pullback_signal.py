"""
NIFTY 50 — 9-EMA Pullback Momentum (5m) push-only signal.

Strategy concept (trend-continuation):
  * Compute a 9-period EMA on 5-minute closes.
  * Track 'bullish' regime (last 3 closes above EMA) and 'bearish' regime
    (last 3 closes below EMA).
  * After a regime is established, watch for a "pullback bar" whose low
    (in bull regime) or high (in bear regime) touches/penetrates the EMA.
  * The next 5m candle that closes back beyond the pullback bar's range in
    the trend direction is the trigger:
       LONG : close > pullback.high  (in bull regime, after pullback touched EMA)
       SHORT: close < pullback.low   (in bear regime, after pullback touched EMA)
  * SL: pullback bar's low (LONG) / high (SHORT) ± buffer.
  * Target: RR × |entry - SL|  (default 1.5).
  * Cap: at most NIFTY_EMA_MAX_SIGNALS_PER_DAY signals/day per direction.
  * Avoid trading before warm-up window (default 09:30 IST) and after cutoff
    (default 14:30 IST).

The signal includes a real ATM CE/PE contract + live premium so the push
shows the actual option entry/SL/target ₹ values. No order is placed.
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
from services.push.strategy_alert_logger import save_strategy_alert
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
class EmaConfig:
    enabled: bool
    ema_period: int
    warmup: time
    cutoff: time
    rr_ratio: float
    sl_buffer_pts: float
    max_signals_per_day: int
    atm_step: int
    atm_delta: float
    lot_size: int
    num_lots: int
    instrument_token: int
    instrument_symbol: str
    user_id_override: Optional[str]


def _load_config() -> EmaConfig:
    return EmaConfig(
        enabled=_env_bool("NIFTY_EMA_SIGNAL_ENABLED", True),
        ema_period=_env_int("NIFTY_EMA_PERIOD", 9, 3, 50),
        warmup=time(
            hour=_env_int("NIFTY_EMA_WARMUP_HOUR", 9, 0, 23),
            minute=_env_int("NIFTY_EMA_WARMUP_MINUTE", 30, 0, 59),
        ),
        cutoff=time(
            hour=_env_int("NIFTY_EMA_CUTOFF_HOUR", 14, 0, 23),
            minute=_env_int("NIFTY_EMA_CUTOFF_MINUTE", 30, 0, 59),
        ),
        rr_ratio=max(0.5, _env_float("NIFTY_EMA_RR_RATIO", 1.5)),
        sl_buffer_pts=max(0.0, _env_float("NIFTY_EMA_SL_BUFFER_POINTS", 2.0)),
        max_signals_per_day=_env_int("NIFTY_EMA_MAX_SIGNALS_PER_DAY", 2, 1, 10),
        atm_step=_env_int("NIFTY_EMA_ATM_STEP", 50, 5, 500),
        atm_delta=min(0.99, max(0.05, _env_float("NIFTY_EMA_ATM_DELTA", 0.5))),
        lot_size=_env_int("NIFTY_EMA_LOT_SIZE", 75, 1, 10000),
        num_lots=_env_int("NIFTY_EMA_NUM_LOTS", 1, 1, 1000),
        instrument_token=_env_int(
            "NIFTY_EMA_INSTRUMENT_TOKEN", 256265, 1, 2_000_000_000
        ),
        instrument_symbol=(
            os.getenv("NIFTY_EMA_INSTRUMENT_SYMBOL", "NSE:NIFTY 50").strip()
            or "NSE:NIFTY 50"
        ),
        user_id_override=(os.getenv("NIFTY_EMA_USER_ID") or "").strip() or None,
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
    cur_candle: Optional[Candle5m] = None
    closed_candles: List[Candle5m] = field(default_factory=list)
    ema: Optional[float] = None
    last_regime: Optional[str] = None    # 'bull' | 'bear' | None
    pending_pullback: Optional[Candle5m] = None  # latest pullback bar awaiting trigger
    pullback_direction: Optional[str] = None     # 'LONG' | 'SHORT' (the direction trigger we need)
    signals_today: int = 0
    last_signal_direction: Optional[str] = None
    last_price: Optional[float] = None


@dataclass
class EmaSignal:
    direction: str
    entry: float
    stop_loss: float
    target: float
    risk_points: float
    reward_points: float
    rr_ratio: float
    ema_value: float
    pullback_bar_start: datetime
    pullback_low: float
    pullback_high: float
    breakout_candle_start: datetime
    breakout_candle_close: float
    atm_strike: int
    option_kind: str
    option_leg: OptionLegEstimate
    lot_size: int
    num_lots: int


_state = DayState()
_state_lock = threading.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None
_registered = False


# ----------------------------- helpers -----------------------------


def _ensure_day_state(now_ist: datetime) -> None:
    today = now_ist.astimezone(IST).date()
    if _state.day != today:
        _state.day = today
        _state.cur_candle = None
        _state.closed_candles = []
        _state.ema = None
        _state.last_regime = None
        _state.pending_pullback = None
        _state.pullback_direction = None
        _state.signals_today = 0
        _state.last_signal_direction = None
        _state.last_price = None


def _update_ema(prev: Optional[float], price: float, period: int) -> float:
    """Standard EMA: EMA_t = price * k + EMA_{t-1} * (1-k); seed with first price."""
    if prev is None:
        return price
    k = 2.0 / (period + 1.0)
    return price * k + prev * (1.0 - k)


def _regime_from_closes(closes: List[float], ema_values: List[float]) -> Optional[str]:
    """
    Determine current regime: 'bull' if the last 3 closes are all above their
    EMA at that bar, 'bear' if all below, else None.
    """
    if len(closes) < 3 or len(ema_values) < 3:
        return None
    last3 = list(zip(closes[-3:], ema_values[-3:]))
    if all(c > e for c, e in last3):
        return "bull"
    if all(c < e for c, e in last3):
        return "bear"
    return None


def _round_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)


def _build_signal(
    *,
    cfg: EmaConfig,
    direction: str,
    pullback_bar: Candle5m,
    breakout_candle: Candle5m,
    ema_value: float,
) -> EmaSignal:
    entry = breakout_candle.close
    if direction == "LONG":
        stop_loss = pullback_bar.low - cfg.sl_buffer_pts
        risk_pts = max(1e-6, entry - stop_loss)
        target = entry + cfg.rr_ratio * risk_pts
        option_kind = "CE"
    else:
        stop_loss = pullback_bar.high + cfg.sl_buffer_pts
        risk_pts = max(1e-6, stop_loss - entry)
        target = entry - cfg.rr_ratio * risk_pts
        option_kind = "PE"

    reward_pts = abs(target - entry)
    atm_strike = _round_to_step(entry, cfg.atm_step)

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

    return EmaSignal(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        target=target,
        risk_points=risk_pts,
        reward_points=reward_pts,
        rr_ratio=cfg.rr_ratio,
        ema_value=ema_value,
        pullback_bar_start=pullback_bar.start,
        pullback_low=pullback_bar.low,
        pullback_high=pullback_bar.high,
        breakout_candle_start=breakout_candle.start,
        breakout_candle_close=breakout_candle.close,
        atm_strike=atm_strike,
        option_kind=option_kind,
        option_leg=option_leg,
        lot_size=cfg.lot_size,
        num_lots=cfg.num_lots,
    )


def _compose_push(sig: EmaSignal) -> Dict[str, Any]:
    arrow = "▲" if sig.direction == "LONG" else "▼"
    title = (
        f"NIFTY 9-EMA Pullback {sig.direction} {arrow} · "
        f"R:R 1:{sig.rr_ratio:g}"
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
        f"9-EMA at {sig.ema_value:.0f} (pullback tested it)",
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
        "type": "nifty_ema_signal",
        "source": "algofeast_backend",
        "strategy": "nifty_9ema_pullback",
        "direction": sig.direction,
        "rr_ratio": f"{sig.rr_ratio:.2f}",
        "entry": f"{sig.entry:.2f}",
        "stop_loss": f"{sig.stop_loss:.2f}",
        "target": f"{sig.target:.2f}",
        "risk_points": f"{sig.risk_points:.2f}",
        "reward_points": f"{sig.reward_points:.2f}",
        "ema_value": f"{sig.ema_value:.2f}",
        "pullback_low": f"{sig.pullback_low:.2f}",
        "pullback_high": f"{sig.pullback_high:.2f}",
        "breakout_close": f"{sig.breakout_candle_close:.2f}",
        "atm_strike": str(sig.atm_strike),
        "option_kind": sig.option_kind,
        "lot_size": str(sig.lot_size),
        "num_lots": str(sig.num_lots),
    }
    data.update(leg.to_payload())
    return {"title": title, "body": body, "data": data, "signal": sig}


# ----------------------------- tick processing -----------------------------


def _process_closed_candle(
    closed: Candle5m, cfg: EmaConfig
) -> Optional[Dict[str, Any]]:
    """Update EMA + regime, detect pullback + trigger. Returns push payload or None."""
    cs = closed.start.astimezone(IST)
    today = cs.date()
    warmup = datetime.combine(today, cfg.warmup, tzinfo=IST)
    cutoff = datetime.combine(today, cfg.cutoff, tzinfo=IST)

    # Maintain EMA from this candle's close
    _state.ema = _update_ema(_state.ema, closed.close, cfg.ema_period)
    # Track closed candles with their EMA at the time of close for regime detection
    _state.closed_candles.append(closed)

    # Limit memory
    if len(_state.closed_candles) > 50:
        _state.closed_candles = _state.closed_candles[-50:]

    # Determine regime using last 3 closed candles + EMAs at those points.
    # Since EMA was rolling, we approximate by recomputing EMA at the last 3 bars.
    if len(_state.closed_candles) >= cfg.ema_period:
        # Recompute EMA series from a window (cheap; tens of bars)
        closes = [c.close for c in _state.closed_candles[-(cfg.ema_period + 5):]]
        ema_series: List[float] = []
        e: Optional[float] = None
        for p in closes:
            e = _update_ema(e, p, cfg.ema_period)
            ema_series.append(float(e))
        regime = _regime_from_closes(closes, ema_series)
    else:
        regime = None

    _state.last_regime = regime

    if cs < warmup or cs >= cutoff:
        return None
    if _state.signals_today >= cfg.max_signals_per_day:
        return None
    if regime is None:
        return None

    # Pullback detection — bar touches/penetrates the EMA against the trend
    e_now = float(_state.ema or 0.0)

    if regime == "bull":
        touches_ema = closed.low <= e_now
        if touches_ema:
            _state.pending_pullback = closed
            _state.pullback_direction = "LONG"
            return None
        if _state.pullback_direction == "LONG" and _state.pending_pullback is not None:
            # Trigger if this candle closes above pullback high
            if closed.close > _state.pending_pullback.high:
                sig = _build_signal(
                    cfg=cfg,
                    direction="LONG",
                    pullback_bar=_state.pending_pullback,
                    breakout_candle=closed,
                    ema_value=e_now,
                )
                _state.signals_today += 1
                _state.last_signal_direction = "LONG"
                _state.pending_pullback = None
                _state.pullback_direction = None
                return _compose_push(sig)

    elif regime == "bear":
        touches_ema = closed.high >= e_now
        if touches_ema:
            _state.pending_pullback = closed
            _state.pullback_direction = "SHORT"
            return None
        if _state.pullback_direction == "SHORT" and _state.pending_pullback is not None:
            if closed.close < _state.pending_pullback.low:
                sig = _build_signal(
                    cfg=cfg,
                    direction="SHORT",
                    pullback_bar=_state.pending_pullback,
                    breakout_candle=closed,
                    ema_value=e_now,
                )
                _state.signals_today += 1
                _state.last_signal_direction = "SHORT"
                _state.pending_pullback = None
                _state.pullback_direction = None
                return _compose_push(sig)

    return None


def _update_candle(tick: Dict[str, Any], cfg: EmaConfig) -> Optional[Candle5m]:
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
    if _loop is None:
        return
    cfg = _load_config()
    if not cfg.enabled:
        return

    pushes: List[Dict[str, Any]] = []
    with _state_lock:
        now_ist = datetime.now(IST)
        _ensure_day_state(now_ist)
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
            log_error(f"[EMA] failed to dispatch push: {e}")


# ----------------------------- push dispatch -----------------------------


async def _dispatch_push(p: Dict[str, Any], *, is_test: bool = False) -> Dict[str, Any]:
    if not push_service.configured():
        log_warning("[EMA] FCM not configured; skipping send.")
        result = {"sent": 0, "failed": 0, "reason": "fcm_not_configured"}
        alert_id = save_strategy_alert(p, result, is_test=is_test)
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
        log_info("[EMA] No registered devices.")
        result = {"sent": 0, "failed": 0, "reason": "no_devices", "payload": p["data"]}
        alert_id = save_strategy_alert(p, result, is_test=is_test)
        if alert_id is not None:
            result["alert_id"] = alert_id
        return result

    totals: Dict[str, Any] = {"sent": 0, "failed": 0, "per_user": {}, "payload": p["data"]}
    for uid in user_ids:
        try:
            r = await push_service.send_to_user(
                user_id=uid, title=p["title"], body=p["body"], data=p["data"]
            )
            totals["per_user"][uid] = r
            totals["sent"] += int(r.get("sent", 0))
            totals["failed"] += int(r.get("failed", 0))
        except Exception as e:  # noqa: BLE001
            log_error(f"[EMA] send failed for user_id={uid!r}: {e}")
            totals["failed"] += 1
            totals["per_user"][uid] = {"sent": 0, "failed": 1, "errors": [str(e)]}
    log_info(
        f"[EMA] dispatched: users={len(user_ids)} sent={totals['sent']} "
        f"failed={totals['failed']} title={p['title']!r}"
    )
    alert_id = save_strategy_alert(p, totals, is_test=is_test)
    if alert_id is not None:
        totals["alert_id"] = alert_id
    return totals


# ----------------------------- public API -----------------------------


def register_nifty_ema_signal(loop: asyncio.AbstractEventLoop) -> None:
    global _registered, _loop
    if _registered:
        return
    cfg = _load_config()
    if not cfg.enabled:
        log_info("[EMA] disabled via NIFTY_EMA_SIGNAL_ENABLED=0")
        _registered = True
        return
    _loop = loop
    from utils.kite_websocket_ticker import register_tick_callback

    register_tick_callback(_on_ticks_batch)
    _registered = True
    log_info(
        f"[EMA] registered tick handler for token={cfg.instrument_token} "
        f"ema={cfg.ema_period} warmup={cfg.warmup.isoformat()} cutoff={cfg.cutoff.isoformat()}"
    )


def get_state_snapshot() -> Dict[str, Any]:
    cfg = _load_config()
    with _state_lock:
        s = _state
        return {
            "enabled": cfg.enabled,
            "config": {
                "ema_period": cfg.ema_period,
                "warmup": cfg.warmup.isoformat(),
                "cutoff": cfg.cutoff.isoformat(),
                "rr_ratio": cfg.rr_ratio,
                "sl_buffer_pts": cfg.sl_buffer_pts,
                "max_signals_per_day": cfg.max_signals_per_day,
                "atm_step": cfg.atm_step,
                "atm_delta": cfg.atm_delta,
                "lot_size": cfg.lot_size,
                "num_lots": cfg.num_lots,
                "instrument_token": cfg.instrument_token,
                "instrument_symbol": cfg.instrument_symbol,
            },
            "day": s.day.isoformat() if s.day else None,
            "ema": s.ema,
            "last_regime": s.last_regime,
            "pending_pullback": (
                {
                    "start": s.pending_pullback.start.astimezone(IST).isoformat(),
                    "high": s.pending_pullback.high,
                    "low": s.pending_pullback.low,
                    "direction": s.pullback_direction,
                }
                if s.pending_pullback
                else None
            ),
            "signals_today": s.signals_today,
            "last_signal_direction": s.last_signal_direction,
            "last_price": s.last_price,
            "closed_candles_today": len(s.closed_candles),
            "fcm_configured": push_service.configured(),
        }


def _fetch_current_spot() -> Optional[float]:
    cfg = _load_config()
    try:
        kite = get_kite_instance()
        q = kite.quote(cfg.instrument_symbol) or {}
        row = q.get(cfg.instrument_symbol) or {}
        lp = row.get("last_price")
        if lp is None:
            ohlc = row.get("ohlc") or {}
            lp = ohlc.get("open") or ohlc.get("close")
        return float(lp) if lp is not None else None
    except Exception as e:  # noqa: BLE001
        log_warning(f"[EMA] _fetch_current_spot failed: {e}")
        return None


def preview_signal(*, direction_override: Optional[str] = None) -> Dict[str, Any]:
    cfg = _load_config()
    with _state_lock:
        last_price = _state.last_price
        ema_now = _state.ema
    if last_price is None:
        last_price = _fetch_current_spot()
    if last_price is None:
        return {"available": False, "reason": "spot price unavailable"}

    direction = (direction_override or "LONG").upper()
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"

    # Synthesize a pullback bar that touched the EMA + a triggering breakout bar.
    ema_value = float(ema_now if ema_now is not None else last_price)
    if direction == "LONG":
        pullback = Candle5m(
            start=_floor_5m(datetime.now(IST)) - timedelta(minutes=5),
            open=ema_value + 3.0,
            high=ema_value + 4.0,
            low=ema_value - 1.0,  # touched EMA
            close=ema_value + 2.0,
        )
        close = max(last_price, pullback.high + 3.0)
        breakout = Candle5m(
            start=_floor_5m(datetime.now(IST)),
            open=pullback.high + 1.0,
            high=close + 2.0,
            low=pullback.high - 0.5,
            close=close,
        )
    else:
        pullback = Candle5m(
            start=_floor_5m(datetime.now(IST)) - timedelta(minutes=5),
            open=ema_value - 3.0,
            high=ema_value + 1.0,  # touched EMA
            low=ema_value - 4.0,
            close=ema_value - 2.0,
        )
        close = min(last_price, pullback.low - 3.0)
        breakout = Candle5m(
            start=_floor_5m(datetime.now(IST)),
            open=pullback.low - 1.0,
            high=pullback.low + 0.5,
            low=close - 2.0,
            close=close,
        )

    sig = _build_signal(
        cfg=cfg,
        direction=direction,
        pullback_bar=pullback,
        breakout_candle=breakout,
        ema_value=ema_value,
    )
    p = _compose_push(sig)
    return {
        "available": True,
        "synthesized": ema_now is None,
        "title": p["title"],
        "body": p["body"],
        "payload": p["data"],
    }


async def force_test_send(*, direction: str = "LONG") -> Dict[str, Any]:
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
    with _state_lock:
        _state.day = None
        _state.cur_candle = None
        _state.closed_candles = []
        _state.ema = None
        _state.last_regime = None
        _state.pending_pullback = None
        _state.pullback_direction = None
        _state.signals_today = 0
        _state.last_signal_direction = None
        _state.last_price = None
    log_info("[EMA] day state reset")
    return {"ok": True}
