"""
NIFTY50 5-minute candle alerts using the existing Kite WebSocket ticker stream.

We aggregate ticks into 5-minute OHLC bars for the NIFTY 50 index and detect the
long-upper-wick + small-body candle (shooting-star style) on candle CLOSE.

Implementation notes:
- Tick callbacks run in KiteTicker's background thread; we must not block.
- We schedule the async push send onto the main FastAPI event loop using
  asyncio.run_coroutine_threadsafe.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from services.push.push_service import push_service
from utils.logger import log_error, log_info, log_warning


IST = ZoneInfo("Asia/Kolkata")


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _floor_5m(dt: datetime) -> datetime:
    dt = dt.astimezone(IST)
    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute, second=0, microsecond=0)


def _tick_time(t: Dict[str, Any]) -> Optional[datetime]:
    ts = t.get("exchange_timestamp") or t.get("last_trade_time")
    if hasattr(ts, "astimezone"):
        return ts.astimezone(IST)
    return None


def _is_long_upper_wick_small_body(c: Dict[str, float]) -> bool:
    o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
    rng = max(0.0, h - l)
    if rng <= 0:
        return False
    body = abs(cl - o)
    upper = h - max(o, cl)
    lower = min(o, cl) - l

    max_body_pct = float(os.getenv("NIFTY_ALERT_MAX_BODY_PCT", "0.30"))
    min_upper_body = float(os.getenv("NIFTY_ALERT_MIN_UPPER_TO_BODY", "2.0"))
    max_close_from_low_pct = float(os.getenv("NIFTY_ALERT_MAX_CLOSE_FROM_LOW_PCT", "0.25"))
    min_range_pts = float(os.getenv("NIFTY_ALERT_MIN_RANGE_POINTS", "5"))

    if rng < min_range_pts:
        return False
    if body / rng > max_body_pct:
        return False
    if body <= 0.01:
        return upper / rng >= 0.60
    if upper / body < min_upper_body:
        return False
    if (cl - l) / rng > max_close_from_low_pct:
        return False
    if lower / rng > 0.45 and upper / rng < 0.55:
        return False
    return True


def _is_long_lower_wick_small_body(c: Dict[str, float]) -> bool:
    """Opposite pattern: long lower wick + small body (hammer / hanging-man style)."""
    o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
    rng = max(0.0, h - l)
    if rng <= 0:
        return False
    body = abs(cl - o)
    upper = h - max(o, cl)
    lower = min(o, cl) - l

    max_body_pct = float(os.getenv("NIFTY_ALERT_MAX_BODY_PCT", "0.30"))
    min_lower_body = float(os.getenv("NIFTY_ALERT_MIN_LOWER_TO_BODY", "2.0"))
    max_close_from_high_pct = float(os.getenv("NIFTY_ALERT_MAX_CLOSE_FROM_HIGH_PCT", "0.25"))
    min_range_pts = float(os.getenv("NIFTY_ALERT_MIN_RANGE_POINTS", "5"))

    if rng < min_range_pts:
        return False
    if body / rng > max_body_pct:
        return False
    if body <= 0.01:
        return lower / rng >= 0.60
    if lower / body < min_lower_body:
        return False
    if (h - cl) / rng > max_close_from_high_pct:
        return False
    # Avoid inverted-hammer-like candles (very long upper wick)
    if upper / rng > 0.45 and lower / rng < 0.55:
        return False
    return True


@dataclass
class Candle:
    start: datetime
    open: float
    high: float
    low: float
    close: float

    def as_dict(self) -> Dict[str, float]:
        return {"open": self.open, "high": self.high, "low": self.low, "close": self.close}


class Nifty5mCandleAggregator:
    def __init__(self, *, token: int) -> None:
        self.token = token
        self._cur: Optional[Candle] = None
        self._last_alert_upper_start: Optional[datetime] = None
        self._last_alert_lower_start: Optional[datetime] = None

    def on_tick(self, tick: Dict[str, Any]) -> Optional[Candle]:
        """Update state. Returns a CLOSED candle when we roll to next window."""
        ts = _tick_time(tick)
        lp = tick.get("last_price")
        if ts is None or lp is None:
            return None
        try:
            price = float(lp)
        except Exception:
            return None

        start = _floor_5m(ts)
        if self._cur is None:
            self._cur = Candle(start=start, open=price, high=price, low=price, close=price)
            return None

        if start == self._cur.start:
            self._cur.high = max(self._cur.high, price)
            self._cur.low = min(self._cur.low, price)
            self._cur.close = price
            return None

        # Rolled into a new 5m window: close previous candle
        closed = self._cur
        self._cur = Candle(start=start, open=price, high=price, low=price, close=price)
        return closed

    def alert_type(self, closed: Candle) -> Optional[str]:
        """
        Returns:
          - 'upper' for long upper wick rejection
          - 'lower' for long lower wick rejection
          - None if no alert
        """
        d = closed.as_dict()
        if self._last_alert_upper_start != closed.start and _is_long_upper_wick_small_body(d):
            self._last_alert_upper_start = closed.start
            return "upper"
        if self._last_alert_lower_start != closed.start and _is_long_lower_wick_small_body(d):
            self._last_alert_lower_start = closed.start
            return "lower"
        return None


_registered = False
_loop: Optional[asyncio.AbstractEventLoop] = None
_agg: Optional[Nifty5mCandleAggregator] = None


def register_nifty_5m_rejection_alerts(loop: asyncio.AbstractEventLoop) -> None:
    """
    Register the tick callback into the existing Kite ticker system.
    Call this once during FastAPI startup (main event loop).
    """
    global _registered, _loop, _agg
    if _registered:
        return
    if not _env_bool("NIFTY_5M_REJECTION_ALERT_ENABLED", True):
        log_info("[NIFTY Alert] Disabled via NIFTY_5M_REJECTION_ALERT_ENABLED=0")
        _registered = True
        return

    token = int(os.getenv("NIFTY_5M_REJECTION_ALERT_TOKEN", "256265"))
    _loop = loop
    _agg = Nifty5mCandleAggregator(token=token)

    from utils.kite_websocket_ticker import register_tick_callback

    register_tick_callback(_on_ticks_batch)
    _registered = True
    log_info(f"[NIFTY Alert] Registered tick candle alerts for token={token} (5m).")


def _on_ticks_batch(ticks: List[Dict]) -> None:
    # Runs in KiteTicker thread — must not block.
    if _loop is None or _agg is None:
        return
    user_id = os.getenv("NIFTY_5M_REJECTION_ALERT_USER_ID", "default").strip() or "default"
    for t in ticks:
        try:
            if int(t.get("instrument_token") or 0) != _agg.token:
                continue
        except Exception:
            continue

        closed = _agg.on_tick(t)
        if not closed:
            continue
        if not push_service.configured():
            continue
        at = _agg.alert_type(closed)
        if not at:
            continue

        close_time = closed.start + timedelta(minutes=5)
        when = f"{closed.start.strftime('%Y-%m-%d %H:%M')}-{close_time.strftime('%H:%M')} IST"

        if at == "upper":
            title = os.getenv(
                "NIFTY_5M_UPPER_WICK_ALERT_TITLE", "NIFTY 5m: long upper wick (rejection)"
            ).strip()
            candle_type = "Long upper wick (rejection)"
            event_type = "nifty_5m_long_upper_wick"
        else:
            title = os.getenv(
                "NIFTY_5M_LOWER_WICK_ALERT_TITLE", "NIFTY 5m: long lower wick (hammer)"
            ).strip()
            candle_type = "Long lower wick (hammer/hanging-man)"
            event_type = "nifty_5m_long_lower_wick"

        body = (
            f"{candle_type} candle at {when}. "
            f"O={closed.open:.2f} H={closed.high:.2f} L={closed.low:.2f} C={closed.close:.2f}."
        )
        data = {
            "type": event_type,
            "candle_type": candle_type,
            "bar_start_ist": closed.start.isoformat(),
            "bar_close_ist": close_time.isoformat(),
            "instrument": "NIFTY 50",
            "timeframe": "5m",
        }

        fut = asyncio.run_coroutine_threadsafe(
            push_service.send_to_user(user_id=user_id, title=title, body=body, data=data),
            _loop,
        )
        # Swallow exceptions to avoid impacting ticker thread.
        fut.add_done_callback(lambda f: f.exception())

