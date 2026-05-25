"""
Live Nifty/VIX indicators from Kite WebSocket ticks (MODE_FULL day OHLC + bar build).

Falls back to Kite Connect historical/quote only for fields not yet available from ticks.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from utils.logger import log_info, log_warning

IST = ZoneInfo("Asia/Kolkata")
NIFTY_TOKEN_DEFAULT = 256265
VIX_TOKEN_DEFAULT = 264969
OR_START = time(9, 15)
OR_END = time(9, 30)
EMA_PERIOD = 9


def _floor_5m(dt: datetime) -> datetime:
    dt = dt.astimezone(IST)
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def _tick_time(tick: Dict[str, Any]) -> Optional[datetime]:
    ts = tick.get("exchange_timestamp") or tick.get("last_trade_time")
    if hasattr(ts, "astimezone"):
        return ts.astimezone(IST)
    return None


def _ema_update(prev: Optional[float], price: float, period: int = EMA_PERIOD) -> float:
    if prev is None:
        return price
    k = 2.0 / (period + 1.0)
    return price * k + prev * (1.0 - k)


@dataclass
class CandleBar:
    start: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class InstrumentLiveState:
    token: int
    day: Optional[date] = None
    last_price: Optional[float] = None
    day_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    prev_close: Optional[float] = None
    pdh: Optional[float] = None
    pdl: Optional[float] = None
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    ema9: Optional[float] = None
    last_5m_close: Optional[float] = None
    or_bar: Optional[CandleBar] = None
    or_locked: bool = False
    cur_5m: Optional[CandleBar] = None
    closed_5m: List[CandleBar] = field(default_factory=list)
    last_tick_at: Optional[datetime] = None
    tick_count_today: int = 0
    historical_seeded: bool = False
    sources: Dict[str, str] = field(default_factory=dict)


_engine_lock = threading.RLock()
_states: Dict[int, InstrumentLiveState] = {}
_registered = False


def _state(token: int) -> InstrumentLiveState:
    if token not in _states:
        _states[token] = InstrumentLiveState(token=token)
    return _states[token]


def _reset_day(st: InstrumentLiveState, today: date) -> None:
    st.day = today
    st.last_price = None
    st.day_open = st.day_high = st.day_low = None
    st.prev_close = None
    st.pdh = st.pdl = None
    st.or_high = st.or_low = None
    st.ema9 = None
    st.last_5m_close = None
    st.or_bar = None
    st.or_locked = False
    st.cur_5m = None
    st.closed_5m = []
    st.last_tick_at = None
    st.tick_count_today = 0
    st.historical_seeded = False
    st.sources = {}


def _sync_or_levels(st: InstrumentLiveState) -> None:
    if st.or_bar:
        st.or_high = st.or_bar.high
        st.or_low = st.or_bar.low


def _apply_tick_ohlc(st: InstrumentLiveState, tick: Dict[str, Any]) -> None:
    ohlc = tick.get("ohlc") or {}
    if not ohlc:
        return
    o = float(ohlc.get("open") or 0)
    h = float(ohlc.get("high") or 0)
    lo = float(ohlc.get("low") or 0)
    c = float(ohlc.get("close") or 0)
    if o > 0:
        st.day_open = o
        st.sources["day_open"] = "kite_ticker"
    if h > 0:
        st.day_high = h
        st.sources["day_high"] = "kite_ticker"
    if lo > 0:
        st.day_low = lo
        st.sources["day_low"] = "kite_ticker"
    if c > 0 and st.prev_close is None:
        st.prev_close = c
        st.sources["prev_close"] = "kite_ticker_ohlc"


def _update_or_from_tick(st: InstrumentLiveState, ts: datetime, price: float) -> None:
    t = ts.time()
    if OR_START <= t < OR_END:
        start = datetime.combine(ts.date(), OR_START, tzinfo=IST)
        if st.or_bar is None or st.or_bar.start.date() != ts.date():
            st.or_bar = CandleBar(start=start, open=price, high=price, low=price, close=price)
        else:
            st.or_bar.high = max(st.or_bar.high, price)
            st.or_bar.low = min(st.or_bar.low, price)
            st.or_bar.close = price
        _sync_or_levels(st)
        st.sources["or_high"] = "kite_ticker"
        st.sources["or_low"] = "kite_ticker"
    elif t >= OR_END and st.or_bar and not st.or_locked:
        st.or_locked = True
        _sync_or_levels(st)


def _update_5m_and_ema(st: InstrumentLiveState, ts: datetime, price: float) -> None:
    start = _floor_5m(ts)
    closed: Optional[CandleBar] = None
    if st.cur_5m is None:
        st.cur_5m = CandleBar(start=start, open=price, high=price, low=price, close=price)
    elif start > st.cur_5m.start:
        closed = st.cur_5m
        st.closed_5m.append(closed)
        if len(st.closed_5m) > 120:
            st.closed_5m = st.closed_5m[-120:]
        st.cur_5m = CandleBar(start=start, open=price, high=price, low=price, close=price)
    else:
        st.cur_5m.high = max(st.cur_5m.high, price)
        st.cur_5m.low = min(st.cur_5m.low, price)
        st.cur_5m.close = price

    if closed is not None:
        st.last_5m_close = closed.close
        st.sources["last_5m_close"] = "kite_ticker"
        st.ema9 = _ema_update(st.ema9, closed.close)
        st.sources["ema9"] = "kite_ticker"


def _on_ticks_batch(ticks: List[Dict[str, Any]]) -> None:
    with _engine_lock:
        for tick in ticks:
            token = tick.get("instrument_token")
            if token is None:
                continue
            token = int(token)
            st = _state(token)
            ts = _tick_time(tick)
            lp = tick.get("last_price")
            if ts is None or lp is None:
                continue
            try:
                price = float(lp)
            except (TypeError, ValueError):
                continue

            today = ts.date()
            if st.day != today:
                _reset_day(st, today)

            st.last_price = price
            st.last_tick_at = ts
            st.tick_count_today += 1
            st.sources["spot"] = "kite_ticker"

            _apply_tick_ohlc(st, tick)
            if token == NIFTY_TOKEN_DEFAULT:
                _update_or_from_tick(st, ts, price)
                _update_5m_and_ema(st, ts, price)


def ensure_kite_live_indicators_registered() -> None:
    global _registered
    if _registered:
        return
    from utils.kite_websocket_ticker import register_tick_callback

    register_tick_callback(_on_ticks_batch)
    _registered = True
    log_info("[KiteLiveIndicators] Tick handler registered (OR/EMA/day OHLC from stream)")


def _historical_seed_5m_ema(st: InstrumentLiveState, kite) -> None:
    if st.historical_seeded:
        return
    now = datetime.now(IST)
    from_dt = datetime.combine(now.date(), datetime.min.time()).replace(tzinfo=IST)
    try:
        c5 = kite.historical_data(
            st.token,
            from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d %H:%M:%S"),
            "5minute",
            continuous=False,
            oi=False,
        )
        if not c5:
            return
        ema: Optional[float] = st.ema9
        for row in c5:
            d = row.get("date")
            if not d:
                continue
            close = float(row["close"])
            if st.sources.get("ema9") != "kite_ticker":
                ema = _ema_update(ema, close)
            if st.sources.get("last_5m_close") != "kite_ticker":
                st.last_5m_close = close
        if ema is not None and st.ema9 is None:
            st.ema9 = ema
            st.sources["ema9"] = "kite_historical_5m"
        if st.last_5m_close and "last_5m_close" not in st.sources:
            st.sources["last_5m_close"] = "kite_historical_5m"
        st.historical_seeded = True
    except Exception as exc:
        log_warning(f"[KiteLiveIndicators] 5m seed: {exc}")


def _historical_fill_missing(st: InstrumentLiveState, kite) -> None:
    """Only populate fields not already set by ticker."""
    now = datetime.now(IST)
    today = now.date()

    if st.pdh is None or st.pdl is None:
        try:
            daily = kite.historical_data(
                st.token,
                (today - timedelta(days=12)).strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d"),
                "day",
                continuous=False,
                oi=False,
            )
            if daily and len(daily) >= 2:
                prev = daily[-2]
                if st.pdh is None:
                    st.pdh = float(prev.get("high") or 0)
                    st.sources["pdh"] = "kite_historical_day"
                if st.pdl is None:
                    st.pdl = float(prev.get("low") or 0)
                    st.sources["pdl"] = "kite_historical_day"
        except Exception as exc:
            log_warning(f"[KiteLiveIndicators] PDH/PDL: {exc}")

    if st.or_high is None or st.or_low is None:
        try:
            from_dt = datetime.combine(today, datetime.min.time()).replace(tzinfo=IST)
            c15 = kite.historical_data(
                st.token,
                from_dt.strftime("%Y-%m-%d %H:%M:%S"),
                now.strftime("%Y-%m-%d %H:%M:%S"),
                "15minute",
                continuous=False,
                oi=False,
            )
            or_bars = [
                c
                for c in (c15 or [])
                if c.get("date")
                and OR_START <= c["date"].astimezone(IST).time() < OR_END
            ]
            if or_bars:
                st.or_high = max(float(c["high"]) for c in or_bars)
                st.or_low = min(float(c["low"]) for c in or_bars)
                st.sources["or_high"] = "kite_historical_15m"
                st.sources["or_low"] = "kite_historical_15m"
        except Exception as exc:
            log_warning(f"[KiteLiveIndicators] OR hist: {exc}")

    if st.ema9 is None:
        _historical_seed_5m_ema(st, kite)

    if st.prev_close is None:
        try:
            key = "NSE:NIFTY 50" if st.token == NIFTY_TOKEN_DEFAULT else None
            if key:
                q = (kite.quote([key]) or {}).get(key, {})
                pc = float((q.get("ohlc") or {}).get("close") or 0)
                if pc > 0:
                    st.prev_close = pc
                    st.sources["prev_close"] = "kite_quote"
        except Exception:
            pass

    if st.last_price is None or st.sources.get("spot") != "kite_ticker":
        try:
            key = "NSE:NIFTY 50" if st.token == NIFTY_TOKEN_DEFAULT else "NSE:INDIA VIX"
            if st.token == VIX_TOKEN_DEFAULT:
                key = "NSE:INDIA VIX"
            q = (kite.quote([key]) or {}).get(key, {})
            lp = float(q.get("last_price") or 0)
            if lp > 0:
                st.last_price = lp
                st.sources["spot"] = "kite_quote"
                o = q.get("ohlc") or {}
                if st.day_open is None:
                    st.day_open = float(o.get("open") or lp)
                    st.sources["day_open"] = "kite_quote"
                if st.day_high is None:
                    st.day_high = float(o.get("high") or lp)
                    st.sources["day_high"] = "kite_quote"
                if st.day_low is None:
                    st.day_low = float(o.get("low") or lp)
                    st.sources["day_low"] = "kite_quote"
        except Exception as exc:
            log_warning(f"[KiteLiveIndicators] quote fallback: {exc}")


def _state_to_snapshot(st: InstrumentLiveState) -> Dict[str, Any]:
    _sync_or_levels(st)
    return {
        "instrument_token": st.token,
        "nifty_spot": st.last_price,
        "prev_close": st.prev_close,
        "day_open": st.day_open,
        "day_high": st.day_high,
        "day_low": st.day_low,
        "pdh": st.pdh,
        "pdl": st.pdl,
        "or_high": st.or_high,
        "or_low": st.or_low,
        "ema9": st.ema9,
        "last_5m_close": st.last_5m_close,
        "last_tick_at": st.last_tick_at.isoformat() if st.last_tick_at else None,
        "tick_count_today": st.tick_count_today,
        "or_locked": st.or_locked,
        "sources": dict(st.sources),
        "updated_at": datetime.now(IST).isoformat(),
    }


def get_live_indicator_snapshot(
    token: int = NIFTY_TOKEN_DEFAULT,
    *,
    fill_historical: bool = True,
) -> Dict[str, Any]:
    """
    Return latest indicators for token; merge ticker state + historical gaps only.
    Call on each V2 refresh / place to re-read ticker-updated values.
    """
    ensure_kite_live_indicators_registered()

    with _engine_lock:
        st = _state(token)
        if fill_historical:
            try:
                from utils.kite_utils import get_kite_instance

                kite = get_kite_instance()
                _historical_fill_missing(st, kite)
            except Exception as exc:
                log_warning(f"[KiteLiveIndicators] fill: {exc}")
        return _state_to_snapshot(st)


def get_vix_snapshot(*, fill_historical: bool = True) -> Dict[str, Any]:
    ensure_kite_live_indicators_registered()
    with _engine_lock:
        st = _state(VIX_TOKEN_DEFAULT)
    snap: Dict[str, Any] = {"ltp": st.last_price, "sources": dict(st.sources)}
    if snap["ltp"] is None or st.sources.get("spot") != "kite_ticker":
        try:
            from utils.kite_utils import get_kite_instance

            q = (get_kite_instance().quote(["NSE:INDIA VIX"]) or {}).get("NSE:INDIA VIX", {})
            snap["ltp"] = float(q.get("last_price") or 0)
            snap["sources"]["vix"] = "kite_quote"
        except Exception:
            snap["ltp"] = 0.0
    return snap


def get_nifty_bundle_for_v2() -> Dict[str, Any]:
    """Unified snapshot for V2 strategy + trade plan (ticker-first)."""
    nifty = get_live_indicator_snapshot(NIFTY_TOKEN_DEFAULT)
    vix = get_vix_snapshot()
    return {
        "nifty_spot": float(nifty.get("nifty_spot") or 0),
        "prev_close": float(nifty.get("prev_close") or nifty.get("nifty_spot") or 0),
        "day_open": float(nifty.get("day_open") or 0),
        "day_high": float(nifty.get("day_high") or 0),
        "day_low": float(nifty.get("day_low") or 0),
        "vix": float(vix.get("ltp") or 0),
        "pdh": nifty.get("pdh"),
        "pdl": nifty.get("pdl"),
        "or_high": nifty.get("or_high"),
        "or_low": nifty.get("or_low"),
        "ema9": nifty.get("ema9"),
        "last_5m_close": nifty.get("last_5m_close"),
        "spot_source": nifty.get("sources", {}).get("spot", "unknown"),
        "indicator_sources": nifty.get("sources", {}),
        "last_tick_at": nifty.get("last_tick_at"),
        "tick_count_today": nifty.get("tick_count_today", 0),
    }


def recalculate_from_ticker() -> Dict[str, Any]:
    """Explicit refresh after tick updates — used by checklist / place."""
    return get_nifty_bundle_for_v2()
