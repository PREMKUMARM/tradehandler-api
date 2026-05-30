"""Technical indicators for crypto BB mean-reversion (RSI, ADX, ATR, bandwidth)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def _last(values: Sequence[float]) -> Optional[float]:
    for v in reversed(values):
        if v is not None:
            return float(v)
    return None


def compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            avg_gain = (avg_gain * (period - 1) + d) / period
            avg_loss = (avg_loss * (period - 1)) / period
        else:
            avg_gain = (avg_gain * (period - 1)) / period
            avg_loss = (avg_loss * (period - 1) - d) / period
    if avg_loss <= 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _true_ranges(bars: List[Dict[str, float]]) -> List[float]:
    out: List[float] = []
    for i, b in enumerate(bars):
        h, l, c = float(b["high"]), float(b["low"]), float(b["close"])
        if i == 0:
            out.append(h - l)
            continue
        pc = float(bars[i - 1]["close"])
        out.append(max(h - l, abs(h - pc), abs(l - pc)))
    return out


def compute_atr(bars: List[Dict[str, float]], period: int = 14) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    trs = _true_ranges(bars)
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def compute_atr_series(bars: List[Dict[str, float]], period: int = 14) -> List[Optional[float]]:
    if len(bars) < period + 1:
        return [None] * len(bars)
    trs = _true_ranges(bars)
    out: List[Optional[float]] = [None] * len(bars)
    atr = sum(trs[:period]) / period
    out[period - 1] = atr
    for i in range(period, len(bars)):
        atr = (atr * (period - 1) + trs[i]) / period
        out[i] = atr
    return out


def compute_adx(bars: List[Dict[str, float]], period: int = 14) -> Optional[float]:
    """Wilder ADX(period) on OHLC bars."""
    if len(bars) < period * 2 + 1:
        return None

    plus_dm: List[float] = []
    minus_dm: List[float] = []
    trs = _true_ranges(bars)

    for i in range(1, len(bars)):
        up = float(bars[i]["high"]) - float(bars[i - 1]["high"])
        down = float(bars[i - 1]["low"]) - float(bars[i]["low"])
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)

    if len(trs) < period + 1:
        return None

    tr14 = sum(trs[1 : period + 1])
    p14 = sum(plus_dm[:period])
    m14 = sum(minus_dm[:period])
    if tr14 <= 0:
        return None

    dx_vals: List[float] = []
    for i in range(period, len(plus_dm)):
        tr14 = tr14 - tr14 / period + trs[i + 1]
        p14 = p14 - p14 / period + plus_dm[i]
        m14 = m14 - m14 / period + minus_dm[i]
        if tr14 <= 0:
            continue
        pdi = 100.0 * p14 / tr14
        mdi = 100.0 * m14 / tr14
        denom = pdi + mdi
        if denom <= 0:
            continue
        dx_vals.append(100.0 * abs(pdi - mdi) / denom)

    if len(dx_vals) < period:
        return None
    adx = sum(dx_vals[:period]) / period
    for dx in dx_vals[period:]:
        adx = (adx * (period - 1) + dx) / period
    return adx


def bb_bandwidth_pct(upper: float, lower: float, middle: float) -> float:
    if middle <= 0:
        return 0.0
    return max(0.0, (upper - lower) / middle * 100.0)


def middle_band_slope(closes: List[float], period: int = 20, lookback: int = 3) -> Optional[float]:
    """Positive = middle band rising (bullish context for longs)."""
    if len(closes) < period + lookback:
        return None
    from services.kite_live_indicators import compute_bollinger_bands

    mids: List[float] = []
    for i in range(lookback + 1):
        end = len(closes) - lookback + i
        if end < period:
            continue
        mid, _, _ = compute_bollinger_bands(closes[:end], period=period)
        if mid is not None:
            mids.append(mid)
    if len(mids) < 2:
        return None
    return mids[-1] - mids[0]


def close_back_inside_long(prev_bar: Dict[str, float], bar: Dict[str, float], lower: float) -> bool:
    """Prev bar closed at/below lower BB; current bar closed back inside."""
    prev_close = float(prev_bar["close"])
    close = float(bar["close"])
    pierced = prev_close <= lower or float(prev_bar["low"]) <= lower
    return pierced and close > lower


def close_back_inside_short(prev_bar: Dict[str, float], bar: Dict[str, float], upper: float) -> bool:
    prev_close = float(prev_bar["close"])
    close = float(bar["close"])
    pierced = prev_close >= upper or float(prev_bar["high"]) >= upper
    return pierced and close < upper
