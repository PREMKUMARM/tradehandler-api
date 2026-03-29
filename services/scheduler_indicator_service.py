"""
Compute indicator snapshots for scheduled Telegram reports (Zerodha Kite historical data).
"""

from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import log_error, log_warning
from utils.indicators import calculate_rsi, calculate_bollinger_bands, calculate_pivot_points


def _closes_highs_lows(candles: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float]]:
    closes, highs, lows = [], [], []
    for c in candles:
        closes.append(float(c.get("close") or 0))
        highs.append(float(c.get("high") or 0))
        lows.append(float(c.get("low") or 0))
    return closes, highs, lows


def _session_vwap_approx(candles: List[Dict[str, Any]]) -> Optional[float]:
    """VWAP using typical price * volume / volume for available bars."""
    num, den = 0.0, 0.0
    for c in candles:
        h, low, cl = float(c.get("high") or 0), float(c.get("low") or 0), float(c.get("close") or 0)
        v = float(c.get("volume") or 0)
        if h <= 0 or low <= 0 or cl <= 0:
            continue
        tp = (h + low + cl) / 3.0
        num += tp * v
        den += v
    if den <= 0:
        return None
    return num / den


def _ema_series(closes: List[float], span: int) -> Optional[float]:
    if len(closes) < span + 1:
        return None
    s = pd.Series(closes, dtype=float)
    return float(s.ewm(span=span, adjust=False).mean().iloc[-1])


def build_indicator_report_text(
    symbol_label: str,
    quote_key: str,
    interval: str,
    candles: List[Dict[str, Any]],
    indicator_ids: List[str],
) -> str:
    """Plain-text report section for Telegram."""
    if not candles:
        return f"No historical candles returned for {quote_key} ({interval})."

    closes, highs, lows = _closes_highs_lows(candles)
    last = closes[-1] if closes else 0.0
    lines: List[str] = [
        f"Instrument: {symbol_label}",
        f"Quote key: {quote_key}",
        f"Interval: {interval}",
        f"Bars: {len(candles)}",
        f"Last close: {last:,.2f}",
        "",
    ]

    for ind in indicator_ids:
        ind = (ind or "").strip().lower()
        try:
            if ind == "rsi_14":
                rsi_arr = calculate_rsi(closes, 14)
                last_rsi = next((x for x in reversed(rsi_arr) if x == x and not np.isnan(x)), None)
                lines.append(f"RSI (14): {last_rsi:.2f}" if last_rsi is not None else "RSI (14): insufficient data")

            elif ind == "ema_9_21":
                e9 = _ema_series(closes, 9)
                e21 = _ema_series(closes, 21)
                if e9 is not None and e21 is not None:
                    state = "bullish (EMA9 > EMA21)" if e9 > e21 else "bearish (EMA9 < EMA21)"
                    lines.append(f"EMA 9: {e9:,.2f} | EMA 21: {e21:,.2f} | {state}")
                else:
                    lines.append("EMA 9/21: insufficient data")

            elif ind == "vwap_distance":
                vw = _session_vwap_approx(candles)
                if vw and last > 0:
                    dist = (last - vw) / vw * 100.0
                    lines.append(f"Session VWAP (approx): {vw:,.2f} | Distance: {dist:+.2f}%")
                else:
                    lines.append("VWAP: insufficient volume or data")

            elif ind == "bollinger_20":
                u, m, lo = calculate_bollinger_bands(closes, 20, 2)
                if u is not None and not (isinstance(u, float) and np.isnan(u)):
                    lines.append(f"Bollinger 20: upper {u:,.2f} | mid {m:,.2f} | lower {lo:,.2f}")
                else:
                    lines.append("Bollinger: insufficient data")

            elif ind == "pivots_daily":
                if len(candles) < 2:
                    lines.append("Pivots: need at least 2 bars")
                else:
                    prev = candles[-2]
                    h, low, cl = float(prev.get("high")), float(prev.get("low")), float(prev.get("close"))
                    pp = calculate_pivot_points(h, low, cl)
                    lines.append(
                        f"Pivots (prev bar): P {pp['pivot']:.2f} | R1 {pp['r1']:.2f} R2 {pp['r2']:.2f} | "
                        f"S1 {pp['s1']:.2f} S2 {pp['s2']:.2f}"
                    )
            else:
                lines.append(f"Unknown indicator id: {ind}")
        except Exception as ex:
            log_warning(f"Indicator {ind} failed: {ex}")
            lines.append(f"{ind}: error ({ex})")

    return "\n".join(lines)


def fetch_candles_sync(
    kite: Any,
    instrument_token: int,
    interval: str,
) -> List[Dict[str, Any]]:
    """Fetch enough history for indicators (sync; run in thread pool)."""
    to_dt = datetime.now()
    if interval == "day":
        from_dt = to_dt - timedelta(days=400)
    else:
        from_dt = to_dt - timedelta(days=14)
    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_dt.date() if interval == "day" else from_dt,
            to_date=to_dt,
            interval=interval,
        )
        return data or []
    except Exception as e:
        log_error(f"historical_data failed: {e}")
        return []
