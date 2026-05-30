"""Live BTCUSDT indicators from Binance futures klines."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from services.crypto_config import SYMBOL
from utils.binance_client import BASE_URL


def _fetch_klines_sync(symbol: str, interval: str, limit: int = 120) -> List[Dict[str, float]]:
    url = f"{BASE_URL}/fapi/v1/klines"
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(
            url,
            params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()
    out: List[Dict[str, float]] = []
    for k in data:
        out.append(
            {
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )
    return out


def _ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def recalculate_from_ticker() -> Dict[str, Any]:
    """OR / PDH / PDL / EMA9 / VWAP / BB / RSI / ADX / ATR for BTCUSDT."""
    sym = SYMBOL
    k5 = _fetch_klines_sync(sym, "5m", 120)
    k1d = _fetch_klines_sync(sym, "1d", 3)
    if not k5:
        return {"connected": False, "message": "No kline data"}

    spot = float(k5[-1]["close"])
    prev_close = float(k1d[-2]["close"]) if len(k1d) >= 2 else spot

    # Opening range: first 6×5m bars of UTC day proxy (first 30m of series window)
    or_bars = k5[:6] if len(k5) >= 6 else k5[:3]
    or_high = max(b["high"] for b in or_bars)
    or_low = min(b["low"] for b in or_bars)

    pdh = float(k1d[-1]["high"]) if k1d else spot
    pdl = float(k1d[-1]["low"]) if k1d else spot
    day_open = float(k1d[-1]["open"]) if k1d else spot
    day_candle_green = spot > day_open
    day_candle_red = spot < day_open

    closes = [b["close"] for b in k5]
    ema9 = _ema(closes, 9) or spot

    from services.kite_live_indicators import compute_bollinger_bands
    from services.crypto_ta import (
        bb_bandwidth_pct,
        compute_adx,
        compute_atr,
        compute_atr_series,
        compute_rsi,
        middle_band_slope,
    )

    bb_mid, bb_upper, bb_lower = compute_bollinger_bands(closes, period=20)
    rsi14 = compute_rsi(closes, period=14)
    adx14 = compute_adx(k5, period=14)
    atr14 = compute_atr(k5, period=14)
    atr_series = compute_atr_series(k5, period=14)
    atr_sma20 = None
    if atr_series:
        recent = [a for a in atr_series[-20:] if a is not None]
        if recent:
            atr_sma20 = sum(recent) / len(recent)
    bb_bw = (
        bb_bandwidth_pct(float(bb_upper), float(bb_lower), float(bb_mid))
        if bb_mid and bb_upper and bb_lower
        else None
    )
    mid_slope = middle_band_slope(closes, period=20, lookback=3)
    atr_ratio = (atr14 / atr_sma20) if atr14 and atr_sma20 and atr_sma20 > 0 else None
    vols = [b["volume"] for b in k5]
    vol_avg20 = sum(vols[-20:]) / min(20, len(vols)) if vols else None

    prev_bar = k5[-3] if len(k5) >= 3 else k5[-2]
    last_bar = k5[-2] if len(k5) >= 2 else k5[-1]
    signal_spot = float(last_bar["close"])

    # Session VWAP on 5m
    cum_pv = 0.0
    cum_v = 0.0
    for b in k5:
        tp = (b["high"] + b["low"] + b["close"]) / 3.0
        v = b["volume"]
        cum_pv += tp * v
        cum_v += v
    vwap = cum_pv / cum_v if cum_v > 0 else spot

    return {
        "connected": True,
        "symbol": sym,
        "btc_spot": round(spot, 2),
        "signal_spot": round(signal_spot, 2),
        "nifty_spot": round(spot, 2),
        "prev_close": round(prev_close, 2),
        "or_high": round(or_high, 2),
        "or_low": round(or_low, 2),
        "pdh": round(pdh, 2),
        "pdl": round(pdl, 2),
        "day_open": round(day_open, 2),
        "day_candle_green": day_candle_green,
        "day_candle_red": day_candle_red,
        "ema9": round(ema9, 2),
        "vwap": round(vwap, 2),
        "bb_middle": round(bb_mid, 2) if bb_mid is not None else None,
        "bb_upper": round(bb_upper, 2) if bb_upper is not None else None,
        "bb_lower": round(bb_lower, 2) if bb_lower is not None else None,
        "bb_bandwidth_pct": round(bb_bw, 3) if bb_bw is not None else None,
        "rsi14": round(rsi14, 2) if rsi14 is not None else None,
        "adx14": round(adx14, 2) if adx14 is not None else None,
        "atr14": round(atr14, 2) if atr14 is not None else None,
        "atr_sma20": round(atr_sma20, 2) if atr_sma20 is not None else None,
        "atr_ratio": round(atr_ratio, 3) if atr_ratio is not None else None,
        "bb_middle_slope": round(mid_slope, 2) if mid_slope is not None else None,
        "volume_avg20": round(vol_avg20, 4) if vol_avg20 is not None else None,
        "prev_5m_bar": prev_bar,
        "last_5m_bar": last_bar,
        "last_5m_close": round(closes[-2], 2) if len(closes) >= 2 else round(spot, 2),
        "data_source": "binance_futures",
        "indicator_sources": {
            "bb_middle": "binance_futures_5m",
            "bb_upper": "binance_futures_5m",
            "bb_lower": "binance_futures_5m",
        },
    }
