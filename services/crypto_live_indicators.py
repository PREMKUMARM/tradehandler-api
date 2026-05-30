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
    """OR / PDH / PDL / EMA9 / VWAP-style context for BTCUSDT."""
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

    bb_mid, bb_upper, bb_lower = compute_bollinger_bands(closes, period=20)

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
        "last_5m_close": round(closes[-2], 2) if len(closes) >= 2 else round(spot, 2),
        "data_source": "binance_futures",
        "indicator_sources": {
            "bb_middle": "binance_futures_5m",
            "bb_upper": "binance_futures_5m",
            "bb_lower": "binance_futures_5m",
        },
    }
