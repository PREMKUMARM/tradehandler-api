"""Live MCX CRUDEOILM indicators (ticker + historical pad, same engine as Nifty)."""
from __future__ import annotations

from typing import Any, Dict

from services.commodity_product_context import get_active_product
from services.commodity_instruments import future_token
from services.kite_live_indicators import ensure_kite_live_indicators_registered, get_live_indicator_snapshot
from utils.kite_utils import get_kite_instance


def _ensure_crude_subscribed() -> None:
    try:
        from utils.kite_websocket_ticker import get_kite_ticker_instance

        ticker = get_kite_ticker_instance()
        if ticker and ticker.is_connected:
            tok = future_token()
            if tok not in (ticker.instrument_tokens or []):
                ticker.subscribe([tok])
    except Exception:
        pass


def _mcx_future_ltp() -> tuple[float, str]:
    """MCX quote for CRUDEOILM future — avoids Nifty/VIX fallback in generic indicator engine."""
    try:
        kite = get_kite_instance()
        key = f"MCX:{get_active_product().future_symbol}"
        row = (kite.quote(key) or {}).get(key, {}) or {}
        ltp = float(row.get("last_price") or 0)
        if ltp <= 0:
            ohlc = row.get("ohlc") or {}
            ltp = float(ohlc.get("close") or ohlc.get("open") or 0)
        return ltp, "kite_quote_mcx"
    except Exception:
        return 0.0, "unknown"


def recalculate_from_ticker() -> Dict[str, Any]:
    ensure_kite_live_indicators_registered()
    _ensure_crude_subscribed()
    snap = get_live_indicator_snapshot(future_token(), fill_historical=True)
    spot = float(snap.get("nifty_spot") or 0)
    spot_src = snap.get("sources", {}).get("spot", "unknown")
    mcx_spot, mcx_src = _mcx_future_ltp()
    if mcx_spot >= 1000:
        spot = mcx_spot
        spot_src = mcx_src
    elif spot < 1000:
        hist_spot = float(snap.get("last_5m_close") or 0)
        if hist_spot >= 1000:
            spot = hist_spot
            spot_src = "kite_hist_5m_close"
        elif mcx_spot > 0:
            spot = mcx_spot
            spot_src = mcx_src
    return {
        "underlying_spot": spot,
        "nifty_spot": spot,
        "prev_close": float(snap.get("prev_close") or spot),
        "day_open": float(snap.get("day_open") or 0),
        "day_high": float(snap.get("day_high") or 0),
        "day_low": float(snap.get("day_low") or 0),
        "pdh": snap.get("pdh"),
        "pdl": snap.get("pdl"),
        "or_high": snap.get("or_high"),
        "or_low": snap.get("or_low"),
        "ema9": snap.get("ema9"),
        "bb_middle": snap.get("bb_middle"),
        "bb_upper": snap.get("bb_upper"),
        "bb_lower": snap.get("bb_lower"),
        "last_5m_close": snap.get("last_5m_close"),
        "vix": None,
        "spot_source": spot_src,
        "indicator_sources": snap.get("sources", {}),
        "last_tick_at": snap.get("last_tick_at"),
        "tick_count_today": snap.get("tick_count_today", 0),
        "hist_pad_bars": snap.get("hist_pad_bars", 0),
        "live_session_bars": snap.get("live_session_bars", 0),
        "indicator_window": snap.get("indicator_window"),
        "future_symbol": snap.get("instrument_token"),
    }
