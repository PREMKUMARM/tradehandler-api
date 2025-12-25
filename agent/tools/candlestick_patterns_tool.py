"""
Candlestick pattern detection tools
"""
from typing import Optional, List, Dict, Any, Union
from langchain_core.tools import tool
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.kite_utils import get_kite_instance
from agent.tools.instrument_resolver import resolve_instrument_name
from kiteconnect.exceptions import KiteException


def detect_doji_candle(open_price: float, high: float, low: float, close: float, body_threshold: float = 0.1) -> bool:
    """
    Detect if a candle is a doji pattern.
    Doji: Open and close are very close (small body relative to range)
    
    Args:
        open_price: Open price
        high: High price
        low: Low price
        close: Close price
        body_threshold: Body must be less than this percentage of range (default: 10%)
    
    Returns:
        True if doji pattern detected
    """
    if high == low:
        return False  # No range
    
    body = abs(close - open_price)
    range_size = high - low
    body_pct = (body / range_size) * 100 if range_size > 0 else 0
    
    return body_pct <= body_threshold


def detect_reversal_after_doji(
    candles: List[Dict[str, Any]],
    doji_idx: int,
    lookback_candles: int = 3,
    reversal_threshold: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    Detect if a reversal occurred after a doji candle.
    
    Args:
        candles: List of candle data
        doji_idx: Index of the doji candle
        lookback_candles: Number of candles to look back for trend
        reversal_threshold: Minimum price change percentage to consider it a reversal
    
    Returns:
        Dict with reversal info or None if no reversal
    """
    if doji_idx < lookback_candles or doji_idx >= len(candles) - 1:
        return None
    
    # Get trend before doji
    before_closes = [candles[i].get("close", 0) for i in range(doji_idx - lookback_candles, doji_idx)]
    if len(before_closes) < 2:
        return None
    
    before_trend = "UP" if before_closes[-1] > before_closes[0] else "DOWN"
    
    # Get price after doji
    doji_candle = candles[doji_idx]
    doji_close = doji_candle.get("close", 0)
    
    # Check next few candles for reversal
    for i in range(doji_idx + 1, min(doji_idx + 5, len(candles))):
        after_candle = candles[i]
        after_close = after_candle.get("close", 0)
        
        price_change_pct = abs((after_close - doji_close) / doji_close * 100) if doji_close > 0 else 0
        
        if price_change_pct >= reversal_threshold:
            # Check if direction reversed
            if before_trend == "UP" and after_close < doji_close:
                return {
                    "reversal_type": "BEARISH",
                    "doji_timestamp": doji_candle.get("date", ""),
                    "reversal_timestamp": after_candle.get("date", ""),
                    "doji_price": doji_close,
                    "reversal_price": after_close,
                    "price_change_pct": price_change_pct,
                    "candles_after": i - doji_idx
                }
            elif before_trend == "DOWN" and after_close > doji_close:
                return {
                    "reversal_type": "BULLISH",
                    "doji_timestamp": doji_candle.get("date", ""),
                    "reversal_timestamp": after_candle.get("date", ""),
                    "doji_price": doji_close,
                    "reversal_price": after_close,
                    "price_change_pct": price_change_pct,
                    "candles_after": i - doji_idx
                }
    
    return None


@tool
def find_candlestick_patterns(
    instrument_names: Union[str, List[str]],
    pattern: str = "doji",
    date: Optional[str] = None,
    interval: str = "minute",
    exchange: str = "NSE",
    check_reversal: bool = True
) -> dict:
    """
    Find candlestick patterns (doji, hammer, engulfing, etc.) in one or multiple instruments.
    Can detect if patterns resulted in reversals.
    
    Args:
        instrument_names: Single instrument name or list of instrument names (e.g., "RELIANCE" or ["RELIANCE", "INFOSYS"])
        pattern: Pattern to detect (doji, hammer, engulfing, etc.) - currently supports "doji"
        date: Date in YYYY-MM-DD format (default: today)
        interval: Time interval (minute, 5minute, 15minute, etc.)
        exchange: Exchange (NSE, NFO, BSE)
        check_reversal: Whether to check if pattern resulted in a reversal
        
    Returns:
        dict with detected patterns and reversals for each instrument
    """
    try:
        kite = get_kite_instance()
        
        # Normalize instrument names to list
        if isinstance(instrument_names, str):
            instrument_names = [instrument_names]
        
        # Parse date
        if date:
            start_dt = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            start_dt = datetime.now().date()
        
        results = {}
        
        for instrument_name in instrument_names:
            # Resolve instrument name
            instrument_info = resolve_instrument_name(instrument_name, exchange)
            if not instrument_info:
                results[instrument_name] = {
                    "status": "error",
                    "error": f"Instrument '{instrument_name}' not found"
                }
                continue
            
            instrument_token = instrument_info["instrument_token"]
            
            # Get historical data
            historical_data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=start_dt,
                to_date=start_dt,
                interval=interval
            )
            
            if not historical_data or len(historical_data) < 5:
                results[instrument_name] = {
                    "status": "error",
                    "error": "Insufficient data for pattern detection"
                }
                continue
            
            # Detect patterns
            patterns_found = []
            
            if pattern.lower() == "doji":
                for idx in range(len(historical_data)):
                    candle = historical_data[idx]
                    open_price = candle.get("open", 0)
                    high = candle.get("high", 0)
                    low = candle.get("low", 0)
                    close = candle.get("close", 0)
                    
                    if detect_doji_candle(open_price, high, low, close):
                        pattern_info = {
                            "timestamp": candle.get("date", ""),
                            "open": float(open_price),
                            "high": float(high),
                            "low": float(low),
                            "close": float(close),
                            "pattern": "DOJI",
                            "reversal": None
                        }
                        
                        # Check for reversal if requested
                        if check_reversal:
                            reversal = detect_reversal_after_doji(historical_data, idx)
                            if reversal:
                                pattern_info["reversal"] = reversal
                        
                        patterns_found.append(pattern_info)
            
            results[instrument_name] = {
                "status": "success",
                "instrument": instrument_info["tradingsymbol"],
                "date": str(start_dt),
                "interval": interval,
                "pattern": pattern.upper(),
                "patterns_found": patterns_found,
                "count": len(patterns_found),
                "reversals_count": sum(1 for p in patterns_found if p.get("reversal") is not None)
            }
        
        # Summary
        total_patterns = sum(r.get("count", 0) for r in results.values() if isinstance(r, dict) and r.get("status") == "success")
        total_reversals = sum(r.get("reversals_count", 0) for r in results.values() if isinstance(r, dict) and r.get("status") == "success")
        
        return {
            "status": "success",
            "date": str(start_dt),
            "pattern": pattern.upper(),
            "instruments_analyzed": len(instrument_names),
            "total_patterns_found": total_patterns,
            "total_reversals": total_reversals,
            "results": results
        }
        
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": f"Error finding candlestick patterns: {str(e)}",
            "traceback": traceback.format_exc()
        }

