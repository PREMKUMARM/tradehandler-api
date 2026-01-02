"""
RSI calculator for Binance data
"""
import pandas as pd
from typing import List, Optional
import numpy as np


def compute_rsi(closes: List[float], window: int = 14) -> Optional[float]:
    """
    Calculate RSI (Relative Strength Index) using Wilder's Smoothing Method
    
    Args:
        closes: List of closing prices
        window: RSI period (default: 14)
    
    Returns:
        Latest RSI value (0-100) or None if insufficient data
    """
    if len(closes) < window + 1:
        return None
    
    try:
        df = pd.DataFrame({"close": closes})
        
        # Calculate price changes
        delta = df["close"].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(alpha=1/window, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/window, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # Return the latest RSI value
        latest_rsi = rsi.iloc[-1]
        return round(float(latest_rsi), 2) if not pd.isna(latest_rsi) else None
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return None


def compute_rsi_batch(data: dict) -> dict:
    """
    Calculate RSI for multiple symbols and timeframes
    
    Args:
        data: Dictionary {symbol: {interval: [closes]}}
    
    Returns:
        Dictionary {symbol: {interval: rsi_value}}
    """
    results = {}
    for symbol, timeframes in data.items():
        results[symbol] = {}
        for interval, closes in timeframes.items():
            if closes and len(closes) > 0:
                rsi = compute_rsi(closes)
                results[symbol][interval] = rsi if rsi is not None else 0
            else:
                results[symbol][interval] = 0
    
    return results

