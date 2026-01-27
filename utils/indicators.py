"""
Technical indicator calculation utilities
"""
import pandas as pd
import numpy as np


def calculate_bollinger_bands(closes, period=20, num_std=2):
    """Calculate Bollinger Bands for mean reversion strategy"""
    if len(closes) < period:
        return None, None, None
    
    df = pd.DataFrame({'close': closes})
    df['sma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['sma'] + (df['std'] * num_std)
    df['lower_band'] = df['sma'] - (df['std'] * num_std)
    
    return df['upper_band'].iloc[-1], df['sma'].iloc[-1], df['lower_band'].iloc[-1]


def calculate_bollinger_bands_full(closes, period=20, num_std=2):
    """Calculate full Bollinger Bands array for chart display - matches TradingView/Zerodha standard"""
    if len(closes) < period:
        return [], [], []
    
    df = pd.DataFrame({'close': closes})
    # Use Simple Moving Average (SMA) - standard for Bollinger Bands
    df['sma'] = df['close'].rolling(window=period, min_periods=1).mean()
    # Use Population Standard Deviation (ddof=0) - matches TradingView default
    # Note: pandas std() uses ddof=1 by default (sample std), we need ddof=0 (population std)
    df['std'] = df['close'].rolling(window=period, min_periods=1).std(ddof=0)
    df['upper_band'] = df['sma'] + (df['std'] * num_std)
    df['lower_band'] = df['sma'] - (df['std'] * num_std)
    
    # Return arrays - keep NaN for first period-1 values (no forward fill for accuracy)
    # Only fill backward for the very first value if needed
    upper = df['upper_band'].fillna(method='bfill', limit=1).tolist()
    middle = df['sma'].fillna(method='bfill', limit=1).tolist()
    lower = df['lower_band'].fillna(method='bfill', limit=1).tolist()
    
    return upper, middle, lower


def calculate_rsi(closes, period=14):
    """Calculate RSI (Relative Strength Index) using Wilder's Smoothing Method - matches TradingView/Zerodha"""
    if len(closes) < period + 1:
        return [np.nan] * len(closes)  # Return NaN if not enough data
    
    # Convert to numpy array for faster computation
    closes_arr = np.array(closes, dtype=float)
    deltas = np.diff(closes_arr)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Initialize arrays for average gain and loss (same length as deltas)
    avg_gains = np.full(len(gains), np.nan, dtype=float)
    avg_losses = np.full(len(losses), np.nan, dtype=float)
    
    # First average: Simple average of first 'period' delta values
    # This corresponds to the RSI value at index 'period' in the closes array
    if len(gains) >= period:
        avg_gains[period - 1] = np.mean(gains[:period])
        avg_losses[period - 1] = np.mean(losses[:period])
        
        # Apply Wilder's smoothing for remaining values
        # Formula: avg = (prev_avg * (period - 1) + current) / period
        for i in range(period, len(gains)):
            avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period
    
    # Calculate RS and RSI
    # Handle division by zero
    rs = np.divide(avg_gains, avg_losses, out=np.full_like(avg_gains, np.nan), where=(avg_losses != 0))
    rsi_deltas = 100 - (100 / (1 + rs))
    
    # RSI array should match closes length
    # First 'period' values are NaN (not enough data)
    # RSI at index i corresponds to close at index i
    rsi_list = [np.nan] * period  # First period values are NaN
    
    # Append calculated RSI values (starting from index period)
    for i in range(period - 1, len(rsi_deltas)):
        val = rsi_deltas[i]
        if np.isnan(val):
            rsi_list.append(np.nan)
        else:
            rsi_list.append(float(val))
    
    # Ensure length matches closes
    while len(rsi_list) < len(closes):
        rsi_list.append(np.nan)
    
    return rsi_list[:len(closes)]


def calculate_pivot_points(high, low, close):
    """Calculate Pivot Points (Traditional/Standard method) - matches TradingView/Zerodha"""
    # Traditional Pivot Point calculation (Standard method)
    # Pivot = (High + Low + Close) / 3
    pivot = (high + low + close) / 3
    
    # Resistance levels (Traditional method)
    # R1 = 2 * Pivot - Low
    # R2 = Pivot + (High - Low)
    # R3 = High + 2 * (Pivot - Low)
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    # Support levels (Traditional method)
    # S1 = 2 * Pivot - High
    # S2 = Pivot - (High - Low)
    # S3 = Low - 2 * (High - Pivot)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        "pivot": float(pivot),
        "r1": float(r1), 
        "r2": float(r2), 
        "r3": float(r3),
        "s1": float(s1), 
        "s2": float(s2), 
        "s3": float(s3)
    }


def calculate_support_resistance(candles, lookback=20):
    """Calculate support and resistance levels"""
    if len(candles) < lookback:
        return None, None
    
    highs = [c.get("high", 0) for c in candles[-lookback:]]
    lows = [c.get("low", 0) for c in candles[-lookback:]]
    
    resistance = max(highs)
    support = min(lows)
    
    return resistance, support






