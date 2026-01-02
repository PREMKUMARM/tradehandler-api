"""
Trading signal generator for Binance Futures using VWAP + indicators
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from utils.binance_vwap import compute_vwap, compute_rsi, compute_macd
from utils.indicators import calculate_rsi


def detect_candlestick_pattern(current_idx: int, df: pd.DataFrame) -> str:
    """Detect candlestick pattern for a given candle index"""
    if current_idx < 0 or current_idx >= len(df):
        return 'Unknown'
    
    row = df.iloc[current_idx]
    open_price = row['open']
    high = row['high']
    low = row['low']
    close = row['close']
    body_size = abs(close - open_price)
    upper_wick = high - max(open_price, close)
    lower_wick = min(open_price, close) - low
    total_range = high - low
    
    if total_range == 0:
        return 'Doji'
    
    body_ratio = body_size / total_range
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range
    is_bullish = close > open_price
    is_bearish = close < open_price
    is_doji = body_ratio < 0.1
    
    prev_row = df.iloc[current_idx - 1] if current_idx > 0 else None
    prev_prev_row = df.iloc[current_idx - 2] if current_idx > 1 else None
    
    if is_doji:
        if upper_wick_ratio > 0.4 and lower_wick_ratio > 0.4:
            return 'Doji'
        elif upper_wick_ratio > 0.6:
            return 'Gravestone Doji'
        elif lower_wick_ratio > 0.6:
            return 'Dragonfly Doji'
        else:
            return 'Doji'
    
    if upper_wick_ratio < 0.05 and lower_wick_ratio < 0.05:
        return 'Bullish Marubozu' if is_bullish else 'Bearish Marubozu'
    
    if lower_wick_ratio > 0.6 and body_ratio < 0.3 and upper_wick_ratio < 0.2:
        return 'Hammer' if is_bullish else 'Hanging Man'
    
    if upper_wick_ratio > 0.6 and body_ratio < 0.3 and lower_wick_ratio < 0.2:
        return 'Inverted Hammer' if is_bullish else 'Shooting Star'
    
    if body_ratio > 0.7:
        return 'Long White Candle' if is_bullish else 'Long Black Candle'
    
    if body_ratio < 0.3:
        return 'Small White Candle' if is_bullish else 'Small Black Candle'
    
    if prev_row is not None:
        prev_open = prev_row['open']
        prev_close = prev_row['close']
        prev_high = prev_row['high']
        prev_low = prev_row['low']
        
        if is_bullish and prev_close < prev_open and close > prev_open and open_price < prev_close:
            return 'Bullish Engulfing'
        if is_bearish and prev_close > prev_open and close < prev_open and open_price > prev_close:
            return 'Bearish Engulfing'
        if is_bullish and prev_close > prev_open and open_price > prev_close and close < prev_open:
            return 'Bullish Harami'
        if is_bearish and prev_close < prev_open and open_price < prev_close and close > prev_open:
            return 'Bearish Harami'
        if is_bullish and prev_close < prev_open and close > (prev_open + prev_close) / 2:
            return 'Piercing Pattern'
        if is_bearish and prev_close > prev_open and close < (prev_open + prev_close) / 2:
            return 'Dark Cloud Cover'
    
    if prev_row is not None and prev_prev_row is not None:
        prev_prev_open = prev_prev_row['open']
        prev_prev_close = prev_prev_row['close']
        prev_open = prev_row['open']
        prev_close = prev_row['close']
        prev_body_size = abs(prev_close - prev_open)
        prev_prev_body_size = abs(prev_prev_close - prev_prev_open)
        
        if is_bullish and prev_prev_close < prev_prev_open and prev_body_size < prev_prev_body_size * 0.5 and close > (prev_prev_open + prev_prev_close) / 2:
            return 'Morning Star'
        if is_bearish and prev_prev_close > prev_prev_open and prev_body_size < prev_prev_body_size * 0.5 and close < (prev_prev_open + prev_prev_close) / 2:
            return 'Evening Star'
        if is_bullish and prev_close > prev_open and prev_prev_close > prev_prev_open and close > prev_close and prev_close > prev_prev_close:
            return 'Three White Soldiers'
        if is_bearish and prev_close < prev_open and prev_prev_close < prev_prev_open and close < prev_close and prev_close < prev_prev_close:
            return 'Three Black Crows'
    
    return 'Small White Candle' if is_bullish else 'Small Black Candle'


def generate_trading_signal(current_idx: int, df: pd.DataFrame) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Generate trading signal based on VWAP strategy logic
    
    Returns:
        (signal, priority, reason) where signal is 'BUY', 'SELL', or None
    """
    if current_idx < 2:
        return (None, None, None)
    
    row = df.iloc[current_idx]
    candle_type = row.get('candle_type', '')
    close = float(row.get('close', 0))
    open_price = float(row.get('open', 0))
    high = float(row.get('high', 0))
    vwap = float(row.get('vwap', 0))
    
    if vwap == 0 or close == 0:
        return (None, None, None)
    
    prev_row = df.iloc[current_idx - 1]
    prev_candle_type = prev_row.get('candle_type', '')
    
    # BUY Signal: Three Black Crows followed by bullish reversal pattern above VWAP
    if prev_candle_type == 'Three Black Crows':
        is_green_candle = close > open_price
        close_above_vwap = close > vwap
        high_above_vwap = high > vwap
        
        vwap_diff_percent = abs(close - vwap) / vwap * 100
        MAX_VWAP_DISTANCE_PCT = 2.0
        
        if vwap_diff_percent > MAX_VWAP_DISTANCE_PCT:
            return (None, None, None)
        
        high_performance_candle_types = [
            'Dragonfly Doji', 'Piercing Pattern',
            'Inverted Hammer', 'Long White Candle'
        ]
        current_candle_matches = any(pattern in candle_type for pattern in high_performance_candle_types)
        
        if not (is_green_candle and (close_above_vwap or high_above_vwap) and current_candle_matches):
            return (None, None, None)
        
        # Additional checks
        if current_idx < 6:
            return (None, None, None)
        
        # RSI check
        if current_idx >= 14:
            current_rsi = row.get('rsi', 50)
            prev_rsi = prev_row.get('rsi', 50) if current_idx > 0 else 50
            
            if pd.notna(current_rsi) and pd.notna(prev_rsi):
                if current_rsi > 65:
                    return (None, None, None)
                if current_rsi < prev_rsi and current_rsi < 40:
                    return (None, None, None)
                if current_rsi > 60 and current_rsi > prev_rsi:
                    return (None, None, None)
                if current_rsi < 30:
                    if current_rsi <= prev_rsi:
                        return (None, None, None)
        
        # MACD check
        if current_idx >= 26:
            current_macd = row.get('macd', 0)
            current_macd_signal = row.get('macd_signal', 0)
            prev_macd = prev_row.get('macd', 0) if current_idx > 0 else 0
            prev_macd_signal = prev_row.get('macd_signal', 0) if current_idx > 0 else 0
            
            if pd.notna(current_macd) and pd.notna(current_macd_signal):
                macd_bullish = current_macd > current_macd_signal
                macd_crossing = (prev_macd <= prev_macd_signal) and (current_macd > current_macd_signal)
                
                if not (macd_bullish or macd_crossing):
                    return (None, None, None)
        
        # Volume check
        if current_idx >= 1:
            current_volume = row.get('volume', 0)
            if current_idx >= 5:
                recent_volumes = [df.iloc[current_idx - i].get('volume', current_volume) for i in range(5)]
                avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else current_volume
                
                if avg_volume > 0 and current_volume < avg_volume * 0.8:
                    if 'Inverted Hammer' in candle_type or 'Piercing Pattern' in candle_type:
                        return (None, None, None)
            elif current_volume == 0:
                return (None, None, None)
        
        # All checks passed - generate BUY signal
        matched_pattern = next((p for p in high_performance_candle_types if p in candle_type), candle_type)
        reason = f"Priority 1: {matched_pattern} candle {'closing' if close_above_vwap else 'high'} above VWAP after Three Black Crows (VWAP: {vwap_diff_percent:.2f}%)"
        return ('BUY', 1, reason)
    
    # SELL Signal: Three White Soldiers followed by bearish reversal pattern below VWAP
    if prev_candle_type == 'Three White Soldiers':
        is_red_candle = close < open_price
        close_below_vwap = close < vwap
        low_below_vwap = row.get('low', close) < vwap
        
        vwap_diff_percent = abs(close - vwap) / vwap * 100
        MAX_VWAP_DISTANCE_PCT = 2.0
        
        if vwap_diff_percent > MAX_VWAP_DISTANCE_PCT:
            return (None, None, None)
        
        bearish_candle_types = [
            'Gravestone Doji', 'Dark Cloud Cover',
            'Shooting Star', 'Long Black Candle'
        ]
        current_candle_matches = any(pattern in candle_type for pattern in bearish_candle_types)
        
        if not (is_red_candle and (close_below_vwap or low_below_vwap) and current_candle_matches):
            return (None, None, None)
        
        # Additional checks similar to BUY
        if current_idx < 6:
            return (None, None, None)
        
        # RSI check for SELL
        if current_idx >= 14:
            current_rsi = row.get('rsi', 50)
            if pd.notna(current_rsi):
                if current_rsi < 35:
                    return (None, None, None)
                if current_rsi > 70:
                    return (None, None, None)
        
        # MACD check for SELL
        if current_idx >= 26:
            current_macd = row.get('macd', 0)
            current_macd_signal = row.get('macd_signal', 0)
            
            if pd.notna(current_macd) and pd.notna(current_macd_signal):
                macd_bearish = current_macd < current_macd_signal
                if not macd_bearish:
                    return (None, None, None)
        
        # All checks passed - generate SELL signal
        matched_pattern = next((p for p in bearish_candle_types if p in candle_type), candle_type)
        reason = f"Priority 1: {matched_pattern} candle {'closing' if close_below_vwap else 'low'} below VWAP after Three White Soldiers (VWAP: {vwap_diff_percent:.2f}%)"
        return ('SELL', 1, reason)
    
    return (None, None, None)


def analyze_symbol_for_signals(symbol: str, klines: list) -> Dict:
    """
    Analyze a symbol's klines to generate trading signals
    
    Args:
        symbol: Trading pair symbol
        klines: List of OHLCV dicts
    
    Returns:
        Dict with candle_pattern, signal, signal_priority, signal_reason
    """
    if not klines or len(klines) < 30:
        return {
            "candle_pattern": "N/A",
            "signal": None,
            "signal_priority": None,
            "signal_reason": None
        }
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(klines)
        # Ensure we have the required columns
        if 'open' not in df.columns or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            return {
                "candle_pattern": "N/A",
                "signal": None,
                "signal_priority": None,
                "signal_reason": None
            }
        
        # Calculate VWAP
        vwap = compute_vwap(klines)
        if vwap is None or vwap == 0:
            return {
                "candle_pattern": "N/A",
                "signal": None,
                "signal_priority": None,
                "signal_reason": None
            }
        
        # Add VWAP to dataframe
        df['vwap'] = vwap
        
        # Calculate RSI
        closes = df['close'].tolist()
        rsi_values = calculate_rsi(closes, period=14)
        if isinstance(rsi_values, list):
            # Pad with NaN if needed to match dataframe length
            if len(rsi_values) < len(df):
                rsi_values = [np.nan] * (len(df) - len(rsi_values)) + rsi_values
            df['rsi'] = rsi_values[:len(df)]
        else:
            df['rsi'] = np.nan
        
        # Calculate MACD for all candles
        if len(closes) >= 35:  # Need at least 26+9 for MACD
            series = pd.Series(closes)
            ema_fast = series.ewm(span=12, adjust=False).mean()
            ema_slow = series.ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
        else:
            df['macd'] = np.nan
            df['macd_signal'] = np.nan
        
        # Detect candlestick patterns
        df['candle_type'] = df.index.map(lambda i: detect_candlestick_pattern(i, df))
        
        # Get latest candle pattern
        latest_idx = len(df) - 1
        candle_pattern = df.iloc[latest_idx]['candle_type']
        
        # Generate trading signal
        signal, priority, reason = generate_trading_signal(latest_idx, df)
        
        return {
            "candle_pattern": candle_pattern,
            "signal": signal,
            "signal_priority": priority,
            "signal_reason": reason
        }
    except Exception as e:
        print(f"Error analyzing {symbol} for signals: {e}")
        import traceback
        traceback.print_exc()
        return {
            "candle_pattern": "Error",
            "signal": None,
            "signal_priority": None,
            "signal_reason": None
        }

