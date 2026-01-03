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


def generate_trading_signal(current_idx: int, df: pd.DataFrame, debug: bool = False) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Generate trading signal - SCALPING strategy
    Focus: More frequent signals, quick profits, tighter stops
    
    Args:
        current_idx: Index of current candle
        df: DataFrame with OHLCV and indicator data
        debug: If True, prints debug information
    
    Returns:
        (signal, priority, reason) where signal is 'BUY', 'SELL', or None
    """
    if current_idx < 3:
        if debug:
            print(f"[Signal Debug] Insufficient data: current_idx={current_idx} (need >= 3)")
        return (None, None, None)
    
    row = df.iloc[current_idx]
    candle_type = row.get('candle_type', '')
    close = float(row.get('close', 0))
    open_price = float(row.get('open', 0))
    high = float(row.get('high', 0))
    low = float(row.get('low', 0))
    vwap = float(row.get('vwap', 0))
    
    if vwap == 0 or close == 0:
        if debug:
            print(f"[Signal Debug] Invalid VWAP or close: vwap={vwap}, close={close}")
        return (None, None, None)
    
    prev_row = df.iloc[current_idx - 1]
    prev_2_row = df.iloc[current_idx - 2] if current_idx >= 2 else None
    
    if debug:
        print(f"[Signal Debug] Current: {candle_type}, VWAP: {vwap:.2f}, Close: {close:.2f}")
    
    # Skip first 5 candles to avoid market open volatility
    if current_idx < 5:
        return (None, None, None)
    
    # Calculate VWAP distance
    vwap_diff_percent = abs(close - vwap) / vwap * 100
    MAX_VWAP_DISTANCE_PCT = 2.5  # Relaxed for more signals (scalping)
    
    if vwap_diff_percent > MAX_VWAP_DISTANCE_PCT:
        return (None, None, None)
    
    # BUY SIGNALS - Multiple patterns for scalping
    bullish_patterns = [
        'Hammer', 'Dragonfly Doji', 'Piercing Pattern', 'Inverted Hammer',
        'Bullish Engulfing', 'Long White Candle', 'Morning Star'
    ]
    
    is_bullish_pattern = any(pattern in candle_type for pattern in bullish_patterns)
    is_green_candle = close > open_price
    close_above_vwap = close > vwap
    high_above_vwap = high > vwap
    
    # BUY: Green candle with bullish pattern near/above VWAP
    if is_green_candle and is_bullish_pattern and (close_above_vwap or high_above_vwap):
        if debug:
            print(f"[Signal Debug] BUY check: Found bullish pattern {candle_type}")
        
        # Simple RSI check - not too overbought
        if current_idx >= 14:
            current_rsi = row.get('rsi', 50)
            if pd.notna(current_rsi):
                if current_rsi > 75:  # Too overbought
                    if debug:
                        print(f"[Signal Debug] BUY failed: RSI too high ({current_rsi:.2f})")
                    return (None, None, None)
                # Prefer RSI between 30-70 for scalping
                if current_rsi < 25:  # Too oversold, might be weak
                    if debug:
                        print(f"[Signal Debug] BUY failed: RSI too low ({current_rsi:.2f})")
                    return (None, None, None)
        
        # Simple MACD check - prefer bullish or neutral
        if current_idx >= 26:
            current_macd = row.get('macd', 0)
            current_macd_signal = row.get('macd_signal', 0)
            if pd.notna(current_macd) and pd.notna(current_macd_signal):
                # Allow if MACD is bullish OR if it's close to crossing
                macd_bullish = current_macd > current_macd_signal
                macd_near_cross = abs(current_macd - current_macd_signal) < abs(current_macd) * 0.1
                if not (macd_bullish or macd_near_cross):
                    if debug:
                        print(f"[Signal Debug] BUY failed: MACD not bullish")
                    return (None, None, None)
        
        # Volume check - ensure some volume
        current_volume = row.get('volume', 0)
        if current_idx >= 5:
            recent_volumes = [df.iloc[current_idx - i].get('volume', current_volume) for i in range(5)]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else current_volume
            if avg_volume > 0 and current_volume < avg_volume * 0.5:  # At least 50% of average
                if debug:
                    print(f"[Signal Debug] BUY failed: Volume too low ({current_volume:.2f} < {avg_volume * 0.5:.2f})")
                return (None, None, None)
        
        # Momentum: Price should be moving up (close higher than 1-2 candles ago)
        if current_idx >= 2:
            price_1_candle_ago = prev_row.get('close', close)
            price_2_candles_ago = prev_2_row.get('close', close) if prev_2_row is not None else close
            # Allow if price is higher than at least one of the previous candles
            if close <= price_1_candle_ago and close <= price_2_candles_ago:
                if debug:
                    print(f"[Signal Debug] BUY failed: No momentum")
                return (None, None, None)
        
        # Pattern-specific confirmations
        if 'Inverted Hammer' in candle_type:
            # For Inverted Hammer, prefer if next candle confirms
            if current_idx < len(df) - 1:
                next_row = df.iloc[current_idx + 1]
                next_close = next_row.get('close', close)
                if next_close > close:
                    matched_pattern = 'Inverted Hammer (Confirmed)'
                else:
                    matched_pattern = 'Inverted Hammer'
            else:
                matched_pattern = 'Inverted Hammer'
        else:
            matched_pattern = next((p for p in bullish_patterns if p in candle_type), candle_type)
        
        reason = f"Scalping BUY: {matched_pattern} above VWAP (VWAP dist: {vwap_diff_percent:.2f}%)"
        if debug:
            print(f"[Signal Debug] BUY signal generated: {reason}")
        return ('BUY', 1, reason)
    
    # SELL SIGNALS - Bearish patterns below VWAP
    bearish_patterns = [
        'Shooting Star', 'Gravestone Doji', 'Dark Cloud Cover',
        'Bearish Engulfing', 'Long Black Candle', 'Evening Star'
    ]
    
    is_bearish_pattern = any(pattern in candle_type for pattern in bearish_patterns)
    is_red_candle = close < open_price
    close_below_vwap = close < vwap
    low_below_vwap = low < vwap
    
    # SELL: Red candle with bearish pattern near/below VWAP
    if is_red_candle and is_bearish_pattern and (close_below_vwap or low_below_vwap):
        if debug:
            print(f"[Signal Debug] SELL check: Found bearish pattern {candle_type}")
        
        # Simple RSI check - not too oversold
        if current_idx >= 14:
            current_rsi = row.get('rsi', 50)
            if pd.notna(current_rsi):
                if current_rsi < 25:  # Too oversold
                    if debug:
                        print(f"[Signal Debug] SELL failed: RSI too low ({current_rsi:.2f})")
                    return (None, None, None)
        
        # Simple MACD check - prefer bearish or neutral
        if current_idx >= 26:
            current_macd = row.get('macd', 0)
            current_macd_signal = row.get('macd_signal', 0)
            if pd.notna(current_macd) and pd.notna(current_macd_signal):
                macd_bearish = current_macd < current_macd_signal
                macd_near_cross = abs(current_macd - current_macd_signal) < abs(current_macd) * 0.1
                if not (macd_bearish or macd_near_cross):
                    if debug:
                        print(f"[Signal Debug] SELL failed: MACD not bearish")
                    return (None, None, None)
        
        # Volume check
        current_volume = row.get('volume', 0)
        if current_idx >= 5:
            recent_volumes = [df.iloc[current_idx - i].get('volume', current_volume) for i in range(5)]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else current_volume
            if avg_volume > 0 and current_volume < avg_volume * 0.5:
                if debug:
                    print(f"[Signal Debug] SELL failed: Volume too low")
                return (None, None, None)
        
        # Momentum: Price should be moving down
        if current_idx >= 2:
            price_1_candle_ago = prev_row.get('close', close)
            price_2_candles_ago = prev_2_row.get('close', close) if prev_2_row is not None else close
            if close >= price_1_candle_ago and close >= price_2_candles_ago:
                if debug:
                    print(f"[Signal Debug] SELL failed: No downward momentum")
                return (None, None, None)
        
        matched_pattern = next((p for p in bearish_patterns if p in candle_type), candle_type)
        reason = f"Scalping SELL: {matched_pattern} below VWAP (VWAP dist: {vwap_diff_percent:.2f}%)"
        if debug:
            print(f"[Signal Debug] SELL signal generated: {reason}")
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
        
        # Calculate validation checks
        prev_row = df.iloc[latest_idx - 1] if latest_idx > 0 else None
        prev_candle_type = prev_row.get('candle_type', 'N/A') if prev_row is not None else 'N/A'
        
        close = float(df.iloc[latest_idx].get('close', 0))
        open_price = float(df.iloc[latest_idx].get('open', 0))
        high = float(df.iloc[latest_idx].get('high', 0))
        low = float(df.iloc[latest_idx].get('low', 0))
        
        is_green = close > open_price
        is_red = close < open_price
        close_above_vwap = close > vwap if vwap else False
        close_below_vwap = close < vwap if vwap else False
        vwap_diff_pct = abs(close - vwap) / vwap * 100 if vwap and vwap > 0 else None
        
        latest_rsi = df.iloc[latest_idx].get('rsi')
        prev_rsi = prev_row.get('rsi') if prev_row is not None else None
        
        latest_macd = df.iloc[latest_idx].get('macd')
        latest_macd_signal = df.iloc[latest_idx].get('macd_signal')
        prev_macd = prev_row.get('macd') if prev_row is not None else None
        prev_macd_signal = prev_row.get('macd_signal') if prev_row is not None else None
        
        macd_bullish = bool(pd.notna(latest_macd) and pd.notna(latest_macd_signal) and latest_macd > latest_macd_signal)
        macd_bearish = bool(pd.notna(latest_macd) and pd.notna(latest_macd_signal) and latest_macd < latest_macd_signal)
        
        # Check BUY conditions (convert all to native Python bool)
        buy_conditions = {
            "prev_candle_is_three_black_crows": bool(prev_candle_type == "Three Black Crows"),
            "current_candle_is_green": bool(is_green),
            "close_or_high_above_vwap": bool(close_above_vwap or (high > vwap if vwap else False)),
            "current_candle_matches_pattern": bool(any(p in candle_pattern for p in ['Dragonfly Doji', 'Piercing Pattern', 'Inverted Hammer', 'Long White Candle'])),
            "vwap_distance_ok": bool(vwap_diff_pct <= 2.0 if vwap_diff_pct else False),
            "rsi_ok": bool(latest_rsi is not None and pd.notna(latest_rsi) and latest_rsi <= 65 and latest_rsi >= 30),
            "macd_bullish": macd_bullish,
            "sufficient_data": bool(latest_idx >= 6)
        }
        
        # Check SELL conditions (convert all to native Python bool)
        sell_conditions = {
            "prev_candle_is_three_white_soldiers": bool(prev_candle_type == "Three White Soldiers"),
            "current_candle_is_red": bool(is_red),
            "close_or_low_below_vwap": bool(close_below_vwap or (low < vwap if vwap else False)),
            "current_candle_matches_pattern": bool(any(p in candle_pattern for p in ['Gravestone Doji', 'Dark Cloud Cover', 'Shooting Star', 'Long Black Candle'])),
            "vwap_distance_ok": bool(vwap_diff_pct <= 2.0 if vwap_diff_pct else False),
            "rsi_ok": bool(latest_rsi is not None and pd.notna(latest_rsi) and latest_rsi >= 35 and latest_rsi <= 70),
            "macd_bearish": macd_bearish,
            "sufficient_data": bool(latest_idx >= 6)
        }
        
        # Count met conditions
        buy_met = sum(1 for v in buy_conditions.values() if v)
        buy_total = len(buy_conditions)
        sell_met = sum(1 for v in sell_conditions.values() if v)
        sell_total = len(sell_conditions)
        
        validation_checks = {
            "buy_conditions_met": buy_met,
            "buy_conditions_total": buy_total,
            "sell_conditions_met": sell_met,
            "sell_conditions_total": sell_total,
            "buy_conditions": buy_conditions,
            "sell_conditions": sell_conditions,
            "indicators": {
                "vwap": round(vwap, 2) if vwap else None,
                "vwap_distance_pct": round(vwap_diff_pct, 2) if vwap_diff_pct else None,
                "rsi": round(float(latest_rsi), 2) if latest_rsi is not None and pd.notna(latest_rsi) else None,
                "macd_bullish": bool(macd_bullish),
                "macd_bearish": bool(macd_bearish)
            },
            "current_candle": {
                "pattern": candle_pattern,
                "prev_pattern": prev_candle_type,
                "is_green": bool(is_green),
                "is_red": bool(is_red),
                "close_above_vwap": bool(close_above_vwap),
                "close_below_vwap": bool(close_below_vwap)
            }
        }
        
        return {
            "candle_pattern": candle_pattern,
            "signal": signal,
            "signal_priority": priority,
            "signal_reason": reason,
            "validation_checks": validation_checks
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

