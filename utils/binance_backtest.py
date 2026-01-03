"""
Shared utility for Binance Futures backtesting
Used by both standalone script and WebSocket endpoint
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.binance_historical import fetch_historical_klines_for_date_range, convert_timeframe_to_binance
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
    Generate trading signal - SCALPING strategy (refined)
    Focus: More frequent signals, quick profits, tighter stops
    
    Returns:
        (signal, priority, reason) where signal is 'BUY', 'SELL', or None
    """
    if current_idx < 3:
        return (None, None, None)
    
    row = df.loc[current_idx]
    candle_type = row.get('candle_type', '')
    close = float(row.get('close', 0))
    open_price = float(row.get('open', 0))
    high = float(row.get('high', 0))
    low = float(row.get('low', 0))
    vwap = float(row.get('vwap', 0))
    
    if vwap == 0 or close == 0:
        return (None, None, None)
    
    prev_row = df.loc[current_idx - 1]
    prev_2_row = df.loc[current_idx - 2] if current_idx >= 2 else None
    
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
        # Trend filter: Check if price is above 20-period EMA (avoid counter-trend)
        if current_idx >= 20:
            closes = df['close'].tolist()
            ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean()
            if close < ema20.iloc[-1]:  # Price below EMA20 suggests downtrend
                return (None, None, None)
        
        # VWAP trend: VWAP should be rising (for better entries)
        if current_idx >= 5:
            prev_vwap = df.loc[current_idx - 1].get('vwap', vwap)
            if vwap < prev_vwap:  # VWAP declining - wait for better entry
                return (None, None, None)
        
        # Fine-tuned RSI check - prefer 45-65 range for better win rate
        if current_idx >= 14:
            current_rsi = row.get('rsi', 50)
            prev_rsi = prev_row.get('rsi', 50)
            if pd.notna(current_rsi) and pd.notna(prev_rsi):
                if current_rsi > 65:  # Too overbought (tighter)
                    return (None, None, None)
                # Prefer RSI between 45-65 for better entries (narrower range)
                if current_rsi < 35:  # Too oversold, might be weak (tighter)
                    return (None, None, None)
                # Prefer RSI rising (momentum building) - stricter
                if current_rsi < 45 and current_rsi <= prev_rsi:
                    return (None, None, None)
                # Avoid RSI declining when already high
                if current_rsi > 60 and current_rsi < prev_rsi:
                    return (None, None, None)
        
        # Stricter MACD check - require clear bullish signal
        if current_idx >= 26:
            current_macd = row.get('macd', 0)
            current_macd_signal = row.get('macd_signal', 0)
            prev_macd = prev_row.get('macd', 0)
            prev_macd_signal = prev_row.get('macd_signal', 0)
            if pd.notna(current_macd) and pd.notna(current_macd_signal):
                macd_bullish = current_macd > current_macd_signal
                macd_crossing = (prev_macd <= prev_macd_signal) and (current_macd > current_macd_signal)
                # Require MACD to be clearly bullish (not just near cross)
                if not (macd_bullish or macd_crossing):
                    return (None, None, None)
                # If MACD is negative, require it to be crossing up (stronger signal)
                if current_macd < 0 and not macd_crossing:
                    return (None, None, None)
        
        # Fine-tuned volume check - require at least 80% of average for better quality
        current_volume = row.get('volume', 0)
        if current_idx >= 5:
            recent_volumes = [df.loc[current_idx - i].get('volume', current_volume) for i in range(5)]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else current_volume
            if avg_volume > 0 and current_volume < avg_volume * 0.8:  # At least 80% of average (tighter)
                return (None, None, None)
            # Prefer above-average volume for stronger signals
            if avg_volume > 0 and current_volume < avg_volume * 1.0 and 'Inverted Hammer' in candle_type:
                return (None, None, None)  # Weak patterns need above-average volume
        
        # Stronger momentum: Price should be clearly moving up
        if current_idx >= 2:
            price_1_candle_ago = prev_row.get('close', close)
            price_2_candles_ago = prev_2_row.get('close', close) if prev_2_row is not None else close
            # Require price to be higher than at least 2 of the last 3 candles
            if close <= price_1_candle_ago:
                if close <= price_2_candles_ago:
                    return (None, None, None)
        
        # Pattern-specific confirmations
        if 'Inverted Hammer' in candle_type:
            if current_idx < len(df) - 1:
                next_row = df.loc[current_idx + 1]
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
        return ('BUY', 1, reason)
    
    return (None, None, None)


def calculate_position_size(df: pd.DataFrame, entry_idx: int, entry_price: float) -> float:
    """Calculate position size based on volatility (ATR)"""
    if entry_idx < 14:
        return 1.0  # Default position size if not enough data
    
    # Calculate ATR for volatility measurement
    true_ranges = []
    for i in range(max(0, entry_idx - 14), entry_idx):
        if i > 0:
            high = float(df.iloc[i].get('high', entry_price))
            low = float(df.iloc[i].get('low', entry_price))
            prev_close = float(df.iloc[i-1].get('close', entry_price))
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
    
    if not true_ranges:
        return 1.0
    
    atr = sum(true_ranges) / len(true_ranges)
    atr_pct = (atr / entry_price) * 100 if entry_price > 0 else 0
    
    # Position sizing: Higher volatility = smaller position size
    # If ATR% > 2%, reduce position size proportionally
    if atr_pct > 2.0:
        position_multiplier = max(0.5, 2.0 / atr_pct)  # Reduce position for high volatility
    else:
        position_multiplier = 1.0  # Full position for normal volatility
    
    return position_multiplier


def calculate_exit_price(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    stop_loss_pct: float = 0.9,
    trailing_stop_pct: float = 0.6,
    profit_target_pct: float = 0.7
) -> Tuple[float, int, str]:
    """
    Calculate exit price, exit index, and exit reason
    
    Returns:
        (exit_price, exit_idx, exit_reason)
    """
    current_price = float(df.iloc[-1]['close']) if len(df) > 0 else entry_price
    exit_price = current_price
    exit_idx = len(df) - 1
    exit_reason = "End of period"
    
    stop_loss_price = entry_price * (1 - stop_loss_pct / 100.0)
    profit_target_price = entry_price * (1 + profit_target_pct / 100.0)
    highest_after_entry = entry_price
    
    for check_idx in range(entry_idx + 1, len(df)):
        check_row = df.loc[check_idx]
        check_close = float(check_row.get('close', entry_price))
        check_high = float(check_row.get('high', entry_price))
        check_low = float(check_row.get('low', entry_price))
        
        if check_close > highest_after_entry:
            highest_after_entry = check_close
        
        # SCALPING: Profit target hit - take quick profit
        if check_high >= profit_target_price:
            exit_price = profit_target_price
            exit_idx = check_idx
            exit_reason = f"Profit target hit ({profit_target_pct}%)"
            break
        
        # Stop-loss check
        if check_low <= stop_loss_price:
            exit_price = stop_loss_price
            exit_idx = check_idx
            exit_reason = f"Stop-loss triggered ({stop_loss_pct}%)"
            break
        
        # Trailing stop (tighter for scalping)
        if highest_after_entry > entry_price:
            trailing_stop_price = highest_after_entry * (1 - trailing_stop_pct / 100.0)
            if check_close <= trailing_stop_price:
                exit_price = check_close
                exit_idx = check_idx
                exit_reason = f"Trailing stop triggered ({trailing_stop_pct}% from high)"
                break
        
        # Quick exit if price moves against us (less aggressive for better win rate)
        if check_idx >= entry_idx + 3:  # Wait at least 3 candles
            price_drop_pct = (entry_price - check_close) / entry_price * 100
            # Exit early if price drops >0.7% (less aggressive than 0.5%)
            if price_drop_pct > 0.7:
                # Also check if next candle confirms weakness
                if check_idx < len(df) - 1:
                    next_row = df.loc[check_idx + 1]
                    next_close = float(next_row.get('close', check_close))
                    if next_close < check_close:
                        exit_price = check_close
                        exit_idx = check_idx
                        exit_reason = "Quick exit: Price moving against trade"
                        break
                else:
                    exit_price = check_close
                    exit_idx = check_idx
                    exit_reason = "Quick exit: Price moving against trade"
                    break
    
    return exit_price, exit_idx, exit_reason


def process_backtest_data(
    df: pd.DataFrame,
    stop_loss_pct: float = 0.9,
    trailing_stop_pct: float = 0.6,
    profit_target_pct: float = 0.7,
    use_position_sizing: bool = True
) -> List[Dict]:
    """
    Process backtest data and calculate P&L for all signals
    
    Args:
        df: DataFrame with OHLCV, indicators, and signals
        stop_loss_pct: Stop loss percentage
        trailing_stop_pct: Trailing stop percentage
        profit_target_pct: Profit target percentage
        use_position_sizing: Whether to apply position sizing based on volatility
    
    Returns:
        List of order dictionaries with P&L information
    """
    # Extract Priority 1 signals
    priority1_rows = df[(df['trading_signal'] == 'BUY') & (df['signal_priority'] == 1)]
    orders = []
    
    for idx, signal_row in priority1_rows.iterrows():
        try:
            entry_price = float(signal_row['close'])
        except (ValueError, TypeError):
            entry_price = float(str(signal_row['close']))
        
        entry_idx = idx
        
        # Calculate position size based on volatility at entry
        position_multiplier = calculate_position_size(df, entry_idx, entry_price) if use_position_sizing else 1.0
        
        # Calculate exit
        exit_price, exit_idx, exit_reason = calculate_exit_price(
            df, entry_idx, entry_price, stop_loss_pct, trailing_stop_pct, profit_target_pct
        )
        
        try:
            exit_price_float = float(exit_price)
        except (ValueError, TypeError):
            exit_price_float = float(str(exit_price))
        
        profit = (exit_price_float - entry_price) * position_multiplier  # Apply position sizing
        
        # Format timestamps
        entry_timestamp = signal_row['timestamp']
        try:
            if isinstance(entry_timestamp, (datetime, pd.Timestamp)):
                entry_time = entry_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                timestamp_val = int(entry_timestamp.timestamp())
            else:
                entry_time = pd.to_datetime(entry_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                timestamp_val = int(pd.to_datetime(entry_timestamp).timestamp())
        except:
            entry_time = str(entry_timestamp) if entry_timestamp else '-'
            timestamp_val = 0
        
        exit_timestamp = df.iloc[exit_idx]['timestamp']
        try:
            if isinstance(exit_timestamp, (datetime, pd.Timestamp)):
                exit_time = exit_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                exit_time = pd.to_datetime(exit_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except:
            exit_time = str(exit_timestamp) if exit_timestamp else '-'
        
        orders.append({
            "date": entry_timestamp.strftime('%Y-%m-%d') if isinstance(entry_timestamp, (datetime, pd.Timestamp)) else str(entry_timestamp)[:10],
            "timestamp": timestamp_val,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price_float, 2),
            "qty": round(position_multiplier, 3),
            "profit": round(profit, 2),
            "profit_percent": round((profit / entry_price * 100) if entry_price > 0 else 0, 2),
            "candle_type": str(signal_row.get('candle_type', '')),
            "signal_reason": str(signal_row.get('signal_reason', '')),
            "exit_reason": exit_reason,
            "position_size": round(position_multiplier, 3)
        })
    
    return orders

