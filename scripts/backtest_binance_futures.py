#!/usr/bin/env python3
"""
Standalone Binance Futures Backtest Script

Usage:
    python backtest_binance_futures.py

Or with custom parameters:
    python backtest_binance_futures.py --config config.json

The script reads parameters from a JSON file or uses defaults.
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import httpx
from typing import List, Dict, Optional

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.binance_historical import fetch_historical_klines_for_date_range, convert_timeframe_to_binance
from utils.indicators import calculate_rsi
from utils.binance_backtest import (
    detect_candlestick_pattern,
    generate_trading_signal,
    process_backtest_data
)


# Default configuration
DEFAULT_CONFIG = {
    "startDateStr": "2025-12-03",
    "endDateStr": "2026-01-03",
    "timeframe": "5minute",
    "symbols": get_binance_symbols_from_env()  # Load from environment
}


# Pattern detection and signal generation are now imported from utils.binance_backtest

def generate_trading_signal_wrapper(current_idx: int, df: pd.DataFrame) -> tuple:
    """
    Generate trading signal - SCALPING strategy
    Focus: More frequent signals, quick profits, tighter stops
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
            # For Inverted Hammer, prefer if next candle confirms
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
        # Simple RSI check - not too oversold
        if current_idx >= 14:
            current_rsi = row.get('rsi', 50)
            if pd.notna(current_rsi):
                if current_rsi < 25:  # Too oversold
                    return (None, None, None)
                if current_rsi > 75:  # Too overbought, might be good for sell
                    pass  # Allow
        
        # Simple MACD check - prefer bearish or neutral
        if current_idx >= 26:
            current_macd = row.get('macd', 0)
            current_macd_signal = row.get('macd_signal', 0)
            if pd.notna(current_macd) and pd.notna(current_macd_signal):
                macd_bearish = current_macd < current_macd_signal
                macd_near_cross = abs(current_macd - current_macd_signal) < abs(current_macd) * 0.1
                if not (macd_bearish or macd_near_cross):
                    return (None, None, None)
        
        # Volume check
        current_volume = row.get('volume', 0)
        if current_idx >= 5:
            recent_volumes = [df.loc[current_idx - i].get('volume', current_volume) for i in range(5)]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else current_volume
            if avg_volume > 0 and current_volume < avg_volume * 0.5:
                return (None, None, None)
        
        # Momentum: Price should be moving down
        if current_idx >= 2:
            price_1_candle_ago = prev_row.get('close', close)
            price_2_candles_ago = prev_2_row.get('close', close) if prev_2_row is not None else close
            if close >= price_1_candle_ago and close >= price_2_candles_ago:
                return (None, None, None)
        
        matched_pattern = next((p for p in bearish_patterns if p in candle_type), candle_type)
        reason = f"Scalping SELL: {matched_pattern} below VWAP (VWAP dist: {vwap_diff_percent:.2f}%)"
        return ('SELL', 1, reason)
    
    return (None, None, None)


async def backtest_symbol(symbol: str, start_date: str, end_date: str, binance_interval: str) -> Dict:
    """Backtest a single symbol"""
    print(f"\n{'='*60}")
    print(f"Processing {symbol}...")
    print(f"{'='*60}")
    
    try:
        # Fetch historical klines
        print(f"Fetching historical data for {symbol}...")
        klines = await fetch_historical_klines_for_date_range(
            symbol,
            binance_interval,
            start_date,
            end_date
        )
        
        if not klines or len(klines) == 0:
            print(f"No data found for {symbol}")
            return {
                "instrument": symbol,
                "total_signals": 0,
                "total_profit": 0.0,
                "orders": []
            }
        
        print(f"Fetched {len(klines)} candles for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(klines)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate VWAP
        print("Calculating VWAP...")
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['cumulative_volume'] = df['volume'].cumsum()
        df['cumulative_tpv'] = (df['typical_price'] * df['volume']).cumsum()
        df['vwap'] = df['cumulative_tpv'] / df['cumulative_volume']
        
        # Calculate RSI
        print("Calculating RSI...")
        closes_list = df['close'].tolist()
        rsi_values = calculate_rsi(closes_list, period=14)
        if isinstance(rsi_values, list):
            if len(rsi_values) < len(df):
                rsi_values = [np.nan] * (len(df) - len(rsi_values)) + rsi_values
            df['rsi'] = rsi_values[:len(df)]
        else:
            df['rsi'] = np.nan
        
        # Calculate MACD
        print("Calculating MACD...")
        def calculate_macd_simple(prices, fast=12, slow=26, signal=9):
            if len(prices) < slow + signal:
                return [np.nan] * len(prices), [np.nan] * len(prices), [np.nan] * len(prices)
            
            prices_arr = np.array(prices, dtype=float)
            ema_fast = pd.Series(prices_arr).ewm(span=fast, adjust=False).mean()
            ema_slow = pd.Series(prices_arr).ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean()
            
            return macd_line.tolist(), signal_line.tolist(), (macd_line - signal_line).tolist()
        
        macd_line, macd_signal, macd_histogram = calculate_macd_simple(closes_list)
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
        
        # Detect candlestick patterns
        print("Detecting candlestick patterns...")
        df['candle_type'] = df.index.map(lambda i: detect_candlestick_pattern(i, df))
        
        # Generate signals
        print("Generating trading signals...")
        df['trading_signal'] = None
        df['signal_priority'] = None
        df['signal_reason'] = None
        
        for i in range(len(df)):
            signal, priority, reason = generate_trading_signal(i, df)
            if signal is not None:
                df.loc[i, 'trading_signal'] = signal
                df.loc[i, 'signal_priority'] = priority
                df.loc[i, 'signal_reason'] = reason
        
        # Calculate P&L using shared utility
        print("Calculating P&L for signals...")
        STOP_LOSS_PCT = 0.9
        TRAILING_STOP_PCT = 0.6
        PROFIT_TARGET_PCT = 0.7
        
        symbol_signals = process_backtest_data(
            df,
            stop_loss_pct=STOP_LOSS_PCT,
            trailing_stop_pct=TRAILING_STOP_PCT,
            profit_target_pct=PROFIT_TARGET_PCT,
            use_position_sizing=True
        )
        
        # Calculate summary from orders
        symbol_profit = sum(order['profit'] for order in symbol_signals)
        symbol_profitable = sum(1 for order in symbol_signals if order['profit'] > 0)
        symbol_losses = sum(1 for order in symbol_signals if order['profit'] < 0)
        
        # Calculate summary from orders
        if len(symbol_signals) > 0:
            avg_profit = symbol_profit / len(symbol_signals)
            win_rate = (symbol_profitable / len(symbol_signals)) * 100
        else:
            avg_profit = 0.0
            win_rate = 0.0
        
        result = {
            "instrument": symbol,
            "instrument_token": symbol,
            "total_signals": len(symbol_signals),
            "total_profit": round(symbol_profit, 2),
            "avg_profit": round(avg_profit, 2),
            "win_rate": round(win_rate, 2),
            "profitable_signals": symbol_profitable,
            "loss_signals": symbol_losses,
            "orders": symbol_signals
        }
        
        print(f"\n{symbol} Results:")
        print(f"  Total Signals: {result['total_signals']}")
        print(f"  Total Profit: ${result['total_profit']:.2f}")
        print(f"  Win Rate: {result['win_rate']:.2f}%")
        print(f"  Profitable: {result['profitable_signals']}, Losses: {result['loss_signals']}")
        
        return result
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "instrument": symbol,
            "total_signals": 0,
            "total_profit": 0.0,
            "orders": [],
            "error": str(e)
        }


async def main():
    """Main backtest function"""
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    # Check for config file argument
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        config_path = sys.argv[2] if len(sys.argv) > 2 else 'config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config.update(json.load(f))
        else:
            print(f"Config file {config_path} not found, using defaults")
    else:
        # Check for config.json in current directory
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config.update(json.load(f))
    
    start_date = config.get('startDateStr') or config.get('start_date')
    end_date = config.get('endDateStr') or config.get('end_date')
    timeframe = config.get('timeframe', '5minute')
    symbols = config.get('symbols', [])
    
    if not start_date or not end_date:
        print("Error: startDateStr and endDateStr are required")
        return
    
    if not symbols:
        print("Error: symbols list is required")
        return
    
    binance_interval = convert_timeframe_to_binance(timeframe)
    
    print("\n" + "="*60)
    print("BINANCE FUTURES BACKTEST")
    print("="*60)
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Timeframe: {timeframe} ({binance_interval})")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Total Symbols: {len(symbols)}")
    print("="*60)
    
    # Run backtest for all symbols
    results = []
    total_signals = 0
    total_profit = 0.0
    
    for symbol in symbols:
        result = await backtest_symbol(symbol, start_date, end_date, binance_interval)
        results.append(result)
        total_signals += result['total_signals']
        total_profit += result['total_profit']
    
    # Calculate summary
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    trading_days = (end_dt - start_dt).days + 1
    
    summary = {
        "total_instruments": len(symbols),
        "total_signals": total_signals,
        "total_profit": round(total_profit, 2),
        "avg_profit_per_signal": round(total_profit / total_signals, 2) if total_signals > 0 else 0.0,
        "profitable_instruments": len([r for r in results if r['total_profit'] > 0]),
        "loss_instruments": len([r for r in results if r['total_profit'] < 0]),
        "test_period": {
            "start_date": start_date,
            "end_date": end_date,
            "trading_days": trading_days
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)
    print(f"Total Instruments: {summary['total_instruments']}")
    print(f"Total Signals: {summary['total_signals']}")
    print(f"Total Profit: ${summary['total_profit']:.2f}")
    print(f"Avg Profit per Signal: ${summary['avg_profit_per_signal']:.2f}")
    print(f"Profitable Instruments: {summary['profitable_instruments']}")
    print(f"Loss Instruments: {summary['loss_instruments']}")
    print(f"Test Period: {start_date} to {end_date} ({trading_days} days)")
    print("="*60)
    
    # Save results to JSON file
    output_file = f"backtest_results_{start_date}_{end_date}.json"
    output_data = {
        "config": config,
        "summary": summary,
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    print("\nBacktest completed!")


if __name__ == "__main__":
    asyncio.run(main())

