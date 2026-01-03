"""
Market data API endpoints (candles, quotes, instruments, etc.)
"""
from fastapi import APIRouter, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import json

from utils.kite_utils import (
    api_key,
    get_access_token,
    get_kite_instance,
    calculate_trend_and_suggestions
)
from kiteconnect.exceptions import KiteException
from core.user_context import get_user_id_from_request
from core.responses import SuccessResponse, ErrorResponse
from core.dependencies import get_request_id
from utils.binance_client import fetch_klines, fetch_multiple_klines
from utils.binance_vwap import compute_vwap, compute_vwap_batch
from utils.binance_signals import analyze_symbol_for_signals
from utils.binance_commentary import get_commentary_generator
from utils.binance_commentary_service import get_commentary_service
from core.config import get_settings

def get_binance_symbols() -> list:
    """Get Binance symbols from environment configuration"""
    settings = get_settings()
    symbols = settings.binance_symbols
    if isinstance(symbols, str):
        # Handle comma-separated string
        symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif isinstance(symbols, list):
        # Ensure all symbols are uppercase
        symbols = [s.upper() if isinstance(s, str) else str(s).upper() for s in symbols]
    else:
        # Fallback to default
        symbols = ["1000PEPEUSDT"]
    return symbols
import httpx
import hmac
import hashlib
import time
from urllib.parse import urlencode

# Binance Futures SDK imports
try:
    from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
        DerivativesTradingUsdsFutures,
        ConfigurationRestAPI
    )
    BINANCE_SDK_AVAILABLE = True
except ImportError:
    BINANCE_SDK_AVAILABLE = False
    print("[Warning] binance-sdk-derivatives-trading-usds-futures not installed. Using manual API calls.")

router = APIRouter(prefix="/market", tags=["Market Data"])

# Binance Monitor Configuration
# BINANCE_SYMBOLS is now loaded from environment via get_binance_symbols()
BINANCE_TIMEFRAMES = ["1m", "5m", "15m", "30m"]
# Global storage for latest VWAP data (in-memory cache)
_latest_binance_vwap_data = {}

# Binance Futures API base URL
BINANCE_FUTURES_API = "https://fapi.binance.com"


@router.get("/candles/{instrument_token}/{interval}/{fromDate}/{toDate}")
def get_candle(instrument_token: str, interval: str, fromDate: str, toDate: str, request: Request = None):
    """Get historical candle data"""
    try:
        # Get user_id from request if available
        user_id = "default"
        if request:
            try:
                user_id = get_user_id_from_request(request)
            except:
                pass
        
        kite = get_kite_instance(user_id=user_id)
        
        # Convert interval format (Kite uses: minute, day, etc.)
        interval_map = {
            '1minute': 'minute',
            '5minute': '5minute',
            '15minute': '15minute',
            '30minute': '30minute',
            '60minute': '60minute',
            'day': 'day'
        }
        kite_interval = interval_map.get(interval, interval)
        
        # Parse dates - Kite expects date objects, not datetime objects
        from_date = datetime.strptime(fromDate, "%Y-%m-%d").date()
        to_date = datetime.strptime(toDate, "%Y-%m-%d").date()
        
        # Get historical data
        try:
            print(f"[get_candle] Calling historical_data with: token={int(instrument_token)}, from={from_date}, to={to_date}, interval={kite_interval}")
            historical_data = kite.historical_data(
                instrument_token=int(instrument_token),
                from_date=from_date,
                to_date=to_date,
                interval=kite_interval
            )
            print(f"[get_candle] Successfully retrieved {len(historical_data) if historical_data else 0} candles")
        except KiteException as e:
            error_msg = str(e)
            error_type = type(e).__name__
            print(f"[get_candle] KiteException ({error_type}): {error_msg}")
            print(f"[get_candle] User ID: {user_id}")
            print(f"[get_candle] Instrument: {instrument_token}, Interval: {interval} ({kite_interval}), Date: {fromDate} to {toDate}")
            print(f"[get_candle] From date object: {from_date}, To date object: {to_date}")
            
            # Get current API key and token for debugging
            from utils.kite_utils import get_kite_api_key, get_access_token
            current_api_key = get_kite_api_key(user_id=user_id)
            current_token = get_access_token()
            print(f"[get_candle] Current API Key: {current_api_key[:15] if current_api_key else 'NOT SET'}...")
            print(f"[get_candle] Current Token length: {len(current_token) if current_token else 0}")
            print(f"[get_candle] Full error: {repr(e)}")
            
            # Check if it's actually a token error or something else
            is_token_error = (
                error_type != "InputException" and
                any(keyword in error_msg.lower() for keyword in ["invalid", "expired", "token", "unauthorized", "authentication"])
            )
            
            # If it's InputException, it's likely an input validation issue, not token
            if error_type == "InputException":
                raise HTTPException(
                    status_code=400,
                    detail=f"Kite API input error: {error_msg}. "
                           f"This might be due to: 1) Invalid instrument token ({instrument_token}), "
                           f"2) Invalid date range ({fromDate} to {toDate} - check if dates are valid trading days), "
                           f"3) Invalid interval ({kite_interval}). "
                           f"Note: Markets are closed on weekends and holidays."
                )
            
            if is_token_error:
                raise HTTPException(
                    status_code=401,
                    detail=f"Kite API error: {error_msg} (Error type: {error_type}). "
                           "Possible causes: 1) Token was generated with a different API key than currently configured, "
                           "2) Token has expired (Kite tokens expire daily), or 3) API key was changed after token generation. "
                           f"Current API Key: {current_api_key[:10] if current_api_key else 'NOT SET'}... "
                           "Solution: Generate a new token using the current API key: "
                           "1) GET /api/v1/auth/google to get login URL, 2) Login through that URL, "
                           "3) POST /api/v1/auth/set-token with the request_token from redirect."
                )
            # For non-token errors, provide more context
            raise HTTPException(
                status_code=400, 
                detail=f"Kite API error ({error_type}): {error_msg}. "
                       f"Request: Instrument={instrument_token}, Interval={kite_interval}, From={fromDate}, To={toDate}"
            )
        
        # Convert to DataFrame for processing
        if historical_data:
            df = pd.DataFrame(historical_data)
            # Rename columns to match expected format
            df = df.rename(columns={
                'date': 'timestamp',
                'oi': 'openinterest'
            })
            
            # Ensure timestamp is datetime for VWAP calculation
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp (oldest first) for VWAP calculation
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate VWAP (Volume Weighted Average Price)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            price_volume = typical_price * df['volume']
            
            # Cumulative VWAP from start of session
            cumulative_price_volume = price_volume.cumsum()
            cumulative_volume = df['volume'].cumsum()
            df['vwap'] = cumulative_price_volume / cumulative_volume.replace(0, np.nan)
            df['vwap'] = df['vwap'].fillna(df['close'])  # Fill NaN with close price if volume is 0
            
            # Calculate VWAP position and differences
            df['is_above_vwap'] = df['close'] > df['vwap']
            df['vwap_position'] = df['is_above_vwap'].map({True: 'Above', False: 'Below'})
            df['vwap_diff'] = (df['close'] - df['vwap']).abs()
            df['vwap_diff_percent'] = (df['close'] - df['vwap']) / df['vwap'] * 100
            
            # Detect candlestick patterns (same logic as before)
            def detect_candlestick_pattern(current_idx, df):
                """Detect candlestick pattern types based on OHLC data"""
                row = df.loc[current_idx]
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
                
                prev_row = df.loc[current_idx - 1] if current_idx > 0 else None
                prev_prev_row = df.loc[current_idx - 2] if current_idx > 1 else None
                
                # Doji patterns
                if is_doji:
                    if upper_wick_ratio > 0.4 and lower_wick_ratio > 0.4:
                        return 'Doji'
                    elif upper_wick_ratio > 0.6:
                        return 'Gravestone Doji'
                    elif lower_wick_ratio > 0.6:
                        return 'Dragonfly Doji'
                    else:
                        return 'Doji'
                
                # Marubozu
                if upper_wick_ratio < 0.05 and lower_wick_ratio < 0.05:
                    if is_bullish:
                        return 'Bullish Marubozu'
                    else:
                        return 'Bearish Marubozu'
                
                # Hammer patterns
                if lower_wick_ratio > 0.6 and body_ratio < 0.3 and upper_wick_ratio < 0.2:
                    if is_bullish:
                        return 'Hammer'
                    else:
                        return 'Hanging Man'
                
                # Inverted Hammer / Shooting Star
                if upper_wick_ratio > 0.6 and body_ratio < 0.3 and lower_wick_ratio < 0.2:
                    if is_bullish:
                        return 'Inverted Hammer'
                    else:
                        return 'Shooting Star'
                
                # Spinning Top
                if body_ratio < 0.3 and upper_wick_ratio > 0.3 and lower_wick_ratio > 0.3:
                    return 'Spinning Top'
                
                # Engulfing patterns
                if prev_row is not None:
                    prev_open = prev_row['open']
                    prev_close = prev_row['close']
                    prev_body_size = abs(prev_close - prev_open)
                    
                    if (is_bullish and prev_close < prev_open and
                        close > prev_open and open_price < prev_close and
                        body_size > prev_body_size * 1.1):
                        return 'Bullish Engulfing'
                    
                    if (is_bearish and prev_close > prev_open and
                        close < prev_open and open_price > prev_close and
                        body_size > prev_body_size * 1.1):
                        return 'Bearish Engulfing'
                
                # Piercing Pattern / Dark Cloud Cover
                if prev_row is not None:
                    prev_open = prev_row['open']
                    prev_close = prev_row['close']
                    
                    if (is_bullish and prev_close < prev_open and
                        open_price < prev_close and
                        close > (prev_open + prev_close) / 2 and
                        close < prev_open):
                        return 'Piercing Pattern'
                    
                    if (is_bearish and prev_close > prev_open and
                        open_price > prev_close and
                        close < (prev_open + prev_close) / 2 and
                        close > prev_open):
                        return 'Dark Cloud Cover'
                
                # Harami patterns
                if prev_row is not None:
                    prev_open = prev_row['open']
                    prev_close = prev_row['close']
                    prev_body_size = abs(prev_close - prev_open)
                    
                    if (prev_close < prev_open and
                        open_price > prev_close and close < prev_open and
                        body_size < prev_body_size * 0.5):
                        return 'Bullish Harami'
                    
                    if (prev_close > prev_open and
                        open_price < prev_close and close > prev_open and
                        body_size < prev_body_size * 0.5):
                        return 'Bearish Harami'
                
                # Three White Soldiers / Three Black Crows
                if prev_row is not None and prev_prev_row is not None:
                    prev_open = prev_row['open']
                    prev_close = prev_row['close']
                    prev_prev_open = prev_prev_row['open']
                    prev_prev_close = prev_prev_row['close']
                    
                    if (is_bullish and prev_close > prev_open and prev_prev_close > prev_prev_open and
                        close > prev_close and prev_close > prev_prev_close):
                        return 'Three White Soldiers'
                    
                    if (is_bearish and prev_close < prev_open and prev_prev_close < prev_prev_open and
                        close < prev_close and prev_close < prev_prev_close):
                        return 'Three Black Crows'
                
                # Morning Star / Evening Star
                if prev_row is not None and prev_prev_row is not None:
                    prev_prev_open = prev_prev_row['open']
                    prev_prev_close = prev_prev_row['close']
                    prev_open = prev_row['open']
                    prev_close = prev_row['close']
                    prev_body_size = abs(prev_close - prev_open)
                    prev_prev_body_size = abs(prev_prev_close - prev_prev_open)
                    
                    if (is_bullish and prev_prev_close < prev_prev_open and
                        prev_body_size < prev_prev_body_size * 0.5 and
                        close > (prev_prev_open + prev_prev_close) / 2):
                        return 'Morning Star'
                    
                    if (is_bearish and prev_prev_close > prev_prev_open and
                        prev_body_size < prev_prev_body_size * 0.5 and
                        close < (prev_prev_open + prev_prev_close) / 2):
                        return 'Evening Star'
                
                # Default patterns
                if body_ratio > 0.7:
                    if is_bullish:
                        return 'Long White Candle'
                    else:
                        return 'Long Black Candle'
                elif body_ratio > 0.4:
                    if is_bullish:
                        return 'White Candle'
                    else:
                        return 'Black Candle'
                else:
                    if is_bullish:
                        return 'Small White Candle'
                    else:
                        return 'Small Black Candle'
            
            # Apply candlestick pattern detection
            df['candle_type'] = None
            for i in range(len(df)):
                df.loc[i, 'candle_type'] = detect_candlestick_pattern(i, df)
            
            # Detect VWAP reversal signals
            df['reversal_signal'] = None
            df['reversal_strength'] = 0.0
            
            for i in range(1, len(df)):
                prev_above = df.loc[i-1, 'is_above_vwap']
                curr_above = df.loc[i, 'is_above_vwap']
                current_reversal = None
                current_strength = 0.0
                
                # Detect crossover
                if prev_above != curr_above:
                    if not prev_above and curr_above:
                        current_reversal = 'Bullish Reversal'
                        prev_volume = df.loc[i-1, 'volume']
                        curr_volume = df.loc[i, 'volume']
                        volume_ratio = curr_volume / prev_volume if prev_volume > 0 else 1
                        price_distance = abs(df.loc[i, 'vwap_diff_percent'])
                        candle_body = abs(df.loc[i, 'close'] - df.loc[i, 'open'])
                        body_percent = (candle_body / df.loc[i, 'close'] * 100) if df.loc[i, 'close'] > 0 else 0
                        
                        current_strength = min(100, 
                            (30 if volume_ratio > 1 else 10) +
                            min(40, price_distance * 10) +
                            min(30, body_percent * 5)
                        )
                    elif prev_above and not curr_above:
                        current_reversal = 'Bearish Reversal'
                        prev_volume = df.loc[i-1, 'volume']
                        curr_volume = df.loc[i, 'volume']
                        volume_ratio = curr_volume / prev_volume if prev_volume > 0 else 1
                        price_distance = abs(df.loc[i, 'vwap_diff_percent'])
                        candle_body = abs(df.loc[i, 'close'] - df.loc[i, 'open'])
                        body_percent = (candle_body / df.loc[i, 'close'] * 100) if df.loc[i, 'close'] > 0 else 0
                        
                        current_strength = min(100,
                            (30 if volume_ratio > 1 else 10) +
                            min(40, price_distance * 10) +
                            min(30, body_percent * 5)
                        )
                
                if current_reversal:
                    df.loc[i, 'reversal_signal'] = current_reversal
                    df.loc[i, 'reversal_strength'] = current_strength
                
                # Confirm reversal
                if i > 1 and current_reversal is None:
                    prev_prev_above = df.loc[i-2, 'is_above_vwap']
                    prev_signal = df.loc[i-1, 'reversal_signal']
                    
                    if (prev_signal == 'Bullish Reversal' and 
                        not prev_prev_above and
                        df.loc[i-1, 'is_above_vwap'] and
                        curr_above):
                        df.loc[i-1, 'reversal_signal'] = 'Confirmed Bullish'
                    
                    if (prev_signal == 'Bearish Reversal' and 
                        prev_prev_above and
                        not df.loc[i-1, 'is_above_vwap'] and
                        not curr_above):
                        df.loc[i-1, 'reversal_signal'] = 'Confirmed Bearish'
            
            # Generate Buy/Sell signals (Priority 1 only)
            def generate_trading_signal(current_idx, df, instrument_token=None):
                instrument_blacklist = ['4701441']  # PERSISTENT
                if instrument_token and str(instrument_token) in instrument_blacklist:
                    return (None, None, None)
                
                row = df.loc[current_idx]
                candle_type = row.get('candle_type', '')
                close = row.get('close', 0)
                open_price = row.get('open', 0)
                high = row.get('high', 0)
                vwap = row.get('vwap', 0)
                
                if current_idx > 0:
                    prev_row = df.loc[current_idx - 1]
                    prev_candle_type = prev_row.get('candle_type', '')
                    is_green_candle = close > open_price
                    close_above_vwap = close > vwap
                    high_above_vwap = high > vwap
                    high_performance_candle_types = [
                        'Dragonfly Doji', 'Piercing Pattern',
                        'Inverted Hammer', 'Long White Candle'
                    ]
                    current_candle_matches = any(pattern in candle_type for pattern in high_performance_candle_types)
                    if (prev_candle_type == 'Three Black Crows' and is_green_candle and 
                        (close_above_vwap or high_above_vwap) and current_candle_matches):
                        matched_pattern = next((p for p in high_performance_candle_types if p in candle_type), candle_type)
                        reason = f"Priority 1: {matched_pattern} candle {'closing' if close_above_vwap else 'high'} above VWAP after Three Black Crows"
                        return ('BUY', 1, reason)
                return (None, None, None)
            
            # Apply trading signal generation
            df['trading_signal'] = None
            df['signal_priority'] = None
            df['signal_reason'] = None
            for i in range(len(df)):
                signal, priority, reason = generate_trading_signal(i, df, instrument_token=instrument_token)
                if signal is not None:
                    df.loc[i, 'trading_signal'] = signal
                    df.loc[i, 'signal_priority'] = priority
                    df.loc[i, 'signal_reason'] = reason
            
            # Convert all datetime columns to Unix timestamps
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = (df[col].astype('int64') // 10**9).astype(int)
                elif 'timestamp' in col.lower() and df[col].dtype == 'object':
                    try:
                        df[col] = (pd.to_datetime(df[col]).astype('int64') // 10**9).astype(int)
                    except:
                        pass
            
            # Convert DataFrame to dict and ensure all values are JSON serializable
            records = df.to_dict(orient='records')
            
            # Convert any remaining non-serializable types
            for record in records:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif isinstance(value, pd.Timestamp):
                        record[key] = int(value.timestamp())
                    elif isinstance(value, pd.Timedelta):
                        record[key] = int(value.total_seconds())
                    elif isinstance(value, (np.integer, np.floating)):
                        record[key] = value.item() if hasattr(value, 'item') else float(value) if isinstance(value, np.floating) else int(value)
                    elif isinstance(value, (np.bool_, bool)):
                        record[key] = bool(value)
                    elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                        record[key] = None
            
            return JSONResponse(content=records)
        else:
            return JSONResponse(content=[])
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting candles: {str(e)}")


@router.get("/resolve-instrument/{instrument_name}")
def resolve_instrument(instrument_name: str, exchange: str = "NSE"):
    """Resolve instrument name to instrument token"""
    try:
        from agent.tools.instrument_resolver import resolve_instrument_name
        result = resolve_instrument_name(instrument_name, exchange)
        if result:
            return {
                "instrument_name": instrument_name,
                "exchange": exchange,
                "instrument_token": result.get("instrument_token"),
                "tradingsymbol": result.get("tradingsymbol"),
                "name": result.get("name"),
                "instrument_type": result.get("instrument_type")
            }
        else:
            raise HTTPException(status_code=404, detail=f"Instrument '{instrument_name}' not found in {exchange}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resolving instrument: {str(e)}")


@router.get("/quote/{instrument_key}")
def get_quote(instrument_key: str):
    """Get quote for an instrument"""
    try:
        kite = get_kite_instance()
        # Parse instrument_key (format: EXCHANGE|TOKEN or just TOKEN)
        if '|' in instrument_key:
            exchange, token = instrument_key.split('|')
            quote = kite.quote(f"{exchange}:{token}")
        else:
            quote = kite.quote(f"NSE:{instrument_key}")
        return {"data": quote}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting quote: {str(e)}")


@router.get("/instruments")
def get_instruments():
    """Get all instruments (for option chain, etc.)"""
    try:
        kite = get_kite_instance()
        # Get instruments for NSE
        instruments = kite.instruments("NSE")
        return {"data": instruments}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting instruments: {str(e)}")


@router.get("/nifty50-options")
def get_nifty50_options():
    """Get Nifty50 options with current strike, 2 ITM and 2 OTM strikes"""
    try:
        kite = get_kite_instance()
        
        # Get Nifty50 current price (NIFTY 50 index)
        nifty_quote = kite.quote("NSE:NIFTY 50")
        nifty_price = nifty_quote.get("NSE:NIFTY 50", {}).get("last_price", 0)
        
        if not nifty_price:
            raise HTTPException(status_code=404, detail="Nifty50 price not found")
        
        # Round to nearest 50 for strike calculation
        current_strike = round(nifty_price / 50) * 50
        
        # Get all NFO instruments (options)
        all_instruments = kite.instruments("NFO")
        
        # Filter for Nifty50 options (NIFTY)
        nifty_options = [
            inst for inst in all_instruments 
            if inst.get("name") == "NIFTY" and inst.get("instrument_type") in ["CE", "PE"]
        ]
        
        # Get current expiry (nearest expiry)
        if not nifty_options:
            raise HTTPException(status_code=404, detail="Nifty50 options not found")
        
        # Group by expiry and get the nearest one
        expiries = sorted(set([inst.get("expiry") for inst in nifty_options if inst.get("expiry")]))
        if not expiries:
            raise HTTPException(status_code=404, detail="No valid expiries found")
        
        current_expiry = expiries[0]
        
        # Filter for current expiry
        current_expiry_options = [
            inst for inst in nifty_options 
            if inst.get("expiry") == current_expiry
        ]
        
        # Calculate strikes: current, 2 ITM, 2 OTM
        strikes_to_find = []
        strikes_to_find.append(current_strike)
        strikes_to_find.append(current_strike - 50)
        strikes_to_find.append(current_strike - 100)
        strikes_to_find.append(current_strike + 50)
        strikes_to_find.append(current_strike + 100)
        
        # Get instrument tokens for these strikes (both CE and PE)
        result_options = []
        for strike in strikes_to_find:
            for option_type in ["CE", "PE"]:
                option = next(
                    (inst for inst in current_expiry_options 
                     if inst.get("strike") == strike and inst.get("instrument_type") == option_type),
                    None
                )
                if option:
                    result_options.append({
                        "instrument_token": option.get("instrument_token"),
                        "tradingsymbol": option.get("tradingsymbol"),
                        "strike": strike,
                        "option_type": option_type,
                        "expiry": current_expiry,
                        "exchange": "NFO"
                    })
        
        # Get quotes for all these instruments
        instrument_keys = [f"NFO:{opt['tradingsymbol']}" for opt in result_options]
        if instrument_keys:
            try:
                quotes = kite.quote(instrument_keys)
                today = datetime.now().date()
                
                # Combine option data with quotes and trend analysis
                for opt in result_options:
                    quote_key = f"NFO:{opt['tradingsymbol']}"
                    quote_data = quotes.get(quote_key, {})
                    if quote_data:
                        opt["last_price"] = quote_data.get("last_price", 0)
                        depth = quote_data.get("depth", {})
                        buy_depth = depth.get("buy", [])
                        sell_depth = depth.get("sell", [])
                        opt["bid"] = buy_depth[0].get("price", 0) if buy_depth and len(buy_depth) > 0 else 0
                        opt["ask"] = sell_depth[0].get("price", 0) if sell_depth and len(sell_depth) > 0 else 0
                        opt["volume"] = quote_data.get("volume", 0)
                        opt["oi"] = quote_data.get("oi", 0)
                    else:
                        opt["last_price"] = 0
                        opt["bid"] = 0
                        opt["ask"] = 0
                        opt["volume"] = 0
                        opt["oi"] = 0
                    
                    # Calculate trend and suggestions
                    try:
                        trend_data = calculate_trend_and_suggestions(
                            kite, 
                            opt["instrument_token"], 
                            opt["last_price"]
                        )
                        opt["trend"] = trend_data.get("trend", "NEUTRAL")
                        opt["trend_strength"] = trend_data.get("trend_strength", 0)
                        opt["buy_price"] = trend_data.get("buy_price", opt["last_price"])
                        opt["sell_price"] = trend_data.get("sell_price", opt["last_price"])
                        opt["reason"] = trend_data.get("reason", "Trend analysis unavailable")
                    except Exception as e:
                        print(f"Error calculating trend for {opt['tradingsymbol']}: {e}")
                        opt["trend"] = "NEUTRAL"
                        opt["trend_strength"] = 0
                        opt["buy_price"] = opt["last_price"]
                        opt["sell_price"] = opt["last_price"]
                        opt["reason"] = f"Error in trend calculation: {str(e)}"
            except Exception as e:
                print(f"Error fetching quotes: {e}")
                for opt in result_options:
                    opt["last_price"] = 0
                    opt["bid"] = 0
                    opt["ask"] = 0
                    opt["volume"] = 0
                    opt["oi"] = 0
                    opt["trend"] = "NEUTRAL"
                    opt["trend_strength"] = 0
                    opt["buy_price"] = 0
                    opt["sell_price"] = 0
                    opt["reason"] = "Quote data unavailable"
        
        return {
            "data": {
                "nifty_price": nifty_price,
                "current_strike": current_strike,
                "expiry": current_expiry,
                "options": result_options
            }
        }
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Nifty50 options: {str(e)}")


@router.get("/ws-nifty50-options")
def get_nifty50_options_ws():
    """Get WebSocket URL for Nifty50 options streaming"""
    try:
        access_token = get_access_token()
        if not access_token:
            raise HTTPException(status_code=401, detail="Access token not found")
        
        # Construct Kite Connect WebSocket URL
        ws_url = f"wss://ws.kite.trade?api_key={api_key}&access_token={access_token}"
        
        return {
            "data": {
                "_authorized_redirect_uri": ws_url,
                "api_key": api_key,
                "access_token": access_token[:20] + "..." if access_token else None,
                "message": "WebSocket URL for Nifty50 options streaming"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting WebSocket info: {str(e)}")


# ============================================================================
# Binance Monitor Endpoints
# ============================================================================

async def update_binance_vwap_data():
    """Background task to update Binance VWAP data periodically"""
    global _latest_binance_vwap_data
    import traceback
    
    # Initial delay to let server fully start
    await asyncio.sleep(2)
    
    print("[Binance Monitor] Starting background VWAP update task...")
    
    while True:
        try:
            # Fetch klines for all symbols and timeframes
            binance_symbols = get_binance_symbols()
            klines_data = await fetch_multiple_klines(binance_symbols, BINANCE_TIMEFRAMES, limit=100)
            
            # Calculate VWAP for all
            vwap_data = compute_vwap_batch(klines_data)
            
            # Update global cache
            _latest_binance_vwap_data = vwap_data
            
            # Log update (first time and periodically)
            if len(_latest_binance_vwap_data) > 0:
                sample_symbol = list(_latest_binance_vwap_data.keys())[0]
                #print(f"[Binance Monitor] Updated VWAP data for {len(_latest_binance_vwap_data)} symbols (sample: {sample_symbol})")
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[Binance Monitor] Error updating VWAP data: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)  # Wait even on error to avoid tight loop


@router.get("/binance-futures/symbols")
async def get_binance_futures_symbols(request: Request):
    """
    Get configured Binance Futures symbols from environment
    
    Returns:
        List of symbols configured in BINANCE_SYMBOLS env variable
    """
    request_id = get_request_id(request)
    
    try:
        symbols = get_binance_symbols()
        return SuccessResponse(
            data={"symbols": symbols},
            message=f"Binance Futures symbols retrieved successfully",
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving symbols: {str(e)}"
        )


@router.get("/binance-vwap")
async def get_binance_vwap(request: Request, symbol: str = None):
    """
    Get current VWAP values for Binance symbols
    
    Args:
        symbol: Optional symbol filter (e.g., "ETHUSDT"). If not provided, returns all symbols.
    
    Returns:
        VWAP data for symbol(s) across all timeframes
    """
    request_id = get_request_id(request)
    
    try:
        if symbol:
            symbol = symbol.upper()
            if symbol in _latest_binance_vwap_data:
                return SuccessResponse(
                    data={symbol: _latest_binance_vwap_data[symbol]},
                    message=f"VWAP data for {symbol}",
                    request_id=request_id
                )
            else:
                return ErrorResponse(
                    message=f"Symbol {symbol} not found",
                    request_id=request_id
                )
        else:
            # Return all symbols
            return SuccessResponse(
                data=_latest_binance_vwap_data,
                message="VWAP data for all Binance symbols",
                request_id=request_id
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Binance VWAP: {str(e)}")


@router.websocket("/ws/binance-vwap")
async def binance_vwap_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time Binance VWAP streaming
    
    Server streams VWAP data every 5 seconds:
    {
        "ETHUSDT": {
            "1m": {"vwap": 2450.23, "current_price": 2455.45, "position": "Above"},
            "5m": {"vwap": 2452.12, "current_price": 2458.34, "position": "Above"},
            "15m": {"vwap": 89700.67, "current_price": 89650.12, "position": "Below"},
            "30m": {"vwap": 89720.45, "current_price": 89680.78, "position": "Below"}
        },
        "ETHUSDT": {
            "1m": {"vwap": 2450.90, "current_price": 2455.12, "position": "Above"},
            ...
        },
        ...
    }
    """
    await websocket.accept()
    print("[Binance WS] Client connected")
    
    try:
        # Wait for initial data to be available (max 10 seconds)
        wait_count = 0
        while not _latest_binance_vwap_data and wait_count < 20:
            await asyncio.sleep(0.5)
            wait_count += 1
        
        if not _latest_binance_vwap_data:
            print("[Binance WS] Warning: No VWAP data available yet, sending empty object")
            await websocket.send_json({})
        else:
            # Log sample data being sent
            sample_symbol = list(_latest_binance_vwap_data.keys())[0] if _latest_binance_vwap_data else None
            if sample_symbol:
                sample_data = _latest_binance_vwap_data[sample_symbol]
                print(f"[Binance WS] Sending initial VWAP data for {len(_latest_binance_vwap_data)} symbols")
                print(f"[Binance WS] Sample {sample_symbol}: {sample_data}")
            await websocket.send_json(_latest_binance_vwap_data)
        
        # Continue streaming updates
        while True:
            await asyncio.sleep(5)
            if _latest_binance_vwap_data:
                # Log sample data periodically (every 10th update = ~50 seconds)
                import random
                if random.randint(1, 10) == 1:
                    sample_symbol = list(_latest_binance_vwap_data.keys())[0] if _latest_binance_vwap_data else None
                    if sample_symbol:
                        sample_data = _latest_binance_vwap_data[sample_symbol]
                        print(f"[Binance WS] Update - Sample {sample_symbol}: {sample_data}")
                await websocket.send_json(_latest_binance_vwap_data)
            else:
                # Send empty object if data not available
                await websocket.send_json({})
                
    except WebSocketDisconnect:
        print("[Binance WS] Client disconnected")
    except Exception as e:
        print(f"[Binance WS] Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close()
        except:
            pass


@router.get("/binance-futures/live-prices")
async def get_binance_futures_live_prices(request: Request, symbols: str = None, include_signals: bool = False):
    """
    Get live prices for Binance Futures symbols
    
    Args:
        symbols: Comma-separated list of symbols (e.g., "1000PEPEUSDT,ETHUSDT"). 
                 If not provided, returns prices for symbols from BINANCE_SYMBOLS env variable.
        include_signals: If True, includes candle pattern and trading signals (slower).
    
    Returns:
        List of ticker data with symbol, price, 24h change, volume, etc.
    """
    request_id = get_request_id(request)
    
    try:
        # Determine which symbols to fetch
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
        else:
            symbol_list = get_binance_symbols()
        
        # Fetch 24h ticker statistics from Binance Futures API
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BINANCE_FUTURES_API}/fapi/v1/ticker/24hr")
            response.raise_for_status()
            all_tickers = response.json()
        
        # Filter to requested symbols and format response
        result = []
        for ticker in all_tickers:
            symbol = ticker.get('symbol', '')
            if symbol in symbol_list:
                ticker_data = {
                    "symbol": symbol,
                    "price": float(ticker.get('lastPrice', 0)),
                    "high_24h": float(ticker.get('highPrice', 0)),
                    "low_24h": float(ticker.get('lowPrice', 0)),
                    "volume_24h": float(ticker.get('volume', 0)),
                    "quote_volume_24h": float(ticker.get('quoteVolume', 0)),
                    "open_price": float(ticker.get('openPrice', 0)),
                    "prev_close_price": float(ticker.get('prevClosePrice', 0)),
                    "bid_price": float(ticker.get('bidPrice', 0)),
                    "ask_price": float(ticker.get('askPrice', 0)),
                    "count": int(ticker.get('count', 0)),
                    "timestamp": int(ticker.get('closeTime', 0)),
                    "candle_pattern": "N/A",
                    "signal": None,
                    "signal_priority": None,
                    "signal_reason": None,
                    "validation_checks": None
                }
                result.append(ticker_data)
        
        # Analyze signals if requested
        if include_signals:
            try:
                from utils.binance_client import fetch_klines
                # Fetch klines for all symbols concurrently
                klines_tasks = {ticker_data['symbol']: fetch_klines(ticker_data['symbol'], '5m', limit=200)  # 200 candles = ~16.7 hours for better signal accuracy 
                               for ticker_data in result}
                
                # Wait for all klines to be fetched
                klines_results = await asyncio.gather(*klines_tasks.values(), return_exceptions=True)
                
                # Map results back to symbols
                for ticker_data in result:
                    symbol = ticker_data['symbol']
                    task_idx = list(klines_tasks.keys()).index(symbol)
                    klines_result = klines_results[task_idx]
                    
                    try:
                        if isinstance(klines_result, Exception):
                            print(f"Error fetching klines for {symbol}: {klines_result}")
                            continue
                        
                        klines = klines_result
                        # Analyze signals
                        signal_data = analyze_symbol_for_signals(symbol, klines)
                        ticker_data['candle_pattern'] = signal_data.get('candle_pattern', 'N/A')
                        ticker_data['signal'] = signal_data.get('signal')
                        ticker_data['signal_priority'] = signal_data.get('signal_priority')
                        ticker_data['signal_reason'] = signal_data.get('signal_reason')
                        ticker_data['validation_checks'] = signal_data.get('validation_checks')
                    except Exception as e:
                        print(f"Error analyzing signals for {symbol}: {e}")
                        # Keep default values
            except Exception as e:
                print(f"Error in signal analysis: {e}")
                # Continue without signals if there's an error
        
        # Sort by symbol
        result.sort(key=lambda x: x['symbol'])
        
        return SuccessResponse(
            data=result,
            message=f"Live prices for {len(result)} Binance Futures symbols",
            request_id=request_id
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Binance API error: {str(e)}"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching Binance Futures prices: {str(e)}"
        )


@router.get("/binance-futures/validate-signals")
async def validate_binance_signals(
    symbol: str = "ETHUSDT",
    request: Request = None
):
    """
    Validation endpoint to debug signal generation
    
    Returns detailed analysis of why signals are/aren't being generated for a symbol
    """
    request_id = get_request_id(request)
    
    try:
        from utils.binance_client import fetch_klines
        import pandas as pd
        
        # Fetch klines
        klines = await fetch_klines(symbol.upper(), '5m', limit=200)
        
        if not klines or len(klines) < 30:
            return SuccessResponse(
                data={
                    "error": f"Insufficient data: {len(klines) if klines else 0} candles (need at least 30)"
                },
                message="Signal validation failed",
                request_id=request_id
            )
        
        # Analyze signals
        signal_data = analyze_symbol_for_signals(symbol, klines)
        
        # Get detailed analysis
        df = pd.DataFrame(klines)
        latest_idx = len(df) - 1
        latest_row = df.iloc[latest_idx]
        prev_row = df.iloc[latest_idx - 1] if latest_idx > 0 else None
        
        # Calculate indicators for display
        from utils.binance_vwap import compute_vwap
        from utils.indicators import calculate_rsi
        
        vwap = compute_vwap(klines)
        closes = df['close'].tolist()
        rsi_values = calculate_rsi(closes, period=14)
        latest_rsi = rsi_values[-1] if isinstance(rsi_values, list) and len(rsi_values) > 0 else None
        
        # Calculate MACD
        macd_info = {}
        if len(closes) >= 35:
            series = pd.Series(closes)
            ema_fast = series.ewm(span=12, adjust=False).mean()
            ema_slow = series.ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_info = {
                "macd": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
                "macd_signal": float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
                "macd_bullish": float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]) if not (pd.isna(macd_line.iloc[-1]) or pd.isna(signal_line.iloc[-1])) else None
            }
        
        # Detect patterns for all candles to get prev_candle_type
        from utils.binance_signals import detect_candlestick_pattern
        df['candle_type'] = df.index.map(lambda i: detect_candlestick_pattern(i, df))
        
        # Check signal conditions
        current_candle_type = signal_data.get('candle_pattern', 'N/A')
        prev_candle_type = df.iloc[latest_idx - 1]['candle_type'] if latest_idx > 0 else 'N/A'
        
        close = float(latest_row.get('close', 0))
        open_price = float(latest_row.get('open', 0))
        high = float(latest_row.get('high', 0))
        low = float(latest_row.get('low', 0))
        
        is_green = close > open_price
        is_red = close < open_price
        close_above_vwap = close > vwap if vwap else False
        close_below_vwap = close < vwap if vwap else False
        vwap_diff_pct = abs(close - vwap) / vwap * 100 if vwap and vwap > 0 else None
        
        # Check BUY conditions
        buy_conditions = {
            "prev_candle_is_three_black_crows": prev_candle_type == "Three Black Crows",
            "current_candle_is_green": is_green,
            "close_or_high_above_vwap": close_above_vwap or high > vwap if vwap else False,
            "current_candle_matches_pattern": any(p in current_candle_type for p in ['Dragonfly Doji', 'Piercing Pattern', 'Inverted Hammer', 'Long White Candle']),
            "vwap_distance_ok": vwap_diff_pct <= 2.0 if vwap_diff_pct else False,
            "rsi_ok": latest_rsi is not None and latest_rsi <= 65 and latest_rsi >= 30,
            "macd_bullish": macd_info.get("macd_bullish", False) if macd_info else False,
            "sufficient_data": latest_idx >= 6
        }
        
        # Check SELL conditions
        sell_conditions = {
            "prev_candle_is_three_white_soldiers": prev_candle_type == "Three White Soldiers",
            "current_candle_is_red": is_red,
            "close_or_low_below_vwap": close_below_vwap or low < vwap if vwap else False,
            "current_candle_matches_pattern": any(p in current_candle_type for p in ['Gravestone Doji', 'Dark Cloud Cover', 'Shooting Star', 'Long Black Candle']),
            "vwap_distance_ok": vwap_diff_pct <= 2.0 if vwap_diff_pct else False,
            "rsi_ok": latest_rsi is not None and latest_rsi >= 35 and latest_rsi <= 70,
            "macd_bearish": macd_info.get("macd_bullish", False) == False if macd_info else False,
            "sufficient_data": latest_idx >= 6
        }
        
        validation_result = {
            "symbol": symbol.upper(),
            "signal_generated": signal_data.get('signal'),
            "signal_priority": signal_data.get('signal_priority'),
            "signal_reason": signal_data.get('signal_reason'),
            "current_candle_pattern": current_candle_type,
            "previous_candle_pattern": prev_candle_type,
            "indicators": {
                "vwap": round(vwap, 2) if vwap else None,
                "current_price": round(close, 2),
                "vwap_distance_percent": round(vwap_diff_pct, 2) if vwap_diff_pct else None,
                "rsi": round(latest_rsi, 2) if latest_rsi else None,
                "macd": macd_info
            },
            "current_candle": {
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "is_green": is_green,
                "is_red": is_red,
                "close_above_vwap": close_above_vwap,
                "close_below_vwap": close_below_vwap
            },
            "buy_conditions": buy_conditions,
            "sell_conditions": sell_conditions,
            "buy_conditions_met": all(buy_conditions.values()),
            "sell_conditions_met": all(sell_conditions.values()),
            "data_points": len(klines),
            "latest_candle_index": latest_idx
        }
        
        return SuccessResponse(
            data=validation_result,
            message=f"Signal validation for {symbol.upper()}",
            request_id=request_id
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error validating signals: {str(e)}"
        )


@router.get("/binance-futures/balance")
async def get_binance_futures_balance(request: Request):
    """
    Get Binance Futures wallet balance using official Binance SDK
    
    Returns:
        Account balance information including USDT and other assets
    """
    request_id = get_request_id(request)
    
    try:
        settings = get_settings()
        api_key = settings.binance_api_key
        api_secret = settings.binance_api_secret
        
        if not api_key or not api_secret:
            raise HTTPException(
                status_code=400,
                detail="Binance API key and secret not configured. Please set BINANCE_API_KEY and BINANCE_API_SECRET in .env file"
            )
        
        # Use official Binance SDK if available, otherwise fallback to manual API calls
        if BINANCE_SDK_AVAILABLE:
            try:
                # Initialize Binance Futures client using official SDK
                # Reference: https://github.com/binance/binance-connector-python/blob/master/clients/derivatives_trading_usds_futures/docs/migration_guide_derivatives_trading_usds_futures_sdk.md
                configuration = ConfigurationRestAPI(
                    api_key=api_key,
                    api_secret=api_secret
                )
                client = DerivativesTradingUsdsFutures(config_rest_api=configuration)
                
                # Get account balance using official SDK
                # Reference: https://binance-docs.github.io/apidocs/futures/en/#futures-account-balance-v2
                # The endpoint is /fapi/v2/balance which returns list of asset balances
                # SDK method name may vary, try common patterns
                balance_data = []
                try:
                    # Try common SDK method names for balance endpoint
                    if hasattr(client.rest_api, 'account_balance_v2'):
                        balance_response = client.rest_api.account_balance_v2()
                    elif hasattr(client.rest_api, 'balance'):
                        balance_response = client.rest_api.balance()
                    elif hasattr(client.rest_api, 'futures_account_balance'):
                        balance_response = client.rest_api.futures_account_balance()
                    else:
                        # Fallback: use account information and extract assets
                        balance_response = client.rest_api.account_information_v2()
                        if isinstance(balance_response, dict) and 'assets' in balance_response:
                            balance_data = balance_response['assets']
                        else:
                            raise AttributeError("Balance method not found")
                    
                    # Handle different response formats from SDK
                    if isinstance(balance_response, list):
                        balance_data = balance_response
                    elif hasattr(balance_response, 'data'):
                        balance_data = balance_response.data if isinstance(balance_response.data, list) else []
                    elif isinstance(balance_response, dict):
                        balance_data = balance_response.get('data', balance_response.get('assets', []))
                        if not isinstance(balance_data, list):
                            balance_data = []
                except (AttributeError, KeyError) as e:
                    print(f"[Binance SDK] Method not found or unexpected response format: {e}")
                    raise  # Re-raise to trigger fallback
                
            except Exception as sdk_error:
                print(f"[Binance SDK] Error using official SDK, falling back to manual API: {sdk_error}")
                # Fallback to manual API call
                balance_data = await _fetch_balance_manual(api_key, api_secret)
        else:
            # Use manual API call if SDK not available
            balance_data = await _fetch_balance_manual(api_key, api_secret)
        
        # Filter and format balance data (show only assets with balance > 0)
        formatted_balances = []
        total_usdt_balance = 0.0
        
        for asset in balance_data:
            balance = float(asset.get('balance', 0))
            available = float(asset.get('availableBalance', 0))
            asset_name = asset.get('asset', '')
            
            if balance > 0 or asset_name == 'USDT':  # Always show USDT
                formatted_balances.append({
                    "asset": asset_name,
                    "balance": round(balance, 8),
                    "available_balance": round(available, 8),
                    "cross_wallet_balance": round(float(asset.get('crossWalletBalance', 0)), 8),
                    "cross_unrealized_pnl": round(float(asset.get('crossUnPnl', 0)), 8),
                    "max_withdraw_amount": round(float(asset.get('maxWithdrawAmount', 0)), 8),
                    "margin_available": asset.get('marginAvailable', False),
                    "update_time": asset.get('updateTime', 0)
                })
                
                if asset_name == 'USDT':
                    total_usdt_balance = balance
        
        return SuccessResponse(
            data={
                "balances": formatted_balances,
                "total_usdt_balance": round(total_usdt_balance, 2),
                "timestamp": int(time.time() * 1000)
            },
            message="Binance Futures wallet balance retrieved successfully",
            request_id=request_id
        )
        
    except httpx.HTTPStatusError as e:
        error_detail = f"Binance API error: {e.response.status_code}"
        error_code = None
        try:
            error_body = e.response.json()
            error_detail = error_body.get('msg', error_detail)
            error_code = error_body.get('code')
            
            # Provide helpful messages for common errors
            if error_code == -1022:
                error_detail = f"Signature validation failed. Please verify: 1) API key and secret are correct, 2) API key has Futures trading permissions, 3) IP address is whitelisted (if IP restriction is enabled). Original error: {error_body.get('msg', 'Unknown')}"
            elif error_code == -2015:
                error_detail = f"Invalid API key or permissions. Please check your API key has Futures trading enabled. Original error: {error_body.get('msg', 'Unknown')}"
            elif error_code == -1021:
                error_detail = f"Timestamp out of sync. Please check your system time. Original error: {error_body.get('msg', 'Unknown')}"
        except:
            pass
        
        raise HTTPException(
            status_code=e.response.status_code,
            detail=error_detail
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching Binance Futures balance: {str(e)}"
        )


async def _fetch_balance_manual(api_key: str, api_secret: str) -> list:
    """
    Fallback function to fetch balance using manual API calls
    Used when official SDK is not available or fails
    
    Reference: https://binance-docs.github.io/apidocs/futures/en/#futures-account-balance-v2
    """
    # Create signature for authenticated request
    # Binance requires: timestamp parameter and signature in query string
    # recvWindow helps with time synchronization issues (5000ms = 5 seconds tolerance)
    timestamp = int(time.time() * 1000)
    recv_window = 5000  # 5 seconds tolerance for timestamp
    
    # Build query string (must be in alphabetical order for signature)
    query_params = {
        "recvWindow": recv_window,
        "timestamp": timestamp
    }
    query_string = urlencode(query_params, doseq=True, safe='')
    
    # Generate signature: HMAC SHA256 of query string
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Add signature to query string
    final_query = f"{query_string}&signature={signature}"
    
    # Make authenticated request to Binance Futures API
    url = f"{BINANCE_FUTURES_API}/fapi/v2/balance?{final_query}"
    headers = {
        "X-MBX-APIKEY": api_key
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=10.0)
        
        # Check for signature errors
        if response.status_code == 400:
            try:
                error_body = response.json()
                if error_body.get('code') == -1022:
                    print(f"[Binance API] Signature error. Query: {query_string}, Signature: {signature[:20]}...")
                    print(f"[Binance API] API Key: {api_key[:10]}..., Secret length: {len(api_secret)}")
            except:
                pass
        
        response.raise_for_status()
        return response.json()


@router.websocket("/ws/binance-futures/live-prices")
async def binance_futures_live_prices_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time Binance Futures price streaming using WebSocket ticker stream
    
    Client can optionally send: {"symbols": ["ETHUSDT", "SOLUSDT"]}
    Server streams price updates in real-time from Binance WebSocket:
    [
        {
            "symbol": "ETHUSDT",
            "price": 2450.23,
            "price_change_24h": 45.67,
            "price_change_percent_24h": 1.42,
            "high_24h": 90200.00,
            "low_24h": 88400.00,
            "volume_24h": 1234567.89,
            "candle_pattern": "Bullish Pattern",
            "signal": "BUY",
            ...
        },
        ...
    ]
    """
    await websocket.accept()
    print("[Binance Futures WS] Client connected")
    
    try:
        # Receive optional symbols list from client
        symbols_to_fetch = set(get_binance_symbols())
        try:
            message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            params = json.loads(message)
            if 'symbols' in params and params['symbols']:
                symbols_to_fetch = {s.upper() for s in params['symbols']}
        except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
            # Use default symbols if no message received or invalid
            pass
        
        print(f"[Binance Futures WS] Streaming prices for {len(symbols_to_fetch)} symbols: {list(symbols_to_fetch)}")
        
        # Start global ticker listener if not already running
        from utils.binance_websocket_ticker import start_global_ticker_listener, get_all_latest_tickers
        listener = await start_global_ticker_listener(symbols_filter=symbols_to_fetch)
        
        # Track last update time for signal analysis and cache signal results
        import time
        last_signal_update = {}
        signal_cache = {}  # Cache signal results: {symbol: {candle_pattern, signal, signal_priority, signal_reason}}
        SIGNAL_UPDATE_INTERVAL = 30  # Update signals every 30 seconds
        
        # Analyze signals immediately for all symbols on first connection
        async def analyze_signals_for_symbols(symbols: set):
            """Analyze signals for multiple symbols concurrently"""
            from utils.binance_client import fetch_klines
            tasks = []
            symbol_list = list(symbols)
            
            for symbol in symbol_list:
                task = fetch_klines(symbol, '5m', limit=200)  # 200 candles = ~16.7 hours for better signal accuracy
                tasks.append((symbol, task))
            
            # Fetch all klines concurrently
            klines_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Analyze signals for each symbol
            for (symbol, _), klines_result in zip(tasks, klines_results):
                try:
                    if isinstance(klines_result, Exception):
                        print(f"Error fetching klines for {symbol}: {klines_result}")
                        continue
                    
                    klines = klines_result
                    signal_data = analyze_symbol_for_signals(symbol, klines)
                    signal_cache[symbol] = {
                        'candle_pattern': signal_data.get('candle_pattern', 'N/A'),
                        'signal': signal_data.get('signal'),
                        'signal_priority': signal_data.get('signal_priority'),
                        'signal_reason': signal_data.get('signal_reason'),
                        'validation_checks': signal_data.get('validation_checks')
                    }
                    last_signal_update[symbol] = time.time()
                except Exception as e:
                    print(f"Error analyzing signals for {symbol}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Initial signal analysis
        print("[Binance Futures WS] Analyzing signals for all symbols on initial connection...")
        await analyze_signals_for_symbols(symbols_to_fetch)
        print("[Binance Futures WS] Initial signal analysis complete")
        
        # Stream updates from WebSocket ticker listener
        while True:
            try:
                # Check if WebSocket is still open
                if websocket.client_state.name != 'CONNECTED':
                    print("[Binance Futures WS] WebSocket not connected, breaking loop")
                    break
                
                # Get latest ticker data from WebSocket listener
                all_tickers = get_all_latest_tickers()
                
                # Filter to requested symbols
                result = []
                current_time = time.time()
                
                # Check which symbols need signal updates
                symbols_to_update = set()
                for symbol in symbols_to_fetch:
                    should_update = (
                        symbol not in last_signal_update or 
                        (current_time - last_signal_update[symbol]) >= SIGNAL_UPDATE_INTERVAL
                    )
                    if should_update:
                        symbols_to_update.add(symbol)
                
                # Update signals for symbols that need it
                if symbols_to_update:
                    await analyze_signals_for_symbols(symbols_to_update)
                
                # Build result with cached signals
                for symbol in symbols_to_fetch:
                    ticker_data = all_tickers.get(symbol)
                    if not ticker_data:
                        continue
                    
                    # Copy ticker data
                    ticker_result = ticker_data.copy()
                    
                    # Add cached signal data
                    if symbol in signal_cache:
                        ticker_result['candle_pattern'] = signal_cache[symbol].get('candle_pattern', 'N/A')
                        ticker_result['signal'] = signal_cache[symbol].get('signal')
                        ticker_result['signal_priority'] = signal_cache[symbol].get('signal_priority')
                        ticker_result['signal_reason'] = signal_cache[symbol].get('signal_reason')
                        ticker_result['validation_checks'] = signal_cache[symbol].get('validation_checks')
                    else:
                        # Initialize with defaults if no cache
                        ticker_result['candle_pattern'] = "N/A"
                        ticker_result['signal'] = None
                        ticker_result['signal_priority'] = None
                        ticker_result['signal_reason'] = None
                        ticker_result['validation_checks'] = None
                    
                    # Remove 24h change fields
                    ticker_result.pop('price_change_24h', None)
                    ticker_result.pop('price_change_percent_24h', None)
                    
                    result.append(ticker_result)
                
                # Sort by symbol
                result.sort(key=lambda x: x['symbol'])
                
                # Check WebSocket state before sending
                if websocket.client_state.name != 'CONNECTED':
                    print("[Binance Futures WS] WebSocket disconnected before send, breaking loop")
                    break
                
                # Send to client
                try:
                    await websocket.send_json(result)
                except (RuntimeError, WebSocketDisconnect) as e:
                    if isinstance(e, WebSocketDisconnect):
                        print("[Binance Futures WS] WebSocket disconnected during send, breaking loop")
                        break
                    elif 'close message' in str(e).lower():
                        print("[Binance Futures WS] WebSocket closed, breaking loop")
                        break
                    raise
                
                # Wait 2 seconds before next update (WebSocket provides real-time updates)
                await asyncio.sleep(2)
                
            except (RuntimeError, WebSocketDisconnect) as e:
                if isinstance(e, WebSocketDisconnect):
                    print("[Binance Futures WS] WebSocket disconnected, breaking loop")
                    break
                elif 'close message' in str(e).lower():
                    print("[Binance Futures WS] WebSocket closed, breaking loop")
                    break
                raise
            except Exception as e:
                print(f"[Binance Futures WS] Error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if websocket.client_state.name == 'CONNECTED':
                        await websocket.send_json({
                            "error": f"Error: {str(e)}"
                        })
                except (RuntimeError, WebSocketDisconnect):
                    print("[Binance Futures WS] WebSocket closed during error send, breaking loop")
                    break
                await asyncio.sleep(2)  # Wait before retry
                
    except WebSocketDisconnect:
        print("[Binance Futures WS] Client disconnected")
    except Exception as e:
        print(f"[Binance Futures WS] Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close()
        except:
            pass


@router.websocket("/ws/binance-futures/commentary")
async def binance_futures_commentary_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for live trading commentary
    
    Sends historical commentary first (from last 100 candles), then streams new commentary
    when new 5-minute candles are formed.
    
    Client can optionally send: {"symbols": ["ETHUSDT", "SOLUSDT"]}
    Server streams commentary messages:
    [
        {
            "timestamp": 1704123456789,
            "symbol": "1000PEPEUSDT",
            "event_type": "new_candle|pattern_detected|signal_generated|...",
            "priority": "high|medium|low",
            "message": "Human-readable message",
            "details": {...}
        },
        ...
    ]
    """
    await websocket.accept()
    print("[Binance Commentary WS] Client connected")
    
    try:
        # Receive optional symbols list from client
        symbols_to_monitor = set(get_binance_symbols())
        try:
            message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            params = json.loads(message)
            if 'symbols' in params and params['symbols']:
                symbols_to_monitor = {s.upper() for s in params['symbols']}
        except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
            # Use default symbols if no message received or invalid
            pass
        
        print(f"[Binance Commentary WS] Client subscribed to {len(symbols_to_monitor)} symbols: {list(symbols_to_monitor)}")
        
        # Get commentary service
        commentary_service = get_commentary_service()
        
        # Send historical commentary first (filtered by symbols if specified)
        historical = commentary_service.get_recent_commentary(count=100)
        if symbols_to_monitor and 'all' not in symbols_to_monitor:
            # Filter by requested symbols
            historical = [msg for msg in historical if msg.get('symbol') in symbols_to_monitor]
        
        if historical:
            # Sort by timestamp (oldest first)
            historical.sort(key=lambda x: x.get('timestamp', 0))
            try:
                await websocket.send_json(historical)
                print(f"[Binance Commentary WS] Sent {len(historical)} historical commentary messages")
            except RuntimeError as e:
                if 'close message' in str(e).lower():
                    print("[Binance Commentary WS] WebSocket closed during historical send")
                    return
                raise
        
        # Start global ticker listener if not already running
        from utils.binance_websocket_ticker import start_global_ticker_listener, get_all_latest_tickers
        listener = await start_global_ticker_listener(symbols_filter=symbols_to_monitor)
        
        # Track last update time for signal analysis
        import time
        last_signal_update = {}
        signal_cache = {}  # {symbol: full_signal_data_dict}
        SIGNAL_UPDATE_INTERVAL = 30  # Update signals every 30 seconds
        
        # Helper to analyze signals for symbols
        async def analyze_signals_for_symbols(symbols: set):
            """Analyze signals for multiple symbols concurrently"""
            from utils.binance_client import fetch_klines
            tasks = []
            symbol_list = list(symbols)
            
            for symbol in symbol_list:
                task = fetch_klines(symbol, '5m', limit=200)
                tasks.append((symbol, task))
            
            klines_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (symbol, _), klines_result in zip(tasks, klines_results):
                try:
                    if isinstance(klines_result, Exception):
                        print(f"Error fetching klines for {symbol}: {klines_result}")
                        continue
                    
                    klines = klines_result
                    signal_data = analyze_symbol_for_signals(symbol, klines)
                    signal_cache[symbol] = signal_data
                    last_signal_update[symbol] = time.time()
                except Exception as e:
                    print(f"Error analyzing signals for {symbol}: {e}")
        
        # Initial signal analysis
        await analyze_signals_for_symbols(symbols_to_monitor)
        
        # Stream new commentary updates
        while True:
            try:
                # Check if WebSocket is still open
                if websocket.client_state.name != 'CONNECTED':
                    print("[Binance Commentary WS] WebSocket not connected, breaking loop")
                    break
                
                # Get latest ticker data
                all_tickers = get_all_latest_tickers()
                current_time = time.time()
                
                # Check which symbols need signal updates
                symbols_to_update = set()
                for symbol in symbols_to_monitor:
                    should_update = (
                        symbol not in last_signal_update or 
                        (current_time - last_signal_update[symbol]) >= SIGNAL_UPDATE_INTERVAL
                    )
                    if should_update:
                        symbols_to_update.add(symbol)
                
                # Update signals for symbols that need it
                if symbols_to_update:
                    await analyze_signals_for_symbols(symbols_to_update)
                
                # Collect new commentary messages
                all_messages = []
                
                for symbol in symbols_to_monitor:
                    ticker_data = all_tickers.get(symbol)
                    if not ticker_data:
                        continue
                    
                    # Get signal data from cache
                    signal_data = signal_cache.get(symbol, {})
                    
                    # Process new candle through commentary service
                    new_messages = await commentary_service.process_new_candle(
                        symbol,
                        ticker_data,
                        signal_data
                    )
                    
                    if new_messages:
                        all_messages.extend(new_messages)
                
                # Send new commentary messages if any
                if all_messages:
                    # Sort by timestamp (oldest first)
                    all_messages.sort(key=lambda x: x.get('timestamp', 0))
                    
                    # Check WebSocket state before sending
                    if websocket.client_state.name != 'CONNECTED':
                        print("[Binance Commentary WS] WebSocket disconnected before send, breaking loop")
                        break
                    
                    try:
                        await websocket.send_json(all_messages)
                        print(f"[Binance Commentary WS] Sent {len(all_messages)} new commentary messages")
                    except RuntimeError as e:
                        if 'close message' in str(e).lower():
                            print("[Binance Commentary WS] WebSocket closed, breaking loop")
                            break
                        raise
                
                # Wait 10 seconds before checking for new candles again
                await asyncio.sleep(10)
                
            except RuntimeError as e:
                if 'close message' in str(e).lower():
                    print("[Binance Commentary WS] WebSocket closed, breaking loop")
                    break
                raise
            except Exception as e:
                print(f"[Binance Commentary WS] Error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if websocket.client_state.name == 'CONNECTED':
                        await websocket.send_json({
                            "error": f"Error: {str(e)}"
                        })
                except RuntimeError:
                    print("[Binance Commentary WS] WebSocket closed during error send, breaking loop")
                    break
                await asyncio.sleep(10)  # Wait before retry
                
    except WebSocketDisconnect:
        print("[Binance Commentary WS] Client disconnected")
    except Exception as e:
        print(f"[Binance Commentary WS] Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close()
        except:
            pass

