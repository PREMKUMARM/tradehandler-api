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
from binance.client import Client
from binance.exceptions import BinanceAPIException

router = APIRouter(prefix="/market", tags=["Market Data"])

# Binance Monitor Configuration
BINANCE_SYMBOLS = ["1000PEPEUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "MATICUSDT"]
BINANCE_TIMEFRAMES = ["1m", "5m", "15m", "30m"]
# Global storage for latest VWAP data (in-memory cache)
_latest_binance_vwap_data = {}

# Initialize Binance client (no API keys needed for public data)
_binance_client = None

def get_binance_client():
    """Get or create Binance client instance"""
    global _binance_client
    if _binance_client is None:
        _binance_client = Client()  # No API keys needed for public data
    return _binance_client


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
            klines_data = await fetch_multiple_klines(BINANCE_SYMBOLS, BINANCE_TIMEFRAMES, limit=100)
            
            # Calculate VWAP for all
            vwap_data = compute_vwap_batch(klines_data)
            
            # Update global cache
            _latest_binance_vwap_data = vwap_data
            
            # Log update (first time and periodically)
            if len(_latest_binance_vwap_data) > 0:
                sample_symbol = list(_latest_binance_vwap_data.keys())[0]
                print(f"[Binance Monitor] Updated VWAP data for {len(_latest_binance_vwap_data)} symbols (sample: {sample_symbol})")
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[Binance Monitor] Error updating VWAP data: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)  # Wait even on error to avoid tight loop


@router.get("/binance-vwap")
async def get_binance_vwap(request: Request, symbol: str = None):
    """
    Get current VWAP values for Binance symbols
    
    Args:
        symbol: Optional symbol filter (e.g., "BTCUSDT"). If not provided, returns all symbols.
    
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
        "BTCUSDT": {
            "1m": {"vwap": 89650.23, "current_price": 89680.45, "position": "Above"},
            "5m": {"vwap": 89680.12, "current_price": 89690.34, "position": "Above"},
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
async def get_binance_futures_live_prices(request: Request, symbols: str = None):
    """
    Get live prices for Binance Futures symbols
    
    Args:
        symbols: Comma-separated list of symbols (e.g., "BTCUSDT,ETHUSDT"). 
                 If not provided, returns prices for default BINANCE_SYMBOLS.
    
    Returns:
        List of ticker data with symbol, price, 24h change, volume, etc.
    """
    request_id = get_request_id(request)
    
    try:
        client = get_binance_client()
        
        # Determine which symbols to fetch
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
        else:
            symbol_list = BINANCE_SYMBOLS
        
        # Fetch 24h ticker statistics for all symbols
        tickers = client.futures_24hr_ticker()
        
        # Filter to requested symbols and format response
        result = []
        for ticker in tickers:
            symbol = ticker.get('symbol', '')
            if symbol in symbol_list:
                result.append({
                    "symbol": symbol,
                    "price": float(ticker.get('lastPrice', 0)),
                    "price_change_24h": float(ticker.get('priceChange', 0)),
                    "price_change_percent_24h": float(ticker.get('priceChangePercent', 0)),
                    "high_24h": float(ticker.get('highPrice', 0)),
                    "low_24h": float(ticker.get('lowPrice', 0)),
                    "volume_24h": float(ticker.get('volume', 0)),
                    "quote_volume_24h": float(ticker.get('quoteVolume', 0)),
                    "open_price": float(ticker.get('openPrice', 0)),
                    "prev_close_price": float(ticker.get('prevClosePrice', 0)),
                    "bid_price": float(ticker.get('bidPrice', 0)),
                    "ask_price": float(ticker.get('askPrice', 0)),
                    "count": int(ticker.get('count', 0)),
                    "timestamp": int(ticker.get('closeTime', 0))
                })
        
        # Sort by symbol
        result.sort(key=lambda x: x['symbol'])
        
        return SuccessResponse(
            data=result,
            message=f"Live prices for {len(result)} Binance Futures symbols",
            request_id=request_id
        )
    except BinanceAPIException as e:
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


@router.websocket("/ws/binance-futures/live-prices")
async def binance_futures_live_prices_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time Binance Futures price streaming
    
    Client can optionally send: {"symbols": ["BTCUSDT", "ETHUSDT"]}
    Server streams price updates every 2 seconds:
    [
        {
            "symbol": "BTCUSDT",
            "price": 89650.23,
            "price_change_24h": 1250.45,
            "price_change_percent_24h": 1.42,
            "high_24h": 90200.00,
            "low_24h": 88400.00,
            "volume_24h": 1234567.89,
            ...
        },
        ...
    ]
    """
    await websocket.accept()
    print("[Binance Futures WS] Client connected")
    
    try:
        client = get_binance_client()
        
        # Receive optional symbols list from client
        symbols_to_fetch = BINANCE_SYMBOLS
        try:
            message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            params = json.loads(message)
            if 'symbols' in params and params['symbols']:
                symbols_to_fetch = [s.upper() for s in params['symbols']]
        except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
            # Use default symbols if no message received or invalid
            pass
        
        print(f"[Binance Futures WS] Streaming prices for {len(symbols_to_fetch)} symbols: {symbols_to_fetch}")
        
        # Stream updates every 2 seconds
        while True:
            try:
                # Fetch 24h ticker statistics
                tickers = client.futures_24hr_ticker()
                
                # Filter to requested symbols and format response
                result = []
                for ticker in tickers:
                    symbol = ticker.get('symbol', '')
                    if symbol in symbols_to_fetch:
                        result.append({
                            "symbol": symbol,
                            "price": float(ticker.get('lastPrice', 0)),
                            "price_change_24h": float(ticker.get('priceChange', 0)),
                            "price_change_percent_24h": float(ticker.get('priceChangePercent', 0)),
                            "high_24h": float(ticker.get('highPrice', 0)),
                            "low_24h": float(ticker.get('lowPrice', 0)),
                            "volume_24h": float(ticker.get('volume', 0)),
                            "quote_volume_24h": float(ticker.get('quoteVolume', 0)),
                            "open_price": float(ticker.get('openPrice', 0)),
                            "prev_close_price": float(ticker.get('prevClosePrice', 0)),
                            "bid_price": float(ticker.get('bidPrice', 0)),
                            "ask_price": float(ticker.get('askPrice', 0)),
                            "count": int(ticker.get('count', 0)),
                            "timestamp": int(ticker.get('closeTime', 0))
                        })
                
                # Sort by symbol
                result.sort(key=lambda x: x['symbol'])
                
                # Send to client
                await websocket.send_json(result)
                
                # Wait 2 seconds before next update
                await asyncio.sleep(2)
                
            except BinanceAPIException as e:
                print(f"[Binance Futures WS] Binance API error: {e}")
                await websocket.send_json({
                    "error": f"Binance API error: {str(e)}"
                })
                await asyncio.sleep(5)  # Wait longer on error
            except Exception as e:
                print(f"[Binance Futures WS] Error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)  # Wait longer on error
                
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

