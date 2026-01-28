"""
Market data API endpoints (candles, quotes, instruments, etc.)
"""
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import json
import time

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
from core.exceptions import ValidationError, NotFoundError, ExternalAPIError, AlgoFeastException
from utils.logger import log_info, log_error, log_debug, log_warning
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


@router.get("/server-time")
async def get_server_time(request: Request):
    """
    Simple endpoint to return current server time.
    Used by frontend to show a live server clock.
    """
    try:
        now = datetime.now()
        return SuccessResponse(
            data={
                "server_time_iso": now.isoformat(),
                "server_timestamp": int(now.timestamp())
            },
            request_id=get_request_id(request)
        )
    except Exception as e:
        log_error(f"Error getting server time: {e}")
        return ErrorResponse(
            message=f"Error getting server time: {str(e)}",
            request_id=get_request_id(request)
        )


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
        
        # Check if date is in the future
        today = datetime.now().date()
        if from_date > today or to_date > today:
            raise ValidationError(
                message=f"Cannot fetch data for future dates. Selected date: {fromDate} to {toDate}, Today: {today.strftime('%Y-%m-%d')}. "
                       f"Please select a date on or before today.",
                field="date"
            )
        
        # Get historical data
        try:
            log_debug(f"[get_candle] Calling historical_data with: token={int(instrument_token)}, from={from_date}, to={to_date}, interval={kite_interval}")
            historical_data = kite.historical_data(
                instrument_token=int(instrument_token),
                from_date=from_date,
                to_date=to_date,
                interval=kite_interval
            )
            log_debug(f"[get_candle] Successfully retrieved {len(historical_data) if historical_data else 0} candles")
        except KiteException as e:
            error_msg = str(e)
            error_type = type(e).__name__
            log_error(f"[get_candle] KiteException ({error_type}): {error_msg}")
            log_debug(f"[get_candle] User ID: {user_id}")
            log_debug(f"[get_candle] Instrument: {instrument_token}, Interval: {interval} ({kite_interval}), Date: {fromDate} to {toDate}")
            log_debug(f"[get_candle] From date object: {from_date}, To date object: {to_date}")
            
            # Get current API key and token for debugging
            from utils.kite_utils import get_kite_api_key, get_access_token
            current_api_key = get_kite_api_key(user_id=user_id)
            current_token = get_access_token()
            log_debug(f"[get_candle] Current API Key: {current_api_key[:15] if current_api_key else 'NOT SET'}...")
            log_debug(f"[get_candle] Current Token length: {len(current_token) if current_token else 0}")
            log_debug(f"[get_candle] Full error: {repr(e)}")
            
            # Check if it's actually a token error or something else
            is_token_error = (
                error_type != "InputException" and
                any(keyword in error_msg.lower() for keyword in ["invalid", "expired", "token", "unauthorized", "authentication"])
            )
            
            # If it's InputException, it's likely an input validation issue, not token
            if error_type == "InputException":
                raise ValidationError(
                    message=f"Kite API input error: {error_msg}. "
                           f"This might be due to: 1) Invalid instrument token ({instrument_token}), "
                           f"2) Invalid date range ({fromDate} to {toDate} - check if dates are valid trading days), "
                           f"3) Invalid interval ({kite_interval}). "
                           f"Note: Markets are closed on weekends and holidays.",
                    field="instrument_token"
                )
            
            if is_token_error:
                from core.exceptions import AuthenticationError
                raise AuthenticationError(
                    message=f"Kite API error: {error_msg} (Error type: {error_type}). "
                           "Possible causes: 1) Token was generated with a different API key than currently configured, "
                           "2) Token has expired (Kite tokens expire daily), or 3) API key was changed after token generation. "
                           f"Current API Key: {current_api_key[:10] if current_api_key else 'NOT SET'}... "
                           "Solution: Generate a new token using the current API key: "
                           "1) GET /api/v1/auth/google to get login URL, 2) Login through that URL, "
                           "3) POST /api/v1/auth/set-token with the request_token from redirect."
                )
            # Check if it's a permission error for indices
            if error_type == "PermissionException" or "permission" in error_msg.lower():
                # Try to determine if this is an index
                try:
                    instruments = kite.instruments("NSE")
                    inst_info = next((inst for inst in instruments if str(inst.get("instrument_token")) == str(instrument_token)), None)
                    if inst_info and (inst_info.get("segment") == "INDICES" or inst_info.get("instrument_type") == "INDEX"):
                        raise ValidationError(
                            message=f"Kite API permission error: {error_msg}. "
                                   f"Index data ({inst_info.get('tradingsymbol', 'N/A')}) may require special API permissions or subscription. "
                                   f"Please check: 1) Your Kite API key has market data permissions, "
                                   f"2) The date {fromDate} is a valid trading day (not a holiday/weekend), "
                                   f"3) Try selecting a different date or use a stock instead of an index.",
                            field="instrument_token"
                        )
                except AlgoFeastException:
                    raise
                except:
                    pass
            
            # For non-token errors, provide more context
            raise ExternalAPIError(
                message=f"Kite API error ({error_type}): {error_msg}. "
                       f"Request: Instrument={instrument_token}, Interval={kite_interval}, From={fromDate}, To={toDate}. "
                       f"Note: If this is an index, ensure your API key has market data permissions and the date is a valid trading day.",
                service="Kite Connect"
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
        log_error(f"Kite API error getting candles: {str(e)}")
        raise ExternalAPIError(
            message=str(e),
            service="Kite Connect"
        )
    except AlgoFeastException:
        raise
    except Exception as e:
        log_error(f"Error getting candles: {str(e)}")
        raise AlgoFeastException(
            message=f"Error getting candles: {str(e)}",
            status_code=500,
            error_code="INTERNAL_ERROR"
        )


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
            raise NotFoundError(
                resource="Instrument",
                identifier=f"{instrument_name} in {exchange}"
            )
    except AlgoFeastException:
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
def get_instruments(exchange: str = "NSE"):
    """Get all instruments (for option chain, etc.)
    
    Args:
        exchange: Exchange to get instruments from (NSE, NFO, BSE, MCX, etc.)
    """
    try:
        kite = get_kite_instance()
        # Get instruments for specified exchange
        instruments = kite.instruments(exchange.upper())
        
        # Filter for Nifty options if NFO exchange
        if exchange.upper() == "NFO":
            # Get a sample of Nifty options to understand structure
            nifty_options = [
                inst for inst in instruments 
                if inst.get("name") == "NIFTY" and inst.get("instrument_type") in ["CE", "PE"]
            ]
            # Return first 10 Nifty options as sample
            return {
                "data": nifty_options[:10] if len(nifty_options) > 10 else nifty_options,
                "total": len(nifty_options),
                "sample_count": min(10, len(nifty_options))
            }
        
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
# Kite Ticker WebSocket Endpoints
# ============================================================================

@router.websocket("/ws/kite-ticker")
async def kite_ticker_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time Kite ticker data streaming
    
    Server streams ticker updates in real-time from Kite WebSocket:
    [
        {
            "instrument_token": 256265,
            "last_price": 24500.50,
            "ohlc": {"open": 24400, "high": 24550, "low": 24350, "close": 24500},
            "volume": 1234567,
            "timestamp": 1234567890,
            ...
        },
        ...
    ]
    """
    try:
        await websocket.accept()
        log_info("[Kite Ticker WS] Client connected")
    except Exception as e:
        log_error(f"[Kite Ticker WS] Error accepting connection: {e}")
        return
    
    try:
        from utils.kite_websocket_ticker import get_kite_ticker_instance, get_all_latest_kite_ticks
        from concurrent.futures import ThreadPoolExecutor
        
        # Get ticker instance (non-blocking check)
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        def get_ticker():
            return get_kite_ticker_instance()
        
        ticker = await loop.run_in_executor(executor, get_ticker)
        
        if not ticker:
            log_warning("[Kite Ticker WS] Ticker not initialized - sending info message")
            try:
                # Send error message to frontend
                await websocket.send_json({
                    "message": "Kite ticker not initialized. Market may be closed or access token not configured.",
                    "error": "Ticker not initialized",
                    "is_initialized": False
                })
                # Keep connection open briefly so frontend can receive the message
                await asyncio.sleep(1)
            except Exception as e:
                log_error(f"[Kite Ticker WS] Error sending message: {e}")
            finally:
                # Close connection gracefully
                try:
                    if websocket.client_state.name == 'CONNECTED':
                        await websocket.close()
                except:
                    pass
            return
        
        # Send initial data (non-blocking)
        try:
            from concurrent.futures import ThreadPoolExecutor
            from datetime import datetime, date
            import json
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            
            def serialize_datetime(obj):
                """Recursively serialize datetime objects to ISO format strings"""
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, date):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: serialize_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_datetime(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    # For any other type, try to convert to string
                    return str(obj)
            
            def get_initial_ticks():
                return get_all_latest_kite_ticks()
            
            initial_ticks = await loop.run_in_executor(executor, get_initial_ticks)
            if initial_ticks:
                # Get first 30-minute and 5-minute candles, and calculate strategy
                def get_candles_and_strategy():
                    from utils.kite_utils import get_kite_instance
                    from datetime import timedelta
                    kite = get_kite_instance()
                    today_date = datetime.now().date()
                    
                    # Get previous day closes for gap calculation
                    prev_date = today_date - timedelta(days=1)
                    while prev_date.weekday() > 4:  # Skip weekends
                        prev_date = prev_date - timedelta(days=1)
                    
                    candles_30min = {}
                    breaking_5min = {}
                    previous_closes = {}
                    strategy_statuses = {}
                    
                    for token in initial_ticks.keys():
                        candles_30min[token] = get_first_30min_candle_ohlc(kite, token, today_date)
                        # Get breaking 5min candle (requires 30min candle data)
                        breaking_5min[token] = get_breaking_5min_candle_ohlc(kite, token, today_date, candles_30min.get(token))
                        
                        # Get previous close
                        try:
                            hist_data = kite.historical_data(
                                instrument_token=token,
                                from_date=prev_date,
                                to_date=prev_date + timedelta(days=1),
                                interval="day"
                            )
                            if hist_data and len(hist_data) > 0:
                                prev_close = hist_data[-1].get('close', 0)
                                if prev_close and prev_close > 0:
                                    previous_closes[token] = float(prev_close)
                        except:
                            pass
                        
                        # Calculate gap
                        gap_value = None
                        previous_close = previous_closes.get(token)
                        tick_data = initial_ticks.get(token, {})
                        if previous_close and tick_data.get('ohlc', {}).get('open'):
                            gap_value = tick_data.get('ohlc', {}).get('open') - previous_close
                        
                        # Calculate strategy status
                        strategy_statuses[token] = calculate_strategy_status(
                            gap_value=gap_value,
                            first_30min_ohlc=candles_30min.get(token),
                            breaking_5min_ohlc=breaking_5min.get(token),
                            previous_close=previous_close
                        )
                    
                    return candles_30min, breaking_5min, strategy_statuses
                
                first_30min_candles, breaking_5min_candles, strategy_statuses = await loop.run_in_executor(executor, get_candles_and_strategy)
                
                # Format and serialize tick data (remove unwanted fields)
                formatted_ticks = []
                for token, tick_data in initial_ticks.items():
                    formatted_tick = {
                        "instrument_token": token,
                        "last_price": tick_data.get('last_price', 0),
                        "ohlc": tick_data.get('ohlc', {}),
                        "timestamp": tick_data.get('timestamp', 0),
                        "change": tick_data.get('change', 0),
                        "depth": tick_data.get('depth', {}),
                        "first_30min_ohlc": first_30min_candles.get(token),
                        "first_5min_ohlc": breaking_5min_candles.get(token),
                        "strategy_status": strategy_statuses.get(token)
                    }
                    # Serialize datetime objects before sending
                    formatted_tick = serialize_datetime(formatted_tick)
                    formatted_ticks.append(formatted_tick)
                await websocket.send_json(formatted_ticks)
        except Exception as e:
            log_error(f"[Kite Ticker WS] Error getting initial ticks: {e}")
            import traceback
            traceback.print_exc()
        
        # Stream updates - send ticks when available, status updates periodically
        last_sent_time = {}
        last_status_sent = 0
        last_ticker_status = None
        
        while True:
            try:
                # Check if WebSocket is still open
                if websocket.client_state.name != 'CONNECTED':
                    log_info("[Kite Ticker WS] WebSocket not connected, breaking loop")
                    break
                
                current_time = time.time()
                
                # Send status update every 30 seconds (instead of polling REST API)
                if current_time - last_status_sent > 30:
                    try:
                        def get_status_update():
                            ticker = get_kite_ticker_instance()
                            if ticker:
                                return {
                                    "type": "status",
                                    "is_connected": ticker.is_connected,
                                    "is_running": ticker.is_running,
                                    "is_market_open": ticker.is_market_open(),
                                    "subscribed_instruments": ticker.instrument_tokens,
                                    "subscribed_count": len(ticker.instrument_tokens)
                                }
                            return None
                        
                        status_update = await loop.run_in_executor(executor, get_status_update)
                        if status_update:
                            # Add instrument details (lightweight, cached lookup)
                            try:
                                from utils.kite_utils import get_kite_instance
                                kite = get_kite_instance()
                                nse_instruments = kite.instruments("NSE")
                                bse_instruments = kite.instruments("BSE")
                                all_instruments = nse_instruments + bse_instruments
                                
                                instrument_details = []
                                for token in status_update.get("subscribed_instruments", []):
                                    for inst in all_instruments:
                                        if inst.get("instrument_token") == token:
                                            instrument_details.append({
                                                "token": token,
                                                "name": inst.get("name") or inst.get("tradingsymbol") or f"Token {token}",
                                                "tradingsymbol": inst.get("tradingsymbol"),
                                                "exchange": inst.get("exchange")
                                            })
                                            break
                                
                                status_update["subscribed_instruments_details"] = instrument_details
                            except Exception as e:
                                log_debug(f"[Kite Ticker WS] Could not get instrument details: {e}")
                            
                            # Only send status if WebSocket is still connected
                            if websocket.client_state.name == 'CONNECTED':
                                try:
                                    await websocket.send_json(status_update)
                                    last_status_sent = current_time
                                    last_ticker_status = status_update
                                except (RuntimeError, WebSocketDisconnect) as send_err:
                                    # If close message already sent or socket closed, break the loop
                                    log_debug(f"[Kite Ticker WS] Status send skipped/failed: {send_err}")
                                    break
                    except Exception as e:
                        log_error(f"[Kite Ticker WS] Error sending status update: {e}")
                
                # Get latest ticker data (non-blocking via executor)
                def get_latest_ticks():
                    return get_all_latest_kite_ticks()
                
                all_ticks = await loop.run_in_executor(executor, get_latest_ticks)
                
                # Only send if there are updates or every 5 seconds
                should_send = False
                
                if all_ticks:
                    # Check if any tick has been updated
                    for token, tick_data in all_ticks.items():
                        tick_time = tick_data.get('timestamp', 0)
                        if token not in last_sent_time or tick_time > last_sent_time.get(token, 0):
                            should_send = True
                            last_sent_time[token] = tick_time
                    
                    # Also send periodic updates (every 5 seconds) even if no change
                    if current_time - max(last_sent_time.values() or [0]) > 5:
                        should_send = True
                
                if should_send and all_ticks:
                    # Format tick data for frontend
                    from datetime import datetime, date
                    
                    def serialize_datetime(obj):
                        """Recursively serialize datetime objects to ISO format strings"""
                        if isinstance(obj, datetime):
                            return obj.isoformat()
                        elif isinstance(obj, date):
                            return obj.isoformat()
                        elif isinstance(obj, dict):
                            return {k: serialize_datetime(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [serialize_datetime(item) for item in obj]
                        elif isinstance(obj, (int, float, str, bool, type(None))):
                            return obj
                        else:
                            # For any other type, try to convert to string
                            return str(obj)
                    
                    # Get first 30-minute and 5-minute candles, and calculate strategy (do this once per update cycle)
                    def get_candles_and_strategy():
                        from utils.kite_utils import get_kite_instance
                        from datetime import timedelta
                        kite = get_kite_instance()
                        today_date = datetime.now().date()
                        
                        # Get previous day closes for gap calculation
                        prev_date = today_date - timedelta(days=1)
                        while prev_date.weekday() > 4:  # Skip weekends
                            prev_date = prev_date - timedelta(days=1)
                        
                        candles_30min = {}
                        breaking_5min = {}
                        previous_closes = {}
                        strategy_statuses = {}
                        
                        for token in all_ticks.keys():
                            candles_30min[token] = get_first_30min_candle_ohlc(kite, token, today_date)
                            # Get breaking 5min candle (requires 30min candle data)
                            breaking_5min[token] = get_breaking_5min_candle_ohlc(kite, token, today_date, candles_30min.get(token))
                            
                            # Get previous close
                            try:
                                hist_data = kite.historical_data(
                                    instrument_token=token,
                                    from_date=prev_date,
                                    to_date=prev_date + timedelta(days=1),
                                    interval="day"
                                )
                                if hist_data and len(hist_data) > 0:
                                    prev_close = hist_data[-1].get('close', 0)
                                    if prev_close and prev_close > 0:
                                        previous_closes[token] = float(prev_close)
                            except:
                                pass
                            
                            # Calculate gap
                            gap_value = None
                            previous_close = previous_closes.get(token)
                            tick_data = all_ticks.get(token, {})
                            if previous_close and tick_data.get('ohlc', {}).get('open'):
                                gap_value = tick_data.get('ohlc', {}).get('open') - previous_close
                            
                            # Calculate strategy status
                            strategy_statuses[token] = calculate_strategy_status(
                                gap_value=gap_value,
                                first_30min_ohlc=candles_30min.get(token),
                                breaking_5min_ohlc=breaking_5min.get(token),
                                previous_close=previous_close
                            )
                        
                        return candles_30min, breaking_5min, strategy_statuses
                    
                    first_30min_candles, breaking_5min_candles, strategy_statuses = await loop.run_in_executor(executor, get_candles_and_strategy)
                    
                    formatted_ticks = []
                    for token, tick_data in all_ticks.items():
                        formatted_tick = {
                            "instrument_token": token,
                            "last_price": tick_data.get('last_price', 0),
                            "ohlc": tick_data.get('ohlc', {}),
                            "timestamp": tick_data.get('timestamp', 0),
                            "change": tick_data.get('change', 0),
                            "depth": tick_data.get('depth', {}),
                            "first_30min_ohlc": first_30min_candles.get(token),
                            "first_5min_ohlc": breaking_5min_candles.get(token),
                            "strategy_status": strategy_statuses.get(token)
                        }
                        # Serialize any datetime objects in the tick data
                        formatted_tick = serialize_datetime(formatted_tick)
                        formatted_ticks.append(formatted_tick)
                    
                    # Send updates
                    try:
                        await websocket.send_json(formatted_ticks)
                    except (RuntimeError, WebSocketDisconnect) as e:
                        if isinstance(e, WebSocketDisconnect):
                            log_info("[Kite Ticker WS] WebSocket disconnected during send")
                        else:
                            log_warning(f"[Kite Ticker WS] WebSocket closed: {e}")
                        break
                
                # Wait 1 second before next update
                await asyncio.sleep(1)
                
            except (RuntimeError, WebSocketDisconnect) as e:
                if isinstance(e, WebSocketDisconnect):
                    log_info("[Kite Ticker WS] WebSocket disconnected")
                else:
                    log_warning(f"[Kite Ticker WS] WebSocket closed: {e}")
                break
            except Exception as e:
                import traceback
                log_error(f"[Kite Ticker WS] Error in streaming loop: {e}")
                log_error(f"[Kite Ticker WS] Traceback: {traceback.format_exc()}")
                try:
                    if websocket.client_state.name == 'CONNECTED':
                        await websocket.send_json({
                            "error": f"Streaming error: {str(e)}"
                        })
                except (RuntimeError, WebSocketDisconnect):
                    pass
                await asyncio.sleep(5)  # Wait before retrying
    
    except WebSocketDisconnect:
        log_info("[Kite Ticker WS] Client disconnected")
    except Exception as e:
        log_error(f"[Kite Ticker WS] Error: {e}")
        try:
            await websocket.close()
        except:
            pass


@router.get("/kite-ticker/latest")
async def get_latest_kite_ticks(request: Request):
    """
    Get latest ticker data from Kite WebSocket
    
    Returns all instruments that are currently subscribed and receiving ticks
    """
    try:
        from utils.kite_websocket_ticker import get_all_latest_kite_ticks, get_kite_ticker_instance
        from concurrent.futures import ThreadPoolExecutor
        import asyncio
        
        # Run sync operations in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        def get_ticker_data():
            ticker = get_kite_ticker_instance()
            if not ticker:
                return None, {}
            all_ticks = get_all_latest_kite_ticks()
            return ticker, all_ticks
        
        ticker, all_ticks = await loop.run_in_executor(executor, get_ticker_data)
        
        if not ticker:
            return SuccessResponse(
                data={
                    "ticks": {},
                    "is_connected": False,
                    "message": "Kite ticker not initialized. Market may be closed or access token not configured."
                },
                request_id=get_request_id(request)
            )
        
        # Format ticks for response (do this in executor to avoid blocking on lock)
        def format_ticks_data():
            from datetime import datetime, timedelta
            kite = get_kite_instance()
            today_date = datetime.now().date()
            
            # Get previous day closes for gap calculation
            prev_date = today_date - timedelta(days=1)
            while prev_date.weekday() > 4:  # Skip weekends
                prev_date = prev_date - timedelta(days=1)
            
            previous_closes = {}
            for token in all_ticks.keys():
                try:
                    hist_data = kite.historical_data(
                        instrument_token=token,
                        from_date=prev_date,
                        to_date=prev_date + timedelta(days=1),
                        interval="day"
                    )
                    if hist_data and len(hist_data) > 0:
                        prev_close = hist_data[-1].get('close', 0)
                        if prev_close and prev_close > 0:
                            previous_closes[token] = float(prev_close)
                except:
                    pass
            
            formatted_ticks = {}
            for token, tick_data in all_ticks.items():
                # Get first 30-minute candle OHLC
                first_30min_ohlc = get_first_30min_candle_ohlc(kite, token, today_date)
                # Get breaking 5-minute candle (requires 30min candle data)
                breaking_5min_ohlc = get_breaking_5min_candle_ohlc(kite, token, today_date, first_30min_ohlc)
                
                # Calculate gap
                gap_value = None
                previous_close = previous_closes.get(token)
                if previous_close and tick_data.get('ohlc', {}).get('open'):
                    gap_value = tick_data.get('ohlc', {}).get('open') - previous_close
                
                # Calculate strategy status
                strategy_status = calculate_strategy_status(
                    gap_value=gap_value,
                    first_30min_ohlc=first_30min_ohlc,
                    breaking_5min_ohlc=breaking_5min_ohlc,
                    previous_close=previous_close
                )
                
                formatted_ticks[token] = {
                    "instrument_token": token,
                    "last_price": tick_data.get('last_price', 0),
                    "ohlc": tick_data.get('ohlc', {}),
                    "timestamp": tick_data.get('timestamp', 0),
                    "change": tick_data.get('change', 0),
                    "depth": tick_data.get('depth', {}),
                    "first_30min_ohlc": first_30min_ohlc,
                    "first_5min_ohlc": breaking_5min_ohlc,
                    "strategy_status": strategy_status
                }
            return formatted_ticks, ticker.is_connected, ticker.is_running, ticker.instrument_tokens
        
        formatted_ticks, is_connected, is_running, subscribed = await loop.run_in_executor(executor, format_ticks_data)
        
        return SuccessResponse(
            data={
                "ticks": formatted_ticks,
                "is_connected": is_connected,
                "is_running": is_running,
                "subscribed_instruments": subscribed,
                "count": len(formatted_ticks)
            },
            request_id=get_request_id(request)
        )
    
    except Exception as e:
        log_error(f"Error getting latest Kite ticks: {e}")
        return ErrorResponse(
            message=f"Error getting latest ticks: {str(e)}",
            request_id=get_request_id(request)
        )


@router.get("/kite-ticker/status")
async def get_kite_ticker_status(request: Request):
    """Get Kite ticker connection status"""
    try:
        from utils.kite_websocket_ticker import get_kite_ticker_instance
        from concurrent.futures import ThreadPoolExecutor
        import asyncio
        
        # Run sync operations in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        def get_status():
            ticker = get_kite_ticker_instance()
            if not ticker:
                return None
            
            # Resolve instrument tokens to names for better display
            instrument_details = []
            try:
                from utils.kite_utils import get_kite_instance
                kite = get_kite_instance()
                
                # Get instruments from both NSE and BSE
                nse_instruments = kite.instruments("NSE")
                bse_instruments = kite.instruments("BSE")
                all_instruments = nse_instruments + bse_instruments
                
                for token in ticker.instrument_tokens:
                    # Find instrument by token
                    inst_info = None
                    for inst in all_instruments:
                        if inst.get("instrument_token") == token:
                            inst_info = inst
                            break
                    
                    if inst_info:
                        instrument_details.append({
                            "token": token,
                            "name": inst_info.get("name") or inst_info.get("tradingsymbol") or f"Token {token}",
                            "tradingsymbol": inst_info.get("tradingsymbol"),
                            "exchange": inst_info.get("exchange")
                        })
                    else:
                        # Fallback if not found
                        instrument_details.append({
                            "token": token,
                            "name": f"Token {token}",
                            "tradingsymbol": None,
                            "exchange": None
                        })
            except Exception as e:
                log_error(f"[Kite Ticker Status] Error resolving instrument names: {e}")
                # Fallback to just tokens if resolution fails
                instrument_details = [{"token": t, "name": f"Token {t}"} for t in ticker.instrument_tokens]
            
            return {
                "is_initialized": True,
                "is_connected": ticker.is_connected,
                "is_running": ticker.is_running,
                "is_market_open": ticker.is_market_open(),
                "subscribed_instruments": ticker.instrument_tokens,
                "subscribed_instruments_details": instrument_details,
                "subscribed_count": len(ticker.instrument_tokens)
            }
        
        status_data = await loop.run_in_executor(executor, get_status)
        
        if not status_data:
            return SuccessResponse(
                data={
                    "is_initialized": False,
                    "is_connected": False,
                    "is_running": False,
                    "message": "Kite ticker not initialized"
                },
                request_id=get_request_id(request)
            )
        
        return SuccessResponse(
            data=status_data,
            request_id=get_request_id(request)
        )
    
    except Exception as e:
        log_error(f"Error getting Kite ticker status: {e}")
        return ErrorResponse(
            message=f"Error getting ticker status: {str(e)}",
            request_id=get_request_id(request)
        )


@router.post("/kite-ticker/subscribe")
async def subscribe_kite_instruments(request: Request):
    """
    Subscribe to additional instruments in Kite ticker
    
    Request body (either instrument_tokens OR instrument_names):
        {
            "instrument_tokens": [256265, 260105, ...]  # Optional: list of instrument tokens
            "instrument_names": ["NIFTY 50", "NIFTY BANK", "SENSEX"]  # Optional: list of instrument names
        }
    """
    try:
        from utils.kite_websocket_ticker import get_kite_ticker_instance
        from agent.tools.instrument_resolver import get_instrument_token
        from concurrent.futures import ThreadPoolExecutor
        import asyncio
        
        # Parse request body
        body = await request.json()
        instrument_tokens = body.get("instrument_tokens", [])
        instrument_names = body.get("instrument_names", [])
        
        if not instrument_tokens and not instrument_names:
            return ErrorResponse(
                message="Either instrument_tokens or instrument_names is required in request body",
                request_id=get_request_id(request)
            )
        
        # Resolve instrument names to tokens if provided
        resolved_tokens = list(instrument_tokens) if instrument_tokens else []
        
        if instrument_names:
            if not isinstance(instrument_names, list):
                return ErrorResponse(
                    message="instrument_names must be a list of strings",
                    request_id=get_request_id(request)
                )
            
            # Resolve names to tokens
            from utils.kite_utils import get_kite_instance
            kite = get_kite_instance()
            
            for name in instrument_names:
                token = None
                if name.upper() == "SENSEX":
                    # For SENSEX, specifically search for INDEX type in BSE
                    try:
                        bse_instruments = kite.instruments("BSE")
                        for inst in bse_instruments:
                            # Look for SENSEX index (not ETF)
                            if (inst.get("name", "").upper() == "SENSEX" and 
                                inst.get("instrument_type", "").upper() == "INDEX"):
                                token = inst.get("instrument_token")
                                log_info(f"[Kite Ticker API] Found SENSEX index: token {token}")
                                break
                    except Exception as e:
                        log_error(f"[Kite Ticker API] Error finding SENSEX index: {e}")
                else:
                    # For other instruments, use standard resolver
                    token = get_instrument_token(name, exchange="NSE")
                    if not token:
                        token = get_instrument_token(name, exchange="BSE")
                
                if token:
                    resolved_tokens.append(token)
                    log_info(f"[Kite Ticker API] Resolved '{name}' to token {token}")
                else:
                    log_warning(f"[Kite Ticker API] Could not resolve instrument name '{name}'")
        
        if not resolved_tokens:
            return ErrorResponse(
                message="No valid instrument tokens found. Please check instrument names or tokens.",
                request_id=get_request_id(request)
            )
        
        if not isinstance(instrument_tokens, list) and instrument_tokens:
            return ErrorResponse(
                message="instrument_tokens must be a list of integers",
                request_id=get_request_id(request)
            )
        
        # Run sync operations in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        def subscribe_instruments():
            ticker = get_kite_ticker_instance()
            if not ticker:
                return None, "Kite ticker not initialized. Please ensure market is open and access token is configured."
            ticker.subscribe(resolved_tokens)
            return ticker, None
        
        ticker, error = await loop.run_in_executor(executor, subscribe_instruments)
        
        if error:
            return ErrorResponse(
                message=error,
                request_id=get_request_id(request)
            )
        
        return SuccessResponse(
            data={
                "message": f"Subscribed to {len(resolved_tokens)} instrument(s)",
                "subscribed_tokens": resolved_tokens,
                "all_subscribed": ticker.instrument_tokens
            },
            request_id=get_request_id(request)
        )
    
    except json.JSONDecodeError:
        return ErrorResponse(
            message="Invalid JSON in request body",
            request_id=get_request_id(request)
        )
    except Exception as e:
        log_error(f"Error subscribing to instruments: {e}")
        return ErrorResponse(
            message=f"Error subscribing: {str(e)}",
            request_id=get_request_id(request)
        )


def get_first_30min_candle_ohlc(kite, instrument_token: int, today_date) -> Optional[Dict]:
    """
    Get the first 30-minute candle OHLC for today
    
    Returns:
        Dict with 'open', 'high', 'low', 'close' or None if not available
    """
    try:
        # Fetch 30-minute candles for today
        hist_data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=today_date,
            to_date=today_date,
            interval="30minute"
        )
        
        if hist_data and len(hist_data) > 0:
            # Get the first candle (index 0) - this is the first 30-minute candle
            first_candle = hist_data[0]
            return {
                "open": float(first_candle.get('open', 0)),
                "high": float(first_candle.get('high', 0)),
                "low": float(first_candle.get('low', 0)),
                "close": float(first_candle.get('close', 0))
            }
    except Exception as e:
        log_debug(f"[30min Candle] Error getting first 30min candle for token {instrument_token}: {e}")
    return None


def get_breaking_5min_candle_ohlc(kite, instrument_token: int, today_date, first_30min_ohlc: Optional[Dict]) -> Optional[Dict]:
    """
    Get the first 5-minute candle that breaks the 30-minute candle region
    (either closes above 30min high or below 30min low)
    
    Args:
        kite: Kite instance
        instrument_token: Instrument token
        today_date: Today's date
        first_30min_ohlc: The first 30-minute candle OHLC data
    
    Returns:
        Dict with 'open', 'high', 'low', 'close' or None if not available
    """
    if not first_30min_ohlc:
        return None
    
    try:
        # Fetch 5-minute candles for today
        hist_data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=today_date,
            to_date=today_date,
            interval="5minute"
        )
        
        if not hist_data or len(hist_data) == 0:
            return None
        
        # Get 30min candle boundaries
        candle_30_high = first_30min_ohlc.get('high', 0)
        candle_30_low = first_30min_ohlc.get('low', 0)
        
        if candle_30_high == 0 or candle_30_low == 0:
            return None
        
        # Find the first 5min candle that breaks the 30min region
        # Start from index 6 (9:45 candle - first candle after 30min)
        # Look for a candle that closes above 30min high (bullish break) or below 30min low (bearish break)
        
        start_index = 6  # First 5min candle after 30min (9:45)
        
        for i in range(start_index, len(hist_data)):
            candle = hist_data[i]
            candle_close = float(candle.get('close', 0))
            candle_high = float(candle.get('high', 0))
            candle_low = float(candle.get('low', 0))
            
            if candle_close == 0:
                continue
            
            # Check if this candle breaks the 30min region
            # Bullish break: close above 30min high
            # Bearish break: close below 30min low
            if candle_close > candle_30_high or candle_close < candle_30_low:
                # Get timestamp from candle
                candle_date = candle.get('date')
                timestamp = None
                if isinstance(candle_date, datetime):
                    timestamp = int(candle_date.timestamp())
                elif candle_date:
                    # Try to parse if it's a string
                    try:
                        if isinstance(candle_date, str):
                            candle_date = datetime.fromisoformat(candle_date.replace('Z', '+00:00'))
                        timestamp = int(candle_date.timestamp())
                    except:
                        pass
                
                return {
                    "open": float(candle.get('open', 0)),
                    "high": candle_high,
                    "low": candle_low,
                    "close": candle_close,
                    "timestamp": timestamp,
                    "date": candle_date.isoformat() if isinstance(candle_date, datetime) else str(candle_date) if candle_date else None
                }
        
        # If no breaking candle found, return None
        return None
        
    except Exception as e:
        log_debug(f"[5min Candle] Error getting breaking 5min candle for token {instrument_token}: {e}")
    return None


def calculate_strategy_status(
    gap_value: Optional[float],
    first_30min_ohlc: Optional[Dict],
    breaking_5min_ohlc: Optional[Dict],
    previous_close: Optional[float]
) -> Dict:
    """
    Calculate strategy status based on:
    1. Gap up/down
    2. 30min candle OHLC
    3. 5min candle after 30min (closing above/below 30min candle)
    4. Strength of 5min candle
    
    Returns:
        Dict with 'status', 'direction', 'buy_price', 'sell_price', 'signal_strength'
    """
    result = {
        "status": "WAITING",
        "direction": None,
        "buy_price": None,
        "sell_price": None,
        "signal_strength": None,
        "message": "Waiting for candles to form"
    }
    
    # Step 1: Check gap
    if gap_value is None or previous_close is None:
        result["message"] = "Gap data not available"
        return result
    
    is_gap_up = gap_value > 0
    is_gap_down = gap_value < 0
    
    # Step 2: Check if 30min candle is available
    if not first_30min_ohlc:
        result["message"] = "Waiting for 30min candle"
        return result
    
    # Step 3: Check if breaking 5min candle is available
    if not breaking_5min_ohlc:
        result["message"] = "Waiting for 5min candle to break 30min region"
        return result
    
    # Extract values
    candle_30_open = first_30min_ohlc.get('open', 0)
    candle_30_high = first_30min_ohlc.get('high', 0)
    candle_30_low = first_30min_ohlc.get('low', 0)
    candle_30_close = first_30min_ohlc.get('close', 0)
    
    candle_5_open = breaking_5min_ohlc.get('open', 0)
    candle_5_high = breaking_5min_ohlc.get('high', 0)
    candle_5_low = breaking_5min_ohlc.get('low', 0)
    candle_5_close = breaking_5min_ohlc.get('close', 0)
    
    if not all([candle_30_close, candle_5_close, candle_30_high, candle_30_low]):
        result["message"] = "Candle data incomplete"
        return result
    
    # Step 3: Check if 5min candle broke above 30min high (bullish) or below 30min low (bearish)
    candle_5_broke_above = candle_5_close > candle_30_high
    candle_5_broke_below = candle_5_close < candle_30_low
    
    # Step 4: Calculate 5min candle strength
    # Strength factors:
    # - Body size (close - open) as percentage of range
    # - Small wicks (high - max(open,close) and min(open,close) - low)
    # - Close near high (for bullish) or near low (for bearish)
    
    candle_5_range = candle_5_high - candle_5_low
    candle_5_body = abs(candle_5_close - candle_5_open)
    
    if candle_5_range == 0:
        result["message"] = "Invalid candle range"
        return result
    
    body_percentage = (candle_5_body / candle_5_range) * 100
    
    # Calculate wick sizes
    upper_wick = candle_5_high - max(candle_5_open, candle_5_close)
    lower_wick = min(candle_5_open, candle_5_close) - candle_5_low
    total_wick = upper_wick + lower_wick
    wick_percentage = (total_wick / candle_5_range) * 100 if candle_5_range > 0 else 0
    
    # Close position in range (0 = at low, 100 = at high)
    close_position = ((candle_5_close - candle_5_low) / candle_5_range) * 100 if candle_5_range > 0 else 50
    
    # Determine strength (strong if: body > 60% of range, wicks < 30% of range)
    is_strong_bullish = (
        candle_5_close > candle_5_open and  # Bullish candle
        body_percentage > 60 and  # Strong body
        wick_percentage < 30 and  # Small wicks
        close_position > 70  # Close near high
    )
    
    is_strong_bearish = (
        candle_5_close < candle_5_open and  # Bearish candle
        body_percentage > 60 and  # Strong body
        wick_percentage < 30 and  # Small wicks
        close_position < 30  # Close near low
    )
    
    # Step 5: Determine direction and prices
    if candle_5_broke_above and is_strong_bullish:
        # BUY signal - broke above 30min high
        result["status"] = "BUY_SIGNAL"
        result["direction"] = "BUY"
        result["buy_price"] = round(candle_5_close, 2)
        # Sell price: 30min high as target (already broken, so use 30min high + some buffer)
        result["sell_price"] = round(candle_30_high, 2)  # Target
        result["signal_strength"] = "STRONG"
        result["message"] = f"Strong bullish 5min candle broke above 30min high"
    elif candle_5_broke_below and is_strong_bearish:
        # SELL signal - broke below 30min low
        result["status"] = "SELL_SIGNAL"
        result["direction"] = "SELL"
        result["sell_price"] = round(candle_5_close, 2)
        # Buy price: 30min low as target (already broken, so use 30min low)
        result["buy_price"] = round(candle_30_low, 2)  # Target
        result["signal_strength"] = "STRONG"
        result["message"] = f"Strong bearish 5min candle broke below 30min low"
    elif candle_5_broke_above:
        # Weak bullish signal - broke above but weak strength
        result["status"] = "WEAK_BUY"
        result["direction"] = "BUY"
        result["buy_price"] = round(candle_5_close, 2)
        result["sell_price"] = round(candle_30_high, 2)
        result["signal_strength"] = "WEAK"
        result["message"] = f"5min candle broke above 30min high but weak strength"
    elif candle_5_broke_below:
        # Weak bearish signal - broke below but weak strength
        result["status"] = "WEAK_SELL"
        result["direction"] = "SELL"
        result["sell_price"] = round(candle_5_close, 2)
        result["buy_price"] = round(candle_30_low, 2)
        result["signal_strength"] = "WEAK"
        result["message"] = f"5min candle broke below 30min low but weak strength"
    else:
        # No clear signal (shouldn't happen if breaking candle is found, but safety check)
        result["status"] = "NO_SIGNAL"
        result["message"] = "5min candle did not break 30min region"
    
    return result


@router.get("/kite-ticker/previous-close")
async def get_previous_day_close(request: Request):
    """
    Get previous trading day's close price for subscribed instruments
    Returns a map of instrument_token -> previous_close_price
    """
    try:
        from utils.kite_websocket_ticker import get_kite_ticker_instance
        from utils.kite_utils import get_kite_instance
        from concurrent.futures import ThreadPoolExecutor
        from datetime import datetime, timedelta
        import asyncio
        
        # Run sync operations in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        def get_previous_closes():
            ticker = get_kite_ticker_instance()
            if not ticker:
                return {}, None
            
            kite = get_kite_instance()
            previous_closes = {}
            
            # Get previous trading day (skip weekends)
            today = datetime.now().date()
            prev_date = today - timedelta(days=1)
            # Skip weekends - go back to Friday if today is Monday
            while prev_date.weekday() > 4:  # Saturday = 5, Sunday = 6
                prev_date = prev_date - timedelta(days=1)
            
            # Try to get previous trading day's data
            for token in ticker.instrument_tokens:
                try:
                    # Get daily candles for previous day
                    hist_data = kite.historical_data(
                        instrument_token=token,
                        from_date=prev_date,
                        to_date=prev_date + timedelta(days=1),
                        interval="day"
                    )
                    
                    if hist_data and len(hist_data) > 0:
                        # Get the last candle's close (previous day close)
                        prev_close = hist_data[-1].get('close', 0)
                        if prev_close and prev_close > 0:
                            previous_closes[token] = float(prev_close)
                    else:
                        # If no daily data, try to get last minute candle of previous day
                        hist_data = kite.historical_data(
                            instrument_token=token,
                            from_date=prev_date,
                            to_date=prev_date + timedelta(days=1),
                            interval="minute"
                        )
                        if hist_data and len(hist_data) > 0:
                            # Get the last candle before market close (15:30)
                            last_candle = None
                            for candle in reversed(hist_data):
                                candle_time = candle.get('date')
                                if isinstance(candle_time, datetime):
                                    if candle_time.time() <= datetime.strptime("15:30", "%H:%M").time():
                                        last_candle = candle
                                        break
                            
                            if last_candle:
                                prev_close = last_candle.get('close', 0)
                                if prev_close and prev_close > 0:
                                    previous_closes[token] = float(prev_close)
                except Exception as e:
                    log_error(f"[Previous Close] Error getting previous close for token {token}: {e}")
                    continue
            
            return previous_closes, prev_date
        
        previous_closes, prev_date = await loop.run_in_executor(executor, get_previous_closes)
        
        return SuccessResponse(
            data={
                "previous_closes": previous_closes,
                "previous_date": str(prev_date)
            },
            request_id=get_request_id(request)
        )
    
    except Exception as e:
        log_error(f"Error getting previous day close: {e}")
        return ErrorResponse(
            message=f"Error getting previous day close: {str(e)}",
            request_id=get_request_id(request)
        )


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

