"""
Market data API endpoints (candles, quotes, instruments, etc.)
"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime

from utils.kite_utils import (
    api_key,
    get_access_token,
    get_kite_instance,
    calculate_trend_and_suggestions
)
from kiteconnect.exceptions import KiteException
from core.user_context import get_user_id_from_request

router = APIRouter(prefix="/market", tags=["Market Data"])


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


