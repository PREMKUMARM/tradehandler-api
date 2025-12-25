"""
Market analysis tools for the agent
"""
from typing import Optional, Union, List
from langchain_core.tools import tool
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.kite_utils import get_kite_instance, calculate_trend_and_suggestions
from kiteconnect.exceptions import KiteException


@tool
def get_quote_tool(
    instrument_key: Optional[Union[str, List[str]]] = None,
    instrument_name: Optional[Union[str, List[str]]] = None
) -> dict:
    """
    Get real-time quote for one or multiple instruments.
    
    Args:
        instrument_key: Single instrument key or list of keys in format "EXCHANGE|TOKEN" (optional if instrument_name provided)
        instrument_name: Single instrument name or list of names like "RELIANCE" or ["RELIANCE", "INFOSYS"] (optional if instrument_key provided)
        
    Returns:
        dict with quote data for single instrument, or dict with results for multiple instruments
    """
    try:
        kite = get_kite_instance()
        from agent.tools.instrument_resolver import resolve_instrument_name
        
        # Normalize to lists
        if instrument_name:
            if isinstance(instrument_name, str):
                instrument_names = [instrument_name]
            else:
                instrument_names = instrument_name
        elif instrument_key:
            if isinstance(instrument_key, str):
                instrument_keys = [instrument_key]
            else:
                instrument_keys = instrument_key
            # Convert keys to names for processing
            instrument_names = instrument_keys  # Will be processed differently
        else:
            return {
                "status": "error",
                "error": "Either instrument_key or instrument_name must be provided"
            }
        
        # Resolve all instrument names to keys
        quote_keys = []
        instrument_info_list = []
        
        for inst in instrument_names:
            if isinstance(inst, str) and ('|' in inst or ':' in inst):
                # Already a key format
                if '|' in inst:
                    exchange, token = inst.split('|')
                    quote_key = f"{exchange}:{token}"
                elif ':' in inst:
                    quote_key = inst
                else:
                    quote_key = f"NSE:{inst}"
                quote_keys.append(quote_key)
                instrument_info_list.append({"tradingsymbol": inst, "exchange": "NSE"})
            else:
                # Resolve name to key
                instrument_info = resolve_instrument_name(inst)
                if instrument_info:
                    quote_key = f"{instrument_info['exchange']}:{instrument_info['tradingsymbol']}"
                    quote_keys.append(quote_key)
                    instrument_info_list.append(instrument_info)
                else:
                    return {
                        "status": "error",
                        "error": f"Instrument '{inst}' not found"
                    }
        
        # Get quotes for all instruments
        quotes = kite.quote(quote_keys)
        
        # Process results
        if len(quote_keys) == 1:
            # Single instrument - return simple format
            quote_key = quote_keys[0]
            if quote_key in quotes:
                quote_data = quotes[quote_key]
                return {
                    "status": "success",
                    "instrument_key": quote_key,
                    "last_price": quote_data.get("last_price", 0),
                    "bid": quote_data.get("depth", {}).get("buy", [{}])[0].get("price", 0) if quote_data.get("depth", {}).get("buy") else 0,
                    "ask": quote_data.get("depth", {}).get("sell", [{}])[0].get("price", 0) if quote_data.get("depth", {}).get("sell") else 0,
                    "volume": quote_data.get("volume", 0),
                    "oi": quote_data.get("oi", 0),
                    "raw": quote_data
                }
            else:
                return {
                    "status": "error",
                    "error": f"Quote not found for {quote_key}"
                }
        else:
            # Multiple instruments - return aggregated format
            results = {}
            for quote_key, inst_info in zip(quote_keys, instrument_info_list):
                if quote_key in quotes:
                    quote_data = quotes[quote_key]
                    results[inst_info["tradingsymbol"]] = {
                        "instrument": inst_info["tradingsymbol"],
                        "last_price": quote_data.get("last_price", 0),
                        "bid": quote_data.get("depth", {}).get("buy", [{}])[0].get("price", 0) if quote_data.get("depth", {}).get("buy") else 0,
                        "ask": quote_data.get("depth", {}).get("sell", [{}])[0].get("price", 0) if quote_data.get("depth", {}).get("sell") else 0,
                        "volume": quote_data.get("volume", 0),
                        "oi": quote_data.get("oi", 0)
                    }
                else:
                    results[inst_info["tradingsymbol"]] = {
                        "status": "error",
                        "error": f"Quote not found"
                    }
            
            return {
                "status": "success",
                "instruments": list(results.keys()),
                "count": len(results),
                "quotes": results
            }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error getting quote: {str(e)}"
        }


@tool
def get_historical_data_tool(
    instrument_token: Optional[Union[int, List[int]]] = None,
    instrument_name: Optional[Union[str, List[str]]] = None,
    interval: str = "5minute",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    days: int = 1,
    exchange: str = "NSE"
) -> dict:
    """
    Get historical candle data for one or multiple instruments.
    
    Args:
        instrument_token: Single instrument token or list of tokens (optional if instrument_name provided)
        instrument_name: Single instrument name or list of names like "RELIANCE" or ["RELIANCE", "INFOSYS"] (optional if instrument_token provided)
        interval: Time interval (minute, 5minute, 15minute, 30minute, 60minute, day)
        from_date: Start date (YYYY-MM-DD format, optional)
        to_date: End date (YYYY-MM-DD format, optional)
        days: Number of days back from today (used if dates not provided)
        exchange: Exchange (NSE, NFO, BSE)
        
    Returns:
        dict with candle data for single instrument, or dict with results for multiple instruments
    """
    try:
        kite = get_kite_instance()
        from agent.tools.instrument_resolver import get_instrument_token, resolve_instrument_name
        
        # Normalize to lists
        if instrument_name:
            if isinstance(instrument_name, str):
                instrument_names = [instrument_name]
            else:
                instrument_names = instrument_name
            # Resolve names to tokens
            instrument_tokens = []
            instrument_info_list = []
            for inst_name in instrument_names:
                token = get_instrument_token(inst_name, exchange)
                if token:
                    instrument_tokens.append(token)
                    info = resolve_instrument_name(inst_name, exchange)
                    instrument_info_list.append(info or {"tradingsymbol": inst_name})
                else:
                    return {
                        "status": "error",
                        "error": f"Instrument '{inst_name}' not found"
                    }
        elif instrument_token:
            if isinstance(instrument_token, (int, float)):
                instrument_tokens = [int(instrument_token)]
            else:
                instrument_tokens = [int(t) for t in instrument_token]
            instrument_info_list = [{"instrument_token": t} for t in instrument_tokens]
        else:
            return {
                "status": "error",
                "error": "Either instrument_token or instrument_name must be provided"
            }
        
        # Parse dates
        if from_date and to_date:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
            to_dt = datetime.strptime(to_date, "%Y-%m-%d").date()
        else:
            to_dt = datetime.now().date()
            from_dt = to_dt - timedelta(days=days)
        
        # Get historical data for all instruments
        if len(instrument_tokens) == 1:
            # Single instrument - return simple format
            historical_data = kite.historical_data(
                instrument_token=instrument_tokens[0],
                from_date=from_dt,
                to_date=to_dt,
                interval=interval
            )
            return {
                "status": "success",
                "instrument_token": instrument_tokens[0],
                "interval": interval,
                "from_date": str(from_dt),
                "to_date": str(to_dt),
                "candles": historical_data,
                "count": len(historical_data) if historical_data else 0
            }
        else:
            # Multiple instruments - return aggregated format
            results = {}
            for token, inst_info in zip(instrument_tokens, instrument_info_list):
                try:
                    historical_data = kite.historical_data(
                        instrument_token=token,
                        from_date=from_dt,
                        to_date=to_dt,
                        interval=interval
                    )
                    results[inst_info.get("tradingsymbol", f"TOKEN_{token}")] = {
                        "instrument_token": token,
                        "candles": historical_data,
                        "count": len(historical_data) if historical_data else 0
                    }
                except Exception as e:
                    results[inst_info.get("tradingsymbol", f"TOKEN_{token}")] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return {
                "status": "success",
                "interval": interval,
                "from_date": str(from_dt),
                "to_date": str(to_dt),
                "instruments": list(results.keys()),
                "count": len(results),
                "results": results
            }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error getting historical data: {str(e)}"
        }


@tool
def analyze_trend_tool(
    instrument_token: Optional[Union[int, List[int]]] = None,
    instrument_name: Optional[Union[str, List[str]]] = None,
    current_price: Optional[float] = None,
    exchange: str = "NSE"
) -> dict:
    """
    Analyze trend across multiple timeframes (5min, 15min, 30min) for one or multiple instruments.
    
    Args:
        instrument_token: Single instrument token or list of tokens (optional if instrument_name provided)
        instrument_name: Single instrument name or list of names like "RELIANCE" or ["RELIANCE", "INFOSYS"] (optional if instrument_token provided)
        current_price: Current price (optional, will be fetched if not provided)
        exchange: Exchange (NSE, NFO, BSE)
        
    Returns:
        dict with trend analysis for single instrument, or dict with results for multiple instruments
    """
    try:
        kite = get_kite_instance()
        from agent.tools.instrument_resolver import get_instrument_token, resolve_instrument_name
        
        # Normalize to lists
        if instrument_name:
            if isinstance(instrument_name, str):
                instrument_names = [instrument_name]
            else:
                instrument_names = instrument_name
            # Resolve names to tokens
            instrument_tokens = []
            instrument_info_list = []
            for inst_name in instrument_names:
                token = get_instrument_token(inst_name, exchange)
                if token:
                    instrument_tokens.append(token)
                    info = resolve_instrument_name(inst_name, exchange)
                    instrument_info_list.append(info or {"tradingsymbol": inst_name})
                else:
                    return {
                        "status": "error",
                        "error": f"Instrument '{inst_name}' not found"
                    }
        elif instrument_token:
            if isinstance(instrument_token, (int, float)):
                instrument_tokens = [int(instrument_token)]
            else:
                instrument_tokens = [int(t) for t in instrument_token]
            instrument_info_list = [{"instrument_token": t} for t in instrument_tokens]
        else:
            return {
                "status": "error",
                "error": "Either instrument_token or instrument_name must be provided"
            }
        
        # Analyze trends for all instruments
        if len(instrument_tokens) == 1:
            # Single instrument - return simple format
            token = instrument_tokens[0]
            if current_price is None:
                # Get current price from quote
                inst_info = instrument_info_list[0]
                quote_key = f"{inst_info.get('exchange', 'NSE')}:{inst_info.get('tradingsymbol', '')}"
                quote = kite.quote(quote_key)
                if quote_key in quote:
                    current_price = quote[quote_key].get("last_price", 0)
                else:
                    current_price = 0
            
            trend_data = calculate_trend_and_suggestions(kite, token, current_price)
            return {
                "status": "success",
                "instrument_token": token,
                "current_price": current_price,
                **trend_data
            }
        else:
            # Multiple instruments - return aggregated format
            results = {}
            for token, inst_info in zip(instrument_tokens, instrument_info_list):
                try:
                    # Get current price
                    quote_key = f"{inst_info.get('exchange', 'NSE')}:{inst_info.get('tradingsymbol', '')}"
                    quote = kite.quote(quote_key)
                    price = quote.get(quote_key, {}).get("last_price", 0) if quote_key in quote else 0
                    
                    trend_data = calculate_trend_and_suggestions(kite, token, price)
                    results[inst_info.get("tradingsymbol", f"TOKEN_{token}")] = {
                        "instrument_token": token,
                        "current_price": price,
                        **trend_data
                    }
                except Exception as e:
                    results[inst_info.get("tradingsymbol", f"TOKEN_{token}")] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return {
                "status": "success",
                "instruments": list(results.keys()),
                "count": len(results),
                "results": results
            }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error analyzing trend: {str(e)}"
        }


@tool
def get_nifty_options_tool() -> dict:
    """
    Get Nifty50 options chain with current strike, 2 ITM and 2 OTM strikes.
    Includes trend analysis and buy/sell suggestions.
    
    Returns:
        dict with nifty_price, current_strike, expiry, and options list
    """
    try:
        kite = get_kite_instance()
        
        # Get Nifty50 current price
        nifty_quote = kite.quote("NSE:NIFTY 50")
        nifty_price = nifty_quote.get("NSE:NIFTY 50", {}).get("last_price", 0)
        
        if not nifty_price:
            return {
                "status": "error",
                "error": "Nifty50 price not found"
            }
        
        # Round to nearest 50 for strike calculation
        current_strike = round(nifty_price / 50) * 50
        
        # Get all NFO instruments
        all_instruments = kite.instruments("NFO")
        
        # Filter for Nifty50 options
        nifty_options = [
            inst for inst in all_instruments 
            if inst.get("name") == "NIFTY" and inst.get("instrument_type") in ["CE", "PE"]
        ]
        
        if not nifty_options:
            return {
                "status": "error",
                "error": "Nifty50 options not found"
            }
        
        # Get current expiry (nearest expiry)
        expiries = sorted(set([inst.get("expiry") for inst in nifty_options if inst.get("expiry")]))
        if not expiries:
            return {
                "status": "error",
                "error": "No valid expiries found"
            }
        
        current_expiry = expiries[0]
        
        # Filter for current expiry
        current_expiry_options = [
            inst for inst in nifty_options 
            if inst.get("expiry") == current_expiry
        ]
        
        # Calculate strikes: current, 2 ITM, 2 OTM
        strikes_to_find = [
            current_strike,
            current_strike - 50,
            current_strike - 100,
            current_strike + 50,
            current_strike + 100
        ]
        
        # Get instrument tokens for these strikes
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
        
        # Get quotes and trend analysis
        instrument_keys = [f"NFO:{opt['tradingsymbol']}" for opt in result_options]
        if instrument_keys:
            quotes = kite.quote(instrument_keys)
            
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
                    
                    # Calculate trend
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
                    except Exception:
                        opt["trend"] = "NEUTRAL"
                        opt["trend_strength"] = 0
        
        return {
            "status": "success",
            "nifty_price": nifty_price,
            "current_strike": current_strike,
            "expiry": current_expiry,
            "options": result_options
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error getting Nifty50 options: {str(e)}"
        }


@tool
def calculate_indicators_tool(
    instrument_token: Optional[Union[int, List[int]]] = None,
    instrument_name: Optional[Union[str, List[str]]] = None,
    interval: str = "5minute",
    indicators: str = "RSI,MACD,BB",
    return_historical: bool = False,
    exchange: str = "NSE"
) -> dict:
    """
    Calculate technical indicators (RSI, MACD, Bollinger Bands) for one or multiple instruments.
    
    Args:
        instrument_token: Single instrument token or list of tokens (optional if instrument_name provided)
        instrument_name: Single instrument name or list of names like "RELIANCE" or ["RELIANCE", "INFOSYS"] (optional if instrument_token provided)
        interval: Time interval (5minute, 15minute, 30minute, day, minute for 1min)
        indicators: Comma-separated list of indicators (RSI, MACD, BB)
        return_historical: If True, return historical indicator values for each candle
        exchange: Exchange (NSE, NFO, BSE)
        
    Returns:
        dict with calculated indicator values for single instrument, or dict with results for multiple instruments
    """
    try:
        kite = get_kite_instance()
        from agent.tools.instrument_resolver import get_instrument_token, resolve_instrument_name
        
        # Normalize to lists
        if instrument_name:
            if isinstance(instrument_name, str):
                instrument_names = [instrument_name]
            else:
                instrument_names = instrument_name
            # Resolve names to tokens
            instrument_tokens = []
            instrument_info_list = []
            for inst_name in instrument_names:
                token = get_instrument_token(inst_name, exchange)
                if token:
                    instrument_tokens.append(token)
                    info = resolve_instrument_name(inst_name, exchange)
                    instrument_info_list.append(info or {"tradingsymbol": inst_name})
                else:
                    return {
                        "status": "error",
                        "error": f"Instrument '{inst_name}' not found"
                    }
        elif instrument_token:
            if isinstance(instrument_token, (int, float)):
                instrument_tokens = [int(instrument_token)]
            else:
                instrument_tokens = [int(t) for t in instrument_token]
            instrument_info_list = [{"instrument_token": t} for t in instrument_tokens]
        else:
            return {
                "status": "error",
                "error": "Either instrument_token or instrument_name must be provided"
            }
        
        # Helper function to calculate indicators for a single instrument
        def calculate_for_instrument(token, inst_info):
            try:
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                
                # Get historical data
                historical_data = kite.historical_data(
                    instrument_token=token,
                    from_date=yesterday,
                    to_date=today,
                    interval=interval
                )
                
                if not historical_data or len(historical_data) < 20:
                    return {
                        "status": "error",
                        "error": "Insufficient data for indicator calculation"
                    }
                
                import pandas as pd
                import numpy as np
                
                df = pd.DataFrame(historical_data)
                if len(df) == 0:
                    return {
                        "status": "error",
                        "error": "No historical data available"
                    }
                
                # Ensure date column exists
                if 'date' not in df.columns and len(df) > 0:
                    df['date'] = pd.to_datetime(df.index) if df.index.name == 'date' else pd.to_datetime(df.get('date', pd.Timestamp.now()))
                
                closes = df['close'].values
                
                result = {
                    "status": "success",
                    "instrument_token": token,
                    "interval": interval,
                    "indicators": {},
                    "historical": [] if return_historical else None
                }
                
                indicator_list = [ind.strip().upper() for ind in indicators.split(",")]
                
                # Calculate RSI
                if "RSI" in indicator_list:
                    period = 14
                    if len(closes) >= period + 1:
                        deltas = np.diff(closes)
                        gains = np.where(deltas > 0, deltas, 0.0)
                        losses = np.where(deltas < 0, -deltas, 0.0)
                        
                        avg_gains = np.full(len(gains), np.nan)
                        avg_losses = np.full(len(losses), np.nan)
                        
                        avg_gains[period - 1] = np.mean(gains[:period])
                        avg_losses[period - 1] = np.mean(losses[:period])
                        
                        for i in range(period, len(gains)):
                            avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
                            avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period
                        
                        rs = avg_gains / (avg_losses + 1e-10)
                        rsi = 100 - (100 / (1 + rs))
                        
                        # Pad RSI array to match closes length (first value is NaN)
                        rsi_full = np.concatenate([[np.nan], rsi])
                        
                        if return_historical:
                            # Return historical RSI values with timestamps
                            historical_rsi = []
                            for idx, (date_val, rsi_val, close_val) in enumerate(zip(df['date'].values, rsi_full, closes)):
                                if not np.isnan(rsi_val):
                                    historical_rsi.append({
                                        "timestamp": date_val.isoformat() if hasattr(date_val, 'isoformat') else str(date_val),
                                        "rsi": float(rsi_val),
                                        "close": float(close_val),
                                        "index": idx
                                    })
                            result["indicators"]["RSI"] = {
                                "current": float(rsi[-1]) if not np.isnan(rsi[-1]) else None,
                                "period": period,
                                "historical": historical_rsi
                            }
                        else:
                            result["indicators"]["RSI"] = {
                                "value": float(rsi[-1]) if not np.isnan(rsi[-1]) else None,
                                "period": period
                            }
                
                # Calculate MACD
                if "MACD" in indicator_list:
                    if len(closes) >= 26:
                        ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
                        ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
                        macd_line = ema12 - ema26
                        signal_line = macd_line.ewm(span=9, adjust=False).mean()
                        histogram = macd_line - signal_line
                        
                        result["indicators"]["MACD"] = {
                            "macd": float(macd_line.iloc[-1]),
                            "signal": float(signal_line.iloc[-1]),
                            "histogram": float(histogram.iloc[-1])
                        }
                
                # Calculate Bollinger Bands
                if "BB" in indicator_list:
                    period = 20
                    if len(closes) >= period:
                        sma = pd.Series(closes).rolling(window=period).mean()
                        std = pd.Series(closes).rolling(window=period).std()
                        upper_band = sma + (std * 2)
                        lower_band = sma - (std * 2)
                        
                        result["indicators"]["BB"] = {
                            "upper": float(upper_band.iloc[-1]),
                            "middle": float(sma.iloc[-1]),
                            "lower": float(lower_band.iloc[-1]),
                            "period": period
                        }
                
                return result
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        # Process single or multiple instruments
        if len(instrument_tokens) == 1:
            # Single instrument - return simple format
            return calculate_for_instrument(instrument_tokens[0], instrument_info_list[0])
        else:
            # Multiple instruments - return aggregated format
            results = {}
            for token, inst_info in zip(instrument_tokens, instrument_info_list):
                result = calculate_for_instrument(token, inst_info)
                results[inst_info.get("tradingsymbol", f"TOKEN_{token}")] = result
            
            return {
                "status": "success",
                "interval": interval,
                "indicators": indicators,
                "instruments": list(results.keys()),
                "count": len(results),
                "results": results
            }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error calculating indicators: {str(e)}"
        }

