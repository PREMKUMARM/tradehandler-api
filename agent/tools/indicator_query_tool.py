"""
Time-based indicator query tools
"""
from typing import Optional, List, Dict, Any, Union
from langchain_core.tools import tool
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.kite_utils import get_kite_instance
from agent.tools.instrument_resolver import resolve_instrument_name, get_instrument_token
from agent.tools.market_tools import calculate_indicators_tool
from kiteconnect.exceptions import KiteException


@tool
def find_indicator_threshold_crossings(
    instrument_name: Union[str, List[str]],
    indicator: str = "RSI",
    threshold: float = 30.0,
    direction: str = "below",  # "below" or "above"
    date: Optional[str] = None,
    interval: str = "5minute",  # 5-minute for better noise filtering
    exchange: str = "NSE"
) -> dict:
    """
    Find when an indicator crossed a threshold (e.g., when RSI went below 30) for one or multiple instruments.
    
    Args:
        instrument_name: Single instrument name or list of names (e.g., "RELIANCE" or ["RELIANCE", "INFOSYS"])
        indicator: Indicator name (RSI, MACD, BB)
        threshold: Threshold value (e.g., 30 for RSI)
        direction: "below" (crossed below threshold) or "above" (crossed above threshold)
        date: Date in YYYY-MM-DD format (default: today)
        interval: Time interval (minute for 1min, 5minute, etc.)
        exchange: Exchange (NSE, NFO, BSE)
        
    Returns:
        dict with list of timestamps when indicator crossed threshold for single instrument, or aggregated results for multiple
    """
    try:
        # Normalize to list
        if isinstance(instrument_name, str):
            instrument_names = [instrument_name]
        else:
            instrument_names = instrument_name
        
        # Parse date
        if date:
            start_dt = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            start_dt = datetime.now().date()
        
        kite = get_kite_instance()
        
        # Helper function to process single instrument
        def process_instrument(inst_name):
            # Resolve instrument name
            instrument_info = resolve_instrument_name(inst_name, exchange)
            if not instrument_info:
                return {
                    "status": "error",
                    "error": f"Instrument '{inst_name}' not found"
                }
            
            instrument_token = instrument_info["instrument_token"]
            
            # Get historical data for the date
            historical_data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=start_dt,
                to_date=start_dt,
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
            closes = df['close'].values
            
            # Calculate RSI
            if indicator.upper() == "RSI":
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
                    
                    # Pad to match closes length
                    rsi_full = np.concatenate([[np.nan], rsi])
                    
                    # Find crossings
                    crossings = []
                    for idx in range(1, len(rsi_full)):
                        prev_rsi = rsi_full[idx - 1]
                        curr_rsi = rsi_full[idx]
                        
                        if np.isnan(prev_rsi) or np.isnan(curr_rsi):
                            continue
                        
                        # Check for crossing
                        if direction.lower() == "below":
                            # Crossed below threshold (was above, now below)
                            if prev_rsi >= threshold and curr_rsi < threshold:
                                candle = historical_data[idx] if idx < len(historical_data) else historical_data[-1]
                                crossings.append({
                                    "timestamp": candle.get("date", ""),
                                    "rsi": float(curr_rsi),
                                    "previous_rsi": float(prev_rsi),
                                    "close": float(candle.get("close", 0)),
                                    "candle_index": idx
                                })
                        elif direction.lower() == "above":
                            # Crossed above threshold (was below, now above)
                            if prev_rsi <= threshold and curr_rsi > threshold:
                                candle = historical_data[idx] if idx < len(historical_data) else historical_data[-1]
                                crossings.append({
                                    "timestamp": candle.get("date", ""),
                                    "rsi": float(curr_rsi),
                                    "previous_rsi": float(prev_rsi),
                                    "close": float(candle.get("close", 0)),
                                    "candle_index": idx
                                })
                    
                    return {
                        "status": "success",
                        "instrument": instrument_info["tradingsymbol"],
                        "indicator": indicator,
                        "threshold": threshold,
                        "direction": direction,
                        "date": str(start_dt),
                        "crossings": crossings,
                        "count": len(crossings),
                        "current_rsi": float(rsi[-1]) if not np.isnan(rsi[-1]) else None
                    }
                else:
                    return {
                        "status": "error",
                        "error": "Insufficient data for RSI calculation"
                    }
            else:
                return {
                    "status": "error",
                    "error": f"Indicator '{indicator}' not yet supported. Currently supports: RSI"
                }
        
        # Process single or multiple instruments
        if len(instrument_names) == 1:
            # Single instrument - return simple format
            return process_instrument(instrument_names[0])
        else:
            # Multiple instruments - return aggregated format
            results = {}
            for inst_name in instrument_names:
                result = process_instrument(inst_name)
                if result.get("status") == "success":
                    results[result.get("instrument", inst_name)] = result
                else:
                    results[inst_name] = result
            
            total_crossings = sum(r.get("count", 0) for r in results.values() if isinstance(r, dict) and r.get("status") == "success")
            
            return {
                "status": "success",
                "indicator": indicator,
                "threshold": threshold,
                "direction": direction,
                "date": str(start_dt),
                "instruments_analyzed": len(instrument_names),
                "total_crossings": total_crossings,
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
            "error": f"Error finding threshold crossings: {str(e)}"
        }


@tool
def get_indicator_history(
    instrument_name: Union[str, List[str]],
    indicator: str = "RSI",
    date: Optional[str] = None,
    interval: str = "5minute",
    exchange: str = "NSE"
) -> dict:
    """
    Get historical indicator values for one or multiple instruments on a specific date.
    
    Args:
        instrument_name: Single instrument name or list of names (e.g., "RELIANCE" or ["RELIANCE", "INFOSYS"])
        indicator: Indicator name (RSI, MACD, BB)
        date: Date in YYYY-MM-DD format (default: today)
        interval: Time interval (minute, 5minute, etc.)
        exchange: Exchange (NSE, NFO, BSE)
        
    Returns:
        dict with historical indicator values for single instrument, or aggregated results for multiple
    """
    try:
        # Resolve instrument name
        instrument_token = get_instrument_token(instrument_name, exchange)
        if not instrument_token:
            return {
                "status": "error",
                "error": f"Instrument '{instrument_name}' not found"
            }
        
        # Use calculate_indicators_tool with return_historical=True
        result = calculate_indicators_tool.invoke({
            "instrument_token": instrument_token,
            "interval": interval,
            "indicators": indicator,
            "return_historical": True
        })
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error getting indicator history: {str(e)}"
        }

