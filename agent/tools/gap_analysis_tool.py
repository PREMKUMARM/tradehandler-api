"""
Gap analysis tools for predicting opening gaps
"""
from typing import Optional, Dict, Any, Union, List
from langchain_core.tools import tool
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.kite_utils import get_kite_instance
from agent.tools.instrument_resolver import resolve_instrument_name
from kiteconnect.exceptions import KiteException


@tool
def analyze_gap_probability(
    instrument_name: Union[str, List[str]],
    exchange: str = "NSE"
) -> dict:
    """
    Analyze the probability of gap up or gap down opening for one or multiple instruments.
    Uses historical gap patterns, current price, and recent volatility.
    
    Args:
        instrument_name: Single instrument name or list of names (e.g., "RELIANCE" or ["RELIANCE", "INFOSYS"])
        exchange: Exchange (NSE, NFO, BSE)
        
    Returns:
        dict with gap analysis for single instrument, or aggregated results for multiple instruments
    """
    try:
        # Normalize to list
        if isinstance(instrument_name, str):
            instrument_names = [instrument_name]
        else:
            instrument_names = instrument_name
        
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
            
            # Get current quote
            quote_key = f"{instrument_info['exchange']}:{instrument_info['tradingsymbol']}"
            quotes = kite.quote(quote_key)
            current_quote = quotes.get(quote_key, {})
            current_price = current_quote.get("last_price", 0)
            
            if not current_price:
                return {
                    "status": "error",
                    "error": "Could not get current price"
                }
            
            # Get historical data for gap analysis (last 30 days)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            
            historical_data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=start_date,
                to_date=end_date,
                interval="day"
            )
            
            if not historical_data or len(historical_data) < 5:
                return {
                    "status": "error",
                    "error": "Insufficient historical data for gap analysis"
                }
            
            import pandas as pd
            import numpy as np
            
            df = pd.DataFrame(historical_data)
            df = df.sort_values('date')
            
            # Calculate gaps
            gaps = []
            gap_up_count = 0
            gap_down_count = 0
            gap_up_percentages = []
            gap_down_percentages = []
            
            for i in range(1, len(df)):
                prev_close = df.iloc[i-1]['close']
                curr_open = df.iloc[i]['open']
                
                gap_pct = ((curr_open - prev_close) / prev_close) * 100
                
                if gap_pct > 0:
                    gap_up_count += 1
                    gap_up_percentages.append(gap_pct)
                elif gap_pct < 0:
                    gap_down_count += 1
                    gap_down_percentages.append(gap_pct)
                
                gaps.append({
                    "date": df.iloc[i]['date'],
                    "gap_percentage": gap_pct,
                    "prev_close": prev_close,
                    "open": curr_open
                })
            
            # Calculate statistics
            total_days = len(gaps)
            gap_up_probability = (gap_up_count / total_days * 100) if total_days > 0 else 0
            gap_down_probability = (gap_down_count / total_days * 100) if total_days > 0 else 0
            
            avg_gap_up = np.mean(gap_up_percentages) if gap_up_percentages else 0
            avg_gap_down = np.mean(gap_down_percentages) if gap_down_percentages else 0
            
            # Analyze recent volatility
            recent_data = df.tail(5)
            recent_volatility = recent_data['close'].pct_change().std() * 100 if len(recent_data) > 1 else 0
            
            # Get last close price
            last_close = df.iloc[-1]['close']
            
            # Check for today's actual gap if data includes today
            actual_gap_today = None
            if len(df) >= 2:
                # Assuming the last row is today if it's during/after market hours
                # and the previous row is the previous trading day
                last_row = df.iloc[-1]
                prev_row = df.iloc[-2]
                
                # Check if last_row date is today
                if last_row['date'].date() == datetime.now().date():
                    actual_gap_val = last_row['open'] - prev_row['close']
                    actual_gap_pct = (actual_gap_val / prev_row['close']) * 100
                    actual_gap_today = {
                        "gap_value": float(actual_gap_val),
                        "gap_percentage": float(actual_gap_pct),
                        "direction": "GAP UP" if actual_gap_pct > 0 else "GAP DOWN" if actual_gap_pct < 0 else "NO GAP",
                        "open": float(last_row['open']),
                        "prev_close": float(prev_row['close'])
                    }

            # Calculate potential gap scenarios for NEXT open
            if gap_up_probability > gap_down_probability:
                likely_direction = "GAP UP"
                probability = gap_up_probability
                estimated_gap_pct = avg_gap_up
            else:
                likely_direction = "GAP DOWN"
                probability = gap_down_probability
                estimated_gap_pct = abs(avg_gap_down)
            
            # Get current market sentiment indicators
            now = datetime.now()
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            is_market_open = market_open <= now <= market_close
            
            # Analyze recent trend
            recent_trend = "NEUTRAL"
            if len(df) >= 3:
                recent_closes = df.tail(3)['close'].values
                if recent_closes[-1] > recent_closes[0]:
                    recent_trend = "BULLISH"
                elif recent_closes[-1] < recent_closes[0]:
                    recent_trend = "BEARISH"
            
            return {
                "status": "success",
                "instrument": instrument_info["tradingsymbol"],
                "current_price": float(current_price),
                "last_close": float(last_close),
                "is_market_open": is_market_open,
                "recent_trend": recent_trend,
                "recent_volatility": float(recent_volatility),
                "actual_gap_today": actual_gap_today,
                "gap_analysis": {
                    "likely_direction": likely_direction,
                    "probability": float(probability),
                    "estimated_gap_percentage": float(estimated_gap_pct),
                    "gap_up_probability": float(gap_up_probability),
                    "gap_down_probability": float(gap_down_probability),
                    "avg_gap_up_pct": float(avg_gap_up),
                    "avg_gap_down_pct": float(abs(avg_gap_down)),
                    "total_days_analyzed": total_days,
                    "gap_up_days": gap_up_count,
                    "gap_down_days": gap_down_count
                },
                "historical_gaps": gaps[-10:],  # Last 10 gaps
                "disclaimer": "This is a statistical analysis based on historical patterns. Actual opening may vary based on overnight news, global markets, and other factors."
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
            
            return {
                "status": "success",
                "instruments_analyzed": len(instrument_names),
                "results": results
            }
        
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": f"Error analyzing gap: {str(e)}",
            "traceback": traceback.format_exc()
        }

