"""
Tools for market simulation and local data management
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Union, Optional
from langchain_core.tools import tool
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.kite_utils import get_kite_instance
from agent.tools.instrument_resolver import resolve_instrument_name, get_instrument_token, INSTRUMENT_GROUPS

DATA_DIR = Path("data/simulation")
DATA_DIR.mkdir(parents=True, exist_ok=True)

@tool
def download_historical_data_to_local_tool(
    instrument_names: Union[str, List[str]],
    from_date: str,
    to_date: Optional[str] = None,
    interval: str = "5minute",
    exchange: str = "NSE"
) -> dict:
    """
    Download historical data from Kite and save it locally for offline simulation and testing.
    
    Args:
        instrument_names: Single instrument name, list of names, or a group name like 'top 10 nifty50 stocks'.
        from_date: Start date in YYYY-MM-DD format.
        to_date: End date in YYYY-MM-DD format (defaults to today).
        interval: Time interval (minute, 5minute, 15minute, day).
        exchange: Exchange name (NSE, NFO, BSE).
        
    Returns:
        dict with download status and file path.
    """
    try:
        kite = get_kite_instance()
        
        # 1. Resolve instruments (handle groups)
        if isinstance(instrument_names, str):
            inst_lower = instrument_names.lower().strip()
            if inst_lower in INSTRUMENT_GROUPS:
                target_names = INSTRUMENT_GROUPS[inst_lower]
            else:
                target_names = [instrument_names]
        else:
            target_names = instrument_names
            
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
            
        start_dt = datetime.strptime(from_date, "%Y-%m-%d")
        end_dt = datetime.strptime(to_date, "%Y-%m-%d")
        
        # Ensure to_date includes the entire day by setting time to 23:59:59
        end_dt_full = end_dt.replace(hour=23, minute=59, second=59)
        
        # We fetch slightly more data (lookback) for indicator calculations
        fetch_start = start_dt - timedelta(days=5)
        
        all_data = {}
        downloaded_count = 0
        
        for name in target_names:
            resolved = resolve_instrument_name(name, exchange)
            if not resolved:
                continue
                
            token = resolved["instrument_token"]
            symbol = resolved["tradingsymbol"]
            
            try:
                # Use the end_dt with full time to ensure we get the last day's data
                data = kite.historical_data(token, fetch_start, end_dt_full, interval)
                if data:
                    # Convert datetime objects to strings for JSON serialization
                    for candle in data:
                        if isinstance(candle.get('date'), datetime):
                            candle['date'] = candle['date'].isoformat()
                    
                    all_data[symbol] = data
                    downloaded_count += 1
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
                
        if not all_data:
            return {"status": "error", "error": "No data could be downloaded for the specified instruments."}
            
        # Save to local file
        filename = f"sim_data_{from_date}_to_{to_date}_{interval}.json".replace(":", "-")
        filepath = DATA_DIR / filename
        
        with open(filepath, "w") as f:
            json.dump({
                "metadata": {
                    "from_date": from_date,
                    "to_date": to_date,
                    "interval": interval,
                    "exchange": exchange,
                    "download_time": datetime.now().isoformat()
                },
                "data": all_data
            }, f, indent=2)
            
        return {
            "status": "success",
            "message": f"Successfully downloaded data for {downloaded_count} instruments.",
            "file_path": str(filepath),
            "instruments": list(all_data.keys()),
            "total_candles": sum(len(v) for v in all_data.values())
        }
        
    except Exception as e:
        return {"status": "error", "error": f"Download failed: {str(e)}"}

@tool
def run_simulation_on_local_data_tool(
    file_path: Optional[str] = None,
    strategy: str = "VWAP"
) -> dict:
    """
    Run a simulated trading session using previously downloaded local data.
    This simulates live market conditions candle-by-candle.
    
    Args:
        file_path: Path to the local JSON data file (optional, will use most recent if empty).
        strategy: The strategy to test (only 'VWAP' supported currently).
        
    Returns:
        dict with simulation results, P&L, and trade list.
    """
    try:
        # SMART PATH FILLING: If no path provided, pick most recent
        if not file_path:
            sim_dir = "data/simulation"
            if os.path.exists(sim_dir):
                files = sorted([f for f in os.listdir(sim_dir) if f.endswith(".json")], reverse=True)
                if files:
                    file_path = os.path.join(sim_dir, files[0])
        
        if not file_path or not os.path.exists(file_path):
            return {"status": "error", "error": f"File not found: {file_path}"}
            
        # Instead of just loading, let's actually RUN the trading opportunities tool logic
        from agent.tools.trading_opportunities_tool import find_indicator_based_trading_opportunities
        
        # We need to extract instruments from the file to tell the tool what to analyze
        with open(file_path, "r") as f:
            sim_content = json.load(f)
            all_data = sim_content.get("data", {})
            instruments = list(all_data.keys())
            
        if not instruments:
            return {"status": "error", "error": "No instruments found in simulation file."}

        # Call the main analysis tool with the local file
        return find_indicator_based_trading_opportunities.invoke({
            "instrument_name": instruments,
            "local_data_file": file_path,
            "interval": "minute" if "minute" in file_path else "5minute"
        })
        
    except Exception as e:
        return {"status": "error", "error": f"Simulation failed: {str(e)}"}

