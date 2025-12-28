from typing import Union
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Response, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio

import json
from pprint import pprint
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

# Kite utilities
from utils.kite_utils import (
    api_key,
    get_access_token,
    get_kite_instance,
    calculate_trend_and_suggestions
)

# Agent imports
from agent.graph import run_agent, get_agent_instance, get_agent_memory
from agent.approval import get_approval_queue
from agent.safety import get_safety_manager
from agent.config import get_agent_config
from agent.autonomous import start_autonomous_agent
from agent.ws_manager import manager, broadcast_agent_update, add_agent_log
from agent.tools.kite_tools import place_order_tool, cancel_order_tool, place_gtt_tool
from database.connection import init_database
from database.repositories import (
    get_log_repository, get_tool_repository, get_simulation_repository,
    get_config_repository, get_chat_repository
)
from database.models import ChatMessage
import uuid

# Initialize FastAPI app with enterprise-level configuration
from core.config import get_settings
from core.responses import APIResponse

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise AI Trading Agent API with Zerodha Kite Connect integration",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add enterprise middleware (order matters!)
from middleware import RequestIDMiddleware, LoggingMiddleware, ErrorHandlerMiddleware

app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlerMiddleware)

# Include API v1 routes
from api.v1 import api_router
app.include_router(api_router)

# Import utilities and modules
from utils.candle_utils import aggregate_to_tf, analyze_trend
from utils.indicators import (
    calculate_bollinger_bands,
    calculate_bollinger_bands_full,
    calculate_rsi,
    calculate_pivot_points,
    calculate_support_resistance
)
from simulation import (
    simulation_state,
    live_logs,
    get_instrument_history,
    add_sim_order,
    calculate_sim_qty,
    find_option
)
from tasks import live_market_scanner, monitor_order_execution
from strategies.runner import run_strategy_on_candles

# Legacy endpoints (maintained for backward compatibility)
# These will be gradually migrated to v1 routes

@app.websocket("/ws/agent")
async def agent_websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages if needed
            data = await websocket.receive_text()
            # For now, we don't need to handle client -> server messages
            # but we need to receive them to keep the connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WS] Error: {e}")
        manager.disconnect(websocket)

# Helper to send agent updates - Redundant definition removed

# Kite Connect credentials - All loaded from environment variables
# IMPORTANT: The redirect_uri must EXACTLY match what's configured in your Kite Connect app settings
# Global API key handled by utils.kite_utils
# All secrets should be in .env file, never hardcoded
api_secret = os.getenv('KITE_API_SECRET')
redirect_uri = os.getenv('KITE_REDIRECT_URI', 'http://localhost:4200/auth-token')

# Note: api_secret validation is done in utils.kite_utils when actually needed
# This allows the server to start even if Kite credentials aren't configured yet

# CORS configuration from settings (already loaded above)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to get access token from file
# Helper function to get KiteConnect instance
# Moved to api/v1/routes/portfolio.py
# @app.get("/live-positions")
# def get_live_positions():
#     """Fetch current open positions from Zerodha Kite"""
#     try:
#         kite = get_kite_instance()
#         positions = kite.positions()
#         
#         # Calculate live MTM and totals
#         net_positions = positions.get("net", [])
#         total_pnl = 0
#         active_count = 0
#         
#         for pos in net_positions:
#             total_pnl += pos.get("pnl", 0)
#             if pos.get("quantity", 0) != 0:
#                 active_count += 1
#                 
#         return {
#             "data": {
#                 "positions": net_positions,
#                 "total_pnl": round(total_pnl, 2),
#                 "active_count": active_count
#             }
#         }
#     except KiteException as e:
#         raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching positions: {str(e)}")

# Moved to api/v1/routes/orders.py
# @app.post("/place-strategy-order")
# async def place_strategy_order(req: Request):
#     """Place a live order based on a strategy signal"""
#     ... (function body moved to api/v1/routes/orders.py)

# Moved to api/v1/routes/orders.py
# @app.post("/exit-all-positions")
# def exit_all_positions():
#     """Kill Switch: Exit all open positions immediately at MARKET price"""
#     ... (function body moved to api/v1/routes/orders.py)

from concurrent.futures import ThreadPoolExecutor

# Moved to tasks/market_scanner.py
# async def live_market_scanner():
#     """Background task to scan market and execute strategies"""
#     ... (function body moved to tasks/market_scanner.py)

# Moved to tasks/market_scanner.py
# async def monitor_order_execution():
#     """
#     Background task to monitor order execution and auto-cancel remaining orders.
#     When SL or Target executes, the other order should be cancelled automatically.
#     """
#     ... (function body moved to tasks/market_scanner.py)

@app.on_event("startup")
async def startup_event():
    # Initialize database
    init_database()
    add_agent_log("Database initialized successfully", "info", "system")

    # Set approval queue callback for WebSocket broadcasting
    approval_queue = get_approval_queue()
    def broadcast_new_approval(approval):
        asyncio.create_task(broadcast_agent_update("NEW_APPROVAL", approval))

    approval_queue.on_create_callback = broadcast_new_approval

    # Start the background scanner
    asyncio.create_task(live_market_scanner())
    # Start the new AI Agent autonomous scanner
    start_autonomous_agent()
    # Start order monitoring task for auto-cancellation
    asyncio.create_task(monitor_order_execution())

# Simulation state and helpers moved to simulation/ module

# Moved to api/v1/routes/simulation.py
# @app.get("/live-logs")
# async def get_live_logs():
#     """Endpoint for UI to fetch latest monitoring logs"""
#     return {"data": live_logs}

# Moved to api/v1/routes/simulation.py
# @app.post("/simulation/speed")
# async def set_simulation_speed(req: Request):
    payload = await req.json()
    speed = payload.get("speed", 1)
    simulation_state["speed"] = max(1, speed)
    return {"status": "success", "speed": simulation_state["speed"]}

# Moved to api/v1/routes/simulation.py
# @app.post("/simulation/start")
# async def start_simulation(req: Request):
    """Start a live-market simulation using historical data"""
    try:
        global live_logs
        live_logs = [] # Clear previous logs
        add_agent_log(f"Initializing simulation...", "info")
        
        kite = get_kite_instance()
        payload = await req.json()
        sim_date_str = payload.get("date", (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
        sim_date = datetime.strptime(sim_date_str, "%Y-%m-%d").date()
        
        add_agent_log(f"Loading historical data for {sim_date_str}...", "info")
        
        # Get Nifty index token
        nse_instruments = kite.instruments("NSE")
        nifty_index = next((inst for inst in nse_instruments if inst.get("tradingsymbol") == "NIFTY 50"), None)
        if not nifty_index: return {"status": "error", "message": "Nifty not found"}
        
        # Fetch 1-minute candles for the chosen day + previous 2 days for context
        # This allows indicators to calculate properly and shows historical context
        start_date = sim_date - timedelta(days=2)
        raw_candles = kite.historical_data(nifty_index["instrument_token"], start_date, sim_date + timedelta(days=1), "minute")
        
        # Filter candles for market hours (09:15 to 15:30 IST)
        all_candles = []
        for candle in raw_candles:
            candle_time = candle["date"].time()
            if time(9, 15) <= candle_time <= time(15, 30):
                all_candles.append(candle)
        
        # Separate: previous days (for context) and simulation day (for replay)
        candles = []  # Simulation day candles
        previous_candles = []  # Previous days for context
        for candle in all_candles:
            candle_date = candle["date"].date()
            if candle_date == sim_date:
                candles.append(candle)
            elif candle_date < sim_date:
                previous_candles.append(candle)

        if not candles:
            return {"status": "error", "message": f"No trading hour data for {sim_date_str}"}
        
        # previous_candles is already populated from the loop above - don't overwrite it!
        # Store previous days' candles for chart context
            
        # Get nifty options for this date (needed by strategies)
        nifty_options = kite.instruments("NFO")
        nifty_options = [inst for inst in nifty_options if inst.get("name") == "NIFTY"]
        
        # Find ATM options for simulation start
        nifty_price = candles[0]["close"]
        current_strike = round(nifty_price / 50) * 50
        
        # Find nearest expiry >= sim_date
        expiries = sorted(list(set([inst["expiry"] for inst in nifty_options if inst.get("expiry")])))
        nearest_expiry = None
        for exp in expiries:
            if exp >= sim_date:
                nearest_expiry = exp
                break
        
        atm_ce = next((inst for inst in nifty_options if inst["expiry"] == nearest_expiry and inst["strike"] == current_strike and inst["instrument_type"] == "CE"), None)
        atm_pe = next((inst for inst in nifty_options if inst["expiry"] == nearest_expiry and inst["strike"] == current_strike and inst["instrument_type"] == "PE"), None)

        simulation_state.update({
            "is_active": True,
            "date": sim_date_str,
            "current_index": 0,
            "speed": 1,
            "candles": candles,
            "previous_candles": previous_candles,  # Store previous days' candles for chart display
            "instrument_history": {}, # Reset cache
            "nifty_options": nifty_options,
            "atm_ce": atm_ce,
            "atm_pe": atm_pe,
            "positions": [],
            "orders": [], # Reset orders
            "executed_strategies": set(),
            "nifty_price": nifty_price,
            "last_update": datetime.now().isoformat()
        })
        
        return {"status": "success", "message": f"Simulation started for {sim_date_str}", "total_minutes": len(candles)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Moved to strategies/runner.py
# async def run_strategy_on_candles(...):

# Moved to simulation/helpers.py
# def get_instrument_history(...):
# def add_sim_order(...):
# def calculate_sim_qty(...):

# Moved to api/v1/routes/simulation.py
# @app.get("/simulation/chart-data")
# async def get_simulation_chart_data(timeframe: str = "1m", indicators: str = ""):
    """Get chart data for simulation replay - returns candles up to current index
    timeframe: 1m, 5m, 15m, 30m, 1h, 1d
    indicators: comma-separated list (e.g., "rsi,bollinger,pivot")
    """
    if not simulation_state["is_active"]:
        return {"data": {"candles": [], "current_index": 0, "total_candles": 0, "timeframe": timeframe}}
    
    idx = simulation_state["current_index"]
    candles_1m = simulation_state["candles"]
    previous_candles = simulation_state.get("previous_candles", [])
    
    # Map timeframe to minutes
    tf_map = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "1d": 1440  # Approximate (6.5 hours * 60 minutes)
    }
    
    tf_minutes = tf_map.get(timeframe, 1)
    
    # CRITICAL: Use previous days' candles for indicator calculation context
    # We need enough historical data for indicators:
    # - RSI (14 period): needs at least 15 candles
    # - Bollinger Bands (20 period): needs at least 20 candles
    # - Pivot Points: needs previous day's H/L/C
    
    # Determine how many previous candles we need based on requested indicators
    max_period_needed = 20  # Bollinger Bands needs most (20 period)
    if indicators:
        indicator_list = [ind.strip().lower() for ind in indicators.split(",") if ind.strip()]
        if "bollinger" in indicator_list:
            max_period_needed = 20
        elif "rsi" in indicator_list:
            max_period_needed = 14
    
    # Get enough previous candles for indicator calculation
    # For aggregated timeframes, we need more 1-minute candles
    # Example: For 30-minute candles with RSI(14), we need 14*30 = 420 minutes = 7 hours
    # But we also need buffer, so let's calculate: max_period_needed * tf_minutes * 2
    # For 30-minute: 14 * 30 * 2 = 840 minutes = 14 hours of data
    # Market hours: 9:15 to 15:30 = 6.25 hours per day
    # So we need at least 3 days of data for 30-minute RSI
    required_prev_candles = max_period_needed * tf_minutes * 2  # 2x buffer for safety
    
    # If we don't have enough previous candles, try to fetch more
    if len(previous_candles) < required_prev_candles and simulation_state.get("date"):
        try:
            kite = get_kite_instance()
            nse_instruments = kite.instruments("NSE")
            nifty_index = next((inst for inst in nse_instruments if inst.get("tradingsymbol") == "NIFTY 50"), None)
            if nifty_index:
                sim_date = simulation_state["date"]
                # Calculate how many days we need
                market_hours_per_day = 6.25  # 9:15 to 15:30 = 6.25 hours
                minutes_per_day = market_hours_per_day * 60  # 375 minutes per day
                days_needed = int((required_prev_candles / minutes_per_day) + 1)  # +1 for safety
                days_needed = max(days_needed, 3)  # At least 3 days
                
                # Fetch more historical data
                start_date = sim_date - timedelta(days=days_needed)
                raw_prev = kite.historical_data(nifty_index["instrument_token"], start_date, sim_date, "minute")
                additional_candles = []
                for candle in raw_prev:
                    candle_time = candle["date"].time()
                    if time(9, 15) <= candle_time <= time(15, 30):
                        additional_candles.append(candle)
                
                # Merge with existing previous candles (avoid duplicates)
                existing_timestamps = {int(c["date"].timestamp()) for c in previous_candles}
                for candle in additional_candles:
                    ts = int(candle["date"].timestamp())
                    if ts not in existing_timestamps:
                        previous_candles.append(candle)
                        existing_timestamps.add(ts)
                
                # Sort by timestamp
                previous_candles.sort(key=lambda x: int(x["date"].timestamp()) if isinstance(x["date"], datetime) else 0)
                simulation_state["previous_candles"] = previous_candles
                print(f"Fetched additional historical data: {len(previous_candles)} total previous candles for {days_needed} days")
        except Exception as e:
            print(f"Warning: Could not fetch additional historical candles: {e}")
    
    previous_for_calc = previous_candles[-required_prev_candles:] if len(previous_candles) > required_prev_candles else previous_candles
    print(f"Using {len(previous_for_calc)} previous candles for calculation (required: {required_prev_candles} for {timeframe} timeframe)")
    
    # Combine: previous (for calculation) + current simulation candles
    # This ensures we have enough historical data for accurate indicator calculation
    # For display: show ALL previous candles (not limited) + current simulation candles up to index
    # This allows users to see historical context before simulation starts
    all_candles_for_display = previous_candles + candles_1m[:idx + 1]  # All previous + current up to index
    all_candles_for_calc = previous_for_calc + candles_1m[:idx + 1]  # Enough for calculation
    
    if not all_candles_for_display:
        return {"data": {"candles": [], "current_index": idx, "total_candles": len(candles_1m), "timeframe": timeframe}}
    
    # Aggregate to selected timeframe - use TIME-BASED aggregation (not sequential chunking)
    # This ensures candles are grouped by actual time windows (e.g., 09:00-09:30, 09:30-10:00)
    def aggregate_candles(candle_list):
        if tf_minutes == 1:
            return candle_list
        
        if not candle_list:
            return []
        
        aggregated = []
        current_chunk = []
        current_window_start = None
        
        for candle in candle_list:
            # Get candle timestamp
            if isinstance(candle["date"], datetime):
                candle_dt = candle["date"]
            else:
                try:
                    candle_dt = datetime.fromisoformat(str(candle["date"]).replace('Z', '+00:00'))
                except:
                    continue
            
            # Calculate which time window this candle belongs to
            # Round down to nearest tf_minutes boundary
            minutes = candle_dt.minute
            rounded_minutes = (minutes // tf_minutes) * tf_minutes
            window_start = candle_dt.replace(minute=rounded_minutes, second=0, microsecond=0)
            
            # If this is a new time window, finalize previous chunk and start new one
            if current_window_start is None or window_start != current_window_start:
                # Finalize previous chunk
                if current_chunk:
                    aggregated.append({
                        "date": current_chunk[0]["date"],
                        "open": float(current_chunk[0]["open"]),
                        "high": float(max(c["high"] for c in current_chunk)),
                        "low": float(min(c["low"] for c in current_chunk)),
                        "close": float(current_chunk[-1]["close"]),
                        "volume": int(sum(c.get("volume", 0) for c in current_chunk))
                    })
                
                # Start new chunk
                current_chunk = [candle]
                current_window_start = window_start
            else:
                # Add to current chunk
                current_chunk.append(candle)
        
        # Don't forget the last chunk
        if current_chunk:
            aggregated.append({
                "date": current_chunk[0]["date"],
                "open": float(current_chunk[0]["open"]),
                "high": float(max(c["high"] for c in current_chunk)),
                "low": float(min(c["low"] for c in current_chunk)),
                "close": float(current_chunk[-1]["close"]),
                "volume": int(sum(c.get("volume", 0) for c in current_chunk))
            })
        
        return aggregated
    
    # Aggregate for display (with limited previous candles for visual context)
    aggregated_candles = aggregate_candles(all_candles_for_display)
    
    # Aggregate for calculation (with more previous candles for accurate indicators)
    aggregated_for_indicators = aggregate_candles(all_candles_for_calc)
    
    # Find the starting index in aggregated_for_indicators that matches the first display candle
    # This allows us to extract indicator values for display candles
    first_display_ts = None
    if aggregated_candles:
        first_candle = aggregated_candles[0]
        if isinstance(first_candle["date"], datetime):
            first_display_ts = int(first_candle["date"].timestamp())
        else:
            try:
                dt = datetime.fromisoformat(str(first_candle["date"]).replace('Z', '+00:00'))
                first_display_ts = int(dt.timestamp())
            except:
                first_display_ts = 0
    
    # Find matching start index in aggregated_for_indicators
    calc_start_idx = 0
    if first_display_ts:
        for i, calc_candle in enumerate(aggregated_for_indicators):
            if isinstance(calc_candle["date"], datetime):
                calc_ts = int(calc_candle["date"].timestamp())
            else:
                try:
                    dt = datetime.fromisoformat(str(calc_candle["date"]).replace('Z', '+00:00'))
                    calc_ts = int(dt.timestamp())
                except:
                    calc_ts = 0
            if calc_ts == first_display_ts:
                calc_start_idx = i
                break
            elif calc_ts > first_display_ts:
                # Found a candle after our display start, use previous one
                calc_start_idx = max(0, i - 1)
                break
    
    # Convert to chart format - ensure proper sorting by timestamp
    chart_candles = []
    for candle in aggregated_candles:
        # Convert datetime to Unix timestamp (seconds)
        if isinstance(candle["date"], datetime):
            timestamp = int(candle["date"].timestamp())
        else:
            # Try to parse string date
            try:
                dt = datetime.fromisoformat(str(candle["date"]).replace('Z', '+00:00'))
                timestamp = int(dt.timestamp())
            except:
                timestamp = int(datetime.now().timestamp())
        
        # Validate OHLC values
        open_val = float(candle.get("open", 0)) or 0
        high_val = float(candle.get("high", 0)) or open_val
        low_val = float(candle.get("low", 0)) or open_val
        close_val = float(candle.get("close", 0)) or open_val
        
        chart_candles.append({
            "time": timestamp,
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "volume": int(candle.get("volume", 0))
        })
    
    # CRITICAL: Sort by timestamp to ensure proper chronological order
    chart_candles.sort(key=lambda x: x["time"])
    
    # Get current positions and orders for markers
    positions = simulation_state.get("positions", [])
    orders = simulation_state.get("orders", [])
    
    # Markers for entry/exit points (for future chart library integration)
    markers = []
    for order in orders[:50]:  # Last 50 orders
        if order.get("order_timestamp"):
            # Find candle index for this order (use original 1m candles for marker positioning)
            order_time_str = order["order_timestamp"]
            for i, c in enumerate(candles_1m):
                c_time_str = c["date"].strftime("%H:%M:%S") if isinstance(c["date"], datetime) else str(c["date"])
                if order_time_str in c_time_str or c_time_str in order_time_str:
                    if isinstance(c["date"], datetime):
                        timestamp = int(c["date"].timestamp())
                    else:
                        try:
                            dt = datetime.fromisoformat(str(c["date"]).replace('Z', '+00:00'))
                            timestamp = int(dt.timestamp())
                        except:
                            timestamp = int(datetime.now().timestamp())
                    
                    markers.append({
                        "time": timestamp,
                        "position": "belowBar" if order["transaction_type"] == "BUY" else "aboveBar",
                        "color": "#26a69a" if order["transaction_type"] == "BUY" else "#ef5350",
                        "shape": "arrowUp" if order["transaction_type"] == "BUY" else "arrowDown",
                        "text": f"{order['strategy']} {order['transaction_type']}"
                    })
                    break
    
    # Calculate indicators if requested
    # CRITICAL: Use aggregated_for_indicators (which includes more historical data) for calculation
    # But map results back to aggregated_candles (display candles) for alignment
    indicator_data = {}
    if indicators:
        indicator_list = [ind.strip().lower() for ind in indicators.split(",") if ind.strip()]
        
        # Get closes/highs/lows for indicator calculation (use full historical dataset)
        closes_for_calc = [float(c["close"]) for c in aggregated_for_indicators]
        highs_for_calc = [float(c["high"]) for c in aggregated_for_indicators]
        lows_for_calc = [float(c["low"]) for c in aggregated_for_indicators]
        
        print(f"Indicator calculation: {len(aggregated_for_indicators)} calc candles, {len(aggregated_candles)} display candles, calc_start_idx={calc_start_idx}")
        
        if "rsi" in indicator_list:
            # Calculate RSI on full historical dataset (for accurate calculation)
            rsi_values_full = calculate_rsi(closes_for_calc, period=14)
            
            # Extract RSI values for display candles (starting from calc_start_idx)
            rsi_data = []
            for i, candle in enumerate(aggregated_candles):
                if isinstance(candle["date"], datetime):
                    timestamp = int(candle["date"].timestamp())
                else:
                    try:
                        dt = datetime.fromisoformat(str(candle["date"]).replace('Z', '+00:00'))
                        timestamp = int(dt.timestamp())
                    except:
                        timestamp = int(datetime.now().timestamp())
                
                # Map display candle index to calculation candle index
                calc_idx = calc_start_idx + i
                
                # Get RSI value from calculated array
                if calc_idx >= 0 and calc_idx < len(rsi_values_full):
                    rsi_val = rsi_values_full[calc_idx]
                    # Include valid RSI values (skip NaN)
                    if not (pd.isna(rsi_val) or np.isnan(rsi_val)):
                        rsi_data.append({
                            "time": timestamp,
                            "value": float(rsi_val)
                        })
            
            # CRITICAL: Sort RSI data by timestamp to match candles
            rsi_data.sort(key=lambda x: x["time"])
            indicator_data["rsi"] = rsi_data
            # Debug log
            print(f"RSI calculation: {len(closes_for_calc)} closes for calc, {len(aggregated_candles)} display candles, {len(rsi_values_full)} RSI values, {len(rsi_data)} valid RSI data points")
            if len(rsi_data) == 0 and len(closes_for_calc) >= 15:
                print(f"WARNING: No RSI data points! calc_start_idx={calc_start_idx}, First 20 RSI values: {rsi_values_full[:20] if len(rsi_values_full) > 0 else 'N/A'}")
        
        if "bollinger" in indicator_list:
            # Calculate Bollinger Bands on full historical dataset (for accurate calculation)
            upper_full, middle_full, lower_full = calculate_bollinger_bands_full(closes_for_calc, period=20, num_std=2)
            
            # Extract BB values for display candles (starting from calc_start_idx)
            bb_data = []
            for i, candle in enumerate(aggregated_candles):
                if isinstance(candle["date"], datetime):
                    timestamp = int(candle["date"].timestamp())
                else:
                    try:
                        dt = datetime.fromisoformat(str(candle["date"]).replace('Z', '+00:00'))
                        timestamp = int(dt.timestamp())
                    except:
                        timestamp = int(datetime.now().timestamp())
                
                # Map display candle index to calculation candle index
                calc_idx = calc_start_idx + i
                
                # Get BB values from calculated arrays
                if calc_idx >= 0 and calc_idx < len(upper_full):
                    # Only include valid values (not NaN)
                    if not (pd.isna(upper_full[calc_idx]) or pd.isna(middle_full[calc_idx]) or pd.isna(lower_full[calc_idx])):
                        bb_data.append({
                            "time": timestamp,
                            "upper": float(upper_full[calc_idx]),
                            "middle": float(middle_full[calc_idx]),
                            "lower": float(lower_full[calc_idx])
                        })
            
            # CRITICAL: Sort Bollinger Bands data by timestamp to match candles
            bb_data.sort(key=lambda x: x["time"])
            indicator_data["bollinger"] = bb_data
            print(f"Bollinger calculation: {len(closes_for_calc)} closes for calc, {len(aggregated_candles)} display candles, calc_start_idx={calc_start_idx}, {len(bb_data)} valid BB data points")
        
        if "pivot" in indicator_list and len(aggregated_candles) > 0:
            # Calculate pivot points from previous day's high, low, close
            # Use the calculation dataset to get previous day's data (more accurate)
            if len(highs_for_calc) > 0 and len(lows_for_calc) > 0 and len(closes_for_calc) > 0:
                # Get previous day's data from the calculation dataset
                # Use first 20% of calculation candles to determine previous day's range
                # This gives us a better representation of the previous trading day
                prev_day_count = max(1, len(aggregated_for_indicators) // 5)
                prev_day_candles = aggregated_for_indicators[:prev_day_count]
                
                if prev_day_candles:
                    prev_high = max(float(c["high"]) for c in prev_day_candles)
                    prev_low = min(float(c["low"]) for c in prev_day_candles)
                    prev_close = float(prev_day_candles[-1]["close"])
                else:
                    # Fallback: use first few display candles
                    prev_high = max(highs[:20] if len(highs) >= 20 else highs) if len(highs) > 0 else highs[0]
                    prev_low = min(lows[:20] if len(lows) >= 20 else lows) if len(lows) > 0 else lows[0]
                    prev_close = closes[0] if len(closes) > 0 else closes[-1]
                
                pivot_points = calculate_pivot_points(prev_high, prev_low, prev_close)
                
                # Get time range for horizontal lines (first to last display candle)
                first_candle = aggregated_candles[0]
                last_candle = aggregated_candles[-1]
                
                def get_timestamp(candle):
                    if isinstance(candle["date"], datetime):
                        return int(candle["date"].timestamp())
                    else:
                        try:
                            dt = datetime.fromisoformat(str(candle["date"]).replace('Z', '+00:00'))
                            return int(dt.timestamp())
                        except:
                            return int(datetime.now().timestamp())
                
                first_time = get_timestamp(first_candle)
                last_time = get_timestamp(last_candle)
                
                # Ensure all values are properly formatted as floats
                indicator_data["pivot"] = {
                    "time_start": first_time,
                    "time_end": last_time,
                    "pivot": float(pivot_points.get("pivot", 0)),
                    "r1": float(pivot_points.get("r1", 0)),
                    "r2": float(pivot_points.get("r2", 0)),
                    "r3": float(pivot_points.get("r3", 0)),
                    "s1": float(pivot_points.get("s1", 0)),
                    "s2": float(pivot_points.get("s2", 0)),
                    "s3": float(pivot_points.get("s3", 0))
                }
                print(f"Pivot points calculated: P={indicator_data['pivot']['pivot']:.2f}, R1={indicator_data['pivot']['r1']:.2f}, R2={indicator_data['pivot']['r2']:.2f}, S1={indicator_data['pivot']['s1']:.2f}, S2={indicator_data['pivot']['s2']:.2f}")
    
    return {
        "data": {
            "candles": chart_candles,
            "markers": markers,
            "current_index": idx,
            "total_candles": len(candles_1m),
            "current_time": simulation_state.get("time", ""),
            "nifty_price": simulation_state.get("nifty_price", 0),
            "timeframe": timeframe,
            "indicators": indicator_data
        }
    }

# Moved to api/v1/routes/simulation.py
# @app.get("/simulation/state")
# async def get_simulation_state():
    """Advance simulation by 1 minute and run active strategies"""
    if not simulation_state["is_active"]:
        return {"data": {"is_active": False}}
        
    idx = simulation_state["current_index"]
    candles = simulation_state["candles"]
    
    if idx >= len(candles):
        simulation_state["is_active"] = False
        add_live_log("Simulation ended successfully.", "info")
        return {"data": {"is_active": False, "message": "Simulation ended"}}
        
    current_candle = candles[idx]
    simulation_state["nifty_price"] = current_candle["close"]
    
    # Advance simulation by 'speed' minutes
    old_idx = simulation_state["current_index"]
    simulation_state["current_index"] += simulation_state["speed"]
    new_idx = simulation_state["current_index"]
    
    # Add Status Logs (Handle speed jumps to keep sync with UI)
    log_interval = 15 # Log every 15 mins of market time
    if (old_idx // log_interval) != (new_idx // log_interval):
        sim_time = current_candle["date"].strftime("%H:%M:%S")
        status = "Scanning" if live_strategy_config["is_auto_trade_enabled"] else "Monitoring"
        
        # Filter strategies that haven't been executed yet
        pending_strats = [s for s in live_strategy_config["active_strategies"] if s not in simulation_state["executed_strategies"]]
        
        if pending_strats:
            active_strats = f" ({', '.join(pending_strats)})"
            add_live_log(f"{status} market @ Nifty {current_candle['close']}{active_strats}", "debug")
        else:
            add_live_log(f"All selected strategies executed. Monitoring open positions @ Nifty {current_candle['close']}", "debug")
    
    # 1. Update existing positions
    total_pnl = 0
    kite = get_kite_instance()
    sim_date = datetime.strptime(simulation_state["date"], "%Y-%m-%d").date()
    
    simulation_state["time"] = current_candle["date"].strftime("%H:%M:%S") if isinstance(current_candle["date"], datetime) else str(current_candle["date"])
    
    for pos in simulation_state["positions"]:
        # If open, update LTP and P&L
        if pos["quantity"] != 0:
            if pos.get("is_multi_leg") and pos.get("legs"):
                current_net_price = 0
                
                for leg in pos["legs"]:
                    # Find/Cache leg instrument history
                    leg_inst = find_option(simulation_state["nifty_options"], leg["strike"], leg["type"], sim_date)
                    if not leg_inst: continue
                    
                    leg_history = get_instrument_history(kite, leg_inst["instrument_token"], sim_date)
                    
                    # Get price at EXACT same index as current Nifty simulation
                    if idx < len(leg_history):
                        leg_current_price = leg_history[idx]["close"]
                    else:
                        leg_current_price = leg_history[-1]["close"] if leg_history else leg["price"]
                    
                    # Standard tick-size floor to prevent negative individual prices
                    leg_current_price = max(0.05, leg_current_price)
                    
                    if leg["action"] == "BUY":
                        current_net_price += leg_current_price
                    else:
                        current_net_price -= leg_current_price
                
                pos["last_price"] = round(current_net_price, 2)
                
                # Correct Multi-Leg P&L Formula: (Current Net Value - Entry Net Value) * Quantity
                # This treats "quantity" as the number of lots of the combined strategy
                open_pnl = (pos["last_price"] - pos["average_price"]) * pos["quantity"]
                pos["pnl"] = round(pos.get("realized_pnl", 0) + open_pnl, 2)
            else:
                # Single leg REAL logic
                symbol = pos["tradingsymbol"]
                inst = next((o for o in simulation_state["nifty_options"] if o.get("tradingsymbol") == symbol), None)
                
                if inst:
                    hist = get_instrument_history(kite, inst["instrument_token"], sim_date)
                    if idx < len(hist):
                        pos["last_price"] = hist[idx]["close"]
                    else:
                        pos["last_price"] = hist[-1]["close"] if hist else pos["average_price"]
                
                open_pnl = (pos["last_price"] - pos["average_price"]) * pos["quantity"]
                pos["pnl"] = round(pos.get("realized_pnl", 0) + open_pnl, 2)

            # 2. Check for SL/Target hits (ONLY for open positions)
            # EXPERT LOGIC: Calculate thresholds from trade value (qty * entry_price)
            trade_value = abs(pos["average_price"]) * abs(pos["quantity"])
            profit_threshold = trade_value * (live_strategy_config["reward_pct"] / 100)
            loss_threshold = trade_value * (live_strategy_config["risk_pct"] / 100)
            
            # Use current trade's P&L for trigger, not the whole day's realized P&L
            current_trade_pnl = pos["pnl"] - pos.get("realized_pnl", 0)
            
            if current_trade_pnl >= profit_threshold:
                pos["realized_pnl"] = pos["pnl"] # Lock in the total P&L
                old_qty = pos["quantity"]
                pos["quantity"] = 0
                pos["exit_reason"] = "Target Hit"
                
                # Add cooldown for ALL strategies that were part of this position
                strats = [s.strip() for s in pos["strategy"].split(",")]
                for s in strats:
                    simulation_state["strategy_cooldown"][s] = simulation_state["time"]
                
                add_live_log(f"SIM EXIT: Target hit for {pos['strategy']} @ P&L: +{current_trade_pnl}", "success")
                add_sim_order(pos["strategy"], pos["tradingsymbol"], "SELL", old_qty, pos["last_price"], reason="Target Hit")
            elif current_trade_pnl <= -loss_threshold:
                pos["realized_pnl"] = pos["pnl"] # Lock in the total P&L
                old_qty = pos["quantity"]
                pos["quantity"] = 0
                pos["exit_reason"] = "Stoploss Hit"
                
                # Add cooldown for ALL strategies that were part of this position
                strats = [s.strip() for s in pos["strategy"].split(",")]
                for s in strats:
                    simulation_state["strategy_cooldown"][s] = simulation_state["time"]
                
                add_live_log(f"SIM EXIT: Stoploss hit for {pos['strategy']} @ P&L: {current_trade_pnl}", "warning")
                add_sim_order(pos["strategy"], pos["tradingsymbol"], "SELL", old_qty, pos["last_price"], reason="Stoploss Hit")
        
        # Always add P&L (both open and closed) to total MTM
        total_pnl += pos["pnl"]

    # 3. Scan for NEW trades if strategies are active
    if live_strategy_config["active_strategies"]:
        # Get historical data up to current point
        atm_ce = simulation_state.get("atm_ce")
        if not atm_ce: return {"data": {"is_active": False, "message": "ATM CE missing"}}
        
        ref_history = get_instrument_history(kite, atm_ce["instrument_token"], sim_date)
        hist_1m = ref_history[:idx+1]
        
        if len(hist_1m) >= 5: # Need at least one 5-minute candle
            hist_5m = aggregate_to_tf(hist_1m, 5)
            
            if len(hist_5m) >= 2: # Need enough for trend analysis
                kite = get_kite_instance()
                
                # Pick active strategies to simulate
                for s_type in live_strategy_config["active_strategies"]:
                    # Optimization: Skip if strategy is in cooldown
                    if s_type in simulation_state["strategy_cooldown"]:
                        last_exit_time_str = simulation_state["strategy_cooldown"][s_type]
                        last_exit_time = datetime.strptime(last_exit_time_str, "%H:%M:%S").time()
                        curr_sim_time = datetime.strptime(simulation_state["time"], "%H:%M:%S").time()
                        diff_mins = (curr_sim_time.hour * 60 + curr_sim_time.minute) - (last_exit_time.hour * 60 + last_exit_time.minute)
                        if diff_mins < 5: continue

                    # Prevent multiple active positions for same strategy
                    if any(p.get("strategy") == s_type for p in simulation_state["positions"] if p["quantity"] > 0): continue
                    
                    # Setup context
                    nifty_price = current_candle["close"]
                    current_strike = round(nifty_price / 50) * 50
                    date_str = simulation_state["date"]
                    atm_ce = simulation_state.get("atm_ce")
                    atm_pe = simulation_state.get("atm_pe")
                    
                    if not atm_ce or not atm_pe: continue

                    # Run strategy on 5-minute aggregated data
                    first_candle_5m = hist_5m[0]
                    
                    res = await run_strategy_on_candles(
                        kite, s_type, hist_5m, first_candle_5m, nifty_price, 
                        current_strike, atm_ce, atm_pe, date_str, 
                        simulation_state.get("nifty_options", []), 
                        datetime.strptime(date_str, "%Y-%m-%d")
                    )
                    
                    if res:
                        # PROFESSIONAL FILTER: Only allow ONE signal per strategy per day for scheduled entries
                        if s_type in ["long_straddle", "long_strangle", "bull_call_spread", "bear_put_spread", "iron_condor"]:
                            simulation_state["strategy_cooldown"][s_type] = "23:59:59" # Block for rest of day
                        
                        entry_price = round(float(res["entry_price"]), 2)
                        option_to_trade = res["option_to_trade"]
                        
                        msg = f"SIGNAL DETECTED: {s_type} triggered @ {entry_price}. Reason: {res['reason']}"
                        if live_strategy_config["is_auto_trade_enabled"]:
                            add_live_log(msg, "success")
                            # Place mock trade - MERGE if ANY position exists for this symbol (Session-wise tracking)
                            existing_pos = next((p for p in simulation_state["positions"] if p["tradingsymbol"] == option_to_trade["tradingsymbol"]), None)
                            
                            # Calculate quantity based on risk formula
                            trade_qty = calculate_sim_qty(
                                entry_price, 
                                live_strategy_config["fund"], 
                                live_strategy_config["risk_pct"], 
                                live_strategy_config["reward_pct"]
                            )
                            
                            if trade_qty == 0:
                                add_live_log(f"SIM SKIP: {s_type} ignored - Risk+Reward gap is too small (<10%). Increase your % settings.", "debug")
                                continue
                            
                            if existing_pos:
                                old_qty = existing_pos["quantity"]
                                # SKIP if position is already active (Prevent infinite additions)
                                if old_qty != 0:
                                    continue
                                
                                # Reopening a closed position
                                existing_pos["average_price"] = entry_price
                                existing_pos["quantity"] = trade_qty
                                existing_pos["exit_reason"] = "" # Clear previous exit reason
                                existing_pos["is_multi_leg"] = res.get("multi_leg", False)
                                existing_pos["legs"] = res.get("legs", [])
                                existing_pos["strategy"] = s_type # Reset to the one that re-opened it
                                
                                existing_pos["entry_nifty"] = nifty_price # Update entry nifty for trail/SL
                                add_sim_order(existing_pos["strategy"], existing_pos["tradingsymbol"], "BUY", trade_qty, entry_price, reason="Re-opened position")
                                add_live_log(f"SIM ORDER: Re-opened position for {option_to_trade['tradingsymbol']} @ {entry_price} (Qty: {trade_qty})", "success")
                            else:
                                simulation_state["positions"].append({
                                    "strategy": s_type,
                                    "tradingsymbol": option_to_trade["tradingsymbol"],
                                    "exchange": "NFO",
                                    "product": "MIS",
                                    "quantity": trade_qty,
                                    "average_price": entry_price,
                                    "entry_nifty": nifty_price,
                                    "last_price": entry_price,
                                    "target": True, # Target logic is now P&L based
                                    "stoploss": True, # SL logic is now P&L based
                                    "is_multi_leg": res.get("multi_leg", False),
                                    "legs": res.get("legs", []),
                                    "pnl": 0,
                                    "realized_pnl": 0
                                })
                                add_sim_order(s_type, option_to_trade["tradingsymbol"], "BUY", trade_qty, entry_price, reason=res['reason'])
                                add_live_log(f"SIM ORDER: {res['option_to_trade']['tradingsymbol']} bought @ {entry_price} (Qty: {trade_qty})", "success")
                        else:
                            add_live_log(f"ALERT: {msg} (Enable Auto-Trade to execute)", "info")
                        # removed break to allow multiple strategies to scan in same minute

    return {
        "data": {
            "is_active": True,
            "time": current_candle["date"].strftime("%H:%M:%S") if isinstance(current_candle["date"], datetime) else str(current_candle["date"]),
            "nifty_price": round(current_candle["close"], 2),
            "positions": simulation_state["positions"],
            "orders": simulation_state["orders"],
            "total_pnl": round(total_pnl, 2),
            "active_count": len([p for p in simulation_state["positions"] if p["quantity"] != 0])
        }
    }

# Moved to api/v1/routes/simulation.py
# @app.post("/simulation/stop")
# def stop_simulation():
    simulation_state["is_active"] = False
    return {"status": "success"}

@app.get("/")
def read_root():
    return {"Hello": "World", "broker": "Zerodha Kite Connect"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# Moved to api/v1/routes/auth.py
# @app.get("/access-token")
def get_access_token_endpoint():
    """Get stored access token and validate it"""
    token = get_access_token()
    if token:
        # Validate token
        token_info = {
            "length": len(token),
            "preview": token[:20] + "..." if len(token) > 20 else token,
            "is_valid_length": len(token) >= 20,
            "status": "unknown"
        }
        
        # Try to validate token if it's long enough
        if len(token) >= 20:
            try:
                from utils.kite_utils import get_kite_api_key
                api_key = get_kite_api_key()
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(token)
                kite.profile()  # Quick validation
                token_info["status"] = "valid"
                token_info["api_key_used"] = api_key[:10] + "..." if api_key and len(api_key) > 10 else "NOT SET"
                return {
                    "access_token": token[:20] + "...",
                    "token_info": token_info,
                    "is_valid": True
                }
            except Exception as e:
                token_info["status"] = "invalid"
                token_info["error"] = str(e)
                token_info["api_key_used"] = api_key[:10] + "..." if api_key and len(api_key) > 10 else "NOT SET"
                return {
                    "access_token": token[:20] + "...",
                    "token_info": token_info,
                    "is_valid": False,
                    "message": f"Token exists but is invalid: {str(e)}"
                }
        else:
            token_info["status"] = "too_short"
            token_info["error"] = "Token is too short to be valid (expected at least 20 chars, got " + str(len(token)) + ")"
            return {
                "access_token": token[:20] + "...",
                "token_info": token_info,
                "is_valid": False,
                "message": "Token is too short. Please regenerate using /auth and /set-token"
            }
    
    return {
        "access_token": None, 
        "message": "No access token found. Please generate one using /auth and /set-token",
        "token_info": None,
        "is_valid": False
    }

@app.get("/test-token")
def test_token():
    """Test the current token with various Kite API calls"""
    try:
        from utils.kite_utils import get_kite_instance, get_access_token, get_kite_api_key
        from datetime import datetime, timedelta
        
        token = get_access_token()
        api_key = get_kite_api_key()
        
        if not token:
            return {"error": "No token found"}
        
        results = {
            "token_length": len(token),
            "api_key": api_key[:10] + "..." if api_key else "NOT SET",
            "tests": {}
        }
        
        kite = get_kite_instance()
        
        # Test 1: Profile
        try:
            profile = kite.profile()
            results["tests"]["profile"] = {
                "status": "success",
                "user_name": profile.get("user_name", "N/A"),
                "user_id": profile.get("user_id", "N/A")
            }
        except Exception as e:
            results["tests"]["profile"] = {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__
            }
        
        # Test 2: Historical data
        try:
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            data = kite.historical_data(2885, yesterday, today, "5minute")
            results["tests"]["historical_data"] = {
                "status": "success",
                "candles_count": len(data) if data else 0
            }
        except Exception as e:
            results["tests"]["historical_data"] = {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "instrument": 2885,
                "date_range": f"{yesterday} to {today}"
            }
        
        return results
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__}

# Moved to api/v1/routes/auth.py
# @app.delete("/access-token")
def delete_access_token():
    """Delete the stored access token (useful for clearing invalid tokens)"""
    try:
        token_path = Path("config/access_token.txt")
        if token_path.exists():
            token_path.unlink()
            return {"status": "success", "message": "Access token deleted successfully"}
        return {"status": "success", "message": "No access token file found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting token: {str(e)}")

# Moved to api/v1/routes/auth.py
# @app.get("/auth")
def get_login_url():
    """Get Kite Connect login URL"""
    try:
        # Validate API key is set
        # Get API key from config (may have been updated via UI)
        from agent.config import get_agent_config
        config = get_agent_config()
        current_api_key = config.kite_api_key or api_key
        if current_api_key == 'your_api_key_here' or not current_api_key:
            raise HTTPException(
                status_code=500, 
                detail="KITE_API_KEY is not configured. Please set it in the Configuration page or environment variables"
            )
        
        kite = KiteConnect(api_key=current_api_key)
        login_url = kite.login_url()
        print(f"Generated login URL with redirect_uri: {redirect_uri}")
        return {
            "login_url": login_url,
            "message": "Redirect user to this URL for authentication",
            "redirect_uri": redirect_uri,
            "note": f"Make sure the redirect URI in your Kite Connect app settings matches: {redirect_uri}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating login URL: {str(e)}")

# Moved to api/v1/routes/auth.py
# @app.post("/set-token")
async def set_token(req: Request):
    """Store access token after authentication"""
    try:
        data = await req.json()
        request_token = data.get('request_token')
        access_token_from_request = data.get('access-token')
        
        # Get user_id from request to use user-specific config
        user_id = "default"
        try:
            from core.user_context import get_user_id_from_request
            user_id = get_user_id_from_request(req)
        except:
            pass
        
        print(f"Received request_token: {request_token[:20] if request_token else None}...")
        print(f"Received access-token from request: {access_token_from_request[:20] if access_token_from_request else None}...")
        print(f"User ID: {user_id}")
        print(f"Redirect URI configured: {redirect_uri}")
        
        if not request_token and not access_token_from_request:
            raise HTTPException(status_code=400, detail="Either request_token or access-token required")
        
        # Initialize access_token variable - will be set from generate_session() if request_token provided
        access_token = None
        
        # If request_token is provided, generate access_token (ignore any access-token from request)
        if request_token:
            # Validate API key and secret are set
            # Get credentials from user-specific config (may have been updated via UI)
            from agent.user_config import get_user_config
            config = get_user_config(user_id=user_id)
            current_api_key = config.kite_api_key or api_key
            current_api_secret = config.kite_api_secret or api_secret
            
            print(f"API Key configured: {current_api_key[:10] if current_api_key and len(current_api_key) > 10 else 'NOT SET'}...")
            
            if current_api_key == 'your_api_key_here' or not current_api_key:
                raise HTTPException(
                    status_code=500, 
                    detail="KITE_API_KEY is not configured. Please set it in the Configuration page or environment variables"
                )
            if current_api_secret == 'your_api_secret_here' or not current_api_secret:
                raise HTTPException(
                    status_code=500, 
                    detail="KITE_API_SECRET is not configured. Please set it in the Configuration page or environment variables"
                )
            
            kite = KiteConnect(api_key=current_api_key)
            try:
                print(f"Attempting to generate session with request_token...")
                print(f"Request token length: {len(request_token) if request_token else 0}")
                print(f"Request token preview: {request_token[:30] if request_token and len(request_token) > 30 else request_token}...")
                
                data_response = kite.generate_session(request_token, api_secret=current_api_secret)
                print(f"generate_session response type: {type(data_response)}")
                print(f"generate_session response: {data_response}")
                
                # Handle both dict and object responses
                if isinstance(data_response, dict):
                    access_token = data_response.get('access_token')
                    print(f"Response is dict, keys: {list(data_response.keys())}")
                else:
                    # If it's an object, try to get the attribute
                    access_token = getattr(data_response, 'access_token', None) if hasattr(data_response, 'access_token') else None
                    print(f"Response is object, has access_token attr: {hasattr(data_response, 'access_token')}")
                
                print(f"Access token extracted: {access_token is not None}")
                print(f"Access token length: {len(access_token) if access_token else 0}")
                if access_token:
                    print(f"Access token preview: {access_token[:30]}...")
                    # Safety check: access_token should NOT be the same as request_token
                    if access_token == request_token:
                        print(f"ERROR: access_token equals request_token! This indicates a problem.")
                        raise HTTPException(
                            status_code=400,
                            detail="Token exchange failed: Received request_token instead of access_token. "
                                   "This usually means: 1) API key/secret mismatch, "
                                   "2) Redirect URI mismatch, or 3) Request token expired. "
                                   f"Please check your Kite Connect app settings. Redirect URI should be: {redirect_uri}"
                        )
                
                if not access_token:
                    print(f"WARNING: No access_token in response! Full response: {data_response}")
                    print(f"Response type: {type(data_response)}")
                    if isinstance(data_response, dict):
                        print(f"Available keys: {list(data_response.keys())}")
                        print(f"Full response content: {data_response}")
                    raise HTTPException(
                        status_code=400, 
                        detail="Failed to get access_token from Kite. The generate_session() call did not return an access_token. "
                               f"Response: {str(data_response)}. "
                               "Please check: 1) API key and secret are correct, "
                               f"2) Redirect URI matches exactly: {redirect_uri}, "
                               "3) Request token is fresh (they expire quickly)."
                    )
            except KiteException as e:
                error_msg = str(e)
                print(f"KiteException: {error_msg}")
                # Provide more helpful error messages
                if "invalid" in error_msg.lower() or "expired" in error_msg.lower():
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid request token: {error_msg}. "
                               f"Please ensure: 1) Redirect URI in Kite Connect app settings matches exactly '{redirect_uri}', "
                               f"2) Request token is used immediately (they expire quickly), "
                               f"3) API key and secret are correct."
                    )
                raise HTTPException(status_code=400, detail=f"Kite API error: {error_msg}")
            except Exception as e:
                # Catch any other unexpected errors
                error_msg = str(e)
                print(f"Unexpected error during token exchange: {error_msg}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error during token exchange: {error_msg}. "
                           "Please check server logs for details."
                )
        else:
            # No request_token provided, use access-token from request body
            access_token = access_token_from_request
        
        if not access_token:
            raise HTTPException(status_code=400, detail="Failed to obtain access token")
        
        # Safety check: Make sure we're not accidentally saving the request_token
        if request_token and access_token == request_token:
            print(f"ERROR: access_token is the same as request_token! This should not happen.")
            print(f"Request token: {request_token}")
            print(f"Access token: {access_token}")
            raise HTTPException(
                status_code=400,
                detail="Internal error: Access token matches request token. "
                       "The token exchange may have failed. Please try again with a fresh request_token."
            )
        
        # Validate token before saving (Kite access tokens are typically 32+ characters)
        # Note: Kite access tokens can be 32 characters, so we use a lower threshold
        if len(access_token) < 20:
            print(f"WARNING: Access token seems invalid (length: {len(access_token)})")
            print(f"Token value: {access_token}")
            print(f"Request token was: {request_token[:20] if request_token else None}...")
            print(f"Are they the same? {access_token == request_token if request_token else 'N/A'}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid access token (too short: {len(access_token)} chars). "
                       "Kite access tokens should be at least 20 characters. "
                       "The token you're trying to save appears to be invalid or corrupted. "
                       "This usually means the token exchange failed. Please check: "
                       "1) Your Kite API Key and Secret are correct in the Config page, "
                       "2) The redirect URI matches exactly in your Kite Connect app settings, "
                       "3) The request_token is fresh (they expire quickly). "
                       "Try generating a new request_token and use it immediately."
            )
        
        # Store access token
        config_path = Path("config")
        config_path.mkdir(exist_ok=True)
        
        with open("config/access_token.txt", "w") as f:
            f.write(access_token.strip())
        
        print(f"Access token stored successfully in config/access_token.txt (length: {len(access_token)})")
        return {
            "status": "success", 
            "message": "Access token stored successfully",
            "access_token": access_token[:20] + "..." if access_token else None
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting token: {str(e)}")

# Moved to api/v1/routes/portfolio.py
# @app.get("/getBalance")
# def get_balance():
#     """Get user margin/funds"""
#     try:
#         kite = get_kite_instance()
#         margins = kite.margins()
#         
#         # Kite Connect margins() returns:
#         # {
#         #   'equity': {
#         #     'enabled': True,
#         #     'net': float,
#         #     'available': float,  # Available margin for trading
#         #     'utilised': float,   # Utilised margin
#         #     'adhoc_margin': float,
#         #     'cash': float,
#         #     'collateral': float,
#         #     'intraday_payin': float,
#         #     'live_balance': float,
#         #     'opening_balance': float
#         #   },
#         #   'commodity': {...}
#         # }
#         
#         equity_data = margins.get('equity', {})
#         
#         print(f"Raw Kite Connect margins response: {margins}")
#         print(f"Equity data: {equity_data}")
#         
#         # Kite Connect 'available' can be a number or dict with 'cash' and 'intraday_payin'
#         # Kite Connect 'utilised' can be a number or dict with 'debits', 'exposure', etc.
#         available_value = equity_data.get('available', 0)
#         utilised_value = equity_data.get('utilised', 0)
#         
#         # Extract numeric values if they're dictionaries
#         if isinstance(available_value, dict):
#             # If available is a dict, use 'cash' (this is the available margin)
#             # Based on user's requirement: _available_margin should be 120608.6 (which is cash)
#             available_margin = available_value.get('cash', 0)
#             if available_margin == 0:
#                 available_margin = equity_data.get('opening_balance', 0) or equity_data.get('live_balance', 0)
#         else:
#             # Use available, or fallback to cash/opening_balance
#             available_margin = available_value if available_value else equity_data.get('cash', 0) or equity_data.get('opening_balance', 0)
#         
#         if isinstance(utilised_value, dict):
#             # If utilised is a dict, use 'debits' (this is the utilised margin)
#             # Based on user's requirement: _utilised_margin should be 15258.75 (which is debits)
#             utilised_margin = utilised_value.get('debits', 0)
#         else:
#             utilised_margin = utilised_value
#         
#         # Total margin is the net value (live_balance or net)
#         # Based on user's requirement: _total_margin should be 64741.85 (which is net/live_balance)
#         total_margin = equity_data.get('net', 0) or equity_data.get('live_balance', 0)
#         
#         print(f"Calculated available_margin: {available_margin}")
#         print(f"Calculated utilised_margin: {utilised_margin}")
#         print(f"Calculated total_margin: {total_margin}")
#         
#         # Transform to match frontend expected format (Upstox-style)
#         # Frontend expects: res.data.equity._available_margin (as a NUMBER, not object)
#         # CRITICAL: _available_margin must be INSIDE equity object, and must be a number
#         transformed_margins = {
#             "equity": {
#                 # Upstox-style fields (what frontend expects) - MUST be numbers, INSIDE equity
#                 "_available_margin": float(available_margin) if available_margin else 0.0,
#                 "_utilised_margin": float(utilised_margin) if utilised_margin else 0.0,
#                 "_total_margin": float(total_margin) if total_margin else 0.0,
#                 # Keep original Kite Connect fields for reference
#                 **equity_data
#             },
#             "commodity": margins.get('commodity', {})
#         }
#         
#         print(f"Transformed margins structure: {transformed_margins}")
#         
#         return {"data": transformed_margins}
#     except KiteException as e:
#         raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error getting balance: {str(e)}")

# Moved to api/v1/routes/portfolio.py
# @app.get("/getPositions")
# def get_positions():
    """Get user positions"""
    try:
        kite = get_kite_instance()
        positions = kite.positions()
        
        # Transform Kite Connect response to match frontend expected format
        # Kite Connect returns: {'net': [...], 'day': [...]}
        # Frontend expects: array of positions with underscore-prefixed fields
        transformed_positions = []
        
        # Combine net and day positions
        all_positions = positions.get('net', []) + positions.get('day', [])
        
        for position in all_positions:
            transformed_position = {}
            # Map Kite Connect fields to Upstox-style fields with underscore prefix
            field_mapping = {
                'tradingsymbol': '_trading_symbol',
                'exchange': '_exchange',
                'instrument_token': '_instrument_token',
                'product': '_product',
                'quantity': '_quantity',
                'average_price': '_average_price',
                'last_price': '_last_price',
                'pnl': '_pnl',
                'net_quantity': '_net_quantity',
                'sell_quantity': '_sell_quantity',
                'buy_quantity': '_buy_quantity',
                'buy_price': '_buy_price',
                'sell_price': '_sell_price',
                'overnight_quantity': '_overnight_quantity',
                'multiplier': '_multiplier'
            }
            
            # Add transformed fields with underscore prefix
            for kite_field, upstox_field in field_mapping.items():
                if kite_field in position:
                    transformed_position[upstox_field] = position[kite_field]
            
            # Also keep original fields for compatibility
            transformed_position.update(position)
            transformed_positions.append(transformed_position)
        
        return {"data": transformed_positions}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting positions: {str(e)}")

# Moved to api/v1/routes/portfolio.py
# @app.get("/getOrders")
# def get_orders():
    """Get user orders"""
    try:
        kite = get_kite_instance()
        orders = kite.orders()
        
        # Transform Kite Connect response to match frontend expected format (Upstox-style)
        # Kite Connect returns orders with fields like: 'tradingsymbol', 'status', 'quantity', etc.
        # Frontend expects fields with underscore prefix: '_trading_symbol', '_status', '_quantity', etc.
        transformed_orders = []
        
        # Status mapping from Kite Connect to Upstox format (lowercase)
        status_mapping = {
            'OPEN': 'open',
            'COMPLETE': 'complete',
            'CANCELLED': 'cancelled',
            'REJECTED': 'rejected',
            'CANCELLED AMO': 'cancelled',
            'TRANSIT': 'transit',
            'PENDING': 'pending',
            'VALIDATION PENDING': 'pending'
        }
        
        for order in orders:
            transformed_order = {}
            # Map Kite Connect fields to Upstox-style fields with underscore prefix
            field_mapping = {
                'order_id': '_order_id',
                'exchange_order_id': '_exchange_order_id',
                'tradingsymbol': '_trading_symbol',
                'exchange': '_exchange',
                'instrument_token': '_instrument_token',
                'transaction_type': '_transaction_type',
                'quantity': '_quantity',
                'price': '_price',
                'trigger_price': '_trigger_price',
                'product': '_product',
                'order_type': '_order_type',
                'status': '_status',
                'status_message': '_status_message',
                'order_timestamp': '_order_timestamp',
                'exchange_timestamp': '_exchange_timestamp',
                'validity': '_validity',
                'variety': '_variety',
                'disclosed_quantity': '_disclosed_quantity',
                'tag': '_tag',
                'average_price': '_average_price',
                'filled_quantity': '_filled_quantity',
                'pending_quantity': '_pending_quantity',
                'cancelled_quantity': '_cancelled_quantity'
            }
            
            # Add transformed fields with underscore prefix
            for kite_field, upstox_field in field_mapping.items():
                if kite_field in order:
                    value = order[kite_field]
                    # Normalize status to lowercase
                    if kite_field == 'status' and value:
                        value = status_mapping.get(value.upper(), value.lower())
                    transformed_order[upstox_field] = value
            
            # Special handling for price field:
            # For MARKET orders, price is 0, so use average_price if available
            # Otherwise use the price field
            if '_price' in transformed_order:
                price_value = transformed_order.get('_price', 0)
                # If price is 0 or None, try to use average_price
                if not price_value or price_value == 0:
                    avg_price = order.get('average_price', 0)
                    if avg_price and avg_price > 0:
                        transformed_order['_price'] = avg_price
                        print(f"Order {order.get('order_id')}: Using average_price {avg_price} instead of price {price_value}")
            
            # Also keep original fields for compatibility
            transformed_order.update(order)
            transformed_orders.append(transformed_order)
        
        return {"data": transformed_orders}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting orders: {str(e)}")

# Moved to api/v1/routes/market.py
# @app.get("/resolve-instrument/{instrument_name}")
# def resolve_instrument(instrument_name: str, exchange: str = "NSE"):
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

# Moved to api/v1/routes/market.py
# @app.get("/getCandle/{instrument_token}/{interval}/{fromDate}/{toDate}")
# def get_candle(instrument_token: str, interval: str, fromDate: str, toDate: str, request: Request = None):
    """Get historical candle data"""
    try:
        # Get user_id from request if available
        user_id = "default"
        if request:
            try:
                from core.user_context import get_user_id_from_request
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
            # InputException with "invalid token" might be misleading - could be invalid input
            is_token_error = (
                error_type != "InputException" and  # InputException usually means bad input, not token
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
                           "1) GET /auth to get login URL, 2) Login through that URL, "
                           "3) POST /set-token with the request_token from redirect."
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
            # VWAP = Σ(Price × Volume) / Σ(Volume)
            # Where Price = (High + Low + Close) / 3 (typical price)
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
            
            # Detect candlestick patterns
            def detect_candlestick_pattern(current_idx, df):
                """
                Detect candlestick pattern types based on OHLC data
                Returns pattern name
                """
                row = df.loc[current_idx]
                open_price = row['open']
                high = row['high']
                low = row['low']
                close = row['close']
                
                # Calculate body and wick sizes
                body_size = abs(close - open_price)
                upper_wick = high - max(open_price, close)
                lower_wick = min(open_price, close) - low
                total_range = high - low
                
                # Avoid division by zero
                if total_range == 0:
                    return 'Doji'
                
                body_ratio = body_size / total_range
                upper_wick_ratio = upper_wick / total_range
                lower_wick_ratio = lower_wick / total_range
                
                is_bullish = close > open_price
                is_bearish = close < open_price
                is_doji = body_ratio < 0.1  # Body is less than 10% of total range
                
                # Get previous candles for multi-candle patterns
                prev_row = df.loc[current_idx - 1] if current_idx > 0 else None
                prev_prev_row = df.loc[current_idx - 2] if current_idx > 1 else None
                
                # 1. Doji patterns (very small body)
                if is_doji:
                    if upper_wick_ratio > 0.4 and lower_wick_ratio > 0.4:
                        return 'Doji'  # Standard Doji
                    elif upper_wick_ratio > 0.6:
                        return 'Gravestone Doji'  # Long upper wick
                    elif lower_wick_ratio > 0.6:
                        return 'Dragonfly Doji'  # Long lower wick
                    else:
                        return 'Doji'
                
                # 2. Marubozu (no wicks, full body)
                if upper_wick_ratio < 0.05 and lower_wick_ratio < 0.05:
                    if is_bullish:
                        return 'Bullish Marubozu'
                    else:
                        return 'Bearish Marubozu'
                
                # 3. Hammer patterns (long lower wick, small body at top)
                if lower_wick_ratio > 0.6 and body_ratio < 0.3 and upper_wick_ratio < 0.2:
                    if is_bullish:
                        return 'Hammer'  # Bullish reversal
                    else:
                        return 'Hanging Man'  # Bearish reversal (at top of uptrend)
                
                # 4. Inverted Hammer / Shooting Star (long upper wick, small body at bottom)
                if upper_wick_ratio > 0.6 and body_ratio < 0.3 and lower_wick_ratio < 0.2:
                    if is_bullish:
                        return 'Inverted Hammer'  # Bullish reversal (at bottom)
                    else:
                        return 'Shooting Star'  # Bearish reversal (at top)
                
                # 5. Spinning Top (small body, wicks on both sides)
                if body_ratio < 0.3 and upper_wick_ratio > 0.3 and lower_wick_ratio > 0.3:
                    return 'Spinning Top'
                
                # 6. Engulfing patterns (need previous candle)
                if prev_row is not None:
                    prev_open = prev_row['open']
                    prev_close = prev_row['close']
                    prev_body_size = abs(prev_close - prev_open)
                    
                    # Bullish Engulfing
                    if (is_bullish and prev_close < prev_open and  # Previous was bearish
                        close > prev_open and open_price < prev_close and  # Current engulfs previous
                        body_size > prev_body_size * 1.1):  # Current body is significantly larger
                        return 'Bullish Engulfing'
                    
                    # Bearish Engulfing
                    if (is_bearish and prev_close > prev_open and  # Previous was bullish
                        close < prev_open and open_price > prev_close and  # Current engulfs previous
                        body_size > prev_body_size * 1.1):  # Current body is significantly larger
                        return 'Bearish Engulfing'
                
                # 7. Piercing Pattern / Dark Cloud Cover (need previous candle)
                if prev_row is not None:
                    prev_open = prev_row['open']
                    prev_close = prev_row['close']
                    
                    # Piercing Pattern (bullish)
                    if (is_bullish and prev_close < prev_open and  # Previous was bearish
                        open_price < prev_close and  # Opens below previous close
                        close > (prev_open + prev_close) / 2 and  # Closes above midpoint of previous body
                        close < prev_open):  # But below previous open
                        return 'Piercing Pattern'
                    
                    # Dark Cloud Cover (bearish)
                    if (is_bearish and prev_close > prev_open and  # Previous was bullish
                        open_price > prev_close and  # Opens above previous close
                        close < (prev_open + prev_close) / 2 and  # Closes below midpoint of previous body
                        close > prev_open):  # But above previous open
                        return 'Dark Cloud Cover'
                
                # 8. Harami patterns (need previous candle)
                if prev_row is not None:
                    prev_open = prev_row['open']
                    prev_close = prev_row['close']
                    prev_body_size = abs(prev_close - prev_open)
                    
                    # Bullish Harami
                    if (prev_close < prev_open and  # Previous was bearish
                        open_price > prev_close and close < prev_open and  # Current is inside previous
                        body_size < prev_body_size * 0.5):  # Current body is much smaller
                        return 'Bullish Harami'
                    
                    # Bearish Harami
                    if (prev_close > prev_open and  # Previous was bullish
                        open_price < prev_close and close > prev_open and  # Current is inside previous
                        body_size < prev_body_size * 0.5):  # Current body is much smaller
                        return 'Bearish Harami'
                
                # 9. Three White Soldiers / Three Black Crows (need 2 previous candles)
                if prev_row is not None and prev_prev_row is not None:
                    prev_open = prev_row['open']
                    prev_close = prev_row['close']
                    prev_prev_open = prev_prev_row['open']
                    prev_prev_close = prev_prev_row['close']
                    
                    # Three White Soldiers (bullish)
                    if (is_bullish and prev_close > prev_open and prev_prev_close > prev_prev_open and
                        close > prev_close and prev_close > prev_prev_close):
                        return 'Three White Soldiers'
                    
                    # Three Black Crows (bearish)
                    if (is_bearish and prev_close < prev_open and prev_prev_close < prev_prev_open and
                        close < prev_close and prev_close < prev_prev_close):
                        return 'Three Black Crows'
                
                # 10. Morning Star / Evening Star (need 2 previous candles)
                if prev_row is not None and prev_prev_row is not None:
                    prev_prev_open = prev_prev_row['open']
                    prev_prev_close = prev_prev_row['close']
                    prev_open = prev_row['open']
                    prev_close = prev_row['close']
                    prev_body_size = abs(prev_close - prev_open)
                    prev_prev_body_size = abs(prev_prev_close - prev_prev_open)
                    
                    # Morning Star (bullish reversal)
                    if (is_bullish and prev_prev_close < prev_prev_open and  # First candle bearish
                        prev_body_size < prev_prev_body_size * 0.5 and  # Middle candle small
                        close > (prev_prev_open + prev_prev_close) / 2):  # Third candle closes above midpoint
                        return 'Morning Star'
                    
                    # Evening Star (bearish reversal)
                    if (is_bearish and prev_prev_close > prev_prev_open and  # First candle bullish
                        prev_body_size < prev_prev_body_size * 0.5 and  # Middle candle small
                        close < (prev_prev_open + prev_prev_close) / 2):  # Third candle closes below midpoint
                        return 'Evening Star'
                
                # 11. Default patterns based on body size
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
                        # Bullish Reversal: crossed from below to above
                        current_reversal = 'Bullish Reversal'
                        
                        # Calculate reversal strength
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
                        # Bearish Reversal: crossed from above to below
                        current_reversal = 'Bearish Reversal'
                        
                        # Calculate reversal strength
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
                
                # Set current candle's reversal signal if there's a new reversal
                if current_reversal:
                    df.loc[i, 'reversal_signal'] = current_reversal
                    df.loc[i, 'reversal_strength'] = current_strength
                
                # Confirm reversal if previous candle had reversal and current maintains position
                # Only confirm if there was NO new reversal on current candle
                if i > 1 and current_reversal is None:
                    prev_prev_above = df.loc[i-2, 'is_above_vwap']
                    prev_signal = df.loc[i-1, 'reversal_signal']
                    
                    # Confirm Bullish Reversal: Previous had bullish reversal, and current maintains above VWAP
                    if (prev_signal == 'Bullish Reversal' and 
                        not prev_prev_above and  # Was below before reversal
                        df.loc[i-1, 'is_above_vwap'] and  # Previous candle is above
                        curr_above):  # Current candle also above (confirmation)
                        df.loc[i-1, 'reversal_signal'] = 'Confirmed Bullish'
                    
                    # Confirm Bearish Reversal: Previous had bearish reversal, and current maintains below VWAP
                    if (prev_signal == 'Bearish Reversal' and 
                        prev_prev_above and  # Was above before reversal
                        not df.loc[i-1, 'is_above_vwap'] and  # Previous candle is below
                        not curr_above):  # Current candle also below (confirmation)
                        df.loc[i-1, 'reversal_signal'] = 'Confirmed Bearish'
            
            # Generate Buy/Sell signals based on candlestick patterns, VWAP position, and reversal signals
            def generate_trading_signal(current_idx, df, instrument_token=None):
                """
                Generate buy/sell signal based on:
                1. Custom buy condition: Green candle (closing above VWAP OR high above VWAP) 
                   AND previous candle is "Three Black Crows"
                   AND current candle is one of high-performing patterns:
                   - Dragonfly Doji (100% win rate)
                   - Piercing Pattern (87.5% win rate)
                   - Inverted Hammer (high profit potential)
                   - Long White Candle (most common profitable pattern)
                   AND instrument is NOT in blacklist (PERSISTENT excluded)
                   Note: Morning Star removed due to 33.3% win rate in actual trading
                2. Strong reversal signals [DISABLED]
                3. Candlestick pattern (bullish patterns = buy, bearish = sell) [DISABLED]
                4. VWAP position (above = bullish, below = bearish) [DISABLED]
                Returns: (signal, priority_level, reason) or (None, None, None) if no signal
                """
                # Instrument blacklist (based on loss analysis)
                # PERSISTENT shows severe losses (₹-50.08 avg loss)
                instrument_blacklist = ['4701441']  # PERSISTENT token
                
                if instrument_token and str(instrument_token) in instrument_blacklist:
                    return (None, None, None)  # Skip blacklisted instruments
                
                row = df.loc[current_idx]
                candle_type = row.get('candle_type', '')
                reversal_signal = row.get('reversal_signal')
                is_above_vwap = row.get('is_above_vwap', False)
                vwap_diff_percent = row.get('vwap_diff_percent', 0)
                close = row.get('close', 0)
                open_price = row.get('open', 0)
                high = row.get('high', 0)
                vwap = row.get('vwap', 0)
                
                # Priority 1: Custom buy condition [ACTIVE - FINE-TUNED]
                # BUY when (green candle closing above vwap) OR (green candle high is above vwap) 
                # AND (previous candle is "Three Black Crows" candle)
                # AND (current candle is one of the high-performing patterns)
                # Morning Star removed - 33.3% win rate in actual trading
                if current_idx > 0:
                    prev_row = df.loc[current_idx - 1]
                    prev_candle_type = prev_row.get('candle_type', '')
                    is_green_candle = close > open_price  # Green/bullish candle
                    close_above_vwap = close > vwap
                    high_above_vwap = high > vwap
                    
                    # High-performing candle types (Morning Star removed - 33.3% win rate)
                    # Based on loss analysis, Morning Star showed poor performance in actual trading
                    high_performance_candle_types = [
                        'Dragonfly Doji',      # 100% win rate
                        'Piercing Pattern',    # 87.5% win rate
                        'Inverted Hammer',   # High profit potential (52.6% win rate)
                        'Long White Candle'  # Most common profitable pattern (44.3% win rate)
                    ]
                    
                    # Check if current candle type matches one of the high-performing patterns
                    current_candle_matches = any(pattern in candle_type for pattern in high_performance_candle_types)
                    
                    if (prev_candle_type == 'Three Black Crows' and is_green_candle and 
                        (close_above_vwap or high_above_vwap) and current_candle_matches):
                        # Get the specific pattern name for the reason
                        matched_pattern = next((p for p in high_performance_candle_types if p in candle_type), candle_type)
                        reason = f"Priority 1: {matched_pattern} candle {'closing' if close_above_vwap else 'high'} above VWAP after Three Black Crows"
                        return ('BUY', 1, reason)
                
                # Priority 2: Strong reversal signals [DISABLED]
                # DISABLED: Only Priority 1 is active based on validation results
                # if reversal_signal:
                #     if 'Bullish' in reversal_signal or 'Confirmed Bullish' in reversal_signal:
                #         reason = f"Priority 2: {reversal_signal}"
                #         return ('BUY', 2, reason)
                #     elif 'Bearish' in reversal_signal or 'Confirmed Bearish' in reversal_signal:
                #         reason = f"Priority 2: {reversal_signal}"
                #         return ('SELL', 2, reason)
                
                # Priority 3: Strong candlestick patterns [DISABLED]
                # DISABLED: Only Priority 1 is active based on validation results
                # bullish_patterns = [
                #     'Bullish Engulfing', 'Hammer', 'Inverted Hammer', 'Morning Star',
                #     'Three White Soldiers', 'Piercing Pattern', 'Bullish Harami',
                #     'Bullish Marubozu', 'Long White Candle'
                # ]
                # bearish_patterns = [
                #     'Bearish Engulfing', 'Shooting Star', 'Hanging Man', 'Evening Star',
                #     'Three Black Crows', 'Dark Cloud Cover', 'Bearish Harami',
                #     'Bearish Marubozu', 'Long Black Candle'
                # ]
                # 
                # for pattern in bullish_patterns:
                #     if pattern in candle_type:
                #         reason = f"Priority 3: {pattern} pattern"
                #         return ('BUY', 3, reason)
                # 
                # for pattern in bearish_patterns:
                #     if pattern in candle_type:
                #         reason = f"Priority 3: {pattern} pattern"
                #         return ('SELL', 3, reason)
                
                # Priority 4: VWAP position with significant distance [DISABLED]
                # DISABLED: Only Priority 1 is active based on validation results
                # if is_above_vwap and vwap_diff_percent > 0.1:  # Significantly above VWAP
                #     reason = f"Priority 4: Significantly above VWAP ({vwap_diff_percent:.2f}%)"
                #     return ('BUY', 4, reason)
                # elif not is_above_vwap and vwap_diff_percent < -0.1:  # Significantly below VWAP
                #     reason = f"Priority 4: Significantly below VWAP ({vwap_diff_percent:.2f}%)"
                #     return ('SELL', 4, reason)
                
                # Priority 5: General VWAP position [DISABLED]
                # DISABLED: Only Priority 1 is active based on validation results
                # if is_above_vwap:
                #     reason = f"Priority 5: Above VWAP ({vwap_diff_percent:.2f}%)"
                #     return ('BUY', 5, reason)
                # else:
                #     reason = f"Priority 5: Below VWAP ({vwap_diff_percent:.2f}%)"
                #     return ('SELL', 5, reason)
                
                # No signal if Priority 1 condition not met
                return (None, None, None)
            
            # Apply trading signal generation
            # Only Priority 1 signals are generated (other priorities disabled)
            df['trading_signal'] = None
            df['signal_priority'] = None
            df['signal_reason'] = None
            for i in range(len(df)):
                signal, priority, reason = generate_trading_signal(i, df, instrument_token=instrument_token)
                if signal is not None:  # Only set if Priority 1 condition is met
                    df.loc[i, 'trading_signal'] = signal
                    df.loc[i, 'signal_priority'] = priority
                    df.loc[i, 'signal_reason'] = reason
            
            # Convert all datetime columns to Unix timestamps (integers)
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

# Backtest WebSocket endpoint moved to api/v1/routes/strategies.py
# Use: /api/v1/strategies/ws/backtest-vwap-strategy

# Moved to api/v1/routes/strategies.py (WebSocket version is active)
# @app.post("/backtest-vwap-strategy")
# async def backtest_vwap_strategy(request: Request):
    """
    Backtest VWAP Priority 1 strategy across all 25 stocks
    Returns order details, PnL, and summary for each instrument
    
    Request body: {
        "start_date": "2025-11-17",
        "end_date": "2025-12-26",
        "timeframe": "5minute"
    }
    """
    try:
        body = await request.json()
        start_date = body.get("start_date")
        end_date = body.get("end_date")
        timeframe = body.get("timeframe", "5minute")
        
        print(f"[backtest] Received timeframe: {timeframe}, type: {type(timeframe)}")
        
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="start_date and end_date are required")
        
        # Get user_id from request if available
        user_id = "default"
        if request:
            try:
                from core.user_context import get_user_id_from_request
                user_id = get_user_id_from_request(request)
            except:
                pass
        
        # All 25 stocks with validated tokens
        instruments = [
            {"name": "RELIANCE", "token": "738561"},
            {"name": "TCS", "token": "2953217"},
            {"name": "HDFCBANK", "token": "341249"},
            {"name": "INFY", "token": "408065"},
            {"name": "ICICIBANK", "token": "1270529"},
            {"name": "SBIN", "token": "779521"},
            {"name": "KOTAKBANK", "token": "492033"},
            {"name": "AXISBANK", "token": "1510401"},
            {"name": "INDUSINDBK", "token": "1346049"},
            {"name": "FEDERALBNK", "token": "261889"},
            {"name": "WIPRO", "token": "969473"},
            {"name": "HCLTECH", "token": "1850625"},
            {"name": "TECHM", "token": "3465729"},
            {"name": "LTIM", "token": "4561409"},
            {"name": "PERSISTENT", "token": "4701441"},
            {"name": "SUNPHARMA", "token": "857857"},
            {"name": "DRREDDY", "token": "225537"},
            {"name": "CIPLA", "token": "177665"},
            {"name": "LUPIN", "token": "2672641"},
            {"name": "DIVISLAB", "token": "2800641"},
            {"name": "BHARTIARTL", "token": "2714625"},
            {"name": "BAJFINANCE", "token": "81153"},
            {"name": "ITC", "token": "424961"},
            {"name": "HINDUNILVR", "token": "356865"},
            {"name": "MARUTI", "token": "2815745"},
        ]
        
        kite = get_kite_instance(user_id=user_id)
        
        # Convert interval format
        interval_map = {
            '1minute': 'minute',
            '5minute': '5minute',
            '15minute': '15minute',
            '30minute': '30minute',
            '60minute': '60minute',
            'day': 'day'
        }
        # Ensure timeframe is a string and strip any whitespace
        timeframe = str(timeframe).strip() if timeframe else "5minute"
        kite_interval = interval_map.get(timeframe, timeframe)
        print(f"[backtest] Converted timeframe '{timeframe}' to kite_interval '{kite_interval}'")
        
        # Parse dates
        from_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        to_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Get all trading dates in range
        trading_dates = []
        current_date = from_date
        while current_date <= to_date:
            if current_date.weekday() < 5:  # Monday=0 to Friday=4
                trading_dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
        
        results = []
        total_signals = 0
        total_profit = 0
        
        # Process each instrument
        for inst in instruments:
            instrument_signals = []
            instrument_profit = 0
            instrument_profitable = 0
            instrument_losses = 0
            
            # Process each trading date
            for date_str in trading_dates:
                try:
                    from_date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                    to_date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                    
                    historical_data = kite.historical_data(
                        instrument_token=int(inst["token"]),
                        from_date=from_date_obj,
                        to_date=to_date_obj,
                        interval=kite_interval
                    )
                    
                    if not historical_data or len(historical_data) == 0:
                        continue
                    
                    # Process same as getCandle endpoint
                    df = pd.DataFrame(historical_data)
                    # Rename 'date' to 'timestamp' to match getCandle processing
                    if 'date' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['date'])
                    elif 'timestamp' not in df.columns:
                        df['timestamp'] = pd.to_datetime(df.index)
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    # Calculate VWAP
                    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
                    df['cumulative_volume'] = df['volume'].cumsum()
                    df['cumulative_tpv'] = (df['typical_price'] * df['volume']).cumsum()
                    df['vwap'] = df['cumulative_tpv'] / df['cumulative_volume']
                    
                    # We need to detect patterns and generate signals
                    # This is complex - let's reuse getCandle's processing
                    # Actually, let's make an internal call to getCandle processing
                    # For now, simplified: just check for Priority 1 conditions
                    
                    # Get current price (last candle close) for PnL - convert to float
                    current_price = float(df.iloc[-1]['close']) if len(df) > 0 else 0.0
                    
                    # Process signals using same logic as getCandle endpoint
                    # Calculate VWAP position and differences
                    df['is_above_vwap'] = df['close'] > df['vwap']
                    df['vwap_position'] = df['is_above_vwap'].map({True: 'Above', False: 'Below'})
                    df['vwap_diff'] = (df['close'] - df['vwap']).abs()
                    df['vwap_diff_percent'] = (df['close'] - df['vwap']) / df['vwap'] * 100
                    
                    # Detect candlestick patterns (reuse logic from getCandle)
                    def detect_candlestick_pattern_bk(current_idx, df):
                        """Detect candlestick pattern - same as getCandle"""
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
                    
                    # Apply pattern detection
                    df['candle_type'] = df.index.map(lambda i: detect_candlestick_pattern_bk(i, df))
                    
                    # Generate trading signals (reuse logic from getCandle)
                    def generate_trading_signal_bk(current_idx, df, instrument_token=None):
                        """Generate trading signal - same as getCandle"""
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
                    
                    # Apply signal generation
                    df['trading_signal'] = None
                    df['signal_priority'] = None
                    df['signal_reason'] = None
                    for i in range(len(df)):
                        signal, priority, reason = generate_trading_signal_bk(i, df, instrument_token=inst["token"])
                        if signal is not None:
                            df.loc[i, 'trading_signal'] = signal
                            df.loc[i, 'signal_priority'] = priority
                            df.loc[i, 'signal_reason'] = reason
                    
                    # Extract Priority 1 signals
                    priority1_rows = df[(df['trading_signal'] == 'BUY') & (df['signal_priority'] == 1)]
                    for idx, signal_row in priority1_rows.iterrows():
                        # Convert numpy types to native Python types
                        try:
                            entry_price_val = signal_row['close']
                            if isinstance(entry_price_val, (pd.Series, np.ndarray)):
                                entry_price = float(entry_price_val.iloc[0] if hasattr(entry_price_val, 'iloc') else entry_price_val[0])
                            else:
                                entry_price = float(entry_price_val)
                        except (ValueError, TypeError) as e:
                            print(f"Error converting entry_price: {e}, value: {signal_row['close']}, type: {type(signal_row['close'])}")
                            entry_price = float(str(signal_row['close']))
                        
                        try:
                            current_price_float = float(current_price)
                        except (ValueError, TypeError):
                            current_price_float = float(str(current_price))
                        
                        profit = current_price_float - entry_price
                        
                        # Format entry time (from signal timestamp)
                        entry_timestamp = signal_row['timestamp']
                        try:
                            if pd.api.types.is_datetime64_any_dtype(entry_timestamp):
                                entry_time = entry_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(entry_timestamp, pd.Timestamp):
                                entry_time = entry_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(entry_timestamp, (int, float, np.integer, np.floating)):
                                # Handle both seconds and milliseconds
                                entry_timestamp_float = float(entry_timestamp)
                                if entry_timestamp_float > 1e10:  # milliseconds
                                    entry_time = datetime.fromtimestamp(entry_timestamp_float / 1000).strftime('%Y-%m-%d %H:%M:%S')
                                else:  # seconds
                                    entry_time = datetime.fromtimestamp(entry_timestamp_float).strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                # Try to convert to datetime
                                try:
                                    entry_time = pd.to_datetime(entry_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                                except:
                                    entry_time = str(entry_timestamp) if entry_timestamp else '-'
                        except Exception as e:
                            print(f"Error formatting entry_time: {e}, timestamp: {entry_timestamp}")
                            entry_time = '-'
                        
                        # Format exit time (last candle timestamp)
                        exit_timestamp = df.iloc[-1]['timestamp']
                        try:
                            if pd.api.types.is_datetime64_any_dtype(exit_timestamp):
                                exit_time = exit_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(exit_timestamp, pd.Timestamp):
                                exit_time = exit_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                            elif isinstance(exit_timestamp, (int, float, np.integer, np.floating)):
                                # Handle both seconds and milliseconds
                                exit_timestamp_float = float(exit_timestamp)
                                if exit_timestamp_float > 1e10:  # milliseconds
                                    exit_time = datetime.fromtimestamp(exit_timestamp_float / 1000).strftime('%Y-%m-%d %H:%M:%S')
                                else:  # seconds
                                    exit_time = datetime.fromtimestamp(exit_timestamp_float).strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                # Try to convert to datetime
                                try:
                                    exit_time = pd.to_datetime(exit_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                                except:
                                    exit_time = str(exit_timestamp) if exit_timestamp else '-'
                        except Exception as e:
                            print(f"Error formatting exit_time: {e}, timestamp: {exit_timestamp}")
                            exit_time = '-'
                        
                        # Convert numpy types to native Python types for JSON serialization
                        timestamp_val = signal_row['timestamp']
                        try:
                            if pd.api.types.is_datetime64_any_dtype(timestamp_val) or isinstance(timestamp_val, pd.Timestamp):
                                timestamp_val = int(timestamp_val.timestamp())
                            elif isinstance(timestamp_val, (int, float, np.integer, np.floating)):
                                timestamp_val = int(float(timestamp_val))
                            else:
                                # Try to convert to datetime first, then to timestamp
                                timestamp_val = int(pd.to_datetime(timestamp_val).timestamp())
                        except Exception as e:
                            print(f"Error converting timestamp: {e}, type: {type(timestamp_val)}, value: {timestamp_val}")
                            timestamp_val = 0
                        
                        instrument_signals.append({
                            "date": date_str,
                            "timestamp": timestamp_val,
                            "entry_time": entry_time,
                            "exit_time": exit_time,
                            "entry_price": float(round(float(entry_price), 2)),
                            "exit_price": float(round(float(current_price), 2)),
                            "qty": 1,  # Default quantity, can be made configurable
                            "profit": float(round(float(profit), 2)),
                            "profit_percent": float(round((float(profit) / float(entry_price) * 100) if entry_price > 0 else 0, 2)),
                            "candle_type": str(signal_row.get('candle_type', '')) if signal_row.get('candle_type') is not None else '',
                            "signal_reason": str(signal_row.get('signal_reason', '')) if signal_row.get('signal_reason') is not None else ''
                        })
                        
                        instrument_profit += float(profit)
                        if float(profit) > 0:
                            instrument_profitable += 1
                        elif float(profit) < 0:
                            instrument_losses += 1
                        
                except Exception as e:
                    print(f"Error processing {inst['name']} on {date_str}: {str(e)}")
                    continue
            
            # Calculate summary for this instrument
            if len(instrument_signals) > 0:
                avg_profit = float(instrument_profit) / len(instrument_signals)
                win_rate = (float(instrument_profitable) / len(instrument_signals)) * 100 if len(instrument_signals) > 0 else 0
            else:
                avg_profit = 0.0
                win_rate = 0.0
            
            results.append({
                "instrument": str(inst["name"]),
                "instrument_token": str(inst["token"]),
                "total_signals": int(len(instrument_signals)),
                "total_profit": float(round(float(instrument_profit), 2)),
                "avg_profit": float(round(float(avg_profit), 2)),
                "win_rate": float(round(float(win_rate), 2)),
                "profitable_signals": int(instrument_profitable),
                "loss_signals": int(instrument_losses),
                "orders": instrument_signals  # Order details
            })
            
            total_signals += int(len(instrument_signals))
            total_profit += float(instrument_profit)
        
        # Overall summary - convert all to native Python types
        summary = {
            "total_instruments": int(len(instruments)),
            "total_signals": int(total_signals),
            "total_profit": float(round(float(total_profit), 2)),
            "avg_profit_per_signal": float(round(float(total_profit) / float(total_signals), 2)) if total_signals > 0 else 0.0,
            "profitable_instruments": int(len([r for r in results if float(r["total_profit"]) > 0])),
            "loss_instruments": int(len([r for r in results if float(r["total_profit"]) < 0])),
            "test_period": {
                "start_date": str(start_date),
                "end_date": str(end_date),
                "trading_days": int(len(trading_dates))
            }
        }
        
        return {
            "status": "success",
            "data": {
                "results": results,
                "summary": summary
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")

# Moved to api/v1/routes/orders.py
# @app.post("/placeOrder")
# async def place_order(req: Request):
    """Place an order"""
    try:
        kite = get_kite_instance()
        payload = await req.json()
        
        # Map Upstox format to Kite Connect format
        kite_order_params = {
            "variety": payload.get('variety', kite.VARIETY_REGULAR),
            "exchange": payload.get('exchange', kite.EXCHANGE_NSE),
            "tradingsymbol": payload.get('tradingsymbol') or payload.get('instrument_token', '').split('|')[-1],
            "transaction_type": payload.get('transaction_type', payload.get('transactionType', 'BUY')),
            "quantity": payload.get('quantity', payload.get('qty', 1)),
            "price": payload.get('price'),
            "product": payload.get('product', kite.PRODUCT_MIS),
            "order_type": payload.get('order_type', payload.get('orderType', kite.ORDER_TYPE_MARKET)),
            "validity": payload.get('validity', kite.VALIDITY_DAY),
            "disclosed_quantity": payload.get('disclosed_quantity', payload.get('disclosedQuantity', 0)),
            "trigger_price": payload.get('trigger_price', payload.get('triggerPrice')),
            "squareoff": payload.get('squareoff'),
            "stoploss": payload.get('stoploss'),
            "trailing_stoploss": payload.get('trailing_stoploss'),
            "tag": payload.get('tag', 'tradehandler')
        }
        
        # Remove None values
        kite_order_params = {k: v for k, v in kite_order_params.items() if v is not None}
        
        order_id = kite.place_order(**kite_order_params)
        return {"data": {"order_id": order_id}, "status": "success"}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing order: {str(e)}")

# Moved to api/v1/routes/orders.py
# @app.post("/modifyOrder")
# async def modify_order(req: Request):
    """Modify an existing order"""
    try:
        kite = get_kite_instance()
        payload = await req.json()
        
        order_id = payload.get('orderId') or payload.get('order_id')
        if not order_id:
            raise HTTPException(status_code=400, detail="orderId is required")
        
        # Map Upstox format to Kite Connect format
        modify_params = {
            "variety": payload.get('variety', kite.VARIETY_REGULAR),
            "order_id": str(order_id),
            "quantity": payload.get('quantity'),
            "price": payload.get('price'),
            "order_type": payload.get('order_type', payload.get('orderType')),
            "validity": payload.get('validity', kite.VALIDITY_DAY),
            "disclosed_quantity": payload.get('disclosed_quantity', payload.get('disclosedQuantity')),
            "trigger_price": payload.get('trigger_price', payload.get('triggerPrice'))
        }
        
        # Remove None values
        modify_params = {k: v for k, v in modify_params.items() if v is not None}
        
        order_id_modified = kite.modify_order(**modify_params)
        return {"data": {"order_id": order_id_modified}, "status": "success"}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error modifying order: {str(e)}")

# Moved to api/v1/routes/orders.py
# @app.post("/cancelOrder")
# async def cancel_order(req: Request):
    """Cancel an order"""
    try:
        kite = get_kite_instance()
        payload = await req.json()
        
        order_id = payload.get('order_id') or payload.get('orderId')
        if not order_id:
            raise HTTPException(status_code=400, detail="order_id is required")
        
        variety = payload.get('variety', kite.VARIETY_REGULAR)
        order_id_cancelled = kite.cancel_order(variety=variety, order_id=str(order_id))
        return {"data": {"order_id": order_id_cancelled}, "status": "success"}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling order: {str(e)}")

# Moved to api/v1/routes/orders.py
# @app.post("/sellOrder")
# async def sell_order(req: Request):
    """Place a sell order (same as placeOrder but with transaction_type=SELL)"""
    try:
        kite = get_kite_instance()
        payload = await req.json()
        
        # Ensure transaction_type is SELL
        payload['transaction_type'] = 'SELL'
        
        # Map to Kite Connect format
        kite_order_params = {
            "variety": payload.get('variety', kite.VARIETY_REGULAR),
            "exchange": payload.get('exchange', kite.EXCHANGE_NSE),
            "tradingsymbol": payload.get('tradingsymbol') or payload.get('instrument_token', '').split('|')[-1],
            "transaction_type": 'SELL',
            "quantity": payload.get('quantity', payload.get('qty', 1)),
            "price": payload.get('price'),
            "product": payload.get('product', kite.PRODUCT_MIS),
            "order_type": payload.get('order_type', payload.get('orderType', kite.ORDER_TYPE_MARKET)),
            "validity": payload.get('validity', kite.VALIDITY_DAY),
            "disclosed_quantity": payload.get('disclosed_quantity', payload.get('disclosedQuantity', 0)),
            "trigger_price": payload.get('trigger_price', payload.get('triggerPrice')),
            "tag": payload.get('tag', 'tradehandler')
        }
        
        # Remove None values
        kite_order_params = {k: v for k, v in kite_order_params.items() if v is not None}
        
        order_id = kite.place_order(**kite_order_params)
        return {"data": {"order_id": order_id}, "status": "success"}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing sell order: {str(e)}")

# Moved to api/v1/routes/portfolio.py
# @app.get("/ws-portfolio")
# def get_portfolio_ws():
    """Get WebSocket authorization for portfolio streaming"""
    try:
        access_token = get_access_token()
        if not access_token:
            raise HTTPException(status_code=401, detail="Access token not found")
        
        # Construct Kite Connect WebSocket URL
        # Format: wss://ws.kite.trade?api_key=xxx&access_token=yyy
        ws_url = f"wss://ws.kite.trade?api_key={api_key}&access_token={access_token}"
        
        # Return in format expected by frontend (Upstox-style)
        return {
            "data": {
                "_authorized_redirect_uri": ws_url,
                "api_key": api_key,
                "access_token": access_token[:20] + "..." if access_token else None,
                "message": "WebSocket URL for portfolio streaming"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting WebSocket info: {str(e)}")

# Moved to api/v1/routes/portfolio.py
# @app.get("/ws-orders")
# def get_orders_ws():
    """Get WebSocket authorization for orders streaming"""
    try:
        access_token = get_access_token()
        if not access_token:
            raise HTTPException(status_code=401, detail="Access token not found")
        
        # Construct Kite Connect WebSocket URL
        # Format: wss://ws.kite.trade?api_key=xxx&access_token=yyy
        ws_url = f"wss://ws.kite.trade?api_key={api_key}&access_token={access_token}"
        
        # Return in format expected by frontend (Upstox-style)
        return {
            "data": {
                "_authorized_redirect_uri": ws_url,
                "api_key": api_key,
                "access_token": access_token[:20] + "..." if access_token else None,
                "message": "WebSocket URL for orders streaming"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting WebSocket info: {str(e)}")

# Moved to api/v1/routes/market.py
# @app.get("/instruments")
# def get_instruments():
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

# Moved to api/v1/routes/market.py
# @app.get("/quote/{instrument_key}")
# def get_quote(instrument_key: str):
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

# Moved to api/v1/routes/market.py
# @app.get("/nifty50-options")
# def get_nifty50_options():
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
        
        # Current strike
        strikes_to_find.append(current_strike)
        
        # 2 ITM strikes (below current for calls, above for puts)
        strikes_to_find.append(current_strike - 50)
        strikes_to_find.append(current_strike - 100)
        
        # 2 OTM strikes (above current for calls, below for puts)
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
                
                # Get today's date for historical data
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
                        # Default values if quote not available
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
                # Set default values if quote fetch fails
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

# Moved to api/v1/routes/market.py
# @app.get("/ws-nifty50-options")
# def get_nifty50_options_ws():
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

# Moved to utils/indicators.py
# def calculate_bollinger_bands(...):
# def calculate_bollinger_bands_full(...):
# def calculate_rsi(...):
# def calculate_pivot_points(...):
# def calculate_support_resistance(...):

def strategy_915_candle_break(kite, trading_candles, first_candle, nifty_price, current_strike, atm_ce, atm_pe, date_str):
    """9:15 Candle Break Strategy - Smart with Nifty trend confirmation"""
    if len(trading_candles) < 5:
        return None
    
    recent_candles = trading_candles[-5:] if len(trading_candles) >= 5 else trading_candles
    closes = [c.get("close", 0) for c in recent_candles if c.get("close", 0) > 0]
    
    if len(closes) < 3:
        return None
    
    first_high = first_candle.get("high", 0)
    first_low = first_candle.get("low", 0)
    first_open = first_candle.get("open", 0)
    first_close = first_candle.get("close", 0)
    current_price = closes[-1]
    
    # Get actual candle time for logs/entry
    last_candle_date = recent_candles[-1].get("date", "")
    if isinstance(last_candle_date, datetime):
        entry_time_val = last_candle_date.strftime("%H:%M:%S")
    else:
        entry_time_val = last_candle_date.split(' ')[1][:8] if ' ' in str(last_candle_date) else "09:30:00"

    # Calculate first candle range and body
    first_range = first_high - first_low
    first_body = abs(first_close - first_open)
    first_body_pct = (first_body / first_range * 100) if first_range > 0 else 0
    
    # Only trade if first candle has significant body (>40% of range) - indicates strong direction
    if first_body_pct < 40:
        # Only log rejection during the early morning breakout window
        if "09:15" <= entry_time_val <= "10:00":
            add_live_log(f"9:15 Strategy: Rejected - First candle body too small ({first_body_pct:.1f}%). Need strong direction.", "debug")
        return None  # Doji or indecision candle - skip
    
    # Check 9:15 candle direction
    first_candle_bullish = first_close > first_open
    first_candle_bearish = first_close < first_open
    
    if first_candle_bullish:
        add_live_log(f"9:15 Strategy: Detected Bullish first candle. Waiting for breakout above {first_high}.", "info")
    else:
        add_live_log(f"9:15 Strategy: Detected Bearish first candle. Waiting for breakout below {first_low}.", "info")

    # Option price trend (last 3 candles)
    option_trend_up = closes[-1] > closes[-2] > closes[-3] if len(closes) >= 3 else False
    option_trend_down = closes[-1] < closes[-2] < closes[-3] if len(closes) >= 3 else False
    
    # Volume confirmation
    volumes = [c.get("volume", 0) for c in recent_candles]
    avg_volume = sum(volumes) / len(volumes) if volumes else 0
    current_volume = volumes[-1] if volumes else 0
    volume_confirmation = current_volume > avg_volume * 1.15 if avg_volume > 0 else True  # 15% above average
    
    if not volume_confirmation:
        add_live_log(f"9:15 Strategy: Price breakout detected but VOLUME confirmation failed.", "debug")

    # Breakout confirmation - price must break with momentum
    above_first_high = current_price > first_high * 1.001  # 0.1% above high for confirmation
    below_first_low = current_price < first_low * 0.999   # 0.1% below low for confirmation
    
    # Bullish: First candle bullish + option trending up + break above high + volume
    if (first_candle_bullish and option_trend_up and above_first_high and volume_confirmation):
        return {
            "trend": "BULLISH",
            "option_to_trade": atm_ce,
            "option_type": "CE",
            "entry_price": current_price,
            "entry_time": entry_time_val,
            "reason": f"9:15 Candle Break - Bullish first candle ({first_body_pct:.0f}% body), option uptrend, break above high with volume"
        }
    # Bearish: First candle bearish + option trending down + break below low + volume
    elif (first_candle_bearish and option_trend_down and below_first_low and volume_confirmation):
        return {
            "trend": "BEARISH",
            "option_to_trade": atm_pe,
            "option_type": "PE",
            "entry_price": current_price,
            "entry_time": entry_time_val,
            "reason": f"9:15 Candle Break - Bearish first candle ({first_body_pct:.0f}% body), option downtrend, break below low with volume"
        }
    
    return None

def strategy_mean_reversion_bollinger(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str):
    """Mean Reversion Strategy - Smart with relaxed but confirmed signals"""
    if len(trading_candles) < 20:
        return None
    
    closes = [c.get("close", 0) for c in trading_candles if c.get("close", 0) > 0]
    if len(closes) < 20:
        return None
    
    upper_band, middle_band, lower_band = calculate_bollinger_bands(closes, period=20, num_std=2)
    
    if upper_band is None or middle_band is None or lower_band is None:
        add_live_log(f"Mean Reversion: Rejected - BB calculation failed.", "debug")
        return None
    
    current_price = closes[-1]
    prev_price = closes[-2] if len(closes) > 1 else current_price
    
    # Calculate RSI (14-period)
    rsi = 50
    if len(closes) >= 14:
        gains = []
        losses = []
        for i in range(1, min(15, len(closes))):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if gains and losses:
            avg_gain = sum(gains) / len(gains)
            avg_loss = sum(losses) / len(losses) if sum(losses) > 0 else 1
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))
    
    # Calculate distance from bands (as percentage)
    distance_from_lower = ((current_price - lower_band) / (upper_band - lower_band) * 100) if (upper_band - lower_band) > 0 else 50
    distance_from_upper = ((upper_band - current_price) / (upper_band - lower_band) * 100) if (upper_band - lower_band) > 0 else 50
    
    # Relaxed but smart conditions:
    # Get actual candle time for entry
    last_candle_date = trading_candles[-1].get("date", "")
    if isinstance(last_candle_date, datetime):
        entry_time_val = last_candle_date.strftime("%H:%M:%S")
    else:
        entry_time_val = last_candle_date.split(' ')[1][:8] if ' ' in str(last_candle_date) else "11:00:00"

    # Buy: Price near lower band (within 5%) OR touching it, RSI < 35 (less strict), reversal signal
    if (distance_from_lower <= 5 or current_price <= lower_band) and rsi < 35 and current_price > prev_price:
        add_live_log(f"Mean Reversion: Buy signal potential. RSI={rsi:.1f}, near lower BB. Waiting for reversal...", "info")
        # Additional confirmation: price should be moving away from lower band
        if len(closes) >= 3 and closes[-1] > closes[-2] > closes[-3]:
            return {
                "trend": "BULLISH",
                "option_to_trade": atm_ce,
                "option_type": "CE",
                "entry_price": current_price,
                "entry_time": entry_time_val,
                "reason": f"Mean Reversion - Price near lower BB ({distance_from_lower:.1f}% from lower), RSI={round(rsi, 1)} (oversold), reversal confirmed"
            }
    
    # Sell: Price near upper band (within 5%) OR touching it, RSI > 65 (less strict), reversal signal
    elif (distance_from_upper <= 5 or current_price >= upper_band) and rsi > 65 and current_price < prev_price:
        add_live_log(f"Mean Reversion: Sell signal potential. RSI={rsi:.1f}, near upper BB. Waiting for reversal...", "info")
        # Additional confirmation: price should be moving away from upper band
        if len(closes) >= 3 and closes[-1] < closes[-2] < closes[-3]:
            return {
                "trend": "BEARISH",
                "option_to_trade": atm_pe,
                "option_type": "PE",
                "entry_price": current_price,
                "entry_time": entry_time_val,
                "reason": f"Mean Reversion - Price near upper BB ({distance_from_upper:.1f}% from upper), RSI={round(rsi, 1)} (overbought), reversal confirmed"
            }
    
    if rsi < 35 or rsi > 65:
        add_live_log(f"Mean Reversion: RSI={rsi:.1f} but price not near Bollinger Bands.", "debug")
    else:
        add_live_log(f"Mean Reversion: Scanning. Price:{current_price:.1f}, RSI:{rsi:.1f}. No signal.", "debug")
    
    return None

def strategy_momentum_breakout(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str):
    """Smart Momentum Breakout - Avoid false breakouts with multiple confirmations"""
    if len(trading_candles) < 15:
        return None
    
    # Ensure closes and volumes arrays are aligned
    closes = []
    volumes = []
    for c in trading_candles:
        close_val = c.get("close", 0)
        volume_val = c.get("volume", 0)
        if close_val > 0:
            closes.append(close_val)
            volumes.append(volume_val if volume_val > 0 else 0)
    
    if len(closes) < 15 or len(volumes) < 15:
        return None
    
    min_len = min(len(closes), len(volumes))
    closes = closes[:min_len]
    volumes = volumes[:min_len]
    
    if len(closes) < 15:
        return None
    
    # Calculate moving averages and indicators
    df = pd.DataFrame({'close': closes, 'volume': volumes})
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=min(20, len(closes))).mean()
    df['vol_avg'] = df['volume'].rolling(window=10).mean()
    
    current_price = closes[-1]
    sma_5 = df['sma_5'].iloc[-1]
    sma_10 = df['sma_10'].iloc[-1]
    sma_20 = df['sma_20'].iloc[-1] if len(df) >= 20 else sma_10
    avg_volume = df['vol_avg'].iloc[-1]
    current_volume = volumes[-1]
    
    # Check for NaN values
    if pd.isna(sma_5) or pd.isna(sma_10) or pd.isna(avg_volume):
        add_live_log(f"Momentum: Rejected - MA calculation failed.", "debug")
        return None
    
    # Calculate momentum strength (rate of change)
    momentum_5 = ((closes[-1] - closes[-6]) / closes[-6] * 100) if len(closes) >= 6 else 0
    momentum_10 = ((closes[-1] - closes[-11]) / closes[-11] * 100) if len(closes) >= 11 else 0
    
    # Smart conditions to avoid false breakouts:
    # 1. Price must be above/below ALL MAs (strong trend)
    # 2. MAs must be aligned (5 > 10 > 20 for bullish, 5 < 10 < 20 for bearish)
    # 3. Strong volume (1.3x average, not just 1.2x)
    # 4. Positive momentum (price accelerating)
    # 5. Recent price action confirms (last 2-3 candles in same direction)
    
    # Bullish: Strong uptrend with all confirmations
    ma_aligned_bullish = sma_5 > sma_10 > sma_20 if not pd.isna(sma_20) else sma_5 > sma_10
    price_above_all = current_price > sma_5 > sma_10
    strong_volume = current_volume > avg_volume * 1.3
    positive_momentum = momentum_5 > 0.5 and momentum_10 > 0.5
    recent_uptrend = len(closes) >= 3 and closes[-1] > closes[-2] > closes[-3]
    
    # Get actual candle time for entry
    last_candle_date = trading_candles[-1].get("date", "")
    if isinstance(last_candle_date, datetime):
        entry_time_val = last_candle_date.strftime("%H:%M:%S")
    else:
        entry_time_val = last_candle_date.split(' ')[1][:8] if ' ' in str(last_candle_date) else "10:00:00"

    if (price_above_all and ma_aligned_bullish and strong_volume and 
        positive_momentum and recent_uptrend):
        return {
            "trend": "BULLISH",
            "option_to_trade": atm_ce,
            "option_type": "CE",
            "entry_price": current_price,
            "entry_time": entry_time_val,
            "reason": f"Momentum Breakout - Strong uptrend (5MA>{round(sma_5, 2)}, 10MA>{round(sma_10, 2)}), momentum {momentum_5:.1f}%, volume {current_volume/avg_volume:.1f}x"
        }
    
    # Bearish: Strong downtrend with all confirmations
    ma_aligned_bearish = sma_5 < sma_10 < sma_20 if not pd.isna(sma_20) else sma_5 < sma_10
    price_below_all = current_price < sma_5 < sma_10
    negative_momentum = momentum_5 < -0.5 and momentum_10 < -0.5
    recent_downtrend = len(closes) >= 3 and closes[-1] < closes[-2] < closes[-3]
    
    if (price_below_all and ma_aligned_bearish and strong_volume and 
        negative_momentum and recent_downtrend):
        return {
            "trend": "BEARISH",
            "option_to_trade": atm_pe,
            "option_type": "PE",
            "entry_price": current_price,
            "entry_time": entry_time_val,
            "reason": f"Momentum Breakout - Strong downtrend (5MA<{round(sma_5, 2)}, 10MA<{round(sma_10, 2)}), momentum {momentum_5:.1f}%, volume {current_volume/avg_volume:.1f}x"
        }
    
    if positive_momentum or negative_momentum:
        add_live_log(f"Momentum: Momentum detected ({momentum_5:.1f}%) but waiting for volume/MA alignment.", "info")
    else:
        add_live_log(f"Momentum: Scanning. SMA5:{sma_5:.1f}, SMA10:{sma_10:.1f}. No momentum.", "debug")
    
    return None

def strategy_support_resistance_breakout(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str):
    """Smart Support/Resistance Breakout - With volume and momentum confirmation"""
    if len(trading_candles) < 20:
        return None
    
    resistance, support = calculate_support_resistance(trading_candles, lookback=20)
    
    if resistance is None or support is None:
        return None
    
    closes = [c.get("close", 0) for c in trading_candles if c.get("close", 0) > 0]
    volumes = [c.get("volume", 0) for c in trading_candles if c.get("volume", 0) > 0]
    
    if len(closes) < 3 or len(volumes) < 3:
        return None
    
    current_price = closes[-1]
    prev_price = closes[-2]
    prev_prev_price = closes[-3] if len(closes) >= 3 else prev_price
    
    # Calculate average volume for confirmation
    avg_volume = sum(volumes[-10:]) / min(10, len(volumes)) if volumes else 0
    current_volume = volumes[-1] if volumes else 0
    
    # Calculate distance from support/resistance
    distance_to_resistance = ((resistance - current_price) / resistance * 100) if resistance > 0 else 100
    distance_to_support = ((current_price - support) / support * 100) if support > 0 else 100
    
    # Breakout above resistance - require confirmation
    # 1. Price must break above resistance
    # 2. Breakout must be with momentum (price accelerating)
    # 3. Volume should be above average (confirmation)
    # 4. Price should stay above resistance (not a false breakout)
    # Get actual candle time for entry
    last_candle_date = trading_candles[-1].get("date", "")
    if isinstance(last_candle_date, datetime):
        entry_time_val = last_candle_date.strftime("%H:%M:%S")
    else:
        entry_time_val = last_candle_date.split(' ')[1][:8] if ' ' in str(last_candle_date) else "11:30:00"

    if (prev_price <= resistance and current_price > resistance * 1.002):  # 0.2% above for confirmation
        add_live_log(f"S/R Breakout: Price broke above resistance ({resistance:.1f}). Checking momentum...", "info")
        # Check momentum and volume
        price_momentum = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        volume_confirmation = current_volume > avg_volume * 1.2 if avg_volume > 0 else True
        
        # Additional confirmation: price should be accelerating upward
        if price_momentum > 0.5 and volume_confirmation and current_price > prev_price > prev_prev_price:
            return {
                "trend": "BULLISH",
                "option_to_trade": atm_ce,
                "option_type": "CE",
                "entry_price": current_price,
                "entry_time": entry_time_val,
                "reason": f"Support/Resistance Breakout - Price broke above resistance ({round(resistance, 2)}) with {price_momentum:.2f}% momentum, volume {current_volume/avg_volume:.1f}x"
            }
    
    # Breakdown below support - require confirmation
    elif (prev_price >= support and current_price < support * 0.998):  # 0.2% below for confirmation
        add_live_log(f"S/R Breakout: Price broke below support ({support:.1f}). Checking momentum...", "info")
        # Check momentum and volume
        price_momentum = ((prev_price - current_price) / prev_price * 100) if prev_price > 0 else 0
        volume_confirmation = current_volume > avg_volume * 1.2 if avg_volume > 0 else True
        
        # Additional confirmation: price should be accelerating downward
        if price_momentum > 0.5 and volume_confirmation and current_price < prev_price < prev_prev_price:
            return {
                "trend": "BEARISH",
                "option_to_trade": atm_pe,
                "option_type": "PE",
                "entry_price": current_price,
                "entry_time": entry_time_val,
                "reason": f"Support/Resistance Breakout - Price broke below support ({round(support, 2)}) with {price_momentum:.2f}% momentum, volume {current_volume/avg_volume:.1f}x"
            }
    
    if distance_to_resistance < 1 or distance_to_support < 1:
        add_live_log(f"S/R Breakout: Price near S/R levels. Res:{resistance:.1f}, Supp:{support:.1f}. Waiting for breakout...", "info")
    else:
        add_live_log(f"S/R Breakout: Scanning. Price:{current_price:.1f}. Not near S/R levels.", "debug")
    
    return None

# ============================================================================
# SENSIBULL-STYLE MULTI-LEG STRATEGIES
# ============================================================================

# Moved to simulation/helpers.py
# def find_option(...):

def get_leg_price_at_time(kite, instrument, date, time_str):
    """Helper to get historical price for a leg at a specific time"""
    if not instrument: return 0
    try:
        # Use a small window around the requested time
        dt = datetime.combine(date, datetime.strptime(time_str, "%H:%M:%S").time())
        # Request data around the time
        hist = kite.historical_data(instrument["instrument_token"], dt - timedelta(minutes=10), dt + timedelta(minutes=10), "5minute")
        return hist[-1]["close"] if hist else 0
    except:
        return 0

def strategy_long_straddle(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date):
    """Long Straddle - Buy ATM CE + ATM PE @ 9:20 AM"""
    entry_time_val = "09:20:00"
    
    # In simulation, we check if current time is around 9:20
    last_candle_date = trading_candles[-1].get("date", "")
    current_time_str = last_candle_date.strftime("%H:%M:%S") if isinstance(last_candle_date, datetime) else str(last_candle_date).split(' ')[1][:8]
    
    # Only trigger around the scheduled time (2-minute window)
    if not ("09:20:00" <= current_time_str <= "09:22:00"):
        if int(current_time_str.split(':')[1]) % 30 == 0:
            add_live_log(f"Long Straddle: Scheduled for {entry_time_val}. Currently scanning...", "debug")
        return None

    ce_price = get_leg_price_at_time(kite, atm_ce, trade_date, entry_time_val)
    pe_price = get_leg_price_at_time(kite, atm_pe, trade_date, entry_time_val)
    
    if ce_price <= 0 or pe_price <= 0: return None
    net_price = ce_price + pe_price
    
    return {
        "trend": "NEUTRAL",
        "option_to_trade": atm_ce,
        "option_type": "STRADDLE",
        "entry_price": net_price,
        "entry_time": entry_time_val,
        "reason": f"Long Straddle - Buy ATM CE + PE @ {current_strike}",
        "multi_leg": True,
        "legs": [
            {"action": "BUY", "type": "CE", "strike": current_strike, "price": ce_price, "tradingsymbol": atm_ce.get("tradingsymbol")},
            {"action": "BUY", "type": "PE", "strike": current_strike, "price": pe_price, "tradingsymbol": atm_pe.get("tradingsymbol")}
        ]
    }

def strategy_long_strangle(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date):
    """Long Strangle - Buy OTM CE + OTM PE @ 9:20 AM"""
    entry_time_val = "09:20:00"
    
    # In simulation, we check if current time is around 9:20
    last_candle_date = trading_candles[-1].get("date", "")
    current_time_str = last_candle_date.strftime("%H:%M:%S") if isinstance(last_candle_date, datetime) else str(last_candle_date).split(' ')[1][:8]
    
    # Only trigger around the scheduled time (2-minute window)
    if not ("09:20:00" <= current_time_str <= "09:22:00"):
        # Avoid flooding logs every minute for multi-leg (only log every 30 mins)
        if int(current_time_str.split(':')[1]) % 30 == 0:
            add_live_log(f"Long Strangle: Scheduled for {entry_time_val}. Currently scanning...", "debug")
        return None

    otm_ce_strike = current_strike + 50
    otm_pe_strike = current_strike - 50
    
    otm_ce = find_option(nifty_options, otm_ce_strike, "CE", trade_date)
    otm_pe = find_option(nifty_options, otm_pe_strike, "PE", trade_date)
    
    ce_price = get_leg_price_at_time(kite, otm_ce, trade_date, entry_time_val)
    pe_price = get_leg_price_at_time(kite, otm_pe, trade_date, entry_time_val)
    
    if ce_price <= 0 or pe_price <= 0: return None
    net_price = ce_price + pe_price
    
    return {
        "trend": "NEUTRAL",
        "option_to_trade": otm_ce or atm_ce,
        "option_type": "STRANGLE",
        "entry_price": net_price,
        "entry_time": entry_time_val,
        "reason": f"Long Strangle - Buy OTM CE ({otm_ce_strike}) + OTM PE ({otm_pe_strike})",
        "multi_leg": True,
        "legs": [
            {"action": "BUY", "type": "CE", "strike": otm_ce_strike, "price": ce_price, "tradingsymbol": otm_ce.get("tradingsymbol") if otm_ce else "N/A"},
            {"action": "BUY", "type": "PE", "strike": otm_pe_strike, "price": pe_price, "tradingsymbol": otm_pe.get("tradingsymbol") if otm_pe else "N/A"}
        ]
    }

def strategy_bull_call_spread(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date):
    """Bull Call Spread - Buy ITM CE, Sell OTM CE @ 9:20 AM"""
    entry_time_val = "09:20:00"
    
    # In simulation, we check if current time is around 9:20
    last_candle_date = trading_candles[-1].get("date", "")
    current_time_str = last_candle_date.strftime("%H:%M:%S") if isinstance(last_candle_date, datetime) else str(last_candle_date).split(' ')[1][:8]
    
    # Only trigger around the scheduled time (2-minute window)
    if not ("09:20:00" <= current_time_str <= "09:22:00"):
        if int(current_time_str.split(':')[1]) % 30 == 0:
            add_live_log(f"Bull Call Spread: Scheduled for {entry_time_val}. Currently scanning...", "debug")
        return None

    itm_ce_strike = current_strike - 50
    otm_ce_strike = current_strike + 50
    
    itm_ce = find_option(nifty_options, itm_ce_strike, "CE", trade_date)
    otm_ce = find_option(nifty_options, otm_ce_strike, "CE", trade_date)
    
    buy_price = get_leg_price_at_time(kite, itm_ce, trade_date, entry_time_val)
    sell_price = get_leg_price_at_time(kite, otm_ce, trade_date, entry_time_val)
    
    if buy_price <= 0 or sell_price <= 0: return None
    # Net value: Buy - Sell
    net_value = buy_price - sell_price
    
    return {
        "trend": "BULLISH",
        "option_to_trade": itm_ce or atm_ce,
        "option_type": "BULL_CALL_SPREAD",
        "entry_price": net_value,
        "entry_time": entry_time_val,
        "reason": f"Bull Call Spread - Buy ITM CE ({itm_ce_strike}), Sell OTM CE ({otm_ce_strike})",
        "multi_leg": True,
        "legs": [
            {"action": "BUY", "type": "CE", "strike": itm_ce_strike, "price": buy_price, "tradingsymbol": itm_ce.get("tradingsymbol") if itm_ce else "N/A"},
            {"action": "SELL", "type": "CE", "strike": otm_ce_strike, "price": sell_price, "tradingsymbol": otm_ce.get("tradingsymbol") if otm_ce else "N/A"}
        ]
    }

def strategy_bear_put_spread(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date):
    """Bear Put Spread - Buy ITM PE, Sell OTM PE @ 9:20 AM"""
    entry_time_val = "09:20:00"
    
    # In simulation, we check if current time is around 9:20
    last_candle_date = trading_candles[-1].get("date", "")
    current_time_str = last_candle_date.strftime("%H:%M:%S") if isinstance(last_candle_date, datetime) else str(last_candle_date).split(' ')[1][:8]
    
    # Only trigger around the scheduled time (2-minute window)
    if not ("09:20:00" <= current_time_str <= "09:22:00"):
        if int(current_time_str.split(':')[1]) % 30 == 0:
            add_live_log(f"Bear Put Spread: Scheduled for {entry_time_val}. Currently scanning...", "debug")
        return None

    itm_pe_strike = current_strike + 50
    otm_pe_strike = current_strike - 50
    
    itm_pe = find_option(nifty_options, itm_pe_strike, "PE", trade_date)
    otm_pe = find_option(nifty_options, otm_pe_strike, "PE", trade_date)
    
    buy_price = get_leg_price_at_time(kite, itm_pe, trade_date, entry_time_val)
    sell_price = get_leg_price_at_time(kite, otm_pe, trade_date, entry_time_val)
    
    if buy_price <= 0 or sell_price <= 0: return None
    # Net value: Buy - Sell
    net_value = buy_price - sell_price
    
    return {
        "trend": "BEARISH",
        "option_to_trade": itm_pe or atm_pe,
        "option_type": "BEAR_PUT_SPREAD",
        "entry_price": net_value,
        "entry_time": entry_time_val,
        "reason": f"Bear Put Spread - Buy ITM PE ({itm_pe_strike}), Sell OTM PE ({otm_pe_strike})",
        "multi_leg": True,
        "legs": [
            {"action": "BUY", "type": "PE", "strike": itm_pe_strike, "price": buy_price, "tradingsymbol": itm_pe.get("tradingsymbol") if itm_pe else "N/A"},
            {"action": "SELL", "type": "PE", "strike": otm_pe_strike, "price": sell_price, "tradingsymbol": otm_pe.get("tradingsymbol") if otm_pe else "N/A"}
        ]
    }

def strategy_iron_condor(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date):
    """Iron Condor - Sell OTM CE/PE, Buy further OTM CE/PE @ 9:20 AM"""
    entry_time_val = "09:20:00"
    
    # In simulation, we check if current time is around 9:20
    last_candle_date = trading_candles[-1].get("date", "")
    current_time_str = last_candle_date.strftime("%H:%M:%S") if isinstance(last_candle_date, datetime) else str(last_candle_date).split(' ')[1][:8]
    
    # Only trigger around the scheduled time (2-minute window)
    if not ("09:20:00" <= current_time_str <= "09:22:00"):
        if int(current_time_str.split(':')[1]) % 30 == 0:
            add_live_log(f"Iron Condor: Scheduled for {entry_time_val}. Currently scanning...", "debug")
        return None

    sell_ce_strike = current_strike + 50
    buy_ce_strike = current_strike + 100
    sell_pe_strike = current_strike - 50
    buy_pe_strike = current_strike - 100
    
    s_ce = find_option(nifty_options, sell_ce_strike, "CE", trade_date)
    b_ce = find_option(nifty_options, buy_ce_strike, "CE", trade_date)
    s_pe = find_option(nifty_options, sell_pe_strike, "PE", trade_date)
    b_pe = find_option(nifty_options, buy_pe_strike, "PE", trade_date)
    
    p1 = get_leg_price_at_time(kite, s_ce, trade_date, entry_time_val)
    p2 = get_leg_price_at_time(kite, b_ce, trade_date, entry_time_val)
    p3 = get_leg_price_at_time(kite, s_pe, trade_date, entry_time_val)
    p4 = get_leg_price_at_time(kite, b_pe, trade_date, entry_time_val)
    
    # Calculate net value: Buy legs - Sell legs (Standard Net Market Value)
    net_value = (p2 - p1) + (p4 - p3)
    
    return {
        "trend": "NEUTRAL",
        "option_to_trade": atm_ce,
        "option_type": "IRON_CONDOR",
        "entry_price": net_value,
        "entry_time": entry_time_val,
        "reason": f"Iron Condor - Sell {sell_ce_strike}/{sell_pe_strike}, Buy {buy_ce_strike}/{buy_pe_strike}",
        "multi_leg": True,
        "legs": [
            {"action": "SELL", "type": "CE", "strike": sell_ce_strike, "price": p1, "tradingsymbol": s_ce.get("tradingsymbol") if s_ce else "N/A"},
            {"action": "BUY", "type": "CE", "strike": buy_ce_strike, "price": p2, "tradingsymbol": b_ce.get("tradingsymbol") if b_ce else "N/A"},
            {"action": "SELL", "type": "PE", "strike": sell_pe_strike, "price": p3, "tradingsymbol": s_pe.get("tradingsymbol") if s_pe else "N/A"},
            {"action": "BUY", "type": "PE", "strike": buy_pe_strike, "price": p4, "tradingsymbol": b_pe.get("tradingsymbol") if b_pe else "N/A"}
        ]
    }

def strategy_macd_crossover(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str):
    """MACD Crossover Strategy - Single Leg Trend Following"""
    if len(trading_candles) < 30:
        return None
    
    closes = [c.get("close", 0) for c in trading_candles if c.get("close", 0) > 0]
    if len(closes) < 26:
        return None
    
    df = pd.DataFrame({'close': closes})
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    current_macd = macd.iloc[-1]
    prev_macd = macd.iloc[-2]
    current_signal = signal.iloc[-1]
    prev_signal = signal.iloc[-2]
    
    # Get actual candle time
    last_candle_date = trading_candles[-1].get("date", "")
    if isinstance(last_candle_date, datetime):
        entry_time_val = last_candle_date.strftime("%H:%M:%S")
    else:
        entry_time_val = last_candle_date.split(' ')[1][:8] if ' ' in str(last_candle_date) else "11:00:00"

    # Bullish Cross
    if prev_macd < prev_signal and current_macd > current_signal:
        return {
            "trend": "BULLISH",
            "option_to_trade": atm_ce,
            "option_type": "CE",
            "entry_price": closes[-1],
            "entry_time": entry_time_val,
            "reason": f"MACD Bullish Cross: MACD({round(current_macd, 2)}) crossed above Signal({round(current_signal, 2)})"
        }
    
    # Bearish Cross
    if prev_macd > prev_signal and current_macd < current_signal:
        return {
            "trend": "BEARISH",
            "option_to_trade": atm_pe,
            "option_type": "PE",
            "entry_price": closes[-1],
            "entry_time": entry_time_val,
            "reason": f"MACD Bearish Cross: MACD({round(current_macd, 2)}) crossed below Signal({round(current_signal, 2)})"
        }
    
    return None

def strategy_rsi_reversal(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str):
    """RSI Reversal Strategy - Single Leg Mean Reversion"""
    if len(trading_candles) < 15:
        return None
    
    closes = [c.get("close", 0) for c in trading_candles if c.get("close", 0) > 0]
    if len(closes) < 14:
        return None
    
    df = pd.DataFrame({'close': closes})
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    current_rsi = df['rsi'].iloc[-1]
    prev_rsi = df['rsi'].iloc[-2]
    
    # Get actual candle time
    last_candle_date = trading_candles[-1].get("date", "")
    if isinstance(last_candle_date, datetime):
        entry_time_val = last_candle_date.strftime("%H:%M:%S")
    else:
        entry_time_val = last_candle_date.split(' ')[1][:8] if ' ' in str(last_candle_date) else "12:00:00"

    # Oversold Reversal (Bullish)
    if prev_rsi < 30 and current_rsi > 30:
        return {
            "trend": "BULLISH",
            "option_to_trade": atm_ce,
            "option_type": "CE",
            "entry_price": closes[-1],
            "entry_time": entry_time_val,
            "reason": f"RSI Reversal: RSI recovered from oversold ({round(prev_rsi, 2)} -> {round(current_rsi, 2)})"
        }
    
    # Overbought Reversal (Bearish)
    if prev_rsi > 70 and current_rsi < 70:
        return {
            "trend": "BEARISH",
            "option_to_trade": atm_pe,
            "option_type": "PE",
            "entry_price": closes[-1],
            "entry_time": entry_time_val,
            "reason": f"RSI Reversal: RSI retreated from overbought ({round(prev_rsi, 2)} -> {round(current_rsi, 2)})"
        }
    
    return None

def strategy_ema_cross(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str):
    """EMA Crossover Strategy (9/21) - Single Leg Trend Following"""
    if len(trading_candles) < 25:
        return None
    
    closes = [c.get("close", 0) for c in trading_candles if c.get("close", 0) > 0]
    if len(closes) < 21:
        return None
    
    df = pd.DataFrame({'close': closes})
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    curr_9 = df['ema_9'].iloc[-1]
    prev_9 = df['ema_9'].iloc[-2]
    curr_21 = df['ema_21'].iloc[-1]
    prev_21 = df['ema_21'].iloc[-2]
    
    # Get actual candle time
    last_candle_date = trading_candles[-1].get("date", "")
    if isinstance(last_candle_date, datetime):
        entry_time_val = last_candle_date.strftime("%H:%M:%S")
    else:
        entry_time_val = last_candle_date.split(' ')[1][:8] if ' ' in str(last_candle_date) else "13:00:00"

    # Golden Cross
    if prev_9 < prev_21 and curr_9 > curr_21:
        return {
            "trend": "BULLISH",
            "option_to_trade": atm_ce,
            "option_type": "CE",
            "entry_price": closes[-1],
            "entry_time": entry_time_val,
            "reason": f"EMA Golden Cross: 9 EMA ({round(curr_9, 2)}) crossed above 21 EMA ({round(curr_21, 2)})"
        }
    
    # Death Cross
    if prev_9 > prev_21 and curr_9 < curr_21:
        return {
            "trend": "BEARISH",
            "option_to_trade": atm_pe,
            "option_type": "PE",
            "entry_price": closes[-1],
            "entry_time": entry_time_val,
            "reason": f"EMA Death Cross: 9 EMA ({round(curr_9, 2)}) crossed below 21 EMA ({round(curr_21, 2)})"
        }
    
    return None

# TODO: Move to api/v1/routes/strategies.py
# @app.post("/backtest-nifty50-options")
# async def backtest_nifty50_options(req: Request):
#     """Backtest Nifty50 options strategy for given date range with multiple strategy options"""
#     ... (function body to be moved to api/v1/routes/strategies.py - ~700 lines removed)
#     pass  # Function body moved to api/v1/routes/strategies.py

# ============================================================================
# AI Agent Endpoints
# NOTE: Agent endpoints are now handled by api/v1/routes/agent.py
# These legacy endpoints are kept for backward compatibility but will be deprecated
# ============================================================================

# Legacy agent endpoints - use /api/v1/agent/* instead
# @app.post("/agent/chat")
async def agent_chat(req: Request):
    """Natural language interaction with the AI agent"""
    try:
        payload = await req.json()
        user_query = payload.get("message", "")
        session_id = payload.get("session_id", "default")

        if not user_query:
            raise HTTPException(status_code=400, detail="Message is required")

        # Save user message to database
        try:
            chat_repo = get_chat_repository()
            user_message = ChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="user",
                content=user_query,
                timestamp=datetime.now()
            )
            chat_repo.save(user_message)
        except Exception as e:
            print(f"Error saving user message: {e}")
        
        # Get context (positions, balance, etc.)
        context = {}
        try:
            kite = get_kite_instance()
            positions = kite.positions().get("net", [])
            margins = kite.margins()
            context = {
                "positions": positions,
                "balance": margins.get("equity", {})
            }
        except:
            pass  # Continue without context if auth fails
        
        # Run agent
        result = await run_agent(user_query, context)

        # Save assistant response to database
        try:
            chat_repo = get_chat_repository()
            assistant_message = ChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content=result.get("response", ""),
                timestamp=datetime.now(),
                metadata={"agent_result": result}
            )
            chat_repo.save(assistant_message)
        except Exception as e:
            print(f"Error saving assistant message: {e}")

        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in agent chat: {str(e)}")

@app.post("/agent/execute")
async def agent_execute(req: Request):
    """Direct agent execution with approval workflow"""
    try:
        payload = await req.json()
        user_query = payload.get("message", "")
        auto_approve = payload.get("auto_approve", False)
        
        if not user_query:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Get context
        context = {}
        try:
            kite = get_kite_instance()
            positions = kite.positions().get("net", [])
            margins = kite.margins()
            context = {
                "positions": positions,
                "balance": margins.get("equity", {})
            }
        except:
            pass
        
        # Run agent
        result = await run_agent(user_query, context)
        
        # If requires approval and not auto-approved, return approval info
        if result.get("requires_approval") and not auto_approve:
            return {
                "status": "pending_approval",
                "data": result,
                "approval_id": result.get("approval_id")
            }
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in agent execute: {str(e)}")

@app.get("/agent/status")
def get_agent_status():
    """Get agent status and context"""
    try:
        memory = get_agent_memory()
        safety = get_safety_manager()
        approval_queue = get_approval_queue()
        
        # Get recent context
        context_summary = memory.get_context_summary()
        safety_status = safety.get_status()
        approval_stats = approval_queue.get_stats()
        
        return {
            "status": "success",
            "data": {
                "context_summary": context_summary,
                "safety_status": safety_status,
                "approval_stats": approval_stats,
                "memory_messages": len(memory.get_messages())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent status: {str(e)}")

@app.get("/agent/approvals")
def get_approvals():
    """Get pending approvals"""
    try:
        approval_queue = get_approval_queue()
        pending = approval_queue.list_pending()

        return {
            "status": "success",
            "data": {
                "approvals": pending,
                "count": len(pending)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting approvals: {str(e)}")

@app.get("/agent/approved-trades")
def get_approved_trades():
    """Get all approved trades"""
    try:
        approval_queue = get_approval_queue()
        approved = approval_queue.list_approved()

        return {
            "status": "success",
            "data": {
                "trades": approved,
                "count": len(approved)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting approved trades: {str(e)}")

@app.get("/agent/logs")
def get_agent_logs(limit: int = 100, component: str = None):
    """Get agent logs from database"""
    try:
        log_repo = get_log_repository()
        logs = log_repo.get_recent(limit=limit, component=component if component else None)

        return {
            "status": "success",
            "data": {
                "logs": [log.model_dump() for log in logs],
                "count": len(logs)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting logs: {str(e)}")

@app.get("/agent/tool-executions")
def get_tool_executions(tool_name: str = None, limit: int = 50):
    """Get tool execution history from database"""
    try:
        tool_repo = get_tool_repository()
        executions = tool_repo.get_recent(tool_name=tool_name if tool_name else None, limit=limit)

        return {
            "status": "success",
            "data": {
                "executions": [exec.model_dump() for exec in executions],
                "count": len(executions)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting tool executions: {str(e)}")

@app.get("/agent/simulations")
def get_simulations(limit: int = 10):
    """Get simulation results from database"""
    try:
        sim_repo = get_simulation_repository()
        simulations = sim_repo.get_recent(limit=limit)

        return {
            "status": "success",
            "data": {
                "simulations": [sim.model_dump() for sim in simulations],
                "count": len(simulations)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting simulations: {str(e)}")

@app.get("/agent/chat-history")
def get_chat_history(session_id: str = None, limit: int = 100):
    """Get chat message history from database"""
    try:
        chat_repo = get_chat_repository()
        if session_id:
            messages = chat_repo.get_session_messages(session_id, limit=limit)
        else:
            messages = chat_repo.get_recent_messages(limit=limit)

        return {
            "status": "success",
            "data": {
                "messages": [msg.model_dump() for msg in messages],
                "count": len(messages)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chat history: {str(e)}")

@app.post("/agent/approve/{approval_id}")
async def approve_action(approval_id: str, req: Request):
    """Approve a pending action"""
    try:
        payload = await req.json() if hasattr(req, 'method') and req.method == "POST" else {}
        approved_by = payload.get("approved_by", "user")
        
        approval_queue = get_approval_queue()
        success = approval_queue.approve(approval_id, approved_by)
        
        if not success:
            raise HTTPException(status_code=400, detail="Approval not found or already processed")
        
        approval = approval_queue.get_approval(approval_id)
        
        # --- EXECUTE ACTUAL TRADE ON ZERODHA ---
        execution_msg = ""
        action = approval.get("action", "")
        details = approval.get("details", {})
        
        if action.startswith("LIVE_") and not details.get("is_simulated", False):
            try:
                symbol = details.get("symbol")
                transaction_type = details.get("type", "BUY")
                quantity = int(details.get("qty", 1))
                price = float(details.get("price", 0))
                
                add_agent_log(f"Executing Approved Trade: {transaction_type} {quantity} {symbol}...", "signal")
                
                # Place actual order on Kite
                order_result = place_order_tool.invoke({
                    "tradingsymbol": symbol,
                    "transaction_type": transaction_type,
                    "quantity": quantity,
                    "order_type": "MARKET", # Using MARKET for approved signals
                    "product": "MIS",       # Default to Intraday
                    "exchange": "NSE"       # Default to NSE for stocks
                })
                
                if order_result.get("status") == "success":
                    entry_order_id = str(order_result.get('order_id'))
                    execution_msg = f"Entry Order Placed: {entry_order_id}"
                    add_agent_log(f"SUCCESS: {execution_msg}", "info")
                    
                    # Store entry order ID
                    approval_queue.repo.update_order_ids(approval_id, entry_order_id=entry_order_id)
                    
                    # --- PLACE EXIT ORDERS (Stop Loss & Target) ---
                    config = get_agent_config()
                    exit_type = "SELL" if transaction_type == "BUY" else "BUY"
                    sl_price = float(details.get("sl", 0))
                    tp_price = float(details.get("tp", 0))
                    sl_order_id = None
                    tp_order_id = None
                    gtt_trigger_id = None
                    
                    # Determine product type
                    product = details.get("product", "MIS")
                    use_gtt = False
                    
                    # Log GTT configuration check
                    add_agent_log(f"GTT Check: use_gtt_orders={config.use_gtt_orders}, product={product}, gtt_for_intraday={config.gtt_for_intraday}, gtt_for_positional={config.gtt_for_positional}", "debug")
                    
                    if config.use_gtt_orders:
                        if product == "CNC" and config.gtt_for_positional:
                            use_gtt = True
                            add_agent_log(f"GTT Enabled: Using GTT for positional trade (CNC)", "info")
                        elif product == "MIS" and config.gtt_for_intraday:
                            use_gtt = True
                            add_agent_log(f"GTT Enabled: Using GTT for intraday trade (MIS)", "info")
                        else:
                            add_agent_log(f"GTT Disabled: Product={product} but gtt_for_intraday={config.gtt_for_intraday}, gtt_for_positional={config.gtt_for_positional}", "debug")
                    else:
                        add_agent_log(f"GTT Disabled: use_gtt_orders is False", "debug")
                    
                    # Use GTT OCO order if enabled
                    if use_gtt and sl_price > 0 and tp_price > 0:
                        try:
                            # Get current price for GTT
                            from utils.kite_utils import get_kite_instance
                            kite = get_kite_instance()
                            quote = kite.quote(f"NSE:{symbol}")
                            instrument_key = f"NSE:{symbol}"
                            current_price = quote[instrument_key].get("last_price", price) if instrument_key in quote else price
                            
                            # Calculate trigger prices
                            # For BUY: SL triggers when price falls to SL, TP triggers when price rises to TP
                            # For SELL: SL triggers when price rises to SL, TP triggers when price falls to TP
                            if transaction_type == "BUY":
                                sl_trigger = sl_price * 1.001  # Trigger when price falls to SL (slightly above to ensure trigger)
                                tp_trigger = tp_price * 0.999  # Trigger when price rises to TP (slightly below to ensure trigger)
                            else:  # SELL
                                sl_trigger = sl_price * 0.999  # Trigger when price rises to SL (slightly below to ensure trigger)
                                tp_trigger = tp_price * 1.001  # Trigger when price falls to TP (slightly above to ensure trigger)
                            
                            gtt_result = place_gtt_tool.invoke({
                                "tradingsymbol": symbol,
                                "exchange": "NSE",
                                "trigger_type": "two-leg",  # OCO order
                                "trigger_prices": [sl_trigger, tp_trigger],
                                "last_price": current_price,
                                "stop_loss_price": round(sl_price, 1),
                                "target_price": round(tp_price, 1),
                                "quantity": quantity,
                                "transaction_type": exit_type,
                                "product": product
                            })
                            
                            if gtt_result.get("status") == "success":
                                gtt_trigger_id = str(gtt_result.get("trigger_id"))
                                add_agent_log(f"✅ GTT OCO Order Placed Successfully!", "info")
                                add_agent_log(f"   Trigger ID: {gtt_trigger_id}", "info")
                                add_agent_log(f"   Stop Loss: ₹{sl_price} (Trigger: ₹{round(sl_trigger, 2)})", "info")
                                add_agent_log(f"   Target: ₹{tp_price} (Trigger: ₹{round(tp_trigger, 2)})", "info")
                                # Store GTT trigger ID in approval details
                                approval_queue.repo.update_order_ids(approval_id, sl_order_id=gtt_trigger_id)
                                execution_msg += f" | GTT OCO: {gtt_trigger_id}"
                            else:
                                error_msg = gtt_result.get('error', 'Unknown error')
                                add_agent_log(f"❌ GTT Order Failed: {error_msg}", "error")
                                add_agent_log(f"   Falling back to regular SL-M + LIMIT orders", "warning")
                                use_gtt = False  # Fall back to regular orders
                        except Exception as e:
                            add_agent_log(f"Error placing GTT order: {e}. Falling back to regular orders.", "warning")
                            use_gtt = False
                    
                    # Fall back to regular SL-M + LIMIT orders if GTT not used
                    if not use_gtt:
                        # 1. Place Stop Loss Order (SL-M)
                        if sl_price > 0:
                            sl_result = place_order_tool.invoke({
                                "tradingsymbol": symbol,
                                "transaction_type": exit_type,
                                "quantity": quantity,
                                "order_type": "SL-M",
                                "trigger_price": round(sl_price, 1), # Kite requires 0.05 or 0.1 precision
                                "product": product,
                                "exchange": "NSE"
                            })
                            if sl_result.get("status") == "success":
                                sl_order_id = str(sl_result.get('order_id'))
                                add_agent_log(f"Stop Loss Order Placed @ ₹{sl_price}: {sl_order_id}", "info")
                                approval_queue.repo.update_order_ids(approval_id, sl_order_id=sl_order_id)
                            else:
                                add_agent_log(f"Stop Loss Order Failed: {sl_result.get('error')}", "error")

                        # 2. Place Target Order (LIMIT)
                        if tp_price > 0:
                            tp_result = place_order_tool.invoke({
                                "tradingsymbol": symbol,
                                "transaction_type": exit_type,
                                "quantity": quantity,
                                "order_type": "LIMIT",
                                "price": round(tp_price, 1),
                                "product": product,
                                "exchange": "NSE"
                            })
                            if tp_result.get("status") == "success":
                                tp_order_id = str(tp_result.get('order_id'))
                                add_agent_log(f"Target Order Placed @ ₹{tp_price}: {tp_order_id}", "info")
                                approval_queue.repo.update_order_ids(approval_id, tp_order_id=tp_order_id)
                            else:
                                add_agent_log(f"Target Order Failed: {tp_result.get('error')}", "error")
                            
                else:
                    execution_msg = f"Order Failed: {order_result.get('error')}"
                    add_agent_log(f"ERROR: {execution_msg}", "error")
                    
            except Exception as e:
                execution_msg = f"Execution Error: {str(e)}"
                add_agent_log(f"CRITICAL: {execution_msg}", "error")
        else:
            execution_msg = "Simulation Approval recorded (No live order placed)"
            add_agent_log(execution_msg, "info")

        # BROADCAST: Notify all clients that an approval was processed
        asyncio.create_task(broadcast_agent_update("APPROVAL_PROCESSED", {
            "approval_id": approval_id,
            "status": "APPROVED",
            "approval": approval,
            "execution_message": execution_msg
        }))

        return {
            "status": "success",
            "data": {
                "approval_id": approval_id,
                "approval": approval,
                "message": execution_msg
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error approving action: {str(e)}")

@app.post("/agent/reject/{approval_id}")
async def reject_action(approval_id: str, req: Request):
    """Reject a pending action"""
    try:
        payload = await req.json() if hasattr(req, 'method') and req.method == "POST" else {}
        reason = payload.get("reason", "")
        rejected_by = payload.get("rejected_by", "user")
        
        approval_queue = get_approval_queue()
        success = approval_queue.reject(approval_id, reason, rejected_by)
        
        if not success:
            raise HTTPException(status_code=400, detail="Approval not found or already processed")
        
        approval = approval_queue.get_approval(approval_id)
        
        # BROADCAST: Notify all clients that an approval was processed
        asyncio.create_task(broadcast_agent_update("APPROVAL_PROCESSED", {
            "approval_id": approval_id,
            "status": "REJECTED",
            "approval": approval
        }))

        return {
            "status": "success",
            "data": {
                "approval_id": approval_id,
                "approval": approval,
                "message": "Rejection successful"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rejecting action: {str(e)}")

@app.get("/agent/config")
def get_agent_config_endpoint():
    """Get agent configuration"""
    try:
        config = get_agent_config()
        
        # Try to get live funds from Zerodha if possible
        zerodha_funds = 0.0
        try:
            kite = get_kite_instance()
            margins = kite.margins()
            equity_data = margins.get('equity', {})
            available_value = equity_data.get('available', 0)
            
            if isinstance(available_value, dict):
                zerodha_funds = float(available_value.get('cash', 0))
                if zerodha_funds == 0:
                    zerodha_funds = float(equity_data.get('opening_balance', 0) or equity_data.get('live_balance', 0))
            else:
                zerodha_funds = float(available_value if available_value else equity_data.get('cash', 0) or equity_data.get('opening_balance', 0))
        except Exception:
            # Silently fail if Kite not initialized
            pass

        return {
            "status": "success",
            "data": {
                "llm_provider": config.llm_provider.value,
                "openai_api_key": config.openai_api_key or "",
                "anthropic_api_key": config.anthropic_api_key or "",
                "ollama_base_url": config.ollama_base_url,
                "agent_model": config.agent_model,
                "agent_temperature": config.agent_temperature,
                "max_tokens": config.max_tokens,
                "auto_trade_threshold": auto_trade_threshold,
                "max_position_size": max_position_size,
                "trading_capital": trading_capital,
                "daily_loss_limit": daily_loss_limit,
                "max_trades_per_day": config.max_trades_per_day,
                "risk_per_trade_pct": config.risk_per_trade_pct,
                "reward_per_trade_pct": config.reward_per_trade_pct,
                "autonomous_mode": config.autonomous_mode,
                "autonomous_scan_interval_mins": config.autonomous_scan_interval_mins,
                "autonomous_target_group": config.autonomous_target_group,
                "active_strategies": config.active_strategies,
                "is_auto_trade_enabled": config.is_auto_trade_enabled,
                "vwap_proximity_pct": config.vwap_proximity_pct,
                "vwap_group_proximity_pct": config.vwap_group_proximity_pct,
                "rejection_shadow_pct": config.rejection_shadow_pct,
                "prime_session_start": config.prime_session_start,
                "prime_session_end": config.prime_session_end,
                "intraday_square_off_time": config.intraday_square_off_time,
                "trading_start_time": config.trading_start_time,
                "trading_end_time": config.trading_end_time,
                "circuit_breaker_enabled": config.circuit_breaker_enabled,
                "circuit_breaker_loss_threshold": circuit_breaker_loss_threshold,
                "use_gtt_orders": config.use_gtt_orders,
                "gtt_for_intraday": config.gtt_for_intraday,
                "gtt_for_positional": config.gtt_for_positional,
                "kite_api_key": config.kite_api_key or "",
                "kite_api_secret": config.kite_api_secret or "",
                "kite_redirect_uri": config.kite_redirect_uri,
                "zerodha_funds": zerodha_funds
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent config: {str(e)}")

@app.post("/agent/config")
async def update_agent_config(req: Request):
    """Update agent configuration"""
    try:
        payload = await req.json()
        config = get_agent_config()
        
        # Update config fields
        if "llm_provider" in payload:
            from agent.config import LLMProvider
            config.llm_provider = LLMProvider(payload["llm_provider"])
        if "openai_api_key" in payload:
            config.openai_api_key = payload["openai_api_key"]
        if "anthropic_api_key" in payload:
            config.anthropic_api_key = payload["anthropic_api_key"]
        if "ollama_base_url" in payload:
            config.ollama_base_url = payload["ollama_base_url"]
        if "agent_model" in payload:
            config.agent_model = payload["agent_model"]
        if "agent_temperature" in payload:
            config.agent_temperature = float(payload["agent_temperature"])
        if "max_tokens" in payload:
            config.max_tokens = int(payload["max_tokens"])
        if "trading_capital" in payload:
            config.trading_capital = float(payload["trading_capital"])
        if "auto_trade_threshold" in payload:
            config.auto_trade_threshold = float(payload["auto_trade_threshold"])
        if "max_position_size" in payload:
            config.max_position_size = float(payload["max_position_size"])
        if "daily_loss_limit" in payload:
            config.daily_loss_limit = float(payload["daily_loss_limit"])
        if "max_trades_per_day" in payload:
            config.max_trades_per_day = int(payload["max_trades_per_day"])
        if "risk_per_trade_pct" in payload:
            config.risk_per_trade_pct = float(payload["risk_per_trade_pct"])
        if "reward_per_trade_pct" in payload:
            config.reward_per_trade_pct = float(payload["reward_per_trade_pct"])
        if "autonomous_mode" in payload:
            config.autonomous_mode = bool(payload["autonomous_mode"])
        if "autonomous_scan_interval_mins" in payload:
            config.autonomous_scan_interval_mins = int(payload["autonomous_scan_interval_mins"])
        if "autonomous_target_group" in payload:
            config.autonomous_target_group = str(payload["autonomous_target_group"])
        if "active_strategies" in payload:
            config.active_strategies = str(payload["active_strategies"])
        if "is_auto_trade_enabled" in payload:
            config.is_auto_trade_enabled = bool(payload["is_auto_trade_enabled"])
        if "vwap_proximity_pct" in payload:
            config.vwap_proximity_pct = float(payload["vwap_proximity_pct"])
        if "vwap_group_proximity_pct" in payload:
            config.vwap_group_proximity_pct = float(payload["vwap_group_proximity_pct"])
        if "rejection_shadow_pct" in payload:
            config.rejection_shadow_pct = float(payload["rejection_shadow_pct"])
        if "prime_session_start" in payload:
            config.prime_session_start = str(payload["prime_session_start"])
        if "prime_session_end" in payload:
            config.prime_session_end = str(payload["prime_session_end"])
        if "intraday_square_off_time" in payload:
            config.intraday_square_off_time = str(payload["intraday_square_off_time"])
        if "trading_start_time" in payload:
            config.trading_start_time = str(payload["trading_start_time"])
        if "trading_end_time" in payload:
            config.trading_end_time = str(payload["trading_end_time"])
        if "circuit_breaker_enabled" in payload:
            config.circuit_breaker_enabled = bool(payload["circuit_breaker_enabled"])
        if "circuit_breaker_loss_threshold" in payload:
            config.circuit_breaker_loss_threshold = float(payload["circuit_breaker_loss_threshold"])
        if "use_gtt_orders" in payload:
            config.use_gtt_orders = bool(payload["use_gtt_orders"])
        if "gtt_for_intraday" in payload:
            config.gtt_for_intraday = bool(payload["gtt_for_intraday"])
        if "gtt_for_positional" in payload:
            config.gtt_for_positional = bool(payload["gtt_for_positional"])
        if "kite_api_key" in payload:
            config.kite_api_key = str(payload["kite_api_key"])
        if "kite_api_secret" in payload:
            config.kite_api_secret = str(payload["kite_api_secret"])
        if "kite_redirect_uri" in payload:
            config.kite_redirect_uri = str(payload["kite_redirect_uri"])
            
        # Persist to .env file
        with open(".env", "w") as f:
            f.write(f"LLM_PROVIDER={config.llm_provider.value}\n")
            f.write(f"OPENAI_API_KEY={config.openai_api_key or ''}\n")
            f.write(f"ANTHROPIC_API_KEY={config.anthropic_api_key or ''}\n")
            f.write(f"OLLAMA_BASE_URL={config.ollama_base_url}\n")
            f.write(f"AGENT_MODEL={config.agent_model}\n")
            f.write(f"AGENT_TEMPERATURE={config.agent_temperature}\n")
            f.write(f"MAX_TOKENS={config.max_tokens}\n")
            f.write(f"AUTO_TRADE_THRESHOLD={config.auto_trade_threshold}\n")
            f.write(f"MAX_POSITION_SIZE={config.max_position_size}\n")
            f.write(f"TRADING_CAPITAL={config.trading_capital}\n")
            f.write(f"DAILY_LOSS_LIMIT={config.daily_loss_limit}\n")
            f.write(f"MAX_TRADES_PER_DAY={config.max_trades_per_day}\n")
            f.write(f"RISK_PER_TRADE_PCT={config.risk_per_trade_pct}\n")
            f.write(f"REWARD_PER_TRADE_PCT={config.reward_per_trade_pct}\n")
            f.write(f"AUTONOMOUS_MODE={config.autonomous_mode}\n")
            f.write(f"AUTONOMOUS_SCAN_INTERVAL_MINS={config.autonomous_scan_interval_mins}\n")
            f.write(f"AUTONOMOUS_TARGET_GROUP={config.autonomous_target_group}\n")
            f.write(f"ACTIVE_STRATEGIES={config.active_strategies}\n")
            f.write(f"IS_AUTO_TRADE_ENABLED={config.is_auto_trade_enabled}\n")
            f.write(f"VWAP_PROXIMITY_PCT={config.vwap_proximity_pct}\n")
            f.write(f"VWAP_GROUP_PROXIMITY_PCT={config.vwap_group_proximity_pct}\n")
            f.write(f"REJECTION_SHADOW_PCT={config.rejection_shadow_pct}\n")
            f.write(f"PRIME_SESSION_START={config.prime_session_start}\n")
            f.write(f"PRIME_SESSION_END={config.prime_session_end}\n")
            f.write(f"INTRADAY_SQUARE_OFF_TIME={config.intraday_square_off_time}\n")
            f.write(f"TRADING_START_TIME={config.trading_start_time}\n")
            f.write(f"TRADING_END_TIME={config.trading_end_time}\n")
            f.write(f"CIRCUIT_BREAKER_ENABLED={config.circuit_breaker_enabled}\n")
            f.write(f"CIRCUIT_BREAKER_LOSS_THRESHOLD={config.circuit_breaker_loss_threshold}\n")
            f.write(f"USE_GTT_ORDERS={config.use_gtt_orders}\n")
            f.write(f"GTT_FOR_INTRADAY={config.gtt_for_intraday}\n")
            f.write(f"GTT_FOR_POSITIONAL={config.gtt_for_positional}\n")
            f.write(f"KITE_API_KEY={config.kite_api_key or ''}\n")
            f.write(f"KITE_API_SECRET={config.kite_api_secret or ''}\n")
            f.write(f"KITE_REDIRECT_URI={config.kite_redirect_uri}\n")
            
        # BROADCAST: Notify all clients that config has changed
        asyncio.create_task(broadcast_agent_update("CONFIG_UPDATED", payload))

        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "data": payload
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating agent config: {str(e)}")
