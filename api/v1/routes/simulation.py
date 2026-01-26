"""
Simulation-related API endpoints
"""
from fastapi import APIRouter, Request, HTTPException
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np

# Import simulation state and helper functions
from simulation import (
    simulation_state,
    live_logs,
    add_sim_order,
    get_instrument_history,
    calculate_sim_qty,
    find_option
)
from utils.kite_utils import get_kite_instance
from utils.candle_utils import aggregate_to_tf
from strategies.runner import run_strategy_on_candles
from agent.config import get_agent_config

# Import add_live_log from simulation helpers (moved for reuse by strategies)
from simulation.helpers import add_live_log

router = APIRouter(prefix="/simulation", tags=["Simulation"])


@router.get("/live-logs")
async def get_live_logs():
    """Endpoint for UI to fetch latest monitoring logs"""
    return {"data": live_logs}


@router.post("/speed")
async def set_simulation_speed(req: Request):
    payload = await req.json()
    speed = payload.get("speed", 1)
    simulation_state["speed"] = max(1, speed)
    return {"status": "success", "speed": simulation_state["speed"]}


@router.post("/start")
async def start_simulation(req: Request):
    """Start a live-market simulation using historical data"""
    try:
        global live_logs
        live_logs = [] # Clear previous logs
        from agent.logging import add_agent_log
        add_agent_log(f"Initializing simulation...", "info")
        
        kite = get_kite_instance()
        payload = await req.json()
        sim_date_str = payload.get("date", (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
        sim_date = datetime.strptime(sim_date_str, "%Y-%m-%d").date()
        
        add_agent_log(f"Loading historical data for {sim_date_str}...", "info")
        
        # Get Nifty index token
        nse_instruments = kite.instruments("NSE")
        nifty_index = next((inst for inst in nse_instruments if inst.get("tradingsymbol") == "NIFTY 50"), None)
        if not nifty_index: 
            return {"status": "error", "message": "Nifty not found"}
        
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


@router.get("/chart-data")
async def get_simulation_chart_data(timeframe: str = "1m", indicators: str = ""):
    """Get chart data for simulation replay - returns candles up to current index
    timeframe: 1m, 5m, 15m, 30m, 1h, 1d
    indicators: comma-separated list (e.g., "rsi,bollinger,pivot")
    """
    # Import helper functions
    from utils.indicators import calculate_rsi, calculate_bollinger_bands_full, calculate_pivot_points
    
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
    required_prev_candles = max_period_needed * tf_minutes * 2  # 2x buffer for safety
    
    # If we don't have enough previous candles, try to fetch more
    if len(previous_candles) < required_prev_candles and simulation_state.get("date"):
        try:
            kite = get_kite_instance()
            nse_instruments = kite.instruments("NSE")
            nifty_index = next((inst for inst in nse_instruments if inst.get("tradingsymbol") == "NIFTY 50"), None)
            if nifty_index:
                sim_date = datetime.strptime(simulation_state["date"], "%Y-%m-%d").date()
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
        except Exception as e:
            print(f"Warning: Could not fetch additional historical candles: {e}")
    
    previous_for_calc = previous_candles[-required_prev_candles:] if len(previous_candles) > required_prev_candles else previous_candles
    
    # Combine: previous (for calculation) + current simulation candles
    all_candles_for_display = previous_candles + candles_1m[:idx + 1]  # All previous + current up to index
    all_candles_for_calc = previous_for_calc + candles_1m[:idx + 1]  # Enough for calculation
    
    if not all_candles_for_display:
        return {"data": {"candles": [], "current_index": idx, "total_candles": len(candles_1m), "timeframe": timeframe}}
    
    # Aggregate to selected timeframe - use TIME-BASED aggregation (not sequential chunking)
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
    
    # Aggregate for display and calculation
    aggregated_candles = aggregate_candles(all_candles_for_display)
    aggregated_for_indicators = aggregate_candles(all_candles_for_calc)
    
    # Find the starting index in aggregated_for_indicators that matches the first display candle
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
                calc_start_idx = max(0, i - 1)
                break
    
    # Convert to chart format - ensure proper sorting by timestamp
    chart_candles = []
    for candle in aggregated_candles:
        # Convert datetime to Unix timestamp (seconds)
        if isinstance(candle["date"], datetime):
            timestamp = int(candle["date"].timestamp())
        else:
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
    
    # Sort by timestamp
    chart_candles.sort(key=lambda x: x["time"])
    
    # Get current positions and orders for markers
    positions = simulation_state.get("positions", [])
    orders = simulation_state.get("orders", [])
    
    # Markers for entry/exit points
    markers = []
    for order in orders[:50]:  # Last 50 orders
        if order.get("order_timestamp"):
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
    indicator_data = {}
    if indicators:
        indicator_list = [ind.strip().lower() for ind in indicators.split(",") if ind.strip()]
        
        # Get closes/highs/lows for indicator calculation
        closes_for_calc = [float(c["close"]) for c in aggregated_for_indicators]
        highs_for_calc = [float(c["high"]) for c in aggregated_for_indicators]
        lows_for_calc = [float(c["low"]) for c in aggregated_for_indicators]
        
        if "rsi" in indicator_list:
            rsi_values_full = calculate_rsi(closes_for_calc, period=14)
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
                
                calc_idx = calc_start_idx + i
                if calc_idx >= 0 and calc_idx < len(rsi_values_full):
                    rsi_val = rsi_values_full[calc_idx]
                    if not (pd.isna(rsi_val) or np.isnan(rsi_val)):
                        rsi_data.append({
                            "time": timestamp,
                            "value": float(rsi_val)
                        })
            
            rsi_data.sort(key=lambda x: x["time"])
            indicator_data["rsi"] = rsi_data
        
        if "bollinger" in indicator_list:
            upper_full, middle_full, lower_full = calculate_bollinger_bands_full(closes_for_calc, period=20, num_std=2)
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
                
                calc_idx = calc_start_idx + i
                if calc_idx >= 0 and calc_idx < len(upper_full):
                    if not (pd.isna(upper_full[calc_idx]) or pd.isna(middle_full[calc_idx]) or pd.isna(lower_full[calc_idx])):
                        bb_data.append({
                            "time": timestamp,
                            "upper": float(upper_full[calc_idx]),
                            "middle": float(middle_full[calc_idx]),
                            "lower": float(lower_full[calc_idx])
                        })
            
            bb_data.sort(key=lambda x: x["time"])
            indicator_data["bollinger"] = bb_data
        
        if "pivot" in indicator_list and len(aggregated_candles) > 0:
            if len(highs_for_calc) > 0 and len(lows_for_calc) > 0 and len(closes_for_calc) > 0:
                prev_day_count = max(1, len(aggregated_for_indicators) // 5)
                prev_day_candles = aggregated_for_indicators[:prev_day_count]
                
                if prev_day_candles:
                    prev_high = max(float(c["high"]) for c in prev_day_candles)
                    prev_low = min(float(c["low"]) for c in prev_day_candles)
                    prev_close = float(prev_day_candles[-1]["close"])
                else:
                    prev_high = max(highs_for_calc[:20] if len(highs_for_calc) >= 20 else highs_for_calc) if len(highs_for_calc) > 0 else highs_for_calc[0]
                    prev_low = min(lows_for_calc[:20] if len(lows_for_calc) >= 20 else lows_for_calc) if len(lows_for_calc) > 0 else lows_for_calc[0]
                    prev_close = closes_for_calc[0] if len(closes_for_calc) > 0 else closes_for_calc[-1]
                
                pivot_points = calculate_pivot_points(prev_high, prev_low, prev_close)
                
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


@router.get("/state")
async def get_simulation_state():
    """Advance simulation by 1 minute and run active strategies"""
    # find_option is already imported at the top
    
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
    
    # Add Status Logs
    log_interval = 15
    config = get_agent_config()
    if (old_idx // log_interval) != (new_idx // log_interval):
        sim_time = current_candle["date"].strftime("%H:%M:%S")
        status = "Scanning" if config.is_auto_trade_enabled else "Monitoring"
        
        active_strats = config.active_strategies.split(",") if config.active_strategies else []
        pending_strats = [s.strip() for s in active_strats if s.strip() not in simulation_state["executed_strategies"]]
        
        if pending_strats:
            active_strats = f" ({', '.join(pending_strats)})"
            add_live_log(f"{status} market @ Nifty {current_candle['close']}{active_strats}", "debug")
        else:
            add_live_log(f"All selected strategies executed. Monitoring open positions @ Nifty {current_candle['close']}", "debug")
    
    # Update existing positions
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
                    leg_inst = find_option(simulation_state["nifty_options"], leg["strike"], leg["type"], sim_date)
                    if not leg_inst: continue
                    
                    leg_history = get_instrument_history(kite, leg_inst["instrument_token"], sim_date)
                    
                    if idx < len(leg_history):
                        leg_current_price = leg_history[idx]["close"]
                    else:
                        leg_current_price = leg_history[-1]["close"] if leg_history else leg["price"]
                    
                    leg_current_price = max(0.05, leg_current_price)
                    
                    if leg["action"] == "BUY":
                        current_net_price += leg_current_price
                    else:
                        current_net_price -= leg_current_price
                
                pos["last_price"] = round(current_net_price, 2)
                open_pnl = (pos["last_price"] - pos["average_price"]) * pos["quantity"]
                pos["pnl"] = round(pos.get("realized_pnl", 0) + open_pnl, 2)
            else:
                # Single leg
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

            # Check for SL/Target hits
            config = get_agent_config()
            trade_value = abs(pos["average_price"]) * abs(pos["quantity"])
            profit_threshold = trade_value * (config.reward_per_trade_pct / 100)
            loss_threshold = trade_value * (config.risk_per_trade_pct / 100)
            
            current_trade_pnl = pos["pnl"] - pos.get("realized_pnl", 0)
            
            if current_trade_pnl >= profit_threshold:
                pos["realized_pnl"] = pos["pnl"]
                old_qty = pos["quantity"]
                pos["quantity"] = 0
                pos["exit_reason"] = "Target Hit"
                
                strats = [s.strip() for s in pos["strategy"].split(",")]
                for s in strats:
                    simulation_state["strategy_cooldown"][s] = simulation_state["time"]
                
                add_live_log(f"SIM EXIT: Target hit for {pos['strategy']} @ P&L: +{current_trade_pnl}", "success")
                add_sim_order(pos["strategy"], pos["tradingsymbol"], "SELL", old_qty, pos["last_price"], reason="Target Hit")
            elif current_trade_pnl <= -loss_threshold:
                pos["realized_pnl"] = pos["pnl"]
                old_qty = pos["quantity"]
                pos["quantity"] = 0
                pos["exit_reason"] = "Stoploss Hit"
                
                strats = [s.strip() for s in pos["strategy"].split(",")]
                for s in strats:
                    simulation_state["strategy_cooldown"][s] = simulation_state["time"]
                
                add_live_log(f"SIM EXIT: Stoploss hit for {pos['strategy']} @ P&L: {current_trade_pnl}", "warning")
                add_sim_order(pos["strategy"], pos["tradingsymbol"], "SELL", old_qty, pos["last_price"], reason="Stoploss Hit")
        
        total_pnl += pos["pnl"]

    # Scan for NEW trades if strategies are active
    config = get_agent_config()
    active_strats = config.active_strategies.split(",") if config.active_strategies else []
    if active_strats:
        atm_ce = simulation_state.get("atm_ce")
        if not atm_ce: 
            return {"data": {"is_active": False, "message": "ATM CE missing"}}
        
        ref_history = get_instrument_history(kite, atm_ce["instrument_token"], sim_date)
        hist_1m = ref_history[:idx+1]
        
        if len(hist_1m) >= 5:
            hist_5m = aggregate_to_tf(hist_1m, 5)
            
            if len(hist_5m) >= 2:
                kite = get_kite_instance()
                
                for s_type in active_strats:
                    # Skip if strategy is in cooldown
                    if s_type in simulation_state["strategy_cooldown"]:
                        last_exit_time_str = simulation_state["strategy_cooldown"][s_type]
                        last_exit_time = datetime.strptime(last_exit_time_str, "%H:%M:%S").time()
                        curr_sim_time = datetime.strptime(simulation_state["time"], "%H:%M:%S").time()
                        diff_mins = (curr_sim_time.hour * 60 + curr_sim_time.minute) - (last_exit_time.hour * 60 + last_exit_time.minute)
                        if diff_mins < 5: continue

                    # Prevent multiple active positions for same strategy
                    if any(p.get("strategy") == s_type for p in simulation_state["positions"] if p["quantity"] > 0): 
                        continue
                    
                    nifty_price = current_candle["close"]
                    current_strike = round(nifty_price / 50) * 50
                    date_str = simulation_state["date"]
                    atm_ce = simulation_state.get("atm_ce")
                    atm_pe = simulation_state.get("atm_pe")
                    
                    if not atm_ce or not atm_pe: 
                        continue

                    first_candle_5m = hist_5m[0]
                    
                    res = await run_strategy_on_candles(
                        kite, s_type, hist_5m, first_candle_5m, nifty_price, 
                        current_strike, atm_ce, atm_pe, date_str, 
                        simulation_state.get("nifty_options", []), 
                        datetime.strptime(date_str, "%Y-%m-%d")
                    )
                    
                    if res:
                        # Only allow ONE signal per strategy per day for scheduled entries
                        if s_type in ["long_straddle", "long_strangle", "bull_call_spread", "bear_put_spread", "iron_condor"]:
                            simulation_state["strategy_cooldown"][s_type] = "23:59:59"
                        
                        entry_price = round(float(res["entry_price"]), 2)
                        option_to_trade = res["option_to_trade"]
                        
                        msg = f"SIGNAL DETECTED: {s_type} triggered @ {entry_price}. Reason: {res['reason']}"
                        if config.is_auto_trade_enabled:
                            add_live_log(msg, "success")
                            existing_pos = next((p for p in simulation_state["positions"] if p["tradingsymbol"] == option_to_trade["tradingsymbol"]), None)
                            
                            trade_qty = calculate_sim_qty(
                                entry_price, 
                                config.trading_capital, 
                                config.risk_per_trade_pct, 
                                config.reward_per_trade_pct
                            )
                            
                            if trade_qty == 0:
                                add_live_log(f"SKIP: Trade filtered (gap < 10% of premium)", "warning")
                                continue
                            
                            if existing_pos:
                                existing_pos["quantity"] += trade_qty
                                existing_pos["average_price"] = ((existing_pos["average_price"] * (existing_pos["quantity"] - trade_qty)) + (entry_price * trade_qty)) / existing_pos["quantity"]
                            else:
                                new_pos = {
                                    "strategy": s_type,
                                    "tradingsymbol": option_to_trade["tradingsymbol"],
                                    "quantity": trade_qty,
                                    "average_price": entry_price,
                                    "last_price": entry_price,
                                    "pnl": 0,
                                    "realized_pnl": 0,
                                    "exit_reason": None,
                                    "is_multi_leg": res.get("is_multi_leg", False),
                                    "legs": res.get("legs", [])
                                }
                                simulation_state["positions"].append(new_pos)
                            
                            add_sim_order(s_type, option_to_trade["tradingsymbol"], "BUY", trade_qty, entry_price, reason=res["reason"])
                            simulation_state["executed_strategies"].add(s_type)
                        else:
                            add_live_log(f"ALERT (Auto-Trade OFF): {msg}", "info")

    return {
        "data": {
            "is_active": True,
            "current_index": simulation_state["current_index"],
            "total_candles": len(candles),
            "time": simulation_state.get("time", ""),
            "nifty_price": simulation_state.get("nifty_price", 0),
            "positions": simulation_state["positions"],
            "total_pnl": round(total_pnl, 2),
            "orders": simulation_state["orders"][:20]  # Last 20 orders
        }
    }


@router.post("/stop")
def stop_simulation():
    simulation_state["is_active"] = False
    return {"status": "success"}

