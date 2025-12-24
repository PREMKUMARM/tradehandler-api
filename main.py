from typing import Union
import os
from pathlib import Path

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import json
from pprint import pprint
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

def calculate_trend_and_suggestions(kite: KiteConnect, instrument_token: int, current_price: float):
    """Calculate trend based on 5min, 15min, 30min candles and provide buy/sell suggestions"""
    try:
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        trend_scores = {
            "5min": 0,
            "15min": 0,
            "30min": 0
        }
        
        intervals = {
            "5min": "5minute",
            "15min": "15minute",
            "30min": "30minute"
        }
        
        all_closes = []
        
        # Get candles for each timeframe
        for timeframe, kite_interval in intervals.items():
            try:
                historical_data = kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=yesterday,
                    to_date=today,
                    interval=kite_interval
                )
                
                if historical_data and len(historical_data) > 0:
                    # Get last 5 candles
                    recent_candles = historical_data[-5:] if len(historical_data) >= 5 else historical_data
                    closes = [candle.get("close", 0) for candle in recent_candles if candle.get("close", 0) > 0]
                    
                    if len(closes) >= 2:
                        all_closes.extend(closes)
                        
                        # Calculate trend: compare last close with previous closes
                        last_close = closes[-1]
                        prev_close = closes[-2] if len(closes) > 1 else closes[0]
                        
                        # Simple trend: if last close > previous, bullish (+1), else bearish (-1)
                        if last_close > prev_close:
                            trend_scores[timeframe] = 1
                        elif last_close < prev_close:
                            trend_scores[timeframe] = -1
                        else:
                            trend_scores[timeframe] = 0
            except Exception as e:
                print(f"Error fetching {timeframe} candles: {e}")
                continue
        
        # Calculate overall trend
        total_score = trend_scores["5min"] + trend_scores["15min"] + trend_scores["30min"]
        
        # Build reason explanation
        reason_parts = []
        
        # Add timeframe analysis to reason
        timeframe_analysis = []
        for tf, score in trend_scores.items():
            if score > 0:
                timeframe_analysis.append(f"{tf} bullish")
            elif score < 0:
                timeframe_analysis.append(f"{tf} bearish")
            else:
                timeframe_analysis.append(f"{tf} neutral")
        
        if timeframe_analysis:
            reason_parts.append(f"Timeframes: {', '.join(timeframe_analysis)}")
        
        # Determine trend
        if total_score >= 2:
            trend = "BULLISH"
            trend_strength = min(abs(total_score) / 3.0, 1.0)  # Normalize to 0-1
            reason_parts.append(f"Strong bullish momentum (score: {total_score}/3)")
        elif total_score <= -2:
            trend = "BEARISH"
            trend_strength = min(abs(total_score) / 3.0, 1.0)
            reason_parts.append(f"Strong bearish momentum (score: {total_score}/3)")
        else:
            trend = "NEUTRAL"
            trend_strength = 0.5
            reason_parts.append(f"Mixed signals (score: {total_score}/3)")
        
        # Calculate buy/sell prices based on trend
        if current_price > 0:
            if trend == "BULLISH":
                # For bullish trend: buy at current or slightly below, sell at target (2-3% above)
                buy_price = round(current_price * 0.998, 2)  # 0.2% below current
                sell_price = round(current_price * 1.025, 2)  # 2.5% above current
                reason_parts.append(f"Buy at ₹{buy_price} (0.2% below) for bullish entry, target ₹{sell_price} (2.5% profit)")
            elif trend == "BEARISH":
                # For bearish trend: buy at support (slightly below), sell at current or slightly above
                buy_price = round(current_price * 0.995, 2)  # 0.5% below current
                sell_price = round(current_price * 1.01, 2)  # 1% above current (quick exit)
                reason_parts.append(f"Buy at ₹{buy_price} (0.5% below support), quick exit at ₹{sell_price} (1% profit)")
            else:  # NEUTRAL
                # For neutral: buy at current, sell at small profit
                buy_price = round(current_price, 2)
                sell_price = round(current_price * 1.015, 2)  # 1.5% above current
                reason_parts.append(f"Buy at ₹{buy_price} (current price), conservative target ₹{sell_price} (1.5% profit)")
        else:
            buy_price = 0
            sell_price = 0
            reason_parts.append("Price data unavailable")
        
        reason = " | ".join(reason_parts)
        
        return {
            "trend": trend,
            "trend_strength": round(trend_strength, 2),
            "buy_price": buy_price,
            "sell_price": sell_price,
            "reason": reason,
            "trend_scores": trend_scores
        }
    except Exception as e:
        print(f"Error in calculate_trend_and_suggestions: {e}")
        return {
            "trend": "NEUTRAL",
            "trend_strength": 0,
            "buy_price": current_price if current_price > 0 else 0,
            "sell_price": current_price if current_price > 0 else 0,
            "reason": "Unable to calculate trend - insufficient candle data"
        }

# Initialize FastAPI app
app = FastAPI()

# Kite Connect credentials - Update these with your actual API key and secret
# IMPORTANT: The redirect_uri must EXACTLY match what's configured in your Kite Connect app settings
api_key = os.getenv('KITE_API_KEY', 'gle4opgggiing1ol')
api_secret = os.getenv('KITE_API_SECRET', 'vmrsky50fsozxonx2v5wwjwdmm6jcjtk')
# For local development, use: http://localhost:4200/auth-token
# For production, use: https://www.tradehandler.com/auth-token
redirect_uri = os.getenv('KITE_REDIRECT_URI', 'http://localhost:4200/auth-token')

origins = [
    "https://www.tradehandler.com",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to get access token from file
def get_access_token():
    """Read access token from config file"""
    config_path = Path("config/access_token.txt")
    if config_path.exists():
        with open(config_path, "r") as f:
            token = f.read().strip()
            return token if token else None
    return None

# Helper function to get KiteConnect instance
def get_kite_instance():
    """Get authenticated KiteConnect instance"""
    access_token = get_access_token()
    if not access_token:
        raise HTTPException(status_code=401, detail="Access token not found. Please authenticate first.")
    
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite

@app.get("/live-positions")
def get_live_positions():
    """Fetch current open positions from Zerodha Kite"""
    try:
        kite = get_kite_instance()
        positions = kite.positions()
        
        # Calculate live MTM and totals
        net_positions = positions.get("net", [])
        total_pnl = 0
        active_count = 0
        
        for pos in net_positions:
            total_pnl += pos.get("pnl", 0)
            if pos.get("quantity", 0) != 0:
                active_count += 1
                
        return {
            "data": {
                "positions": net_positions,
                "total_pnl": round(total_pnl, 2),
                "active_count": active_count
            }
        }
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching positions: {str(e)}")

@app.post("/place-strategy-order")
async def place_strategy_order(req: Request):
    """Place a live order based on a strategy signal"""
    try:
        kite = get_kite_instance()
        payload = await req.json()
        
        strategy_type = payload.get("strategy_type")
        symbol = payload.get("tradingsymbol")
        exchange = payload.get("exchange", "NFO")
        transaction_type = payload.get("transaction_type", "BUY")
        quantity = payload.get("quantity", 75)
        order_type = payload.get("order_type", "MARKET")
        product = payload.get("product", "MIS") # MIS for Intraday
        
        is_multi_leg = payload.get("multi_leg", False)
        legs = payload.get("legs", [])
        
        order_ids = []
        
        if is_multi_leg and legs:
            # Place multi-leg orders (e.g., Bull Call Spread)
            for leg in legs:
                order_id = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=exchange,
                    tradingsymbol=leg.get("tradingsymbol"),
                    transaction_type=leg.get("action"), # BUY or SELL
                    quantity=quantity,
                    product=product,
                    order_type=order_type
                )
                order_ids.append(order_id)
        else:
            # Place single leg order
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=order_type
            )
            order_ids.append(order_id)
            
        return {
            "status": "success",
            "message": f"Successfully placed {len(order_ids)} order(s)",
            "order_ids": order_ids
        }
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing order: {str(e)}")

@app.post("/exit-all-positions")
def exit_all_positions():
    """Kill Switch: Exit all open positions immediately at MARKET price"""
    try:
        # Check if in simulation mode
        if simulation_state["is_active"]:
            exit_count = 0
            for pos in simulation_state["positions"]:
                if pos["quantity"] != 0:
                    old_qty = pos["quantity"]
                    pos["quantity"] = 0
                    pos["exit_reason"] = "Manual Exit"
                    add_sim_order(pos["strategy"], pos["tradingsymbol"], "SELL", old_qty, pos["last_price"], reason="Manual Exit (Emergency)")
                    exit_count += 1
            return {"status": "success", "message": f"Simulation: Exited {exit_count} positions"}

        kite = get_kite_instance()
        positions = kite.positions().get("net", [])
        exit_orders = []
        
        for pos in positions:
            qty = pos.get("quantity", 0)
            if qty != 0:
                # Opposite transaction to close
                trans_type = kite.TRANSACTION_TYPE_SELL if qty > 0 else kite.TRANSACTION_TYPE_BUY
                abs_qty = abs(qty)
                
                order_id = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=pos.get("exchange"),
                    tradingsymbol=pos.get("tradingsymbol"),
                    transaction_type=trans_type,
                    quantity=abs_qty,
                    product=pos.get("product"),
                    order_type=kite.ORDER_TYPE_MARKET
                )
                exit_orders.append({
                    "symbol": pos.get("tradingsymbol"),
                    "order_id": order_id
                })
                
        return {
            "status": "success",
            "message": f"Exited {len(exit_orders)} positions",
            "exits": exit_orders
        }
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exiting positions: {str(e)}")

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Global state for Live Trading
live_strategy_config = {
    "active_strategies": set(), # e.g., {"915_candle_break", "bull_call_spread"}
    "is_auto_trade_enabled": False,
    "last_run_time": None,
    "fund": 200000,
    "risk_pct": 1.0,
    "reward_pct": 3.0,
    "daily_loss_limit": 5000,
    "current_daily_pnl": 0
}

@app.get("/live-strategy-config")
def get_live_config():
    return {"data": {
        **live_strategy_config,
        "active_strategies": list(live_strategy_config["active_strategies"])
    }}

@app.post("/toggle-live-strategy")
async def toggle_live_strategy(req: Request):
    payload = await req.json()
    strategy = payload.get("strategy")
    action = payload.get("action") # "enable" or "disable"
    
    # Also handle fund/risk/reward updates if present
    if "fund" in payload: live_strategy_config["fund"] = float(payload["fund"])
    if "risk_pct" in payload: live_strategy_config["risk_pct"] = float(payload["risk_pct"])
    if "reward_pct" in payload: live_strategy_config["reward_pct"] = float(payload["reward_pct"])
    
    if action == "enable":
        live_strategy_config["active_strategies"].add(strategy)
    else:
        live_strategy_config["active_strategies"].discard(strategy)
        
    return {"status": "success", "active_strategies": list(live_strategy_config["active_strategies"])}

@app.post("/toggle-auto-trade")
async def toggle_auto_trade(req: Request):
    payload = await req.json()
    enabled = payload.get("enabled", False)
    live_strategy_config["is_auto_trade_enabled"] = enabled
    return {"status": "success", "is_auto_trade_enabled": enabled}

def aggregate_to_tf(candles_1m, tf_min):
    """
    Aggregate 1-minute candles into any timeframe (5, 15, 30, 60 min)
    
    OHLC Aggregation Rules (TradingView standard):
    - Open: First value in period
    - High: Maximum value in period
    - Low: Minimum value in period
    - Close: Last value in period
    - Volume: Sum of volumes in period
    
    Note: For better performance with large datasets, consider using pandas resample:
        df = pd.DataFrame(candles_1m)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        resampled = df.resample(f'{tf_min}T').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        })
    """
    if not candles_1m: return []
    
    aggregated = []
    for i in range(0, len(candles_1m), tf_min):
        chunk = candles_1m[i:i+tf_min]
        if not chunk: continue
        
        aggregated.append({
            'date': chunk[0]['date'],
            'open': chunk[0]['open'],
            'high': max(c['high'] for c in chunk),
            'low': min(c['low'] for c in chunk),
            'close': chunk[-1]['close'],
            'volume': sum(c.get('volume', 0) for c in chunk)
        })
    return aggregated

def analyze_trend(candles):
    """Simple trend analysis for a set of candles"""
    if len(candles) < 5: return "NEUTRAL"
    
    closes = [c['close'] for c in candles]
    # Simple moving average (5)
    sma_5 = sum(closes[-5:]) / 5
    current_price = closes[-1]
    
    # Calculate RSI for trend strength
    rsi = 50
    if len(closes) >= 14:
        deltas = np.diff(closes)
        seed = deltas[:14]
        up = seed[seed >= 0].sum() / 14
        down = -seed[seed < 0].sum() / 14
        if down == 0:
            rsi = 100
        else:
            rs = up / down
            rsi = 100 - (100 / (1 + rs))
            
    if current_price > sma_5 * 1.002 and rsi > 55: return "BULLISH"
    if current_price < sma_5 * 0.998 and rsi < 45: return "BEARISH"
    return "SIDEWAYS"

async def live_market_scanner():
    """Background task to scan market and execute strategies"""
    add_live_log("Live Market Scanner initializing...", "info")
    
    # Heartbeat tracker to avoid log spam
    last_heartbeat_min = -1
    last_analysis_min = -1
    
    while True:
        try:
            # Only scan between 9:15 AM and 3:30 PM IST
            now = datetime.now()
            
            # Indian Market Hours (9:15 AM to 3:30 PM)
            if now.hour >= 9 and (now.hour < 15 or (now.hour == 15 and now.minute <= 30)):
                if live_strategy_config["active_strategies"]:
                    # 1. Fetch live quotes/candles
                    kite = get_kite_instance()
                    
                    # Get Nifty index price
                    quote = kite.quote(["NSE:NIFTY 50"])
                    nifty_data = quote.get("NSE:NIFTY 50", {})
                    nifty_price = nifty_data.get("last_price", 0)
                    
                    if nifty_price > 0:
                        # Multi-Timeframe Analysis Feed
                        # Log detailed analysis every 3 minutes
                        if now.minute % 3 == 0 and now.minute != last_analysis_min:
                            from_date = now - timedelta(days=5) # Get 5 days for enough 1h/Day data
                            nifty_candles_1m = kite.historical_data(256265, from_date, now, "minute")
                            
                            if nifty_candles_1m:
                                analysis_parts = []
                                for tf_name, tf_min in [("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60)]:
                                    tf_candles = aggregate_to_tf(nifty_candles_1m, tf_min)
                                    trend = analyze_trend(tf_candles)
                                    # Add small indicator like 🟢 🔴 ⚪
                                    icon = "🟢" if trend == "BULLISH" else "🔴" if trend == "BEARISH" else "⚪"
                                    analysis_parts.append(f"{tf_name}: {icon} {trend}")
                                
                                # Fetch Day candle separately from Kite for accuracy
                                day_hist = kite.historical_data(256265, now - timedelta(days=30), now, "day")
                                day_trend = analyze_trend(day_hist)
                                day_icon = "🟢" if day_trend == "BULLISH" else "🔴" if day_trend == "BEARISH" else "⚪"
                                analysis_parts.append(f"DAY: {day_icon} {day_trend}")
                                
                                add_live_log(f"ANALYSIS: " + " | ".join(analysis_parts), "info")
                                last_analysis_min = now.minute

                        # Heartbeat tracker for Nifty price and active strategies
                        if now.minute % 5 == 0 and now.minute != last_heartbeat_min:
                            active_strats = ", ".join(live_strategy_config["active_strategies"])
                            add_live_log(f"LIVE SCANNING: Nifty @ {nifty_price} | Active: {active_strats}", "info")
                            last_heartbeat_min = now.minute
                            
                        current_strike = round(nifty_price / 50) * 50
                        
                        # Get NFO instruments for expiry matching
                        all_instruments = kite.instruments("NFO")
                        nifty_options = [inst for inst in all_instruments if inst.get("name") == "NIFTY"]
                        
                        # Find nearest expiry
                        today = now.date()
                        valid_options = [inst for inst in nifty_options if inst.get("expiry") and inst.get("expiry") >= today]
                        valid_options.sort(key=lambda x: x["expiry"])
                        
                        if valid_options:
                            nearest_expiry = valid_options[0]["expiry"]
                            atm_options = [inst for inst in valid_options if inst["expiry"] == nearest_expiry and inst["strike"] == current_strike]
                            
                            atm_ce = next((inst for inst in atm_options if inst["instrument_type"] == "CE"), None)
                            atm_pe = next((inst for inst in atm_options if inst["instrument_type"] == "PE"), None)
                            
                            if atm_ce and atm_pe:
                                # Fetch recent 1-minute candles for Nifty 50 to run strategies (mostly 5m based)
                                from_date_strat = now - timedelta(hours=6)
                                nifty_candles_strat = kite.historical_data(256265, from_date_strat, now, "minute")
                                
                                if len(nifty_candles_strat) >= 20:
                                    trading_candles_5m = aggregate_to_tf(nifty_candles_strat, 5)
                                    first_candle = trading_candles_5m[0] if trading_candles_5m else None
                                    
                                    # Run each active strategy
                                    for strategy_id in live_strategy_config["active_strategies"]:
                                        res = await run_strategy_on_candles(
                                            kite, strategy_id, trading_candles_5m, first_candle, 
                                            nifty_price, current_strike, atm_ce, atm_pe, 
                                            now.strftime("%Y-%m-%d"), nifty_options, today
                                        )
                                        
                                        if res:
                                            # Signal detected in LIVE market
                                            message = f"LIVE SIGNAL: {strategy_id} triggered. Reason: {res['reason']}"
                                            
                                            if live_strategy_config["is_auto_trade_enabled"]:
                                                add_live_log(f"EXECUTING LIVE ORDER: {message}", "signal")
                                            else:
                                                add_live_log(f"ALERT (Auto-Trade OFF): {message}", "info")
                else:
                    # Log once that no strategies are active
                    if now.minute % 30 == 0 and now.minute != last_heartbeat_min:
                        add_live_log("LIVE MONITORING: Waiting for strategies to be selected...", "info")
                        last_heartbeat_min = now.minute
            else:
                # Outside market hours
                if now.minute % 60 == 0 and now.minute != last_heartbeat_min:
                    add_live_log(f"LIVE FEED: Market is currently CLOSED ({now.strftime('%H:%M')}). Scanning resumes at 09:15 AM.", "info")
                    last_heartbeat_min = now.minute
            
            # Run every 10 seconds (more responsive scanner)
            await asyncio.sleep(10)
        except Exception as e:
            print(f"Scanner error: {e}")
            await asyncio.sleep(30) # Wait longer on error

@app.on_event("startup")
async def startup_event():
    # Start the background scanner
    asyncio.create_task(live_market_scanner())

# Global state for Simulation
simulation_state = {
    "is_active": False,
    "date": None,
    "current_index": 0,
    "speed": 1,
    "candles": [],
    "instrument_history": {}, # Cache for all traded instrument historical data
    "nifty_options": [],
    "atm_ce": None,
    "atm_pe": None,
    "positions": [],
    "orders": [], # Track entry and exit orders
    "strategy_cooldown": {}, # Track last exit time per strategy to prevent infinite loops
    "executed_strategies": set(),
    "nifty_price": 0,
    "last_update": None
}

# Global state for Live Monitoring Logs
live_logs = []

def add_live_log(message, log_type="info"):
    """Helper to add a log message with timestamp for UI monitoring"""
    global live_logs
    
    # Get current time (either real or simulated)
    if simulation_state["is_active"] and simulation_state["candles"]:
        idx = min(simulation_state["current_index"], len(simulation_state["candles"]) - 1)
        timestamp = simulation_state["candles"][idx]["date"].strftime("%H:%M:%S")
        timestamp = f"[SIM {timestamp}]"
    else:
        timestamp = datetime.now().strftime("%H:%M:%S")
    
    live_logs.insert(0, {
        "timestamp": timestamp,
        "message": message,
        "type": log_type
    })
    
    # Keep only last 100 logs
    if len(live_logs) > 100:
        live_logs = live_logs[:100]
    
    # Also print to terminal for debugging
    print(f"{timestamp} {log_type.upper()}: {message}")

@app.get("/live-logs")
async def get_live_logs():
    """Endpoint for UI to fetch latest monitoring logs"""
    return {"data": live_logs}

@app.post("/simulation/speed")
async def set_simulation_speed(req: Request):
    payload = await req.json()
    speed = payload.get("speed", 1)
    simulation_state["speed"] = max(1, speed)
    return {"status": "success", "speed": simulation_state["speed"]}

@app.post("/simulation/start")
async def start_simulation(req: Request):
    """Start a live-market simulation using historical data"""
    try:
        global live_logs
        live_logs = [] # Clear previous logs
        add_live_log(f"Initializing simulation...", "info")
        
        kite = get_kite_instance()
        payload = await req.json()
        sim_date_str = payload.get("date", (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
        sim_date = datetime.strptime(sim_date_str, "%Y-%m-%d").date()
        
        add_live_log(f"Loading historical data for {sim_date_str}...", "info")
        
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

async def run_strategy_on_candles(kite, strategy_type, trading_candles, first_candle, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date):
    """Unified function to run any strategy on a set of candles"""
    if strategy_type == "915_candle_break":
        return strategy_915_candle_break(kite, trading_candles, first_candle, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "mean_reversion":
        return strategy_mean_reversion_bollinger(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "momentum_breakout":
        return strategy_momentum_breakout(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "support_resistance":
        return strategy_support_resistance_breakout(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "long_straddle":
        return strategy_long_straddle(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date)
    elif strategy_type == "long_strangle":
        return strategy_long_strangle(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date)
    elif strategy_type == "bull_call_spread":
        return strategy_bull_call_spread(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date)
    elif strategy_type == "bear_put_spread":
        return strategy_bear_put_spread(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date)
    elif strategy_type == "iron_condor":
        return strategy_iron_condor(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date)
    elif strategy_type == "macd_crossover":
        return strategy_macd_crossover(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "rsi_reversal":
        return strategy_rsi_reversal(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    elif strategy_type == "ema_cross":
        return strategy_ema_cross(kite, trading_candles, nifty_price, current_strike, atm_ce, atm_pe, date_str)
    return None

def get_instrument_history(kite, token, sim_date):
    """Helper to fetch and cache full day history for an instrument during simulation"""
    cache_key = f"{token}_{sim_date}"
    if cache_key in simulation_state["instrument_history"]:
        return simulation_state["instrument_history"][cache_key]
    
    try:
        data = kite.historical_data(token, sim_date, sim_date + timedelta(days=1), "minute")
        # Filter for market hours to match Nifty candles index
        filtered_data = [c for c in data if time(9, 15) <= c["date"].time() <= time(15, 30)]
        simulation_state["instrument_history"][cache_key] = filtered_data
        return filtered_data
    except Exception as e:
        print(f"Error caching instrument {token}: {e}")
        return []

def add_sim_order(strategy, symbol, action, quantity, price, order_type="MARKET", status="COMPLETE", reason=""):
    """Helper to track mock orders in simulation"""
    order = {
        "order_id": f"SIM{datetime.now().strftime('%H%M%S%f')[:-3]}",
        "order_timestamp": simulation_state["time"] if "time" in simulation_state else datetime.now().strftime("%H:%M:%S"),
        "strategy": strategy,
        "tradingsymbol": symbol,
        "transaction_type": action, # BUY/SELL
        "quantity": quantity,
        "average_price": price,
        "order_type": order_type,
        "status": status,
        "reason": reason
    }
    simulation_state["orders"].insert(0, order) # Keep newest at top
    return order

def calculate_sim_qty(entry_price, fund, risk_pct, reward_pct):
    """Calculates quantity based on User's Risk Management formula: Qty = Risk / (Target - SL)"""
    # Professional Filter: Ignore trades if total gap (Risk + Reward) is less than 10% of premium
    # This prevents entering trades where the targets are too small to cover brokerage/slippage
    # and results in unrealistically high quantities.
    total_gap_pct = reward_pct + risk_pct
    if total_gap_pct < 10:
        return 0 # Signal to skip this trade
        
    risk_amount = fund * (risk_pct / 100)
    
    # price_range = Target - SL = (Reward% + Risk%) of Entry
    price_range = abs(entry_price) * total_gap_pct / 100
    
    if price_range <= 0.1: price_range = 0.1 # Tick size safety
    
    target_qty = risk_amount / price_range
    
    # Round to nearest lot of 75, minimum 75
    lots = round(target_qty / 75)
    if lots < 1: lots = 1
    
    return lots * 75

@app.get("/simulation/chart-data")
async def get_simulation_chart_data(timeframe: str = "1m", indicators: str = ""):
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

@app.get("/simulation/state")
async def get_simulation_state():
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

@app.post("/simulation/stop")
def stop_simulation():
    simulation_state["is_active"] = False
    return {"status": "success"}

@app.get("/")
def read_root():
    return {"Hello": "World", "broker": "Zerodha Kite Connect"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/access-token")
def get_access_token_endpoint():
    """Get stored access token"""
    token = get_access_token()
    if token:
        return {"access_token": token}
    return {"access_token": None, "message": "No access token found"}

@app.get("/auth")
def get_login_url():
    """Get Kite Connect login URL"""
    try:
        # Validate API key is set
        if api_key == 'your_api_key_here' or not api_key:
            raise HTTPException(
                status_code=500, 
                detail="KITE_API_KEY is not configured. Please set it in environment variables or main.py"
            )
        
        kite = KiteConnect(api_key=api_key)
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

@app.post("/set-token")
async def set_token(req: Request):
    """Store access token after authentication"""
    try:
        data = await req.json()
        request_token = data.get('request_token')
        access_token = data.get('access-token')
        
        print(f"Received request_token: {request_token[:20] if request_token else None}...")
        print(f"API Key configured: {api_key[:10] if api_key and len(api_key) > 10 else 'NOT SET'}...")
        print(f"Redirect URI configured: {redirect_uri}")
        
        if not request_token and not access_token:
            raise HTTPException(status_code=400, detail="Either request_token or access-token required")
        
        # If request_token is provided, generate access_token
        if request_token:
            # Validate API key and secret are set
            if api_key == 'your_api_key_here' or not api_key:
                raise HTTPException(
                    status_code=500, 
                    detail="KITE_API_KEY is not configured. Please set it in environment variables or main.py"
                )
            if api_secret == 'your_api_secret_here' or not api_secret:
                raise HTTPException(
                    status_code=500, 
                    detail="KITE_API_SECRET is not configured. Please set it in environment variables or main.py"
                )
            
            kite = KiteConnect(api_key=api_key)
            try:
                print(f"Attempting to generate session with request_token...")
                data_response = kite.generate_session(request_token, api_secret=api_secret)
                access_token = data_response.get('access_token')
                print(f"Access token generated successfully: {access_token[:20] if access_token else None}...")
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
        
        if not access_token:
            raise HTTPException(status_code=400, detail="Failed to obtain access token")
        
        # Store access token
        config_path = Path("config")
        config_path.mkdir(exist_ok=True)
        
        with open("config/access_token.txt", "w") as f:
            f.write(access_token)
        
        print(f"Access token stored successfully in config/access_token.txt")
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

@app.get("/getBalance")
def get_balance():
    """Get user margin/funds"""
    try:
        kite = get_kite_instance()
        margins = kite.margins()
        
        # Kite Connect margins() returns:
        # {
        #   'equity': {
        #     'enabled': True,
        #     'net': float,
        #     'available': float,  # Available margin for trading
        #     'utilised': float,   # Utilised margin
        #     'adhoc_margin': float,
        #     'cash': float,
        #     'collateral': float,
        #     'intraday_payin': float,
        #     'live_balance': float,
        #     'opening_balance': float
        #   },
        #   'commodity': {...}
        # }
        
        equity_data = margins.get('equity', {})
        
        print(f"Raw Kite Connect margins response: {margins}")
        print(f"Equity data: {equity_data}")
        
        # Kite Connect 'available' can be a number or dict with 'cash' and 'intraday_payin'
        # Kite Connect 'utilised' can be a number or dict with 'debits', 'exposure', etc.
        available_value = equity_data.get('available', 0)
        utilised_value = equity_data.get('utilised', 0)
        
        # Extract numeric values if they're dictionaries
        if isinstance(available_value, dict):
            # If available is a dict, use 'cash' (this is the available margin)
            # Based on user's requirement: _available_margin should be 120608.6 (which is cash)
            available_margin = available_value.get('cash', 0)
            if available_margin == 0:
                available_margin = equity_data.get('opening_balance', 0) or equity_data.get('live_balance', 0)
        else:
            # Use available, or fallback to cash/opening_balance
            available_margin = available_value if available_value else equity_data.get('cash', 0) or equity_data.get('opening_balance', 0)
        
        if isinstance(utilised_value, dict):
            # If utilised is a dict, use 'debits' (this is the utilised margin)
            # Based on user's requirement: _utilised_margin should be 15258.75 (which is debits)
            utilised_margin = utilised_value.get('debits', 0)
        else:
            utilised_margin = utilised_value
        
        # Total margin is the net value (live_balance or net)
        # Based on user's requirement: _total_margin should be 64741.85 (which is net/live_balance)
        total_margin = equity_data.get('net', 0) or equity_data.get('live_balance', 0)
        
        print(f"Calculated available_margin: {available_margin}")
        print(f"Calculated utilised_margin: {utilised_margin}")
        print(f"Calculated total_margin: {total_margin}")
        
        # Transform to match frontend expected format (Upstox-style)
        # Frontend expects: res.data.equity._available_margin (as a NUMBER, not object)
        # CRITICAL: _available_margin must be INSIDE equity object, and must be a number
        transformed_margins = {
            "equity": {
                # Upstox-style fields (what frontend expects) - MUST be numbers, INSIDE equity
                "_available_margin": float(available_margin) if available_margin else 0.0,
                "_utilised_margin": float(utilised_margin) if utilised_margin else 0.0,
                "_total_margin": float(total_margin) if total_margin else 0.0,
                # Keep original Kite Connect fields for reference
                **equity_data
            },
            "commodity": margins.get('commodity', {})
        }
        
        print(f"Transformed margins structure: {transformed_margins}")
        
        return {"data": transformed_margins}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting balance: {str(e)}")

@app.get("/getPositions")
def get_positions():
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

@app.get("/getOrders")
def get_orders():
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

@app.get("/getCandle/{instrument_token}/{interval}/{fromDate}/{toDate}")
def get_candle(instrument_token: str, interval: str, fromDate: str, toDate: str):
    """Get historical candle data"""
    try:
        kite = get_kite_instance()
        
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
        
        # Parse dates
        from_date = datetime.strptime(fromDate, "%Y-%m-%d")
        to_date = datetime.strptime(toDate, "%Y-%m-%d")
        
        # Get historical data
        historical_data = kite.historical_data(
            instrument_token=int(instrument_token),
            from_date=from_date,
            to_date=to_date,
            interval=kite_interval
        )
        
        # Convert to DataFrame for consistency
        if historical_data:
            df = pd.DataFrame(historical_data)
            # Rename columns to match expected format
            df = df.rename(columns={
                'date': 'timestamp',
                'oi': 'openinterest'
            })
            # Ensure timestamp is in the right format
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
            
            return Response(df.to_json(orient='records'), media_type="application/json")
        else:
            return Response(json.dumps([]), media_type="application/json")
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting candles: {str(e)}")

@app.post("/placeOrder")
async def place_order(req: Request):
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

@app.post("/modifyOrder")
async def modify_order(req: Request):
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

@app.post("/cancelOrder")
async def cancel_order(req: Request):
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

@app.post("/sellOrder")
async def sell_order(req: Request):
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

@app.get("/ws-portfolio")
def get_portfolio_ws():
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

@app.get("/ws-orders")
def get_orders_ws():
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

@app.get("/instruments")
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

@app.get("/quote/{instrument_key}")
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

@app.get("/nifty50-options")
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

@app.get("/ws-nifty50-options")
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

def calculate_bollinger_bands(closes, period=20, num_std=2):
    """Calculate Bollinger Bands for mean reversion strategy"""
    if len(closes) < period:
        return None, None, None
    
    df = pd.DataFrame({'close': closes})
    df['sma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper_band'] = df['sma'] + (df['std'] * num_std)
    df['lower_band'] = df['sma'] - (df['std'] * num_std)
    
    return df['upper_band'].iloc[-1], df['sma'].iloc[-1], df['lower_band'].iloc[-1]

def calculate_bollinger_bands_full(closes, period=20, num_std=2):
    """Calculate full Bollinger Bands array for chart display - matches TradingView/Zerodha standard"""
    if len(closes) < period:
        return [], [], []
    
    df = pd.DataFrame({'close': closes})
    # Use Simple Moving Average (SMA) - standard for Bollinger Bands
    df['sma'] = df['close'].rolling(window=period, min_periods=1).mean()
    # Use Population Standard Deviation (ddof=0) - matches TradingView default
    # Note: pandas std() uses ddof=1 by default (sample std), we need ddof=0 (population std)
    df['std'] = df['close'].rolling(window=period, min_periods=1).std(ddof=0)
    df['upper_band'] = df['sma'] + (df['std'] * num_std)
    df['lower_band'] = df['sma'] - (df['std'] * num_std)
    
    # Return arrays - keep NaN for first period-1 values (no forward fill for accuracy)
    # Only fill backward for the very first value if needed
    upper = df['upper_band'].fillna(method='bfill', limit=1).tolist()
    middle = df['sma'].fillna(method='bfill', limit=1).tolist()
    lower = df['lower_band'].fillna(method='bfill', limit=1).tolist()
    
    return upper, middle, lower

def calculate_rsi(closes, period=14):
    """Calculate RSI (Relative Strength Index) using Wilder's Smoothing Method - matches TradingView/Zerodha"""
    if len(closes) < period + 1:
        return [np.nan] * len(closes)  # Return NaN if not enough data
    
    # Convert to numpy array for faster computation
    closes_arr = np.array(closes, dtype=float)
    deltas = np.diff(closes_arr)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Initialize arrays for average gain and loss (same length as deltas)
    avg_gains = np.full(len(gains), np.nan, dtype=float)
    avg_losses = np.full(len(losses), np.nan, dtype=float)
    
    # First average: Simple average of first 'period' delta values
    # This corresponds to the RSI value at index 'period' in the closes array
    if len(gains) >= period:
        avg_gains[period - 1] = np.mean(gains[:period])
        avg_losses[period - 1] = np.mean(losses[:period])
        
        # Apply Wilder's smoothing for remaining values
        # Formula: avg = (prev_avg * (period - 1) + current) / period
        for i in range(period, len(gains)):
            avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period
    
    # Calculate RS and RSI
    # Handle division by zero
    rs = np.divide(avg_gains, avg_losses, out=np.full_like(avg_gains, np.nan), where=(avg_losses != 0))
    rsi_deltas = 100 - (100 / (1 + rs))
    
    # RSI array should match closes length
    # First 'period' values are NaN (not enough data)
    # RSI at index i corresponds to close at index i
    rsi_list = [np.nan] * period  # First period values are NaN
    
    # Append calculated RSI values (starting from index period)
    for i in range(period - 1, len(rsi_deltas)):
        val = rsi_deltas[i]
        if np.isnan(val):
            rsi_list.append(np.nan)
        else:
            rsi_list.append(float(val))
    
    # Ensure length matches closes
    while len(rsi_list) < len(closes):
        rsi_list.append(np.nan)
    
    return rsi_list[:len(closes)]

def calculate_pivot_points(high, low, close):
    """Calculate Pivot Points (Traditional/Standard method) - matches TradingView/Zerodha"""
    # Traditional Pivot Point calculation (Standard method)
    # Pivot = (High + Low + Close) / 3
    pivot = (high + low + close) / 3
    
    # Resistance levels (Traditional method)
    # R1 = 2 * Pivot - Low
    # R2 = Pivot + (High - Low)
    # R3 = High + 2 * (Pivot - Low)
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    # Support levels (Traditional method)
    # S1 = 2 * Pivot - High
    # S2 = Pivot - (High - Low)
    # S3 = Low - 2 * (High - Pivot)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        "pivot": float(pivot),
        "r1": float(r1), 
        "r2": float(r2), 
        "r3": float(r3),
        "s1": float(s1), 
        "s2": float(s2), 
        "s3": float(s3)
    }

def calculate_support_resistance(candles, lookback=20):
    """Calculate support and resistance levels"""
    if len(candles) < lookback:
        return None, None
    
    highs = [c.get("high", 0) for c in candles[-lookback:]]
    lows = [c.get("low", 0) for c in candles[-lookback:]]
    
    resistance = max(highs)
    support = min(lows)
    
    return resistance, support

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

def find_option(nifty_options, strike, type, sim_date=None):
    """Helper to find an option instrument by strike and type, with expiry awareness"""
    matches = [o for o in nifty_options if o.get("strike") == strike and o.get("instrument_type") == type]
    if not matches: return None
    
    # If in simulation, find the nearest expiry >= simulation date
    if sim_date:
        if isinstance(sim_date, str):
            sim_date = datetime.strptime(sim_date, "%Y-%m-%d").date()
        elif isinstance(sim_date, datetime):
            sim_date = sim_date.date()
            
        expiries = sorted(list(set([o["expiry"] for o in matches if o.get("expiry")])))
        for exp in expiries:
            if exp >= sim_date:
                return next((o for o in matches if o["expiry"] == exp), matches[0])
                
    return matches[0]

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

@app.post("/backtest-nifty50-options")
async def backtest_nifty50_options(req: Request):
    """Backtest Nifty50 options strategy for given date range with multiple strategy options"""
    try:
        kite = get_kite_instance()
        payload = await req.json()
        
        start_date_str = payload.get("start_date")
        end_date_str = payload.get("end_date")
        strategy_type = payload.get("strategy_type", "915_candle_break")  # Default strategy
        fund = payload.get("fund", 200000)
        risk_pct = payload.get("risk", 1) / 100
        reward_pct = payload.get("reward", 3) / 100
        
        # Debug logging
        print(f"Backtest request received:")
        print(f"  Start date: {start_date_str}")
        print(f"  End date: {end_date_str}")
        print(f"  Strategy type: {strategy_type}")
        print(f"  Full payload: {payload}")
        
        if not start_date_str or not end_date_str:
            raise HTTPException(status_code=400, detail="start_date and end_date are required (format: YYYY-MM-DD)")
        
        # Parse dates
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
        # Get all NFO instruments
        all_instruments = kite.instruments("NFO")
        
        # Filter for Nifty50 options
        nifty_options = [
            inst for inst in all_instruments 
            if inst.get("name") == "NIFTY" and inst.get("instrument_type") in ["CE", "PE"]
        ]
        
        if not nifty_options:
            raise HTTPException(status_code=404, detail="Nifty50 options not found")
        
        # Generate list of trading dates (excluding weekends)
        trading_dates = []
        current_date = start_date
        while current_date <= end_date:
            # Check if weekday (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday to Friday
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        if not trading_dates:
            raise HTTPException(status_code=400, detail="No trading days found in date range")
        
        # Backtest results
        backtest_results = {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "strategy_type": strategy_type,
            "total_trading_days": len(trading_dates),
            "trades": [],
            "statistics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "average_profit": 0.0,
                "average_loss": 0.0,
                "max_profit": 0.0,
                "max_loss": 0.0
            }
        }
        
        # Get Nifty50 index instrument token for historical data
        nse_instruments = kite.instruments("NSE")
        nifty_index = next(
            (inst for inst in nse_instruments if inst.get("tradingsymbol") == "NIFTY 50"),
            None
        )
        
        if not nifty_index:
            raise HTTPException(status_code=404, detail="Nifty50 index not found")
        
        nifty_token = nifty_index.get("instrument_token")
        
        print(f"Backtesting from {start_date} to {end_date}")
        print(f"Total trading days: {len(trading_dates)}")
        print(f"Nifty50 token: {nifty_token}")
        print(f"Total Nifty options found: {len(nifty_options)}")
        
        # Process each trading day
        for trade_date in trading_dates:
            try:
                date_str = trade_date.strftime("%Y-%m-%d")
                
                # Get historical Nifty50 price for this date and previous day (for gap analysis)
                try:
                    # Get current day and previous day data
                    nifty_historical = kite.historical_data(
                        instrument_token=nifty_token,
                        from_date=trade_date - timedelta(days=2),
                        to_date=trade_date + timedelta(days=1),
                        interval="day"
                    )
                    
                    if not nifty_historical or len(nifty_historical) < 2:
                        # Try wider range
                        nifty_historical = kite.historical_data(
                            instrument_token=nifty_token,
                            from_date=trade_date - timedelta(days=7),
                            to_date=trade_date + timedelta(days=1),
                            interval="day"
                        )
                    
                    if not nifty_historical or len(nifty_historical) == 0:
                        print(f"No Nifty50 historical data available for {date_str}")
                        continue
                    
                    # Find current day and previous day candles
                    nifty_candle = None
                    prev_day_candle = None
                    
                    for candle in nifty_historical:
                        candle_date_str = candle.get("date", "")
                        if candle_date_str:
                            try:
                                candle_date = datetime.strptime(candle_date_str.split()[0], "%Y-%m-%d").date()
                                if candle_date == trade_date:
                                    nifty_candle = candle
                                elif candle_date == trade_date - timedelta(days=1):
                                    prev_day_candle = candle
                            except:
                                continue
                    
                    # If no exact match, use the last candle as current
                    if not nifty_candle and nifty_historical:
                        nifty_candle = nifty_historical[-1]
                    
                    # Get previous day candle (for gap analysis)
                    if not prev_day_candle and len(nifty_historical) >= 2:
                        prev_day_candle = nifty_historical[-2]
                    
                    if not nifty_candle:
                        print(f"Could not find Nifty50 candle for {date_str}")
                        continue
                    
                    nifty_price = nifty_candle.get("close", 0)
                    if not nifty_price or nifty_price == 0:
                        print(f"Invalid Nifty50 price for {date_str}: {nifty_price}")
                        continue
                    
                    # Gap Analysis - Skip trades on significant gap days (risk management)
                    gap_percent = 0
                    if prev_day_candle:
                        prev_close = prev_day_candle.get("close", 0)
                        if prev_close > 0:
                            gap_percent = ((nifty_price - prev_close) / prev_close) * 100
                            # Skip if gap > 1% (too volatile/unpredictable for options)
                            if abs(gap_percent) > 1.0:
                                print(f"{date_str}: Skipping trade due to large gap ({gap_percent:.2f}%)")
                                continue
                    
                    print(f"{date_str}: Nifty50 price = {nifty_price}, Gap = {gap_percent:.2f}%")
                    
                except Exception as e:
                    print(f"Error getting Nifty50 price for {date_str}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Calculate current strike
                current_strike = round(nifty_price / 50) * 50
                
                # Get options for this date (nearest expiry)
                # Filter options that would have been active on this date
                # Options expire on Thursdays, so we need expiry >= trade_date
                expiries_raw = [inst.get("expiry") for inst in nifty_options if inst.get("expiry")]
                
                # Convert expiries to date objects (they might be strings or date objects)
                # Kite Connect returns expiry as datetime.date objects
                expiries_dates = []
                seen_dates = set()  # Track unique dates to avoid duplicates
                
                for expiry in expiries_raw:
                    try:
                        # Check if already a date object FIRST (datetime.date has year, month, day attributes)
                        # This is the most common case from Kite Connect
                        if hasattr(expiry, 'year') and hasattr(expiry, 'month') and hasattr(expiry, 'day'):
                            expiry_date = expiry
                        elif isinstance(expiry, datetime):
                            expiry_date = expiry.date()
                        elif isinstance(expiry, str):
                            expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
                        else:
                            continue
                        
                        # Avoid duplicates
                        if expiry_date not in seen_dates:
                            seen_dates.add(expiry_date)
                            expiries_dates.append((expiry_date, expiry))  # Store both date and original
                    except Exception as e:
                        print(f"Error parsing expiry {expiry} (type: {type(expiry)}): {e}")
                        continue
                
                # Sort by date
                expiries_dates.sort(key=lambda x: x[0])
                expiries = [exp[1] for exp in expiries_dates]  # Original format
                expiries_date_objects = [exp[0] for exp in expiries_dates]  # Date objects
                
                if not expiries:
                    print(f"No expiries found for {date_str}")
                    continue
                
                # Find expiry that's >= trade_date (options expire on or after trade date)
                nearest_expiry = None
                nearest_expiry_date = None
                for i, expiry_date_obj in enumerate(expiries_date_objects):
                    if expiry_date_obj >= trade_date:
                        nearest_expiry = expiries[i]
                        nearest_expiry_date = expiry_date_obj
                        break
                
                if not nearest_expiry:
                    print(f"No valid expiry found for {date_str} (trade_date: {trade_date}, available expiries: {expiries_date_objects[:5]})")
                    continue
                
                print(f"{date_str}: Using expiry {nearest_expiry_date} ({nearest_expiry}), strike: {current_strike}")
                
                # Get ATM CE and PE options for this expiry
                # Compare expiry properly (handle both string and date formats)
                atm_ce = None
                atm_pe = None
                
                for inst in nifty_options:
                    inst_expiry = inst.get("expiry")
                    if not inst_expiry:
                        continue
                    
                    # Normalize expiry for comparison
                    try:
                        # Check if already a date object FIRST (most common case from Kite Connect)
                        if hasattr(inst_expiry, 'year') and hasattr(inst_expiry, 'month') and hasattr(inst_expiry, 'day'):
                            inst_expiry_date = inst_expiry
                        elif isinstance(inst_expiry, datetime):
                            inst_expiry_date = inst_expiry.date()
                        elif isinstance(inst_expiry, str):
                            inst_expiry_date = datetime.strptime(inst_expiry, "%Y-%m-%d").date()
                        else:
                            continue
                        
                        # Check if this instrument matches our criteria
                        if inst_expiry_date == nearest_expiry_date:
                            if inst.get("strike") == current_strike:
                                if inst.get("instrument_type") == "CE" and not atm_ce:
                                    atm_ce = inst
                                elif inst.get("instrument_type") == "PE" and not atm_pe:
                                    atm_pe = inst
                    except Exception as e:
                        continue
                
                if not atm_ce or not atm_pe:
                    print(f"No ATM options found for {date_str}, strike: {current_strike}, expiry: {nearest_expiry}")
                    continue
                
                # Get historical data for trend analysis and entry price
                # Indian Market Strategy: Use 9:15 AM candle + trend confirmation
                try:
                    # Get 5min candles for the trading day
                    option_historical = kite.historical_data(
                        instrument_token=atm_ce.get("instrument_token"),
                        from_date=trade_date,
                        to_date=trade_date + timedelta(days=1),
                        interval="5minute"
                    )
                    
                    if not option_historical or len(option_historical) < 2:
                        # Try to get from previous day
                        option_historical = kite.historical_data(
                            instrument_token=atm_ce.get("instrument_token"),
                            from_date=trade_date - timedelta(days=1),
                            to_date=trade_date + timedelta(days=1),
                            interval="5minute"
                        )
                    
                    if option_historical and len(option_historical) >= 2:
                        # Filter candles for trading hours (9:15 AM to 3:30 PM IST)
                        trading_candles = []
                        for candle in option_historical:
                            candle_date = candle.get("date", "")
                            if candle_date:
                                try:
                                    if isinstance(candle_date, str):
                                        if ' ' in candle_date:
                                            time_part = candle_date.split(' ')[1][:8] if len(candle_date.split(' ')) > 1 else ""
                                            if time_part and "09:15" <= time_part <= "15:30":
                                                trading_candles.append(candle)
                                    elif isinstance(candle_date, datetime):
                                        time_str = candle_date.strftime("%H:%M:%S")
                                        if "09:15" <= time_str <= "15:30":
                                            trading_candles.append(candle)
                                except:
                                    pass
                        
                        # If no filtered candles, use all candles
                        if not trading_candles:
                            trading_candles = option_historical
                        
                        if len(trading_candles) < 2:
                            print(f"Insufficient trading candles for {date_str}")
                            continue
                        
                        # Get 9:15 AM candle (first candle of the day) - needed for 9:15 strategy
                        first_candle = None
                        for candle in trading_candles:
                            candle_date = candle.get("date", "")
                            if candle_date:
                                try:
                                    time_str = ""
                                    if isinstance(candle_date, str) and ' ' in candle_date:
                                        time_str = candle_date.split(' ')[1][:8] if len(candle_date.split(' ')) > 1 else ""
                                    elif isinstance(candle_date, datetime):
                                        time_str = candle_date.strftime("%H:%M:%S")
                                    
                                    if time_str and "09:15" <= time_str <= "09:20":
                                        first_candle = candle
                                        break
                                except:
                                    pass
                        
                        # If no 9:15 candle found, use first candle
                        if not first_candle:
                            first_candle = trading_candles[0]
                        
                        # Apply selected strategy
                        strategy_result = None
                        
                        # Debug: Log which strategy is being used
                        print(f"  [{date_str}] Applying strategy: {strategy_type}")
                        
                        # Apply selected strategy
                        if strategy_type == "915_candle_break":
                            strategy_result = strategy_915_candle_break(
                                kite, trading_candles, first_candle, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str
                            )
                        elif strategy_type == "mean_reversion":
                            strategy_result = strategy_mean_reversion_bollinger(
                                kite, trading_candles, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str
                            )
                        elif strategy_type == "momentum_breakout":
                            strategy_result = strategy_momentum_breakout(
                                kite, trading_candles, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str
                            )
                        elif strategy_type == "support_resistance":
                            strategy_result = strategy_support_resistance_breakout(
                                kite, trading_candles, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str
                            )
                        # Sensibull-style multi-leg strategies
                        elif strategy_type == "long_straddle":
                            strategy_result = strategy_long_straddle(
                                kite, trading_candles, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date
                            )
                        elif strategy_type == "long_strangle":
                            strategy_result = strategy_long_strangle(
                                kite, trading_candles, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date
                            )
                        elif strategy_type == "bull_call_spread":
                            strategy_result = strategy_bull_call_spread(
                                kite, trading_candles, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date
                            )
                        elif strategy_type == "bear_put_spread":
                            strategy_result = strategy_bear_put_spread(
                                kite, trading_candles, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date
                            )
                        elif strategy_type == "iron_condor":
                            strategy_result = strategy_iron_condor(
                                kite, trading_candles, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str, nifty_options, trade_date
                            )
                        elif strategy_type == "macd_crossover":
                            strategy_result = strategy_macd_crossover(
                                kite, trading_candles, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str
                            )
                        elif strategy_type == "rsi_reversal":
                            strategy_result = strategy_rsi_reversal(
                                kite, trading_candles, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str
                            )
                        elif strategy_type == "ema_cross":
                            strategy_result = strategy_ema_cross(
                                kite, trading_candles, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str
                            )
                        else:
                            # Default to 9:15 candle break
                            strategy_result = strategy_915_candle_break(
                                kite, trading_candles, first_candle, nifty_price, 
                                current_strike, atm_ce, atm_pe, date_str
                            )
                        
                        if not strategy_result:
                            print(f"No signal from {strategy_type} strategy for {date_str}")
                            continue
                        
                        # Extract strategy results
                        trend = strategy_result["trend"]
                        option_to_trade = strategy_result["option_to_trade"]
                        option_type = strategy_result["option_type"]
                        entry_price = strategy_result["entry_price"]
                        entry_time = strategy_result["entry_time"]
                        strategy_reason = strategy_result["reason"]
                        is_multi_leg = strategy_result.get("multi_leg", False)
                        strategy_legs = strategy_result.get("legs", [])
                        
                        if entry_price <= 0:
                            continue
                        
                        # PROFESSIONAL RISK MANAGEMENT & POSITION SIZING:
                        # 1. Calculate Risk and Reward absolute amounts from Fund
                        risk_amount = fund * risk_pct      # e.g., 200,000 * 0.01 = 2,000
                        reward_amount = fund * reward_pct  # e.g., 200,000 * 0.03 = 6,000
                        
                        # 2. Determine Quantity based on User's Risk Formula:
                        # Filter: Ignore trades where the combined gap % is less than 10%
                        # This prevents unrealistic quantity sizes and "noise" trades.
                        total_gap_pct = (reward_pct + risk_pct) * 100
                        if total_gap_pct < 10:
                            print(f"Skipping trade: Gap % too small ({total_gap_pct:.1f}%)")
                            continue
                            
                        # qty = (Fund * Risk%) / (TargetPrice - StoplossPrice)
                        price_diff = entry_price * (reward_pct + risk_pct)
                        
                        if price_diff > 0:
                            target_qty = risk_amount / price_diff
                            lots = round(target_qty / 75)
                            if lots < 1: lots = 1
                        else:
                            lots = 1
                            
                        # Final check: Ensure total trade value doesn't exceed fund
                        if (lots * 75 * entry_price) > fund:
                            lots = int(fund / (entry_price * 75))
                            if lots < 1: lots = 1
                            
                        quantity = lots * 75
                        
                        # 3. Calculate Stoploss and Target points PER UNIT based on the configured percentages
                        sl_points = entry_price * risk_pct
                        target_points = entry_price * reward_pct
                        
                        if trend == "BULLISH" or trend == "NEUTRAL":
                            target_price = entry_price + target_points
                            stoploss_price = entry_price - sl_points
                        else: # BEARISH
                            target_price = entry_price - target_points
                            stoploss_price = entry_price + sl_points
                        
                        # 4. Check if target/stoploss was hit during the day
                        target_hit = False
                        stoploss_hit = False
                        intraday_exit_price = None
                        intraday_exit_time = None
                        
                        # Check intraday candles for target/stoploss - ONLY AFTER ENTRY TIME
                        entry_time_obj = datetime.strptime(entry_time, "%H:%M:%S").time()
                        
                        for candle in trading_candles:
                            candle_date = candle.get("date", "")
                            if not candle_date: continue
                                
                            try:
                                if isinstance(candle_date, str):
                                    time_str = candle_date.split(' ')[1][:8] if ' ' in candle_date else ""
                                else:
                                    time_str = candle_date.strftime("%H:%M:%S")
                                    
                                if not time_str: continue
                                    
                                candle_time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
                                if candle_time_obj <= entry_time_obj: continue
                                    
                                candle_high = candle.get("high", 0)
                                candle_low = candle.get("low", 0)
                                candle_close = candle.get("close", 0)
                                
                                # Check for target/stoploss (Only for single-leg trades)
                                # For multi-leg, we exit at EOD (2:30 PM) for accuracy
                                if not is_multi_leg:
                                    if trend == "BULLISH" or trend == "NEUTRAL":
                                        if candle_high >= target_price:
                                            target_hit, intraday_exit_price, intraday_exit_time = True, target_price, time_str
                                            break
                                        elif candle_low <= stoploss_price:
                                            stoploss_hit, intraday_exit_price, intraday_exit_time = True, stoploss_price, time_str
                                            break
                                    else: # BEARISH
                                        if candle_low <= target_price:
                                            target_hit, intraday_exit_price, intraday_exit_time = True, target_price, time_str
                                            break
                                        elif candle_high >= stoploss_price:
                                            stoploss_hit, intraday_exit_price, intraday_exit_time = True, stoploss_price, time_str
                                            break
                                        
                                # Time-based exit: 2:30 PM (14:30) for all trades
                                if time_str >= "14:30:00":
                                    intraday_exit_price, intraday_exit_time = candle_close, time_str
                                    break
                            except: continue
                        
                        # Determine final exit details
                        if intraday_exit_price is not None:
                            exit_price, exit_time = intraday_exit_price, intraday_exit_time
                        else:
                            last_candle = trading_candles[-1]
                            exit_price = last_candle.get("close", entry_price)
                            exit_time = last_candle.get("date").split(' ')[1][:8] if isinstance(last_candle.get("date"), str) else last_candle.get("date").strftime("%H:%M:%S")

                        # For multi-leg trades, fetch individual leg exit prices at exit_time
                        final_legs = []
                        if is_multi_leg and strategy_legs:
                            for leg in strategy_legs:
                                leg_inst = find_option(nifty_options, leg['strike'], leg['type'])
                                if leg_inst:
                                    # Fetch price at exit_time
                                    try:
                                        # Use a small window around exit_time to find the candle
                                        exit_dt = datetime.combine(trade_date, datetime.strptime(exit_time, "%H:%M:%S").time())
                                        leg_hist = kite.historical_data(leg_inst["instrument_token"], exit_dt - timedelta(minutes=5), exit_dt + timedelta(minutes=5), "5minute")
                                        leg_exit_price = leg_hist[-1]["close"] if leg_hist else leg['price']
                                    except:
                                        leg_exit_price = leg['price'] # Fallback
                                    
                                    # Calculate individual leg P&L
                                    leg_pnl = (leg_exit_price - leg['price']) * quantity
                                    if leg['action'] == "SELL":
                                        leg_pnl = -leg_pnl
                                    
                                    final_legs.append({
                                        **leg,
                                        "exit_price": round(leg_exit_price, 2),
                                        "pnl": round(leg_pnl, 2)
                                    })
                        
                        # Calculate P&L
                        if is_multi_leg and final_legs:
                            # Accurate P&L for multi-leg is the sum of leg P&Ls
                            gross_pnl = sum([leg['pnl'] for leg in final_legs])
                            # Re-calculate exit_price as net price for reporting
                            exit_price = sum([leg['exit_price'] if leg['action'] == "BUY" else -leg['exit_price'] for leg in final_legs])
                        else:
                            gross_pnl = (exit_price - entry_price) * quantity if trend != "BEARISH" else (entry_price - exit_price) * quantity
                        
                        brokerage = 40 if not is_multi_leg else (40 * len(strategy_legs))
                        pnl = gross_pnl - brokerage
                        
                        # Skip trades with very small profit potential (less than costs)
                        # This prevents taking trades that can't be profitable
                        min_profit_needed = 60 # approx total costs for a lot
                        if gross_pnl < min_profit_needed and gross_pnl > 0:
                            print(f"{date_str}: Skipping trade - profit ({gross_pnl:.2f}) too close to costs")
                            continue
                        
                        # Build reason for entry and exit
                        reason_parts = []
                        
                        # Entry reason with strategy details
                        if is_multi_leg and strategy_legs:
                            legs_desc = ", ".join([f"{leg['action']} {leg['type']} @ {leg['strike']}" for leg in strategy_legs])
                            entry_reason = f"Entry: {option_type} (Multi-leg: {legs_desc}) @ Net ₹{round(entry_price, 2)}"
                        else:
                            entry_reason = f"Entry: {option_type} @ ₹{round(entry_price, 2)}"
                        entry_reason += f" | Strategy: {strategy_type.replace('_', ' ').title()}"
                        entry_reason += f" | {strategy_reason}"
                        entry_reason += f" | Strike: {current_strike}, Nifty: ₹{round(nifty_price, 2)}"
                        reason_parts.append(entry_reason)
                        
                        # Calculate actual stoploss percentage for reporting
                        stoploss_pct_actual = (sl_points / entry_price * 100) if entry_price > 0 else 0
                        target_pct_actual = (target_points / entry_price * 100) if entry_price > 0 else 0
                        
                        # Exit reason
                        exit_reason = ""
                        if target_hit:
                            exit_reason = f"Exit: Target hit @ ₹{round(exit_price, 2)} at {exit_time} (Target Profit ₹{reward_amount} achieved)"
                        elif stoploss_hit:
                            exit_reason = f"Exit: Stoploss hit @ ₹{round(exit_price, 2)} at {exit_time} (Risk Limit ₹{risk_amount} triggered)"
                        else:
                            exit_reason = f"Exit: EOD @ ₹{round(exit_price, 2)} at {exit_time}"
                        
                        reason_parts.append(exit_reason)
                        
                        # Add P&L explanation
                        if pnl > 0:
                            reason_parts.append(f"P&L: +₹{round(pnl, 2)} (Goal was ₹{reward_amount})")
                        elif pnl < 0:
                            reason_parts.append(f"P&L: -₹{round(abs(pnl), 2)} (Risk Limit was ₹{risk_amount})")
                        else:
                            reason_parts.append(f"P&L: ₹0 (Breakeven)")
                        
                        reason = " | ".join(reason_parts)
                        
                        trade_result = {
                            "date": date_str,
                            "nifty_price": round(nifty_price, 2),
                            "strike": current_strike,
                            "option_type": option_type,
                            "tradingsymbol": option_to_trade.get("tradingsymbol"),
                            "quantity": quantity,
                            "entry_price": round(entry_price, 2),
                            "entry_time": entry_time,
                            "exit_price": round(exit_price, 2),
                            "exit_time": exit_time,
                            "pnl": round(pnl, 2),
                            "trend": trend,
                            "status": "WIN" if pnl > 0 else "LOSS",
                            "reason": reason,
                            "legs": final_legs
                        }
                        
                        backtest_results["trades"].append(trade_result)
                        print(f"Trade added for {date_str}: {option_type} @ ₹{entry_price}, Entry: {entry_time}, Exit: {exit_time}, Qty: {quantity}, P&L: ₹{pnl}")
                    else:
                        print(f"No historical data for option on {date_str}")
                            
                except Exception as e:
                    print(f"Error processing trade for {date_str}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
            except Exception as e:
                print(f"Error processing date {trade_date}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate statistics with advanced metrics
        trades = backtest_results["trades"]
        if trades:
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t["pnl"] > 0])
            losing_trades = total_trades - winning_trades
            total_pnl = sum([t["pnl"] for t in trades])
            
            profits = [t["pnl"] for t in trades if t["pnl"] > 0]
            losses = [t["pnl"] for t in trades if t["pnl"] < 0]
            
            # Calculate advanced metrics
            avg_profit = sum(profits) / len(profits) if profits else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            profit_factor = abs(sum(profits) / sum(losses)) if losses and sum(losses) != 0 else 0
            
            # Calculate drawdown
            cumulative_pnl = 0
            peak = 0
            max_drawdown = 0
            for trade in trades:
                cumulative_pnl += trade["pnl"]
                if cumulative_pnl > peak:
                    peak = cumulative_pnl
                drawdown = peak - cumulative_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Calculate average holding time (in minutes)
            holding_times = []
            for trade in trades:
                try:
                    entry_time_obj = datetime.strptime(trade.get("entry_time", "09:20:00"), "%H:%M:%S")
                    exit_time_obj = datetime.strptime(trade.get("exit_time", "15:30:00"), "%H:%M:%S")
                    # If exit is next day, add 6.25 hours (market hours)
                    if exit_time_obj < entry_time_obj:
                        exit_time_obj = datetime.strptime("15:30:00", "%H:%M:%S")
                    time_diff = (exit_time_obj.hour * 60 + exit_time_obj.minute) - (entry_time_obj.hour * 60 + entry_time_obj.minute)
                    if time_diff < 0:
                        time_diff += 375  # Add market hours (6.25 hours = 375 minutes)
                    holding_times.append(time_diff)
                except:
                    holding_times.append(375)  # Default to full day
            
            avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
            
            # Risk-reward ratio
            risk_reward_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
            
            backtest_results["statistics"] = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "total_pnl": round(total_pnl, 2),
                "win_rate": round((winning_trades / total_trades * 100) if total_trades > 0 else 0, 2),
                "average_profit": round(avg_profit, 2),
                "average_loss": round(avg_loss, 2),
                "max_profit": round(max(profits) if profits else 0, 2),
                "max_loss": round(min(losses) if losses else 0, 2),
                "profit_factor": round(profit_factor, 2),
                "max_drawdown": round(max_drawdown, 2),
                "avg_holding_time_minutes": round(avg_holding_time, 0),
                "risk_reward_ratio": round(risk_reward_ratio, 2)
            }
        
        return {"data": backtest_results}
        
    except HTTPException:
        raise
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error backtesting strategy: {str(e)}")
