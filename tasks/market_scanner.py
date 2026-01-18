"""
Background tasks for market scanning and order monitoring
"""
import asyncio
from datetime import datetime, timedelta

from agent.config import get_agent_config
from agent.ws_manager import add_agent_log
from agent.tools.kite_tools import cancel_order_tool
from agent.approval import get_approval_queue
from utils.kite_utils import get_kite_instance
from utils.candle_utils import aggregate_to_tf, analyze_trend
from strategies.runner import run_strategy_on_candles


async def live_market_scanner():
    """Background task to scan market and execute strategies"""
    add_agent_log("Live Market Scanner initializing...", "info")
    config = get_agent_config()
    
    # Heartbeat tracker to avoid log spam
    last_heartbeat_min = -1
    last_analysis_min = -1
    
    while True:
        try:
            # Only scan between market hours from config
            now = datetime.now()
            start_time = datetime.strptime(config.trading_start_time, "%H:%M").time()
            end_time = datetime.strptime(config.trading_end_time, "%H:%M").time()
            
            # Indian Market Hours (From Config)
            if start_time <= now.time() <= end_time:
                active_strats = config.active_strategies.split(",") if config.active_strategies else []
                if active_strats:
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
                            from_date = now - timedelta(days=5)  # Get 5 days for enough 1h/Day data
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
                                
                                add_agent_log(f"ANALYSIS: " + " | ".join(analysis_parts), "info")
                                last_analysis_min = now.minute

                        # Heartbeat tracker for Nifty price and active strategies
                        if now.minute % 5 == 0 and now.minute != last_heartbeat_min:
                            active_strats_str = ", ".join(active_strats) if active_strats else "None"
                            add_agent_log(f"LIVE SCANNING: Nifty @ {nifty_price} | Active: {active_strats_str}", "info")
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
                                    for strategy_id in active_strats:
                                        res = await run_strategy_on_candles(
                                            kite, strategy_id, trading_candles_5m, first_candle, 
                                            nifty_price, current_strike, atm_ce, atm_pe, 
                                            now.strftime("%Y-%m-%d"), nifty_options, today
                                        )
                                        
                                        if res:
                                            # Signal detected in LIVE market
                                            message = f"LIVE SIGNAL: {strategy_id} triggered. Reason: {res['reason']}"
                                            
                                            if config.is_auto_trade_enabled:
                                                add_agent_log(f"EXECUTING LIVE ORDER: {message}", "signal")
                                            else:
                                                add_agent_log(f"ALERT (Auto-Trade OFF): {message}", "info")
                else:
                    # Log once that no strategies are active
                    if now.minute % 30 == 0 and now.minute != last_heartbeat_min:
                        add_agent_log("LIVE MONITORING: Waiting for strategies to be selected...", "info")
                        last_heartbeat_min = now.minute
            else:
                # Outside market hours
                if now.minute % 60 == 0 and now.minute != last_heartbeat_min:
                    add_agent_log(f"LIVE FEED: Market is currently CLOSED ({now.strftime('%H:%M')}). Scanning resumes at 09:15 AM.", "info")
                    last_heartbeat_min = now.minute
            
            # Run every 10 seconds (more responsive scanner)
            await asyncio.sleep(10)
        except Exception as e:
            print(f"Scanner error: {e}")
            await asyncio.sleep(30)  # Wait longer on error


async def monitor_order_execution():
    """
    Background task to monitor order execution and auto-cancel remaining orders.
    When SL or Target executes, the other order should be cancelled automatically.
    """
    
    while True:
        try:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            approval_queue = get_approval_queue()
            approved_trades = approval_queue.list_approved()
            
            if not approved_trades:
                continue
            
            kite = get_kite_instance()
            
            # Get all orders from Kite
            try:
                orders = kite.orders()
            except Exception as e:
                add_agent_log(f"Error fetching orders for monitoring: {e}", "error")
                continue
            
            # Process each approved trade
            for approval in approved_trades:
                details = approval.get("details", {})
                if details.get("is_simulated", False):
                    continue  # Skip simulated trades
                
                approval_id = approval.get("approval_id")
                approval_obj = approval_queue.get_approval(approval_id)
                
                if not approval_obj:
                    continue
                
                sl_order_id = approval_obj.get("sl_order_id")
                tp_order_id = approval_obj.get("tp_order_id")
                
                # Skip if no exit orders
                if not sl_order_id and not tp_order_id:
                    continue
                
                # Check order statuses
                sl_executed = False
                tp_executed = False
                sl_cancelled = False
                tp_cancelled = False
                
                for order in orders:
                    order_id_str = str(order.get("order_id", ""))
                    
                    if sl_order_id and order_id_str == str(sl_order_id):
                        status = order.get("status", "").upper()
                        if status in ["COMPLETE", "FILLED"]:
                            sl_executed = True
                        elif status in ["CANCELLED", "REJECTED"]:
                            sl_cancelled = True
                    
                    if tp_order_id and order_id_str == str(tp_order_id):
                        status = order.get("status", "").upper()
                        if status in ["COMPLETE", "FILLED"]:
                            tp_executed = True
                        elif status in ["CANCELLED", "REJECTED"]:
                            tp_cancelled = True
                
                # Auto-cancel logic
                if sl_executed and tp_order_id and not tp_cancelled:
                    # SL executed, cancel Target
                    try:
                        cancel_result = cancel_order_tool.invoke({
                            "order_id": str(tp_order_id),
                            "variety": "regular"
                        })
                        if cancel_result.get("status") == "success":
                            add_agent_log(f"✅ Auto-cancelled Target order {tp_order_id} (SL executed)", "info")
                        else:
                            add_agent_log(f"⚠️ Failed to cancel Target order {tp_order_id}: {cancel_result.get('error')}", "warning")
                    except Exception as e:
                        add_agent_log(f"Error cancelling Target order: {e}", "error")
                
                elif tp_executed and sl_order_id and not sl_cancelled:
                    # Target executed, cancel SL
                    try:
                        cancel_result = cancel_order_tool.invoke({
                            "order_id": str(sl_order_id),
                            "variety": "regular"
                        })
                        if cancel_result.get("status") == "success":
                            add_agent_log(f"✅ Auto-cancelled Stop Loss order {sl_order_id} (Target executed)", "info")
                        else:
                            add_agent_log(f"⚠️ Failed to cancel SL order {sl_order_id}: {cancel_result.get('error')}", "warning")
                    except Exception as e:
                        add_agent_log(f"Error cancelling SL order: {e}", "error")
                        
        except Exception as e:
            add_agent_log(f"Error in order monitoring task: {e}", "error")
            await asyncio.sleep(30)  # Wait longer on error





