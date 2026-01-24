"""
30-Minute Range Breakout Strategy Backtest
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
from datetime import datetime, timedelta, date

from utils.kite_utils import get_kite_instance
from .shared import to_bool, send_log, get_trading_dates

router = APIRouter()

@router.websocket("/ws/backtest-range-breakout-30min")
async def backtest_range_breakout_30min_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for 30-minute range breakout strategy backtest
    Strategy: Wait for 9:15-9:45 AM 30-min candle, then trade first 5-min breakout
    Client sends: {"start_date": "2025-01-01", "end_date": "2025-01-31", "instrument_token": "738561", "instrument_name": "RELIANCE", "risk_per_trade": 1.0, "capital": 100000}
    """
    await websocket.accept()
    try:
        message = await websocket.receive_text()
        params = json.loads(message)
        
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        instrument_token = params.get("instrument_token")
        instrument_name = params.get("instrument_name", "Unknown")
        risk_per_trade = params.get("risk_per_trade", 1.0)  # 1% default
        reward_ratio = params.get("reward_ratio", 2.0)  # 2x risk default
        capital = params.get("capital", 100000.0)
        
        # Strategy configuration (with defaults) - use shared to_bool function
        allow_options_selling = to_bool(params.get("allow_options_selling"), False)
        skip_pe_on_gap_up = to_bool(params.get("skip_pe_on_gap_up"), True)
        skip_ce_on_gap_down = to_bool(params.get("skip_ce_on_gap_down"), True)
        skip_long_on_gap_down = to_bool(params.get("skip_long_on_gap_down"), True)
        skip_short_on_gap_up = to_bool(params.get("skip_short_on_gap_up"), True)
        skip_exhaustion_candles = to_bool(params.get("skip_exhaustion_candles"), True)
        
        if not start_date or not end_date or not instrument_token:
            await websocket.send_json({
                "type": "error",
                "message": "start_date, end_date, and instrument_token are required"
            })
            await websocket.close()
            return
        
        # Get user_id
        user_id = "default"
        try:
            user_id_param = websocket.query_params.get("user_id")
            if user_id_param:
                user_id = user_id_param
        except:
            pass
        
        kite = get_kite_instance(user_id=user_id)
        
        # Get instrument details (lot size, exchange, option type) once at the start
        lot_size = 1  # Default lot size for equity (1 share)
        instrument_exchange = "NSE"  # Default exchange
        instrument_type = None  # CE, PE, or None for equity
        is_index = False  # Flag to indicate if instrument is an index
        index_name = None  # Name of the index (e.g., "NIFTY", "BANKNIFTY")
        
        try:
            # Check if instrument is an index (NIFTY 50, BANKNIFTY, etc.)
            nse_instruments = kite.instruments("NSE")
            for inst in nse_instruments:
                if inst.get("instrument_token") == int(instrument_token):
                    inst_type = inst.get("instrument_type", "").upper()
                    inst_segment = inst.get("segment", "").upper()
                    if inst_type == "INDEX" or inst_segment == "INDICES":
                        is_index = True
                        index_name = inst.get("name", "").upper()
                        # Map common index names to option names
                        if "NIFTY" in index_name and "BANK" not in index_name:
                            index_name = "NIFTY"
                        elif "BANKNIFTY" in index_name or "NIFTY BANK" in index_name:
                            index_name = "BANKNIFTY"
                        elif "FINNIFTY" in index_name:
                            index_name = "FINNIFTY"
                        print(f"[Backtest] Instrument is an INDEX: {index_name}")
                        break
            
            # Check if instrument is in NFO exchange (direct option)
            if not is_index:
                nfo_instruments = kite.instruments("NFO")
                for inst in nfo_instruments:
                    if inst.get("instrument_token") == int(instrument_token):
                        lot_size = inst.get("lot_size", 1)
                        instrument_exchange = "NFO"
                        instrument_type = inst.get("instrument_type")  # CE or PE
                        print(f"[Backtest] Instrument is NFO with lot_size: {lot_size}, type: {instrument_type}")
                        break
        except Exception as e:
            print(f"[Backtest] Could not fetch instrument details, assuming equity (lot_size=1): {e}")
        
        # If it's an index, we'll need to get lot size from options later
        if is_index:
            # Default lot size for Nifty options (will be fetched when selecting option)
            lot_size = 65  # Default for NIFTY, will be updated based on actual option
        
        # Parse dates
        from_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        to_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Get trading dates (weekdays only)
        trading_dates = []
        current_date = from_date
        while current_date <= to_date:
            if current_date.weekday() < 5:  # Monday=0 to Friday=4
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        print(f"[Backtest] Starting backtest for {instrument_name} ({instrument_token}) from {start_date} to {end_date}")
        print(f"[Backtest] Trading dates: {len(trading_dates)} days")
        print(f"[Backtest] Risk: {risk_per_trade}%, Reward Ratio: {reward_ratio}x, Capital: ₹{capital:,.2f}")
        print(f"[Backtest] Raw params received: allow_options_selling={params.get('allow_options_selling')} (type: {type(params.get('allow_options_selling'))}), skip_pe_on_gap_up={params.get('skip_pe_on_gap_up')} (type: {type(params.get('skip_pe_on_gap_up'))})")
        print(f"[Backtest] Strategy Configuration (after conversion): allow_options_selling={allow_options_selling} (type: {type(allow_options_selling)}), skip_pe_on_gap_up={skip_pe_on_gap_up} (type: {type(skip_pe_on_gap_up)}), skip_ce_on_gap_down={skip_ce_on_gap_down}, skip_long_on_gap_down={skip_long_on_gap_down}, skip_short_on_gap_up={skip_short_on_gap_up}, skip_exhaustion_candles={skip_exhaustion_candles}")
        
        await send_log(websocket, "info", "Strategy configuration loaded", None, {
            "allow_options_selling": allow_options_selling,
            "skip_pe_on_gap_up": skip_pe_on_gap_up,
            "skip_ce_on_gap_down": skip_ce_on_gap_down,
            "skip_long_on_gap_down": skip_long_on_gap_down,
            "skip_short_on_gap_up": skip_short_on_gap_up,
            "skip_exhaustion_candles": skip_exhaustion_candles,
            "reason": "Configuration from UI will be applied during backtest"
        })
        
        await websocket.send_json({
            "type": "start",
            "message": "Range Breakout backtest started",
            "params": {
                "start_date": start_date,
                "end_date": end_date,
                "instrument": instrument_name,
                "total_days": len(trading_dates)
            }
        })
        await asyncio.sleep(0.01)
        
        results = []
        total_trades = 0
        total_profit = 0.0
        cancelled = False
        
        for day_idx, trade_date in enumerate(trading_dates):
            if cancelled:
                break
            
            try:
                await websocket.send_json({
                    "type": "progress",
                    "date": trade_date.strftime("%Y-%m-%d"),
                    "progress": f"{day_idx + 1}/{len(trading_dates)}",
                    "message": f"Processing {trade_date.strftime('%Y-%m-%d')}..."
                })
                await asyncio.sleep(0.05)
            except:
                cancelled = True
                break
            
            try:
                # For index, we need to get gap first, then select option
                # For non-index, use instrument directly
                gap_calculation_token = instrument_token
                
                # Fetch 30-minute candles for gap calculation (use index if index, otherwise use instrument)
                try:
                    candles_30min = kite.historical_data(
                        int(gap_calculation_token),
                        trade_date,
                        trade_date,
                        "30minute"
                    )
                except Exception as e:
                    print(f"[Backtest] Error fetching 30min candles for {trade_date}: {e}")
                    continue
                
                if not candles_30min or len(candles_30min) == 0:
                    await send_log(websocket, "warning", f"No 30min candles available for {trade_date.strftime('%Y-%m-%d')}", trade_date.strftime("%Y-%m-%d"))
                    print(f"[Backtest] No 30min candles for {trade_date}")
                    continue
                
                # Get first 30-minute candle (9:15-9:45 AM IST) for gap calculation
                gap_reference_candle = None
                market_open = datetime.combine(trade_date, datetime.min.time()).replace(hour=9, minute=15)
                market_open_30min_end = market_open + timedelta(minutes=30)
                
                for candle in candles_30min:
                    candle_time = candle.get('date')
                    if isinstance(candle_time, str):
                        try:
                            candle_time = datetime.strptime(candle_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                        except:
                            continue
                    elif isinstance(candle_time, datetime):
                        # Make timezone-naive if it's timezone-aware
                        if candle_time.tzinfo is not None:
                            candle_time = candle_time.replace(tzinfo=None)
                    else:
                        continue
                    
                    # Ensure both are naive for comparison
                    if candle_time.tzinfo is not None:
                        candle_time = candle_time.replace(tzinfo=None)
                    
                    if market_open <= candle_time <= market_open_30min_end:
                        gap_reference_candle = {
                            'open': float(candle.get('open', 0)),
                            'high': float(candle.get('high', 0)),
                            'low': float(candle.get('low', 0)),
                            'close': float(candle.get('close', 0))
                        }
                        break
                
                if not gap_reference_candle:
                    await send_log(websocket, "warning", f"No reference candle (9:15-9:45) found for {trade_date.strftime('%Y-%m-%d')}", trade_date.strftime("%Y-%m-%d"))
                    print(f"[Backtest] No reference candle (9:15-9:45) found for {trade_date}")
                    continue
                
                # Get previous trading day's close to check for gap
                # Must find the actual previous trading day (excluding holidays and weekends)
                gap_status = "No Gap"
                gap_percent = 0.0
                prev_day_close = None
                prev_trading_date = None
                
                try:
                    # Find the previous trading day by going back day by day
                    # until we find a day with actual trading data
                    prev_date = trade_date - timedelta(days=1)
                    max_lookback_days = 10  # Maximum days to look back (to avoid infinite loop)
                    lookback_count = 0
                    found_prev_trading_day = False
                    
                    await send_log(websocket, "info", f"Finding previous trading day for {trade_date.strftime('%Y-%m-%d')} (weekday: {trade_date.strftime('%A')})", trade_date.strftime("%Y-%m-%d"), {
                        "trade_date": trade_date.strftime("%Y-%m-%d"),
                        "weekday": trade_date.strftime("%A"),
                        "reason": "Starting search for previous trading day to calculate gap"
                    })
                    
                    while lookback_count < max_lookback_days and not found_prev_trading_day:
                        # Skip weekends
                        if prev_date.weekday() >= 5:  # Saturday=5, Sunday=6
                            await send_log(websocket, "info", f"Skipping weekend: {prev_date.strftime('%Y-%m-%d')} ({prev_date.strftime('%A')})", trade_date.strftime("%Y-%m-%d"), {
                                "skipped_date": prev_date.strftime("%Y-%m-%d"),
                                "weekday": prev_date.strftime("%A"),
                                "reason": "Weekend - not a trading day"
                            })
                            prev_date = prev_date - timedelta(days=1)
                            lookback_count += 1
                            continue
                        
                        # Try to get data for this date
                        try:
                            # Fetch minute-level data to get the actual closing price at 3:30 PM
                            prev_candles_minute = kite.historical_data(
                                int(gap_calculation_token),
                                prev_date,
                                prev_date,
                                "minute"
                            )
                            
                            # Check if we got valid data (not a holiday)
                            if prev_candles_minute and len(prev_candles_minute) > 0:
                                # Filter candles for the specific date and get the last candle (should be around 3:30 PM)
                                prev_day_candles = []
                                for candle in prev_candles_minute:
                                    candle_time = candle.get('date')
                                    if isinstance(candle_time, str):
                                        try:
                                            candle_dt = datetime.strptime(candle_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                                        except:
                                            try:
                                                candle_dt = datetime.strptime(candle_time.split(' ')[0], '%Y-%m-%d')
                                            except:
                                                continue
                                    elif isinstance(candle_time, datetime):
                                        candle_dt = candle_time
                                    else:
                                        continue
                                    
                                    # Check if candle is for the previous trading date
                                    if candle_dt.date() == prev_date:
                                        prev_day_candles.append(candle)
                                
                                # If we have candles for this date, use the last one's close (actual market close at 3:30 PM)
                                if prev_day_candles:
                                    # Sort by time to get the last candle
                                    prev_day_candles.sort(key=lambda c: c.get('date'))
                                    last_candle = prev_day_candles[-1]
                                    candle_close = last_candle.get('close', 0)
                                    
                                    # If we have valid close price, use it
                                    if candle_close and float(candle_close) > 0:
                                        prev_day_close = float(candle_close)
                                        prev_trading_date = prev_date
                                        found_prev_trading_day = True
                                        
                                        # Get the time of the last candle for logging
                                        last_candle_time = last_candle.get('date')
                                        if isinstance(last_candle_time, datetime):
                                            last_candle_time_str = last_candle_time.strftime('%H:%M:%S')
                                        else:
                                            last_candle_time_str = str(last_candle_time)
                                        
                                        print(f"[Backtest] Found previous trading day for {trade_date}: {prev_date} (close: {prev_day_close:.2f} at {last_candle_time_str})")
                                        await send_log(websocket, "success", f"Found previous trading day: {prev_date.strftime('%Y-%m-%d')} ({prev_date.strftime('%A')}) - Close: ₹{prev_day_close:.2f} (at {last_candle_time_str})", trade_date.strftime("%Y-%m-%d"), {
                                            "prev_trading_date": prev_date.strftime("%Y-%m-%d"),
                                            "weekday": prev_date.strftime("%A"),
                                            "close": prev_day_close,
                                            "close_time": last_candle_time_str,
                                            "candles_count": len(prev_day_candles),
                                            "reason": f"Previous trading day found after looking back {lookback_count} day(s) - using last minute candle close"
                                        })
                                        break
                                    else:
                                        await send_log(websocket, "info", f"No valid close price for {prev_date.strftime('%Y-%m-%d')}, continuing search", trade_date.strftime("%Y-%m-%d"), {
                                            "checked_date": prev_date.strftime("%Y-%m-%d"),
                                            "reason": "Data exists but close price is invalid"
                                        })
                                else:
                                    await send_log(websocket, "info", f"No candles found for {prev_date.strftime('%Y-%m-%d')}, trying daily candles as fallback", trade_date.strftime("%Y-%m-%d"), {
                                        "checked_date": prev_date.strftime("%Y-%m-%d"),
                                        "reason": "No minute candles found, trying daily candles"
                                    })
                                    # Fallback to daily candles
                                    try:
                                        prev_candles_daily = kite.historical_data(
                                            int(gap_calculation_token),
                                            prev_date,
                                            prev_date,
                                            "day"
                                        )
                                        if prev_candles_daily and len(prev_candles_daily) > 0:
                                            candle_close = prev_candles_daily[-1].get('close', 0)
                                            if candle_close and float(candle_close) > 0:
                                                prev_day_close = float(candle_close)
                                                prev_trading_date = prev_date
                                                found_prev_trading_day = True
                                                print(f"[Backtest] Found previous trading day (daily candle) for {trade_date}: {prev_date} (close: {prev_day_close:.2f})")
                                                await send_log(websocket, "success", f"Found previous trading day: {prev_date.strftime('%Y-%m-%d')} ({prev_date.strftime('%A')}) - Close: ₹{prev_day_close:.2f} (from daily candle)", trade_date.strftime("%Y-%m-%d"), {
                                                    "prev_trading_date": prev_date.strftime("%Y-%m-%d"),
                                                    "weekday": prev_date.strftime("%A"),
                                                    "close": prev_day_close,
                                                    "reason": f"Previous trading day found using daily candle fallback"
                                                })
                                                break
                                    except Exception as daily_e:
                                        await send_log(websocket, "info", f"Daily candle fallback also failed for {prev_date.strftime('%Y-%m-%d')}, continuing search", trade_date.strftime("%Y-%m-%d"), {
                                            "checked_date": prev_date.strftime("%Y-%m-%d"),
                                            "error": str(daily_e)[:50],
                                            "reason": "Both minute and daily candles failed"
                                        })
                            else:
                                await send_log(websocket, "info", f"No data for {prev_date.strftime('%Y-%m-%d')} (likely holiday), continuing search", trade_date.strftime("%Y-%m-%d"), {
                                    "checked_date": prev_date.strftime("%Y-%m-%d"),
                                    "reason": "No candles returned - likely a market holiday"
                                })
                        except Exception as e:
                            # This date might be a holiday or error, continue to previous day
                            print(f"[Backtest] No data for {prev_date} (might be holiday): {e}")
                            await send_log(websocket, "info", f"Error fetching data for {prev_date.strftime('%Y-%m-%d')}: {str(e)[:50]}", trade_date.strftime("%Y-%m-%d"), {
                                "checked_date": prev_date.strftime("%Y-%m-%d"),
                                "error": str(e)[:100],
                                "reason": "Error fetching data - likely a holiday or API issue"
                            })
                        
                        # Move to previous day
                        prev_date = prev_date - timedelta(days=1)
                        lookback_count += 1
                    
                    # Calculate gap if we found previous trading day's close
                    if prev_day_close and prev_day_close > 0 and prev_trading_date:
                        current_open = gap_reference_candle['open']
                        gap = current_open - prev_day_close
                        gap_percent = (gap / prev_day_close) * 100 if prev_day_close > 0 else 0
                        
                        await send_log(websocket, "info", f"Gap calculation: Open={current_open:.2f}, Prev Close={prev_day_close:.2f}, Gap={gap:.2f} ({gap_percent:.2f}%)", trade_date.strftime("%Y-%m-%d"), {
                            "current_open": current_open,
                            "prev_close": prev_day_close,
                            "gap": gap,
                            "gap_percent": gap_percent,
                            "prev_trading_date": prev_trading_date.strftime("%Y-%m-%d"),
                            "reason": f"Gap calculated: {gap:.2f} points ({gap_percent:.2f}%)"
                        })
                        
                        if gap_percent > 0.1:  # Gap-up if > 0.1%
                            gap_status = f"Gap-Up ({gap_percent:.2f}%)"
                            is_gap_up = True
                            is_gap_down = False
                        elif gap_percent < -0.1:  # Gap-down if < -0.1%
                            gap_status = f"Gap-Down ({abs(gap_percent):.2f}%)"
                            is_gap_up = False
                            is_gap_down = True
                        else:
                            gap_status = f"No Gap ({gap_percent:.2f}%)"
                            is_gap_up = False
                            is_gap_down = False
                            await send_log(websocket, "info", f"Gap is within threshold (-0.1% to +0.1%), treating as 'No Gap'", trade_date.strftime("%Y-%m-%d"), {
                                "gap_percent": gap_percent,
                                "threshold": "±0.1%",
                                "reason": "Gap is too small to be considered significant"
                            })
                    else:
                        print(f"[Backtest] Could not find previous trading day for {trade_date} (looked back {lookback_count} days)")
                        gap_status = "N/A"
                        is_gap_up = False
                        is_gap_down = False
                        await send_log(websocket, "warning", f"Could not find previous trading day (looked back {lookback_count} days)", trade_date.strftime("%Y-%m-%d"), {
                            "lookback_days": lookback_count,
                            "reason": "No valid trading day found within lookback period"
                        })
                        
                except Exception as e:
                    print(f"[Backtest] Error fetching previous trading day data for {trade_date}: {e}")
                    import traceback
                    traceback.print_exc()
                    gap_status = "N/A"
                    is_gap_up = False
                    is_gap_down = False
                
                # If instrument is an index, select appropriate option based on gap direction
                actual_instrument_token = instrument_token
                actual_instrument_name = instrument_name
                actual_lot_size = lot_size
                actual_instrument_type = instrument_type
                reference_candle = gap_reference_candle  # Default to gap reference candle
                selected_option_strike = None  # Store selected option strike for index backtests
                selected_option_expiry = None  # Store selected option expiry for index backtests
                
                if is_index and index_name:
                    try:
                        # Determine option type based on gap direction
                        # Gap-up → CE (Call) option, Gap-down → PE (Put) option
                        if is_gap_up:
                            selected_option_type = "CE"
                        elif is_gap_down:
                            selected_option_type = "PE"
                        else:
                            # No gap, skip this day
                            await send_log(websocket, "skip", f"No gap detected, skipping option selection", trade_date.strftime("%Y-%m-%d"))
                            print(f"[Backtest] No gap for {trade_date}, skipping option selection for index")
                            continue
                        
                        # Get current index price (use gap reference candle open as proxy)
                        current_index_price = gap_reference_candle['open']
                        
                        # Determine strike interval based on index
                        if index_name == "NIFTY":
                            strike_interval = 50
                        elif index_name == "BANKNIFTY":
                            strike_interval = 100
                        elif index_name == "FINNIFTY":
                            strike_interval = 50
                        else:
                            strike_interval = 50  # Default
                        
                        # Round to nearest strike
                        current_strike = round(current_index_price / strike_interval) * strike_interval
                        
                        # Get all NFO instruments
                        nfo_instruments = kite.instruments("NFO")
                        
                        # Filter for options matching index name and type
                        matching_options = [
                            inst for inst in nfo_instruments
                            if inst.get("name") == index_name 
                            and inst.get("instrument_type") == selected_option_type
                            and inst.get("strike") == current_strike
                        ]
                        
                        if not matching_options:
                            await send_log(websocket, "warning", f"No {selected_option_type} option found at strike {current_strike}", trade_date.strftime("%Y-%m-%d"), {"strike": current_strike, "type": selected_option_type})
                            print(f"[Backtest] No {selected_option_type} option found for {index_name} at strike {current_strike} on {trade_date}")
                            continue
                        
                        # Get nearest expiry (expiry >= trade_date)
                        valid_options = []
                        for inst in matching_options:
                            expiry = inst.get("expiry")
                            if expiry:
                                # Handle both string and date types
                                if isinstance(expiry, str):
                                    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
                                elif isinstance(expiry, date):
                                    expiry_date = expiry
                                elif isinstance(expiry, datetime):
                                    expiry_date = expiry.date()
                                else:
                                    continue
                                
                                if expiry_date >= trade_date:
                                    valid_options.append(inst)
                        
                        if not valid_options:
                            await send_log(websocket, "warning", f"No valid expiry found for {index_name} {selected_option_type}", trade_date.strftime("%Y-%m-%d"))
                            print(f"[Backtest] No valid expiry found for {index_name} {selected_option_type} on {trade_date}")
                            continue
                        
                        # Sort by expiry and get nearest
                        def get_expiry_date(inst):
                            expiry = inst.get("expiry")
                            if isinstance(expiry, str):
                                return datetime.strptime(expiry, "%Y-%m-%d").date()
                            elif isinstance(expiry, date):
                                return expiry
                            elif isinstance(expiry, datetime):
                                return expiry.date()
                            return datetime.max.date()  # Fallback
                        
                        valid_options.sort(key=get_expiry_date)
                        selected_option = valid_options[0]
                        
                        actual_instrument_token = str(selected_option.get("instrument_token"))
                        actual_lot_size = selected_option.get("lot_size", 65)
                        actual_instrument_type = selected_option.get("instrument_type")
                        actual_instrument_name = f"{index_name} {current_strike} {selected_option_type}"
                        selected_option_strike = current_strike
                        
                        # Store expiry date
                        expiry = selected_option.get("expiry")
                        if expiry:
                            if isinstance(expiry, str):
                                selected_option_expiry = expiry
                            elif isinstance(expiry, date):
                                selected_option_expiry = expiry.strftime("%Y-%m-%d")
                            elif isinstance(expiry, datetime):
                                selected_option_expiry = expiry.strftime("%Y-%m-%d")
                        
                        # Build reason for option selection
                        reason_parts = []
                        if is_gap_up:
                            reason_parts.append(f"Gap-Up ({gap_percent:.2f}%) → Selected CE (Call) option")
                        elif is_gap_down:
                            reason_parts.append(f"Gap-Down ({abs(gap_percent):.2f}%) → Selected PE (Put) option")
                        reason_parts.append(f"Strike {current_strike} (Index price: ₹{current_index_price:.2f}, Interval: {strike_interval})")
                        reason_parts.append(f"Nearest expiry: {selected_option_expiry}")
                        
                        selection_reason = " | ".join(reason_parts)
                        
                        print(f"[Backtest] Selected option for {trade_date}: {actual_instrument_name} (token: {actual_instrument_token}, lot_size: {actual_lot_size}, expiry: {selected_option_expiry})")
                        await send_log(websocket, "success", f"Selected {actual_instrument_name}", trade_date.strftime("%Y-%m-%d"), {
                            "strike": selected_option_strike,
                            "type": actual_instrument_type,
                            "expiry": selected_option_expiry,
                            "lot_size": actual_lot_size,
                            "index_price": current_index_price,
                            "strike_interval": strike_interval,
                            "gap_percent": gap_percent,
                            "gap_status": gap_status,
                            "reason": selection_reason
                        })
                        
                        # Fetch 30-minute candles for the selected option to get reference candle
                        try:
                            option_candles_30min = kite.historical_data(
                                int(actual_instrument_token),
                                trade_date,
                                trade_date,
                                "30minute"
                            )
                            
                            # Get first 30-minute candle (9:15-9:45 AM IST) for the option
                            for candle in option_candles_30min:
                                candle_time = candle.get('date')
                                if isinstance(candle_time, str):
                                    try:
                                        candle_time = datetime.strptime(candle_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                                    except:
                                        continue
                                elif isinstance(candle_time, datetime):
                                    if candle_time.tzinfo is not None:
                                        candle_time = candle_time.replace(tzinfo=None)
                                else:
                                    continue
                                
                                if candle_time.tzinfo is not None:
                                    candle_time = candle_time.replace(tzinfo=None)
                                
                                if market_open <= candle_time <= market_open_30min_end:
                                    reference_candle = {
                                        'open': float(candle.get('open', 0)),
                                        'high': float(candle.get('high', 0)),
                                        'low': float(candle.get('low', 0)),
                                        'close': float(candle.get('close', 0))
                                    }
                                    break
                            
                            if not reference_candle:
                                await send_log(websocket, "warning", f"No 30-minute reference candle found for selected option on {trade_date.strftime('%Y-%m-%d')}", trade_date.strftime("%Y-%m-%d"), {
                                    "option": actual_instrument_name,
                                    "reason": "30-minute candle (9:15-9:45 AM) not found for selected option"
                                })
                                print(f"[Backtest] No reference candle found for selected option on {trade_date}")
                                continue
                            
                            # Verify we're using the option's reference candle, not the index's
                            # Log reference candle once (will be logged again below, but with more context)
                            # Skip this duplicate log
                                
                        except Exception as e:
                            await send_log(websocket, "error", f"Error fetching option 30min candles: {str(e)}", trade_date.strftime("%Y-%m-%d"), {
                                "option": actual_instrument_name
                            })
                            print(f"[Backtest] Error fetching option 30min candles for {trade_date}: {e}")
                            continue
                        
                    except Exception as e:
                        print(f"[Backtest] Error selecting option for index on {trade_date}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Log which reference candle is being used (only once)
                instrument_used = actual_instrument_name if is_index else instrument_name
                await send_log(websocket, "info", f"Reference candle (9:15-9:45 AM): HIGH=₹{reference_candle['high']:.2f}, LOW=₹{reference_candle['low']:.2f}", trade_date.strftime("%Y-%m-%d"), {
                    "ref_high": reference_candle['high'],
                    "ref_low": reference_candle['low'],
                    "instrument": instrument_used,
                    "gap_status": gap_status,
                    "reason": f"{'Selected option' if is_index else 'Instrument'}'s 30-minute candle closed at 9:45 AM. Ready for breakout detection."
                })
                
                print(f"[Backtest] Reference candle for {trade_date}: HIGH={reference_candle['high']:.2f}, LOW={reference_candle['low']:.2f}, Gap: {gap_status} (from {'option' if is_index else 'instrument'})")
                
                # Fetch 5-minute candles (use actual instrument token - may be option if index)
                try:
                    candles_5min = kite.historical_data(
                        int(actual_instrument_token),
                        trade_date,
                        trade_date,
                        "5minute"
                    )
                except Exception as e:
                    await send_log(websocket, "error", f"Error fetching 5min candles: {str(e)}", trade_date.strftime("%Y-%m-%d"), {
                        "instrument": instrument_used
                    })
                    print(f"[Backtest] Error fetching 5min candles for {trade_date}: {e}")
                    continue
                
                if not candles_5min or len(candles_5min) == 0:
                    await send_log(websocket, "warning", f"No 5-minute candles available for {trade_date.strftime('%Y-%m-%d')}", trade_date.strftime("%Y-%m-%d"), {
                        "instrument": instrument_used
                    })
                    print(f"[Backtest] No 5min candles for {trade_date}")
                    continue
                
                # Filter candles from 9:45 AM onwards (after 30-min reference candle closes)
                trading_candles = []
                candle_start_time = datetime.strptime('09:45', '%H:%M').time()
                
                for candle in candles_5min:
                    candle_time = candle.get('date')
                    if isinstance(candle_time, str):
                        try:
                            candle_time = datetime.strptime(candle_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                        except:
                            continue
                    elif isinstance(candle_time, datetime):
                        # Make timezone-naive if it's timezone-aware
                        if candle_time.tzinfo is not None:
                            candle_time = candle_time.replace(tzinfo=None)
                    else:
                        continue
                    
                    # Ensure timezone-naive
                    if candle_time.tzinfo is not None:
                        candle_time = candle_time.replace(tzinfo=None)
                    
                    # Only include candles from 9:45 AM onwards (after 30-min reference candle closes)
                    if candle_time.time() >= candle_start_time:
                        trading_candles.append(candle)
                
                # Skip this log - already logged reference candle above
                
                # Gap direction already determined above (is_gap_up and is_gap_down are set)
                
                # Find first breakout signal (only 1 trade per day)
                signal = None
                entry_candle_idx = None
                skipped_reason = None
                
                for idx, candle in enumerate(trading_candles):
                    close_5min = float(candle.get('close', 0))
                    high_5min = float(candle.get('high', 0))
                    low_5min = float(candle.get('low', 0))
                    open_5min = float(candle.get('open', 0))
                    
                    ref_high = reference_candle['high']
                    ref_low = reference_candle['low']
                    
                    # LONG: 5-min candle closes above 30-min HIGH
                    if close_5min > ref_high:
                        # Don't log detection separately - will log in skip or success message
                        
                        # Filter: For PE options, skip LONG on gap-up days (PE profits when price goes down)
                        if skip_pe_on_gap_up and (instrument_exchange == "NFO" or is_index) and actual_instrument_type == "PE" and is_gap_up:
                            skipped_reason = f"LONG skipped: PE on Gap-Up ({gap_percent:.2f}%)"
                            await send_log(websocket, "skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
                                "close": close_5min,
                                "ref_high": ref_high,
                                "reason": "PE profits when price goes down, but gap-up suggests upward momentum"
                            })
                            print(f"[Backtest] {skipped_reason} for {trade_date}")
                            continue  # Skip this signal, continue looking
                        
                        # Filter: For CE options, skip LONG on gap-down days (CE profits when price goes up)
                        if skip_ce_on_gap_down and (instrument_exchange == "NFO" or is_index) and actual_instrument_type == "CE" and is_gap_down:
                            skipped_reason = f"LONG skipped: CE on Gap-Down ({gap_percent:.2f}%)"
                            await send_log(websocket, "skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
                                "close": close_5min,
                                "ref_high": ref_high,
                                "reason": "CE profits when price goes up, but gap-down suggests downward momentum"
                            })
                            print(f"[Backtest] {skipped_reason} for {trade_date}")
                            continue  # Skip this signal, continue looking
                        
                        # Filter: Don't take LONG on gap-down days (against momentum) - for equity only
                        if skip_long_on_gap_down and instrument_exchange != "NFO" and is_gap_down:
                            skipped_reason = f"LONG skipped: Gap-Down ({gap_percent:.2f}%)"
                            await send_log(websocket, "skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
                                "close": close_5min,
                                "ref_high": ref_high,
                                "reason": "Gap-down suggests downward momentum, LONG would be against trend"
                            })
                            print(f"[Backtest] {skipped_reason} for {trade_date}")
                            continue  # Skip this signal, continue looking
                        
                        # Filter: Avoid candles where close = high (exhaustion signal)
                        # Allow small tolerance (0.01% or 0.01 rupees) for floating point comparison
                        if skip_exhaustion_candles and abs(close_5min - high_5min) < max(0.01, high_5min * 0.0001):
                            skipped_reason = f"LONG skipped: Exhaustion candle (Close=High)"
                            await send_log(websocket, "skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
                                "close": close_5min,
                                "reason": "Close equals high indicates exhaustion - buyers may be exhausted"
                            })
                            print(f"[Backtest] {skipped_reason} for {trade_date} at {close_5min:.2f}")
                            continue  # Skip this signal, continue looking
                        
                        entry_price = close_5min
                        stop_loss = low_5min
                        risk = entry_price - stop_loss
                        target = entry_price + (reward_ratio * risk)
                        
                        signal = {
                            'direction': 'LONG',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'target': target,
                            'entry_candle': candle
                        }
                        entry_candle_idx = idx
                        break
                    
                    # SHORT: 5-min candle closes below 30-min LOW
                    elif close_5min < ref_low:
                        # Don't log detection separately - will log in skip or success message
                        
                        # Filter: Skip SHORT positions for NFO instruments (options selling not possible)
                        if not allow_options_selling and (instrument_exchange == "NFO" or is_index):
                            skipped_reason = f"SHORT skipped: Options selling not allowed"
                            await send_log(websocket, "skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
                                "close": close_5min,
                                "ref_low": ref_low,
                                "reason": "Options selling requires margin/funds not available, only options buying is allowed"
                            })
                            print(f"[Backtest] {skipped_reason} for {trade_date}")
                            continue  # Skip this signal, continue looking
                        
                        # Filter: Don't take SHORT on gap-up days (against momentum)
                        if skip_short_on_gap_up and is_gap_up:
                            skipped_reason = f"SHORT skipped: Gap-Up ({gap_percent:.2f}%)"
                            await send_log(websocket, "skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
                                "close": close_5min,
                                "ref_low": ref_low,
                                "reason": "Gap-up suggests upward momentum, SHORT would be against trend"
                            })
                            print(f"[Backtest] {skipped_reason} for {trade_date}")
                            continue  # Skip this signal, continue looking
                        
                        # Filter: Avoid candles where close = low (exhaustion signal)
                        # Allow small tolerance (0.01% or 0.01 rupees) for floating point comparison
                        if skip_exhaustion_candles and abs(close_5min - low_5min) < max(0.01, low_5min * 0.0001):
                            skipped_reason = f"SHORT skipped: Exhaustion candle (Close=Low)"
                            await send_log(websocket, "skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
                                "close": close_5min,
                                "reason": "Close equals low indicates exhaustion - sellers may be exhausted"
                            })
                            print(f"[Backtest] {skipped_reason} for {trade_date} at {close_5min:.2f}")
                            continue  # Skip this signal, continue looking
                        
                        entry_price = close_5min
                        stop_loss = high_5min
                        risk = stop_loss - entry_price
                        target = entry_price - (reward_ratio * risk)
                        
                        signal = {
                            'direction': 'SHORT',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'target': target,
                            'entry_candle': candle
                        }
                        entry_candle_idx = idx
                        break
                
                if not signal:
                    if skipped_reason:
                        print(f"[Backtest] {skipped_reason} - No valid signal for {trade_date}")
                    else:
                        print(f"[Backtest] No breakout signal for {trade_date}")
                    continue
                
                print(f"[Backtest] Signal found for {trade_date}: {signal['direction']} at {signal['entry_price']:.2f}")
                await send_log(websocket, "success", f"{signal['direction']} signal found at ₹{signal['entry_price']:.2f}", trade_date.strftime("%Y-%m-%d"), {
                    "direction": signal['direction'],
                    "entry_price": signal['entry_price'],
                    "stop_loss": signal['stop_loss'],
                    "target": signal['target'],
                    "ref_high": reference_candle['high'],
                    "ref_low": reference_candle['low'],
                    "gap_status": gap_status,
                    "gap_percent": gap_percent,
                    "reason": f"5-min candle closed {'above' if signal['direction'] == 'LONG' else 'below'} 30-min reference {'high' if signal['direction'] == 'LONG' else 'low'}"
                })
                
                # Calculate position size using capital allocation approach
                # Formula: position_size = risk_amount / entry_price
                # This allocates a fixed percentage of capital (risk_per_trade%) per trade
                risk_amount = capital * (risk_per_trade / 100.0)
                
                if signal['entry_price'] == 0:
                    print(f"[Backtest] Entry price is 0 for {trade_date}, skipping trade")
                    continue
                
                # Calculate position size: risk_amount / entry_price
                position_size_float = risk_amount / signal['entry_price']
                
                # Also check capital constraint: qty * entry_price <= capital
                # This ensures we don't exceed available capital
                position_size_by_capital = capital / signal['entry_price']
                
                # Use the minimum of both to ensure we don't exceed capital
                position_size_float = min(position_size_float, position_size_by_capital)
                
                # For NFO instruments (or index options), round to lot size multiples
                if (instrument_exchange == "NFO" or is_index) and actual_lot_size > 1:
                    # Round down to nearest lot size multiple
                    lots = int(position_size_float / actual_lot_size)
                    # Ensure minimum 1 lot
                    if lots < 1:
                        lots = 1
                    position_size = lots * actual_lot_size
                    print(f"[Backtest] NFO instrument: rounded to {lots} lot(s) × {actual_lot_size} = {position_size} qty")
                else:
                    # For equity, round down to nearest integer (no decimal quantities)
                    position_size = int(position_size_float)
                
                if position_size == 0:
                    print(f"[Backtest] Position size is 0 for {trade_date} after rounding, skipping trade. Risk-based: {risk_amount / signal['entry_price']:.2f}, Capital-based: {position_size_by_capital:.2f}")
                    continue
                
                # Verify the investment doesn't exceed capital
                total_investment = position_size * signal['entry_price']
                if total_investment > capital:
                    print(f"[Backtest] Warning: Investment {total_investment:.2f} exceeds capital {capital:.2f} for {trade_date}")
                    # Further reduce if needed
                    if instrument_exchange == "NFO" and lot_size > 1:
                        # Reduce by lot size
                        max_lots = int(capital / (signal['entry_price'] * lot_size))
                        if max_lots < 1:
                            print(f"[Backtest] Cannot afford even 1 lot ({lot_size} qty) for {trade_date}, skipping trade")
                            continue
                        position_size = max_lots * lot_size
                    else:
                        position_size = int(capital / signal['entry_price'])
                    total_investment = position_size * signal['entry_price']
                
                print(f"[Backtest] Position size: {position_size}, Entry: {signal['entry_price']:.2f}, Investment: {total_investment:.2f}, Risk Amount: {risk_amount:.2f}, Capital: {capital:.2f}, Lot Size: {lot_size}, Exchange: {instrument_exchange}")
                
                # Get candles after entry for exit calculation
                candles_after_entry = trading_candles[entry_candle_idx + 1:]
                eod_time = datetime.combine(trade_date, datetime.min.time()).replace(hour=15, minute=30)
                
                # Calculate exit
                exit_price = signal['entry_price']
                exit_time = None
                exit_reason = "End of day (3:30 PM)"
                
                for candle in candles_after_entry:
                    candle_time = candle.get('date')
                    if isinstance(candle_time, str):
                        try:
                            candle_time = datetime.strptime(candle_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                        except:
                            continue
                    elif isinstance(candle_time, datetime):
                        # Make timezone-naive if it's timezone-aware
                        if candle_time.tzinfo is not None:
                            candle_time = candle_time.replace(tzinfo=None)
                    else:
                        continue
                    
                    # Ensure timezone-naive
                    if candle_time.tzinfo is not None:
                        candle_time = candle_time.replace(tzinfo=None)
                    
                    if candle_time.time() > eod_time.time():
                        break
                    
                    high = float(candle.get('high', 0))
                    low = float(candle.get('low', 0))
                    close = float(candle.get('close', 0))
                    
                    if signal['direction'] == 'LONG':
                        if high >= signal['target']:
                            exit_price = signal['target']
                            exit_time = candle_time
                            exit_reason = f"Target hit ({reward_ratio}x risk)"
                            break
                        if low <= signal['stop_loss']:
                            exit_price = signal['stop_loss']
                            exit_time = candle_time
                            exit_reason = "Stop loss triggered"
                            break
                    else:  # SHORT
                        if low <= signal['target']:
                            exit_price = signal['target']
                            exit_time = candle_time
                            exit_reason = f"Target hit ({reward_ratio}x risk)"
                            break
                        if high >= signal['stop_loss']:
                            exit_price = signal['stop_loss']
                            exit_time = candle_time
                            exit_reason = "Stop loss triggered"
                            break
                
                # If no exit found, use last candle close
                if exit_time is None and candles_after_entry:
                    last_candle = candles_after_entry[-1]
                    exit_price = float(last_candle.get('close', signal['entry_price']))
                    exit_time_str = last_candle.get('date')
                    if isinstance(exit_time_str, str):
                        try:
                            exit_time = datetime.strptime(exit_time_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                        except:
                            exit_time = eod_time
                    elif isinstance(exit_time_str, datetime):
                        exit_time = exit_time_str
                        # Make timezone-naive if it's timezone-aware
                        if exit_time.tzinfo is not None:
                            exit_time = exit_time.replace(tzinfo=None)
                    else:
                        exit_time = eod_time
                else:
                    exit_time = exit_time or eod_time
                
                # Calculate P&L
                if signal['direction'] == 'LONG':
                    profit = (exit_price - signal['entry_price']) * position_size
                    # Expected profit if target is hit
                    expected_profit = (signal['target'] - signal['entry_price']) * position_size
                    # Expected loss if stop loss is hit
                    expected_loss = (signal['entry_price'] - signal['stop_loss']) * position_size
                else:
                    profit = (signal['entry_price'] - exit_price) * position_size
                    # Expected profit if target is hit
                    expected_profit = (signal['entry_price'] - signal['target']) * position_size
                    # Expected loss if stop loss is hit
                    expected_loss = (signal['stop_loss'] - signal['entry_price']) * position_size
                
                profit_pct = (profit / (signal['entry_price'] * position_size)) * 100 if position_size > 0 else 0
                
                # Calculate buy cost and sell cost
                buy_cost = position_size * signal['entry_price']
                sell_cost = position_size * exit_price
                
                # Get entry time
                entry_candle_time = signal['entry_candle'].get('date')
                if isinstance(entry_candle_time, str):
                    try:
                        entry_time = datetime.strptime(entry_candle_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                    except:
                        entry_time = datetime.combine(trade_date, datetime.min.time())
                elif isinstance(entry_candle_time, datetime):
                    entry_time = entry_candle_time
                    # Make timezone-naive if it's timezone-aware
                    if entry_time.tzinfo is not None:
                        entry_time = entry_time.replace(tzinfo=None)
                else:
                    entry_time = datetime.combine(trade_date, datetime.min.time())
                
                trade_result = {
                    'date': trade_date.strftime('%Y-%m-%d'),
                    'direction': signal['direction'],
                    'entry_price': round(signal['entry_price'], 2),
                    'exit_price': round(exit_price, 2),
                    'stop_loss': round(signal['stop_loss'], 2),
                    'target': round(signal['target'], 2),
                    'position_size': int(position_size),  # Ensure integer quantity
                    'buy_cost': round(buy_cost, 2),
                    'sell_cost': round(sell_cost, 2),
                    'profit': round(profit, 2),
                    'profit_pct': round(profit_pct, 2),
                    'expected_profit': round(expected_profit, 2),
                    'expected_loss': round(expected_loss, 2),
                    'entry_time': entry_time.strftime('%H:%M:%S'),
                    'exit_time': exit_time.strftime('%H:%M:%S'),
                    'exit_reason': exit_reason,
                    'ref_high': round(reference_candle['high'], 2),
                    'ref_low': round(reference_candle['low'], 2),
                    'gap_status': gap_status,
                    'gap_percent': round(gap_percent, 2),
                    'prev_day_close': round(prev_day_close, 2) if prev_day_close else None,
                    'selected_option_strike': selected_option_strike,  # Strike price for index backtests
                    'selected_option_type': actual_instrument_type if is_index else None,  # CE/PE for index backtests
                    'selected_option_expiry': selected_option_expiry  # Expiry date for index backtests
                }
                
                results.append(trade_result)
                total_trades += 1
                total_profit += profit
                
                # Send individual trade result
                await websocket.send_json({
                    "type": "result",
                    "data": trade_result
                })
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"Error processing {trade_date}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate summary (always send summary, even if no trades)
        profitable_trades = len([r for r in results if r['profit'] > 0])
        losing_trades = len([r for r in results if r['profit'] < 0])
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        summary = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'total_profit': round(total_profit, 2),
            'avg_profit': round(avg_profit, 2),
            'win_rate': round(win_rate, 2),
            'total_days_processed': len(trading_dates)
        }
        
        print(f"[Backtest] Summary: {summary}")
        
        # Send summary
        try:
            await websocket.send_json({
                "type": "summary",
                "data": summary
            })
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[Backtest] Error sending summary: {e}")
        
    except Exception as e:
        print(f"[Backtest] Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Error running backtest: {str(e)}"
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

