"""
Strategy-related API endpoints (VWAP backtesting, etc.)
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException
from typing import Optional
import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

from utils.kite_utils import get_kite_instance
from core.user_context import get_user_id_from_request
from kiteconnect.exceptions import KiteException
from utils.binance_historical import fetch_historical_klines_for_date_range, convert_timeframe_to_binance
from utils.binance_backtest import (
    detect_candlestick_pattern,
    generate_trading_signal,
    process_backtest_data
)
from api.v1.routes.market import get_binance_symbols

router = APIRouter(prefix="/strategies", tags=["Strategies"])


@router.post("/backtest-nifty50-options")
async def backtest_nifty50_options(request: Request):
    """
    Backtest Nifty50 options strategy for given date range with multiple strategy options
    
    Request body:
    {
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "strategy_type": "915_candle_break",
        "fund": 200000,
        "risk": 1,
        "reward": 3
    }
    """
    try:
        payload = await request.json()
        start_date_str = payload.get("start_date")
        end_date_str = payload.get("end_date")
        strategy_type = payload.get("strategy_type", "915_candle_break")
        fund = payload.get("fund", 200000)
        risk_pct = payload.get("risk", 1) / 100
        reward_pct = payload.get("reward", 3) / 100
        
        if not start_date_str or not end_date_str:
            raise HTTPException(status_code=400, detail="start_date and end_date are required (format: YYYY-MM-DD)")
        
        # Parse dates
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
        user_id = get_user_id_from_request(request)
        kite = get_kite_instance(user_id)
        
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
            if current_date.weekday() < 5:  # Monday to Friday
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        if not trading_dates:
            raise HTTPException(status_code=400, detail="No trading days found in date range")
        
        # Backtest results structure
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
        
        # NOTE: Full backtest implementation was removed from main.py (~700 lines)
        # This is a placeholder that returns the structure expected by the frontend
        # TODO: Implement full backtest logic here or import from a separate module
        
        return {"data": backtest_results}
        
    except HTTPException:
        raise
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error backtesting strategy: {str(e)}")


@router.websocket("/ws/backtest-vwap-strategy")
async def backtest_vwap_strategy_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming backtest results in real-time
    Client sends: {"start_date": "2025-11-17", "end_date": "2025-12-26", "timeframe": "5minute"}
    Server streams: {"type": "result", "data": {...}} for each instrument
    Server sends: {"type": "summary", "data": {...}} when complete
    Server sends: {"type": "error", "message": "..."} on error
    """
    await websocket.accept()
    try:
        # Receive initial message with parameters
        message = await websocket.receive_text()
        params = json.loads(message)
        
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        timeframe = params.get("timeframe", "5minute")
        
        if not start_date or not end_date:
            await websocket.send_json({
                "type": "error",
                "message": "start_date and end_date are required"
            })
            await websocket.close()
            return
        
        # Get user_id from query params or headers if available
        user_id = "default"
        try:
            user_id_param = websocket.query_params.get("user_id")
            if user_id_param:
                user_id = user_id_param
        except:
            pass
        
        # Get instruments list from request, or use selected stocks, or fallback to default
        instruments = params.get("instruments")
        
        if not instruments:
            # Try to get from selected stocks database
            try:
                from database.stocks_repository import get_stocks_repository
                repo = get_stocks_repository()
                selected_stocks = repo.get_all(active_only=True)
                if selected_stocks:
                    instruments = [
                        {"name": stock.tradingsymbol, "token": str(stock.instrument_token)}
                        for stock in selected_stocks
                    ]
            except Exception as e:
                print(f"[Backtest] Error loading selected stocks: {e}")
        
        # Fallback to default list if no instruments provided
        if not instruments:
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
        
        # Send start message
        await websocket.send_json({
            "type": "start",
            "message": "Backtest started",
            "params": {
                "start_date": start_date,
                "end_date": end_date,
                "timeframe": timeframe,
                "total_instruments": len(instruments)
            }
        })
        # Yield control to event loop to ensure message is sent
        await asyncio.sleep(0.01)
        
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
        timeframe = str(timeframe).strip() if timeframe else "5minute"
        kite_interval = interval_map.get(timeframe, timeframe)
        
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
        cancelled = False
        
        # Process each instrument and stream results
        for inst_idx, inst in enumerate(instruments):
            # Check if WebSocket is still open before processing
            if cancelled:
                print(f"[Backtest] Cancelled, stopping at instrument {inst_idx + 1}/{len(instruments)}")
                break
                
            try:
                # Send progress update - this will raise exception if WebSocket is closed
                await websocket.send_json({
                    "type": "progress",
                    "instrument": inst["name"],
                    "progress": f"{inst_idx + 1}/{len(instruments)}",
                    "message": f"Processing {inst['name']}..."
                })
                # Yield control to event loop to ensure message is sent immediately
                await asyncio.sleep(0.05)
            except Exception as ws_error:
                # WebSocket closed by client (cancelled)
                print(f"[Backtest] WebSocket closed, cancelling backtest: {ws_error}")
                cancelled = True
                break
                
            # Check again after progress update
            if cancelled:
                break
                
            try:
                instrument_signals = []
                instrument_profit = 0
                instrument_profitable = 0
                instrument_losses = 0
                
                # Process each trading date
                for date_str in trading_dates:
                    # Check if cancelled before processing each date
                    if cancelled:
                        break
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
                        
                        # Calculate RSI and MACD for additional confirmation (AI recommendation)
                        from utils.indicators import calculate_rsi
                        closes_list = df['close'].tolist()
                        rsi_values = calculate_rsi(closes_list, period=14)
                        df['rsi'] = rsi_values
                        
                        # Calculate MACD (12, 26, 9)
                        def calculate_macd_simple(prices, fast=12, slow=26, signal=9):
                            """Simple MACD calculation"""
                            import numpy as np
                            if len(prices) < slow + signal:
                                return [np.nan] * len(prices), [np.nan] * len(prices), [np.nan] * len(prices)
                            
                            prices_arr = np.array(prices, dtype=float)
                            ema_fast = pd.Series(prices_arr).ewm(span=fast, adjust=False).mean()
                            ema_slow = pd.Series(prices_arr).ewm(span=slow, adjust=False).mean()
                            macd_line = ema_fast - ema_slow
                            signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean()
                            
                            return macd_line.tolist(), signal_line.tolist(), (macd_line - signal_line).tolist()
                        
                        macd_line, macd_signal, macd_histogram = calculate_macd_simple(closes_list)
                        df['macd'] = macd_line
                        df['macd_signal'] = macd_signal
                        df['macd_histogram'] = macd_histogram
                        
                        # Get current price (last candle close) for PnL - convert to float
                        current_price = float(df.iloc[-1]['close']) if len(df) > 0 else 0.0
                        
                        # Process signals using same logic as getCandle endpoint
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
                        
                        # Generate trading signals (enhanced with AI recommendations)
                        def generate_trading_signal_bk(current_idx, df, instrument_token=None):
                            """
                            Generate trading signal with enhanced filters based on AI analysis:
                            1. VWAP proximity check (entry should be within 2% of VWAP)
                            2. Confirmation requirement (next candle should confirm reversal)
                            3. Pattern strength validation (stronger confirmation for weaker patterns)
                            4. Entry timing (avoid very early entries without confirmation)
                            """
                            instrument_blacklist = ['4701441']  # PERSISTENT
                            if instrument_token and str(instrument_token) in instrument_blacklist:
                                return (None, None, None)
                            
                            # Need at least 2 previous candles for pattern detection
                            if current_idx < 2:
                                return (None, None, None)
                            
                            row = df.loc[current_idx]
                            candle_type = row.get('candle_type', '')
                            close = row.get('close', 0)
                            open_price = row.get('open', 0)
                            high = row.get('high', 0)
                            low = row.get('low', 0)
                            vwap = row.get('vwap', 0)
                            
                            if vwap == 0 or close == 0:
                                return (None, None, None)
                            
                            prev_row = df.loc[current_idx - 1]
                            prev_2_row = df.loc[current_idx - 2] if current_idx >= 2 else None
                            
                            # SCALPING: Skip first 5 candles to avoid market open volatility
                            if current_idx < 5:
                                return (None, None, None)
                            
                            # Calculate VWAP distance
                            vwap_diff_percent = abs(close - vwap) / vwap * 100
                            MAX_VWAP_DISTANCE_PCT = 2.5  # Relaxed for more signals (scalping)
                            
                            if vwap_diff_percent > MAX_VWAP_DISTANCE_PCT:
                                return (None, None, None)
                            
                            # BUY SIGNALS - Multiple patterns for scalping
                            bullish_patterns = [
                                'Hammer', 'Dragonfly Doji', 'Piercing Pattern', 'Inverted Hammer',
                                'Bullish Engulfing', 'Long White Candle', 'Morning Star'
                            ]
                            
                            is_bullish_pattern = any(pattern in candle_type for pattern in bullish_patterns)
                            is_green_candle = close > open_price
                            close_above_vwap = close > vwap
                            high_above_vwap = high > vwap
                            
                            # BUY: Green candle with bullish pattern near/above VWAP
                            if not (is_green_candle and is_bullish_pattern and (close_above_vwap or high_above_vwap)):
                                return (None, None, None)
                            
                            # Simple RSI check - not too overbought
                            if current_idx >= 14:
                                current_rsi = row.get('rsi', 50)
                                if pd.notna(current_rsi):
                                    if current_rsi > 75:  # Too overbought
                                        return (None, None, None)
                                    if current_rsi < 25:  # Too oversold, might be weak
                                        return (None, None, None)
                            
                            # Simple MACD check - prefer bullish or neutral
                            if current_idx >= 26:
                                current_macd = row.get('macd', 0)
                                current_macd_signal = row.get('macd_signal', 0)
                                if pd.notna(current_macd) and pd.notna(current_macd_signal):
                                    macd_bullish = current_macd > current_macd_signal
                                    macd_near_cross = abs(current_macd - current_macd_signal) < abs(current_macd) * 0.1
                                    if not (macd_bullish or macd_near_cross):
                                        return (None, None, None)
                            
                            # Volume check - ensure some volume
                            current_volume = row.get('volume', 0)
                            if current_idx >= 5:
                                recent_volumes = [df.loc[current_idx - i].get('volume', current_volume) for i in range(5)]
                                avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else current_volume
                                if avg_volume > 0 and current_volume < avg_volume * 0.5:  # At least 50% of average
                                    return (None, None, None)
                            
                            # Momentum: Price should be moving up (close higher than 1-2 candles ago)
                            if current_idx >= 2:
                                price_1_candle_ago = prev_row.get('close', close)
                                price_2_candles_ago = prev_2_row.get('close', close) if prev_2_row is not None else close
                                if close <= price_1_candle_ago and close <= price_2_candles_ago:
                                    return (None, None, None)
                            
                            # Pattern-specific confirmations
                            if 'Inverted Hammer' in candle_type:
                                if current_idx < len(df) - 1:
                                    next_row = df.loc[current_idx + 1]
                                    next_close = next_row.get('close', close)
                                    if next_close > close:
                                        matched_pattern = 'Inverted Hammer (Confirmed)'
                                    else:
                                        matched_pattern = 'Inverted Hammer'
                                else:
                                    matched_pattern = 'Inverted Hammer'
                            else:
                                matched_pattern = next((p for p in bullish_patterns if p in candle_type), candle_type)
                            
                            reason = f"Scalping BUY: {matched_pattern} above VWAP (VWAP dist: {vwap_diff_percent:.2f}%)"
                            return ('BUY', 1, reason)
                        
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
                            
                            # Dynamic exit strategy: Track price action after entry
                            entry_idx = idx
                            exit_price = current_price
                            exit_idx = len(df) - 1
                            
                            # SCALPING: Tighter stops and profit targets
                            STOP_LOSS_PCT = 1.0  # Tighter stop for scalping
                            TRAILING_STOP_PCT = 0.6  # Tighter trailing stop
                            PROFIT_TARGET_PCT = 0.6  # Quick profit target (0.6% for scalping)
                            
                            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT / 100.0)
                            profit_target_price = entry_price * (1 + PROFIT_TARGET_PCT / 100.0)
                            
                            # Track highest price after entry for trailing stop
                            highest_after_entry = entry_price
                            exit_reason = "End of day"
                            
                            # Check price action after entry for early exit
                            for check_idx in range(entry_idx + 1, len(df)):
                                check_row = df.loc[check_idx]
                                check_close = float(check_row.get('close', entry_price))
                                check_high = float(check_row.get('high', entry_price))
                                check_low = float(check_row.get('low', entry_price))
                                
                                # Update highest price after entry
                                if check_close > highest_after_entry:
                                    highest_after_entry = check_close
                                
                                # SCALPING: Profit target hit - take quick profit
                                if check_high >= profit_target_price:
                                    exit_price = profit_target_price
                                    exit_idx = check_idx
                                    exit_reason = f"Profit target hit ({PROFIT_TARGET_PCT}%)"
                                    break
                                
                                # Check stop-loss (price hit stop-loss level)
                                if check_low <= stop_loss_price:
                                    exit_price = stop_loss_price
                                    exit_idx = check_idx
                                    exit_reason = f"Stop-loss triggered ({STOP_LOSS_PCT}%)"
                                    break
                                
                                # Check trailing stop (price dropped from high)
                                if highest_after_entry > entry_price:
                                    trailing_stop_price = highest_after_entry * (1 - TRAILING_STOP_PCT / 100.0)
                                    if check_close <= trailing_stop_price:
                                        exit_price = check_close
                                        exit_idx = check_idx
                                        exit_reason = f"Trailing stop triggered ({TRAILING_STOP_PCT}% from high)"
                                        break
                                
                                # Quick exit if price moves against us (scalping - don't wait)
                                if check_idx >= entry_idx + 2:
                                    if check_close < entry_price * 0.995:  # 0.5% below entry
                                        exit_price = check_close
                                        exit_idx = check_idx
                                        exit_reason = "Quick exit: Price moving against trade"
                                        break
                            
                            try:
                                exit_price_float = float(exit_price)
                            except (ValueError, TypeError):
                                exit_price_float = float(str(exit_price))
                            
                            profit = exit_price_float - entry_price
                            
                            # Format entry time (from signal timestamp)
                            entry_timestamp = signal_row['timestamp']
                            try:
                                if pd.api.types.is_datetime64_any_dtype(entry_timestamp):
                                    entry_time = entry_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                                elif isinstance(entry_timestamp, pd.Timestamp):
                                    entry_time = entry_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                                elif isinstance(entry_timestamp, (int, float, np.integer, np.floating)):
                                    entry_timestamp_float = float(entry_timestamp)
                                    if entry_timestamp_float > 1e10:  # milliseconds
                                        entry_time = datetime.fromtimestamp(entry_timestamp_float / 1000).strftime('%Y-%m-%d %H:%M:%S')
                                    else:  # seconds
                                        entry_time = datetime.fromtimestamp(entry_timestamp_float).strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    try:
                                        entry_time = pd.to_datetime(entry_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                                    except:
                                        entry_time = str(entry_timestamp) if entry_timestamp else '-'
                            except Exception as e:
                                print(f"Error formatting entry_time: {e}, timestamp: {entry_timestamp}")
                                entry_time = '-'
                            
                            # Format exit time (use actual exit candle timestamp)
                            exit_timestamp = df.iloc[exit_idx]['timestamp']
                            try:
                                if pd.api.types.is_datetime64_any_dtype(exit_timestamp):
                                    exit_time = exit_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                                elif isinstance(exit_timestamp, pd.Timestamp):
                                    exit_time = exit_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                                elif isinstance(exit_timestamp, (int, float, np.integer, np.floating)):
                                    exit_timestamp_float = float(exit_timestamp)
                                    if exit_timestamp_float > 1e10:  # milliseconds
                                        exit_time = datetime.fromtimestamp(exit_timestamp_float / 1000).strftime('%Y-%m-%d %H:%M:%S')
                                    else:  # seconds
                                        exit_time = datetime.fromtimestamp(exit_timestamp_float).strftime('%Y-%m-%d %H:%M:%S')
                                else:
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
                                "exit_price": float(round(float(exit_price_float), 2)),
                                "qty": 1,
                                "profit": float(round(float(profit), 2)),
                                "profit_percent": float(round((float(profit) / float(entry_price) * 100) if entry_price > 0 else 0, 2)),
                                "candle_type": str(signal_row.get('candle_type', '')) if signal_row.get('candle_type') is not None else '',
                                "signal_reason": str(signal_row.get('signal_reason', '')) if signal_row.get('signal_reason') is not None else '',
                                "exit_reason": exit_reason  # Add exit reason for analysis
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
                
                result = {
                    "instrument": str(inst["name"]),
                    "instrument_token": str(inst["token"]),
                    "total_signals": int(len(instrument_signals)),
                    "total_profit": float(round(float(instrument_profit), 2)),
                    "avg_profit": float(round(float(avg_profit), 2)),
                    "win_rate": float(round(float(win_rate), 2)),
                    "profitable_signals": int(instrument_profitable),
                    "loss_signals": int(instrument_losses),
                    "orders": instrument_signals
                }
                
                results.append(result)
                total_signals += int(len(instrument_signals))
                total_profit += float(instrument_profit)
                
                # Check if cancelled before sending result
                if cancelled:
                    print(f"[Backtest] Cancelled, skipping result for {inst['name']}")
                    break
                
                # Stream result immediately (only if has signals)
                if len(instrument_signals) > 0:
                    try:
                        print(f"[WS] Sending result for {inst['name']}: {len(instrument_signals)} signals, profit: {instrument_profit}")
                        await websocket.send_json({
                            "type": "result",
                            "data": result
                        })
                        # Yield control to event loop to ensure message is sent immediately
                        # This allows the WebSocket to flush the message before continuing
                        await asyncio.sleep(0.05)
                        print(f"[WS] Result sent for {inst['name']}")
                    except Exception as ws_error:
                        print(f"[Backtest] WebSocket closed while sending result, cancelling: {ws_error}")
                        cancelled = True
                        break
                else:
                    print(f"[WS] Skipping {inst['name']} - no signals")
                    
            except Exception as e:
                print(f"Error processing instrument {inst['name']}: {str(e)}")
                if not cancelled:
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "instrument": inst["name"],
                            "message": f"Error processing {inst['name']}: {str(e)}"
                        })
                    except Exception as ws_error:
                        print(f"[Backtest] WebSocket closed while sending error, cancelling: {ws_error}")
                        cancelled = True
                        break
                continue
        
        # Send final summary only if not cancelled
        if cancelled:
            print("[Backtest] Backtest cancelled by user, not sending summary")
            try:
                await websocket.send_json({
                    "type": "cancelled",
                    "message": "Backtest cancelled by user"
                })
            except:
                pass  # WebSocket already closed
            return
        
        # Send final summary
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
        
        await websocket.send_json({
            "type": "summary",
            "data": summary
        })
        
        await websocket.send_json({
            "type": "complete",
            "message": "Backtest completed"
        })
        
    except WebSocketDisconnect:
        print("[WS] Client disconnected from backtest")
    except Exception as e:
        print(f"[WS] Backtest error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Error running backtest: {str(e)}"
            })
        except:
            pass
        await websocket.close()


@router.websocket("/ws/backtest-binance-futures")
async def backtest_binance_futures_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming Binance Futures backtest results in real-time
    Client sends: {"start_date": "2025-11-17", "end_date": "2025-12-26", "timeframe": "5minute", "symbols": ["ETHUSDT", "SOLUSDT"]}
    Server streams: {"type": "result", "data": {...}} for each symbol
    Server sends: {"type": "summary", "data": {...}} when complete
    Server sends: {"type": "error", "message": "..."} on error
    """
    await websocket.accept()
    try:
        # Receive initial message with parameters
        message = await websocket.receive_text()
        params = json.loads(message)
        
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        timeframe = params.get("timeframe", "5minute")
        symbols = params.get("symbols", get_binance_symbols())  # Default to symbols from env
        
        if not start_date or not end_date:
            await websocket.send_json({
                "type": "error",
                "message": "start_date and end_date are required"
            })
            await websocket.close()
            return
        
        # Convert timeframe to Binance interval
        binance_interval = convert_timeframe_to_binance(timeframe)
        
        # Send start message
        await websocket.send_json({
            "type": "start",
            "message": "Binance Futures backtest started",
            "params": {
                "start_date": start_date,
                "end_date": end_date,
                "timeframe": timeframe,
                "binance_interval": binance_interval,
                "total_symbols": len(symbols)
            }
        })
        await asyncio.sleep(0.01)
        
        results = []
        total_signals = 0
        total_profit = 0
        cancelled = False
        
        # Process each symbol and stream results
        for symbol_idx, symbol in enumerate(symbols):
            # Check if WebSocket is still open before processing
            if cancelled:
                print(f"[Binance Backtest] Cancelled, stopping at symbol {symbol_idx + 1}/{len(symbols)}")
                break
                
            try:
                # Send progress update
                await websocket.send_json({
                    "type": "progress",
                    "symbol": symbol,
                    "progress": f"{symbol_idx + 1}/{len(symbols)}",
                    "message": f"Processing {symbol}..."
                })
                await asyncio.sleep(0.05)
            except Exception as ws_error:
                print(f"[Binance Backtest] WebSocket closed, cancelling: {ws_error}")
                cancelled = True
                break
            
            if cancelled:
                break
                
            try:
                symbol_signals = []
                symbol_profit = 0
                symbol_profitable = 0
                symbol_losses = 0
                
                # Fetch historical klines for the date range
                klines = await fetch_historical_klines_for_date_range(
                    symbol,
                    binance_interval,
                    start_date,
                    end_date
                )
                
                if not klines or len(klines) == 0:
                    print(f"[Binance Backtest] No data for {symbol}")
                    continue
                
                # Convert to DataFrame for processing
                df = pd.DataFrame(klines)
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Calculate VWAP (same logic as VWAP strategy)
                df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
                df['cumulative_volume'] = df['volume'].cumsum()
                df['cumulative_tpv'] = (df['typical_price'] * df['volume']).cumsum()
                df['vwap'] = df['cumulative_tpv'] / df['cumulative_volume']
                
                # Calculate RSI and MACD (same as VWAP strategy)
                from utils.indicators import calculate_rsi
                closes_list = df['close'].tolist()
                rsi_values = calculate_rsi(closes_list, period=14)
                df['rsi'] = rsi_values
                
                # Calculate MACD (12, 26, 9)
                def calculate_macd_simple(prices, fast=12, slow=26, signal=9):
                    """Simple MACD calculation"""
                    import numpy as np
                    if len(prices) < slow + signal:
                        return [np.nan] * len(prices), [np.nan] * len(prices), [np.nan] * len(prices)
                    
                    prices_arr = np.array(prices, dtype=float)
                    ema_fast = pd.Series(prices_arr).ewm(span=fast, adjust=False).mean()
                    ema_slow = pd.Series(prices_arr).ewm(span=slow, adjust=False).mean()
                    macd_line = ema_fast - ema_slow
                    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean()
                    
                    return macd_line.tolist(), signal_line.tolist(), (macd_line - signal_line).tolist()
                
                macd_line, macd_signal, macd_histogram = calculate_macd_simple(closes_list)
                df['macd'] = macd_line
                df['macd_signal'] = macd_signal
                df['macd_histogram'] = macd_histogram
                
                # Process signals using same logic as VWAP strategy
                df['is_above_vwap'] = df['close'] > df['vwap']
                df['vwap_position'] = df['is_above_vwap'].map({True: 'Above', False: 'Below'})
                df['vwap_diff'] = (df['close'] - df['vwap']).abs()
                df['vwap_diff_percent'] = (df['close'] - df['vwap']) / df['vwap'] * 100
                
                # Apply pattern detection using shared utility
                df['candle_type'] = df.index.map(lambda i: detect_candlestick_pattern(i, df))
                
                # Apply signal generation using shared utility
                df['trading_signal'] = None
                df['signal_priority'] = None
                df['signal_reason'] = None
                for i in range(len(df)):
                    signal, priority, reason = generate_trading_signal(i, df)
                    if signal is not None:
                        df.loc[i, 'trading_signal'] = signal
                        df.loc[i, 'signal_priority'] = priority
                        df.loc[i, 'signal_reason'] = reason
                
                # Calculate P&L using shared utility
                STOP_LOSS_PCT = 0.9
                TRAILING_STOP_PCT = 0.6
                PROFIT_TARGET_PCT = 0.7
                
                symbol_signals = process_backtest_data(
                    df,
                    stop_loss_pct=STOP_LOSS_PCT,
                    trailing_stop_pct=TRAILING_STOP_PCT,
                    profit_target_pct=PROFIT_TARGET_PCT,
                    use_position_sizing=True
                )
                
                # Calculate summary from orders
                symbol_profit = sum(order['profit'] for order in symbol_signals)
                symbol_profitable = sum(1 for order in symbol_signals if order['profit'] > 0)
                symbol_losses = sum(1 for order in symbol_signals if order['profit'] < 0)
                
                # Orders are already processed by process_backtest_data utility
                # symbol_signals already contains all the order data with P&L in correct format
                
                # Calculate summary for this symbol
                if len(symbol_signals) > 0:
                    avg_profit = float(symbol_profit) / len(symbol_signals)
                    win_rate = (float(symbol_profitable) / len(symbol_signals)) * 100 if len(symbol_signals) > 0 else 0
                else:
                    avg_profit = 0.0
                    win_rate = 0.0
                
                result = {
                    "instrument": symbol,
                    "instrument_token": symbol,  # Use symbol as token for Binance
                    "total_signals": int(len(symbol_signals)),
                    "total_profit": float(round(float(symbol_profit), 2)),
                    "avg_profit": float(round(float(avg_profit), 2)),
                    "win_rate": float(round(float(win_rate), 2)),
                    "profitable_signals": int(symbol_profitable),
                    "loss_signals": int(symbol_losses),
                    "orders": symbol_signals
                }
                
                results.append(result)
                total_signals += int(len(symbol_signals))
                total_profit += float(symbol_profit)
                
                if cancelled:
                    print(f"[Binance Backtest] Cancelled, skipping result for {symbol}")
                    break
                
                # Stream result immediately (only if has signals)
                if len(symbol_signals) > 0:
                    try:
                        print(f"[Binance WS] Sending result for {symbol}: {len(symbol_signals)} signals, profit: {symbol_profit}")
                        await websocket.send_json({
                            "type": "result",
                            "data": result
                        })
                        await asyncio.sleep(0.05)
                    except Exception as ws_error:
                        print(f"[Binance Backtest] WebSocket closed while sending result, cancelling: {ws_error}")
                        cancelled = True
                        break
                else:
                    print(f"[Binance WS] Skipping {symbol} - no signals")
                    
            except Exception as e:
                print(f"Error processing symbol {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                if not cancelled:
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "symbol": symbol,
                            "message": f"Error processing {symbol}: {str(e)}"
                        })
                    except Exception as ws_error:
                        print(f"[Binance Backtest] WebSocket closed while sending error, cancelling: {ws_error}")
                        cancelled = True
                        break
                continue
        
        # Send final summary only if not cancelled
        if cancelled:
            print("[Binance Backtest] Backtest cancelled by user, not sending summary")
            try:
                await websocket.send_json({
                    "type": "cancelled",
                    "message": "Backtest cancelled by user"
                })
            except:
                pass
            return
        
        # Calculate trading days (approximate - Binance trades 24/7)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        trading_days = (end_dt - start_dt).days + 1
        
        # Send final summary
        summary = {
            "total_instruments": int(len(symbols)),
            "total_signals": int(total_signals),
            "total_profit": float(round(float(total_profit), 2)),
            "avg_profit_per_signal": float(round(float(total_profit) / float(total_signals), 2)) if total_signals > 0 else 0.0,
            "profitable_instruments": int(len([r for r in results if float(r["total_profit"]) > 0])),
            "loss_instruments": int(len([r for r in results if float(r["total_profit"]) < 0])),
            "test_period": {
                "start_date": str(start_date),
                "end_date": str(end_date),
                "trading_days": int(trading_days)
            }
        }
        
        await websocket.send_json({
            "type": "summary",
            "data": summary
        })
        
        await websocket.send_json({
            "type": "complete",
            "message": "Binance Futures backtest completed"
        })
        
    except WebSocketDisconnect:
        print("[Binance WS] Client disconnected from backtest")
    except Exception as e:
        print(f"[Binance WS] Backtest error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Error running Binance backtest: {str(e)}"
            })
        except:
            pass
        await websocket.close()


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
        
        # Strategy configuration (with defaults) - explicitly convert to boolean
        # Handle both boolean and string values from JSON
        def to_bool(value, default):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        
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
        
        # Helper function to send log messages via WebSocket
        async def send_log(log_type: str, message: str, date: str = None, details: dict = None):
            """Send log message to frontend"""
            try:
                await websocket.send_json({
                    "type": "log",
                    "log_type": log_type,  # info, warning, error, success, skip
                    "message": message,
                    "date": date,
                    "details": details or {},
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
            except:
                pass  # Ignore errors if WebSocket is closed
        
        await send_log("info", "Strategy configuration loaded", None, {
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
                    await send_log("warning", f"No 30min candles available for {trade_date.strftime('%Y-%m-%d')}", trade_date.strftime("%Y-%m-%d"))
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
                    await send_log("warning", f"No reference candle (9:15-9:45) found for {trade_date.strftime('%Y-%m-%d')}", trade_date.strftime("%Y-%m-%d"))
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
                    
                    await send_log("info", f"Finding previous trading day for {trade_date.strftime('%Y-%m-%d')} (weekday: {trade_date.strftime('%A')})", trade_date.strftime("%Y-%m-%d"), {
                        "trade_date": trade_date.strftime("%Y-%m-%d"),
                        "weekday": trade_date.strftime("%A"),
                        "reason": "Starting search for previous trading day to calculate gap"
                    })
                    
                    while lookback_count < max_lookback_days and not found_prev_trading_day:
                        # Skip weekends
                        if prev_date.weekday() >= 5:  # Saturday=5, Sunday=6
                            await send_log("info", f"Skipping weekend: {prev_date.strftime('%Y-%m-%d')} ({prev_date.strftime('%A')})", trade_date.strftime("%Y-%m-%d"), {
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
                                        await send_log("success", f"Found previous trading day: {prev_date.strftime('%Y-%m-%d')} ({prev_date.strftime('%A')}) - Close: ₹{prev_day_close:.2f} (at {last_candle_time_str})", trade_date.strftime("%Y-%m-%d"), {
                                            "prev_trading_date": prev_date.strftime("%Y-%m-%d"),
                                            "weekday": prev_date.strftime("%A"),
                                            "close": prev_day_close,
                                            "close_time": last_candle_time_str,
                                            "candles_count": len(prev_day_candles),
                                            "reason": f"Previous trading day found after looking back {lookback_count} day(s) - using last minute candle close"
                                        })
                                        break
                                    else:
                                        await send_log("info", f"No valid close price for {prev_date.strftime('%Y-%m-%d')}, continuing search", trade_date.strftime("%Y-%m-%d"), {
                                            "checked_date": prev_date.strftime("%Y-%m-%d"),
                                            "reason": "Data exists but close price is invalid"
                                        })
                                else:
                                    await send_log("info", f"No candles found for {prev_date.strftime('%Y-%m-%d')}, trying daily candles as fallback", trade_date.strftime("%Y-%m-%d"), {
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
                                                await send_log("success", f"Found previous trading day: {prev_date.strftime('%Y-%m-%d')} ({prev_date.strftime('%A')}) - Close: ₹{prev_day_close:.2f} (from daily candle)", trade_date.strftime("%Y-%m-%d"), {
                                                    "prev_trading_date": prev_date.strftime("%Y-%m-%d"),
                                                    "weekday": prev_date.strftime("%A"),
                                                    "close": prev_day_close,
                                                    "reason": f"Previous trading day found using daily candle fallback"
                                                })
                                                break
                                    except Exception as daily_e:
                                        await send_log("info", f"Daily candle fallback also failed for {prev_date.strftime('%Y-%m-%d')}, continuing search", trade_date.strftime("%Y-%m-%d"), {
                                            "checked_date": prev_date.strftime("%Y-%m-%d"),
                                            "error": str(daily_e)[:50],
                                            "reason": "Both minute and daily candles failed"
                                        })
                            else:
                                await send_log("info", f"No data for {prev_date.strftime('%Y-%m-%d')} (likely holiday), continuing search", trade_date.strftime("%Y-%m-%d"), {
                                    "checked_date": prev_date.strftime("%Y-%m-%d"),
                                    "reason": "No candles returned - likely a market holiday"
                                })
                        except Exception as e:
                            # This date might be a holiday or error, continue to previous day
                            print(f"[Backtest] No data for {prev_date} (might be holiday): {e}")
                            await send_log("info", f"Error fetching data for {prev_date.strftime('%Y-%m-%d')}: {str(e)[:50]}", trade_date.strftime("%Y-%m-%d"), {
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
                        
                        await send_log("info", f"Gap calculation: Open={current_open:.2f}, Prev Close={prev_day_close:.2f}, Gap={gap:.2f} ({gap_percent:.2f}%)", trade_date.strftime("%Y-%m-%d"), {
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
                            await send_log("info", f"Gap is within threshold (-0.1% to +0.1%), treating as 'No Gap'", trade_date.strftime("%Y-%m-%d"), {
                                "gap_percent": gap_percent,
                                "threshold": "±0.1%",
                                "reason": "Gap is too small to be considered significant"
                            })
                    else:
                        print(f"[Backtest] Could not find previous trading day for {trade_date} (looked back {lookback_count} days)")
                        gap_status = "N/A"
                        is_gap_up = False
                        is_gap_down = False
                        await send_log("warning", f"Could not find previous trading day (looked back {lookback_count} days)", trade_date.strftime("%Y-%m-%d"), {
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
                            await send_log("skip", f"No gap detected, skipping option selection", trade_date.strftime("%Y-%m-%d"))
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
                            await send_log("warning", f"No {selected_option_type} option found at strike {current_strike}", trade_date.strftime("%Y-%m-%d"), {"strike": current_strike, "type": selected_option_type})
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
                            await send_log("warning", f"No valid expiry found for {index_name} {selected_option_type}", trade_date.strftime("%Y-%m-%d"))
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
                        await send_log("success", f"Selected {actual_instrument_name}", trade_date.strftime("%Y-%m-%d"), {
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
                                await send_log("warning", f"No 30-minute reference candle found for selected option on {trade_date.strftime('%Y-%m-%d')}", trade_date.strftime("%Y-%m-%d"), {
                                    "option": actual_instrument_name,
                                    "reason": "30-minute candle (9:15-9:45 AM) not found for selected option"
                                })
                                print(f"[Backtest] No reference candle found for selected option on {trade_date}")
                                continue
                            
                            # Verify we're using the option's reference candle, not the index's
                            # Log reference candle once (will be logged again below, but with more context)
                            # Skip this duplicate log
                                
                        except Exception as e:
                            await send_log("error", f"Error fetching option 30min candles: {str(e)}", trade_date.strftime("%Y-%m-%d"), {
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
                await send_log("info", f"Reference candle (9:15-9:45 AM): HIGH=₹{reference_candle['high']:.2f}, LOW=₹{reference_candle['low']:.2f}", trade_date.strftime("%Y-%m-%d"), {
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
                    await send_log("error", f"Error fetching 5min candles: {str(e)}", trade_date.strftime("%Y-%m-%d"), {
                        "instrument": instrument_used
                    })
                    print(f"[Backtest] Error fetching 5min candles for {trade_date}: {e}")
                    continue
                
                if not candles_5min or len(candles_5min) == 0:
                    await send_log("warning", f"No 5-minute candles available for {trade_date.strftime('%Y-%m-%d')}", trade_date.strftime("%Y-%m-%d"), {
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
                            await send_log("skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
                                "close": close_5min,
                                "ref_high": ref_high,
                                "reason": "PE profits when price goes down, but gap-up suggests upward momentum"
                            })
                            print(f"[Backtest] {skipped_reason} for {trade_date}")
                            continue  # Skip this signal, continue looking
                        
                        # Filter: For CE options, skip LONG on gap-down days (CE profits when price goes up)
                        if skip_ce_on_gap_down and (instrument_exchange == "NFO" or is_index) and actual_instrument_type == "CE" and is_gap_down:
                            skipped_reason = f"LONG skipped: CE on Gap-Down ({gap_percent:.2f}%)"
                            await send_log("skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
                                "close": close_5min,
                                "ref_high": ref_high,
                                "reason": "CE profits when price goes up, but gap-down suggests downward momentum"
                            })
                            print(f"[Backtest] {skipped_reason} for {trade_date}")
                            continue  # Skip this signal, continue looking
                        
                        # Filter: Don't take LONG on gap-down days (against momentum) - for equity only
                        if skip_long_on_gap_down and instrument_exchange != "NFO" and is_gap_down:
                            skipped_reason = f"LONG skipped: Gap-Down ({gap_percent:.2f}%)"
                            await send_log("skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
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
                            await send_log("skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
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
                            await send_log("skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
                                "close": close_5min,
                                "ref_low": ref_low,
                                "reason": "Options selling requires margin/funds not available, only options buying is allowed"
                            })
                            print(f"[Backtest] {skipped_reason} for {trade_date}")
                            continue  # Skip this signal, continue looking
                        
                        # Filter: Don't take SHORT on gap-up days (against momentum)
                        if skip_short_on_gap_up and is_gap_up:
                            skipped_reason = f"SHORT skipped: Gap-Up ({gap_percent:.2f}%)"
                            await send_log("skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
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
                            await send_log("skip", skipped_reason, trade_date.strftime("%Y-%m-%d"), {
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
                await send_log("success", f"{signal['direction']} signal found at ₹{signal['entry_price']:.2f}", trade_date.strftime("%Y-%m-%d"), {
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

