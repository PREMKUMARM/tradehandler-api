"""
Strategy-related API endpoints (VWAP backtesting, etc.)
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException
from typing import Optional
import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.kite_utils import get_kite_instance
from core.user_context import get_user_id_from_request
from kiteconnect.exceptions import KiteException

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
                            prev_candle_type = prev_row.get('candle_type', '')
                            
                            # Check for Three Black Crows pattern
                            if prev_candle_type != 'Three Black Crows':
                                return (None, None, None)
                            
                            is_green_candle = close > open_price
                            close_above_vwap = close > vwap
                            high_above_vwap = high > vwap
                            
                            # VWAP proximity check: entry should be within 2% of VWAP
                            vwap_diff_percent = abs(close - vwap) / vwap * 100
                            MAX_VWAP_DISTANCE_PCT = 2.0  # Maximum 2% distance from VWAP
                            
                            if vwap_diff_percent > MAX_VWAP_DISTANCE_PCT:
                                return (None, None, None)  # Entry too far from VWAP
                            
                            high_performance_candle_types = [
                                'Dragonfly Doji', 'Piercing Pattern',
                                'Inverted Hammer', 'Long White Candle'
                            ]
                            current_candle_matches = any(pattern in candle_type for pattern in high_performance_candle_types)
                            
                            if not (is_green_candle and (close_above_vwap or high_above_vwap) and current_candle_matches):
                                return (None, None, None)
                            
                            # Pattern strength validation: Inverted Hammer needs stronger confirmation
                            if 'Inverted Hammer' in candle_type:
                                # For Inverted Hammer, require next candle confirmation
                                if current_idx < len(df) - 1:
                                    next_row = df.loc[current_idx + 1]
                                    next_close = next_row.get('close', 0)
                                    next_open = next_row.get('open', 0)
                                    # Next candle should be bullish and close higher than entry
                                    if not (next_close > next_open and next_close > close):
                                        return (None, None, None)  # Inverted Hammer not confirmed
                                else:
                                    # Can't confirm if it's the last candle
                                    return (None, None, None)
                            
                            # Entry timing check: Avoid entries in first 30 minutes (volatile period)
                            # AI analysis: Entry at 09:50:00 (shortly after market open) was problematic
                            # For backtest, we'll use index position as proxy
                            # Skip if it's one of the first 6 candles (assuming 5min candles = first 30 mins)
                            # This gives market time to stabilize after opening volatility
                            if current_idx < 6:
                                return (None, None, None)
                            
                            # Confirmation requirement: Check if previous reversal pattern is strong
                            # Look at the candle before Three Black Crows to ensure proper context
                            if current_idx >= 3:
                                prev_prev_row = df.loc[current_idx - 2]
                                # Ensure we're not entering during a strong downtrend
                                if prev_prev_row.get('close', 0) < prev_row.get('close', 0):
                                    # Still in downtrend, need stronger confirmation
                                    if 'Inverted Hammer' in candle_type:
                                        return (None, None, None)  # Inverted Hammer too weak in strong downtrend
                            
                            # AI Recommendation: Additional confirmation from RSI and MACD
                            # RSI check (needs 14 candles) - run earlier than MACD
                            if current_idx >= 14:
                                current_rsi = row.get('rsi', 50)
                                prev_rsi = prev_row.get('rsi', 50) if current_idx > 0 else 50
                                
                                # RSI confirmation: Stricter overbought check (AI analysis showed RSI 79.809)
                                # For reversal patterns, RSI should be in neutral to slightly oversold range
                                if pd.notna(current_rsi) and pd.notna(prev_rsi):
                                    # Reject if RSI is overbought (> 65) - stricter than before (was 70)
                                    # RSI > 65 indicates strong upward momentum that may reverse
                                    if current_rsi > 65:  # Overbought - avoid entry (stricter threshold)
                                        return (None, None, None)
                                    # Prefer RSI in neutral range (30-60) or recovering from oversold
                                    if current_rsi < prev_rsi and current_rsi < 40:
                                        # RSI declining and oversold - wait for confirmation
                                        return (None, None, None)
                                    # Additional check: If RSI is very high (> 60), require it to be turning down
                                    # This prevents entries at the peak of momentum
                                    if current_rsi > 60 and current_rsi > prev_rsi:
                                        # RSI high and still rising - likely overbought, avoid entry
                                        return (None, None, None)
                                    # AI Recommendation: RSI should be moving out of oversold/overbought territory
                                    # For reversal patterns, prefer RSI that's recovering from oversold (< 40)
                                    # If RSI is oversold (< 30), require it to be turning up (recovering)
                                    if current_rsi < 30:
                                        # Very oversold - require RSI to be turning up for reversal confirmation
                                        if current_rsi <= prev_rsi:
                                            # RSI still declining or flat - wait for recovery
                                            return (None, None, None)
                            
                            # MACD confirmation (needs 26 candles) - additional layer of confirmation
                            if current_idx >= 26:
                                current_macd = row.get('macd', 0)
                                current_macd_signal = row.get('macd_signal', 0)
                                prev_macd = prev_row.get('macd', 0) if current_idx > 0 else 0
                                prev_macd_signal = prev_row.get('macd_signal', 0) if current_idx > 0 else 0
                                
                                if pd.notna(current_macd) and pd.notna(current_macd_signal):
                                    # MACD should be above signal (bullish) or crossing above
                                    macd_bullish = current_macd > current_macd_signal
                                    macd_crossing = (prev_macd <= prev_macd_signal) and (current_macd > current_macd_signal)
                                    
                                    if not (macd_bullish or macd_crossing):
                                        # MACD not confirming - skip trade
                                        return (None, None, None)
                            
                            # AI Recommendation: Volume confirmation for reversal patterns
                            # High volume confirms the strength of the reversal signal
                            if current_idx >= 1:
                                current_volume = row.get('volume', 0)
                                # Calculate average volume over last 5 candles (if available)
                                if current_idx >= 5:
                                    recent_volumes = [df.loc[current_idx - i].get('volume', current_volume) for i in range(5)]
                                    avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else current_volume
                                    
                                    # Require current volume to be at least 80% of average volume
                                    # Low volume suggests weak conviction in the reversal
                                    if avg_volume > 0 and current_volume < avg_volume * 0.8:
                                        # Volume too low - pattern may not be reliable
                                        # For weaker patterns, be more strict
                                        if 'Inverted Hammer' in candle_type or 'Piercing Pattern' in candle_type:
                                            return (None, None, None)  # Weak patterns need volume confirmation
                                elif current_volume == 0:
                                    # No volume data - skip trade
                                    return (None, None, None)
                            
                            # Check for significant price movement before entry (AI recommendation)
                            # If price moved significantly in recent candles, may indicate exhaustion
                            if current_idx >= 3:
                                # Check price change in last 3 candles
                                price_3_candles_ago = df.loc[current_idx - 3].get('close', close) if current_idx >= 3 else close
                                price_change_pct = abs(close - price_3_candles_ago) / price_3_candles_ago * 100
                                
                                # If price moved more than 2% in last 3 candles, be cautious
                                if price_change_pct > 2.0:
                                    # Significant movement - require stronger confirmation
                                    # For weaker patterns, reject if price moved too much
                                    if 'Inverted Hammer' in candle_type or 'Piercing Pattern' in candle_type:
                                        return (None, None, None)  # Weaker patterns need more stability
                                    # For stronger patterns, still allow but with caution
                                    # (Long White Candle is strong, so we allow it but note the risk)
                            
                            # Pattern reliability in context: Check recent price action
                            # Look at last 5 candles to assess market sentiment
                            if current_idx >= 5:
                                recent_closes = [df.loc[current_idx - i].get('close', close) for i in range(6)]
                                # Check if price is in a strong downtrend (declining closes)
                                declining_count = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] < recent_closes[i-1])
                                if declining_count >= 4:  # 4 out of 5 recent candles declining
                                    # Strong downtrend - pattern may not be reliable
                                    if 'Inverted Hammer' in candle_type or 'Piercing Pattern' in candle_type:
                                        # Weaker patterns need stronger confirmation in downtrend
                                        return (None, None, None)
                            
                            matched_pattern = next((p for p in high_performance_candle_types if p in candle_type), candle_type)
                            reason = f"Priority 1: {matched_pattern} candle {'closing' if close_above_vwap else 'high'} above VWAP after Three Black Crows (VWAP: {vwap_diff_percent:.2f}%)"
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
                            
                            # Stop-loss: 1.5% below entry price (tighter as per AI recommendation)
                            STOP_LOSS_PCT = 1.5
                            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT / 100.0)
                            
                            # Trailing stop: Exit if price drops 1% from highest point after entry
                            TRAILING_STOP_PCT = 1.0
                            
                            # Track highest price after entry for trailing stop
                            highest_after_entry = entry_price
                            exit_reason = "End of day"
                            
                            # Check price action after entry for early exit
                            for check_idx in range(entry_idx + 1, len(df)):
                                check_row = df.loc[check_idx]
                                check_close = float(check_row.get('close', entry_price))
                                check_low = float(check_row.get('low', entry_price))
                                
                                # Update highest price after entry
                                if check_close > highest_after_entry:
                                    highest_after_entry = check_close
                                
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
                                
                                # Early exit if price doesn't confirm (price drops below entry after 2+ candles)
                                if check_idx >= entry_idx + 2:
                                    if check_close < entry_price * 0.995:  # 0.5% below entry
                                        exit_price = check_close
                                        exit_idx = check_idx
                                        exit_reason = "Early exit: Price not confirming trade"
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

