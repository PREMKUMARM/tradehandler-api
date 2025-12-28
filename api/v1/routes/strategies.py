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
        
        # Send start message
        await websocket.send_json({
            "type": "start",
            "message": "Backtest started",
            "params": {"start_date": start_date, "end_date": end_date, "timeframe": timeframe}
        })
        # Yield control to event loop to ensure message is sent
        await asyncio.sleep(0.01)
        
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
        
        # Process each instrument and stream results
        for inst_idx, inst in enumerate(instruments):
            try:
                # Send progress update
                await websocket.send_json({
                    "type": "progress",
                    "instrument": inst["name"],
                    "progress": f"{inst_idx + 1}/{len(instruments)}",
                    "message": f"Processing {inst['name']}..."
                })
                # Yield control to event loop to ensure message is sent immediately
                await asyncio.sleep(0.05)
                
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
                            
                            # Format exit time (last candle timestamp)
                            exit_timestamp = df.iloc[-1]['timestamp']
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
                                "exit_price": float(round(float(current_price), 2)),
                                "qty": 1,
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
                
                # Stream result immediately (only if has signals)
                if len(instrument_signals) > 0:
                    print(f"[WS] Sending result for {inst['name']}: {len(instrument_signals)} signals, profit: {instrument_profit}")
                    await websocket.send_json({
                        "type": "result",
                        "data": result
                    })
                    # Yield control to event loop to ensure message is sent immediately
                    # This allows the WebSocket to flush the message before continuing
                    await asyncio.sleep(0.05)
                    print(f"[WS] Result sent for {inst['name']}")
                else:
                    print(f"[WS] Skipping {inst['name']} - no signals")
                    
            except Exception as e:
                print(f"Error processing instrument {inst['name']}: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "instrument": inst["name"],
                    "message": f"Error processing {inst['name']}: {str(e)}"
                })
                continue
        
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

