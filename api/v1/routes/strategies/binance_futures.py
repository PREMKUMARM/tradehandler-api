"""
Binance Futures Backtest Strategy
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
from datetime import datetime, timedelta

from utils.binance_historical import convert_timeframe_to_binance
from utils.binance_backtest import (
    detect_candlestick_pattern,
    generate_trading_signal,
    process_backtest_data
)
from api.v1.routes.market import get_binance_symbols

router = APIRouter()

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


