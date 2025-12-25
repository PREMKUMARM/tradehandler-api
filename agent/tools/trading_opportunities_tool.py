"""
Trading opportunities based on indicators - Dynamic multi-indicator support
"""
from typing import Optional, List, Dict, Any, Union
from langchain_core.tools import tool
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.kite_utils import get_kite_instance
from agent.tools.instrument_resolver import resolve_instrument_name, get_instrument_token
from agent.config import get_agent_config
from agent.tools.risk_tools import suggest_position_size_tool
from kiteconnect.exceptions import KiteException


def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate RSI indicator"""
    import numpy as np
    if len(prices) < period + 1:
        return [np.nan] * len(prices)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gains = np.full(len(gains), np.nan)
    avg_losses = np.full(len(losses), np.nan)
    
    avg_gains[period - 1] = np.mean(gains[:period])
    avg_losses[period - 1] = np.mean(losses[:period])
    
    for i in range(period, len(gains)):
        avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period
    
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return [np.nan] + rsi.tolist()


def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
    """Calculate MACD indicator"""
    import pandas as pd
    if len(prices) < slow:
        return {"macd": [0.0] * len(prices), "signal": [0.0] * len(prices), "histogram": [0.0] * len(prices)}
    
    series = pd.Series(prices)
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        "macd": macd_line.tolist(),
        "signal": signal_line.tolist(),
        "histogram": histogram.tolist()
    }


def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, List[float]]:
    """Calculate Bollinger Bands"""
    import pandas as pd
    if len(prices) < period:
        return {"upper": [0.0] * len(prices), "middle": [0.0] * len(prices), "lower": [0.0] * len(prices)}
    
    series = pd.Series(prices)
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return {
        "upper": upper_band.tolist(),
        "middle": sma.tolist(),
        "lower": lower_band.tolist()
    }


def check_indicator_condition(
    indicator_name: str,
    indicator_data: Any,
    condition: str,
    idx: int,
    prev_idx: Optional[int] = None
) -> bool:
    """Check if indicator condition is met at given index"""
    if indicator_name.upper() == "RSI":
        rsi_values = indicator_data
        if idx >= len(rsi_values) or (prev_idx is not None and prev_idx >= len(rsi_values)):
            return False
            
        current_rsi = rsi_values[idx]
        prev_rsi = rsi_values[prev_idx] if prev_idx is not None else None
        
        if current_rsi is None or (prev_idx is not None and prev_rsi is None):
            return False

        # --- REVERSAL STRATEGY LOGIC ---
        if "reversal" in condition.lower():
            if "oversold" in condition.lower() or "below 30" in condition.lower():
                # Buy when RSI crosses ABOVE 30 from below
                return prev_rsi is not None and prev_rsi < 30.0 and current_rsi >= 30.0
            elif "overbought" in condition.lower() or "above 70" in condition.lower():
                # Sell when RSI crosses BELOW 70 from above
                return prev_rsi is not None and prev_rsi > 70.0 and current_rsi <= 70.0
        
        # --- ORIGINAL ZONE LOGIC ---
        # Professional Default: Use Confirmation logic even if "reversal" not explicitly requested
        # This prevents "falling knife" entries and rapid exits.
        if "oversold" in condition.lower() or "below 30" in condition.lower():
            # Standard Buy: Wait for RSI to cross ABOVE 30 (Confirmation)
            return prev_rsi is not None and prev_rsi < 30.0 and current_rsi >= 30.0
        elif "overbought" in condition.lower() or "above 70" in condition.lower():
            # Standard Sell: Wait for RSI to cross BELOW 70 (Confirmation)
            return prev_rsi is not None and prev_rsi > 70.0 and current_rsi <= 70.0
        elif condition.lower() == "bullish crossover":
            return prev_rsi is not None and prev_rsi < 50 and current_rsi >= 50
        elif condition.lower() == "bearish crossover":
            return prev_rsi is not None and prev_rsi > 50 and current_rsi <= 50
    
    elif indicator_name.upper() == "MACD":
        macd = indicator_data.get("macd", [])
        signal = indicator_data.get("signal", [])
        histogram = indicator_data.get("histogram", [])
        
        if idx >= len(macd) or idx >= len(signal):
            return False
        
        macd_val = macd[idx]
        signal_val = signal[idx]
        hist_val = histogram[idx] if idx < len(histogram) else 0
        
        if condition.lower() in ["bullish crossover", "bullish", "cross above"]:
            if prev_idx is not None and prev_idx < len(macd) and prev_idx < len(signal):
                return macd[prev_idx] <= signal[prev_idx] and macd_val > signal_val
            return macd_val > signal_val and hist_val > 0
        elif condition.lower() in ["bearish crossover", "bearish", "cross below"]:
            if prev_idx is not None and prev_idx < len(macd) and prev_idx < len(signal):
                return macd[prev_idx] >= signal[prev_idx] and macd_val < signal_val
            return macd_val < signal_val and hist_val < 0
        elif condition.lower() == "above zero":
            return macd_val > 0
        elif condition.lower() == "below zero":
            return macd_val < 0
    
    elif indicator_name.upper() in ["BB", "BOLLINGER", "BOLLINGER BANDS"]:
        upper = indicator_data.get("upper", [])
        lower = indicator_data.get("lower", [])
        middle = indicator_data.get("middle", [])
        prices = indicator_data.get("prices", [])
        
        if idx >= len(upper) or idx >= len(lower) or idx >= len(prices):
            return False
        
        price = prices[idx]
        upper_band = upper[idx]
        lower_band = lower[idx]
        
        if condition.lower() in ["touch lower", "lower band", "oversold"]:
            return price <= lower_band * 1.01  # Within 1% of lower band
        elif condition.lower() in ["touch upper", "upper band", "overbought"]:
            return price >= upper_band * 0.99  # Within 1% of upper band
        elif condition.lower() == "squeeze":
            band_width = (upper_band - lower_band) / middle[idx] if middle[idx] > 0 else 0
            return band_width < 0.02  # Less than 2% band width
    
    return False


def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate EMA indicator"""
    import pandas as pd
    return pd.Series(prices).ewm(span=period, adjust=False).mean().tolist()


def calculate_vwap(df: Any) -> List[float]:
    """Calculate VWAP indicator - Resets fresh every trading day"""
    import pandas as pd
    # Ensure date column is properly handled for grouping
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    df_copy['date_only'] = df_copy['date'].dt.date
    
    # Calculate Typical Price
    tp = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
    pv = tp * df_copy['volume']
    
    # Calculate cumulative values per day
    # Grouping by date_only ensures VWAP resets daily
    cpv = pv.groupby(df_copy['date_only']).cumsum()
    cv = df_copy.groupby(df_copy['date_only'])['volume'].cumsum()
    
    # Avoid division by zero
    vwap_series = cpv / cv.replace(0, float('nan'))
    # Forward fill to handle any NaN from zero volume periods
    vwap_series = vwap_series.ffill().fillna(df_copy['close'])
    return vwap_series.tolist()


def get_swing_level(prices: List[float], idx: int, lookback: int = 5, find_high: bool = False) -> float:
    """Find recent swing high or low"""
    start = max(0, idx - lookback)
    window = prices[start:idx+1]
    return max(window) if find_high else min(window)


def is_rejection_candle(open_p: float, high: float, low: float, close: float, candle_type: str = "BULLISH") -> bool:
    """Detect institutional rejection candle (pin bar / hammer)"""
    body = abs(close - open_p)
    candle_range = high - low
    if candle_range == 0: return False
    
    if candle_type == "BULLISH":
        # Hammer/Pin Bar: Lower shadow should be at least 35% of total range (relaxed from 45%)
        lower_shadow = min(open_p, close) - low
        return (lower_shadow >= candle_range * 0.35) and (close > low + candle_range * 0.3)
    else:
        # Shooting Star: Upper shadow should be at least 35% of total range
        upper_shadow = high - max(open_p, close)
        return (upper_shadow >= candle_range * 0.35) and (close < high - candle_range * 0.3)


def is_engulfing(curr_o, curr_c, prev_o, prev_c, candle_type="BULLISH"):
    """Detect engulfing pattern"""
    if candle_type == "BULLISH":
        return curr_c > curr_o and prev_c < prev_o and curr_c > prev_o and curr_o < prev_c
    else:
        return curr_c < curr_o and prev_c > prev_o and curr_c < prev_o and curr_o > prev_c


@tool
def find_indicator_based_trading_opportunities(
    instrument_name: Union[str, List[str]],
    indicators: Union[str, List[str]] = "RSI",  # Can be "RSI", "MACD", "BB", or ["RSI", "MACD"]
    conditions: Union[str, List[str]] = "oversold",  # Can be single or list matching indicators
    date: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    interval: str = "5minute",
    exchange: str = "NSE",
    use_risk_reward: bool = True,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    local_data_file: Optional[str] = None
) -> dict:
    """
    Find trading opportunities based on single or multiple indicator conditions.
    Uses configured risk/reward ratios and position sizing.
    
    Args:
        instrument_name: Instrument name (e.g., "RELIANCE", "reliance", "NIFTY 50")
        indicators: Single indicator name or list (RSI, MACD, BB, or combinations)
        conditions: Single condition or list matching indicators (e.g., ["oversold", "bullish crossover"])
        date: Single date in YYYY-MM-DD format (deprecated, use from_date/to_date)
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format (default: today)
        interval: Time interval (minute, 5minute, etc.)
        exchange: Exchange (NSE, NFO, BSE)
        use_risk_reward: Whether to use configured risk/reward ratios for exit calculation
        stop_loss_pct: Custom stop loss percentage (overrides config if provided)
        take_profit_pct: Custom take profit percentage (overrides config if provided)
        local_data_file: Optional path to a local JSON data file (offline simulation)
        
    Returns:
        dict with trading opportunities including entry/exit suggestions with risk management
    """
    try:
        config = get_agent_config()
        
        # 1. Resolve instruments (handle groups)
        if isinstance(instrument_name, str):
            inst_lower = instrument_name.lower().strip()
            from agent.tools.instrument_resolver import INSTRUMENT_GROUPS
            if inst_lower in INSTRUMENT_GROUPS:
                instrument_names = INSTRUMENT_GROUPS[inst_lower]
            else:
                instrument_names = [instrument_name]
        else:
            instrument_names = instrument_name
        
        # Load local data if provided
        local_data = None
        sim_metadata = {}
        if local_data_file:
            import json
            import os
            if os.path.exists(local_data_file):
                with open(local_data_file, "r") as f:
                    sim_json = json.load(f)
                    local_data = sim_json.get("data", {})
                    sim_metadata = sim_json.get("metadata", {})
                    print(f"[DEBUG] Trading Opportunities | Loaded local data from {local_data_file}")
                    
                    # SMART DATE PICKING: If simulating local data, use the dates from the file
                    if not from_date and not date and sim_metadata.get("from_date"):
                        start_dt = datetime.strptime(sim_metadata["from_date"], "%Y-%m-%d").date()
                        end_dt = datetime.strptime(sim_metadata.get("to_date", sim_metadata["from_date"]), "%Y-%m-%d").date()
                        print(f"[DEBUG] Trading Opportunities | Auto-selected dates from sim file: {start_dt} to {end_dt}")
            else:
                print(f"[DEBUG] Trading Opportunities | Local data file {local_data_file} not found")

        kite = get_kite_instance()
        
        # Helper function to process single instrument
        def process_instrument(inst_name):
            # Check local data first
            if local_data and inst_name in local_data:
                historical_data_all = local_data[inst_name]
                # Convert back to datetime if needed or handle as ISO strings
                import pandas as pd
                df_all = pd.DataFrame(historical_data_all)
                df_all['date'] = pd.to_datetime(df_all['date'])
                df_all['date_only'] = df_all['date'].dt.date
                tradingsymbol = inst_name
                instrument_info = {"tradingsymbol": inst_name}
            else:
                # Resolve instrument name
                resolved = resolve_instrument_name(inst_name, exchange, return_multiple=False)
                if not resolved:
                    return {"status": "error", "error": f"Instrument '{inst_name}' not found"}
                
                instrument_token = resolved["instrument_token"]
                tradingsymbol = resolved["tradingsymbol"]
                
                # Fetch data with lookback
                if interval == "day":
                    fetch_start = start_dt - timedelta(days=200)
                else:
                    fetch_start = start_dt - timedelta(days=5)
                
                historical_data_all = kite.historical_data(instrument_token, fetch_start, end_dt, interval)
                if not historical_data_all:
                    return {"status": "error", "error": f"No data found for {tradingsymbol}"}
                
                import pandas as pd
                df_all = pd.DataFrame(historical_data_all)
                df_all['date_only'] = pd.to_datetime(df_all['date']).dt.date

            # ... rest of the single instrument logic continues ...

            # Check if we have data within the target range
            df_target = df_all[(df_all['date_only'] >= start_dt) & (df_all['date_only'] <= end_dt)]
            
            if df_target.empty:
                return {
                    "status": "error",
                    "error": f"Market was likely closed for {instrument_info['tradingsymbol']} during the period {start_dt} to {end_dt}"
                }
            
            # We need historical lookback for indicators (EMA 200 needs 200 candles)
            if len(df_all) < 200:
                # If day interval and not enough data, try fetching more
                if interval == "day":
                     fetch_start = start_dt - timedelta(days=365)
                     historical_data_all = kite.historical_data(
                        instrument_token=instrument_token,
                        from_date=fetch_start,
                        to_date=end_dt,
                        interval=interval
                     )
                     df_all = pd.DataFrame(historical_data_all)
                     df_all['date_only'] = pd.to_datetime(df_all['date']).dt.date
            
            if len(df_all) < 50:
                return {
                    "status": "error",
                    "error": f"Insufficient historical data (found {len(df_all)} candles, need at least 50) for {instrument_info['tradingsymbol']}"
                }
            
            closes = df_all['close'].values.tolist()
            highs = df_all['high'].values.tolist()
            lows = df_all['low'].values.tolist()
            opens = df_all['open'].values.tolist()
            
            # Calculate all required indicators
            indicator_data = {}
            
            # 1. Base RSI
            rsi_values = calculate_rsi(closes)
            indicator_data["RSI"] = rsi_values
            
            # 2. Trend Filters (EMAs)
            ema50 = calculate_ema(closes, 50)
            ema200 = calculate_ema(closes, 200)
            indicator_data["EMA50"] = ema50
            indicator_data["EMA200"] = ema200
            
            # 3. Price Action Confirmers (VWAP)
            vwap = calculate_vwap(df_all)
            indicator_data["VWAP"] = vwap
            
            # 4. Optional Indicators (if specifically asked)
            if "MACD" in [ind.upper() for ind in indicators]:
                indicator_data["MACD"] = calculate_macd(closes)
            if "BB" in [ind.upper() for ind in indicators] or "BOLLINGER" in [ind.upper() for ind in indicators]:
                indicator_data["BB"] = calculate_bollinger_bands(closes)
            
            opportunities = []
            current_capital = config.trading_capital # Compounding capital
            
            # Find opportunities where ALL indicator conditions are met
            idx = 200 # Need 200 candles for EMA 200 to stabilize
            while idx < len(closes):
                # ONLY process opportunities that occurred WITHIN THE TARGET RANGE
                row = df_all.iloc[idx]
                curr_date_only = row['date_only']
                if curr_date_only < start_dt or curr_date_only > end_dt:
                    idx += 1
                    continue
                
                # --- MARKET REGIME CLASSIFICATION ---
                price = closes[idx]
                e50 = ema50[idx]
                e200 = ema200[idx]
                
                regime = "RANGE"
                if price > e50 and price > e200:
                    regime = "UPTREND"
                elif price < e50 and price < e200:
                    regime = "DOWNTREND"
                
                # --- PRIME SESSION FILTER (10:15 AM - 02:45 PM) ---
                # We avoid the extreme volatile open (9:15-10:15) and erratic closing (after 14:45)
                is_trade_window = datetime.strptime("10:15", "%H:%M").time() <= curr_time <= datetime.strptime("14:45", "%H:%M").time()
                
                # --- ONLY STRATEGY: INSTITUTIONAL VWAP + RSI ---
                signal_type = None
                signal_reason = ""
                
                # ONLY ENTER DURING THE TRADE WINDOW
                if is_trade_window:
                    current_rsi = rsi_values[idx]
                    prev_rsi = rsi_values[idx-1]
                    v_price = vwap[idx]
                    
                    # Check near VWAP (widened to 0.5% from 0.35%)
                    is_near_vwap = abs(price - v_price) / v_price <= 0.005
                    
                    if is_near_vwap:
                        if price > v_price: # Potential BUY (Uptrend)
                            # RSI Filter: RSI should be below 45 during pullback (slightly more inclusive)
                            if current_rsi < 45: 
                                bullish_rejection = is_rejection_candle(opens[idx], highs[idx], lows[idx], closes[idx], "BULLISH")
                                bullish_engulfing = is_engulfing(opens[idx], closes[idx], opens[idx-1], closes[idx-1], "BULLISH")
                                rsi_turning_up = current_rsi > prev_rsi
                                
                                if (bullish_rejection or bullish_engulfing) and rsi_turning_up:
                                    signal_type = "BUY"
                                    signal_reason = f"Institutional VWAP: Bullish @ VWAP + RSI {current_rsi:.1f} Turn Up"
                        
                        elif price < v_price: # Potential SELL (Downtrend)
                            # RSI Filter: RSI should be above 55 during pullback
                            if current_rsi > 55:
                                bearish_rejection = is_rejection_candle(opens[idx], highs[idx], lows[idx], closes[idx], "BEARISH")
                                bearish_engulfing = is_engulfing(opens[idx], closes[idx], opens[idx-1], closes[idx-1], "BEARISH")
                                rsi_turning_down = current_rsi < prev_rsi
                                
                                if (bearish_rejection or bearish_engulfing) and rsi_turning_down:
                                    signal_type = "SELL"
                                    signal_reason = f"Institutional VWAP: Bearish @ VWAP + RSI {current_rsi:.1f} Turn Down"

                if signal_type:
                    entry_price = price
                    entry_time = row['date']
                    
                    # --- INSTITUTIONAL STOP LOSS & TARGET ---
                    rr_ratio = config.reward_per_trade_pct / config.risk_per_trade_pct
                    if signal_type == "BUY":
                        v_price = vwap[idx]
                        stop_loss_price = min(v_price * 0.998, entry_price * 0.995) 
                        risk = entry_price - stop_loss_price
                        take_profit_price = entry_price + (risk * rr_ratio)
                    else:
                        v_price = vwap[idx]
                        stop_loss_price = max(v_price * 1.002, entry_price * 1.005)
                        risk = stop_loss_price - entry_price
                        take_profit_price = entry_price - (risk * rr_ratio)
                    
                    # Position sizing
                    position_size_result = suggest_position_size_tool.invoke({
                        "entry_price": entry_price,
                        "stop_loss_price": stop_loss_price,
                        "available_capital": current_capital, # Use dynamic compounding capital
                        "risk_percentage": config.risk_per_trade_pct
                    })
                    qty = position_size_result.get("suggested_quantity", 1)
                    
                    # --- EXECUTION / EXIT SEARCH ---
                    exit_idx = None
                    exit_price = None
                    exit_reason = ""
                    
                    for search_idx in range(idx + 1, len(closes)):
                        curr_p = closes[search_idx]
                        curr_rsi_val = rsi_values[search_idx]
                        
                        # 1. SL/TP Hit
                        if signal_type == "BUY":
                            if curr_p <= stop_loss_price:
                                exit_price = stop_loss_price
                                exit_reason = "Stop Loss Hit"
                                break
                            if curr_p >= take_profit_price:
                                exit_price = take_profit_price
                                exit_reason = "Target Hit"
                                break
                            # RSI overbought exit
                            if curr_rsi_val > 70:
                                exit_price = curr_p
                                exit_reason = "RSI Overbought reached"
                                break
                        else:
                            if curr_p >= stop_loss_price:
                                exit_price = stop_loss_price
                                exit_reason = "Stop Loss Hit"
                                break
                            if curr_p <= take_profit_price:
                                exit_price = take_profit_price
                                exit_reason = "Target Hit"
                                break
                            # RSI oversold exit
                            if curr_rsi_val < 30:
                                exit_price = curr_p
                                exit_reason = "RSI Oversold reached"
                                break
                        
                        # 2. End of Day Exit (Intraday Square-off at 3:15 PM)
                        row_date = df_all.iloc[search_idx]['date']
                        if row_date.time() >= datetime.strptime("15:15", "%H:%M").time() or df_all.iloc[search_idx]['date_only'] != curr_date_only:
                            exit_price = curr_p
                            exit_reason = "3:15 PM Square-off" if row_date.time() >= datetime.strptime("15:15", "%H:%M").time() else "End of Day"
                            break
                    
                    if exit_price is None:
                        exit_idx = len(closes) - 1
                        exit_price = closes[exit_idx]
                        exit_reason = "Data End"
                    else:
                        exit_idx = search_idx

                    # P&L Calculation
                    pnl = (exit_price - entry_price) * qty if signal_type == "BUY" else (entry_price - exit_price) * qty
                    pnl_pct = ((exit_price - entry_price) / entry_price * 100) if signal_type == "BUY" else ((entry_price - exit_price) / entry_price * 100)
                    
                    # Update compounding capital
                    current_capital += pnl
                    
                    opportunities.append({
                        "signal_type": signal_type,
                        "signal_reason": signal_reason + ( " [Prime]" if is_trade_window else "" ),
                        "entry_time": str(entry_time),
                        "entry_price": float(entry_price),
                        "stop_loss_price": float(stop_loss_price),
                        "take_profit_price": float(take_profit_price),
                        "exit_time": str(df_all.iloc[exit_idx]['date']),
                        "exit_price": float(exit_price),
                        "exit_reason": exit_reason,
                        "suggested_quantity": int(qty),
                        "pnl": float(pnl),
                        "pnl_percent": float(pnl_pct),
                        "risk_reward_ratio": rr_ratio,
                        "duration_candles": exit_idx - idx,
                        "available_funds": float(current_capital)
                    })
                    
                    idx = exit_idx + 1 # Skip to next candle after exit
                else:
                    idx += 1
            
            # Calculate summary statistics
            if opportunities:
                total_pnl = sum(opp["pnl"] for opp in opportunities)
                total_pnl_percent = sum(opp["pnl_percent"] for opp in opportunities)
                winning_trades = [opp for opp in opportunities if opp["pnl"] > 0]
                losing_trades = [opp for opp in opportunities if opp["pnl"] <= 0]
                
                summary = {
                    "total_opportunities": len(opportunities),
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                    "win_rate": (len(winning_trades) / len(opportunities) * 100) if opportunities else 0,
                    "total_pnl": float(total_pnl),
                    "total_pnl_percent": float(total_pnl_percent),
                    "avg_pnl_per_trade": float(total_pnl / len(opportunities)) if opportunities else 0,
                    "avg_risk_reward_ratio": float(sum(opp["risk_reward_ratio"] for opp in opportunities) / len(opportunities)) if opportunities else 0,
                    "best_trade": max(opportunities, key=lambda x: x["pnl"]) if opportunities else None,
                    "worst_trade": min(opportunities, key=lambda x: x["pnl"]) if opportunities else None,
                    "config_used": {
                        "risk_per_trade_pct": config.risk_per_trade_pct,
                        "reward_per_trade_pct": config.reward_per_trade_pct,
                        "max_position_size": config.max_position_size
                    }
                }
            else:
                summary = {
                    "total_opportunities": 0,
                    "message": f"No opportunities found for {', '.join(indicators)} with conditions {', '.join(conditions)} from {start_dt} to {end_dt}"
                }
            
            return {
                "status": "success",
                "instrument": instrument_info["tradingsymbol"],
                "indicators": indicators,
                "conditions": conditions,
                "date": f"{start_dt} to {end_dt}",
                "interval": interval,
                "opportunities": opportunities,
                "summary": summary
            }
        
        # Process single or multiple instruments
        if len(instrument_names) == 1:
            # Single instrument - return simple format
            return process_instrument(instrument_names[0])
        else:
            # --- GLOBAL SEQUENTIAL ANALYSIS FOR MULTIPLE INSTRUMENTS ---
            import pandas as pd
            print(f"[DEBUG] Starting Global Sequential Analysis for {len(instrument_names)} instruments")
            
            all_instrument_data = {}
            print(f"[DEBUG] Local data keys: {list(local_data.keys()) if local_data else 'None'}")
            print(f"[DEBUG] Target instruments: {instrument_names}")
            print(f"[DEBUG] Target range: {start_dt} to {end_dt}")

            for inst_name in instrument_names:
                # Check local data first
                if local_data and inst_name in local_data:
                    data = local_data[inst_name]
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df['date_only'] = df['date'].dt.date
                    df['symbol'] = inst_name
                    symbol = inst_name
                    print(f"[DEBUG] Loaded {len(df)} candles for {symbol} from local data")
                else:
                    # Resolve instrument name
                    resolved = resolve_instrument_name(inst_name, exchange, return_multiple=False)
                    if not resolved: 
                        print(f"[DEBUG] Could not resolve {inst_name}")
                        continue
                    
                    token = resolved["instrument_token"]
                    symbol = resolved["tradingsymbol"]
                    
                    # Fetch data with lookback
                    if interval == "day":
                        fetch_start = start_dt - timedelta(days=200)
                    else:
                        fetch_start = start_dt - timedelta(days=5)
                    
                    try:
                        print(f"[DEBUG] Fetching {symbol} from Kite...")
                        data = kite.historical_data(token, fetch_start, end_dt, interval)
                        if not data: continue
                        df = pd.DataFrame(data)
                        df['date'] = pd.to_datetime(df['date'])
                        df['date_only'] = df['date'].dt.date
                        df['symbol'] = symbol
                    except Exception as fe:
                        print(f"[DEBUG] Kite fetch failed for {symbol}: {fe}")
                        continue
                
                # Calculate indicators for this instrument
                closes = df['close'].values.tolist()
                df['rsi'] = calculate_rsi(closes)
                df['vwap'] = calculate_vwap(df)
                df['prev_rsi'] = df['rsi'].shift(1)
                df['prev_open'] = df['open'].shift(1)
                df['prev_close'] = df['close'].shift(1)
                
                # Store processed data (only target range)
                print(f"[DEBUG] {symbol} | Data Date Range in file: {df['date_only'].min()} to {df['date_only'].max()}")
                df_target = df[(df['date_only'] >= start_dt) & (df['date_only'] <= end_dt)]
                print(f"[DEBUG] {symbol} | Target data size: {len(df_target)}")
                if not df_target.empty:
                    all_instrument_data[symbol] = df_target

            if not all_instrument_data:
                return {"status": "error", "error": "No data found for any instruments in the group"}

            # Combine all candles and sort by time
            combined_timeline = pd.concat(all_instrument_data.values()).sort_values('date')
            
            global_opportunities = []
            active_trade = None # Stores instrument symbol if in trade
            current_capital = config.trading_capital # SMART: Start with configured capital
            
            print(f"[DEBUG] Processing {len(combined_timeline)} total candles across the group timeline")
            
            # Group by timestamp to process all stocks simultaneously at each candle
            for timestamp, group in combined_timeline.groupby('date'):
                # 1. If in a trade, only check for exit of that specific instrument
                if active_trade:
                    # Find the current candle for the active instrument
                    current_inst_row = group[group['symbol'] == active_trade['symbol']]
                    if current_inst_row.empty: 
                        continue 
                    
                    row = current_inst_row.iloc[0]
                    curr_p = row['close']
                    curr_rsi = row['rsi']
                    curr_date_only = row['date_only']
                    
                    # Check Exit Conditions (SL/TP/Time)
                    exit_triggered = False
                    exit_price = None
                    exit_reason = ""
                    
                    if active_trade['type'] == "BUY":
                        if curr_p <= active_trade['sl']:
                            exit_triggered, exit_price, exit_reason = True, active_trade['sl'], "Stop Loss Hit"
                        elif curr_p >= active_trade['tp']:
                            exit_triggered, exit_price, exit_reason = True, active_trade['tp'], "Target Hit"
                        elif not pd.isna(curr_rsi) and curr_rsi > 70:
                            exit_triggered, exit_price, exit_reason = True, curr_p, "RSI Overbought reached"
                    else: # SELL
                        if curr_p >= active_trade['sl']:
                            exit_triggered, exit_price, exit_reason = True, active_trade['sl'], "Stop Loss Hit"
                        elif curr_p <= active_trade['tp']:
                            exit_triggered, exit_price, exit_reason = True, active_trade['tp'], "Target Hit"
                        elif not pd.isna(curr_rsi) and curr_rsi < 30:
                            exit_triggered, exit_price, exit_reason = True, curr_p, "RSI Oversold reached"
                    
                    # End of Day Exit (Intraday Square-off at 3:15 PM)
                    if not exit_triggered and (timestamp.time() >= datetime.strptime("15:15", "%H:%M").time() or curr_date_only != active_trade['entry_date_only']):
                        exit_triggered, exit_price, exit_reason = True, curr_p, "3:15 PM Square-off" if timestamp.time() >= datetime.strptime("15:15", "%H:%M").time() else "End of Day"
                    
                    if exit_triggered:
                        # Record the trade
                        qty = active_trade['qty']
                        pnl = (exit_price - active_trade['entry_price']) * qty if active_trade['type'] == "BUY" else (active_trade['entry_price'] - exit_price) * qty
                        pnl_pct = ((exit_price - active_trade['entry_price']) / active_trade['entry_price'] * 100) if active_trade['type'] == "BUY" else ((active_trade['entry_price'] - exit_price) / active_trade['entry_price'] * 100)
                        
                        current_capital += pnl
                        
                        global_opportunities.append({
                            "instrument": active_trade['symbol'],
                            "signal_type": active_trade['type'],
                            "signal_reason": active_trade['reason'],
                            "entry_time": str(active_trade['entry_time']),
                            "entry_price": float(active_trade['entry_price']),
                            "exit_time": str(timestamp),
                            "exit_price": float(exit_price),
                            "exit_reason": exit_reason,
                            "suggested_quantity": int(qty),
                            "pnl": float(pnl),
                            "pnl_percent": float(pnl_pct),
                            "available_funds": float(current_capital)
                        })
                        active_trade = None 
                    
                    continue 

                # 2. If NOT in a trade, check for signals across ALL instruments in this candle
                # --- PRIME SESSION FILTER (10:15 AM - 02:45 PM) ---
                curr_time = timestamp.time()
                is_trade_window = datetime.strptime("10:15", "%H:%M").time() <= curr_time <= datetime.strptime("14:45", "%H:%M").time()
                
                if is_trade_window:
                    for _, row in group.iterrows():
                        symbol = row['symbol']
                        price = row['close']
                        v_price = row['vwap']
                        current_rsi = row['rsi']
                        prev_rsi = row['prev_rsi']
                        
                        if pd.isna(v_price) or pd.isna(current_rsi): continue
                        
                        # More inclusive zone for multi-stock scan (0.75% widened from 0.5%)
                        is_near_vwap = abs(price - v_price) / v_price <= 0.0075

                        if is_near_vwap:
                            signal_type = None
                            if price > v_price and current_rsi < 50: # Potential BUY
                                bullish_rejection = is_rejection_candle(row['open'], row['high'], row['low'], row['close'], "BULLISH")
                                bullish_engulfing = is_engulfing(row['open'], row['close'], row['prev_open'], row['prev_close'], "BULLISH")
                                rsi_turning_up = True if pd.isna(prev_rsi) else current_rsi > prev_rsi
                                
                                if (bullish_rejection or bullish_engulfing) and rsi_turning_up:
                                    signal_type = "BUY"
                            elif price < v_price and current_rsi > 50: 
                                bearish_rejection = is_rejection_candle(row['open'], row['high'], row['low'], row['close'], "BEARISH")
                                bearish_engulfing = is_engulfing(row['open'], row['close'], row['prev_open'], row['prev_close'], "BEARISH")
                                rsi_turning_down = True if pd.isna(prev_rsi) else current_rsi < prev_rsi
                                
                                if (bearish_rejection or bearish_engulfing) and rsi_turning_down:
                                    signal_type = "SELL"
                            
                            if signal_type:
                                print(f"[DEBUG] Global Scan | Found {signal_type} Signal for {symbol} at {timestamp}")
                                # Found a valid signal! Enter and lock capital.
                                entry_price = price
                                
                                # Calculate SL/TP
                                rr_ratio = config.reward_per_trade_pct / config.risk_per_trade_pct
                                if signal_type == "BUY":
                                    sl = min(v_price * 0.998, entry_price * 0.995)
                                    tp = entry_price + (abs(entry_price - sl) * rr_ratio)
                                else:
                                    sl = max(v_price * 1.002, entry_price * 1.005)
                                    tp = entry_price - (abs(sl - entry_price) * rr_ratio)
                                
                                # SMART POSITION SIZING: Use 'current_capital' (includes previous gains/losses)
                                # This implements compounding and loss-based scaling.
                                ps = suggest_position_size_tool.invoke({
                                    "entry_price": entry_price, "stop_loss_price": sl,
                                    "available_capital": current_capital, # Use current dynamic funds
                                    "risk_percentage": config.risk_per_trade_pct
                                })
                                
                                active_trade = {
                                    "symbol": symbol, "type": signal_type, "entry_price": entry_price,
                                    "entry_time": timestamp, "entry_date_only": row['date_only'],
                                    "sl": sl, "tp": tp, "qty": ps.get("suggested_quantity", 1),
                                    "reason": f"Institutional VWAP: {signal_type} @ {symbol}"
                                }
                                break # Only take the first signal found in this candle group

            # Results aggregation
            if not global_opportunities:
                return {
                    "status": "success", "total_opportunities": 0,
                    "message": f"No sequential trades found for the group across {start_dt} to {end_dt}"
                }
            
            total_pnl = sum(o["pnl"] for o in global_opportunities)
            return {
                "status": "success",
                "indicators": indicators, "date": f"{start_dt} to {end_dt}",
                "instruments_analyzed": len(instrument_names),
                "total_opportunities": len(global_opportunities),
                "total_pnl": float(total_pnl),
                "opportunities": global_opportunities, # Flat list for sequential display
                "is_sequential": True
            }
            
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": f"Error finding trading opportunities: {str(e)}",
            "traceback": traceback.format_exc()
        }
