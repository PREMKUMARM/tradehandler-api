"""
Strategy implementations for Nifty50 options trading
All strategy functions extracted from main.py for better organization
"""
from datetime import datetime, timedelta
import pandas as pd

# Import helper functions
from utils.indicators import calculate_bollinger_bands, calculate_support_resistance
from simulation.helpers import find_option, add_live_log

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


# Helper function for multi-leg strategies
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
