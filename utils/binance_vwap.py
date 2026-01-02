"""
VWAP calculator for Binance data with additional trading indicators
"""
from typing import List, Optional, Dict
import pandas as pd
import numpy as np


def compute_vwap(klines: List[Dict]) -> Optional[float]:
    """
    Calculate VWAP (Volume Weighted Average Price) from OHLCV data
    
    VWAP = Σ(Price × Volume) / Σ(Volume)
    Where Price = (High + Low + Close) / 3 (typical price)
    
    Args:
        klines: List of dicts with {"open": float, "high": float, "low": float, "close": float, "volume": float}
    
    Returns:
        Latest VWAP value or None if insufficient data
    """
    if not klines or len(klines) == 0:
        return None
    
    try:
        df = pd.DataFrame(klines)
        
        # Calculate typical price (High + Low + Close) / 3
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate price × volume
        df['price_volume'] = df['typical_price'] * df['volume']
        
        # Calculate cumulative sums
        df['cumulative_price_volume'] = df['price_volume'].cumsum()
        df['cumulative_volume'] = df['volume'].cumsum()
        
        # Calculate VWAP
        df['vwap'] = df['cumulative_price_volume'] / df['cumulative_volume'].replace(0, np.nan)
        
        # Fill NaN with close price if volume is 0
        df['vwap'] = df['vwap'].fillna(df['close'])
        
        # Return the latest VWAP value, rounded to 2 decimal places
        latest_vwap = df['vwap'].iloc[-1]
        return round(float(latest_vwap), 2) if not pd.isna(latest_vwap) else None
    except Exception as e:
        print(f"Error calculating VWAP: {e}")
        return None


def compute_rsi(closes: List[float], window: int = 14) -> Optional[float]:
    """
    Calculate RSI (Relative Strength Index) using Wilder's Smoothing Method
    
    Args:
        closes: List of closing prices
        window: RSI period (default: 14)
    
    Returns:
        Latest RSI value (0-100) or None if insufficient data
    """
    if len(closes) < window + 1:
        return None
    
    try:
        df = pd.DataFrame({"close": closes})
        delta = df["close"].diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        avg_gains = gains.ewm(alpha=1/window, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/window, adjust=False).mean()
        rs = avg_gains / avg_losses.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]
        return round(float(latest_rsi), 2) if not pd.isna(latest_rsi) else None
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return None


def compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict]:
    """
    Calculate MACD indicator
    
    Args:
        closes: List of closing prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
    
    Returns:
        Dict with {"macd": float, "signal": float, "histogram": float, "trend": "Bullish"|"Bearish"|"Neutral"}
        or None if insufficient data
    """
    if len(closes) < slow + signal:
        return None
    
    try:
        series = pd.Series(closes)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        latest_macd = float(macd_line.iloc[-1])
        latest_signal = float(signal_line.iloc[-1])
        latest_histogram = float(histogram.iloc[-1])
        
        # Determine trend
        if latest_macd > latest_signal and latest_histogram > 0:
            trend = "Bullish"
        elif latest_macd < latest_signal and latest_histogram < 0:
            trend = "Bearish"
        else:
            trend = "Neutral"
        
        return {
            "macd": round(latest_macd, 4),
            "signal": round(latest_signal, 4),
            "histogram": round(latest_histogram, 4),
            "trend": trend
        }
    except Exception as e:
        print(f"Error calculating MACD: {e}")
        return None


def compute_volume_ratio(volumes: List[float], lookback: int = 20) -> Optional[float]:
    """
    Calculate volume ratio (current volume / average volume)
    
    Args:
        volumes: List of volume values
        lookback: Number of periods for average (default: 20)
    
    Returns:
        Volume ratio or None if insufficient data
    """
    if len(volumes) < lookback + 1:
        return None
    
    try:
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-lookback:])
        if avg_volume > 0:
            return round(current_volume / avg_volume, 2)
        return None
    except Exception as e:
        print(f"Error calculating volume ratio: {e}")
        return None


def generate_trade_signal(vwap_position: str, rsi: Optional[float], macd_trend: Optional[str], 
                         vwap_deviation: float, volume_ratio: Optional[float]) -> str:
    """
    Generate trade signal based on combined indicators
    
    Args:
        vwap_position: "Above" or "Below"
        rsi: RSI value (0-100)
        macd_trend: "Bullish", "Bearish", or "Neutral"
        vwap_deviation: Percentage deviation from VWAP
        volume_ratio: Current volume / average volume
    
    Returns:
        "BUY", "SELL", or "HOLD"
    """
    buy_signals = 0
    sell_signals = 0
    
    # VWAP position signal
    if vwap_position == "Above":
        buy_signals += 1
    elif vwap_position == "Below":
        sell_signals += 1
    
    # RSI signal
    if rsi is not None:
        if rsi < 30:  # Oversold - potential buy
            buy_signals += 1
        elif rsi > 70:  # Overbought - potential sell
            sell_signals += 1
        elif 30 <= rsi <= 50:  # Neutral to slightly oversold
            buy_signals += 0.5
        elif 50 < rsi <= 70:  # Neutral to slightly overbought
            sell_signals += 0.5
    
    # MACD signal
    if macd_trend == "Bullish":
        buy_signals += 1
    elif macd_trend == "Bearish":
        sell_signals += 1
    
    # VWAP deviation signal (strong deviation = stronger signal)
    if abs(vwap_deviation) > 1.0:  # More than 1% deviation
        if vwap_deviation > 0:  # Price above VWAP
            buy_signals += 0.5
        else:  # Price below VWAP
            sell_signals += 0.5
    
    # Volume confirmation (high volume = stronger signal)
    if volume_ratio is not None and volume_ratio > 1.2:  # 20% above average
        if buy_signals > sell_signals:
            buy_signals += 0.5
        elif sell_signals > buy_signals:
            sell_signals += 0.5
    
    # Generate signal
    if buy_signals >= 2.5:
        return "BUY"
    elif sell_signals >= 2.5:
        return "SELL"
    else:
        return "HOLD"


def compute_vwap_batch(data: dict) -> dict:
    """
    Calculate comprehensive trading indicators for multiple symbols and timeframes
    
    Args:
        data: Dictionary {symbol: {interval: [klines]}}
    
    Returns:
        Dictionary {symbol: {interval: {
            "vwap": float,
            "current_price": float,
            "position": "Above"|"Below",
            "vwap_deviation_pct": float,
            "rsi": float,
            "macd": {"macd": float, "signal": float, "histogram": float, "trend": str},
            "volume_ratio": float,
            "trade_signal": "BUY"|"SELL"|"HOLD"
        }}}
    """
    results = {}
    for symbol, timeframes in data.items():
        results[symbol] = {}
        for interval, klines in timeframes.items():
            if klines and len(klines) > 0:
                # Extract data
                closes = [float(k["close"]) for k in klines]
                volumes = [float(k["volume"]) for k in klines]
                current_price = closes[-1]
                
                # Calculate VWAP
                vwap = compute_vwap(klines)
                
                if vwap is not None and vwap > 0:
                    # VWAP position
                    position = "Above" if current_price > vwap else "Below"
                    
                    # VWAP deviation percentage
                    vwap_deviation_pct = round(((current_price - vwap) / vwap) * 100, 2)
                    
                    # Calculate RSI
                    rsi = compute_rsi(closes)
                    
                    # Calculate MACD
                    macd_data = compute_macd(closes)
                    
                    # Calculate volume ratio
                    volume_ratio = compute_volume_ratio(volumes)
                    
                    # Generate trade signal
                    trade_signal = generate_trade_signal(
                        position,
                        rsi,
                        macd_data["trend"] if macd_data else None,
                        vwap_deviation_pct,
                        volume_ratio
                    )
                    
                    results[symbol][interval] = {
                        "vwap": round(vwap, 2),
                        "current_price": round(current_price, 2),
                        "position": position,
                        "vwap_deviation_pct": vwap_deviation_pct,
                        "rsi": round(rsi, 2) if rsi is not None else None,
                        "macd": macd_data if macd_data else None,
                        "volume_ratio": volume_ratio,
                        "trade_signal": trade_signal
                    }
                else:
                    results[symbol][interval] = {
                        "vwap": 0,
                        "current_price": round(current_price, 2),
                        "position": "N/A",
                        "vwap_deviation_pct": 0,
                        "rsi": None,
                        "macd": None,
                        "volume_ratio": None,
                        "trade_signal": "HOLD"
                    }
            else:
                results[symbol][interval] = {
                    "vwap": 0,
                    "current_price": 0,
                    "position": "N/A",
                    "vwap_deviation_pct": 0,
                    "rsi": None,
                    "macd": None,
                    "volume_ratio": None,
                    "trade_signal": "HOLD"
                }
    
    return results

