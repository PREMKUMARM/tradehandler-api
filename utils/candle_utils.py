"""
Candle aggregation and analysis utilities
"""
import numpy as np


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
    if not candles_1m: 
        return []
    
    aggregated = []
    for i in range(0, len(candles_1m), tf_min):
        chunk = candles_1m[i:i+tf_min]
        if not chunk: 
            continue
        
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
    if len(candles) < 5: 
        return "NEUTRAL"
    
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
            
    if current_price > sma_5 * 1.002 and rsi > 55: 
        return "BULLISH"
    if current_price < sma_5 * 0.998 and rsi < 45: 
        return "BEARISH"
    return "SIDEWAYS"

