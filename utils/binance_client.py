"""
Binance API client for fetching cryptocurrency market data
"""
import httpx
from typing import List
import asyncio

BASE_URL = "https://fapi.binance.com"  # Binance Futures API


async def fetch_klines(symbol: str, interval: str, limit: int = 100) -> List[dict]:
    """
    Fetch candlestick data from Binance Futures API
    
    Args:
        symbol: Trading pair symbol (e.g., "ETHUSDT")
        interval: Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        limit: Number of candles to fetch (default: 100)
    
    Returns:
        List of dicts with OHLCV data: [{"open": float, "high": float, "low": float, "close": float, "volume": float}, ...]
    """
    url = f"{BASE_URL}/fapi/v1/klines"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url, 
                params={
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "limit": limit
                }
            )
            resp.raise_for_status()
            data = resp.json()
            # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
            # Extract OHLCV data with timestamps
            # Indices: 0=open_time, 1=open, 2=high, 3=low, 4=close, 5=volume, 6=close_time
            klines = [
                {
                    "open_time": int(k[0]),  # Open time in milliseconds
                    "close_time": int(k[6]),  # Close time in milliseconds
                    "timestamp": int(k[6]),  # Use close_time as primary timestamp
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                }
                for k in data
            ]
            return klines
    except httpx.HTTPStatusError as e:
        raise Exception(f"Binance API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"Error fetching Binance data: {str(e)}")


async def fetch_multiple_klines(symbols: List[str], intervals: List[str], limit: int = 100) -> dict:
    """
    Fetch klines for multiple symbols and intervals concurrently
    
    Args:
        symbols: List of trading pair symbols
        intervals: List of timeframes
        limit: Number of candles to fetch
    
    Returns:
        Dictionary: {symbol: {interval: [{"open": float, "high": float, "low": float, "close": float, "volume": float}, ...]}}
    """
    # Create all tasks
    tasks = []
    task_info = []  # Store (symbol, interval) for each task
    
    for symbol in symbols:
        for interval in intervals:
            task = fetch_klines(symbol, interval, limit)
            tasks.append(task)
            task_info.append((symbol, interval))
    
    # Execute all tasks concurrently
    results = {}
    try:
        # Wait for all tasks to complete
        klines_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for (symbol, interval), klines in zip(task_info, klines_list):
            if isinstance(klines, Exception):
                print(f"Error fetching {symbol} {interval}: {klines}")
                klines = []
            
            if symbol not in results:
                results[symbol] = {}
            results[symbol][interval] = klines if isinstance(klines, list) else []
    except Exception as e:
        print(f"Error in fetch_multiple_klines: {e}")
        # Initialize empty results for all symbols
        for symbol in symbols:
            if symbol not in results:
                results[symbol] = {}
            for interval in intervals:
                results[symbol][interval] = []
    
    return results

