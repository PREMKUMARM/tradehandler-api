"""
Binance historical data fetcher for backtesting
"""
import httpx
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio


BASE_URL = "https://fapi.binance.com"  # Binance Futures API


def binance_interval_to_minutes(interval: str) -> int:
    """Convert Binance interval to minutes"""
    interval_map = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '8h': 480,
        '12h': 720,
        '1d': 1440,
        '3d': 4320,
        '1w': 10080,
        '1M': 43200
    }
    return interval_map.get(interval.lower(), 5)


def convert_timeframe_to_binance(timeframe: str) -> str:
    """Convert frontend timeframe to Binance interval"""
    timeframe_map = {
        '1minute': '1m',
        '5minute': '5m',
        '15minute': '15m',
        '30minute': '30m',
        '60minute': '1h',
        '4hour': '4h',
        'day': '1d'
    }
    return timeframe_map.get(timeframe.lower(), '5m')


async def fetch_historical_klines(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
    limit: int = 1000
) -> List[Dict]:
    """
    Fetch historical klines from Binance Futures API for a date range
    
    Args:
        symbol: Trading pair symbol (e.g., "ETHUSDT")
        interval: Binance interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        start_time: Start datetime
        end_time: End datetime
        limit: Maximum candles per request (default: 1000, max: 1500)
    
    Returns:
        List of dicts with OHLCV data and timestamp
    """
    url = f"{BASE_URL}/fapi/v1/klines"
    
    # Convert datetime to milliseconds timestamp
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    all_klines = []
    current_start = start_ms
    
    try:
        while current_start < end_ms:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    url,
                    params={
                        "symbol": symbol.upper(),
                        "interval": interval,
                        "startTime": current_start,
                        "endTime": end_ms,
                        "limit": limit
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                
                if not data:
                    break
                
                # Convert to our format
                for k in data:
                    kline_time = datetime.fromtimestamp(k[0] / 1000)
                    all_klines.append({
                        "timestamp": kline_time,
                        "date": kline_time,
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5])
                    })
                
                # If we got less than limit, we've reached the end
                if len(data) < limit:
                    break
                
                # Update start time for next batch (use last candle's close time + 1ms)
                current_start = k[6] + 1  # k[6] is close time in milliseconds
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
        
        return all_klines
    except httpx.HTTPStatusError as e:
        raise Exception(f"Binance API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"Error fetching Binance historical data: {str(e)}")


async def fetch_historical_klines_for_date_range(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str
) -> List[Dict]:
    """
    Fetch historical klines for a date range (handles multiple days)
    
    Args:
        symbol: Trading pair symbol
        interval: Binance interval
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    
    Returns:
        List of klines sorted by timestamp
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    # Set end time to end of day
    end_dt = end_dt.replace(hour=23, minute=59, second=59)
    
    # Fetch all klines for the date range
    klines = await fetch_historical_klines(symbol, interval, start_dt, end_dt)
    
    # Sort by timestamp
    klines.sort(key=lambda x: x["timestamp"])
    
    return klines

