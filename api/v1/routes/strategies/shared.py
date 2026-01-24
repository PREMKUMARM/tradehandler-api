"""
Shared utilities for strategy backtests
"""
from datetime import datetime, timedelta, date
from typing import Optional, Callable
from fastapi import WebSocket
import asyncio


def to_bool(value, default: bool) -> bool:
    """Convert value to boolean, handling various input types"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)


async def send_log(websocket: WebSocket, log_type: str, message: str, date: str = None, details: dict = None):
    """Send log message to frontend via WebSocket"""
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


def get_trading_dates(start_date: date, end_date: date) -> list:
    """Get list of trading dates (weekdays only) between start and end dates"""
    trading_dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday=0 to Friday=4
            trading_dates.append(current_date)
        current_date += timedelta(days=1)
    return trading_dates

