"""
Enterprise scheduler catalog: registered strategies and supported technical indicators.
Used by GET /telegram-scheduler/catalog and strategy overview notifications.
"""

from typing import Any, Dict, List

# Aligns with api/v1/routes/strategies sub-routers and product features
STRATEGY_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "nifty50_options",
        "name": "Nifty50 options",
        "description": "Nifty 50 index options chain with strikes, quotes, and trend context.",
        "api": "/api/v1/strategies/nifty50-options",
    },
    {
        "id": "vwap_intraday",
        "name": "VWAP intraday",
        "description": "Volume-weighted average price signals with pattern and VWAP distance filters.",
        "api": "/api/v1/strategies/vwap",
    },
    {
        "id": "range_breakout_30m",
        "name": "Range breakout (30m)",
        "description": "30-minute range breakout with risk and execution hooks.",
        "api": "/api/v1/strategies/range-breakout-30min",
    },
    {
        "id": "binance_futures",
        "name": "Binance USDT-M futures",
        "description": "Crypto futures VWAP / RSI commentary and live checks.",
        "api": "/api/v1/strategies/binance-futures",
    },
]

# Supported in scheduled indicator reports (Kite historical data)
INDICATOR_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "rsi_14",
        "name": "RSI (14)",
        "description": "Relative Strength Index on closing prices, Wilder smoothing.",
        "params": [],
    },
    {
        "id": "ema_9_21",
        "name": "EMA 9 / 21",
        "description": "Exponential moving averages; reports last values and crossover state.",
        "params": [],
    },
    {
        "id": "vwap_distance",
        "name": "Session VWAP distance",
        "description": "Approximate session VWAP from recent bars and distance of last close.",
        "params": [],
    },
    {
        "id": "bollinger_20",
        "name": "Bollinger (20, 2)",
        "description": "Upper, middle, lower bands vs last close.",
        "params": [],
    },
    {
        "id": "pivots_daily",
        "name": "Classic pivots",
        "description": "Previous bar pivot, R1–R3, S1–S3 vs last close.",
        "params": [],
    },
]

KITE_INTERVALS: List[Dict[str, str]] = [
    {"id": "minute", "label": "1 minute"},
    {"id": "3minute", "label": "3 minutes"},
    {"id": "5minute", "label": "5 minutes"},
    {"id": "15minute", "label": "15 minutes"},
    {"id": "60minute", "label": "1 hour"},
    {"id": "day", "label": "1 day"},
]


def get_catalog_payload() -> Dict[str, Any]:
    return {
        "strategies": STRATEGY_DEFINITIONS,
        "indicators": INDICATOR_DEFINITIONS,
        "intervals": KITE_INTERVALS,
        "version": "1.0",
    }
