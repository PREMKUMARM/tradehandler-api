"""
Strategies package - modular backtest implementations
"""
from fastapi import APIRouter

# Import all strategy routers
from .nifty50_options import router as nifty50_router
from .vwap_strategy import router as vwap_router
from .binance_futures import router as binance_router
from .range_breakout_30min import router as range_breakout_router

# Create main router with prefix
router = APIRouter(prefix="/strategies", tags=["Strategies"])

# Include all strategy routers
router.include_router(nifty50_router)
router.include_router(vwap_router)
router.include_router(binance_router)
router.include_router(range_breakout_router)

__all__ = ["router"]

