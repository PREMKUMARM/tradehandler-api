"""
Automated trading strategies API endpoints
"""
from fastapi import APIRouter, Request, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import time

from utils.strategy_executor import strategy_executor, StrategyConfig, StrategyType
from utils.trade_limits import trade_limits
from utils.logger import log_info, log_error

router = APIRouter(prefix="/strategies", tags=["Automated Strategies"])


class StrategyRequest(BaseModel):
    """Request to add/update a strategy"""
    strategy_type: str = Field(..., description="Strategy type")
    symbol: str = Field(..., description="Trading symbol")
    quantity: int = Field(..., gt=0, description="Order quantity")
    product: str = Field(default="MIS", description="Product type")
    order_type: str = Field(default="MARKET", description="Order type")
    stoploss_pct: float = Field(default=0.02, gt=0, description="Stoploss percentage")
    target_pct: float = Field(default=0.04, gt=0, description="Target percentage")
    trailing_stoploss: bool = Field(default=False, description="Enable trailing stoploss")
    max_trades_per_day: int = Field(default=3, gt=0, description="Max trades per day")
    active_hours_start: str = Field(default="09:15", description="Active hours start (HH:MM)")
    active_hours_end: str = Field(default="15:25", description="Active hours end (HH:MM)")
    enabled: bool = Field(default=True, description="Enable strategy")


@router.post("/start")
async def start_strategy_executor():
    """Start the automated strategy executor"""
    try:
        await strategy_executor.start_execution()
        return {"status": "success", "message": "Strategy executor started"}
    except Exception as e:
        log_error(f"Error starting strategy executor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_strategy_executor():
    """Stop the automated strategy executor"""
    try:
        await strategy_executor.stop_execution()
        return {"status": "success", "message": "Strategy executor stopped"}
    except Exception as e:
        log_error(f"Error stopping strategy executor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add")
async def add_strategy(strategy_id: str, request: StrategyRequest):
    """Add a new automated strategy"""
    try:
        # Validate strategy type
        try:
            strategy_type = StrategyType(request.strategy_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy type. Valid types: {[s.value for s in StrategyType]}"
            )
        
        # Parse time strings
        try:
            start_time = time.fromisoformat(request.active_hours_start)
            end_time = time.fromisoformat(request.active_hours_end)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid time format. Use HH:MM format (e.g., 09:15)"
            )
        
        # Create strategy config
        config = StrategyConfig(
            strategy_type=strategy_type,
            symbol=request.symbol,
            quantity=request.quantity,
            product=request.product,
            order_type=request.order_type,
            stoploss_pct=request.stoploss_pct,
            target_pct=request.target_pct,
            trailing_stoploss=request.trailing_stoploss,
            max_trades_per_day=request.max_trades_per_day,
            active_hours_start=start_time,
            active_hours_end=end_time,
            enabled=request.enabled
        )
        
        # Add strategy
        strategy_executor.add_strategy(strategy_id, config)
        
        return {
            "status": "success",
            "message": f"Strategy {strategy_id} added successfully",
            "strategy_id": strategy_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error adding strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/remove/{strategy_id}")
async def remove_strategy(strategy_id: str):
    """Remove a strategy"""
    try:
        strategy_executor.remove_strategy(strategy_id)
        return {
            "status": "success",
            "message": f"Strategy {strategy_id} removed successfully"
        }
    except Exception as e:
        log_error(f"Error removing strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable/{strategy_id}")
async def enable_strategy(strategy_id: str):
    """Enable a strategy"""
    try:
        strategy_executor.enable_strategy(strategy_id)
        return {
            "status": "success",
            "message": f"Strategy {strategy_id} enabled"
        }
    except Exception as e:
        log_error(f"Error enabling strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable/{strategy_id}")
async def disable_strategy(strategy_id: str):
    """Disable a strategy"""
    try:
        strategy_executor.disable_strategy(strategy_id)
        return {
            "status": "success",
            "message": f"Strategy {strategy_id} disabled"
        }
    except Exception as e:
        log_error(f"Error disabling strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_strategy_status():
    """Get status of all strategies and active trades"""
    try:
        status = strategy_executor.get_strategy_status()
        active_trades = strategy_executor.get_active_trades()
        limits_status = trade_limits.get_limits_status()
        
        return {
            "data": {
                "executor_status": status,
                "active_trades": active_trades,
                "trade_limits": limits_status
            }
        }
    except Exception as e:
        log_error(f"Error getting strategy status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active-trades")
async def get_active_trades():
    """Get list of active trades"""
    try:
        trades = strategy_executor.get_active_trades()
        return {"data": trades}
    except Exception as e:
        log_error(f"Error getting active trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_available_strategies():
    """Get list of available strategy types"""
    try:
        strategies = [
            {
                "value": strategy.value,
                "name": strategy.value.replace("_", " ").title(),
                "description": _get_strategy_description(strategy)
            }
            for strategy in StrategyType
        ]
        return {"data": strategies}
    except Exception as e:
        log_error(f"Error getting strategy types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_strategy_description(strategy_type: StrategyType) -> str:
    """Get description for a strategy type"""
    descriptions = {
        StrategyType.CANDLE_BREAK_915: "9:15 AM candle breakout strategy with volume confirmation",
        StrategyType.MEAN_REVERSION: "Mean reversion using Bollinger Bands",
        StrategyType.MOMENTUM_BREAKOUT: "Momentum-based breakout strategy",
        StrategyType.SUPPORT_RESISTANCE: "Support/Resistance level breakout",
        StrategyType.RSI_REVERSAL: "RSI-based reversal strategy",
        StrategyType.MACD_CROSSOVER: "MACD crossover strategy",
        StrategyType.EMA_CROSS: "Exponential moving average crossover"
    }
    return descriptions.get(strategy_type, "Automated trading strategy")
