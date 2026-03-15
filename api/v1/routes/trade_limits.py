"""
Trade limits management API endpoints
"""
from fastapi import APIRouter, Request
from typing import Optional
from pydantic import BaseModel, Field

from utils.trade_limits import trade_limits
from utils.logger import log_info

router = APIRouter(prefix="/trade-limits", tags=["Trade Limits"])


class LimitsConfig(BaseModel):
    max_trades_per_day: Optional[int] = Field(None, gt=0, description="Maximum trades per day")
    max_profit_per_day_pct: Optional[float] = Field(None, gt=0, description="Maximum profit percentage per day")
    max_loss_per_day_pct: Optional[float] = Field(None, gt=0, description="Maximum loss percentage per day")


@router.get("/status")
def get_limits_status():
    """Get current trade limits status"""
    try:
        status = trade_limits.get_limits_status()
        return {"data": status}
    except Exception as e:
        log_info(f"Error getting limits status: {e}")
        return {"error": str(e)}


@router.post("/configure")
def configure_limits(config: LimitsConfig):
    """Configure trade limits"""
    try:
        trade_limits.update_limits(
            max_trades=config.max_trades_per_day,
            max_profit_pct=config.max_profit_per_day_pct,
            max_loss_pct=config.max_loss_per_day_pct
        )
        return {"status": "success", "message": "Trade limits updated successfully"}
    except Exception as e:
        log_info(f"Error configuring limits: {e}")
        return {"error": str(e)}


@router.post("/reset")
def reset_daily_limits():
    """Reset daily trade limits"""
    try:
        trade_limits.reset_daily_limits()
        return {"status": "success", "message": "Daily trade limits reset"}
    except Exception as e:
        log_info(f"Error resetting limits: {e}")
        return {"error": str(e)}
