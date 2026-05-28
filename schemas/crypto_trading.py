"""Crypto (Binance BTCUSDT) wizard schemas."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CryptoTradePreviewRequest(BaseModel):
    completed_steps: Optional[List[bool]] = Field(default=None, min_length=8, max_length=8)
    auto_execute: bool = False
    direction: str = Field(default="AUTO", description="LONG, SHORT, or AUTO")
    risk_percentage: Optional[float] = Field(default=None, gt=0, le=100)
    reward_percentage: Optional[float] = Field(default=None, gt=0, le=100)
    quantity_btc: Optional[float] = Field(default=0.001, gt=0, le=10)


class CryptoTradePlaceRequest(CryptoTradePreviewRequest):
    confirm: bool = False
    trade_plan: Optional[Dict[str, Any]] = None


class CryptoWatchArmRequest(BaseModel):
    direction: str = "AUTO"
    quantity_btc: Optional[float] = Field(default=0.001, gt=0, le=10)
    mode: str = "autonomous"
    auto_place_on_signal: bool = True
    disarm_after_place: bool = True
    auto_execute_checklist: bool = True
