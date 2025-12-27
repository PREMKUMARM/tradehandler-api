"""
Market data request/response schemas
"""
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from datetime import date


class QuoteRequest(BaseModel):
    """Request schema for getting quote"""
    exchange: str = Field(default="NSE", description="Exchange")
    symbol: str = Field(..., min_length=1, description="Trading symbol")


class HistoricalDataRequest(BaseModel):
    """Request schema for historical data"""
    exchange: str = Field(default="NSE", description="Exchange")
    symbol: str = Field(..., min_length=1, description="Trading symbol")
    interval: str = Field(..., description="Time interval (minute, day, etc.)")
    from_date: date = Field(..., description="Start date")
    to_date: date = Field(..., description="End date")

    @validator("interval")
    def validate_interval(cls, v):
        valid_intervals = ["minute", "day", "3minute", "5minute", "15minute", "30minute", "60minute"]
        if v.lower() not in valid_intervals:
            raise ValueError(f"interval must be one of {valid_intervals}")
        return v.lower()

    @validator("to_date")
    def validate_date_range(cls, v, values):
        if "from_date" in values and v < values["from_date"]:
            raise ValueError("to_date must be after from_date")
        return v


class PositionsResponse(BaseModel):
    """Response schema for positions"""
    net_positions: List[dict] = Field(default_factory=list, description="Net positions")
    total_pnl: float = Field(default=0.0, description="Total P&L")
    active_count: int = Field(default=0, description="Number of active positions")

