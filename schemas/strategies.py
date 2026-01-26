"""
Strategy backtest request/response schemas
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import date


class Nifty50OptionsBacktestRequest(BaseModel):
    """Request schema for Nifty50 options backtest"""
    start_date: date = Field(..., description="Start date for backtest (YYYY-MM-DD)")
    end_date: date = Field(..., description="End date for backtest (YYYY-MM-DD)")
    strategy_type: str = Field(default="915_candle_break", description="Strategy type")
    fund: float = Field(default=200000, gt=0, description="Trading capital")
    risk: float = Field(default=1, gt=0, le=100, description="Risk percentage (1-100)")
    reward: float = Field(default=3, gt=0, description="Reward ratio")

    @validator("end_date")
    def validate_date_range(cls, v, values):
        if "start_date" in values and v < values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class RangeBreakout30MinBacktestRequest(BaseModel):
    """Request schema for 30-minute range breakout backtest"""
    instrument: str = Field(..., min_length=1, description="Instrument name or symbol")
    from_date: date = Field(..., description="Start date (YYYY-MM-DD)")
    to_date: date = Field(..., description="End date (YYYY-MM-DD)")
    capital: float = Field(..., gt=0, description="Trading capital")
    risk_percent: float = Field(default=1.0, gt=0, le=100, description="Risk percentage per trade")
    reward_ratio: float = Field(default=2.0, gt=0, description="Reward to risk ratio")
    
    # Strategy configuration flags
    enable_options_selling: bool = Field(default=False, description="Enable options selling")
    enable_gap_filter: bool = Field(default=True, description="Enable gap direction filter")
    enable_exhaustion_candle_filter: bool = Field(default=True, description="Enable exhaustion candle filter")
    
    # Segment-specific filters
    equity_gap_filter: bool = Field(default=True, description="Gap filter for equity segment")
    equity_exhaustion_filter: bool = Field(default=True, description="Exhaustion filter for equity segment")
    index_gap_filter: bool = Field(default=True, description="Gap filter for index segment")
    index_exhaustion_filter: bool = Field(default=True, description="Exhaustion filter for index segment")
    fno_gap_filter: bool = Field(default=True, description="Gap filter for FnO segment")
    fno_exhaustion_filter: bool = Field(default=True, description="Exhaustion filter for FnO segment")

    @validator("to_date")
    def validate_date_range(cls, v, values):
        if "from_date" in values and v < values["from_date"]:
            raise ValueError("to_date must be after from_date")
        return v


class VWAPStrategyBacktestRequest(BaseModel):
    """Request schema for VWAP strategy backtest"""
    instrument: str = Field(..., min_length=1, description="Instrument name or symbol")
    from_date: date = Field(..., description="Start date (YYYY-MM-DD)")
    to_date: date = Field(..., description="End date (YYYY-MM-DD)")
    capital: float = Field(..., gt=0, description="Trading capital")
    risk_percent: float = Field(default=1.0, gt=0, le=100, description="Risk percentage per trade")
    proximity_percent: float = Field(default=0.5, gt=0, description="VWAP proximity percentage")

    @validator("to_date")
    def validate_date_range(cls, v, values):
        if "from_date" in values and v < values["from_date"]:
            raise ValueError("to_date must be after from_date")
        return v


class BinanceFuturesBacktestRequest(BaseModel):
    """Request schema for Binance futures backtest"""
    symbol: str = Field(..., min_length=1, description="Trading symbol (e.g., BTCUSDT)")
    from_date: date = Field(..., description="Start date (YYYY-MM-DD)")
    to_date: date = Field(..., description="End date (YYYY-MM-DD)")
    capital: float = Field(..., gt=0, description="Trading capital")
    risk_percent: float = Field(default=1.0, gt=0, le=100, description="Risk percentage per trade")

    @validator("to_date")
    def validate_date_range(cls, v, values):
        if "from_date" in values and v < values["from_date"]:
            raise ValueError("to_date must be after from_date")
        return v


class BacktestResult(BaseModel):
    """Response schema for backtest results"""
    date: str = Field(..., description="Trading date")
    instrument: Optional[str] = Field(None, description="Instrument name")
    direction: Optional[str] = Field(None, description="Trade direction (LONG/SHORT)")
    entry_price: Optional[float] = Field(None, description="Entry price")
    exit_price: Optional[float] = Field(None, description="Exit price")
    quantity: Optional[int] = Field(None, description="Trade quantity")
    profit: Optional[float] = Field(None, description="Profit/Loss")
    gap_status: Optional[str] = Field(None, description="Gap status")
    buy_cost: Optional[float] = Field(None, description="Buy cost (qty * entry_price)")
    sell_cost: Optional[float] = Field(None, description="Sell cost (qty * exit_price)")
    expected_profit: Optional[float] = Field(None, description="Expected profit if target hit")
    expected_loss: Optional[float] = Field(None, description="Expected loss if stop loss hit")


class BacktestSummary(BaseModel):
    """Response schema for backtest summary"""
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    win_rate: float = Field(..., description="Win rate percentage")
    total_profit: float = Field(..., description="Total profit/loss")
    max_profit: float = Field(..., description="Maximum single trade profit")
    max_loss: float = Field(..., description="Maximum single trade loss")
    avg_profit: float = Field(..., description="Average profit per trade")
    profit_factor: Optional[float] = Field(None, description="Profit factor (gross profit / gross loss)")

