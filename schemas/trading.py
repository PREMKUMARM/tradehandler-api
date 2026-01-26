"""
Trading-related request/response schemas
"""
from typing import Optional
from pydantic import BaseModel, Field


class TradeValidationRequest(BaseModel):
    """Request schema for trade validation"""
    entry_price: float = Field(..., gt=0, description="Entry price")
    stoploss: float = Field(..., gt=0, description="Stop loss price")
    target_price: float = Field(..., gt=0, description="Target price")
    quantity: int = Field(..., gt=0, description="Quantity")
    capital: float = Field(..., gt=0, description="Total capital/margin")
    risk_percentage: float = Field(..., gt=0, le=100, description="Risk percentage per trade (e.g., 1.0 for 1%)")
    reward_percentage: float = Field(..., gt=0, le=100, description="Reward percentage per trade (e.g., 2.0 for 2%)")


class TradeValidationResponse(BaseModel):
    """Response schema for trade validation"""
    is_good_trade: bool = Field(..., description="Whether the trade meets validation criteria")
    risk_amount: float = Field(..., description="Actual risk amount (entry - stoploss) × quantity")
    reward_amount: float = Field(..., description="Actual reward amount (target - entry) × quantity")
    max_risk_amount: float = Field(..., description="Maximum allowed risk (capital × risk%)")
    min_required_reward: float = Field(..., description="Minimum required reward (risk × reward ratio)")
    reward_ratio: float = Field(..., description="Reward ratio (reward% / risk%)")
    risk_within_limit: bool = Field(..., description="Whether risk is within configured limit")
    reward_meets_requirement: bool = Field(..., description="Whether reward meets minimum requirement")
    validation_details: dict = Field(..., description="Detailed validation information")

