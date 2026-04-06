"""Request bodies for risk, multi-leg execution, and strategy catalog APIs."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class KillSwitchUpdate(BaseModel):
    active: bool


class BasketLeg(BaseModel):
    exchange: str
    tradingsymbol: str
    transaction_type: str
    quantity: int
    order_type: str = "MARKET"
    product: str = "MIS"
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    strategy_run_id: Optional[str] = None
    stoploss: Optional[float] = Field(None, gt=0, description="Paper mode: stop-loss price for auto-exit")
    target: Optional[float] = Field(None, gt=0, description="Paper mode: target price for auto-exit")
    trailing_stoploss: Optional[float] = Field(None, gt=0, description="Paper mode: trailing amount")


class BasketPlaceRequest(BaseModel):
    legs: List[BasketLeg] = Field(..., min_length=1)


class StrategyDefinitionIn(BaseModel):
    id: Optional[str] = None
    name: str
    spec: Dict[str, Any] = Field(default_factory=dict)


class StrategyRunIn(BaseModel):
    definition_id: Optional[str] = None
    mode: str = "paper"
    meta: Dict[str, Any] = Field(default_factory=dict)


class RunStatusPatch(BaseModel):
    status: str
    ended: bool = True


class StrategyFillIn(BaseModel):
    broker_order_id: str
    tradingsymbol: str
    side: str
    quantity: int = Field(..., gt=0)
    price: Optional[float] = None

    @field_validator("side")
    @classmethod
    def side_upper(cls, v: str) -> str:
        u = (v or "").upper()
        if u not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        return u
