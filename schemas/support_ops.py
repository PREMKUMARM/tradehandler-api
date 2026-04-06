"""Request bodies for risk, multi-leg execution, and strategy catalog APIs."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class KillSwitchUpdate(BaseModel):
    active: bool


class BasketLeg(BaseModel):
    exchange: str
    tradingsymbol: str
    transaction_type: str
    quantity: int
    order_type: str = "LIMIT"
    product: str = "MIS"
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    strategy_run_id: Optional[str] = None

    @model_validator(mode="after")
    def price_for_order_type(self):
        ot = (self.order_type or "").upper()
        if ot == "LIMIT":
            if self.price is None or float(self.price) <= 0:
                raise ValueError("price is required and must be > 0 for LIMIT orders")
        if ot == "SL":
            if self.price is None or float(self.price) <= 0:
                raise ValueError("price is required for SL orders")
            if self.trigger_price is None or float(self.trigger_price) <= 0:
                raise ValueError("trigger_price is required for SL orders")
        if ot == "SL-M":
            if self.trigger_price is None or float(self.trigger_price) <= 0:
                raise ValueError("trigger_price is required for SL-M orders")
        return self


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
