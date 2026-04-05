"""Request bodies for risk, multi-leg execution, and strategy catalog APIs."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
