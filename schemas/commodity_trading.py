"""V2 pre-buy wizard trade execution schemas."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CommodityTradePreviewRequest(BaseModel):
    completed_steps: Optional[List[bool]] = Field(
        default=None,
        min_length=11,
        max_length=11,
        description="Wizard step completion flags (11 steps); ignored when auto_execute",
    )
    auto_execute: bool = Field(
        default=False,
        description="Run all checklist steps server-side and skip manual marks",
    )
    direction: str = Field(
        default="AUTO",
        description="CE, PE, or AUTO (infer from Crude vs prior close)",
    )
    risk_percentage: Optional[float] = Field(default=None, gt=0, le=100)
    reward_percentage: Optional[float] = Field(default=None, gt=0, le=100)
    num_lots: Optional[int] = Field(default=1, ge=1, le=50)
    future_symbol: Optional[str] = Field(
        default=None,
        description="Ignored — app uses CRUDEOILM only (nearest listed month)",
    )


class CommodityTradePlaceRequest(CommodityTradePreviewRequest):
    confirm: bool = Field(default=False, description="Must be true to place live orders")
    trade_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional plan snapshot from preview — avoids re-build mismatch on confirm",
    )


class ChecklistStepStatus(BaseModel):
    index: int
    title: str
    completed: bool
    server_ok: bool
    message: str
    output: Optional[str] = None


class TradePlanOut(BaseModel):
    tradingsymbol: str
    exchange: str = "MCX"
    option_type: str
    strike: int
    expiry: str
    quantity: int
    lot_size: int = 75
    num_lots: int = 1
    product: str = "NRML"
    entry_order_type: str = "LIMIT"
    entry_limit_price: Optional[float] = None
    exit_order_type: str = "GTT_OCO"
    entry_premium: float
    stop_loss_premium: float
    target_premium: float
    indicators: Optional[Dict[str, Any]] = None
    nifty_spot: float
    spot_stop_loss: float
    spot_target: float
    risk_inr: float
    reward_inr: float
    reward_ratio: float
    estimated_premium: bool = False
    note: Optional[str] = None
    strategy_id: Optional[str] = None
    strategy_name: Optional[str] = None
    strike_moneyness: Optional[str] = None
    pattern_tag: Optional[str] = None
    delta_used: Optional[float] = None
    atm_reference: Optional[int] = None
    pricing_note: Optional[str] = None


class StrategyAnalysisOut(BaseModel):
    selected_id: str
    selected_name: str
    selected_score: int
    selected_fit: str
    selected_option_kind: str
    context: Dict[str, Any] = {}
    strategies: List[Dict[str, Any]] = []
    output_summary: str = ""


class CommodityTradePreviewResponse(BaseModel):
    can_place: bool
    checklist_ready: bool
    missing_steps: List[int]
    step_statuses: List[ChecklistStepStatus]
    trade_plan: Optional[TradePlanOut] = None
    validation: Optional[Dict[str, Any]] = None
    messages: List[str] = []
    market_open: bool = False
    allow_test_place: bool = False
    strategy_analysis: Optional[Dict[str, Any]] = None


class CommodityTradePlaceResponse(CommodityTradePreviewResponse):
    placed: bool = False
    entry_order_id: Optional[str] = None
    gtt_trigger_id: Optional[str] = None
    errors: List[str] = []


class CommodityWatchArmRequest(BaseModel):
    direction: str = Field(default="AUTO", description="CE, PE, or AUTO")
    num_lots: Optional[int] = Field(default=1, ge=1, le=50)
    risk_percentage: Optional[float] = Field(default=None, gt=0, le=100)
    reward_percentage: Optional[float] = Field(default=None, gt=0, le=100)
    mode: str = Field(
        default="autonomous",
        description="autonomous = server auto LIMIT+GTT when setup confirms; alert = notify only",
    )
    auto_place_on_signal: bool = Field(
        default=True,
        description="For autonomous mode: place without browser (ignored in alert mode)",
    )
    disarm_after_place: bool = Field(
        default=True,
        description="Stop watch after one successful autonomous order per day",
    )
    auto_execute_checklist: bool = Field(
        default=True,
        description="Run full live checklist server-side while watching",
    )
