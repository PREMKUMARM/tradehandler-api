"""V2 wizard trade preview and execution."""
from fastapi import APIRouter, Request
from core.responses import SuccessResponse
from schemas.v2_trading import V2TradePlaceRequest, V2TradePreviewRequest
from services import v2_trade_service

router = APIRouter(prefix="/v2/trade", tags=["V2 Trading"])


@router.post("/preview")
async def preview_v2_trade(request: Request, body: V2TradePreviewRequest):
    """Validate checklist and build Nifty GTT trade plan without placing orders."""
    data = v2_trade_service.preview_trade(
        completed_steps=body.completed_steps,
        direction=body.direction,
        risk_percentage=body.risk_percentage,
        reward_percentage=body.reward_percentage,
        num_lots=body.num_lots or 1,
        auto_execute=body.auto_execute,
    )
    return SuccessResponse(data=data, message="Trade preview ready")


@router.post("/place")
async def place_v2_trade(request: Request, body: V2TradePlaceRequest):
    """Validate checklist, then place MARKET entry + GTT OCO exit on NFO."""
    data = v2_trade_service.place_trade(
        completed_steps=body.completed_steps,
        direction=body.direction,
        risk_percentage=body.risk_percentage,
        reward_percentage=body.reward_percentage,
        num_lots=body.num_lots or 1,
        confirm=body.confirm,
        auto_execute=body.auto_execute,
    )
    msg = "Order placed" if data.get("placed") else "Trade not placed"
    return SuccessResponse(data=data, message=msg)
