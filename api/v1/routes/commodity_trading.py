"""Commodity wizard trade preview and execution."""
from fastapi import APIRouter, HTTPException, Request
from core.responses import SuccessResponse
from schemas.commodity_trading import CommodityTradePlaceRequest, CommodityTradePreviewRequest, CommodityWatchArmRequest
from services import commodity_trade_service
from services.commodity_strategy_watch import (
    arm_watch,
    disarm_watch,
    get_watch_events,
    get_watch_status,
    watch_autonomous_allowed,
    watch_auto_place_allowed,
)

router = APIRouter(prefix="/commodity/trade", tags=["Commodity Trading"])


@router.get("/scan-affordable")
async def scan_affordable_commodity(request: Request, direction: str = "AUTO"):
    """CRUDEOILM premium cost vs available commodity margin."""
    from services.commodity_affordable_scan import scan_affordable_commodities

    data = scan_affordable_commodities(direction=direction)
    return SuccessResponse(data=data, message="Affordable commodity scan complete")


@router.get("/checklist-live")
async def checklist_live_commodity(
    request: Request,
    direction: str = "AUTO",
    risk_percentage: float | None = None,
    reward_percentage: float | None = None,
    num_lots: int = 1,
):
    """All 11 checklist steps from live MCX Crude ticker + quotes + chain OI."""
    data = commodity_trade_service.get_checklist_live(
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        num_lots=num_lots,
    )
    return SuccessResponse(data=data, message="Live checklist refreshed")


@router.get("/checklist-analyze")
async def checklist_analyze_commodity(
    request: Request,
    step: int,
    direction: str = "AUTO",
    risk_percentage: float | None = None,
    reward_percentage: float | None = None,
    num_lots: int = 1,
):
    """Realtime analysis for one step and its prerequisite steps (0-based index)."""
    data = commodity_trade_service.get_checklist_analyze(
        step=step,
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        num_lots=num_lots,
    )
    return SuccessResponse(data=data, message="Step analysis ready")


@router.get("/strategy-analysis")
async def strategy_analysis_commodity(request: Request, direction: str = "AUTO"):
    """Rank top Crude F&O strategies using session data from prior checklist steps."""
    data = commodity_trade_service.get_strategy_analysis(direction=direction)
    return SuccessResponse(data=data, message="Strategy analysis ready")


@router.post("/preview")
async def preview_commodity_trade(request: Request, body: CommodityTradePreviewRequest):
    """Validate checklist and build MCX GTT trade plan without placing orders."""
    data = commodity_trade_service.preview_trade(
        completed_steps=body.completed_steps,
        direction=body.direction,
        risk_percentage=body.risk_percentage,
        reward_percentage=body.reward_percentage,
        num_lots=body.num_lots or 1,
        auto_execute=body.auto_execute,
        future_symbol=body.future_symbol,
    )
    return SuccessResponse(data=data, message="Trade preview ready")


@router.post("/place")
async def place_commodity_trade(request: Request, body: CommodityTradePlaceRequest):
    """Validate checklist, place entry LIMIT; GTT OCO attaches after entry fills."""
    data = commodity_trade_service.place_trade(
        completed_steps=body.completed_steps,
        direction=body.direction,
        risk_percentage=body.risk_percentage,
        reward_percentage=body.reward_percentage,
        num_lots=body.num_lots or 1,
        future_symbol=body.future_symbol,
        confirm=body.confirm,
        auto_execute=body.auto_execute,
        trade_plan_snapshot=body.trade_plan,
    )
    msg = "Order placed" if data.get("placed") else "Trade not placed"
    return SuccessResponse(data=data, message=msg)


@router.post("/watch/arm")
async def arm_commodity_watch(request: Request, body: CommodityWatchArmRequest):
    """Arm background Crude strategy watch (poll entry_ready, alert / auto-place)."""
    try:
        data = arm_watch(
            direction=body.direction,
            num_lots=body.num_lots or 1,
            risk_percentage=body.risk_percentage,
            reward_percentage=body.reward_percentage,
            mode=body.mode,
            auto_place_on_signal=body.auto_place_on_signal,
            auto_execute_checklist=body.auto_execute_checklist,
            disarm_after_place=body.disarm_after_place,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data["autonomous_server_allowed"] = watch_autonomous_allowed()
    data["auto_place_server_allowed"] = watch_auto_place_allowed()
    msg = (
        "Autonomous trading started — server will place when setup confirms"
        if data.get("autonomous")
        else "Strategy watch armed"
    )
    return SuccessResponse(data=data, message=msg)


@router.post("/watch/disarm")
async def disarm_commodity_watch(request: Request):
    data = disarm_watch()
    return SuccessResponse(data=data, message="Strategy watch stopped")


@router.get("/watch/status")
async def watch_status_commodity(request: Request):
    data = get_watch_status()
    data["autonomous_server_allowed"] = watch_autonomous_allowed()
    data["auto_place_server_allowed"] = watch_auto_place_allowed()
    return SuccessResponse(data=data, message="Watch status")


@router.get("/watch/events")
async def watch_events_commodity(request: Request, limit: int = 20):
    data = {"events": get_watch_events(limit=limit)}
    return SuccessResponse(data=data, message="Watch events")
