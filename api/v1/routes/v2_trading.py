"""V2 wizard trade preview and execution."""
from fastapi import APIRouter, HTTPException, Request
from core.responses import SuccessResponse
from schemas.v2_trading import V2TradePlaceRequest, V2TradePreviewRequest, V2WatchArmRequest
from services import v2_trade_service
from services.v2_strategy_watch import (
    arm_watch,
    disarm_watch,
    get_watch_events,
    get_watch_status,
    watch_autonomous_allowed,
    watch_auto_place_allowed,
)

router = APIRouter(prefix="/v2/trade", tags=["V2 Trading"])


@router.get("/checklist-live")
async def checklist_live_v2(
    request: Request,
    direction: str = "AUTO",
    risk_percentage: float | None = None,
    reward_percentage: float | None = None,
    num_lots: int = 1,
):
    """All 12 checklist steps from live Kite ticker + quotes + chain OI."""
    data = v2_trade_service.get_checklist_live(
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        num_lots=num_lots,
    )
    return SuccessResponse(data=data, message="Live checklist refreshed")


@router.get("/checklist-analyze")
async def checklist_analyze_v2(
    request: Request,
    step: int,
    direction: str = "AUTO",
    risk_percentage: float | None = None,
    reward_percentage: float | None = None,
    num_lots: int = 1,
):
    """Realtime analysis for one step and its prerequisite steps (0-based index)."""
    data = v2_trade_service.get_checklist_analyze(
        step=step,
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        num_lots=num_lots,
    )
    return SuccessResponse(data=data, message="Step analysis ready")


@router.get("/strategy-analysis")
async def strategy_analysis_v2(request: Request, direction: str = "AUTO"):
    """Rank top 4 Nifty F&O strategies using session data from prior checklist steps."""
    data = v2_trade_service.get_strategy_analysis(direction=direction)
    return SuccessResponse(data=data, message="Strategy analysis ready")


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
    """Validate checklist, place entry (LIMIT or MARKET+protection), then GTT OCO exit on NFO."""
    data = v2_trade_service.place_trade(
        completed_steps=body.completed_steps,
        direction=body.direction,
        risk_percentage=body.risk_percentage,
        reward_percentage=body.reward_percentage,
        num_lots=body.num_lots or 1,
        confirm=body.confirm,
        auto_execute=body.auto_execute,
        trade_plan_snapshot=body.trade_plan,
    )
    msg = "Order placed" if data.get("placed") else "Trade not placed"
    return SuccessResponse(data=data, message=msg)


@router.post("/watch/arm")
async def arm_v2_watch(request: Request, body: V2WatchArmRequest):
    """
    Arm background strategy watch (Streak-style deploy).

    Polls live entry_ready every few seconds; fires push + WebSocket on signal.
    """
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
async def disarm_v2_watch(request: Request):
    data = disarm_watch()
    return SuccessResponse(data=data, message="Strategy watch stopped")


@router.get("/watch/status")
async def watch_status_v2(request: Request):
    data = get_watch_status()
    data["autonomous_server_allowed"] = watch_autonomous_allowed()
    data["auto_place_server_allowed"] = watch_auto_place_allowed()
    return SuccessResponse(data=data, message="Watch status")


@router.get("/watch/events")
async def watch_events_v2(request: Request, limit: int = 20):
    data = {"events": get_watch_events(limit=limit)}
    return SuccessResponse(data=data, message="Watch events")
