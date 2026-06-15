"""Sensex wizard trade preview and execution."""
from fastapi import APIRouter, HTTPException, Request
from core.responses import SuccessResponse
from schemas.sensex_trading import SensexTradePlaceRequest, SensexTradePreviewRequest, SensexWatchArmRequest
from services import sensex_trade_service
from services.sensex_run_params import SensexRunParams
from services.sensex_strategy_watch import (
    arm_watch,
    disarm_watch,
    get_watch_events,
    get_watch_status,
    nuclear_reset_watch,
    watch_autonomous_allowed,
    watch_auto_place_allowed,
)

router = APIRouter(prefix="/sensex/trade", tags=["Sensex Trading"])


def _run_params_kwargs(**kwargs) -> dict:
    return SensexRunParams.from_mapping(kwargs, direction=kwargs.get("direction")).to_dict()


@router.get("/checklist-live")
async def checklist_live_sensex(
    request: Request,
    direction: str = "AUTO",
    risk_percentage: float | None = None,
    reward_percentage: float | None = None,
    num_lots: int = 1,
    capital: float | None = None,
    risk_pct: float | None = None,
    sl_inr: float | None = None,
    entry_band_low: float | None = None,
    entry_band_high: float | None = None,
    min_target_low: float | None = None,
    min_target_high: float | None = None,
    entry_scan_start_ist: str | None = None,
    entry_scan_end_ist: str | None = None,
):
    """All 12 checklist steps from live Kite ticker + quotes + chain OI."""
    data = sensex_trade_service.get_checklist_live(
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        num_lots=num_lots,
        run_params=_run_params_kwargs(
            direction=direction,
            capital=capital,
            risk_pct=risk_pct,
            risk_percentage=risk_percentage,
            sl_inr=sl_inr,
            entry_band_low=entry_band_low,
            entry_band_high=entry_band_high,
            min_target_low=min_target_low,
            min_target_high=min_target_high,
            entry_scan_start_ist=entry_scan_start_ist,
            entry_scan_end_ist=entry_scan_end_ist,
            num_lots=num_lots,
        ),
    )
    return SuccessResponse(data=data, message="Live checklist refreshed")


@router.get("/checklist-analyze")
async def checklist_analyze_sensex(
    request: Request,
    step: int,
    direction: str = "AUTO",
    risk_percentage: float | None = None,
    reward_percentage: float | None = None,
    num_lots: int = 1,
    capital: float | None = None,
    risk_pct: float | None = None,
    sl_inr: float | None = None,
    entry_band_low: float | None = None,
    entry_band_high: float | None = None,
    min_target_low: float | None = None,
    min_target_high: float | None = None,
    entry_scan_start_ist: str | None = None,
    entry_scan_end_ist: str | None = None,
):
    """Realtime analysis for one step and its prerequisite steps (0-based index)."""
    data = sensex_trade_service.get_checklist_analyze(
        step=step,
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        num_lots=num_lots,
        run_params=_run_params_kwargs(
            direction=direction,
            capital=capital,
            risk_pct=risk_pct,
            risk_percentage=risk_percentage,
            sl_inr=sl_inr,
            entry_band_low=entry_band_low,
            entry_band_high=entry_band_high,
            min_target_low=min_target_low,
            min_target_high=min_target_high,
            entry_scan_start_ist=entry_scan_start_ist,
            entry_scan_end_ist=entry_scan_end_ist,
            num_lots=num_lots,
        ),
    )
    return SuccessResponse(data=data, message="Step analysis ready")


@router.get("/strategy-analysis")
async def strategy_analysis_sensex(
    request: Request,
    direction: str = "AUTO",
    capital: float | None = None,
    risk_pct: float | None = None,
    sl_inr: float | None = None,
    entry_band_low: float | None = None,
    entry_band_high: float | None = None,
    min_target_low: float | None = None,
    min_target_high: float | None = None,
    entry_scan_start_ist: str | None = None,
    entry_scan_end_ist: str | None = None,
):
    """Rank top 4 Sensex F&O strategies using session data from prior checklist steps."""
    data = sensex_trade_service.get_strategy_analysis(
        direction=direction,
        run_params=_run_params_kwargs(
            direction=direction,
            capital=capital,
            risk_pct=risk_pct,
            sl_inr=sl_inr,
            entry_band_low=entry_band_low,
            entry_band_high=entry_band_high,
            min_target_low=min_target_low,
            min_target_high=min_target_high,
            entry_scan_start_ist=entry_scan_start_ist,
            entry_scan_end_ist=entry_scan_end_ist,
        ),
    )
    return SuccessResponse(data=data, message="Strategy analysis ready")


@router.post("/preview")
async def preview_sensex_trade(request: Request, body: SensexTradePreviewRequest):
    """Validate checklist and build Sensex GTT trade plan without placing orders."""
    data = sensex_trade_service.preview_trade(
        completed_steps=body.completed_steps,
        direction=body.direction,
        risk_percentage=body.risk_percentage,
        reward_percentage=body.reward_percentage,
        num_lots=body.num_lots or 1,
        auto_execute=body.auto_execute,
        run_params=SensexRunParams.from_mapping(body.to_resolve_kwargs(), direction=body.direction).to_dict(),
    )
    return SuccessResponse(data=data, message="Trade preview ready")


@router.post("/place")
async def place_sensex_trade(request: Request, body: SensexTradePlaceRequest):
    """Validate checklist, place entry (LIMIT or MARKET+protection), then GTT OCO exit on NFO."""
    data = sensex_trade_service.place_trade(
        completed_steps=body.completed_steps,
        direction=body.direction,
        risk_percentage=body.risk_percentage,
        reward_percentage=body.reward_percentage,
        num_lots=body.num_lots or 1,
        confirm=body.confirm,
        auto_execute=body.auto_execute,
        trade_plan_snapshot=body.trade_plan,
        run_params=SensexRunParams.from_mapping(body.to_resolve_kwargs(), direction=body.direction).to_dict(),
    )
    msg = "Order placed" if data.get("placed") else "Trade not placed"
    return SuccessResponse(data=data, message=msg)


@router.post("/watch/arm")
async def arm_sensex_watch(request: Request, body: SensexWatchArmRequest):
    """
    Arm background strategy watch (Streak-style deploy).

    Polls live entry_ready every few seconds; fires push + WebSocket on signal.
    """
    try:
        rp = SensexRunParams.from_mapping(body.to_resolve_kwargs(), direction=body.direction)
        data = arm_watch(
            direction=body.direction,
            num_lots=body.num_lots or rp.num_lots,
            risk_percentage=body.risk_percentage or rp.risk_pct,
            reward_percentage=body.reward_percentage,
            mode=body.mode,
            auto_place_on_signal=body.auto_place_on_signal,
            auto_execute_checklist=body.auto_execute_checklist,
            disarm_after_place=body.disarm_after_place,
            run_params=rp.to_dict(),
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
async def disarm_sensex_watch(request: Request):
    data = disarm_watch()
    return SuccessResponse(data=data, message="Strategy watch stopped")


@router.post("/watch/nuclear-reset")
async def nuclear_reset_sensex_watch(request: Request):
    data = nuclear_reset_watch()
    return SuccessResponse(
        data=data,
        message="Sensex watch state reset — event log and daily counters cleared",
    )


@router.get("/watch/status")
async def watch_status_sensex(request: Request):
    data = get_watch_status()
    data["autonomous_server_allowed"] = watch_autonomous_allowed()
    data["auto_place_server_allowed"] = watch_auto_place_allowed()
    return SuccessResponse(data=data, message="Watch status")


@router.get("/watch/events")
async def watch_events_sensex(request: Request, limit: int = 20):
    data = {"events": get_watch_events(limit=limit)}
    return SuccessResponse(data=data, message="Watch events")
