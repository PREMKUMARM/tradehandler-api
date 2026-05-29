"""Binance BTCUSDT futures wizard API."""
from fastapi import APIRouter, HTTPException, Request

from core.responses import SuccessResponse
from schemas.crypto_trading import CryptoTradePlaceRequest, CryptoTradePreviewRequest, CryptoWatchArmRequest
from services import crypto_trade_service
from services.crypto_config import DEFAULT_LEVERAGE, SYMBOL
from services.crypto_strategy_watch import (
    arm_watch,
    disarm_watch,
    get_watch_events,
    get_watch_status,
    nuclear_reset_watch,
)

router = APIRouter(prefix="/crypto/trade", tags=["Crypto Trading"])


@router.get("/balance")
async def crypto_balance(request: Request):
    """USDT balance + last BTC spot (REST fallback; prefer segment WebSocket stream)."""
    from services.segment_balance import get_crypto_balance_payload

    data = get_crypto_balance_payload()
    return SuccessResponse(
        data=data,
        message="Binance balance" if data.get("connected") else "Binance not connected",
    )


@router.get("/config")
async def crypto_config(request: Request):
    return SuccessResponse(
        data={
            "symbol": SYMBOL,
            "leverage": DEFAULT_LEVERAGE,
            "exchange": "BINANCE",
        },
        message="Crypto config",
    )


@router.get("/checklist-live")
async def checklist_live_crypto(
    request: Request,
    direction: str = "AUTO",
    quantity_btc: float = 0.001,
):
    data = crypto_trade_service.get_checklist_live(
        direction=direction,
        quantity_btc=quantity_btc,
    )
    return SuccessResponse(data=data, message="Live checklist refreshed")


@router.get("/checklist-analyze")
async def checklist_analyze_crypto(
    request: Request,
    step: int,
    direction: str = "AUTO",
    quantity_btc: float = 0.001,
):
    data = crypto_trade_service.get_checklist_analyze(
        step=step,
        direction=direction,
        quantity_btc=quantity_btc,
    )
    return SuccessResponse(data=data, message="Step analysis ready")


@router.get("/strategy-analysis")
async def strategy_analysis_crypto(request: Request, direction: str = "AUTO"):
    data = crypto_trade_service.get_strategy_analysis(direction=direction)
    return SuccessResponse(data=data, message="Strategy analysis ready")


@router.post("/preview")
async def preview_crypto_trade(request: Request, body: CryptoTradePreviewRequest):
    data = crypto_trade_service.preview_trade(
        completed_steps=body.completed_steps,
        direction=body.direction,
        risk_percentage=body.risk_percentage,
        reward_percentage=body.reward_percentage,
        quantity_btc=body.quantity_btc or 0.001,
        auto_execute=body.auto_execute,
    )
    return SuccessResponse(data=data, message="Trade preview ready")


@router.post("/place")
async def place_crypto_trade(request: Request, body: CryptoTradePlaceRequest):
    data = crypto_trade_service.place_trade(
        completed_steps=body.completed_steps,
        direction=body.direction,
        risk_percentage=body.risk_percentage,
        reward_percentage=body.reward_percentage,
        quantity_btc=body.quantity_btc or 0.001,
        confirm=body.confirm,
        auto_execute=body.auto_execute,
        trade_plan_snapshot=body.trade_plan,
    )
    msg = "Order placed" if data.get("placed") else "Trade not placed"
    return SuccessResponse(data=data, message=msg)


@router.post("/watch/arm")
async def arm_crypto_watch(request: Request, body: CryptoWatchArmRequest):
    try:
        data = arm_watch(
            direction=body.direction,
            quantity_btc=body.quantity_btc or 0.001,
            mode=body.mode,
            auto_place_on_signal=body.auto_place_on_signal,
            auto_execute_checklist=body.auto_execute_checklist,
            disarm_after_place=body.disarm_after_place,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SuccessResponse(data=data, message="Crypto watch armed")


@router.post("/watch/disarm")
async def disarm_crypto_watch(request: Request):
    return SuccessResponse(data=disarm_watch(), message="Crypto watch disarmed")


@router.post("/watch/nuclear-reset")
async def nuclear_reset_crypto_watch(request: Request):
    return SuccessResponse(data=nuclear_reset_watch(), message="Crypto watch reset")


@router.get("/watch/status")
async def crypto_watch_status(request: Request):
    return SuccessResponse(data=get_watch_status(), message="Watch status")


@router.get("/watch/events")
async def crypto_watch_events(request: Request, limit: int = 20):
    return SuccessResponse(data=get_watch_events(limit), message="Watch events")
