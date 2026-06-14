"""Sensex 20rupees-strategy backtest via Dhan Data API."""
import asyncio

from fastapi import APIRouter, HTTPException, Request

from core.responses import SuccessResponse
from schemas.sensex_backtest import SensexBacktestRunRequest
from services.sensex_dhan_backtest import (
    BacktestParams,
    check_dhan_status,
    list_available_sessions,
    run_sensex_dhan_backtest,
)

router = APIRouter(prefix="/sensex/backtest", tags=["Sensex Backtest"])


@router.get("/sessions")
async def backtest_sessions(_: Request):
    """Available weekly expiry sessions with Dhan cache status."""
    sessions = list_available_sessions()
    return SuccessResponse(
        data={"sessions": sessions, "count": len(sessions)},
        message=f"{len(sessions)} expiry sessions available",
    )


@router.get("/status")
async def backtest_dhan_status(_: Request):
    """Dhan Data API subscription status."""
    try:
        status = check_dhan_status()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SuccessResponse(data=status, message="Dhan status loaded")


@router.post("/run")
async def run_backtest(_: Request, body: SensexBacktestRunRequest):
    """Run 20rupees-strategy backtest on Dhan 5m rolling options."""
    params = BacktestParams(
        capital=body.capital,
        risk_pct=body.risk_pct,
        sl_inr=body.sl_inr,
        entry_band_low=body.entry_band_low,
        entry_band_high=body.entry_band_high,
        min_target_low=body.min_target_low,
        min_target_high=body.min_target_high,
        direction=body.direction,
        mode=body.mode,
        expiry_dates=body.expiry_dates,
        refresh_dhan=body.refresh_dhan,
    )
    try:
        result = await asyncio.to_thread(run_sensex_dhan_backtest, params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc
    return SuccessResponse(data=result, message="Backtest complete")
