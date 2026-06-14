"""Sensex Dhan intraday data viewer API."""
import asyncio

from fastapi import APIRouter, HTTPException, Query, Request

from core.responses import SuccessResponse
from services.sensex_dhan_backtest import list_available_sessions
from services.sensex_dhan_viewer import get_dhan_ohlc, list_dhan_contracts

router = APIRouter(prefix="/sensex/dhan", tags=["Sensex Dhan Data"])


@router.get("/sessions")
async def dhan_sessions(_: Request):
    """Weekly expiry sessions with Dhan cache status."""
    sessions = list_available_sessions()
    return SuccessResponse(
        data={"sessions": sessions, "count": len(sessions)},
        message=f"{len(sessions)} expiry sessions available",
    )


@router.get("/contracts")
async def dhan_contracts(
    _: Request,
    session_date: str = Query(..., description="Expiry session date YYYY-MM-DD"),
    kind: str | None = Query(None, description="CE or PE (omit for both)"),
    refresh: bool = Query(False, description="Re-fetch from Dhan API"),
):
    """List ATM / OTM / ITM rolling contracts for a session."""
    try:
        data = await asyncio.to_thread(
            list_dhan_contracts,
            session_date,
            kind=kind,
            refresh=refresh,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SuccessResponse(data=data, message=f"{data['count']} contracts for {session_date}")


@router.get("/ohlc")
async def dhan_ohlc(
    _: Request,
    session_date: str = Query(..., description="Expiry session date YYYY-MM-DD"),
    kind: str = Query(..., description="CE or PE"),
    offset: str = Query("ATM", description="Rolling offset e.g. ATM, ATM+1, ATM-2"),
    refresh: bool = Query(False, description="Re-fetch from Dhan API"),
):
    """5m OHLC rows for a selected rolling contract."""
    try:
        data = await asyncio.to_thread(
            get_dhan_ohlc,
            session_date,
            kind=kind,
            offset=offset,
            refresh=refresh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SuccessResponse(data=data, message=f"{data['bars']} bars loaded")
