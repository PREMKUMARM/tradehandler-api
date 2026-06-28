"""Dhan intraday rolling-option viewer API (Sensex cache + Nifty DuckDB)."""
import asyncio

from fastapi import APIRouter, HTTPException, Query, Request

from core.responses import SuccessResponse
from services.dhan_viewer import get_dhan_ohlc, list_dhan_contracts, list_dhan_sessions

router = APIRouter(prefix="/sensex/dhan", tags=["Sensex Dhan Data"])


@router.get("/sessions")
async def dhan_sessions(
    _: Request,
    segment: str = Query("sensex", description="sensex or nifty50"),
):
    """Trading sessions with Dhan cache status."""
    sessions = list_dhan_sessions(segment)
    return SuccessResponse(
        data={"sessions": sessions, "count": len(sessions), "segment": segment},
        message=f"{len(sessions)} sessions available ({segment})",
    )


@router.get("/contracts")
async def dhan_contracts(
    _: Request,
    session_date: str = Query(..., description="Expiry session date YYYY-MM-DD"),
    kind: str | None = Query(None, description="CE or PE (omit for both)"),
    refresh: bool = Query(False, description="Re-fetch from Dhan API"),
    segment: str = Query("sensex", description="sensex or nifty50"),
):
    """List ATM / OTM / ITM rolling contracts for a session."""
    try:
        data = await asyncio.to_thread(
            list_dhan_contracts,
            segment,
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
    segment: str = Query("sensex", description="sensex or nifty50"),
    entry: float | None = Query(None, description="Trade entry premium for chart overlay"),
    exit: float | None = Query(None, description="Trade exit premium for chart overlay"),
    sl: float | None = Query(None, description="Stop-loss premium for chart overlay"),
    target: float | None = Query(None, description="Target premium for chart overlay"),
    entry_datetime_ist: str | None = Query(None, description="Entry bar time for marker"),
    exit_datetime_ist: str | None = Query(None, description="Exit bar time for marker"),
    exit_reason: str | None = Query(None, description="Exit reason label"),
    pnl_inr: float | None = Query(None, description="Trade P&L for chart subtitle"),
    trade_kind: str | None = Query(None, description="Trade leg CE or PE for decision explain"),
    strike_source: str | None = Query(None, description="Strike offset label e.g. ATM"),
):
    """5m OHLC rows for a selected rolling contract, optional trade overlay."""
    trade = None
    if entry is not None:
        trade = {
            "entry": entry,
            "exit": exit or 0,
            "sl": sl or 9,
            "target": target or 0,
            "entry_datetime_ist": entry_datetime_ist or "",
            "exit_datetime_ist": exit_datetime_ist or "",
            "exit_reason": exit_reason or "",
            "pnl_inr": pnl_inr,
            "kind": trade_kind or kind,
            "strike_source": strike_source or "",
        }
    try:
        data = await asyncio.to_thread(
            get_dhan_ohlc,
            segment,
            session_date,
            kind=kind,
            offset=offset,
            refresh=refresh,
            trade=trade,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SuccessResponse(data=data, message=f"{data['bars']} bars loaded")
