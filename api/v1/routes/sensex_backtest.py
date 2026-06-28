"""20rupees-strategy backtest via Dhan Data API (Sensex + Nifty50)."""
import asyncio

from fastapi import APIRouter, HTTPException, Query, Request

from core.responses import SuccessResponse
from schemas.sensex_backtest import SensexBacktestRunRequest
from services.backtest_job_runner import get_backtest_job, prune_old_jobs, start_backtest_job
from services.dhan_viewer import list_dhan_sessions, load_cached_backtest_bundle
from services.nifty_dhan_backtest import run_nifty_dhan_backtest
from services.sensex_dhan_backtest import (
    BacktestParams,
    backtest_session_calendar,
    check_dhan_status,
    list_available_sessions,
    run_sensex_dhan_backtest,
    sensex_entry_cutoff_minutes,
    sensex_entry_scan_start_minutes,
)

router = APIRouter(prefix="/sensex/backtest", tags=["Sensex Backtest"])


def _backtest_calendar_for_segment(segment: str) -> dict:
    seg = (segment or "sensex").lower()
    if seg in ("nifty50", "nifty"):
        sessions = list_dhan_sessions("nifty50")
        dates = [s["session_date"] for s in sessions if s.get("session_date")]
        scan_start = sensex_entry_scan_start_minutes()
        scan_end = sensex_entry_cutoff_minutes()
        return {
            "segment": "nifty50",
            "start_date": min(dates) if dates else None,
            "end_date": max(dates) if dates else None,
            "trading_days": len(dates),
            "cached_count": len(dates),
            "default_entry_scan_start_ist": f"{scan_start // 60:02d}:{scan_start % 60:02d}",
            "default_entry_scan_end_ist": f"{scan_end // 60:02d}:{scan_end % 60:02d}",
        }
    cal = backtest_session_calendar()
    cal["segment"] = "sensex"
    return cal


@router.get("/sessions")
async def backtest_sessions(
    _: Request,
    segment: str = Query("sensex", description="sensex or nifty50"),
):
    """Trading-day calendar with Dhan cache status (for date-range picker)."""
    seg = (segment or "sensex").lower()
    if seg in ("nifty50", "nifty"):
        sessions = list_dhan_sessions("nifty50")
        calendar = _backtest_calendar_for_segment("nifty50")
    else:
        sessions = list_available_sessions()
        calendar = _backtest_calendar_for_segment("sensex")
    return SuccessResponse(
        data={
            "sessions": sessions,
            "count": len(sessions),
            "calendar": calendar,
            "segment": calendar.get("segment", seg),
        },
        message=f"{len(sessions)} trading days · {calendar.get('cached_count', 0)} cached",
    )


@router.get("/cached")
async def backtest_cached_results(_: Request):
    """Load latest on-disk backtest JSON (Nifty + Sensex) and combined trade CSV."""
    try:
        bundle = load_cached_backtest_bundle()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SuccessResponse(data=bundle, message="Cached backtest results loaded")


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
    """Run 20rupees-strategy backtest on Dhan rolling options (configurable bar intervals)."""
    params = BacktestParams(
        capital=body.capital,
        risk_pct=body.risk_pct,
        sl_inr=body.sl_inr,
        entry_band_low=body.entry_band_low,
        entry_band_high=body.entry_band_high,
        min_target_low=body.min_target_low,
        min_target_high=body.min_target_high,
        direction=body.direction,
        start_date=body.start_date,
        end_date=body.end_date,
        expiry_dates=body.expiry_dates,
        refresh_dhan=body.refresh_dhan,
        timeframes_min=body.timeframes_min,
        entry_scan_start_ist=body.entry_scan_start_ist,
        entry_scan_end_ist=body.entry_scan_end_ist,
    )
    runner = run_nifty_dhan_backtest if body.segment == "nifty50" else run_sensex_dhan_backtest
    try:
        result = await asyncio.to_thread(runner, params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc
    return SuccessResponse(data=result, message="Backtest complete")


@router.post("/run/async")
async def run_backtest_async(_: Request, body: SensexBacktestRunRequest):
    """Start backtest in background; poll GET /run/jobs/{job_id} for live progress."""
    prune_old_jobs()
    params = BacktestParams(
        capital=body.capital,
        risk_pct=body.risk_pct,
        sl_inr=body.sl_inr,
        entry_band_low=body.entry_band_low,
        entry_band_high=body.entry_band_high,
        min_target_low=body.min_target_low,
        min_target_high=body.min_target_high,
        direction=body.direction,
        start_date=body.start_date,
        end_date=body.end_date,
        expiry_dates=body.expiry_dates,
        refresh_dhan=body.refresh_dhan,
        timeframes_min=body.timeframes_min,
        entry_scan_start_ist=body.entry_scan_start_ist,
        entry_scan_end_ist=body.entry_scan_end_ist,
    )
    segment = body.segment or "sensex"
    runner = run_nifty_dhan_backtest if segment == "nifty50" else run_sensex_dhan_backtest
    try:
        job_id = start_backtest_job(params, segment=segment, runner=runner)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SuccessResponse(
        data={"job_id": job_id, "status": "running"},
        message="Backtest started — poll /run/jobs/{job_id}",
    )


@router.get("/run/jobs/{job_id}")
async def backtest_job_status(_: Request, job_id: str):
    """Live backtest progress and result when complete."""
    job = get_backtest_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Backtest job not found or expired")
    return SuccessResponse(data=job, message=job.get("status") or "unknown")
