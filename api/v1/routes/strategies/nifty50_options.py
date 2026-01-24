"""
Nifty50 Options Backtest Strategy
"""
from fastapi import APIRouter, Request, HTTPException
from datetime import datetime, timedelta
from utils.kite_utils import get_kite_instance
from core.user_context import get_user_id_from_request
from kiteconnect.exceptions import KiteException

router = APIRouter()


@router.post("/backtest-nifty50-options")
async def backtest_nifty50_options(request: Request):
    """
    Backtest Nifty50 options strategy for given date range with multiple strategy options
    
    Request body:
    {
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "strategy_type": "915_candle_break",
        "fund": 200000,
        "risk": 1,
        "reward": 3
    }
    """
    try:
        payload = await request.json()
        start_date_str = payload.get("start_date")
        end_date_str = payload.get("end_date")
        strategy_type = payload.get("strategy_type", "915_candle_break")
        fund = payload.get("fund", 200000)
        risk_pct = payload.get("risk", 1) / 100
        reward_pct = payload.get("reward", 3) / 100
        
        if not start_date_str or not end_date_str:
            raise HTTPException(status_code=400, detail="start_date and end_date are required (format: YYYY-MM-DD)")
        
        # Parse dates
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
        user_id = get_user_id_from_request(request)
        kite = get_kite_instance(user_id)
        
        # Get all NFO instruments
        all_instruments = kite.instruments("NFO")
        
        # Filter for Nifty50 options
        nifty_options = [
            inst for inst in all_instruments 
            if inst.get("name") == "NIFTY" and inst.get("instrument_type") in ["CE", "PE"]
        ]
        
        if not nifty_options:
            raise HTTPException(status_code=404, detail="Nifty50 options not found")
        
        # Generate list of trading dates (excluding weekends)
        trading_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday to Friday
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        if not trading_dates:
            raise HTTPException(status_code=400, detail="No trading days found in date range")
        
        # Backtest results structure
        backtest_results = {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "strategy_type": strategy_type,
            "total_trading_days": len(trading_dates),
            "trades": [],
            "statistics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "average_profit": 0.0,
                "average_loss": 0.0,
                "max_profit": 0.0,
                "max_loss": 0.0
            }
        }
        
        # NOTE: Full backtest implementation was removed from main.py (~700 lines)
        # This is a placeholder that returns the structure expected by the frontend
        # TODO: Implement full backtest logic here or import from a separate module
        
        return {"data": backtest_results}
        
    except HTTPException:
        raise
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error backtesting strategy: {str(e)}")

