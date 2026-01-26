"""
Nifty50 Options Backtest Strategy
"""
from fastapi import APIRouter, Request
from datetime import datetime, timedelta
from kiteconnect.exceptions import KiteException
from utils.kite_utils import get_kite_instance
from core.user_context import get_user_id_from_request
from core.exceptions import ValidationError, NotFoundError, ExternalAPIError, AlgoFeastException
from schemas.strategies import Nifty50OptionsBacktestRequest
from utils.logger import log_error, log_info

router = APIRouter()


@router.post("/backtest-nifty50-options")
async def backtest_nifty50_options(request: Request, backtest_request: Nifty50OptionsBacktestRequest):
    """
    Backtest Nifty50 options strategy for given date range with multiple strategy options
    """
    try:
        start_date = backtest_request.start_date
        end_date = backtest_request.end_date
        strategy_type = backtest_request.strategy_type
        fund = backtest_request.fund
        risk_pct = backtest_request.risk / 100
        reward_pct = backtest_request.reward / 100
        
        log_info(f"Starting Nifty50 options backtest: {start_date} to {end_date}, strategy: {strategy_type}")
        
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
            raise NotFoundError(
                resource="Nifty50 options",
                identifier="NFO exchange"
            )
        
        # Generate list of trading dates (excluding weekends)
        trading_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday to Friday
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        if not trading_dates:
            raise ValidationError(
                message="No trading days found in date range",
                field="date_range",
                details={"start_date": str(start_date), "end_date": str(end_date)}
            )
        
        # Backtest results structure
        backtest_results = {
            "start_date": str(start_date),
            "end_date": str(end_date),
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
        # The actual backtest logic should be implemented using the strategies/runner module
        # or by importing from a dedicated backtest module
        
        return {"data": backtest_results}
        
    except AlgoFeastException:
        raise
    except KiteException as e:
        log_error(f"Kite API error in Nifty50 options backtest: {str(e)}")
        raise ExternalAPIError(message=f"Kite API error: {str(e)}", service="Kite Connect")
    except Exception as e:
        log_error(f"Error backtesting Nifty50 options strategy: {str(e)}")
        raise AlgoFeastException(
            message=f"Error backtesting strategy: {str(e)}",
            status_code=500,
            error_code="INTERNAL_ERROR"
        )

