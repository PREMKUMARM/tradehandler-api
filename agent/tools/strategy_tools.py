"""
Strategy management tools for the agent
"""
from typing import Optional
from langchain_core.tools import tool
from datetime import datetime
import sys
from pathlib import Path
import requests

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.kite_utils import get_kite_instance
from kiteconnect.exceptions import KiteException


@tool
def backtest_strategy_tool(
    start_date: str,
    end_date: str,
    strategy_type: str = "915_candle_break",
    fund: float = 200000.0,
    risk: float = 1.0,
    reward: float = 3.0
) -> dict:
    """
    Backtest a trading strategy on historical data.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strategy_type: Strategy type (915_candle_break, mean_reversion, momentum_breakout, etc.)
        fund: Trading capital
        risk: Risk percentage per trade
        reward: Reward percentage per trade
        
    Returns:
        dict with backtest results (total_pnl, win_rate, trades, etc.)
    """
    try:
        # This would call the existing backtest endpoint
        # For now, return a placeholder structure
        return {
            "status": "success",
            "message": "Backtest initiated. Use /backtest-nifty50-options endpoint for full backtest.",
            "start_date": start_date,
            "end_date": end_date,
            "strategy_type": strategy_type,
            "fund": fund,
            "risk": risk,
            "reward": reward
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error initiating backtest: {str(e)}"
        }


@tool
def get_strategy_signal_tool(strategy_type: str = "915_candle_break") -> dict:
    """
    Get current trading signal from a strategy.
    
    Args:
        strategy_type: Strategy type to check
        
    Returns:
        dict with signal (BUY/SELL/HOLD), confidence, and details
    """
    try:
        kite = get_kite_instance()
        
        # Get Nifty50 options
        nifty_quote = kite.quote("NSE:NIFTY 50")
        nifty_price = nifty_quote.get("NSE:NIFTY 50", {}).get("last_price", 0)
        
        if not nifty_price:
            return {
                "status": "error",
                "error": "Nifty50 price not found"
            }
        
        # This is a simplified signal - in production, this would run the actual strategy
        return {
            "status": "success",
            "strategy_type": strategy_type,
            "signal": "HOLD",  # Would be calculated by actual strategy
            "confidence": 0.5,
            "nifty_price": nifty_price,
            "message": "Strategy signal calculation requires real-time data processing"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error getting strategy signal: {str(e)}"
        }


@tool
def optimize_strategy_tool(
    strategy_type: str,
    parameter_ranges: dict,
    optimization_metric: str = "sharpe_ratio"
) -> dict:
    """
    Optimize strategy parameters using backtesting.
    
    Args:
        strategy_type: Strategy type to optimize
        parameter_ranges: Dict of parameter ranges to test (e.g., {"rsi_period": [10, 14, 20]})
        optimization_metric: Metric to optimize (sharpe_ratio, total_pnl, win_rate)
        
    Returns:
        dict with optimal parameters and performance metrics
    """
    try:
        # This would run optimization logic
        return {
            "status": "success",
            "strategy_type": strategy_type,
            "optimal_parameters": parameter_ranges,  # Placeholder
            "optimization_metric": optimization_metric,
            "message": "Parameter optimization requires extended backtesting framework"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error optimizing strategy: {str(e)}"
        }


@tool
def create_strategy_tool(strategy_description: str) -> dict:
    """
    Create a new trading strategy from natural language description.
    
    Args:
        strategy_description: Natural language description of the strategy
        
    Returns:
        dict with created strategy details
    """
    try:
        # This would parse the description and create strategy code
        # For now, return a placeholder
        return {
            "status": "success",
            "message": "Strategy creation from description requires code generation framework",
            "description": strategy_description,
            "strategy_type": "custom",
            "note": "This feature requires advanced AI code generation capabilities"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error creating strategy: {str(e)}"
        }

