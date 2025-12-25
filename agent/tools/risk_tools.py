"""
Risk management tools for the agent
"""
from typing import Optional
from langchain_core.tools import tool
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.kite_utils import get_kite_instance
from agent.config import get_agent_config
from kiteconnect.exceptions import KiteException


@tool
def calculate_risk_tool(
    entry_price: float,
    stop_loss_price: float,
    quantity: int,
    current_price: Optional[float] = None
) -> dict:
    """
    Calculate risk metrics for a trade.
    
    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        quantity: Quantity
        current_price: Current price (for existing positions)
        
    Returns:
        dict with risk metrics (risk_amount, risk_percentage, reward_potential, etc.)
    """
    try:
        if current_price is None:
            current_price = entry_price
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        total_risk = risk_per_share * quantity
        
        # Calculate risk percentage
        position_value = entry_price * quantity
        risk_percentage = (total_risk / position_value) * 100 if position_value > 0 else 0
        
        # Calculate reward potential using config
        config = get_agent_config()
        rr_ratio = config.reward_per_trade_pct / config.risk_per_trade_pct
        reward_per_share = risk_per_share * rr_ratio
        total_reward = reward_per_share * quantity
        reward_percentage = (total_reward / position_value) * 100 if position_value > 0 else 0
        
        return {
            "status": "success",
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "quantity": quantity,
            "position_value": position_value,
            "risk_per_share": round(risk_per_share, 2),
            "total_risk": round(total_risk, 2),
            "risk_percentage": round(risk_percentage, 2),
            "reward_per_share": round(reward_per_share, 2),
            "total_reward": round(total_reward, 2),
            "reward_percentage": round(reward_percentage, 2),
            "risk_reward_ratio": 1.0 / rr_ratio
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error calculating risk: {str(e)}"
        }


@tool
def check_risk_limits_tool(
    trade_value: float,
    risk_amount: float
) -> dict:
    """
    Check if a trade violates risk limits.
    
    Args:
        trade_value: Total value of the trade
        risk_amount: Risk amount for this trade
        
    Returns:
        dict with validation result and warnings
    """
    try:
        config = get_agent_config()
        kite = get_kite_instance()
        
        # Get current balance
        margins = kite.margins()
        equity_data = margins.get('equity', {})
        available_value = equity_data.get('available', 0)
        
        if isinstance(available_value, dict):
            available_margin = available_value.get('cash', 0)
        else:
            available_margin = available_value if available_value else 0
        
        # Check limits
        violations = []
        warnings = []
        
        # Check max position size
        if trade_value > config.max_position_size:
            violations.append(f"Trade value {trade_value} exceeds max position size {config.max_position_size}")
        
        # Check available margin
        if trade_value > available_margin:
            violations.append(f"Trade value {trade_value} exceeds available margin {available_margin}")
        
        # Check risk percentage
        risk_pct = (risk_amount / trade_value) * 100 if trade_value > 0 else 0
        if risk_pct > config.risk_per_trade_pct:
            warnings.append(f"Risk percentage {risk_pct:.2f}% exceeds configured {config.risk_per_trade_pct}%")
        
        # Check daily loss limit (would need to track daily PnL)
        # This is a placeholder - would need actual daily PnL tracking
        
        return {
            "status": "success",
            "trade_value": trade_value,
            "risk_amount": risk_amount,
            "available_margin": available_margin,
            "max_position_size": config.max_position_size,
            "risk_percentage": round(risk_pct, 2),
            "violations": violations,
            "warnings": warnings,
            "is_valid": len(violations) == 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error checking risk limits: {str(e)}"
        }


@tool
def get_portfolio_risk_tool() -> dict:
    """
    Get overall portfolio risk assessment.
    
    Returns:
        dict with portfolio risk metrics
    """
    try:
        kite = get_kite_instance()
        positions = kite.positions().get("net", [])
        
        total_exposure = 0
        total_pnl = 0
        open_positions = 0
        
        for pos in positions:
            qty = pos.get("quantity", 0)
            if qty != 0:
                open_positions += 1
                avg_price = pos.get("average_price", 0)
                total_exposure += abs(avg_price * qty)
                total_pnl += pos.get("pnl", 0)
        
        # Get margins
        margins = kite.margins()
        equity_data = margins.get('equity', {})
        available_value = equity_data.get('available', 0)
        
        if isinstance(available_value, dict):
            available_margin = available_value.get('cash', 0)
        else:
            available_margin = available_value if available_value else 0
        
        utilised_margin = equity_data.get('utilised', 0)
        if isinstance(utilised_margin, dict):
            utilised_margin = utilised_margin.get('debits', 0)
        
        margin_utilization = (utilised_margin / (available_margin + utilised_margin)) * 100 if (available_margin + utilised_margin) > 0 else 0
        
        return {
            "status": "success",
            "total_exposure": round(total_exposure, 2),
            "total_pnl": round(total_pnl, 2),
            "open_positions": open_positions,
            "available_margin": round(available_margin, 2),
            "utilised_margin": round(utilised_margin, 2),
            "margin_utilization": round(margin_utilization, 2),
            "risk_level": "LOW" if margin_utilization < 50 else "MEDIUM" if margin_utilization < 80 else "HIGH"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error getting portfolio risk: {str(e)}"
        }


@tool
def suggest_position_size_tool(
    entry_price: float,
    stop_loss_price: float,
    available_capital: Optional[float] = None,
    risk_percentage: float = 1.0,
    max_position_size: Optional[float] = None
) -> dict:
    """
    Suggest optimal position size based on risk management rules.
    
    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        available_capital: Available capital (if None, uses trading_capital from config)
        risk_percentage: Risk percentage per trade (default: 1%)
        max_position_size: Maximum position size (overrides config if provided)
        
    Returns:
        dict with suggested quantity and position details
    """
    try:
        config = get_agent_config()
        
        # Use provided capital or config capital
        capital = available_capital if available_capital is not None else config.trading_capital
        
        # Determine max position size
        max_size = max_position_size if max_position_size is not None else config.max_position_size
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return {
                "status": "error",
                "error": "Entry price and stop loss price cannot be the same"
            }
        
        # Calculate risk amount (e.g. 1% of capital)
        risk_amount = (capital * risk_percentage) / 100
        
        # Calculate quantity based on risk
        quantity = int(risk_amount / risk_per_share)
        
        # Cap quantity by max position size
        if entry_price * quantity > max_size:
            quantity = int(max_size / entry_price)
            
        # Calculate final position value
        position_value = entry_price * quantity
        
        return {
            "status": "success",
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "available_capital": capital,
            "max_position_size": max_size,
            "risk_percentage": risk_percentage,
            "risk_per_share": round(risk_per_share, 2),
            "suggested_quantity": quantity,
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "actual_risk_percentage": round((risk_amount / position_value) * 100, 2) if position_value > 0 else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error suggesting position size: {str(e)}"
        }

