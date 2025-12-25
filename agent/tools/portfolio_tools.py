"""
Portfolio management tools for the agent
"""
from langchain_core.tools import tool
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.kite_utils import get_kite_instance
from kiteconnect.exceptions import KiteException


@tool
def get_portfolio_summary_tool() -> dict:
    """
    Get portfolio overview with positions, PnL, and margin details.
    
    Returns:
        dict with portfolio summary
    """
    try:
        kite = get_kite_instance()
        
        # Get positions
        positions = kite.positions().get("net", [])
        
        # Get margins
        margins = kite.margins()
        equity_data = margins.get('equity', {})
        available_value = equity_data.get('available', 0)
        utilised_value = equity_data.get('utilised', 0)
        
        if isinstance(available_value, dict):
            available_margin = available_value.get('cash', 0)
        else:
            available_margin = available_value if available_value else 0
        
        if isinstance(utilised_value, dict):
            utilised_margin = utilised_value.get('debits', 0)
        else:
            utilised_margin = utilised_value
        
        total_margin = equity_data.get('net', 0) or equity_data.get('live_balance', 0)
        
        # Calculate portfolio metrics
        total_pnl = 0
        open_positions = []
        total_exposure = 0
        
        for pos in positions:
            qty = pos.get("quantity", 0)
            if qty != 0:
                pnl = pos.get("pnl", 0)
                total_pnl += pnl
                avg_price = pos.get("average_price", 0)
                exposure = abs(avg_price * qty)
                total_exposure += exposure
                
                open_positions.append({
                    "tradingsymbol": pos.get("tradingsymbol"),
                    "quantity": qty,
                    "average_price": avg_price,
                    "last_price": pos.get("last_price", 0),
                    "pnl": pnl,
                    "exposure": exposure
                })
        
        return {
            "status": "success",
            "total_pnl": round(total_pnl, 2),
            "open_positions_count": len(open_positions),
            "open_positions": open_positions,
            "total_exposure": round(total_exposure, 2),
            "available_margin": round(available_margin, 2),
            "utilised_margin": round(utilised_margin, 2),
            "total_margin": round(total_margin, 2),
            "margin_utilization": round((utilised_margin / total_margin) * 100, 2) if total_margin > 0 else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error getting portfolio summary: {str(e)}"
        }


@tool
def analyze_performance_tool(days: int = 30) -> dict:
    """
    Analyze portfolio performance over a period.
    
    Args:
        days: Number of days to analyze (default: 30)
        
    Returns:
        dict with performance metrics
    """
    try:
        kite = get_kite_instance()
        
        # Get orders for the period
        orders = kite.orders()
        
        # Filter orders by date
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_orders = []
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        
        for order in orders:
            order_time = order.get("order_timestamp")
            if order_time:
                # Parse order timestamp (format may vary)
                try:
                    if isinstance(order_time, str):
                        order_dt = datetime.fromisoformat(order_time.replace('Z', '+00:00'))
                    else:
                        order_dt = order_time
                    
                    if order_dt >= cutoff_date:
                        recent_orders.append(order)
                        status = order.get("status", "").upper()
                        if status == "COMPLETE":
                            total_trades += 1
                            # Calculate PnL if available
                            avg_price = order.get("average_price", 0)
                            price = order.get("price", 0)
                            if avg_price > 0 and price > 0:
                                pnl = (price - avg_price) * order.get("filled_quantity", 0)
                                if pnl > 0:
                                    winning_trades += 1
                                    total_profit += pnl
                                else:
                                    losing_trades += 1
                                    total_loss += abs(pnl)
                except Exception:
                    continue
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        net_pnl = total_profit - total_loss
        
        return {
            "status": "success",
            "period_days": days,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "total_profit": round(total_profit, 2),
            "total_loss": round(total_loss, 2),
            "net_pnl": round(net_pnl, 2),
            "average_profit_per_trade": round(total_profit / winning_trades, 2) if winning_trades > 0 else 0,
            "average_loss_per_trade": round(total_loss / losing_trades, 2) if losing_trades > 0 else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error analyzing performance: {str(e)}"
        }


@tool
def rebalance_portfolio_tool() -> dict:
    """
    Analyze portfolio and suggest rebalancing actions.
    
    Returns:
        dict with rebalancing suggestions
    """
    try:
        kite = get_kite_instance()
        positions = kite.positions().get("net", [])
        
        suggestions = []
        
        # Analyze each position
        for pos in positions:
            qty = pos.get("quantity", 0)
            if qty != 0:
                pnl = pos.get("pnl", 0)
                pnl_percentage = (pnl / (pos.get("average_price", 1) * abs(qty))) * 100 if qty != 0 else 0
                
                # Suggest exit if significant profit or loss
                if pnl_percentage > 10:
                    suggestions.append({
                        "action": "PARTIAL_EXIT",
                        "tradingsymbol": pos.get("tradingsymbol"),
                        "reason": f"Significant profit: {pnl_percentage:.2f}%",
                        "suggested_quantity": abs(qty) // 2
                    })
                elif pnl_percentage < -5:
                    suggestions.append({
                        "action": "EXIT",
                        "tradingsymbol": pos.get("tradingsymbol"),
                        "reason": f"Significant loss: {pnl_percentage:.2f}%",
                        "suggested_quantity": abs(qty)
                    })
        
        return {
            "status": "success",
            "suggestions": suggestions,
            "count": len(suggestions)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error analyzing rebalancing: {str(e)}"
        }

