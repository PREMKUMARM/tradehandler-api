"""
Kite Connect trading tools for the agent
"""
from typing import Optional
from langchain_core.tools import tool
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException
import sys
from pathlib import Path

# Add parent directory to path to import get_kite_instance
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.kite_utils import get_kite_instance


@tool
def place_order_tool(
    tradingsymbol: str,
    exchange: str = "NFO",
    transaction_type: str = "BUY",
    quantity: int = 75,
    order_type: str = "MARKET",
    product: str = "MIS",
    price: Optional[float] = None,
    trigger_price: Optional[float] = None,
    validity: str = "DAY",
) -> dict:
    """
    Place a buy or sell order on Zerodha Kite.
    
    Args:
        tradingsymbol: Trading symbol (e.g., "NIFTY24JAN24500CE")
        exchange: Exchange (NSE, NFO, BSE, etc.) - default: NFO
        transaction_type: BUY or SELL
        quantity: Number of lots/shares
        order_type: MARKET, LIMIT, SL, SL-M
        product: MIS (intraday), CNC (delivery), NRML (carry forward)
        price: Price for LIMIT orders
        trigger_price: Trigger price for SL orders
        validity: DAY, IOC, TTL
        
    Returns:
        dict with order_id and status
    """
    try:
        kite = get_kite_instance()
        
        order_params = {
            "variety": kite.VARIETY_REGULAR,
            "exchange": exchange,
            "tradingsymbol": tradingsymbol,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "product": product,
            "order_type": order_type,
            "validity": validity,
        }
        
        if price is not None:
            order_params["price"] = price
        if trigger_price is not None:
            order_params["trigger_price"] = trigger_price
            
        order_id = kite.place_order(**order_params)
        
        return {
            "status": "success",
            "order_id": order_id,
            "message": f"Order placed successfully: {transaction_type} {quantity} {tradingsymbol}"
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error placing order: {str(e)}"
        }


@tool
def modify_order_tool(
    order_id: str,
    quantity: Optional[int] = None,
    price: Optional[float] = None,
    order_type: Optional[str] = None,
    trigger_price: Optional[float] = None,
    validity: Optional[str] = None,
) -> dict:
    """
    Modify an existing order.
    
    Args:
        order_id: Order ID to modify
        quantity: New quantity (optional)
        price: New price for LIMIT orders (optional)
        order_type: New order type (optional)
        trigger_price: New trigger price (optional)
        validity: New validity (optional)
        
    Returns:
        dict with modified order_id and status
    """
    try:
        kite = get_kite_instance()
        
        modify_params = {
            "variety": kite.VARIETY_REGULAR,
            "order_id": order_id,
        }
        
        if quantity is not None:
            modify_params["quantity"] = quantity
        if price is not None:
            modify_params["price"] = price
        if order_type is not None:
            modify_params["order_type"] = order_type
        if trigger_price is not None:
            modify_params["trigger_price"] = trigger_price
        if validity is not None:
            modify_params["validity"] = validity
            
        modified_order_id = kite.modify_order(**modify_params)
        
        return {
            "status": "success",
            "order_id": modified_order_id,
            "message": f"Order {order_id} modified successfully"
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error modifying order: {str(e)}"
        }


@tool
def cancel_order_tool(order_id: str, variety: str = "regular") -> dict:
    """
    Cancel an existing order.
    
    Args:
        order_id: Order ID to cancel
        variety: Order variety (regular, amo, etc.)
        
    Returns:
        dict with cancellation status
    """
    try:
        kite = get_kite_instance()
        
        kite_variety = kite.VARIETY_REGULAR if variety == "regular" else variety
        cancelled_order_id = kite.cancel_order(
            variety=kite_variety,
            order_id=order_id
        )
        
        return {
            "status": "success",
            "order_id": cancelled_order_id,
            "message": f"Order {order_id} cancelled successfully"
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error cancelling order: {str(e)}"
        }


@tool
def get_positions_tool() -> dict:
    """
    Get current open positions from Zerodha Kite.
    
    Returns:
        dict with positions list, total_pnl, and active_count
    """
    try:
        kite = get_kite_instance()
        positions = kite.positions()
        
        net_positions = positions.get("net", [])
        total_pnl = 0
        active_count = 0
        
        for pos in net_positions:
            total_pnl += pos.get("pnl", 0)
            if pos.get("quantity", 0) != 0:
                active_count += 1
                
        return {
            "status": "success",
            "positions": net_positions,
            "total_pnl": round(total_pnl, 2),
            "active_count": active_count
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error fetching positions: {str(e)}"
        }


@tool
def get_orders_tool() -> dict:
    """
    Get order history from Zerodha Kite.
    
    Returns:
        dict with orders list
    """
    try:
        kite = get_kite_instance()
        orders = kite.orders()
        
        return {
            "status": "success",
            "orders": orders,
            "count": len(orders)
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error fetching orders: {str(e)}"
        }


@tool
def get_balance_tool() -> dict:
    """
    Get account balance and margin information.
    
    Returns:
        dict with equity margin details (available, utilised, total)
    """
    try:
        kite = get_kite_instance()
        margins = kite.margins()
        
        equity_data = margins.get('equity', {})
        available_value = equity_data.get('available', 0)
        utilised_value = equity_data.get('utilised', 0)
        total_margin = equity_data.get('net', 0) or equity_data.get('live_balance', 0)
        
        if isinstance(available_value, dict):
            available_margin = available_value.get('cash', 0)
            if available_margin == 0:
                available_margin = equity_data.get('opening_balance', 0) or equity_data.get('live_balance', 0)
        else:
            available_margin = available_value if available_value else equity_data.get('cash', 0) or equity_data.get('opening_balance', 0)
        
        if isinstance(utilised_value, dict):
            utilised_margin = utilised_value.get('debits', 0)
        else:
            utilised_margin = utilised_value
        
        return {
            "status": "success",
            "available_margin": float(available_margin) if available_margin else 0.0,
            "utilised_margin": float(utilised_margin) if utilised_margin else 0.0,
            "total_margin": float(total_margin) if total_margin else 0.0,
            "equity": equity_data
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error getting balance: {str(e)}"
        }


@tool
def exit_position_tool(tradingsymbol: str, exchange: str = "NFO") -> dict:
    """
    Exit a specific position by placing opposite order.
    
    Args:
        tradingsymbol: Trading symbol to exit
        exchange: Exchange (default: NFO)
        
    Returns:
        dict with exit order status
    """
    try:
        kite = get_kite_instance()
        positions = kite.positions().get("net", [])
        
        # Find the position
        position = None
        for pos in positions:
            if pos.get("tradingsymbol") == tradingsymbol and pos.get("exchange") == exchange:
                position = pos
                break
        
        if not position:
            return {
                "status": "error",
                "error": f"Position not found for {tradingsymbol}"
            }
        
        qty = position.get("quantity", 0)
        if qty == 0:
            return {
                "status": "error",
                "error": f"No open position for {tradingsymbol}"
            }
        
        # Place opposite order
        trans_type = kite.TRANSACTION_TYPE_SELL if qty > 0 else kite.TRANSACTION_TYPE_BUY
        abs_qty = abs(qty)
        
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            transaction_type=trans_type,
            quantity=abs_qty,
            product=position.get("product", "MIS"),
            order_type=kite.ORDER_TYPE_MARKET
        )
        
        return {
            "status": "success",
            "order_id": order_id,
            "message": f"Exit order placed for {tradingsymbol}: {trans_type} {abs_qty}"
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error exiting position: {str(e)}"
        }


@tool
def place_gtt_tool(
    tradingsymbol: str,
    exchange: str = "NSE",
    trigger_type: str = "single",  # "single" or "two-leg" (OCO)
    trigger_price: float = None,  # For single trigger
    trigger_prices: list = None,  # For OCO: [stop_loss_trigger, target_trigger]
    last_price: float = None,  # Current market price
    stop_loss_price: float = None,  # Stop-loss limit price
    target_price: float = None,  # Target limit price
    quantity: int = 1,
    transaction_type: str = "SELL",  # BUY or SELL
    product: str = "MIS"
) -> dict:
    """
    Place a Good Till Triggered (GTT) order.
    
    GTT orders remain active until trigger price is reached (valid up to 1 year).
    OCO (One Cancels Other) type allows setting both stop-loss and target.
    
    Args:
        tradingsymbol: Trading symbol (e.g., "RELIANCE")
        exchange: Exchange (NSE, NFO, BSE)
        trigger_type: "single" for single trigger, "two-leg" for OCO (stop-loss + target)
        trigger_price: Trigger price for single trigger GTT
        trigger_prices: [stop_loss_trigger, target_trigger] for OCO GTT
        last_price: Current market price (required)
        stop_loss_price: Stop-loss limit price (for OCO or single stop-loss)
        target_price: Target limit price (for OCO)
        quantity: Number of shares/lots
        transaction_type: BUY or SELL
        product: MIS (intraday), CNC (delivery), NRML (carry forward)
        
    Returns:
        dict with GTT trigger_id and status
    """
    try:
        kite = get_kite_instance()
        
        if not last_price:
            # Try to get current price
            quote = kite.quote(f"{exchange}:{tradingsymbol}")
            instrument_key = f"{exchange}:{tradingsymbol}"
            if instrument_key in quote:
                last_price = quote[instrument_key].get("last_price", 0)
            else:
                return {
                    "status": "error",
                    "error": "last_price is required. Could not fetch from market."
                }
        
        # Build trigger_values based on trigger_type
        if trigger_type == "single" or trigger_type == kite.GTT_TYPE_SINGLE:
            if not trigger_price:
                return {
                    "status": "error",
                    "error": "trigger_price is required for single trigger GTT"
                }
            trigger_values = {"ltp": trigger_price}
            orders = [{
                "transaction_type": transaction_type,
                "quantity": quantity,
                "price": stop_loss_price or trigger_price * 0.99,  # Default to 1% below trigger
                "product": product,
                "order_type": "LIMIT"
            }]
        elif trigger_type == "two-leg" or trigger_type == kite.GTT_TYPE_OCO:
            if not trigger_prices or len(trigger_prices) != 2:
                return {
                    "status": "error",
                    "error": "trigger_prices list with 2 values [stop_loss_trigger, target_trigger] required for OCO GTT"
                }
            if not stop_loss_price or not target_price:
                return {
                    "status": "error",
                    "error": "Both stop_loss_price and target_price are required for OCO GTT"
                }
            trigger_values = {
                "ltp": [
                    {"ltp": trigger_prices[0]},  # Stop-loss trigger
                    {"ltp": trigger_prices[1]}  # Target trigger
                ]
            }
            orders = [
                {
                    "transaction_type": transaction_type,
                    "quantity": quantity,
                    "price": stop_loss_price,
                    "product": product,
                    "order_type": "LIMIT"
                },
                {
                    "transaction_type": transaction_type,
                    "quantity": quantity,
                    "price": target_price,
                    "product": product,
                    "order_type": "LIMIT"
                }
            ]
        else:
            return {
                "status": "error",
                "error": f"Invalid trigger_type: {trigger_type}. Use 'single' or 'two-leg'"
            }
        
        # Place GTT order
        trigger_id = kite.place_gtt(
            trigger_type=kite.GTT_TYPE_SINGLE if trigger_type == "single" else kite.GTT_TYPE_OCO,
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            trigger_values=trigger_values,
            last_price=last_price,
            orders=orders
        )
        
        return {
            "status": "success",
            "trigger_id": trigger_id,
            "message": f"GTT order placed successfully. Trigger ID: {trigger_id}",
            "trigger_type": trigger_type
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error placing GTT order: {str(e)}"
        }


@tool
def modify_gtt_tool(
    trigger_id: int,
    tradingsymbol: str,
    exchange: str = "NSE",
    trigger_type: str = "single",
    trigger_price: float = None,
    trigger_prices: list = None,
    last_price: float = None,
    stop_loss_price: float = None,
    target_price: float = None,
    quantity: int = 1,
    transaction_type: str = "SELL",
    product: str = "MIS"
) -> dict:
    """
    Modify an existing GTT order.
    
    Args:
        trigger_id: GTT trigger ID to modify
        (other parameters same as place_gtt_tool)
        
    Returns:
        dict with modification status
    """
    try:
        kite = get_kite_instance()
        
        if not last_price:
            quote = kite.quote(f"{exchange}:{tradingsymbol}")
            instrument_key = f"{exchange}:{tradingsymbol}"
            if instrument_key in quote:
                last_price = quote[instrument_key].get("last_price", 0)
            else:
                return {
                    "status": "error",
                    "error": "last_price is required. Could not fetch from market."
                }
        
        # Build trigger_values and orders (same logic as place_gtt_tool)
        if trigger_type == "single" or trigger_type == kite.GTT_TYPE_SINGLE:
            if not trigger_price:
                return {
                    "status": "error",
                    "error": "trigger_price is required for single trigger GTT"
                }
            trigger_values = {"ltp": trigger_price}
            orders = [{
                "transaction_type": transaction_type,
                "quantity": quantity,
                "price": stop_loss_price or trigger_price * 0.99,
                "product": product,
                "order_type": "LIMIT"
            }]
        elif trigger_type == "two-leg" or trigger_type == kite.GTT_TYPE_OCO:
            if not trigger_prices or len(trigger_prices) != 2:
                return {
                    "status": "error",
                    "error": "trigger_prices list with 2 values required for OCO GTT"
                }
            if not stop_loss_price or not target_price:
                return {
                    "status": "error",
                    "error": "Both stop_loss_price and target_price are required for OCO GTT"
                }
            trigger_values = {
                "ltp": [
                    {"ltp": trigger_prices[0]},
                    {"ltp": trigger_prices[1]}
                ]
            }
            orders = [
                {
                    "transaction_type": transaction_type,
                    "quantity": quantity,
                    "price": stop_loss_price,
                    "product": product,
                    "order_type": "LIMIT"
                },
                {
                    "transaction_type": transaction_type,
                    "quantity": quantity,
                    "price": target_price,
                    "product": product,
                    "order_type": "LIMIT"
                }
            ]
        else:
            return {
                "status": "error",
                "error": f"Invalid trigger_type: {trigger_type}"
            }
        
        modified_id = kite.modify_gtt(
            trigger_id=trigger_id,
            trigger_type=kite.GTT_TYPE_SINGLE if trigger_type == "single" else kite.GTT_TYPE_OCO,
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            trigger_values=trigger_values,
            last_price=last_price,
            orders=orders
        )
        
        return {
            "status": "success",
            "trigger_id": modified_id,
            "message": f"GTT order modified successfully. Trigger ID: {modified_id}"
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error modifying GTT order: {str(e)}"
        }


@tool
def delete_gtt_tool(trigger_id: int) -> dict:
    """
    Delete a GTT order.
    
    Args:
        trigger_id: GTT trigger ID to delete
        
    Returns:
        dict with deletion status
    """
    try:
        kite = get_kite_instance()
        deleted_id = kite.delete_gtt(trigger_id)
        
        return {
            "status": "success",
            "trigger_id": deleted_id,
            "message": f"GTT order deleted successfully. Trigger ID: {deleted_id}"
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error deleting GTT order: {str(e)}"
        }


@tool
def get_gtt_tool(trigger_id: int) -> dict:
    """
    Get details of a specific GTT order.
    
    Args:
        trigger_id: GTT trigger ID
        
    Returns:
        dict with GTT details
    """
    try:
        kite = get_kite_instance()
        gtt = kite.get_gtt(trigger_id)
        
        return {
            "status": "success",
            "data": gtt
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error getting GTT: {str(e)}"
        }


@tool
def get_gtts_tool() -> dict:
    """
    Get list of all GTT orders in the account.
    
    Returns:
        dict with list of GTT orders
    """
    try:
        kite = get_kite_instance()
        gtts = kite.get_gtts()
        
        return {
            "status": "success",
            "data": gtts,
            "count": len(gtts) if isinstance(gtts, list) else 0
        }
    except KiteException as e:
        return {
            "status": "error",
            "error": f"Kite API error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error getting GTTs: {str(e)}"
        }

