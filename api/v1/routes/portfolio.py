"""
Portfolio management API endpoints (balance, positions, orders, etc.)
"""
from fastapi import APIRouter, Request, HTTPException

from utils.kite_utils import (
    api_key,
    get_access_token,
    get_kite_instance
)
from kiteconnect.exceptions import KiteException
from core.user_context import get_user_id_from_request

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


@router.get("/balance")
def get_balance(request: Request):
    """Get user margin/funds"""
    try:
        user_id = "default"
        try:
            user_id = get_user_id_from_request(request)
        except:
            pass
        
        kite = get_kite_instance(user_id=user_id)
        margins = kite.margins()
        
        equity_data = margins.get('equity', {})
        
        print(f"Raw Kite Connect margins response: {margins}")
        print(f"Equity data: {equity_data}")
        
        available_value = equity_data.get('available', 0)
        utilised_value = equity_data.get('utilised', 0)
        
        # Extract numeric values if they're dictionaries
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
        
        total_margin = equity_data.get('net', 0) or equity_data.get('live_balance', 0)
        
        print(f"Calculated available_margin: {available_margin}")
        print(f"Calculated utilised_margin: {utilised_margin}")
        print(f"Calculated total_margin: {total_margin}")
        
        # Transform to match frontend expected format (Upstox-style)
        transformed_margins = {
            "equity": {
                "_available_margin": float(available_margin) if available_margin else 0.0,
                "_utilised_margin": float(utilised_margin) if utilised_margin else 0.0,
                "_total_margin": float(total_margin) if total_margin else 0.0,
                **equity_data
            },
            "commodity": margins.get('commodity', {})
        }
        
        print(f"Transformed margins structure: {transformed_margins}")
        
        return {"data": transformed_margins}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting balance: {str(e)}")


@router.get("/positions")
def get_positions(request: Request):
    """Get user positions"""
    try:
        user_id = "default"
        try:
            user_id = get_user_id_from_request(request)
        except:
            pass
        
        kite = get_kite_instance(user_id=user_id)
        positions = kite.positions()
        
        # Transform Kite Connect response to match frontend expected format
        transformed_positions = []
        
        # Combine net and day positions
        all_positions = positions.get('net', []) + positions.get('day', [])
        
        for position in all_positions:
            transformed_position = {}
            # Map Kite Connect fields to Upstox-style fields with underscore prefix
            field_mapping = {
                'tradingsymbol': '_trading_symbol',
                'exchange': '_exchange',
                'instrument_token': '_instrument_token',
                'product': '_product',
                'quantity': '_quantity',
                'average_price': '_average_price',
                'last_price': '_last_price',
                'pnl': '_pnl',
                'net_quantity': '_net_quantity',
                'sell_quantity': '_sell_quantity',
                'buy_quantity': '_buy_quantity',
                'buy_price': '_buy_price',
                'sell_price': '_sell_price',
                'overnight_quantity': '_overnight_quantity',
                'multiplier': '_multiplier'
            }
            
            # Add transformed fields with underscore prefix
            for kite_field, upstox_field in field_mapping.items():
                if kite_field in position:
                    transformed_position[upstox_field] = position[kite_field]
            
            # Also keep original fields for compatibility
            transformed_position.update(position)
            transformed_positions.append(transformed_position)
        
        return {"data": transformed_positions}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting positions: {str(e)}")


@router.get("/orders")
def get_orders(request: Request):
    """Get user orders"""
    try:
        user_id = "default"
        try:
            user_id = get_user_id_from_request(request)
        except:
            pass
        
        kite = get_kite_instance(user_id=user_id)
        orders = kite.orders()
        
        # Transform Kite Connect response to match frontend expected format (Upstox-style)
        transformed_orders = []
        
        # Status mapping from Kite Connect to Upstox format (lowercase)
        status_mapping = {
            'OPEN': 'open',
            'COMPLETE': 'complete',
            'CANCELLED': 'cancelled',
            'REJECTED': 'rejected',
            'CANCELLED AMO': 'cancelled',
            'TRANSIT': 'transit',
            'PENDING': 'pending',
            'VALIDATION PENDING': 'pending'
        }
        
        for order in orders:
            transformed_order = {}
            # Map Kite Connect fields to Upstox-style fields with underscore prefix
            field_mapping = {
                'order_id': '_order_id',
                'exchange_order_id': '_exchange_order_id',
                'tradingsymbol': '_trading_symbol',
                'exchange': '_exchange',
                'instrument_token': '_instrument_token',
                'transaction_type': '_transaction_type',
                'quantity': '_quantity',
                'price': '_price',
                'trigger_price': '_trigger_price',
                'product': '_product',
                'order_type': '_order_type',
                'status': '_status',
                'status_message': '_status_message',
                'order_timestamp': '_order_timestamp',
                'exchange_timestamp': '_exchange_timestamp',
                'validity': '_validity',
                'variety': '_variety',
                'disclosed_quantity': '_disclosed_quantity',
                'tag': '_tag',
                'average_price': '_average_price',
                'filled_quantity': '_filled_quantity',
                'pending_quantity': '_pending_quantity',
                'cancelled_quantity': '_cancelled_quantity'
            }
            
            # Add transformed fields with underscore prefix
            for kite_field, upstox_field in field_mapping.items():
                if kite_field in order:
                    value = order[kite_field]
                    # Normalize status to lowercase
                    if kite_field == 'status' and value:
                        value = status_mapping.get(value.upper(), value.lower())
                    transformed_order[upstox_field] = value
            
            # Special handling for price field
            if '_price' in transformed_order:
                price_value = transformed_order.get('_price', 0)
                if not price_value or price_value == 0:
                    avg_price = order.get('average_price', 0)
                    if avg_price and avg_price > 0:
                        transformed_order['_price'] = avg_price
                        print(f"Order {order.get('order_id')}: Using average_price {avg_price} instead of price {price_value}")
            
            # Also keep original fields for compatibility
            transformed_order.update(order)
            transformed_orders.append(transformed_order)
        
        return {"data": transformed_orders}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting orders: {str(e)}")


@router.get("/live-positions")
def get_live_positions(request: Request):
    """Fetch current open positions from Zerodha Kite"""
    try:
        user_id = "default"
        try:
            user_id = get_user_id_from_request(request)
        except:
            pass
        
        kite = get_kite_instance(user_id=user_id)
        positions = kite.positions()
        
        # Calculate live MTM and totals
        net_positions = positions.get("net", [])
        total_pnl = 0
        active_count = 0
        
        for pos in net_positions:
            total_pnl += pos.get("pnl", 0)
            if pos.get("quantity", 0) != 0:
                active_count += 1
                
        return {
            "data": {
                "positions": net_positions,
                "total_pnl": round(total_pnl, 2),
                "active_count": active_count
            }
        }
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching positions: {str(e)}")


@router.get("/ws-portfolio")
def get_portfolio_ws(request: Request):
    """Get WebSocket authorization for portfolio streaming"""
    try:
        access_token = get_access_token()
        if not access_token:
            raise HTTPException(status_code=401, detail="Access token not found")
        
        # Construct Kite Connect WebSocket URL
        ws_url = f"wss://ws.kite.trade?api_key={api_key}&access_token={access_token}"
        
        # Return in format expected by frontend (Upstox-style)
        return {
            "data": {
                "_authorized_redirect_uri": ws_url,
                "api_key": api_key,
                "access_token": access_token[:20] + "..." if access_token else None,
                "message": "WebSocket URL for portfolio streaming"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting WebSocket info: {str(e)}")


@router.get("/ws-orders")
def get_orders_ws(request: Request):
    """Get WebSocket authorization for orders streaming"""
    try:
        access_token = get_access_token()
        if not access_token:
            raise HTTPException(status_code=401, detail="Access token not found")
        
        # Construct Kite Connect WebSocket URL
        ws_url = f"wss://ws.kite.trade?api_key={api_key}&access_token={access_token}"
        
        # Return in format expected by frontend (Upstox-style)
        return {
            "data": {
                "_authorized_redirect_uri": ws_url,
                "api_key": api_key,
                "access_token": access_token[:20] + "..." if access_token else None,
                "message": "WebSocket URL for orders streaming"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting WebSocket info: {str(e)}")

