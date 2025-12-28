"""
Order management API endpoints
"""
from fastapi import APIRouter, Request, HTTPException
from typing import Optional

from utils.kite_utils import get_kite_instance
from kiteconnect.exceptions import KiteException
from core.user_context import get_user_id_from_request

router = APIRouter(prefix="/orders", tags=["Orders"])


@router.post("/place")
async def place_order(req: Request):
    """Place an order"""
    try:
        user_id = "default"
        try:
            user_id = get_user_id_from_request(req)
        except:
            pass
        
        kite = get_kite_instance(user_id=user_id)
        payload = await req.json()
        
        # Map Upstox format to Kite Connect format
        kite_order_params = {
            "variety": payload.get('variety', kite.VARIETY_REGULAR),
            "exchange": payload.get('exchange', kite.EXCHANGE_NSE),
            "tradingsymbol": payload.get('tradingsymbol') or payload.get('instrument_token', '').split('|')[-1],
            "transaction_type": payload.get('transaction_type', payload.get('transactionType', 'BUY')),
            "quantity": payload.get('quantity', payload.get('qty', 1)),
            "price": payload.get('price'),
            "product": payload.get('product', kite.PRODUCT_MIS),
            "order_type": payload.get('order_type', payload.get('orderType', kite.ORDER_TYPE_MARKET)),
            "validity": payload.get('validity', kite.VALIDITY_DAY),
            "disclosed_quantity": payload.get('disclosed_quantity', payload.get('disclosedQuantity', 0)),
            "trigger_price": payload.get('trigger_price', payload.get('triggerPrice')),
            "squareoff": payload.get('squareoff'),
            "stoploss": payload.get('stoploss'),
            "trailing_stoploss": payload.get('trailing_stoploss'),
            "tag": payload.get('tag', 'algofeast')
        }
        
        # Remove None values
        kite_order_params = {k: v for k, v in kite_order_params.items() if v is not None}
        
        order_id = kite.place_order(**kite_order_params)
        return {"data": {"order_id": order_id}, "status": "success"}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing order: {str(e)}")


@router.post("/modify")
async def modify_order(req: Request):
    """Modify an existing order"""
    try:
        user_id = "default"
        try:
            user_id = get_user_id_from_request(req)
        except:
            pass
        
        kite = get_kite_instance(user_id=user_id)
        payload = await req.json()
        
        order_id = payload.get('orderId') or payload.get('order_id')
        if not order_id:
            raise HTTPException(status_code=400, detail="orderId is required")
        
        # Map Upstox format to Kite Connect format
        modify_params = {
            "variety": payload.get('variety', kite.VARIETY_REGULAR),
            "order_id": str(order_id),
            "quantity": payload.get('quantity'),
            "price": payload.get('price'),
            "order_type": payload.get('order_type', payload.get('orderType')),
            "validity": payload.get('validity', kite.VALIDITY_DAY),
            "disclosed_quantity": payload.get('disclosed_quantity', payload.get('disclosedQuantity')),
            "trigger_price": payload.get('trigger_price', payload.get('triggerPrice'))
        }
        
        # Remove None values
        modify_params = {k: v for k, v in modify_params.items() if v is not None}
        
        order_id_modified = kite.modify_order(**modify_params)
        return {"data": {"order_id": order_id_modified}, "status": "success"}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error modifying order: {str(e)}")


@router.post("/cancel")
async def cancel_order(req: Request):
    """Cancel an order"""
    try:
        user_id = "default"
        try:
            user_id = get_user_id_from_request(req)
        except:
            pass
        
        kite = get_kite_instance(user_id=user_id)
        payload = await req.json()
        
        order_id = payload.get('order_id') or payload.get('orderId')
        if not order_id:
            raise HTTPException(status_code=400, detail="order_id is required")
        
        variety = payload.get('variety', kite.VARIETY_REGULAR)
        order_id_cancelled = kite.cancel_order(variety=variety, order_id=str(order_id))
        return {"data": {"order_id": order_id_cancelled}, "status": "success"}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling order: {str(e)}")


@router.post("/sell")
async def sell_order(req: Request):
    """Place a sell order (same as placeOrder but with transaction_type=SELL)"""
    try:
        user_id = "default"
        try:
            user_id = get_user_id_from_request(req)
        except:
            pass
        
        kite = get_kite_instance(user_id=user_id)
        payload = await req.json()
        
        # Ensure transaction_type is SELL
        payload['transaction_type'] = 'SELL'
        
        # Map to Kite Connect format
        kite_order_params = {
            "variety": payload.get('variety', kite.VARIETY_REGULAR),
            "exchange": payload.get('exchange', kite.EXCHANGE_NSE),
            "tradingsymbol": payload.get('tradingsymbol') or payload.get('instrument_token', '').split('|')[-1],
            "transaction_type": 'SELL',
            "quantity": payload.get('quantity', payload.get('qty', 1)),
            "price": payload.get('price'),
            "product": payload.get('product', kite.PRODUCT_MIS),
            "order_type": payload.get('order_type', payload.get('orderType', kite.ORDER_TYPE_MARKET)),
            "validity": payload.get('validity', kite.VALIDITY_DAY),
            "disclosed_quantity": payload.get('disclosed_quantity', payload.get('disclosedQuantity', 0)),
            "trigger_price": payload.get('trigger_price', payload.get('triggerPrice')),
            "tag": payload.get('tag', 'algofeast')
        }
        
        # Remove None values
        kite_order_params = {k: v for k, v in kite_order_params.items() if v is not None}
        
        order_id = kite.place_order(**kite_order_params)
        return {"data": {"order_id": order_id}, "status": "success"}
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing sell order: {str(e)}")


@router.post("/place-strategy-order")
async def place_strategy_order(req: Request):
    """Place a live order based on a strategy signal"""
    try:
        user_id = "default"
        try:
            user_id = get_user_id_from_request(req)
        except:
            pass
        
        kite = get_kite_instance(user_id=user_id)
        payload = await req.json()
        
        strategy_type = payload.get("strategy_type")
        symbol = payload.get("tradingsymbol")
        exchange = payload.get("exchange", "NFO")
        transaction_type = payload.get("transaction_type", "BUY")
        quantity = payload.get("quantity", 75)
        order_type = payload.get("order_type", "MARKET")
        product = payload.get("product", "MIS")  # MIS for Intraday
        
        is_multi_leg = payload.get("multi_leg", False)
        legs = payload.get("legs", [])
        
        order_ids = []
        
        if is_multi_leg and legs:
            # Place multi-leg orders (e.g., Bull Call Spread)
            for leg in legs:
                order_id = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=exchange,
                    tradingsymbol=leg.get("tradingsymbol"),
                    transaction_type=leg.get("action"),  # BUY or SELL
                    quantity=quantity,
                    product=product,
                    order_type=order_type
                )
                order_ids.append(order_id)
        else:
            # Place single leg order
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=order_type
            )
            order_ids.append(order_id)
            
        return {
            "status": "success",
            "message": f"Successfully placed {len(order_ids)} order(s)",
            "order_ids": order_ids
        }
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing order: {str(e)}")


@router.post("/exit-all-positions")
def exit_all_positions(req: Request):
    """Kill Switch: Exit all open positions immediately at MARKET price"""
    try:
        user_id = "default"
        try:
            user_id = get_user_id_from_request(req)
        except:
            pass
        
        # Check if in simulation mode (import from main if needed)
        try:
            from main import simulation_state, add_sim_order
            if simulation_state.get("is_active", False):
                exit_count = 0
                for pos in simulation_state.get("positions", []):
                    if pos.get("quantity", 0) != 0:
                        old_qty = pos["quantity"]
                        pos["quantity"] = 0
                        pos["exit_reason"] = "Manual Exit"
                        add_sim_order(pos.get("strategy"), pos.get("tradingsymbol"), "SELL", old_qty, pos.get("last_price"), reason="Manual Exit (Emergency)")
                        exit_count += 1
                return {"status": "success", "message": f"Simulation: Exited {exit_count} positions"}
        except (ImportError, AttributeError):
            # Simulation state not available, proceed with real positions
            pass

        kite = get_kite_instance(user_id=user_id)
        positions = kite.positions().get("net", [])
        exit_orders = []
        
        for pos in positions:
            qty = pos.get("quantity", 0)
            if qty != 0:
                # Opposite transaction to close
                trans_type = kite.TRANSACTION_TYPE_SELL if qty > 0 else kite.TRANSACTION_TYPE_BUY
                abs_qty = abs(qty)
                
                order_id = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=pos.get("exchange"),
                    tradingsymbol=pos.get("tradingsymbol"),
                    transaction_type=trans_type,
                    quantity=abs_qty,
                    product=pos.get("product"),
                    order_type=kite.ORDER_TYPE_MARKET
                )
                exit_orders.append({
                    "symbol": pos.get("tradingsymbol"),
                    "order_id": order_id
                })
                
        return {
            "status": "success",
            "message": f"Exited {len(exit_orders)} positions",
            "exits": exit_orders
        }
    except KiteException as e:
        raise HTTPException(status_code=400, detail=f"Kite API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exiting positions: {str(e)}")


