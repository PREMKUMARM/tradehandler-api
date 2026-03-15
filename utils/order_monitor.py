"""
Order monitoring service for stoploss and target management
"""
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import json
from utils.logger import log_info, log_error, log_warning
from utils.kite_utils import get_kite_instance
from kiteconnect.exceptions import KiteException


class OrderMonitor:
    """Monitors active orders and places stoploss/target orders"""
    
    def __init__(self):
        self.monitored_orders: Dict[str, Dict] = {}
        self.is_running = False
        self.monitor_task = None
    
    async def start_monitoring(self):
        """Start the order monitoring service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        log_info("Order monitoring service started")
    
    async def stop_monitoring(self):
        """Stop the order monitoring service"""
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        log_info("Order monitoring service stopped")
    
    def add_order(self, order_id: str, symbol: str, transaction_type: str, 
                  quantity: int, stoploss: Optional[float] = None, 
                  target: Optional[float] = None, trailing_stoploss: Optional[float] = None):
        """Add an order to monitor for stoploss/target"""
        self.monitored_orders[order_id] = {
            "order_id": order_id,
            "symbol": symbol,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "stoploss": stoploss,
            "target": target,
            "trailing_stoploss": trailing_stoploss,
            "original_stoploss": stoploss,
            "highest_price": 0,
            "lowest_price": float('inf'),
            "created_at": datetime.now(),
            "stoploss_triggered": False,
            "target_triggered": False
        }
        log_info(f"Added order {order_id} to monitoring: SL={stoploss}, TGT={target}")
    
    def remove_order(self, order_id: str):
        """Remove an order from monitoring"""
        if order_id in self.monitored_orders:
            del self.monitored_orders[order_id]
            log_info(f"Removed order {order_id} from monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                await self._check_orders()
                await asyncio.sleep(2)  # Check every 2 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(f"Error in order monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_orders(self):
        """Check all monitored orders for stoploss/target triggers"""
        if not self.monitored_orders:
            return
        
        try:
            kite = get_kite_instance()
            
            # Get current prices for all symbols
            symbols = list(set(order["symbol"] for order in self.monitored_orders.values()))
            quotes = {}
            
            for symbol in symbols:
                try:
                    quote = kite.quote([symbol])
                    if symbol in quote:
                        quotes[symbol] = quote[symbol]["last_price"]
                except KiteException as e:
                    log_error(f"Error getting quote for {symbol}: {e}")
                    continue
            
            # Check each order
            orders_to_remove = []
            for order_id, order in self.monitored_orders.items():
                if order["stoploss_triggered"] or order["target_triggered"]:
                    continue
                
                symbol = order["symbol"]
                if symbol not in quotes:
                    continue
                
                current_price = quotes[symbol]
                transaction_type = order["transaction_type"]
                
                # Update trailing values
                if transaction_type == "BUY":
                    order["highest_price"] = max(order["highest_price"], current_price)
                else:
                    order["lowest_price"] = min(order["lowest_price"], current_price)
                
                # Check stoploss trigger
                if order["stoploss"] and not order["stoploss_triggered"]:
                    if self._should_trigger_stoploss(order, current_price, transaction_type):
                        await self._place_exit_order(order, "STOPLOSS", current_price)
                        order["stoploss_triggered"] = True
                        orders_to_remove.append(order_id)
                        continue
                
                # Check target trigger
                if order["target"] and not order["target_triggered"]:
                    if self._should_trigger_target(order, current_price, transaction_type):
                        await self._place_exit_order(order, "TARGET", current_price)
                        order["target_triggered"] = True
                        orders_to_remove.append(order_id)
                        continue
                
                # Update trailing stoploss
                if order["trailing_stoploss"] and not order["stoploss_triggered"]:
                    self._update_trailing_stoploss(order, current_price, transaction_type)
            
            # Remove triggered orders
            for order_id in orders_to_remove:
                self.remove_order(order_id)
                
        except Exception as e:
            log_error(f"Error checking orders: {e}")
    
    def _should_trigger_stoploss(self, order: Dict, current_price: float, transaction_type: str) -> bool:
        """Check if stoploss should be triggered"""
        stoploss = order["stoploss"]
        
        if transaction_type == "BUY":
            # For BUY orders, stoploss triggers when price falls below stoploss
            return current_price <= stoploss
        else:
            # For SELL orders, stoploss triggers when price rises above stoploss
            return current_price >= stoploss
    
    def _should_trigger_target(self, order: Dict, current_price: float, transaction_type: str) -> bool:
        """Check if target should be triggered"""
        target = order["target"]
        
        if transaction_type == "BUY":
            # For BUY orders, target triggers when price reaches or exceeds target
            return current_price >= target
        else:
            # For SELL orders, target triggers when price falls to or below target
            return current_price <= target
    
    def _update_trailing_stoploss(self, order: Dict, current_price: float, transaction_type: str):
        """Update trailing stoploss"""
        if not order["trailing_stoploss"]:
            return
        
        trailing_amount = order["trailing_stoploss"]
        
        if transaction_type == "BUY":
            # For BUY orders, trail stoploss up as price increases
            new_stoploss = order["highest_price"] - trailing_amount
            if new_stoploss > order["stoploss"]:
                order["stoploss"] = new_stoploss
                log_info(f"Updated trailing stoploss for {order['order_id']}: {new_stoploss}")
        else:
            # For SELL orders, trail stoploss down as price decreases
            new_stoploss = order["lowest_price"] + trailing_amount
            if new_stoploss < order["stoploss"]:
                order["stoploss"] = new_stoploss
                log_info(f"Updated trailing stoploss for {order['order_id']}: {new_stoploss}")
    
    async def _place_exit_order(self, order: Dict, reason: str, current_price: float):
        """Place exit order for stoploss/target"""
        try:
            kite = get_kite_instance()
            
            # Determine exit transaction type
            exit_transaction_type = "SELL" if order["transaction_type"] == "BUY" else "BUY"
            
            # Place market order for immediate exit
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,  # Default to NSE, should be configurable
                tradingsymbol=order["symbol"],
                transaction_type=exit_transaction_type,
                quantity=order["quantity"],
                product=kite.PRODUCT_MIS,  # Default to MIS
                order_type=kite.ORDER_TYPE_MARKET,
                tag=f"algofeast-{reason.lower()}"
            )
            
            log_info(f"Placed {reason} exit order {order_id} for {order['symbol']} at {current_price}")
            
        except KiteException as e:
            log_error(f"Error placing {reason} exit order: {e}")
        except Exception as e:
            log_error(f"Unexpected error placing {reason} exit order: {e}")


# Global instance
order_monitor = OrderMonitor()
