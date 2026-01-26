"""
View all GTT orders in the account
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from agent.tools.kite_tools import get_gtts_tool, get_gtt_tool
from utils.kite_utils import get_kite_instance
from kiteconnect import KiteConnect

def view_gtt_orders():
    """View all GTT orders"""
    try:
        print("=" * 70)
        print("GTT Orders in Account")
        print("=" * 70)
        
        # Get all GTTs
        result = get_gtts_tool.invoke({})
        
        if result.get("status") != "success":
            print(f"✗ Error: {result.get('error')}")
            return
        
        gtts = result.get("data", [])
        count = result.get("count", 0)
        
        print(f"\nTotal GTT Orders: {count}\n")
        
        if count == 0:
            print("No GTT orders found in your account.")
            return
        
        # Display each GTT
        for idx, gtt in enumerate(gtts, 1):
            print(f"{'=' * 70}")
            print(f"GTT #{idx}")
            print(f"{'=' * 70}")
            
            # Extract key information
            trigger_id = gtt.get("id", "N/A")
            status = gtt.get("status", "N/A")
            tradingsymbol = gtt.get("tradingsymbol", "N/A")
            exchange = gtt.get("exchange", "N/A")
            created_at = gtt.get("created_at", "N/A")
            updated_at = gtt.get("updated_at", "N/A")
            expires_at = gtt.get("expires_at", "N/A")
            
            condition = gtt.get("condition", {})
            trigger_type = condition.get("type", "N/A")
            
            orders = gtt.get("orders", [])
            
            print(f"Trigger ID: {trigger_id}")
            print(f"Status: {status}")
            print(f"Symbol: {tradingsymbol} ({exchange})")
            print(f"Type: {trigger_type}")
            print(f"Created: {created_at}")
            print(f"Updated: {updated_at}")
            print(f"Expires: {expires_at}")
            
            # Show trigger conditions
            if trigger_type == "single":
                trigger_values = condition.get("trigger_values", {})
                ltp = trigger_values.get("ltp", "N/A")
                print(f"\nTrigger Condition:")
                print(f"  - LTP: ₹{ltp}")
            elif trigger_type == "two-leg":
                trigger_values = condition.get("trigger_values", {})
                ltp_list = trigger_values.get("ltp", [])
                if len(ltp_list) >= 2:
                    print(f"\nTrigger Conditions (OCO):")
                    print(f"  - Stop Loss Trigger: ₹{ltp_list[0].get('ltp', 'N/A')}")
                    print(f"  - Target Trigger: ₹{ltp_list[1].get('ltp', 'N/A')}")
            
            # Show orders
            print(f"\nOrders ({len(orders)}):")
            for order_idx, order in enumerate(orders, 1):
                print(f"  Order {order_idx}:")
                print(f"    - Transaction: {order.get('transaction_type', 'N/A')}")
                print(f"    - Quantity: {order.get('quantity', 'N/A')}")
                print(f"    - Price: ₹{order.get('price', 'N/A')}")
                print(f"    - Product: {order.get('product', 'N/A')}")
                print(f"    - Order Type: {order.get('order_type', 'N/A')}")
            
            print()
        
        print("=" * 70)
        print("To get details of a specific GTT, use:")
        print("  python3.12 view_gtt_details.py <trigger_id>")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    view_gtt_orders()

