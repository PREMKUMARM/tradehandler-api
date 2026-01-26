"""
Test script to generate a BUY signal for ITC and place GTT order
This script:
1. Gets current ITC price
2. Calculates stop-loss and target based on VWAP strategy
3. Creates an approval
4. Places GTT OCO order when approved
5. Shows how to view the GTT order
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from utils.kite_utils import get_kite_instance
from agent.approval import get_approval_queue
from agent.config import get_agent_config
from agent.tools.kite_tools import place_gtt_tool, get_gtts_tool, get_gtt_tool
from agent.tools.risk_tools import suggest_position_size_tool

def get_current_price(symbol, exchange="NSE"):
    """Get current market price"""
    try:
        kite = get_kite_instance()
        quote = kite.quote(f"{exchange}:{symbol}")
        instrument_key = f"{exchange}:{symbol}"
        if instrument_key in quote:
            return quote[instrument_key].get("last_price", 0)
        return 0
    except Exception as e:
        print(f"Error getting price: {e}")
        return 0

def calculate_vwap_sl_tp(entry_price, transaction_type="BUY"):
    """Calculate stop-loss and target based on VWAP strategy"""
    config = get_agent_config()
    rr_ratio = config.reward_per_trade_pct / config.risk_per_trade_pct
    
    if transaction_type == "BUY":
        # Stop-loss: 0.5% below entry or VWAP (we'll use 0.5% below entry)
        sl = entry_price * 0.995
        # Target: 3x risk distance
        risk = entry_price - sl
        tp = entry_price + (risk * rr_ratio)
    else:  # SELL
        # Stop-loss: 0.5% above entry
        sl = entry_price * 1.005
        # Target: 3x risk distance
        risk = sl - entry_price
        tp = entry_price - (risk * rr_ratio)
    
    return round(sl, 2), round(tp, 2)

def test_gtt_itc_signal():
    """Generate ITC BUY signal and place GTT order"""
    try:
        print("=" * 70)
        print("ITC BUY Signal Test with GTT Order")
        print("=" * 70)
        
        symbol = "ITC"
        exchange = "NSE"
        transaction_type = "BUY"
        
        # 1. Get current price
        print(f"\n1. Getting current price for {symbol}...")
        current_price = get_current_price(symbol, exchange)
        if current_price == 0:
            print("   ✗ Could not get current price")
            return
        
        print(f"   ✓ Current Price: ₹{current_price}")
        
        # 2. Calculate stop-loss and target
        print(f"\n2. Calculating stop-loss and target...")
        sl, tp = calculate_vwap_sl_tp(current_price, transaction_type)
        print(f"   ✓ Entry Price: ₹{current_price}")
        print(f"   ✓ Stop Loss: ₹{sl}")
        print(f"   ✓ Target: ₹{tp}")
        print(f"   ✓ Risk: ₹{round(current_price - sl, 2)}")
        print(f"   ✓ Reward: ₹{round(tp - current_price, 2)}")
        print(f"   ✓ R:R Ratio: {round((tp - current_price) / (current_price - sl), 2)}:1")
        
        # 3. Calculate position size
        print(f"\n3. Calculating position size...")
        config = get_agent_config()
        position_size = suggest_position_size_tool.invoke({
            "entry_price": current_price,
            "stop_loss_price": sl,
            "available_capital": config.trading_capital,
            "risk_percentage": config.risk_per_trade_pct
        })
        quantity = position_size.get("suggested_quantity", 1)
        print(f"   ✓ Suggested Quantity: {quantity} shares")
        print(f"   ✓ Trade Value: ₹{round(current_price * quantity, 2)}")
        
        # 4. Create approval
        print(f"\n4. Creating approval...")
        approval_queue = get_approval_queue()
        risk_amt = abs(current_price - sl) * quantity
        reward_amt = abs(tp - current_price) * quantity
        
        approval_id = approval_queue.create_approval(
            action=f"LIVE_{transaction_type}",
            details={
                "symbol": symbol,
                "type": transaction_type,
                "price": current_price,
                "qty": quantity,
                "sl": sl,
                "tp": tp,
                "is_simulated": False,
                "timestamp": datetime.now().isoformat(),
                "signal_timestamp": datetime.now().isoformat(),
                "candle_timestamp": datetime.now().isoformat(),
                "timeframe": "5minute",
                "product": "MIS"  # Can change to "CNC" for positional
            },
            trade_value=current_price * quantity,
            risk_amount=risk_amt,
            reward_amount=reward_amt,
            reasoning=f"Institutional VWAP: {transaction_type} @ {symbol} (Test Signal)"
        )
        print(f"   ✓ Approval Created: {approval_id}")
        print(f"   ✓ Approval Status: PENDING")
        
        # 5. Show approval details
        approval = approval_queue.get_approval(approval_id)
        if approval:
            approval_dict = approval if isinstance(approval, dict) else approval.model_dump() if hasattr(approval, 'model_dump') else approval
            details = approval_dict.get('details', {}) if isinstance(approval_dict.get('details'), dict) else {}
            print(f"\n5. Approval Details:")
            print(f"   - Symbol: {details.get('symbol', 'N/A')}")
            print(f"   - Type: {details.get('type', 'N/A')}")
            print(f"   - Entry: ₹{details.get('price', 0)}")
            print(f"   - Quantity: {details.get('qty', 0)}")
            print(f"   - Stop Loss: ₹{details.get('sl', 0)}")
            print(f"   - Target: ₹{details.get('tp', 0)}")
            print(f"   - Risk: ₹{approval_dict.get('risk_amount', 0)}")
            print(f"   - Reward: ₹{approval_dict.get('reward_amount', 0)}")
            print(f"   - R:R Ratio: {approval_dict.get('rr_ratio', 0):.2f}:1")
            print(f"   - Product: {details.get('product', 'MIS')}")
        
        # 6. Check GTT configuration and show instructions
        print(f"\n" + "=" * 70)
        print("GTT CONFIGURATION CHECK:")
        print("=" * 70)
        print(f"  use_gtt_orders: {config.use_gtt_orders}")
        print(f"  gtt_for_intraday: {config.gtt_for_intraday}")
        print(f"  gtt_for_positional: {config.gtt_for_positional}")
        print(f"  Product: MIS (intraday)")
        
        will_use_gtt = config.use_gtt_orders and config.gtt_for_intraday
        if not will_use_gtt:
            print(f"\n⚠️  GTT WILL NOT BE USED!")
            if not config.use_gtt_orders:
                print(f"   Reason: use_gtt_orders is False")
            elif not config.gtt_for_intraday:
                print(f"   Reason: gtt_for_intraday is False (product is MIS)")
            print(f"\n   The system will place regular SL-M + LIMIT orders instead.")
            print(f"\n   To use GTT, enable it first:")
            print(f"   Quick way: algo-env/bin/python3.12 enable_gtt.py")
            print(f"   Or update config via API/UI:")
            print(f"      use_gtt_orders = true")
            print(f"      gtt_for_intraday = true")
            print(f"   2. Then approve this trade")
        else:
            print(f"\n✅ GTT IS ENABLED - Will use GTT OCO order!")
        
        print(f"\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print(f"1. {'Enable GTT first (see above), then ' if not will_use_gtt else ''}Go to Approvals UI")
        print(f"2. Approve approval_id: {approval_id}")
        print(f"3. System will place:")
        print(f"   - Entry order (MARKET)")
        if will_use_gtt:
            print(f"   - ✅ GTT OCO order (Stop-loss + Target)")
            print(f"4. After approval, run: algo-env/bin/python3.12 view_gtt_orders.py")
        else:
            print(f"   - ⚠️  Regular SL-M + LIMIT orders (2 separate orders)")
            print(f"4. Check logs for order placement details")
        print("=" * 70)
        
        return {
            "approval_id": approval_id,
            "symbol": symbol,
            "entry_price": current_price,
            "stop_loss": sl,
            "target": tp,
            "quantity": quantity
        }
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_gtt_itc_signal()
    if result:
        print(f"\n✅ Test signal created successfully!")
        print(f"   Approval ID: {result['approval_id']}")

