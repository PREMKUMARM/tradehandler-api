"""
Simulation helper functions
"""
from datetime import datetime, timedelta, time
from .state import simulation_state


def get_instrument_history(kite, token, sim_date):
    """Helper to fetch and cache full day history for an instrument during simulation"""
    cache_key = f"{token}_{sim_date}"
    if cache_key in simulation_state["instrument_history"]:
        return simulation_state["instrument_history"][cache_key]
    
    try:
        data = kite.historical_data(token, sim_date, sim_date + timedelta(days=1), "minute")
        # Filter for market hours to match Nifty candles index
        filtered_data = [c for c in data if time(9, 15) <= c["date"].time() <= time(15, 30)]
        simulation_state["instrument_history"][cache_key] = filtered_data
        return filtered_data
    except Exception as e:
        print(f"Error caching instrument {token}: {e}")
        return []


def add_sim_order(strategy, symbol, action, quantity, price, order_type="MARKET", status="COMPLETE", reason=""):
    """Helper to track mock orders in simulation"""
    order = {
        "order_id": f"SIM{datetime.now().strftime('%H%M%S%f')[:-3]}",
        "order_timestamp": simulation_state["time"] if "time" in simulation_state else datetime.now().strftime("%H:%M:%S"),
        "strategy": strategy,
        "tradingsymbol": symbol,
        "transaction_type": action,  # BUY/SELL
        "quantity": quantity,
        "average_price": price,
        "order_type": order_type,
        "status": status,
        "reason": reason
    }
    simulation_state["orders"].insert(0, order)  # Keep newest at top
    return order


def calculate_sim_qty(entry_price, fund, risk_pct, reward_pct):
    """Calculates quantity based on User's Risk Management formula: Qty = Risk / (Target - SL)"""
    # Professional Filter: Ignore trades if total gap (Risk + Reward) is less than 10% of premium
    # This prevents entering trades where the targets are too small to cover brokerage/slippage
    # and results in unrealistically high quantities.
    total_gap_pct = reward_pct + risk_pct
    if total_gap_pct < 10:
        return 0  # Signal to skip this trade
        
    risk_amount = fund * (risk_pct / 100)
    
    # price_range = Target - SL = (Reward% + Risk%) of Entry
    price_range = abs(entry_price) * total_gap_pct / 100
    
    if price_range <= 0.1: 
        price_range = 0.1  # Tick size safety
    
    target_qty = risk_amount / price_range
    
    # Round to nearest lot of 75, minimum 75
    lots = round(target_qty / 75)
    if lots < 1: 
        lots = 1
    
    return lots * 75


def find_option(nifty_options, strike, type, sim_date=None):
    """Helper to find an option instrument by strike and type, with expiry awareness"""
    matches = [o for o in nifty_options if o.get("strike") == strike and o.get("instrument_type") == type]
    if not matches: 
        return None
    
    # If in simulation, find the nearest expiry >= simulation date
    if sim_date:
        if isinstance(sim_date, str):
            sim_date = datetime.strptime(sim_date, "%Y-%m-%d").date()
        elif isinstance(sim_date, datetime):
            sim_date = sim_date.date()
            
        expiries = sorted(list(set([o["expiry"] for o in matches if o.get("expiry")])))
        for exp in expiries:
            if exp >= sim_date:
                return next((o for o in matches if o["expiry"] == exp), matches[0])
                
    return matches[0]





