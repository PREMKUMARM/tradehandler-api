"""
Simulation state management
"""
# Global state for Simulation
simulation_state = {
    "is_active": False,
    "date": None,
    "current_index": 0,
    "speed": 1,
    "candles": [],
    "instrument_history": {},  # Cache for all traded instrument historical data
    "nifty_options": [],
    "atm_ce": None,
    "atm_pe": None,
    "positions": [],
    "orders": [],  # Track entry and exit orders
    "strategy_cooldown": {},  # Track last exit time per strategy to prevent infinite loops
    "executed_strategies": set(),
    "nifty_price": 0,
    "last_update": None
}

# Global state for Live Monitoring Logs
live_logs = []

