import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Mock the environment to run the tool logic
sys.path.append(os.getcwd())
from agent.tools.trading_opportunities_tool import find_indicator_based_trading_opportunities
from utils.kite_utils import get_kite_instance

# Run for ONGC yesterday (2025-12-24)
result = find_indicator_based_trading_opportunities.invoke({
    "instrument_name": "ONGC",
    "from_date": "2025-12-24",
    "to_date": "2025-12-24",
    "interval": "5minute"
})

print(f"Status: {result.get('status')}")
if result.get('status') == 'success':
    print(f"Instrument: {result.get('instrument')}")
    opps = result.get('opportunities', [])
    print(f"Total Opportunities: {len(opps)}")
    if not opps:
        print(f"Message: {result.get('summary', {}).get('message')}")
    for i, opp in enumerate(opps):
        print(f"Trade {i+1}: {opp['signal_type']} at {opp['entry_time']} Price: {opp['entry_price']} Exit: {opp['exit_reason']}")
else:
    print(f"Error: {result.get('error')}")
