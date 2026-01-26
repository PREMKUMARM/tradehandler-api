import sys
import os
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from agent.tools.market_tools import get_historical_data_tool
from agent.tools.trading_opportunities_tool import calculate_vwap, calculate_rsi

# Fetch ONGC data for Dec 24, 2025
data = get_historical_data_tool.invoke({
    "instrument_name": "ONGC",
    "from_date": "2025-12-18", # Fetch more for indicators
    "to_date": "2025-12-24",
    "interval": "5minute"
})

if "data" in data:
    df = pd.DataFrame(data["data"])
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    
    # Calculate indicators
    df['vwap'] = calculate_vwap(df)
    df['rsi'] = calculate_rsi(df['close'].tolist())
    
    # Filter for Dec 24
    df_yesterday = df[df['date_only'] == datetime(2025, 12, 24).date()]
    
    print(f"Total rows for yesterday: {len(df_yesterday)}")
    if not df_yesterday.empty:
        # Check vwap distance and rsi
        df_yesterday['vwap_dist_pct'] = abs(df_yesterday['close'] - df_yesterday['vwap']) / df_yesterday['vwap'] * 100
        
        # Look for potential signals (Price near VWAP < 0.35% AND RSI < 45 or > 55)
        buy_pullback = df_yesterday[(df_yesterday['close'] > df_yesterday['vwap']) & (df_yesterday['vwap_dist_pct'] <= 0.35) & (df_yesterday['rsi'] < 45)]
        sell_pullback = df_yesterday[(df_yesterday['close'] < df_yesterday['vwap']) & (df_yesterday['vwap_dist_pct'] <= 0.35) & (df_yesterday['rsi'] > 55)]
        
        print(f"Potential Buy Pullbacks found: {len(buy_pullback)}")
        print(f"Potential Sell Pullbacks found: {len(sell_pullback)}")
        
        if not buy_pullback.empty:
             print("\nSample Buy Pullbacks (Price > VWAP, Dist <= 0.35%, RSI < 45):")
             print(buy_pullback[['date', 'close', 'vwap', 'vwap_dist_pct', 'rsi']].head())
             
        if not sell_pullback.empty:
             print("\nSample Sell Pullbacks (Price < VWAP, Dist <= 0.35%, RSI > 55):")
             print(sell_pullback[['date', 'close', 'vwap', 'vwap_dist_pct', 'rsi']].head())
    else:
        print("No data for yesterday in the result.")
else:
    print(f"Error fetching data: {data}")
