import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.kite_utils import get_kite_instance

def check_reliance():
    kite = get_kite_instance()
    instruments = kite.instruments("NSE")
    
    print("Matches for 'RELIANCE' in NSE:")
    for inst in instruments:
        symbol = inst.get("tradingsymbol", "")
        if "RELIANCE" in symbol:
            print(f"Symbol: {symbol}, Name: {inst.get('name')}, Token: {inst.get('instrument_token')}")
            # Get last price
            try:
                quote = kite.quote(f"NSE:{symbol}")
                lp = quote.get(f"NSE:{symbol}", {}).get("last_price")
                print(f"  Last Price: {lp}")
            except:
                print("  Could not get quote")

if __name__ == "__main__":
    check_reliance()

