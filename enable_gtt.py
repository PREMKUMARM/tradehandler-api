"""
Quick script to enable GTT orders via API
"""
import sys
from pathlib import Path
import requests
import json

sys.path.append(str(Path(__file__).parent))

def enable_gtt(api_url="http://localhost:8000", enable_intraday=True, enable_positional=True):
    """Enable GTT orders via API"""
    try:
        url = f"{api_url}/agent/config"
        payload = {
            "use_gtt_orders": True,
            "gtt_for_intraday": enable_intraday,
            "gtt_for_positional": enable_positional
        }
        
        print("=" * 70)
        print("Enabling GTT Orders")
        print("=" * 70)
        print(f"\nUpdating configuration...")
        print(f"  use_gtt_orders: True")
        print(f"  gtt_for_intraday: {enable_intraday}")
        print(f"  gtt_for_positional: {enable_positional}")
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                print(f"\n✅ GTT enabled successfully!")
                print(f"\nNow when you approve trades:")
                if enable_intraday:
                    print(f"  - MIS (intraday) trades will use GTT OCO orders")
                if enable_positional:
                    print(f"  - CNC (positional) trades will use GTT OCO orders")
                return True
            else:
                print(f"\n❌ Error: {result.get('message', 'Unknown error')}")
                return False
        else:
            print(f"\n❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to API at {api_url}")
        print(f"   Make sure the server is running: python3 -m uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    enable_intraday = True
    enable_positional = True
    
    if len(sys.argv) > 1:
        if "intraday" in sys.argv[1].lower():
            enable_positional = False
        elif "positional" in sys.argv[1].lower():
            enable_intraday = False
    
    success = enable_gtt(enable_intraday=enable_intraday, enable_positional=enable_positional)
    if success:
        print(f"\n" + "=" * 70)
        print("Next: Run test_gtt_itc_signal.py to create a test approval")
        print("=" * 70)
    sys.exit(0 if success else 1)

