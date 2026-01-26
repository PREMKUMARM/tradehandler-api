"""
View details of a specific GTT order
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python3.12 view_gtt_details.py <trigger_id>")
    sys.exit(1)

trigger_id = int(sys.argv[1])

sys.path.append(str(Path(__file__).parent))

from agent.tools.kite_tools import get_gtt_tool
import json

def view_gtt_details(trigger_id):
    """View details of a specific GTT"""
    try:
        print("=" * 70)
        print(f"GTT Order Details - Trigger ID: {trigger_id}")
        print("=" * 70)
        
        result = get_gtt_tool.invoke({"trigger_id": trigger_id})
        
        if result.get("status") != "success":
            print(f"✗ Error: {result.get('error')}")
            return
        
        gtt = result.get("data", {})
        
        # Pretty print the GTT details
        print("\nFull GTT Details:")
        print(json.dumps(gtt, indent=2, default=str))
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    view_gtt_details(trigger_id)

