"""
Check GTT configuration and test GTT placement
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from agent.config import get_agent_config
from agent.tools.kite_tools import get_gtts_tool

def check_gtt_config():
    """Check if GTT is properly configured"""
    print("=" * 70)
    print("GTT Configuration Check")
    print("=" * 70)
    
    config = get_agent_config()
    
    print(f"\nCurrent GTT Settings:")
    print(f"  use_gtt_orders: {config.use_gtt_orders}")
    print(f"  gtt_for_intraday: {config.gtt_for_intraday}")
    print(f"  gtt_for_positional: {config.gtt_for_positional}")
    
    print(f"\nGTT Usage Logic:")
    print(f"  - For MIS (intraday) trades: {'✅ Will use GTT' if (config.use_gtt_orders and config.gtt_for_intraday) else '❌ Will NOT use GTT'}")
    print(f"  - For CNC (positional) trades: {'✅ Will use GTT' if (config.use_gtt_orders and config.gtt_for_positional) else '❌ Will NOT use GTT'}")
    
    if not config.use_gtt_orders:
        print(f"\n⚠️  GTT is DISABLED!")
        print(f"   To enable: Set use_gtt_orders=true in config")
    elif not config.gtt_for_intraday and not config.gtt_for_positional:
        print(f"\n⚠️  GTT is enabled but not configured for any product type!")
        print(f"   Enable gtt_for_intraday or gtt_for_positional")
    
    # Check existing GTT orders
    print(f"\nExisting GTT Orders:")
    try:
        result = get_gtts_tool.invoke({})
        if result.get("status") == "success":
            gtts = result.get("data", [])
            print(f"  Total GTTs: {len(gtts)}")
            if gtts:
                for gtt in gtts[:3]:  # Show first 3
                    print(f"    - ID: {gtt.get('id')}, Status: {gtt.get('status')}, Symbol: {gtt.get('tradingsymbol')}")
        else:
            print(f"  Error: {result.get('error')}")
    except Exception as e:
        print(f"  Error checking GTTs: {e}")
    
    print("\n" + "=" * 70)
    print("To enable GTT for intraday trades, update config:")
    print("  use_gtt_orders = true")
    print("  gtt_for_intraday = true")
    print("=" * 70)

if __name__ == "__main__":
    check_gtt_config()

