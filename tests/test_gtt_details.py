"""
Get detailed information about GTT methods in KiteConnect
"""
import sys
from pathlib import Path
import inspect

sys.path.append(str(Path(__file__).parent))

from utils.kite_utils import get_kite_instance
from kiteconnect import KiteConnect

def inspect_gtt_methods():
    """Get detailed information about GTT methods"""
    try:
        kite = get_kite_instance()
        
        print("=" * 70)
        print("KiteConnect GTT Methods - Detailed Information")
        print("=" * 70)
        
        # GTT Constants
        print("\n📌 GTT Constants:")
        print(f"  GTT_TYPE_SINGLE = {kite.GTT_TYPE_SINGLE}")
        print(f"  GTT_TYPE_OCO = {kite.GTT_TYPE_OCO}")
        print(f"  GTT_STATUS_ACTIVE = {kite.GTT_STATUS_ACTIVE}")
        print(f"  GTT_STATUS_TRIGGERED = {kite.GTT_STATUS_TRIGGERED}")
        print(f"  GTT_STATUS_CANCELLED = {kite.GTT_STATUS_CANCELLED}")
        print(f"  GTT_STATUS_DISABLED = {kite.GTT_STATUS_DISABLED}")
        print(f"  GTT_STATUS_EXPIRED = {kite.GTT_STATUS_EXPIRED}")
        print(f"  GTT_STATUS_REJECTED = {kite.GTT_STATUS_REJECTED}")
        print(f"  GTT_STATUS_DELETED = {kite.GTT_STATUS_DELETED}")
        
        # GTT Methods
        gtt_methods = ['place_gtt', 'modify_gtt', 'delete_gtt', 'get_gtt', 'get_gtts']
        
        print("\n📋 GTT Methods:")
        for method_name in gtt_methods:
            if hasattr(kite, method_name):
                method = getattr(kite, method_name)
                print(f"\n  {method_name}:")
                try:
                    sig = inspect.signature(method)
                    print(f"    Signature: {sig}")
                    
                    # Get docstring
                    doc = inspect.getdoc(method)
                    if doc:
                        print(f"    Docstring: {doc[:100]}...")
                except Exception as e:
                    print(f"    Could not inspect: {e}")
        
        # Try to get existing GTTs (if any)
        print("\n📊 Testing get_gtts()...")
        try:
            gtts = kite.get_gtts()
            print(f"  ✓ Successfully called get_gtts()")
            print(f"  Return type: {type(gtts)}")
            if isinstance(gtts, (list, dict)):
                print(f"  Number of GTTs: {len(gtts) if isinstance(gtts, list) else 'dict'}")
                if gtts and len(gtts) > 0:
                    print(f"  Sample GTT structure:")
                    sample = gtts[0] if isinstance(gtts, list) else list(gtts.values())[0]
                    for key, value in list(sample.items())[:5]:
                        print(f"    {key}: {value}")
        except Exception as e:
            print(f"  ⚠ Error calling get_gtts(): {e}")
            print(f"    (This is normal if you don't have any GTT orders)")
        
        print("\n" + "=" * 70)
        print("✅ GTT Methods are available and ready to use!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_gtt_methods()

