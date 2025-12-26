"""
Test script to check if GTT (Good Till Triggered) methods exist in KiteConnect API
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.kite_utils import get_kite_instance
from kiteconnect import KiteConnect

def test_gtt_methods():
    """Test if GTT methods are available in KiteConnect"""
    try:
        print("=" * 60)
        print("Testing KiteConnect for GTT Methods")
        print("=" * 60)
        
        # Get KiteConnect instance
        print("\n1. Getting KiteConnect instance...")
        kite = get_kite_instance()
        print("   ✓ KiteConnect instance created successfully")
        
        # Get all methods/attributes of the KiteConnect object
        print("\n2. Inspecting KiteConnect object methods...")
        all_methods = [method for method in dir(kite) if not method.startswith('_')]
        
        # Filter for GTT-related methods
        gtt_methods = [method for method in all_methods if 'gtt' in method.lower()]
        
        print(f"\n   Total methods/attributes: {len(all_methods)}")
        print(f"   GTT-related methods found: {len(gtt_methods)}")
        
        if gtt_methods:
            print("\n   ✓ GTT Methods Found:")
            for method in gtt_methods:
                print(f"      - {method}")
        else:
            print("\n   ✗ No GTT methods found")
        
        # Check for common GTT method names
        print("\n3. Checking for common GTT method patterns...")
        common_gtt_patterns = [
            'gtt_place',
            'gtt_modify',
            'gtt_delete',
            'gtt_list',
            'gtt_get',
            'place_gtt',
            'modify_gtt',
            'delete_gtt',
            'list_gtt',
            'get_gtt'
        ]
        
        found_patterns = []
        for pattern in common_gtt_patterns:
            if hasattr(kite, pattern):
                found_patterns.append(pattern)
                print(f"   ✓ Found: {pattern}")
        
        if not found_patterns:
            print("   ✗ No common GTT patterns found")
        
        # Try to inspect the KiteConnect class source if available
        print("\n4. Inspecting KiteConnect class...")
        try:
            import inspect
            source = inspect.getsource(KiteConnect)
            if 'gtt' in source.lower():
                print("   ✓ 'gtt' found in KiteConnect source code")
                # Find lines with gtt
                lines = source.split('\n')
                gtt_lines = [line.strip() for line in lines if 'gtt' in line.lower() and not line.strip().startswith('#')]
                if gtt_lines:
                    print(f"   Found {len(gtt_lines)} lines with 'gtt':")
                    for line in gtt_lines[:5]:  # Show first 5
                        print(f"      {line[:80]}...")
            else:
                print("   ✗ 'gtt' not found in KiteConnect source code")
        except Exception as e:
            print(f"   ⚠ Could not inspect source: {e}")
        
        # Try to call a GTT method if it exists (with error handling)
        print("\n5. Testing GTT method calls...")
        if gtt_methods:
            for method_name in gtt_methods[:2]:  # Test first 2 methods
                try:
                    method = getattr(kite, method_name)
                    print(f"   Testing {method_name}...")
                    # Just check if it's callable, don't actually call it
                    if callable(method):
                        print(f"      ✓ {method_name} is callable")
                        # Try to get signature
                        try:
                            import inspect
                            sig = inspect.signature(method)
                            print(f"      Signature: {sig}")
                        except:
                            print(f"      Could not get signature")
                    else:
                        print(f"      ⚠ {method_name} is not callable (might be a property)")
                except Exception as e:
                    print(f"      ✗ Error testing {method_name}: {e}")
        else:
            print("   ⚠ No GTT methods to test")
        
        # Check KiteConnect version
        print("\n6. Checking KiteConnect version...")
        try:
            import kiteconnect
            print(f"   KiteConnect version: {kiteconnect.__version__}")
        except:
            print("   ⚠ Could not determine version")
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if gtt_methods or found_patterns:
            print("✓ GTT methods appear to be available in KiteConnect")
            print("  You can use these methods for Good Till Triggered orders")
        else:
            print("✗ GTT methods do not appear to be available in KiteConnect")
            print("  GTT orders may only be available through Zerodha web/mobile interface")
        print("=" * 60)
        
        return {
            "gtt_methods_found": len(gtt_methods) > 0,
            "gtt_methods": gtt_methods,
            "found_patterns": found_patterns
        }
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "gtt_methods_found": False
        }

if __name__ == "__main__":
    result = test_gtt_methods()
    print(f"\nTest result: {result}")

