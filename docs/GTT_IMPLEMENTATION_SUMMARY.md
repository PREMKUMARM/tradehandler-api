# GTT (Good Till Triggered) Methods - Implementation Summary

## ✅ Test Results

**Status: GTT methods ARE available in KiteConnect API**

The test confirmed that KiteConnect Python library includes full GTT support with the following methods and constants.

## 📋 Available GTT Methods

### Core Methods

1. **`place_gtt()`** - Place a new GTT order
   - Signature: `(trigger_type, tradingsymbol, exchange, trigger_values, last_price, orders)`
   - Creates a new GTT order

2. **`modify_gtt()`** - Modify an existing GTT order
   - Signature: `(trigger_id, trigger_type, tradingsymbol, exchange, trigger_values, last_price, orders)`
   - Updates an existing GTT order

3. **`delete_gtt()`** - Delete a GTT order
   - Signature: `(trigger_id)`
   - Removes a GTT order

4. **`get_gtt()`** - Get details of a specific GTT
   - Signature: `(trigger_id)`
   - Fetches details of a single GTT order

5. **`get_gtts()`** - List all GTT orders
   - Signature: `()`
   - Returns list of all GTT orders in the account

### GTT Constants

#### GTT Types
- `GTT_TYPE_SINGLE = "single"` - Single trigger GTT
- `GTT_TYPE_OCO = "two-leg"` - One Cancels Other (OCO) GTT

#### GTT Statuses
- `GTT_STATUS_ACTIVE = "active"` - GTT is active and waiting
- `GTT_STATUS_TRIGGERED = "triggered"` - GTT has been triggered
- `GTT_STATUS_CANCELLED = "cancelled"` - GTT was cancelled
- `GTT_STATUS_DISABLED = "disabled"` - GTT is disabled
- `GTT_STATUS_EXPIRED = "expired"` - GTT expired (after 1 year)
- `GTT_STATUS_REJECTED = "rejected"` - GTT was rejected
- `GTT_STATUS_DELETED = "deleted"` - GTT was deleted

## 💡 Usage Examples

### Example 1: Place Single Trigger GTT (Stop Loss)

```python
from utils.kite_utils import get_kite_instance

kite = get_kite_instance()

# Place a stop-loss GTT for RELIANCE
trigger_id = kite.place_gtt(
    trigger_type=kite.GTT_TYPE_SINGLE,
    tradingsymbol="RELIANCE",
    exchange="NSE",
    trigger_values={
        "ltp": 1550.0,  # Trigger when LTP reaches this price
    },
    last_price=1580.0,  # Current market price
    orders=[{
        "transaction_type": "SELL",
        "quantity": 10,
        "price": 1545.0,  # Limit price when triggered
        "product": "MIS",
        "order_type": "LIMIT"
    }]
)
```

### Example 2: Place OCO GTT (Stop Loss + Target)

```python
# Place OCO GTT with both stop-loss and target
trigger_id = kite.place_gtt(
    trigger_type=kite.GTT_TYPE_OCO,
    tradingsymbol="RELIANCE",
    exchange="NSE",
    trigger_values={
        "ltp": [
            {"ltp": 1540.0},  # Stop-loss trigger
            {"ltp": 1620.0}   # Target trigger
        ]
    },
    last_price=1580.0,
    orders=[
        {
            "transaction_type": "SELL",
            "quantity": 10,
            "price": 1535.0,  # Stop-loss limit price
            "product": "MIS",
            "order_type": "LIMIT"
        },
        {
            "transaction_type": "SELL",
            "quantity": 10,
            "price": 1615.0,  # Target limit price
            "product": "MIS",
            "order_type": "LIMIT"
        }
    ]
)
```

### Example 3: List All GTT Orders

```python
gtts = kite.get_gtts()
for gtt in gtts:
    print(f"GTT ID: {gtt['id']}")
    print(f"Status: {gtt['status']}")
    print(f"Symbol: {gtt['tradingsymbol']}")
    print(f"Type: {gtt['condition']['type']}")
```

### Example 4: Modify GTT Order

```python
kite.modify_gtt(
    trigger_id=12345,
    trigger_type=kite.GTT_TYPE_SINGLE,
    tradingsymbol="RELIANCE",
    exchange="NSE",
    trigger_values={
        "ltp": 1555.0,  # Updated trigger price
    },
    last_price=1580.0,
    orders=[{
        "transaction_type": "SELL",
        "quantity": 10,
        "price": 1550.0,  # Updated limit price
        "product": "MIS",
        "order_type": "LIMIT"
    }]
)
```

### Example 5: Delete GTT Order

```python
kite.delete_gtt(trigger_id=12345)
```

## 🎯 Benefits for Your Trading Agent

### Current Implementation (SL-M + LIMIT Orders)
- ✅ Works for intraday trading
- ✅ Orders expire at end of day
- ✅ Requires monitoring and auto-cancellation logic

### With GTT Implementation
- ✅ **Long-term stop-losses**: GTT orders valid for up to 1 year
- ✅ **OCO orders**: Automatic stop-loss + target in one order
- ✅ **No monitoring needed**: GTT handles trigger automatically
- ✅ **Reduced complexity**: No need for auto-cancellation background task
- ✅ **Better for swing/positional trades**: Orders persist across days

## 🔄 Migration Strategy

### Option 1: Hybrid Approach (Recommended)
- Use **GTT for positional/swing trades** (CNC product)
- Use **SL-M + LIMIT for intraday trades** (MIS product)
- Keep current implementation for intraday
- Add GTT support for longer-term positions

### Option 2: Full GTT Migration
- Replace all SL/TP logic with GTT orders
- Use OCO GTT for automatic stop-loss + target
- Remove auto-cancellation background task
- Simplify order management

## 📝 Implementation Notes

### GTT Limitations
- Maximum 250 active GTTs per account
- Valid for up to 1 year
- Canceled during corporate actions
- Requires sufficient funds/holdings when triggered

### GTT vs Regular Orders
- **GTT**: Persists until triggered (up to 1 year)
- **SL-M/LIMIT**: Expires at end of day (intraday)
- **GTT OCO**: Automatic cancellation of opposite order
- **Regular orders**: Require manual cancellation

## ✅ Implementation Complete

### What Was Implemented

1. **GTT Tools Created** (`agent/tools/kite_tools.py`):
   - ✅ `place_gtt_tool()` - Place single or OCO GTT orders
   - ✅ `modify_gtt_tool()` - Modify existing GTT orders
   - ✅ `delete_gtt_tool()` - Delete GTT orders
   - ✅ `get_gtt_tool()` - Get GTT details
   - ✅ `get_gtts_tool()` - List all GTT orders

2. **Configuration Added** (`agent/config.py`):
   - ✅ `use_gtt_orders` - Enable/disable GTT usage (default: False)
   - ✅ `gtt_for_intraday` - Use GTT for intraday trades (MIS) (default: False)
   - ✅ `gtt_for_positional` - Use GTT for positional trades (CNC) (default: True)

3. **Approval Handler Updated** (`main.py`):
   - ✅ Automatic GTT OCO order placement when enabled
   - ✅ Falls back to regular SL-M + LIMIT orders if GTT fails
   - ✅ Supports both intraday (MIS) and positional (CNC) trades
   - ✅ Stores GTT trigger ID in approval details

4. **Tools Exported** (`agent/tools/__init__.py`):
   - ✅ All GTT tools exported and available to agent

5. **Configuration API Updated** (`main.py`):
   - ✅ GET `/agent/config` returns GTT settings
   - ✅ POST `/agent/config` accepts GTT settings
   - ✅ GTT settings persisted to `.env` file

### How It Works

When a trade is approved:
1. Entry order is placed (MARKET order)
2. If `use_gtt_orders` is enabled:
   - For **positional trades (CNC)**: Uses GTT if `gtt_for_positional = True`
   - For **intraday trades (MIS)**: Uses GTT if `gtt_for_intraday = True`
3. If GTT is used:
   - Places **OCO GTT order** with both stop-loss and target
   - One trigger cancels the other automatically
   - Valid for up to 1 year
4. If GTT not used or fails:
   - Falls back to regular **SL-M + LIMIT** orders
   - Requires auto-cancellation background task

### Configuration

Add to `.env` file:
```env
USE_GTT_ORDERS=true
GTT_FOR_INTRADAY=false
GTT_FOR_POSITIONAL=true
```

Or update via API:
```json
{
  "use_gtt_orders": true,
  "gtt_for_intraday": false,
  "gtt_for_positional": true
}
```

### Benefits

- ✅ **OCO Orders**: Automatic stop-loss + target in one order
- ✅ **No Monitoring**: GTT handles triggers automatically
- ✅ **Long-term**: Valid up to 1 year (vs end-of-day for regular orders)
- ✅ **Simplified**: No need for auto-cancellation background task
- ✅ **Flexible**: Can use GTT for positional, regular orders for intraday

## 📚 References

- KiteConnect Python Library: https://kite.trade/docs/connect/v3/
- Zerodha GTT Documentation: https://support.zerodha.com/category/trading-and-markets/gtt

---

**Implementation Date**: 2025-12-26  
**KiteConnect Version**: Available (tested successfully)  
**Status**: ✅ **FULLY IMPLEMENTED AND READY TO USE**

