# GTT Test Scripts - Quick Guide

## Test Scripts Created

1. **`test_gtt_itc_signal.py`** - Generates a BUY signal for ITC and creates an approval
2. **`view_gtt_orders.py`** - View all GTT orders in your account
3. **`view_gtt_details.py`** - View details of a specific GTT order

## How to Test GTT Orders

### Step 1: Enable GTT in Configuration

Before running the test, enable GTT orders:

**Option A: Via .env file**
```env
USE_GTT_ORDERS=true
GTT_FOR_INTRADAY=true
GTT_FOR_POSITIONAL=true
```

**Option B: Via API**
```bash
curl -X POST http://localhost:8000/agent/config \
  -H "Content-Type: application/json" \
  -d '{
    "use_gtt_orders": true,
    "gtt_for_intraday": true,
    "gtt_for_positional": true
  }'
```

### Step 2: Generate Test Signal

Run the test script to create an ITC BUY signal:

```bash
cd tradehandler-api
algo-env/bin/python3.12 test_gtt_itc_signal.py
```

This will:
- Get current ITC price
- Calculate stop-loss and target
- Create an approval
- Display approval details

**Output Example:**
```
======================================================================
ITC BUY Signal Test with GTT Order
======================================================================

1. Getting current price for ITC...
   ✓ Current Price: ₹405.50

2. Calculating stop-loss and target...
   ✓ Entry Price: ₹405.50
   ✓ Stop Loss: ₹403.27
   ✓ Target: ₹411.96
   ✓ Risk: ₹2.23
   ✓ Reward: ₹6.46
   ✓ R:R Ratio: 2.90:1

3. Calculating position size...
   ✓ Suggested Quantity: 24 shares
   ✓ Trade Value: ₹9,732.00

4. Creating approval...
   ✓ Approval Created: abc123-def456-...
   ✓ Approval Status: PENDING
```

### Step 3: Approve the Trade

1. Go to the **Approvals UI** in your frontend
2. Find the ITC BUY approval
3. Click **"Approve & Place Order"**

The system will automatically:
- Place entry order (MARKET)
- Place GTT OCO order (Stop-loss + Target) if GTT is enabled

### Step 4: View GTT Orders

After approval, view your GTT orders:

```bash
# View all GTT orders
algo-env/bin/python3.12 view_gtt_orders.py

# View specific GTT details
algo-env/bin/python3.12 view_gtt_details.py <trigger_id>
```

**Output Example:**
```
======================================================================
GTT Orders in Account
======================================================================

Total GTT Orders: 1

======================================================================
GTT #1
======================================================================
Trigger ID: 12345
Status: active
Symbol: ITC (NSE)
Type: two-leg
Created: 2025-12-26T13:30:00+05:30
Updated: 2025-12-26T13:30:00+05:30
Expires: 2026-12-26T13:30:00+05:30

Trigger Conditions (OCO):
  - Stop Loss Trigger: ₹403.50
  - Target Trigger: ₹411.80

Orders (2):
  Order 1:
    - Transaction: SELL
    - Quantity: 24
    - Price: ₹403.27
    - Product: MIS
    - Order Type: LIMIT
  Order 2:
    - Transaction: SELL
    - Quantity: 24
    - Price: ₹411.96
    - Product: MIS
    - Order Type: LIMIT
```

## GTT Order Behavior

### OCO (One Cancels Other)
- When **stop-loss trigger** is hit → Stop-loss order is placed, target is canceled
- When **target trigger** is hit → Target order is placed, stop-loss is canceled
- Only one order executes, the other is automatically canceled

### Validity
- GTT orders are valid for **up to 1 year**
- Automatically expire if not triggered within 1 year

### Status
- `active` - Waiting for trigger
- `triggered` - Trigger condition met, order placed
- `cancelled` - Manually cancelled
- `expired` - Expired after 1 year

## Troubleshooting

### GTT Not Placed
- Check if `use_gtt_orders` is enabled in config
- Check if `gtt_for_intraday` or `gtt_for_positional` matches your product type
- Check logs for error messages

### GTT Not Visible
- Wait a few seconds after approval (GTT placement is async)
- Check if GTT was successfully placed (check logs)
- Verify trigger_id was stored in approval

### GTT Status Issues
- `disabled` - GTT was disabled (check Zerodha account)
- `rejected` - GTT was rejected (check error in logs)
- `expired` - GTT expired (create new one)

## Notes

- GTT orders are **separate from regular orders**
- GTT trigger ID is stored in approval's `sl_order_id` field
- GTT orders persist even if the server restarts
- Maximum 250 active GTTs per account

