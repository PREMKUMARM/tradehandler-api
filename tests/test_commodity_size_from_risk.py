"""Commodity Kite qty sizing from entry, SL, capital, and R:R policy."""
from services.commodity_indicator_plan import size_from_risk


def test_auto_qty_from_risk_and_capital():
    # ₹2L, 1% risk, entry 420, SL 409.8 → ₹10.2/bbl × 10 = ₹102 risk/qty
    # max risk ₹2000 → ~19 qty; capital 420×10×19=79800 < 200k → 19
    qty, order_qty, risk_inr, sizing = size_from_risk(
        capital=100_000,
        risk_pct=1.0,
        reward_pct=2.0,
        entry_premium=414.0,
        sl_premium=403.65,
        target_premium=427.60,
        lot_size=10,
        max_qty_cap=1,
    )
    assert qty == order_qty
    assert qty == 9, sizing
    assert risk_inr <= 100_000 * 0.01 + 1
    assert sizing["max_qty_from_capital"] == 24


def test_max_qty_cap_when_user_sets_ceiling():
    qty, _, _, _ = size_from_risk(
        capital=500_000,
        risk_pct=1.0,
        reward_pct=2.0,
        entry_premium=400.0,
        sl_premium=390.0,
        target_premium=420.0,
        lot_size=10,
        max_qty_cap=5,
    )
    assert qty <= 5


def test_entry_cost_mcx_multiplier():
    from services.paper_funds import entry_cost_from_payload

    cost = entry_cost_from_payload(
        {
            "exchange": "MCX",
            "tradingsymbol": "CRUDEOILM26JUN8300PE",
            "quantity": 3,
            "lot_size": 10,
            "price": 420.0,
        },
        fill_price=420.0,
    )
    assert cost == 420.0 * 3 * 10
