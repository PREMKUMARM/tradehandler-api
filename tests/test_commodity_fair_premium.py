"""Commodity fair premium must not collapse to tick when BB trigger is on premium scale."""
from services.commodity_entry_pricing import _patient_buy_limit, _structure_fair_premium


def test_bb_trigger_does_not_crush_fair_to_tick():
    """Crude spot ~7636, option LTP ~70, BB mid ~76 — fair must stay near LTP not ₹0.05."""
    fair = _structure_fair_premium(
        ltp=70.10,
        spot=7636.0,
        spot_trigger=76.08,
        kind="CE",
        delta=0.5,
    )
    assert fair >= 35.0
    assert fair <= 70.10

    limit, _ = _patient_buy_limit({"bid": 69.0, "ask": 71.0, "ltp": 70.10}, fair)
    assert limit >= 35.0
