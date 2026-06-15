"""Sensex smart strike selection tests."""
from services.sensex_strike_selection import (
    pick_smart_from_chain,
    resolve_sensex_strike_for_plan,
    score_strike,
)


def _sample_chain(atm: int = 82000) -> dict:
    return {
        "atm": atm,
        "atm_ce": {"ltp": 45.0, "oi": 1000, "spread_pct": 0.5},
        "atm_pe": {"ltp": 42.0, "oi": 900, "spread_pct": 0.6},
        "ranked_ce_oi": [
            {
                "strike": atm + 700,
                "ltp": 19.5,
                "oi": 5000,
                "oi_change": 800,
                "spread_pct": 1.2,
                "symbol": f"SENSEX-{atm + 700}-CE",
            },
            {
                "strike": atm + 900,
                "ltp": 20.0,
                "oi": 8000,
                "oi_change": 200,
                "spread_pct": 3.5,
                "symbol": f"SENSEX-{atm + 900}-CE",
            },
        ],
        "ranked_pe_oi": [
            {
                "strike": atm - 700,
                "ltp": 21.0,
                "oi": 6000,
                "oi_change": 500,
                "spread_pct": 1.0,
                "symbol": f"SENSEX-{atm - 700}-PE",
            },
        ],
    }


def test_prefers_near_atm_in_band_over_deep_otm():
    chain = _sample_chain()
    picked = pick_smart_from_chain(chain, 82050.0, kinds=("CE",), prev_close=82000.0)
    assert picked is not None
    assert picked.strike == 82700
    assert 17 <= picked.ltp <= 23


def test_wide_spread_penalized():
    low_spread = score_strike(
        oi=5000, strike=82700, atm=82000, kind="CE", max_oi=8000, ltp=20.0, spread_pct=1.0
    )
    high_spread = score_strike(
        oi=8000, strike=82900, atm=82000, kind="CE", max_oi=8000, ltp=20.0, spread_pct=3.5
    )
    assert low_spread > high_spread


def test_resolve_uses_smart_when_anchor_out_of_band():
    chain = _sample_chain()
    resolved = resolve_sensex_strike_for_plan(
        spot=82050.0,
        option_kind="CE",
        chain_oi=chain,
        strategy_id="20rupees_strategy",
        anchor_strike=82000,
        band_low=17.0,
        band_high=23.0,
    )
    assert resolved.source == "smart_oi"
    assert resolved.strike == 82700


def test_resolve_keeps_valid_anchor():
    chain = _sample_chain()
    resolved = resolve_sensex_strike_for_plan(
        spot=82050.0,
        option_kind="CE",
        chain_oi=chain,
        strategy_id="20rupees_strategy",
        anchor_strike=82700,
        band_low=17.0,
        band_high=23.0,
    )
    assert resolved.source == "anchor"
    assert resolved.strike == 82700


def test_resolve_pe_smart_pick():
    chain = _sample_chain()
    resolved = resolve_sensex_strike_for_plan(
        spot=81900.0,
        option_kind="PE",
        chain_oi=chain,
        strategy_id="20rupees_strategy",
        band_low=17.0,
        band_high=23.0,
        prev_close=82200.0,
    )
    assert resolved.kind == "PE"
    assert resolved.strike == 81300
