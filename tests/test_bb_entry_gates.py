"""Entry gates that block counter-trend BB / weak ORB setups."""
from services.commodity_entry_pricing import compute_strategy_entry
from services.kite_live_indicators import bb_auto_option_kind, bb_mean_reversion_index_gate
from services.v2_entry_pricing import compute_strategy_entry as nifty_entry


def test_bb_index_gate_blocks_ce_at_upper_band():
    msg = bb_mean_reversion_index_gate(
        "CE",
        spot=24002.0,
        lower=23925.0,
        middle=23962.0,
        upper=23998.0,
    )
    assert msg is not None
    assert "CE blocked" in msg


def test_bb_index_gate_allows_pe_at_upper_band():
    assert (
        bb_mean_reversion_index_gate(
            "PE",
            spot=24002.0,
            lower=23925.0,
            middle=23962.0,
            upper=23998.0,
        )
        is None
    )


def test_bb_auto_kind_prefers_pe_at_upper_index():
    kind = bb_auto_option_kind(
        24002.0,
        23925.0,
        23962.0,
        23998.0,
        prev_close=23856.0,
    )
    assert kind == "PE"


def test_nifty_bb_blocks_ce_on_bullish_day_at_contract_middle():
    intra = {
        "nifty_spot": 23964.0,
        "prev_close": 23856.0,
        "index_bb_lower": 23934.0,
        "index_bb_middle": 23968.0,
        "index_bb_upper": 24001.0,
        "bb_lower": 85.0,
        "bb_middle": 92.0,
        "bb_upper": 100.0,
        "contract_ltp": 92.0,
        "bb_on_contract": "NIFTY2661623950CE",
        "indicator_sources": {"bb_middle": "kite_historical_5m_option"},
    }
    result = nifty_entry(
        strategy_id="bb_5m_mean_reversion",
        option_kind="CE",
        quote={"ltp": 92.0, "bid": 91.5, "ask": 92.5},
        spot=23964.0,
        strike=23950,
        delta=0.45,
        intra=intra,
        prev_close=23856.0,
    )
    assert result.entry_ready is False
    assert "prior close" in (result.block_reason or "").lower()


def test_nifty_bb_blocks_ce_when_index_extended():
    intra = {
        "nifty_spot": 23970.0,
        "prev_close": 23856.0,
        "index_bb_lower": 23930.0,
        "index_bb_middle": 23965.0,
        "index_bb_upper": 24000.0,
        "bb_lower": 90.0,
        "bb_middle": 100.0,
        "bb_upper": 110.0,
        "contract_ltp": 100.0,
        "bb_on_contract": "NIFTY2661623950CE",
        "indicator_sources": {"bb_middle": "kite_historical_5m_option"},
    }
    result = nifty_entry(
        strategy_id="bb_5m_mean_reversion",
        option_kind="CE",
        quote={"ltp": 100.0, "bid": 99.5, "ask": 100.5},
        spot=23970.0,
        strike=23950,
        delta=0.45,
        intra=intra,
        prev_close=23856.0,
    )
    assert result.entry_ready is False
    assert result.block_reason is not None
    assert "CE blocked" in result.block_reason


def test_commodity_orb_blocks_pe_without_confirmed_break():
    intra = {
        "or_high": 7620.0,
        "or_low": 7580.0,
        "day_open": 7610.0,
        "session_minutes": 10 * 60 + 30,
        "last_5m_close": 7595.0,
    }
    result = compute_strategy_entry(
        strategy_id="orb_15m_breakout",
        option_kind="PE",
        quote={"ltp": 140.0, "bid": 139.5, "ask": 140.5},
        spot=7590.0,
        strike=7600,
        delta=0.4,
        intra=intra,
        prev_close=7610.0,
    )
    assert result.entry_ready is False
    assert result.block_reason is not None


def test_commodity_orb_blocks_pe_on_bullish_day_above_open():
    intra = {
        "or_high": 7630.0,
        "or_low": 7595.0,
        "day_open": 7580.0,
        "session_minutes": 10 * 60 + 30,
        "last_5m_close": 7587.0,
    }
    result = compute_strategy_entry(
        strategy_id="orb_15m_breakout",
        option_kind="PE",
        quote={"ltp": 140.0, "bid": 139.5, "ask": 140.5},
        spot=7588.0,
        strike=7600,
        delta=0.4,
        intra=intra,
        prev_close=7580.0,
    )
    assert result.entry_ready is False
    assert "day open" in (result.block_reason or "").lower()
