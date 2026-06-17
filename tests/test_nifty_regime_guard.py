"""Nifty whipsaw / chop regime gates."""
from services.nifty_regime_guard import (
    nifty_autonomous_regime_allowed,
    nifty_intraday_regime_block,
    opposite_direction_blocked,
)


def test_pe_blocked_after_bounce_from_lower_band():
    """Simulates 17-Jun: low ~23970, spot recovered to 24005+."""
    msg = nifty_intraday_regime_block(
        "PE",
        spot=24005.0,
        prev_close=23989.0,
        day_low=23970.0,
        day_high=24072.0,
        index_bb_lower=23967.0,
        index_bb_middle=24010.0,
        index_bb_upper=24050.0,
        contract_zone="upper",
    )
    assert msg is not None
    assert "bounced" in msg.lower()


def test_pe_allowed_at_low_before_bounce():
    msg = nifty_intraday_regime_block(
        "PE",
        spot=23975.0,
        prev_close=23989.0,
        day_low=23970.0,
        day_high=24020.0,
        index_bb_lower=23967.0,
        index_bb_middle=24010.0,
        index_bb_upper=24050.0,
        contract_zone="upper",
    )
    assert msg is None


def test_chop_blocks_mid_range():
    msg = nifty_intraday_regime_block(
        "PE",
        spot=24020.0,
        prev_close=23989.0,
        day_low=23970.0,
        day_high=24072.0,
        index_bb_lower=23950.0,
        index_bb_middle=24020.0,
        index_bb_upper=24090.0,
        contract_zone="between",
    )
    assert msg is not None
    assert "chop" in msg.lower()


def test_opposite_direction_cooldown():
    msg = opposite_direction_blocked("PE", "CE", seconds_since_last=120.0)
    assert msg is not None
    assert "flip" in msg.lower()


def test_autonomous_lot_cap():
    plan = {
        "strategy_id": "bb_5m_mean_reversion",
        "option_type": "PE",
        "num_lots": 11,
        "lot_size": 65,
        "quantity": 715,
        "entry_ready": True,
        "indicators": {
            "nifty_spot": 23975.0,
            "prev_close": 23989.0,
            "day_low": 23970.0,
            "day_high": 24050.0,
            "index_bb_lower": 23950.0,
            "index_bb_middle": 24000.0,
            "index_bb_upper": 24050.0,
            "bb_lower": 150,
            "bb_middle": 170,
            "bb_upper": 190,
            "option_ltp": 185,
        },
    }
    ok, msg = nifty_autonomous_regime_allowed(plan)
    assert ok is False
    assert "lot cap" in msg.lower()
