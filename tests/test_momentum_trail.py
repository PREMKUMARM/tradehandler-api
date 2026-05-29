"""Tests for momentum trailing exit logic."""
from services.momentum_trail import (
    compute_trailed_levels,
    get_momentum_trail_config,
    should_activate_trail,
)


def test_activate_when_target_reached_in_profit():
    assert should_activate_trail(429.0, 418.9, 429.0, trail_active=False) is True
    assert should_activate_trail(410.0, 418.9, 429.0, trail_active=False) is False


def test_first_trail_moves_sl_to_breakeven_and_extends_tp():
    cfg = get_momentum_trail_config()
    sl, tp, peak, active, note = compute_trailed_levels(
        entry=418.9,
        peak=418.9,
        ltp=429.0,
        current_sl=409.35,
        current_tp=429.0,
        trail_active=False,
        cfg=cfg,
    )
    assert active is True
    assert sl >= 418.9
    assert tp > 429.0
    assert peak == 429.0
    assert note


def test_trail_ratchet_on_new_high():
    cfg = get_momentum_trail_config()
    sl1, tp1, peak1, _, _ = compute_trailed_levels(
        entry=418.9,
        peak=429.0,
        ltp=429.0,
        current_sl=418.95,
        current_tp=433.0,
        trail_active=True,
        cfg=cfg,
    )
    sl2, tp2, peak2, _, _ = compute_trailed_levels(
        entry=418.9,
        peak=peak1,
        ltp=435.0,
        current_sl=sl1,
        current_tp=tp1,
        trail_active=True,
        cfg=cfg,
    )
    assert peak2 == 435.0
    assert sl2 >= sl1
    assert tp2 >= tp1
