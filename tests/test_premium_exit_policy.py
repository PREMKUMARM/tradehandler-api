"""Tests for minimum premium exit spacing."""
from services.premium_exit_policy import enforce_min_premium_exits


def test_tight_8250pe_exits_widened():
    """Reproduce 2026-05-29 live bug: TP ₹392.15 on entry ₹392.1."""
    sl, tp = enforce_min_premium_exits(
        392.1,
        386.55,
        392.15,
        risk_pct=1.0,
        reward_pct=2.0,
        min_rr=1.5,
    )
    assert tp - 392.1 >= 5.0
    assert tp >= 392.1 + (392.1 - sl) * 1.5
    assert sl < 392.1


def test_entry_initial_rr_defaults_to_one_to_one():
    from services.premium_exit_policy import entry_initial_rr, entry_validation_skips_reward

    assert entry_initial_rr() == 1.0
    assert entry_validation_skips_reward() is True
    sl, tp = enforce_min_premium_exits(400.0, 380.0, 405.0, risk_pct=1.0, reward_pct=2.0)
    assert tp >= 400.0 + (400.0 - sl) * entry_initial_rr()


def test_respects_actual_sl_when_wider_than_pct():
    entry = 400.0
    sl, tp = enforce_min_premium_exits(
        entry, 370.0, 405.0, risk_pct=1.0, reward_pct=2.0, min_rr=2.0
    )
    assert sl == 370.0
    assert tp >= entry + (entry - sl) * 2.0 * 0.99
