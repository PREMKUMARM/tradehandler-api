"""Tests for trail improvements: breakeven buffer, partial qty, activation hold."""
from services.momentum_trail import breakeven_stop, get_momentum_trail_config
from services.trail_ops import (
    activation_ready,
    check_time_stop,
    get_exit_policy_summary,
    partial_exit_qty,
)


def test_breakeven_uses_pct_and_r_fraction():
    cfg = get_momentum_trail_config()
    entry = 400.0
    R = 20.0
    sl = breakeven_stop(entry, R, cfg)
    assert sl > entry
    assert sl >= entry + entry * cfg.breakeven_pct * 0.99
    assert sl >= entry + R * cfg.breakeven_r_fraction * 0.99


def test_partial_exit_qty_respects_single_lot():
    cfg = get_momentum_trail_config()
    assert partial_exit_qty(1, cfg) == 0
    assert partial_exit_qty(65, cfg, lot_size=65) == 0
    assert partial_exit_qty(130, cfg, lot_size=65) == 65
    assert partial_exit_qty(10, cfg) >= 1
    assert partial_exit_qty(10, cfg) <= 9


def test_activation_hold_blocks_until_elapsed():
    cfg = get_momentum_trail_config()
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")
    now = datetime.now(IST)
    trail = {"trail_active": 0, "target_touch_since": (now - timedelta(seconds=5)).isoformat()}
    ready, _ = activation_ready(
        trail,
        ltp=110.0,
        activation_target=110.0,
        now=now,
        cfg=cfg,
    )
    assert ready is False


def test_time_stop_before_1r():
    cfg = get_momentum_trail_config()
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")
    old = datetime.now(IST) - timedelta(minutes=cfg.time_stop_minutes + 5)
    trail = {"created_at": old.isoformat(), "trail_active": 0}
    reason = check_time_stop(trail, now=datetime.now(IST), trail_active=False, cfg=cfg)
    assert reason is not None


def test_exit_policy_summary_has_trail_lines():
    policy = get_exit_policy_summary("orb_15m_breakout", quantity=10)
    assert policy.get("entry_rr") == 1.0
    assert policy.get("regime") == "trend"
    assert len(policy.get("summary_lines") or []) >= 2


def test_exit_policy_summary_single_lot():
    policy = get_exit_policy_summary("orb_15m_breakout", quantity=1)
    assert "single lot" in policy["summary_lines"][1].lower()
