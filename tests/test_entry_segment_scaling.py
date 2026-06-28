"""Segment-scaled intraday entry thresholds (Nifty vs Sensex)."""
from __future__ import annotations

from services.entry_quality import (
    entry_chase_day_pts,
    entry_intraday_context_ok,
    entry_intraday_thresholds,
    entry_min_day_move_pts,
    entry_pt_scale,
)


def test_nifty_uses_absolute_points():
    assert entry_pt_scale("nifty50", 24000.0) == 1.0
    assert entry_min_day_move_pts(segment="nifty50", index_open=24000.0) == 30.0
    assert entry_chase_day_pts(segment="nifty50", index_open=24000.0) == 250.0


def test_sensex_scales_with_index_open():
    scale = entry_pt_scale("sensex", 75000.0)
    assert 3.0 < scale < 3.2
    thr = entry_intraday_thresholds("sensex", 74827.0)
    assert 90 < thr.min_day_pts < 100
    assert 750 < thr.chase_day_pts < 800
    assert 30 < thr.chase_bar_pts < 35
    assert 75 < thr.chase_min_bar_pts < 85


def test_sensex_mar25_winner_bar_passes_scaled_chase():
    """Mar 25 CE @ 14:45 — +572 pt day is below scaled chase (~779), should pass."""
    ok, reason = entry_intraday_context_ok(
        kind="CE",
        index_open=74827.0,
        spot=75400.0,
        spot_prev=75466.0,
        bar_minutes=14 * 60 + 45,
        scan_start_minutes=14 * 60,
        segment="sensex",
    )
    assert ok, reason


def test_nifty_jun9_pe_bounce_still_blocked():
    ok, reason = entry_intraday_context_ok(
        kind="PE",
        index_open=23235.0,
        spot=23195.0,
        spot_prev=23206.0,
        bar_minutes=14 * 60 + 20,
        scan_start_minutes=14 * 60,
        session_low_so_far=23113.0,
        session_high_so_far=23275.0,
        segment="nifty50",
    )
    assert not ok
    assert reason == "bounce_from_low"
