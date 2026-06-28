"""Entry intraday context filters — warmup, weak day, chase."""
from __future__ import annotations

from services.entry_quality import entry_intraday_context_ok


def test_scan_warmup_blocks_first_bar():
    ok, reason = entry_intraday_context_ok(
        kind="CE",
        index_open=100.0,
        spot=101.0,
        spot_prev=100.5,
        bar_minutes=14 * 60,
        scan_start_minutes=14 * 60,
    )
    assert not ok
    assert reason == "scan_warmup"


def test_weak_day_blocks_marginal_pe():
    ok, reason = entry_intraday_context_ok(
        kind="PE",
        index_open=23235.0,
        spot=23206.0,
        spot_prev=23208.0,
        bar_minutes=14 * 60 + 10,
        scan_start_minutes=14 * 60,
    )
    assert not ok
    assert reason == "weak_day_trend"


def test_chase_blocks_extended_ce_without_momentum():
    ok, reason = entry_intraday_context_ok(
        kind="CE",
        index_open=23578.0,
        spot=23866.0,
        spot_prev=23861.0,
        bar_minutes=14 * 60 + 25,
        scan_start_minutes=14 * 60,
    )
    assert not ok
    assert reason == "chase_exhaustion"


def test_bounce_blocks_pe_shorting_recovery():
    ok, reason = entry_intraday_context_ok(
        kind="PE",
        index_open=23235.0,
        spot=23195.0,
        spot_prev=23206.0,
        bar_minutes=14 * 60 + 20,
        scan_start_minutes=14 * 60,
        session_low_so_far=23113.0,
        session_high_so_far=23275.0,
    )
    assert not ok
    assert reason == "bounce_from_low"


def test_capitulation_blocks_extended_pe_panic_bar():
    ok, reason = entry_intraday_context_ok(
        kind="PE",
        index_open=23738.0,
        spot=23438.0,
        spot_prev=23469.0,
        bar_minutes=14 * 60 + 30,
        scan_start_minutes=14 * 60,
        segment="nifty50",
    )
    assert not ok
    assert reason == "capitulation_bar"
