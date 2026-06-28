"""Two-candle confirmation entry for 20rupees band strategy."""
from __future__ import annotations

from services.entry_quality import (
    band_bar_touched_band,
    band_setup_bar_ok,
    confirmation_candle_entry_ok,
)


def test_setup_bar_band_touch_wick():
    assert band_bar_touched_band(14.0, 24.0, 16.0, 17.0, 23.0) is True
    assert band_bar_touched_band(25.0, 30.0, 26.0, 17.0, 23.0) is False


def test_ce_confirmation_ok(monkeypatch):
    monkeypatch.setenv("ENTRY_REQUIRE_CONFIRMATION_CANDLE", "true")
    ok, why = confirmation_candle_entry_ok(
        kind="CE",
        setup_open=18.0,
        setup_high=20.0,
        setup_low=17.5,
        setup_close=19.0,
        confirm_open=19.5,
        confirm_high=21.5,
        confirm_low=19.0,
        confirm_close=21.0,
        band_low=17.0,
        band_high=23.0,
    )
    assert ok is True
    assert why == ""


def test_ce_confirmation_rejects_no_progress(monkeypatch):
    monkeypatch.setenv("ENTRY_REQUIRE_CONFIRMATION_CANDLE", "true")
    ok, why = confirmation_candle_entry_ok(
        kind="CE",
        setup_open=18.0,
        setup_high=22.0,
        setup_low=17.5,
        setup_close=20.0,
        confirm_open=19.5,
        confirm_high=21.0,
        confirm_low=19.0,
        confirm_close=19.8,
        band_low=17.0,
        band_high=23.0,
    )
    assert ok is False
    assert why == "confirm_no_progress"


def test_pe_confirmation_ok(monkeypatch):
    monkeypatch.setenv("ENTRY_REQUIRE_CONFIRMATION_CANDLE", "true")
    ok, why = confirmation_candle_entry_ok(
        kind="PE",
        setup_open=20.0,
        setup_high=22.0,
        setup_low=18.0,
        setup_close=19.5,
        confirm_open=19.0,
        confirm_high=19.5,
        confirm_low=17.5,
        confirm_close=17.8,
        band_low=17.0,
        band_high=23.0,
    )
    assert ok is True
    assert why == ""
