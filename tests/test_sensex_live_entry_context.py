"""Live 20rupees entry gates — intraday context filters."""
from __future__ import annotations

from services.sensex_entry_pricing import _analyze_20rupees


def test_live_blocks_pe_bounce_from_low():
    intra = {
        "day_open": 23235.0,
        "day_low": 23113.0,
        "day_high": 23275.0,
        "session_low_so_far": 23113.0,
        "session_high_so_far": 23275.0,
        "spot": 23195.0,
        "index_last_5m_close": 23206.0,
        "bar_minutes": 14 * 60 + 20,
        "option_kind": "PE",
        "sensex_run_params": {
            "entry_scan_start_ist": "14:00",
            "entry_scan_end_ist": "14:45",
            "entry_band_low": 17.0,
            "entry_band_high": 23.0,
            "sl_inr": 9.0,
        },
    }
    quote = {"bid": 17.0, "ask": 17.5, "ltp": 17.15}
    ready, *_rest = _analyze_20rupees(quote, intra, prev_close=23200.0)
    assert not ready
    assert _rest[-1] == "20rupees_bounce_from_low"
