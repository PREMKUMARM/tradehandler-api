"""Tests for pending entry invalidation while LIMIT is open."""
from services.watch_pending_invalidation import pending_entry_invalidated


def _pending_pe_orb():
    return {
        "tradingsymbol": "CRUDEOILM26JUN8300PE",
        "option_type": "PE",
        "strategy_id": "orb_15m_breakout",
        "spot_stop_loss": 8361.0,
        "entry_spot_trigger": 8421.0,
        "indicators": {"or_low": 8421.0, "or_high": 8479.0},
    }


def _current_valid(spot: float):
    return {
        "tradingsymbol": "CRUDEOILM26JUN8300PE",
        "entry_ready": True,
        "entry_confirmation_score": 85,
        "nifty_spot": spot,
        "indicators": {"or_low": 8421.0, "or_high": 8479.0, "last_5m_close": spot - 10},
    }


def test_pending_still_valid_when_spot_below_or_low():
    invalid, _ = pending_entry_invalidated(
        pending_plan=_pending_pe_orb(),
        pending_symbol="CRUDEOILM26JUN8300PE",
        current_plan=_current_valid(8338.0),
        min_score=65,
    )
    assert invalid is False


def test_pending_invalid_when_entry_ready_false():
    current = _current_valid(8338.0)
    current["entry_ready"] = False
    current["entry_block_reason"] = "Setup must reset after last trade"
    invalid, reason = pending_entry_invalidated(
        pending_plan=_pending_pe_orb(),
        pending_symbol="CRUDEOILM26JUN8300PE",
        current_plan=current,
        min_score=65,
    )
    assert invalid is True
    assert "reset" in reason.lower()


def test_pending_invalid_when_spot_reclaims_or_low():
    pending = _pending_pe_orb()
    pending["spot_stop_loss"] = 8500.0
    invalid, reason = pending_entry_invalidated(
        pending_plan=pending,
        pending_symbol="CRUDEOILM26JUN8300PE",
        current_plan=_current_valid(8425.0),
        min_score=65,
    )
    assert invalid is True
    assert "OR low" in reason


def test_pending_invalid_when_spot_hits_setup_sl():
    invalid, reason = pending_entry_invalidated(
        pending_plan=_pending_pe_orb(),
        pending_symbol="CRUDEOILM26JUN8300PE",
        current_plan=_current_valid(8365.0),
        min_score=65,
    )
    assert invalid is True
    assert "setup SL" in reason


def test_adjacent_strike_change_does_not_cancel_by_default():
    current = _current_valid(8338.0)
    current["tradingsymbol"] = "CRUDEOILM26JUN8250PE"
    invalid, _ = pending_entry_invalidated(
        pending_plan=_pending_pe_orb(),
        pending_symbol="CRUDEOILM26JUN8300PE",
        current_plan=current,
        min_score=65,
    )
    assert invalid is False


def test_adjacent_strike_block_reason_does_not_invalidate_pending():
    """Live preview on 3950PE must not abort open/filled 4000PE."""
    pending = {
        "tradingsymbol": "NIFTY2661624000PE",
        "option_type": "PE",
        "strategy_id": "bb_5m_mean_reversion",
        "stop_loss_premium": 57.0,
        "spot_stop_loss": 56.0,
        "entry_premium": 62.3,
        "indicators": {"nifty_spot": 23970.0, "option_ltp": 62.3},
    }
    current = {
        "tradingsymbol": "NIFTY2661623950PE",
        "entry_ready": False,
        "entry_confirmation_score": 28,
        "entry_block_reason": (
            "Wait for rally to BB middle (56.40) or upper (81.73) — price 34.10 at lower"
        ),
        "indicators": {"nifty_spot": 23970.0},
    }
    invalid, reason = pending_entry_invalidated(
        pending_plan=pending,
        pending_symbol="NIFTY2661624000PE",
        current_plan=current,
        min_score=65,
    )
    assert invalid is False, reason


def test_post_fill_skips_unrelated_live_preview():
    pending = {
        "tradingsymbol": "NIFTY2661624000PE",
        "option_type": "PE",
        "strategy_id": "bb_5m_mean_reversion",
        "stop_loss_premium": 57.0,
        "spot_stop_loss": 56.0,
        "entry_premium": 62.3,
        "indicators": {"nifty_spot": 23970.0, "option_ltp": 62.3},
    }
    live = {
        "tradingsymbol": "NIFTY2661623950PE",
        "entry_ready": False,
        "entry_block_reason": "Wait for rally",
    }
    invalid, _ = pending_entry_invalidated(
        pending_plan=pending,
        pending_symbol="NIFTY2661624000PE",
        current_plan=live,
        min_score=65,
        post_fill=True,
    )
    assert invalid is False


def test_bb_premium_trigger_not_compared_to_nifty_spot():
    """Contract BB trigger (~174 premium) must not invalidate against Nifty spot (~24000)."""
    pending = {
        "tradingsymbol": "NIFTY2662324000PE",
        "option_type": "PE",
        "strategy_id": "bb_5m_mean_reversion",
        "entry_spot_trigger": 174.0,
        "entry_premium": 168.0,
        "indicators": {"nifty_spot": 23990.0, "option_ltp": 168.0},
    }
    current = {
        "tradingsymbol": "NIFTY2662324000PE",
        "entry_ready": True,
        "entry_confirmation_score": 88,
        "nifty_spot": 23990.0,
    }
    invalid, reason = pending_entry_invalidated(
        pending_plan=pending,
        pending_symbol="NIFTY2662324000PE",
        current_plan=current,
        min_score=65,
    )
    assert invalid is False, reason


def test_index_spot_trigger_requires_buffer():
    pending = {
        "tradingsymbol": "NIFTY2662324000PE",
        "option_type": "PE",
        "strategy_id": "orb_15m_breakout",
        "entry_ready": True,
        "entry_confirmation_score": 88,
        "entry_spot_trigger": 24003.0,
        "indicators": {"nifty_spot": 24005.0, "option_ltp": 170.0},
    }
    invalid, _ = pending_entry_invalidated(
        pending_plan=pending,
        pending_symbol="NIFTY2662324000PE",
        current_plan=pending,
        min_score=65,
    )
    assert invalid is False

    pending["indicators"]["nifty_spot"] = 24015.0
    invalid, reason = pending_entry_invalidated(
        pending_plan=pending,
        pending_symbol="NIFTY2662324000PE",
        current_plan=pending,
        min_score=65,
    )
    assert invalid is True
    assert "reclaimed" in reason.lower()


def test_large_strike_drift_cancels_when_enabled(monkeypatch):
    monkeypatch.setenv("COMMODITY_WATCH_CANCEL_ON_SYMBOL_DRIFT", "1")
    monkeypatch.setenv("COMMODITY_WATCH_SYMBOL_DRIFT_MIN_STRIKES", "2")
    current = _current_valid(8338.0)
    current["tradingsymbol"] = "CRUDEOILM26JUN8200PE"
    invalid, reason = pending_entry_invalidated(
        pending_plan=_pending_pe_orb(),
        pending_symbol="CRUDEOILM26JUN8300PE",
        current_plan=current,
        min_score=65,
    )
    assert invalid is True
    assert "8200PE" in reason
