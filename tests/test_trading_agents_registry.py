"""Tests for trading agent registry."""
from services.trading_agents import get_segment, segment_registry_payload


def test_registry_lists_three_segments():
    payload = segment_registry_payload()
    ids = {s["id"] for s in payload["segments"]}
    assert ids == {"nifty50", "commodity", "crypto"}


def test_nifty_has_invalidation_agent():
    seg = get_segment("nifty50")
    assert seg is not None
    assert "InvalidationAgent" in seg.agents
    assert seg.supports_gtt is True


def test_commodity_has_eod_flatten():
    seg = get_segment("commodity")
    assert seg is not None
    assert seg.supports_eod_flatten is True


def test_crypto_skips_gtt_agents():
    seg = get_segment("crypto")
    assert seg is not None
    assert "GttAgent" not in seg.agents
