"""Tests for segment balance / Kite connection helpers."""
from unittest.mock import patch

from services.segment_balance import get_kite_balance_payload, is_kite_broker_connected


def test_is_kite_broker_connected_when_margin_ok():
    with patch(
        "services.segment_balance.get_kite_balance_payload",
        return_value={"connected": True, "available_margin": 0.7},
    ):
        assert is_kite_broker_connected() is True


def test_is_kite_broker_connected_when_margin_fails():
    with patch(
        "services.segment_balance.get_kite_balance_payload",
        return_value={"connected": False, "message": "token expired"},
    ):
        assert is_kite_broker_connected() is False
