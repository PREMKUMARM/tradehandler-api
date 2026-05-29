"""Tests for commodity daily trading cutoff."""
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from services.commodity_config import (
    commodity_trading_cutoff_minutes,
    is_commodity_new_trading_allowed,
    is_past_commodity_trading_cutoff,
)

IST = ZoneInfo("Asia/Kolkata")


def test_default_cutoff_is_2315():
    assert commodity_trading_cutoff_minutes() == 23 * 60 + 15


@patch("services.commodity_config.datetime")
def test_trading_allowed_before_cutoff(mock_dt):
    mock_dt.now.return_value = datetime(2026, 5, 29, 22, 0, tzinfo=IST)
    assert is_commodity_new_trading_allowed() is True
    assert is_past_commodity_trading_cutoff() is False


@patch("services.commodity_config.datetime")
def test_trading_blocked_at_cutoff(mock_dt):
    mock_dt.now.return_value = datetime(2026, 5, 29, 23, 15, tzinfo=IST)
    assert is_commodity_new_trading_allowed() is False
    assert is_past_commodity_trading_cutoff() is True


@patch("services.commodity_config.datetime")
def test_trading_blocked_after_cutoff(mock_dt):
    mock_dt.now.return_value = datetime(2026, 5, 29, 23, 20, tzinfo=IST)
    assert is_commodity_new_trading_allowed() is False
    assert is_past_commodity_trading_cutoff() is True
