"""Tests for Sensex 20rupees entry cutoff (avoid last-minute trading)."""
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from services.sensex_constants import (
    is_past_sensex_entry_cutoff,
    is_sensex_new_entry_allowed,
    sensex_entry_cutoff_minutes,
)

IST = ZoneInfo("Asia/Kolkata")


def test_default_cutoff_is_1500():
    assert sensex_entry_cutoff_minutes() == 15 * 60


@patch("services.sensex_constants.datetime")
def test_new_entry_allowed_before_cutoff(mock_dt):
    mock_dt.now.return_value = datetime(2026, 5, 29, 14, 45, tzinfo=IST)
    assert is_past_sensex_entry_cutoff() is False
    assert is_sensex_new_entry_allowed() is True


@patch("services.sensex_constants.datetime")
def test_new_entry_blocked_at_cutoff(mock_dt):
    mock_dt.now.return_value = datetime(2026, 5, 29, 15, 0, tzinfo=IST)
    assert is_past_sensex_entry_cutoff() is True
    assert is_sensex_new_entry_allowed() is False


@patch("services.sensex_constants.datetime")
def test_new_entry_blocked_near_close(mock_dt):
    mock_dt.now.return_value = datetime(2026, 5, 29, 15, 25, tzinfo=IST)
    assert is_past_sensex_entry_cutoff() is True
    assert is_sensex_new_entry_allowed() is False
