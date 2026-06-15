"""NSE/BSE session weekday must include Monday (Python weekday Monday=0)."""
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from services import sensex_trade_service, v2_trade_service

IST = ZoneInfo("Asia/Kolkata")


def _monday_open():
    # Monday 2026-06-15 10:00 IST — inside 9:15–15:30
    return datetime(2026, 6, 15, 10, 0, tzinfo=IST)


def _monday_early():
    return datetime(2026, 6, 15, 8, 0, tzinfo=IST)


def _saturday():
    return datetime(2026, 6, 13, 10, 0, tzinfo=IST)


def test_nifty_session_open_monday_morning():
    with patch("services.v2_trade_service.datetime") as mock_dt:
        mock_dt.now.return_value = _monday_open()
        assert v2_trade_service.is_market_session_open() is True


def test_nifty_session_closed_monday_before_open():
    with patch("services.v2_trade_service.datetime") as mock_dt:
        mock_dt.now.return_value = _monday_early()
        assert v2_trade_service.is_market_session_open() is False


def test_nifty_session_closed_saturday():
    with patch("services.v2_trade_service.datetime") as mock_dt:
        mock_dt.now.return_value = _saturday()
        assert v2_trade_service.is_market_session_open() is False


def test_sensex_session_open_monday_morning():
    with patch("services.sensex_trade_service.datetime") as mock_dt:
        mock_dt.now.return_value = _monday_open()
        assert sensex_trade_service.is_market_session_open() is True
