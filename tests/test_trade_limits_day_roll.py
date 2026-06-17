"""IST day rollover for long-running API."""
from datetime import date
from unittest.mock import patch

from utils.trade_limits import TradeLimits, _ist_today


def test_roll_day_resets_counters(tmp_path):
    limits_file = tmp_path / "trade_limits.json"
    tl = TradeLimits()
    tl.limits_file = str(limits_file)
    tl.limits = {
        "max_trades_per_day": 10,
        "max_profit_per_day": 0.02,
        "max_loss_per_day": 0.05,
        "max_premium_inr_per_day": 0,
        "max_loss_inr_per_day": 0,
        "current_day": "2026-06-16",
        "trades_today": 10,
        "profit_today": 0.01,
        "loss_today": 0.02,
        "total_investment_today": 16172.0,
        "pnl_inr_today": -500.0,
    }
    tl._save_limits(tl.limits)

    with patch("utils.trade_limits._ist_today", return_value=date(2026, 6, 17)):
        tl._roll_day_if_needed()

    assert tl.limits["current_day"] == "2026-06-17"
    assert tl.limits["trades_today"] == 0
    assert tl.limits["total_investment_today"] == 0.0
    assert tl.limits["pnl_inr_today"] == 0.0
    can, _ = tl.can_place_trade()
    assert can is True


def test_get_limits_status_rolls_before_can_trade(tmp_path):
    limits_file = tmp_path / "trade_limits.json"
    tl = TradeLimits()
    tl.limits_file = str(limits_file)
    tl.limits["current_day"] = "2026-06-16"
    tl.limits["trades_today"] = 10
    tl._save_limits(tl.limits)

    with patch("utils.trade_limits._ist_today", return_value=date(2026, 6, 17)):
        status = tl.get_limits_status()

    assert status["current_day"] == "2026-06-17"
    assert status["trades_today"] == 0
    assert status["can_trade"] is True
