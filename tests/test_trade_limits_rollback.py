"""Trade limits rollback on cancelled entry."""
from utils.trade_limits import trade_limits


def test_rollback_trade_decrements_counters():
    trade_limits.limits["trades_today"] = 3
    trade_limits.limits["total_investment_today"] = 5000.0
    trade_limits.rollback_trade(1200.0)
    assert trade_limits.limits["trades_today"] == 2
    assert trade_limits.limits["total_investment_today"] == 3800.0
