"""Commodity watch should stay armed outside session (weekend / pre-open)."""
from unittest.mock import patch

from services.commodity_strategy_watch import CommodityStrategyWatch


def test_arm_stays_true_when_trading_not_yet_allowed():
    watch = CommodityStrategyWatch()
    with patch(
        "services.commodity_strategy_watch.watch_autonomous_globally_disabled",
        return_value=False,
    ), patch.object(watch, "_kill_switch_active", return_value=False), patch.object(
        watch, "_ensure_loop_task"
    ):
        out = watch.arm(mode="autonomous", auto_place_on_signal=True)
    assert out["armed"] is True
    assert watch._armed is True
