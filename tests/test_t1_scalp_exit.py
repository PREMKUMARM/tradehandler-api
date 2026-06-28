"""T1 scalp exit mode."""
from __future__ import annotations

from services.dhan_data_client import OptionSeries
from services.sensex_dhan_backtest import _simulate_from_entry


def _series(highs: list[float], lows: list[float] | None = None) -> OptionSeries:
    lows = lows or highs
    n = len(highs)
    return OptionSeries(
        kind="CE",
        offset="ATM",
        timestamps=list(range(n)),
        open=highs,
        high=highs,
        low=lows,
        close=highs,
        oi=[0.0] * n,
        spot=[0.0] * n,
        strike=[82000.0] * n,
    )


def test_t1_scalp_exits_at_target(monkeypatch):
    monkeypatch.setenv("EXIT_MODEL", "t1_scalp")
    entry, sl, t1 = 20.0, 10.0, 30.0
    series = _series([20.0, t1, 20.0], lows=[20.0, 20.0, 9.0])
    exit_px, reason, _ = _simulate_from_entry(entry, series, 0, sl, t1)
    assert reason == "target_t1"
    assert exit_px == t1


def test_t1_scalp_stops_before_target(monkeypatch):
    monkeypatch.setenv("EXIT_MODEL", "t1_scalp")
    entry, sl, t1 = 20.0, 10.0, 30.0
    series = _series([20.0, 19.0], lows=[20.0, 9.0])
    exit_px, reason, _ = _simulate_from_entry(entry, series, 0, sl, t1)
    assert reason == "stop_loss"
    assert exit_px == sl
