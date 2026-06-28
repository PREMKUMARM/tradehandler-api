"""Backtest exit sim matches live T1/T2 stepped SL."""
from __future__ import annotations

from services.dhan_data_client import OptionSeries
from services.sensex_dhan_backtest import _simulate_from_entry, _stepped_sl_targets


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


def test_stepped_targets_t1_t2():
    entry, sl = 20.0, 10.0
    r, t1, t2 = _stepped_sl_targets(entry, sl)
    assert r == 10.0
    assert t1 == 30.0
    assert t2 == 40.0


def test_simulate_stops_at_initial_sl(monkeypatch):
    monkeypatch.setenv("EXIT_MODEL", "stepped")
    entry = 20.0
    sl = 10.0
    # bar0 entry, bar1 low hits SL
    series = _series([20.0, 19.0], lows=[20.0, 9.0])
    exit_px, reason, _ = _simulate_from_entry(entry, series, 0, sl, entry + 10)
    assert reason == "stop_loss"
    assert exit_px == sl


def test_simulate_ratchet_to_entry_at_t1(monkeypatch):
    monkeypatch.setenv("EXIT_MODEL", "stepped")
    entry = 20.0
    sl = 10.0
    t1 = entry + 10
    # bar1 hits T1 (low stays above entry), bar2 dips to entry (breakeven SL)
    series = _series([20.0, t1 + 1, entry], lows=[20.0, entry + 0.5, entry - 0.5])
    exit_px, reason, _ = _simulate_from_entry(entry, series, 0, sl, t1)
    assert reason == "trail_t1"
    assert exit_px == entry


def test_simulate_ratchet_to_t1_at_t2(monkeypatch):
    monkeypatch.setenv("EXIT_MODEL", "stepped")
    entry = 20.0
    sl = 10.0
    t1 = 30.0
    t2 = 40.0
    # bar1 T1, bar2 T2 (low above T1), bar3 low at T1
    series = _series([20.0, t1, t2, t1], lows=[20.0, entry + 1, t1 + 1, t1 - 0.5])
    exit_px, reason, _ = _simulate_from_entry(entry, series, 0, sl, t1)
    assert reason == "trail_t2"
    assert exit_px == t1


def test_simulate_ratchet_to_t2_at_t3(monkeypatch):
    monkeypatch.setenv("EXIT_MODEL", "stepped")
    entry = 20.0
    sl = 10.0
    t1 = 30.0
    t2 = 40.0
    t3 = 50.0
    series = _series(
        [20.0, t1, t2, t3, t2],
        lows=[20.0, entry + 1, t1 + 1, t2 + 1, t2 - 0.5],
    )
    exit_px, reason, _ = _simulate_from_entry(entry, series, 0, sl, t1)
    assert reason == "trail_t3"
    assert exit_px == t2


def test_t1_wick_only_does_not_ratchet(monkeypatch):
    """Wick above T1 with close below should not move SL to entry when close-confirm is on."""
    monkeypatch.setenv("EXIT_T1_CLOSE_CONFIRM", "true")
    from services.dhan_data_client import OptionSeries

    entry = 20.0
    sl = 10.0
    t1 = 30.0
    series = OptionSeries(
        kind="CE",
        offset="ATM",
        timestamps=[0, 1, 2],
        open=[20.0, 20.0, 20.0],
        high=[20.0, t1 + 2, 20.0],
        low=[20.0, 19.0, 9.0],
        close=[20.0, 25.0, 9.0],
        oi=[0.0, 0.0, 0.0],
        spot=[0.0, 0.0, 0.0],
        strike=[82000.0] * 3,
    )
    exit_px, reason, _ = _simulate_from_entry(entry, series, 0, sl, t1)
    assert reason == "stop_loss"
    assert exit_px == sl
