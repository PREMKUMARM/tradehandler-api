"""BB/EMA window helpers — Zerodha uses SMA(20) + 2*stdev on 5m closes."""
from services.kite_live_indicators import (
    BB_PERIOD,
    compute_bollinger_bands,
    _patch_forming_close,
    InstrumentLiveState,
)


def test_bollinger_sma_20():
    closes = [float(100 + i) for i in range(20)]
    mid, upper, lower = compute_bollinger_bands(closes)
    assert mid == sum(closes) / BB_PERIOD
    assert upper > mid > lower


def test_patch_forming_close_uses_ticker():
    st = InstrumentLiveState(token=1)
    st.last_price = 105.0
    out = _patch_forming_close(st, [100.0, 102.0])
    assert out[-1] == 105.0
