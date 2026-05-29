"""Option-contract 5m Bollinger from Kite historical closes."""
from services.kite_live_indicators import compute_bollinger_bands, get_option_bollinger_snapshot


def test_compute_bollinger_on_option_premiums():
    closes = [100 + i * 0.5 for i in range(25)]
    mid, upper, lower = compute_bollinger_bands(closes)
    assert mid is not None and upper is not None and lower is not None
    assert lower < mid < upper


def test_option_snapshot_missing_symbol():
    out = get_option_bollinger_snapshot("")
    assert out.get("error") == "missing_symbol"
