"""Circuit limit validation for patient LIMIT entries."""
from utils.kite_order_utils import validate_buy_limit_price


def test_stale_limit_below_circuit_blocked():
    quote = {"ltp": 8500.0, "lower_circuit_limit": 8135.0, "upper_circuit_limit": 9000.0}
    limit, ok, msg = validate_buy_limit_price(470.0, quote=quote)
    assert ok is False
    assert "8135" in msg
    assert limit == 470.0


def test_minor_bump_to_circuit_floor():
    quote = {"ltp": 8140.0, "lower_circuit_limit": 8135.0, "upper_circuit_limit": 9000.0}
    limit, ok, msg = validate_buy_limit_price(8130.0, quote=quote)
    assert ok is True
    assert limit == 8135.0


def test_limit_inside_band_unchanged():
    quote = {"ltp": 8200.0, "lower_circuit_limit": 8135.0, "upper_circuit_limit": 9000.0}
    limit, ok, _ = validate_buy_limit_price(8180.0, quote=quote)
    assert ok is True
    assert limit == 8180.0
