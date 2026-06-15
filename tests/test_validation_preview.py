"""Preview validation when market is closed."""
from services.validation_preview import soften_validation_for_closed_market


def test_softens_failed_validation_when_market_closed():
    raw = {
        "is_good_trade": False,
        "failure_reasons": ["Max loss too high"],
        "summary": "bad",
    }
    out = soften_validation_for_closed_market(raw, market_open=False)
    assert out["preview_only"] is True
    assert "Preview only" in out["summary"]
    assert out["is_good_trade"] is False


def test_unchanged_when_market_open():
    raw = {"is_good_trade": False, "failure_reasons": ["x"]}
    out = soften_validation_for_closed_market(raw, market_open=True)
    assert out is raw


def test_softens_even_when_offhours_test_enabled():
    raw = {"is_good_trade": False, "failure_reasons": ["x"]}
    out = soften_validation_for_closed_market(raw, market_open=False, allow_test_place=True)
    assert out["preview_only"] is True
