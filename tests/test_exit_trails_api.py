"""Tests for exit trails API serialization."""
from services.exit_trails_api import list_exit_trails_for_api, serialize_exit_trail


def test_serialize_single_lot_trail():
    row = serialize_exit_trail(
        {
            "id": 1,
            "segment": "commodity",
            "tradingsymbol": "CRUDEOIL24JUNFUT",
            "entry_price": 100.0,
            "stop_loss": 90.0,
            "target": 110.0,
            "initial_target": 110.0,
            "peak_ltp": 105.0,
            "quantity": 1,
            "trail_active": False,
            "partial_exit_done": 0,
        }
    )
    assert "single lot" in row["partial_note"].lower()
    assert row["risk_unit"] == 10.0


def test_list_exit_trails_filter_segment():
    rows = list_exit_trails_for_api(segment="nifty50", limit=5)
    assert isinstance(rows, list)
