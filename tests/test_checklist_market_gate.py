"""Tests for market-closed checklist gating."""
from services.checklist_step_utils import apply_market_closed_gate


def test_apply_market_closed_gate_dict_steps():
    steps = [
        {"index": 0, "server_ok": True, "completed": True, "message": "Pass"},
        {"index": 1, "server_ok": True, "completed": True, "message": "Pass"},
        {"index": 2, "server_ok": True, "completed": True, "message": "Pass"},
    ]
    out = apply_market_closed_gate(
        steps,
        market_open=False,
        allow_offhours=False,
        gated_indices=[2],
        closed_message="Market closed",
    )
    assert out[0]["server_ok"] is True
    assert out[1]["server_ok"] is True
    assert out[2]["server_ok"] is False
    assert out[2]["message"] == "Market closed"


def test_apply_market_closed_gate_skips_when_open():
    steps = [{"index": 2, "server_ok": True, "completed": True, "message": "Pass"}]
    out = apply_market_closed_gate(
        steps,
        market_open=True,
        allow_offhours=False,
        gated_indices=[2],
    )
    assert out[0]["server_ok"] is True
