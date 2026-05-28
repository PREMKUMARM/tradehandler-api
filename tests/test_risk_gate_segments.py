"""Kill switch segment and fail-closed behavior."""
import json
from pathlib import Path

import services.risk_gate as rg


def test_segment_kill_switch_nifty_only(tmp_path, monkeypatch):
    ks_file = tmp_path / "kill_switch.json"
    ks_file.write_text(json.dumps({"nifty": True, "commodity": False}), encoding="utf-8")
    monkeypatch.setattr(rg, "KILL_SWITCH_PATH", Path(ks_file))
    monkeypatch.delenv("EXECUTION_KILL_SWITCH", raising=False)

    assert rg.is_kill_switch_active("nifty") is True
    assert rg.is_kill_switch_active("commodity") is False
    assert rg.is_kill_switch_active() is True


def test_kill_switch_read_error_fail_closed(tmp_path, monkeypatch):
    ks_file = tmp_path / "kill_switch.json"
    ks_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(rg, "KILL_SWITCH_PATH", Path(ks_file))
    monkeypatch.delenv("EXECUTION_KILL_SWITCH", raising=False)

    assert rg.is_kill_switch_active("commodity") is True
