from services.gate_audit import (
    _classify_verdict,
    emit_gate_audit_day_summary,
    record_gate_audit,
    record_gate_audit_placed,
)


def test_classify_allow_autonomous():
    v, _ = _classify_verdict(
        market_open=True,
        entry_ready=True,
        can_execute=True,
        try_autonomous=True,
        block_reason="",
    )
    assert v == "ALLOW_AUTONOMOUS"


def test_classify_blocked_gate():
    v, detail = _classify_verdict(
        market_open=True,
        entry_ready=False,
        can_execute=False,
        try_autonomous=False,
        block_reason="CE blocked: spot above prior close",
    )
    assert v == "BLOCKED_GATE"
    assert "prior close" in detail


def test_record_gate_audit_logs_blocked(monkeypatch):
    lines = []

    def fake_log(msg):
        lines.append(msg)

    monkeypatch.setattr("services.gate_audit.log_info", fake_log)
    monkeypatch.setenv("GATE_AUDIT_DISABLE", "0")

    plan = {
        "tradingsymbol": "NIFTY2661623950CE",
        "option_type": "CE",
        "entry_ready": False,
        "entry_confirmation_score": 28,
        "entry_block_reason": "CE blocked: spot above prior close",
        "strategy_name": "5m Bollinger Bands",
    }
    record_gate_audit(
        "nifty50",
        plan,
        {},
        market_open=True,
        can_execute=False,
        entry_ready=False,
    )
    assert any("[GateAudit:nifty50] BLOCKED_GATE" in ln for ln in lines)


def test_day_summary_emits(monkeypatch):
    lines = []

    def fake_log(msg):
        lines.append(msg)

    monkeypatch.setattr("services.gate_audit.log_info", fake_log)
    monkeypatch.setenv("GATE_AUDIT_DISABLE", "0")

    plan = {
        "tradingsymbol": "X",
        "option_type": "PE",
        "entry_ready": False,
        "entry_block_reason": "ORB PE blocked",
        "strategy_name": "ORB",
    }
    record_gate_audit("commodity", plan, {}, market_open=True, entry_ready=False)
    record_gate_audit_placed("commodity", "CRUDEOILM26JUN7600PE")
    emit_gate_audit_day_summary("commodity")
    assert any("day_summary" in ln and "placed=1" in ln for ln in lines)
