"""Tests for multi-entry commodity pending + segment guard."""
from services.commodity_watch_pending import (
    migrate_pending_entries,
    pending_needing_gtt,
    register_pending_entry,
    sync_legacy_pending_fields,
)
from services.trading_agents.guard_agent import has_any_exchange_position


def test_migrate_legacy_single_pending():
    data = {
        "pending_entry_order_id": "A1",
        "pending_symbol": "CRUDEOILM26JUN8750PE",
        "pending_entry_placed_at": "2026-06-02T09:01:49+05:30",
        "pending_trade_plan": {"tradingsymbol": "CRUDEOILM26JUN8750PE"},
        "pending_gtt_trigger_id": None,
    }
    entries = migrate_pending_entries(data)
    assert "A1" in entries
    assert entries["A1"]["trade_plan"]["tradingsymbol"] == "CRUDEOILM26JUN8750PE"


def test_two_pending_entries_sync_legacy_oldest_needing_gtt():
    entries = {}
    register_pending_entry(
        entries,
        order_id="A1",
        symbol="CRUDEOILM26JUN8750PE",
        placed_at="t1",
        trade_plan={"x": 1},
    )
    register_pending_entry(
        entries,
        order_id="A2",
        symbol="CRUDEOILM26JUN8700PE",
        placed_at="t2",
        trade_plan={"x": 2},
    )
    assert pending_needing_gtt(entries)
    oid, *_ = sync_legacy_pending_fields(entries)
    assert oid == "A1"


def test_has_any_exchange_position_no_kite(monkeypatch):
    monkeypatch.setattr(
        "utils.kite_utils.get_kite_instance",
        lambda: (_ for _ in ()).throw(RuntimeError("no kite")),
    )
    blocked, msg = has_any_exchange_position(exchange="MCX", log_prefix="Test")
    assert blocked is True
    assert "Could not verify" in msg
