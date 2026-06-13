"""Tests for checklist step None coercion and paper entry floor."""
from schemas.v2_trading import ChecklistStepStatus
from services.checklist_step_utils import parse_checklist_step
from services.paper_order_guard import paper_entry_levels_valid


def test_checklist_step_none_coerced_to_false():
    st = parse_checklist_step(
        {
            "index": 0,
            "title": "Session",
            "completed": None,
            "server_ok": None,
            "message": "x",
        },
        ChecklistStepStatus,
    )
    assert st.completed is False
    assert st.server_ok is False


def test_paper_entry_levels_reject_degenerate():
    ok, msg = paper_entry_levels_valid(0.05, 0.05, 0.10)
    assert ok is False
    assert "minimum" in msg.lower()


def test_paper_entry_levels_accept_normal():
    ok, _ = paper_entry_levels_valid(150.0, 130.0, 180.0)
    assert ok is True
