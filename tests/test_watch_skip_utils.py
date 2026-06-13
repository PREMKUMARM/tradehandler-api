"""Tests for watch skip messages and paper can_execute gates."""
from services.trading_agents.execution_agent import resolve_can_execute
from services.watch_skip_utils import normalize_skip_message, validation_skip_message


def test_normalize_skip_message_empty_uses_fallback():
    assert normalize_skip_message("") == "Autonomous placement blocked"
    assert normalize_skip_message("  ") == "Autonomous placement blocked"
    assert normalize_skip_message("Score too low") == "Score too low"


def test_validation_skip_message_risk_reward():
    preview = {
        "can_place": False,
        "validation": {
            "is_good_trade": False,
            "failure_reasons": [
                "Risk ₹5,000 exceeds 1% of capital (max ₹3,000)",
                "Fix: reduce lots in Settings",
            ],
        },
    }
    msg = validation_skip_message(preview)
    assert "Risk ₹5,000" in msg
    assert "reduce lots" in msg


def test_validation_skip_message_reward_policy():
    preview = {
        "validation": {
            "is_good_trade": False,
            "failure_reasons": ["Reward does not meet risk/reward policy"],
        }
    }
    assert "Reward does not meet" in validation_skip_message(preview)


def test_paper_can_execute_blocks_bad_validation():
    preview = {
        "checklist_ready": True,
        "paper_trading_mode": True,
        "can_place": False,
        "validation": {"is_good_trade": False, "failure_reasons": ["Risk too high"]},
        "trade_plan": {
            "entry_ready": True,
            "entry_limit_price": 150.0,
            "stop_loss_premium": 130.0,
            "target_premium": 180.0,
        },
    }
    plan = preview["trade_plan"]
    assert resolve_can_execute(preview, plan) is False


def test_paper_can_execute_blocks_degenerate_premium():
    preview = {
        "checklist_ready": True,
        "paper_trading_mode": True,
        "can_place": False,
        "validation": {"is_good_trade": True},
        "trade_plan": {
            "entry_ready": True,
            "entry_limit_price": 0.05,
            "stop_loss_premium": 0.05,
            "target_premium": 0.10,
        },
    }
    plan = preview["trade_plan"]
    assert resolve_can_execute(preview, plan) is False


def test_paper_can_execute_allows_valid_plan():
    preview = {
        "checklist_ready": True,
        "paper_trading_mode": True,
        "can_place": False,
        "validation": {"is_good_trade": True},
        "trade_plan": {
            "entry_ready": True,
            "entry_limit_price": 150.0,
            "stop_loss_premium": 130.0,
            "target_premium": 180.0,
        },
    }
    plan = preview["trade_plan"]
    assert resolve_can_execute(preview, plan) is True
