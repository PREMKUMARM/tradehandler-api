"""HTTP surface for execution risk controls (kill switch, paper trading)."""
from fastapi import APIRouter

from core.exceptions import ValidationError
from schemas.support_ops import KillSwitchUpdate
from services.paper_trading import (
    is_paper_mode,
    paper_trading_env_locks_ui,
    set_paper_trading_active,
)
from services.risk_gate import is_kill_switch_active, set_kill_switch

router = APIRouter(prefix="/risk", tags=["Risk"])


@router.get("/execution-status")
def get_execution_status():
    """Kill switch + paper mode + whether paper is locked by PAPER_TRADING_MODE in .env."""
    return {
        "data": {
            "kill_switch_active": is_kill_switch_active(),
            "paper_trading_mode": is_paper_mode(),
            "paper_trading_env_locks_ui": paper_trading_env_locks_ui(),
        }
    }


@router.get("/kill-switch")
def get_kill_switch():
    return {"data": {"active": is_kill_switch_active()}}


@router.post("/kill-switch")
def post_kill_switch(body: KillSwitchUpdate):
    set_kill_switch(body.active)
    return {"data": {"active": is_kill_switch_active()}}


@router.post("/paper-trading")
def post_paper_trading(body: KillSwitchUpdate):
    """Persist paper mode on/off (same shape as kill switch: { active: bool })."""
    try:
        set_paper_trading_active(body.active)
    except ValueError as e:
        raise ValidationError(message=str(e), field="paper_trading") from e
    return {"data": {"active": is_paper_mode(), "paper_trading_env_locks_ui": paper_trading_env_locks_ui()}}
