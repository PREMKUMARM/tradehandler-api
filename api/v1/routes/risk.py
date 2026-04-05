"""HTTP surface for execution risk controls (kill switch)."""
from fastapi import APIRouter

from schemas.support_ops import KillSwitchUpdate
from services.paper_trading import is_paper_mode
from services.risk_gate import is_kill_switch_active, set_kill_switch

router = APIRouter(prefix="/risk", tags=["Risk"])


@router.get("/execution-status")
def get_execution_status():
    """Single read for UI banners: kill switch + whether server is in paper mode (env)."""
    return {
        "data": {
            "kill_switch_active": is_kill_switch_active(),
            "paper_trading_mode": is_paper_mode(),
        }
    }


@router.get("/kill-switch")
def get_kill_switch():
    return {"data": {"active": is_kill_switch_active()}}


@router.post("/kill-switch")
def post_kill_switch(body: KillSwitchUpdate):
    set_kill_switch(body.active)
    return {"data": {"active": is_kill_switch_active()}}
