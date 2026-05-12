"""HTTP surface for execution risk controls (kill switch, paper trading, auto-trade)."""
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from core.exceptions import ValidationError
from schemas.support_ops import KillSwitchUpdate
from services.paper_trading import (
    is_paper_mode,
    paper_trading_env_locks_ui,
    set_paper_trading_active,
)
from services.risk_gate import is_kill_switch_active, set_kill_switch
from services.strategy_auto_trader import get_config as get_auto_trade_config
from services.strategy_auto_trader import update_config as update_auto_trade_config

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


class AutoTradeUpdate(BaseModel):
    enabled: Optional[bool] = None
    place_sl_order: Optional[bool] = None
    place_target_order: Optional[bool] = None
    product: Optional[str] = None
    strategies: Optional[Dict[str, bool]] = None


@router.get("/auto-trade")
def get_auto_trade():
    """Read the strategy auto-trade configuration.

    Combined with `/risk/execution-status`, the UI can show "Auto-trade is ON,
    live broker routing" or "Auto-trade is ON, paper mode" etc.
    """
    return {"data": get_auto_trade_config()}


@router.post("/auto-trade")
def post_auto_trade(body: AutoTradeUpdate):
    """Persist a partial update for strategy auto-trade.

    Any field omitted from the body is left unchanged. Master ``enabled`` flag,
    per-strategy toggles in ``strategies``, and the optional SL/target placement
    flags are all supported.
    """
    patch: Dict[str, Any] = {}
    if body.enabled is not None:
        patch["enabled"] = bool(body.enabled)
    if body.place_sl_order is not None:
        patch["place_sl_order"] = bool(body.place_sl_order)
    if body.place_target_order is not None:
        patch["place_target_order"] = bool(body.place_target_order)
    if body.product is not None:
        patch["product"] = str(body.product).upper()
    if body.strategies is not None:
        patch["strategies"] = {str(k): bool(v) for k, v in body.strategies.items()}
    return {"data": update_auto_trade_config(patch)}
