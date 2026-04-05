"""
P0/P1: Central pre-trade checks — kill switch, session window, exchange policy, daily trade limits.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Tuple

from zoneinfo import ZoneInfo

from agent.config import get_agent_config
from utils.logger import log_info, log_warning
from utils.trade_limits import trade_limits

KILL_SWITCH_PATH = Path(os.getenv("KILL_SWITCH_FILE", "data/kill_switch.json"))


def is_kill_switch_active() -> bool:
    env = os.getenv("EXECUTION_KILL_SWITCH", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    try:
        if KILL_SWITCH_PATH.exists():
            data = json.loads(KILL_SWITCH_PATH.read_text(encoding="utf-8"))
            return bool(data.get("active"))
    except Exception as e:
        log_warning(f"[RiskGate] Kill-switch file read failed: {e}")
    return False


def set_kill_switch(active: bool) -> None:
    KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    KILL_SWITCH_PATH.write_text(json.dumps({"active": bool(active)}, indent=2), encoding="utf-8")
    log_info(f"[RiskGate] Kill switch set to active={active}")


def _within_session_ist() -> Tuple[bool, str]:
    cfg = get_agent_config()
    now = datetime.now(ZoneInfo("Asia/Kolkata")).time()
    try:
        start = datetime.strptime(cfg.trading_start_time, "%H:%M").time()
        end = datetime.strptime(cfg.trading_end_time, "%H:%M").time()
    except Exception:
        start, end = dt_time(9, 15), dt_time(15, 30)
    if start <= now <= end:
        return True, "ok"
    return False, f"Outside trading session (IST {cfg.trading_start_time}-{cfg.trading_end_time})"


def _exchange_allowed(exchange: str) -> Tuple[bool, str]:
    raw = os.getenv("ALLOWED_ORDER_EXCHANGES", "NFO,NSE,BSE")
    allowed = {x.strip().upper() for x in raw.split(",") if x.strip()}
    ex = (exchange or "").upper()
    if ex in allowed:
        return True, "ok"
    return False, f"Exchange '{exchange}' not permitted (ALLOWED_ORDER_EXCHANGES={raw})"


def check_order_allowed(
    exchange: str,
    tradingsymbol: str,
    quantity: int,
    transaction_type: str,
    estimated_value_inr: float = 0.0,
    *,
    skip_session_check: bool = False,
) -> Tuple[bool, str]:
    if is_kill_switch_active():
        return False, "Execution kill switch is ON — orders blocked."

    ok_ex, msg_ex = _exchange_allowed(exchange)
    if not ok_ex:
        return False, msg_ex

    if not skip_session_check:
        ok_s, msg_s = _within_session_ist()
        if not ok_s:
            return False, msg_s

    cfg = get_agent_config()
    if quantity <= 0:
        return False, "Quantity must be positive."

    if cfg.max_trades_per_day and trade_limits.limits.get("trades_today", 0) >= cfg.max_trades_per_day:
        return False, f"Daily trade count cap reached ({cfg.max_trades_per_day})."

    ok, msg = trade_limits.can_place_trade(estimated_value_inr or 0.0)
    if not ok:
        return False, msg

    return True, "ok"


def record_order_placed(investment_amount: float = 0.0) -> None:
    trade_limits.record_trade(investment_amount)
