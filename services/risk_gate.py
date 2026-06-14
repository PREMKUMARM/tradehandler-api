"""
P0/P1: Central pre-trade checks — kill switch, session window, exchange policy, daily trade limits.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from zoneinfo import ZoneInfo

from agent.config import get_agent_config
from utils.logger import log_info, log_warning
from utils.trade_limits import trade_limits

KILL_SWITCH_PATH = Path(os.getenv("KILL_SWITCH_FILE", "data/kill_switch.json"))


def _normalize_segment(segment: Optional[str]) -> Optional[str]:
    if not segment:
        return None
    s = segment.strip().lower()
    if s in ("v2", "nifty", "nifty50", "nfo"):
        return "nifty"
    if s in ("commodity", "mcx", "crude"):
        return "commodity"
    if s in ("crypto", "binance", "btc"):
        return "crypto"
    if s in ("sensex", "bfo", "bse"):
        return "sensex"
    return s


def _read_kill_switch_data() -> Dict[str, Any]:
    if not KILL_SWITCH_PATH.exists():
        return {}
    try:
        raw = json.loads(KILL_SWITCH_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception as e:
        log_warning(f"[RiskGate] Kill-switch file read failed (fail-closed): {e}")
        return {"_read_error": True}


def is_kill_switch_active(segment: Optional[str] = None) -> bool:
    """Global env/file `active` or per-segment `nifty` / `commodity` flags."""
    env = os.getenv("EXECUTION_KILL_SWITCH", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True

    data = _read_kill_switch_data()
    if data.get("_read_error"):
        return True

    if bool(data.get("active")):
        return True

    seg = _normalize_segment(segment)
    if seg == "nifty":
        return bool(data.get("nifty") or data.get("nifty50"))
    if seg == "commodity":
        return bool(data.get("commodity"))
    if seg == "crypto":
        return bool(data.get("crypto"))
    if seg == "sensex":
        return bool(data.get("sensex"))
    return bool(data.get("nifty") or data.get("nifty50") or data.get("commodity") or data.get("crypto") or data.get("sensex"))


def get_kill_switch_status() -> Dict[str, Any]:
    data = _read_kill_switch_data()
    return {
        "active": is_kill_switch_active(),
        "global_active": bool(data.get("active")),
        "nifty": bool(data.get("nifty") or data.get("nifty50")),
        "commodity": bool(data.get("commodity")),
        "crypto": bool(data.get("crypto")),
        "sensex": bool(data.get("sensex")),
        "read_error": bool(data.get("_read_error")),
    }


def set_kill_switch(active: bool, segment: Optional[str] = None) -> None:
    KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _read_kill_switch_data()
    data.pop("_read_error", None)

    seg = _normalize_segment(segment)
    if seg is None:
        data["active"] = bool(active)
        log_info(f"[RiskGate] Global kill switch set to active={active}")
    elif seg == "nifty":
        data["nifty"] = bool(active)
        log_info(f"[RiskGate] Nifty kill switch set to active={active}")
    elif seg == "commodity":
        data["commodity"] = bool(active)
        log_info(f"[RiskGate] Commodity kill switch set to active={active}")
    elif seg == "crypto":
        data["crypto"] = bool(active)
        log_info(f"[RiskGate] Crypto kill switch set to active={active}")
    elif seg == "sensex":
        data["sensex"] = bool(active)
        log_info(f"[RiskGate] Sensex kill switch set to active={active}")
    else:
        data[seg] = bool(active)

    KILL_SWITCH_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _segment_for_exchange(exchange: str) -> Optional[str]:
    ex = (exchange or "").upper()
    if ex == "MCX":
        return "commodity"
    if ex == "BFO":
        return "sensex"
    if ex in ("NFO", "NSE", "BSE"):
        return "nifty"
    return None


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


def _within_session_for_exchange(exchange: str) -> Tuple[bool, str]:
    """
    Exchange-aware session window.
    - NSE/NFO/BSE: use agent config (default 09:15–15:30 IST)
    - MCX: use commodity session (default 09:00–23:30 IST)
    """
    ex = (exchange or "").upper()
    if ex == "MCX":
        try:
            from services.commodity_config import (
                MCX_OPEN_MINUTES,
                commodity_trading_cutoff_label,
                commodity_trading_cutoff_minutes,
            )

            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            if now.weekday() > 4:
                return False, "Outside trading session (IST 09:00-cutoff)"
            minutes = now.hour * 60 + now.minute
            cutoff = commodity_trading_cutoff_minutes()
            if MCX_OPEN_MINUTES <= minutes < cutoff:
                return True, "ok"
            if minutes >= cutoff:
                return False, (
                    f"Commodity trading closed for the day (cutoff {commodity_trading_cutoff_label()} IST)"
                )
            return False, "Outside trading session (IST 09:00-cutoff)"
        except Exception:
            now_t = datetime.now(ZoneInfo("Asia/Kolkata")).time()
            if dt_time(9, 0) <= now_t <= dt_time(23, 15):
                return True, "ok"
            return False, "Outside trading session (IST 09:00-23:15)"
    return _within_session_ist()


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
    segment: Optional[str] = None,
) -> Tuple[bool, str]:
    seg = _normalize_segment(segment) or _segment_for_exchange(exchange)
    if is_kill_switch_active(seg):
        label = f" ({seg})" if seg else ""
        return False, f"Execution kill switch is ON{label} — orders blocked."

    paper_seg = seg
    if paper_seg == "nifty":
        paper_seg = "nifty50"
    try:
        from services.paper_trading import is_paper_mode_for_segment

        if is_paper_mode_for_segment(paper_seg):
            ok_ex, msg_ex = _exchange_allowed(exchange)
            if not ok_ex:
                return False, msg_ex
            if quantity <= 0:
                return False, "Quantity must be positive."
            return True, "ok (paper ledger — funds checked at place)"
    except Exception:
        pass

    ok_ex, msg_ex = _exchange_allowed(exchange)
    if not ok_ex:
        return False, msg_ex

    if not skip_session_check:
        ok_s, msg_s = _within_session_for_exchange(exchange)
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


def rollback_order_reserved(investment_amount: float = 0.0) -> None:
    """Undo a reserved slot when a submitted entry is cancelled before fill."""
    trade_limits.rollback_trade(investment_amount)


def get_risk_limits_snapshot() -> dict:
    """Daily caps and P&L for watch status / UI."""
    return trade_limits.get_limits_status()
