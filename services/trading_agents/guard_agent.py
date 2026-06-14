"""GuardAgent — entry quality and duplicate order/position checks (NFO / MCX)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from services.trading_agents.types import OPEN_ORDER_STATUSES
from utils.logger import log_warning


@dataclass(frozen=True)
class GuardAgentConfig:
    segment: str
    exchange: str
    min_score_env: str
    max_spread_env: str
    log_prefix: str
    placed_today_message: str
    pre_check: Optional[Callable[[], Tuple[bool, str]]] = None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def min_entry_confirmation_score(config: GuardAgentConfig) -> int:
    default = 55 if config.segment == "commodity" else 65
    return max(40, min(100, _env_int(config.min_score_env, default)))


def entry_quality_for_autonomous(
    plan: Dict[str, Any],
    *,
    config: GuardAgentConfig,
) -> Tuple[bool, str]:
    """Strict entry gate — autonomous must not fire on missing or weak confirmation."""
    min_score = min_entry_confirmation_score(config)
    max_spread = _env_float(config.max_spread_env, 2.5)

    if not plan:
        return False, "No trade plan"
    if plan.get("entry_ready") is not True:
        reason = plan.get("entry_block_reason") or "Entry not confirmed by indicators"
        return False, reason
    block = plan.get("entry_block_reason")
    if block:
        return False, str(block)
    score = int(plan.get("entry_confirmation_score") or 0)
    if score < min_score:
        return False, (
            f"Confirmation score {score} below minimum {min_score} "
            f"for autonomous entry"
        )
    style = str(plan.get("entry_style") or "")
    if "blocked" in style.lower() or style == "blocked_wait":
        return False, "Entry style blocked — wait for setup"
    limit_px = float(plan.get("entry_limit_price") or 0)
    fair = float(plan.get("entry_fair_premium") or limit_px)
    if limit_px <= 0:
        return False, "Invalid entry limit price"
    if fair > 0 and limit_px > fair * 1.025:
        return False, f"Limit ₹{limit_px} chases above fair ₹{fair:.2f} — no autonomous chase"
    ind = plan.get("indicators") or {}
    bid = float(ind.get("option_bid") or 0)
    ask = float(ind.get("option_ask") or 0)
    ltp = float(ind.get("option_ltp") or plan.get("entry_premium") or 0)
    if bid > 0 and ask > bid and ltp > 0:
        spread = (ask - bid) / ltp * 100.0
        if spread > max_spread:
            return False, f"Option spread {spread:.1f}% too wide for autonomous entry"
    return True, "Entry confirmed"


def has_pending_exchange_order(
    tradingsymbol: str,
    *,
    exchange: str,
    log_prefix: str,
) -> Tuple[bool, str]:
    sym = (tradingsymbol or "").strip().upper()
    ex = (exchange or "").strip().upper()
    if not sym:
        return False, ""
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance()
        orders: List[Dict[str, Any]] = list(kite.orders() or [])
        for o in orders:
            if str(o.get("exchange") or "").upper() != ex:
                continue
            if str(o.get("tradingsymbol") or "").upper() != sym:
                continue
            status = str(o.get("status") or "").upper()
            if status not in OPEN_ORDER_STATUSES:
                continue
            if str(o.get("transaction_type") or "").upper() == "BUY":
                oid = o.get("order_id")
                return True, f"Open BUY order {oid} already exists for {sym}"
        return False, ""
    except Exception as exc:
        log_warning(f"[{log_prefix}] orders check failed: {exc}")
        return True, f"Could not verify open orders — blocked: {exc}"


def has_any_exchange_position(
    *,
    exchange: str,
    log_prefix: str,
) -> Tuple[bool, str]:
    """True if any open long position exists on the exchange (segment-wide gate)."""
    ex = (exchange or "").strip().upper()
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance()
        positions = kite.positions() or {}
        held: List[str] = []
        for bucket in ("net", "day"):
            for p in positions.get(bucket) or []:
                if str(p.get("exchange") or "").upper() != ex:
                    continue
                qty = int(p.get("quantity") or 0)
                if qty > 0:
                    sym = str(p.get("tradingsymbol") or "").upper()
                    if sym and sym not in held:
                        held.append(sym)
        if held:
            label = held[0] if len(held) == 1 else f"{held[0]} +{len(held) - 1} more"
            return (
                True,
                f"Open {ex} position on {label} — exit before new commodity entry",
            )
        return False, ""
    except Exception as exc:
        log_warning(f"[{log_prefix}] positions check failed: {exc}")
        return True, f"Could not verify positions — blocked: {exc}"


def has_exchange_position(
    tradingsymbol: str,
    *,
    exchange: str,
    log_prefix: str,
) -> Tuple[bool, str]:
    sym = (tradingsymbol or "").strip().upper()
    ex = (exchange or "").strip().upper()
    if not sym:
        return False, ""
    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance()
        positions = kite.positions() or {}
        for bucket in ("net", "day"):
            for p in positions.get(bucket) or []:
                if str(p.get("exchange") or "").upper() != ex:
                    continue
                if str(p.get("tradingsymbol") or "").upper() != sym:
                    continue
                qty = int(p.get("quantity") or 0)
                if qty > 0:
                    return True, f"Open position on {sym} (qty={qty}) — exit before re-entry"
        return False, ""
    except Exception as exc:
        log_warning(f"[{log_prefix}] positions check failed: {exc}")
        return True, f"Could not verify positions — blocked: {exc}"


def autonomous_place_allowed(
    plan: Dict[str, Any],
    *,
    config: GuardAgentConfig,
    placed_today: bool,
) -> Tuple[bool, str]:
    if config.pre_check:
        ok, msg = config.pre_check()
        if not ok:
            return False, msg
    if placed_today:
        return False, config.placed_today_message
    try:
        from services.paper_trading import is_paper_mode_for_segment

        if is_paper_mode_for_segment(config.segment):
            from services.paper_order_guard import paper_autonomous_place_allowed

            return paper_autonomous_place_allowed(
                plan,
                placed_today=placed_today,
                segment=config.segment,
            )
    except Exception:
        pass
    sym = str(plan.get("tradingsymbol") or "")
    ok, msg = entry_quality_for_autonomous(plan, config=config)
    if not ok:
        return False, msg
    if config.exchange == "MCX":
        any_pos, any_msg = has_any_exchange_position(
            exchange=config.exchange, log_prefix=config.log_prefix
        )
        if any_pos:
            return False, any_msg
    pending, pend_msg = has_pending_exchange_order(
        sym, exchange=config.exchange, log_prefix=config.log_prefix
    )
    if pending:
        return False, pend_msg
    pos, pos_msg = has_exchange_position(
        sym, exchange=config.exchange, log_prefix=config.log_prefix
    )
    if pos:
        return False, pos_msg
    return True, "OK"


def _commodity_cutoff_check() -> Tuple[bool, str]:
    try:
        from services.commodity_config import (
            commodity_trading_cutoff_label,
            is_commodity_new_trading_allowed,
        )

        if not is_commodity_new_trading_allowed():
            return False, (
                f"Commodity trading closed for the day "
                f"(cutoff {commodity_trading_cutoff_label()} IST)"
            )
    except Exception:
        pass
    return True, "OK"


def _commodity_pre_check() -> Tuple[bool, str]:
    ok, msg = _commodity_cutoff_check()
    if not ok:
        return ok, msg
    return has_any_exchange_position(exchange="MCX", log_prefix="CommodityGuard")


NIFTY_GUARD = GuardAgentConfig(
    segment="nifty50",
    exchange="NFO",
    min_score_env="NIFTY_AUTO_MIN_ENTRY_SCORE",
    max_spread_env="NIFTY_AUTO_MAX_SPREAD_PCT",
    log_prefix="V2Guard",
    placed_today_message="Max autonomous Nifty trades per day reached",
)

SENSEX_GUARD = GuardAgentConfig(
    segment="sensex",
    exchange="BFO",
    min_score_env="SENSEX_AUTO_MIN_ENTRY_SCORE",
    max_spread_env="SENSEX_AUTO_MAX_SPREAD_PCT",
    log_prefix="SensexGuard",
    placed_today_message="Max autonomous Sensex trades per day reached",
)

COMMODITY_GUARD = GuardAgentConfig(
    segment="commodity",
    exchange="MCX",
    min_score_env="COMMODITY_AUTO_MIN_ENTRY_SCORE",
    max_spread_env="COMMODITY_AUTO_MAX_SPREAD_PCT",
    log_prefix="CommodityGuard",
    placed_today_message="Max autonomous commodity trades per day reached",
    pre_check=_commodity_pre_check,
)
