"""BTCUSDT perp trade plan — strategy stub (awaiting configuration)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from services.crypto_config import (
    DEFAULT_LEVERAGE,
    EXCHANGE,
    SYMBOL,
    compute_crypto_quantity,
)
from services.crypto_live_indicators import recalculate_from_ticker
from utils.binance_order_utils import get_usdt_balance, round_price, round_quantity

STRATEGY_ID = "pending"
STRATEGY_NAME = "Crypto strategy (not configured)"
STRATEGY_PENDING_REASON = "Strategy reset — awaiting configuration"


def is_strategy_configured() -> bool:
    """Flip to True once entry/exit rules are implemented."""
    return False


def bb_reentry_reset_zone(_live: Dict[str, Any]) -> bool:
    """Compat hook for watch loop — always ready when strategy is unset."""
    return True


def passes_min_tp_reward(
    _entry: float, _tp: float, _side: str, _qty: float
) -> Tuple[bool, str]:
    return False, STRATEGY_PENDING_REASON


def passes_min_rr(
    _entry: float, _sl: float, _tp: float, _side: str
) -> Tuple[bool, str]:
    return False, STRATEGY_PENDING_REASON


def refresh_exits_at_fill(plan: Dict[str, Any], *, fill_price: float) -> Dict[str, Any]:
    """No exit logic until strategy is configured."""
    out = dict(plan)
    if fill_price > 0:
        out["entry_limit_price"] = round_price(SYMBOL, fill_price)
        out["entry_premium"] = out["entry_limit_price"]
    return out


def _resolve_side(direction: str) -> str:
    d = (direction or "AUTO").upper()
    if d in ("LONG", "SHORT"):
        return d
    return "LONG"


def build_trade_plan(
    *,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    quantity_btc: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    live = recalculate_from_ticker()
    messages: List[str] = [STRATEGY_PENDING_REASON]

    if not live.get("connected"):
        return {}, ["Binance not connected — check API keys"]

    side = _resolve_side(direction)
    spot = float(live.get("signal_spot") or live.get("btc_spot") or 0)
    if spot <= 0:
        return {}, ["BTCUSDT spot price unavailable"]

    if quantity_btc is None:
        from services.paper_trading import is_paper_mode_for_segment

        paper = is_paper_mode_for_segment("crypto")
        if paper:
            from services.paper_funds import get_available_balance

            usdt = float(get_available_balance("crypto") or 0)
            messages.append(f"Paper fund: ${usdt:,.2f} USDT available")
        else:
            try:
                usdt = float(get_usdt_balance() or 0)
            except Exception:
                usdt = 0.0
        qty, size_msgs = compute_crypto_quantity(usdt, spot, paper=paper, symbol=SYMBOL)
        messages.extend(size_msgs)
    else:
        from services.paper_trading import is_paper_mode_for_segment

        qty = round_quantity(SYMBOL, float(quantity_btc))
        if not is_paper_mode_for_segment("crypto"):
            try:
                usdt = float(get_usdt_balance() or 0)
            except Exception:
                usdt = 0.0
            need_margin = (qty * spot) / max(1, DEFAULT_LEVERAGE)
            if usdt + 0.01 < need_margin:
                messages.append(
                    f"Insufficient margin for {qty} BTC: need ${need_margin:,.2f} @ {DEFAULT_LEVERAGE}x, "
                    f"have ${usdt:,.2f} USDT"
                )
                qty = 0.0

    entry_limit = round_price(SYMBOL, spot)
    stop_loss_price = 0.0
    target_price = 0.0

    plan: Dict[str, Any] = {
        "tradingsymbol": SYMBOL,
        "exchange": EXCHANGE,
        "side": side,
        "option_type": side,
        "leverage": DEFAULT_LEVERAGE,
        "strike": 0,
        "expiry": "PERP",
        "quantity": qty,
        "num_lots": 1,
        "lot_size": qty,
        "product": "USDT-M",
        "entry_order_type": "LIMIT",
        "exit_order_type": "STOP_MARKET",
        "entry_limit_price": entry_limit,
        "entry_premium": entry_limit,
        "stop_loss_premium": stop_loss_price,
        "target_premium": target_price,
        "spot_stop_loss": stop_loss_price,
        "spot_target": target_price,
        "nifty_spot": spot,
        "risk_inr": 0.0,
        "reward_inr": 0.0,
        "reward_ratio": 0.0,
        "entry_ready": False,
        "entry_block_reason": STRATEGY_PENDING_REASON,
        "entry_style": "pending",
        "entry_confirmation_score": 0,
        "strategy_id": STRATEGY_ID,
        "strategy_name": STRATEGY_NAME,
        "trail_enabled": False,
        "indicators": live,
    }
    messages.append(
        f"{SYMBOL} {side} {DEFAULT_LEVERAGE}x · spot ${spot:,.2f} · qty {qty} BTC · strategy not configured"
    )
    return plan, messages


def refresh_plan_at_execution(plan: Dict[str, Any]) -> Dict[str, Any]:
    direction = plan.get("side") or plan.get("option_type") or "AUTO"
    fresh, _ = build_trade_plan(direction=str(direction), quantity_btc=None)
    if not fresh:
        return plan
    out = dict(plan)
    out.update(fresh)
    out["indicators"] = {**(plan.get("indicators") or {}), **(fresh.get("indicators") or {})}
    return out
