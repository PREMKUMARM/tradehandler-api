"""BTCUSDT perp trade plan from live Binance data."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from services.crypto_config import (
    DEFAULT_LEVERAGE,
    DEFAULT_QUANTITY_BTC,
    DEFAULT_REWARD_RATIO,
    DEFAULT_RISK_PCT,
    EXCHANGE,
    MAX_NOTIONAL_PCT_OF_USDT,
    SYMBOL,
)
from services.crypto_live_indicators import recalculate_from_ticker
from utils.binance_order_utils import get_usdt_balance, get_symbol_price, round_price, round_quantity


def _resolve_side(direction: str, live: Dict[str, Any]) -> str:
    d = (direction or "AUTO").upper()
    if d in ("LONG", "SHORT"):
        return d
    spot = float(live.get("btc_spot") or 0)
    prev = float(live.get("prev_close") or spot)
    ema9 = float(live.get("ema9") or spot)
    if spot >= ema9 and spot >= prev:
        return "LONG"
    return "SHORT"


def _spot_levels(
    side: str, spot: float, live: Dict[str, Any], rr: float
) -> Tuple[float, float, float]:
    or_high = float(live.get("or_high") or spot)
    or_low = float(live.get("or_low") or spot)
    pdh = float(live.get("pdh") or spot)
    pdl = float(live.get("pdl") or spot)
    risk_pts = max(50.0, (or_high - or_low) * 0.5, abs(spot - (pdl if side == "LONG" else pdh)) * 0.15)
    if side == "LONG":
        sl = min(or_low, pdl, spot - risk_pts)
        tp = spot + risk_pts * rr
    else:
        sl = max(or_high, pdh, spot + risk_pts)
        tp = spot - risk_pts * rr
    return spot, sl, tp


def build_trade_plan(
    *,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    quantity_btc: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    live = recalculate_from_ticker()
    messages: List[str] = []
    if not live.get("connected"):
        return {}, ["Binance not connected — check API keys"]

    side = _resolve_side(direction, live)
    spot = float(live.get("btc_spot") or 0)
    spot_entry, spot_sl, spot_tp = _spot_levels(
        side, spot, live, reward_percentage or DEFAULT_REWARD_RATIO
    )

    risk_pct = float(risk_percentage or DEFAULT_RISK_PCT)
    rr = float(reward_percentage or DEFAULT_REWARD_RATIO) if reward_percentage else DEFAULT_REWARD_RATIO

    entry_limit = round_price(SYMBOL, spot * 0.9995 if side == "LONG" else spot * 1.0005)
    sl_dist = abs(spot_entry - spot_sl)
    tp_dist = sl_dist * rr
    if side == "LONG":
        stop_loss_price = round_price(SYMBOL, spot_entry - sl_dist)
        target_price = round_price(SYMBOL, spot_entry + tp_dist)
    else:
        stop_loss_price = round_price(SYMBOL, spot_entry + sl_dist)
        target_price = round_price(SYMBOL, spot_entry - tp_dist)

    if quantity_btc is None:
        # Size from available USDT so the wizard works out-of-box without manual qty.
        try:
            usdt = float(get_usdt_balance() or 0)
        except Exception:
            usdt = 0.0
        budget = usdt * (MAX_NOTIONAL_PCT_OF_USDT / 100.0) * DEFAULT_LEVERAGE
        sized_qty = (budget / spot) if (spot > 0 and budget > 0) else DEFAULT_QUANTITY_BTC
        qty = round_quantity(SYMBOL, float(sized_qty))
        messages.append(f"Sizing: {MAX_NOTIONAL_PCT_OF_USDT:.0f}% of USDT balance @ {DEFAULT_LEVERAGE}x")
    else:
        qty = round_quantity(SYMBOL, float(quantity_btc or DEFAULT_QUANTITY_BTC))
    notional = qty * spot
    risk_inr = notional * (risk_pct / 100.0) * DEFAULT_LEVERAGE

    ema9 = float(live.get("ema9") or spot)
    entry_ready = (side == "LONG" and spot > ema9) or (side == "SHORT" and spot < ema9)
    block = None if entry_ready else f"Wait for {side} alignment vs EMA9"

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
        "spot_stop_loss": spot_sl,
        "spot_target": spot_tp,
        "nifty_spot": spot_entry,
        "risk_inr": round(risk_inr, 2),
        "reward_inr": round(risk_inr * rr, 2),
        "reward_ratio": rr,
        "entry_ready": entry_ready,
        "entry_block_reason": block,
        "entry_style": "patient_limit",
        "entry_confirmation_score": 70 if entry_ready else 45,
        "strategy_id": "btc_trend",
        "strategy_name": f"BTCUSDT {side} {DEFAULT_LEVERAGE}x",
        "indicators": live,
    }
    messages.append(
        f"{SYMBOL} {side} {DEFAULT_LEVERAGE}x · LIMIT ${entry_limit:,.2f} · "
        f"SL ${stop_loss_price:,.2f} TP ${target_price:,.2f} · qty {qty} BTC"
    )
    return plan, messages


def refresh_plan_at_execution(plan: Dict[str, Any]) -> Dict[str, Any]:
    direction = plan.get("side") or plan.get("option_type") or "AUTO"
    fresh, _ = build_trade_plan(
        direction=str(direction),
        quantity_btc=float(plan.get("quantity") or DEFAULT_QUANTITY_BTC),
    )
    if not fresh:
        return plan
    out = dict(plan)
    out.update(fresh)
    out["indicators"] = {**(plan.get("indicators") or {}), **(fresh.get("indicators") or {})}
    return out
