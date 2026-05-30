"""BTCUSDT perp trade plan from live Binance data — 5m Bollinger Bands mean reversion."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from services.crypto_config import (
    DEFAULT_LEVERAGE,
    DEFAULT_QUANTITY_BTC,
    DEFAULT_REWARD_RATIO,
    DEFAULT_RISK_PCT,
    EXCHANGE,
    SYMBOL,
    compute_crypto_quantity,
)
from services.crypto_live_indicators import recalculate_from_ticker
from services.kite_live_indicators import bollinger_zone
from utils.binance_order_utils import get_usdt_balance, round_price, round_quantity


def _resolve_side(direction: str, live: Dict[str, Any]) -> str:
    d = (direction or "AUTO").upper()
    if d in ("LONG", "SHORT"):
        return d

    spot = float(live.get("btc_spot") or 0)
    mid = live.get("bb_middle")
    upper = live.get("bb_upper")
    lower = live.get("bb_lower")
    if mid is not None and upper is not None and lower is not None:
        bb_ce = bollinger_zone(spot, float(mid), float(upper), float(lower), "CE")
        if bb_ce.get("preferred"):
            return "LONG"
        bb_pe = bollinger_zone(spot, float(mid), float(upper), float(lower), "PE")
        if bb_pe.get("preferred"):
            return "SHORT"
        zone = str(bb_ce.get("zone") or "")
        if zone in ("lower", "middle"):
            return "LONG"
        if zone == "upper":
            return "SHORT"

    prev = float(live.get("prev_close") or spot)
    ema9 = float(live.get("ema9") or spot)
    if spot >= ema9 and spot >= prev:
        return "LONG"
    return "SHORT"


def _bb_entry_analysis(
    side: str, spot: float, live: Dict[str, Any]
) -> Tuple[bool, Optional[str], int, str, float, Optional[str]]:
    """5m BB entry: LONG at lower/middle, SHORT at upper/middle; block extension."""
    kind = "CE" if side == "LONG" else "PE"
    mid = live.get("bb_middle")
    upper = live.get("bb_upper")
    lower = live.get("bb_lower")
    if mid is None or upper is None or lower is None:
        return (
            False,
            "5m Bollinger Bands not ready — need 20×5m bars on BTCUSDT",
            0,
            "blocked_wait",
            spot,
            None,
        )

    bb = bollinger_zone(spot, float(mid), float(upper), float(lower), kind)
    zone = str(bb.get("zone") or "")
    style = f"bb5m_{zone}"

    if bb.get("extended"):
        return False, str(bb.get("wait_msg") or "BB extension — wait for pullback"), 28, style, float(bb["trigger"]), zone

    if bb.get("preferred"):
        trigger = float(bb["trigger"])
        limit = round_price(SYMBOL, trigger)
        return True, None, 88, style, limit, zone

    if zone == "between":
        limit = round_price(SYMBOL, float(mid))
        return True, None, 70, f"{style}_patient", limit, zone

    return False, str(bb.get("wait_msg") or f"Wait for {side} BB setup"), 38, style, float(bb["trigger"]), zone


def _bb_exit_levels(side: str, spot: float, live: Dict[str, Any], rr: float) -> Tuple[float, float, float]:
    upper = float(live.get("bb_upper") or spot)
    lower = float(live.get("bb_lower") or spot)
    width = max(upper - lower, spot * 0.002)
    if side == "LONG":
        sl = round_price(SYMBOL, min(lower, spot - width * 0.25))
        tp = round_price(SYMBOL, spot + width * rr * 0.5)
    else:
        sl = round_price(SYMBOL, max(upper, spot + width * 0.25))
        tp = round_price(SYMBOL, spot - width * rr * 0.5)
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
    entry_ready, block, score, entry_style, entry_limit, bb_zone = _bb_entry_analysis(side, spot, live)
    spot_entry, stop_loss_price, target_price = _bb_exit_levels(
        side, spot, live, reward_percentage or DEFAULT_REWARD_RATIO
    )

    risk_pct = float(risk_percentage or DEFAULT_RISK_PCT)
    rr = float(reward_percentage or DEFAULT_REWARD_RATIO) if reward_percentage else DEFAULT_REWARD_RATIO

    if not entry_ready:
        entry_limit = round_price(SYMBOL, spot * 0.9995 if side == "LONG" else spot * 1.0005)

    sl_dist = abs(spot_entry - stop_loss_price)
    risk_inr = 0.0

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
        if qty <= 0 and not paper:
            block = block or (size_msgs[-1] if size_msgs else "Insufficient margin for min lot")
            entry_ready = False
            score = min(int(score or 0), 38)
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
                block = (
                    block
                    or f"Insufficient margin for {qty} BTC: need ${need_margin:,.2f} @ {DEFAULT_LEVERAGE}x, "
                    f"have ${usdt:,.2f} USDT"
                )
                entry_ready = False
                score = min(int(score or 0), 38)
                messages.append(block)
    notional = qty * spot
    risk_inr = notional * (risk_pct / 100.0) * DEFAULT_LEVERAGE

    bb_lo = live.get("bb_lower")
    bb_mid = live.get("bb_middle")
    bb_up = live.get("bb_upper")
    if bb_lo is not None and bb_mid is not None and bb_up is not None:
        messages.append(
            f"5m BB L ${float(bb_lo):,.2f} M ${float(bb_mid):,.2f} U ${float(bb_up):,.2f} · zone {bb_zone or '—'}"
        )

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
        "nifty_spot": spot_entry,
        "risk_inr": round(risk_inr, 2),
        "reward_inr": round(risk_inr * rr, 2),
        "reward_ratio": rr,
        "entry_ready": entry_ready,
        "entry_block_reason": block,
        "entry_style": entry_style,
        "entry_confirmation_score": score,
        "strategy_id": "bb_5m_mean_reversion",
        "strategy_name": f"BTCUSDT {side} BB 5m {DEFAULT_LEVERAGE}x",
        "bb_lower": bb_lo,
        "bb_middle": bb_mid,
        "bb_upper": bb_up,
        "bb_zone": bb_zone,
        "indicators": live,
    }
    messages.append(
        f"{SYMBOL} {side} {DEFAULT_LEVERAGE}x · LIMIT ${entry_limit:,.2f} · "
        f"SL ${stop_loss_price:,.2f} TP ${target_price:,.2f} · qty {qty} BTC"
    )
    if block:
        messages.append(block)
    return plan, messages


def refresh_plan_at_execution(plan: Dict[str, Any]) -> Dict[str, Any]:
    direction = plan.get("side") or plan.get("option_type") or "AUTO"
    fresh, _ = build_trade_plan(
        direction=str(direction),
        quantity_btc=None,
    )
    if not fresh:
        return plan
    out = dict(plan)
    out.update(fresh)
    out["indicators"] = {**(plan.get("indicators") or {}), **(fresh.get("indicators") or {})}
    return out
