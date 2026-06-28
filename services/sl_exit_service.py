"""Stop-loss exit protection after entry fill (replaces Zerodha GTT OCO)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from agent.tools.kite_tools import modify_order_tool, place_order_tool
from utils.kite_order_utils import round_to_tick
from utils.logger import log_info, log_warning


def stepped_targets_from_plan(plan: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Return (entry, sl, t1, t2) premium levels for a long option plan.

    T1 = first target (target_premium). T2 = entry + 2R (second reward step).
    """
    entry = float(plan.get("entry_limit_price") or plan.get("entry_premium") or 0)
    sl = float(plan.get("stop_loss_premium") or 0)
    t1 = float(plan.get("target_premium") or 0)
    if entry <= 0 or sl <= 0:
        return entry, sl, t1, t1
    risk = max(0.05, entry - sl)
    if t1 <= entry:
        t1 = round_to_tick(entry + risk)
    t2 = float(plan.get("target_2_premium") or 0)
    if t2 <= t1:
        t2 = round_to_tick(entry + 2 * risk)
    return entry, sl, round_to_tick(t1), round_to_tick(t2)


def place_sl_exit_for_plan(
    plan: Dict[str, Any],
    *,
    fill_price: Optional[float] = None,
    entry_order_id: Optional[str] = None,
    segment: str = "nifty50",
) -> Dict[str, Any]:
    """Place SL-M SELL after entry fill; register exit trail for T1/T2 SL ratchet."""
    from services.premium_exit_policy import enforce_plan_exits
    from services.v2_indicator_plan import refresh_plan_at_execution

    result: Dict[str, Any] = {
        "sl_order_id": None,
        "gtt_trigger_id": None,
        "errors": [],
        "messages": [],
        "ok": False,
        "trade_plan": None,
        "exit_deferred": False,
    }

    working = dict(plan)
    if fill_price is not None and fill_price > 0:
        working["entry_limit_price"] = float(fill_price)
        working["entry_premium"] = float(fill_price)

    plan = refresh_plan_at_execution(working)
    if fill_price is not None and fill_price > 0:
        plan["entry_limit_price"] = float(fill_price)
        plan["entry_premium"] = float(fill_price)

    entry_limit = float(plan.get("entry_limit_price") or plan.get("entry_premium") or 0)
    plan = enforce_plan_exits(plan, entry=entry_limit)
    entry, sl, t1, t2 = stepped_targets_from_plan(plan)
    plan["target_2_premium"] = t2
    result["trade_plan"] = plan

    symbol = plan.get("tradingsymbol")
    if not symbol:
        result["errors"].append("No tradingsymbol in plan")
        return result

    qty = int(plan.get("quantity") or plan.get("num_lots") or 1)
    if sl <= 0:
        result["errors"].append("Invalid stop_loss_premium")
        return result

    exchange = str(plan.get("exchange") or ("MCX" if segment == "commodity" else "NFO")).upper()
    product = str(plan.get("product") or "NRML")

    if segment == "nifty50":
        from services.v2_constants import resolve_v2_nfo_product

        product = resolve_v2_nfo_product(plan)
    elif segment == "sensex":
        from services.sensex_trade_service import resolve_sensex_bfo_product

        product = resolve_sensex_bfo_product(plan)
    elif segment == "commodity":
        from services.commodity_trade_service import resolve_commodity_product

        product = resolve_commodity_product(plan)

    sl_trig = round_to_tick(sl * 1.002)
    sl_order = place_order_tool.invoke(
        {
            "tradingsymbol": symbol,
            "exchange": exchange,
            "transaction_type": "SELL",
            "quantity": qty,
            "order_type": "SL-M",
            "product": product,
            "trigger_price": sl_trig,
            "segment": segment,
            "skip_session_check": True,
        }
    )

    if sl_order.get("status") != "success":
        result["errors"].append(sl_order.get("error") or "SL order failed")
        return result

    sl_id = str(sl_order.get("order_id"))
    result["sl_order_id"] = sl_id
    result["ok"] = True
    result["messages"].append(
        f"SL-M {sl_id} @ trigger ₹{sl_trig:.2f} (T1 ₹{t1:.2f} → SL entry, T2+ → ratchet SL until exit)"
    )

    if entry_order_id:
        try:
            from services.exit_trail_register import register_from_trade_plan

            register_from_trade_plan(
                plan,
                entry_order_id=str(entry_order_id),
                sl_order_id=sl_id,
                segment=segment,
                fill_price=fill_price,
                paper=False,
                target_2=t2,
            )
        except Exception as exc:
            log_warning(f"Exit trail register failed: {exc}")

    log_info(f"[SLExit] {segment} {symbol} entry={entry_order_id} sl_order={sl_id}")
    return result


def modify_sl_exit_order(
    *,
    sl_order_id: str,
    tradingsymbol: str,
    exchange: str,
    product: str,
    quantity: int,
    new_sl_premium: float,
    segment: str = "nifty50",
) -> bool:
    """Modify live SL-M trigger to a new stop premium."""
    _ = (tradingsymbol, exchange, product, quantity, segment)
    trig = round_to_tick(float(new_sl_premium) * 1.002)
    res = modify_order_tool.invoke(
        {
            "order_id": str(sl_order_id),
            "order_type": "SL-M",
            "trigger_price": trig,
            "quantity": int(quantity),
        }
    )
    if res.get("status") == "success":
        return True
    log_warning(f"[SLExit] modify {sl_order_id} failed: {res.get('error') or res}")
    return False


# Backward-compat shim — do not place GTT.
def place_gtt_for_plan(*args, **kwargs) -> Dict[str, Any]:
    segment = kwargs.pop("segment", "nifty50")
    return place_sl_exit_for_plan(*args, segment=segment, **kwargs)
