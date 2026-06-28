"""Register exit trails after GTT or paper entry placement."""
from __future__ import annotations

from typing import Any, Dict, Optional

from services.exit_trail_store import register_exit_trail


def register_from_trade_plan(
    plan: Dict[str, Any],
    *,
    entry_order_id: str,
    gtt_trigger_id: Optional[str] = None,
    sl_order_id: Optional[str] = None,
    segment: str,
    fill_price: Optional[float] = None,
    paper: bool = False,
    target_2: Optional[float] = None,
) -> None:
    symbol = plan.get("tradingsymbol")
    if not symbol or not entry_order_id:
        return
    entry = float(
        fill_price
        or plan.get("entry_limit_price")
        or plan.get("entry_premium")
        or 0
    )
    sl = float(plan.get("stop_loss_premium") or 0)
    tp = float(plan.get("target_premium") or 0)
    t2 = float(target_2 or plan.get("target_2_premium") or 0)
    qty = int(plan.get("quantity") or plan.get("num_lots") or 1)
    exchange = str(plan.get("exchange") or ("MCX" if segment == "commodity" else "NFO"))
    product = str(plan.get("product") or "NRML")

    register_exit_trail(
        segment=segment,
        entry_order_id=entry_order_id,
        gtt_trigger_id=gtt_trigger_id,
        sl_order_id=sl_order_id or gtt_trigger_id,
        tradingsymbol=str(symbol),
        exchange=exchange,
        product=product,
        quantity=qty,
        entry_price=entry,
        stop_loss=sl,
        target=tp,
        target_2=t2,
        paper=paper,
        paper_order_id=entry_order_id if paper else None,
        strategy_id=str(plan.get("strategy_id") or ""),
    )
