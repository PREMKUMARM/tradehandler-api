"""GttAgent — OCO trigger prices for Zerodha GTT (NFO / MCX long options)."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from utils.kite_order_utils import round_to_tick


def gtt_triggers_from_plan(plan: Dict[str, Any]) -> Tuple[float, float, float]:
    """OCO trigger prices and last_price for GTT placement."""
    from services.commodity_indicator_plan import _normalize_long_option_exits

    sl_prem = float(plan["stop_loss_premium"])
    tgt_prem = float(plan["target_premium"])
    entry_ref = float(plan.get("entry_limit_price") or plan.get("entry_premium") or 0)
    sl_prem, tgt_prem = _normalize_long_option_exits(entry_ref, sl_prem, tgt_prem)
    last_price = entry_ref
    min_gap = max(0.05, last_price * 0.0026) if last_price > 0 else 0.05

    sl_trigger = round_to_tick(sl_prem * 1.002)
    tp_trigger = round_to_tick(tgt_prem * 0.998)

    if last_price > 0:
        if last_price - sl_trigger < min_gap:
            sl_trigger = round_to_tick(max(0.05, last_price - min_gap))
        if tp_trigger - last_price < min_gap:
            tp_trigger = round_to_tick(last_price + min_gap)
        if sl_trigger >= last_price:
            sl_trigger = round_to_tick(max(0.05, last_price - min_gap))
        if tp_trigger <= last_price:
            tp_trigger = round_to_tick(last_price + min_gap)
    return sl_trigger, tp_trigger, last_price
