"""Human-readable preview of what would be sent to Kite for a trade plan."""
from __future__ import annotations

from typing import Any, Dict, Optional


def build_kite_order_preview(
    plan: Dict[str, Any],
    *,
    paper_mode: bool = False,
    sizing_capital_inr: float = 0.0,
    sizing_capital_source: str = "",
    kite_margin_inr: Optional[float] = None,
) -> Dict[str, Any]:
    sym = str(plan.get("tradingsymbol") or "")
    exchange = str(plan.get("exchange") or "MCX")
    qty = int(plan.get("quantity") or plan.get("num_lots") or 1)
    ls = int(plan.get("lot_size") or 10)
    entry = float(plan.get("entry_limit_price") or plan.get("entry_premium") or 0)
    sl = float(plan.get("stop_loss_premium") or 0)
    tp = float(plan.get("target_premium") or 0)
    product = str(plan.get("product") or "NRML")
    prem_risk = max(0.0, entry - sl) if entry > sl else 0.0
    prem_reward = max(0.0, tp - entry) if tp > entry else 0.0
    risk_inr = float(plan.get("risk_inr") or prem_risk * ls * qty)
    reward_inr = float(plan.get("reward_inr") or prem_reward * ls * qty)
    premium_lock = entry * ls * qty
    qty_sizing = plan.get("qty_sizing") if isinstance(plan.get("qty_sizing"), dict) else {}

    lines = [
        f"{'[PAPER ledger]' if paper_mode else '[LIVE Kite]'} BUY {qty} × {sym} @ LIMIT ₹{entry:.2f}",
        f"Exchange {exchange} · {product} · {ls} bbl/qty · est. premium ₹{premium_lock:,.0f}",
        f"GTT OCO exit — SL ₹{sl:.2f} · TP ₹{tp:.2f} (premium)",
        f"Risk ₹{risk_inr:,.0f} · Reward ₹{reward_inr:,.0f} · R:R {plan.get('reward_ratio', '—')}",
    ]
    if sizing_capital_source:
        lines.append(
            f"Qty sized from {sizing_capital_source}: ₹{sizing_capital_inr:,.0f}"
        )
    if kite_margin_inr is not None and not paper_mode:
        lines.append(f"Kite available margin (placement check): ₹{kite_margin_inr:,.0f}")

    return {
        "mode": "paper" if paper_mode else "live",
        "exchange": exchange,
        "tradingsymbol": sym,
        "transaction_type": "BUY",
        "order_type": "LIMIT",
        "product": product,
        "quantity": qty,
        "lot_size": ls,
        "units_total": qty * ls,
        "limit_price": round(entry, 2),
        "stop_loss_premium": round(sl, 2),
        "target_premium": round(tp, 2),
        "exit_type": "GTT_OCO",
        "estimated_premium_inr": round(premium_lock, 2),
        "risk_inr": round(risk_inr, 2),
        "reward_inr": round(reward_inr, 2),
        "reward_ratio": plan.get("reward_ratio"),
        "sizing_capital_inr": round(float(sizing_capital_inr), 2),
        "sizing_capital_source": sizing_capital_source,
        "kite_margin_inr": round(float(kite_margin_inr), 2) if kite_margin_inr else None,
        "qty_sizing": qty_sizing,
        "summary_lines": lines,
        "summary": " · ".join(lines[:3]),
    }
