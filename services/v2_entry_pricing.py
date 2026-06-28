"""
V2 — 20rupees-strategy entry only (patient LIMIT inside premium band).
"""
from __future__ import annotations

from typing import Any, Dict

from services.sensex_entry_pricing import EntryAnalysis, compute_strategy_entry as _compute
from services.sensex_strategy_analysis import STRATEGY_ID

__all__ = ["EntryAnalysis", "STRATEGY_ID", "compute_strategy_entry"]


def compute_strategy_entry(
    *,
    strategy_id: str,
    option_kind: str,
    quote: Dict[str, float],
    spot: float,
    strike: int,
    delta: float,
    intra: Dict[str, Any],
    prev_close: float = 0.0,
) -> EntryAnalysis:
    return _compute(
        strategy_id=STRATEGY_ID,
        option_kind=option_kind,
        quote=quote,
        spot=spot,
        strike=strike,
        delta=delta,
        intra=intra,
        prev_close=prev_close,
        segment="nifty50",
    )
