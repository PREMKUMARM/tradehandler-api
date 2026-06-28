"""
V2 Nifty50 — 20rupees-strategy only (premium band ₹17–₹23, fixed SL, 1R target).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from services.sensex_run_params import SensexRunParams
from services.sensex_strategy_analysis import (
    FIXED_LOTS,
    FIXED_SL_PREMIUM,
    STRATEGY_DESC,
    STRATEGY_ID,
    STRATEGY_IDS,
    STRATEGY_NAME,
    analyze_fno_strategies as _analyze_20rupees,
)

__all__ = [
    "STRATEGY_ID",
    "STRATEGY_NAME",
    "STRATEGY_DESC",
    "STRATEGY_IDS",
    "FIXED_SL_PREMIUM",
    "FIXED_LOTS",
    "analyze_fno_strategies",
]


def analyze_fno_strategies(
    direction_pref: str = "AUTO",
    margin: float = 0.0,
    hypothesis_note: Optional[str] = None,
    chain_oi: Optional[Dict[str, Any]] = None,
    run_params: Optional[SensexRunParams] = None,
) -> Dict[str, Any]:
    return _analyze_20rupees(
        direction_pref=direction_pref,
        margin=margin,
        hypothesis_note=hypothesis_note,
        chain_oi=chain_oi,
        run_params=run_params,
        segment="nifty50",
    )
