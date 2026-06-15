"""Soften trade-plan validation when the exchange session is closed (preview-only)."""
from __future__ import annotations

from typing import Any, Dict, Optional


def soften_validation_for_closed_market(
    validation: Optional[Dict[str, Any]],
    *,
    market_open: bool,
    allow_test_place: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Keep raw risk numbers for review, but mark preview so UI does not block/warn
    before the session opens.
    """
    if not validation or market_open or allow_test_place:
        return validation
    out = dict(validation)
    out["preview_only"] = True
    if not out.get("is_good_trade"):
        out["summary"] = (
            "Preview only — market closed. "
            "Risk/reward checks run again when the session opens."
        )
    else:
        out["summary"] = out.get("summary") or "Preview only — market closed"
    return out
