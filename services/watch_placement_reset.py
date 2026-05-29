"""Reset autonomous watch daily placement counters when paper journal is cleared."""
from __future__ import annotations

from services.paper_trading import normalize_segment


def reset_watch_placement_for_segment(segment: str) -> None:
    seg = normalize_segment(segment)
    if seg == "nifty50":
        from services.v2_strategy_watch import reset_watch_placement_counters

        reset_watch_placement_counters()
    elif seg == "commodity":
        from services.commodity_strategy_watch import reset_commodity_watch_placement_counters

        reset_commodity_watch_placement_counters()
