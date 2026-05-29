"""Reset autonomous watch daily placement counters when paper journal is cleared or mode toggles."""
from __future__ import annotations

from services.paper_trading import normalize_segment
from utils.logger import log_info, log_warning


def reset_watch_placement_for_segment(segment: str) -> None:
    seg = normalize_segment(segment)
    if seg == "nifty50":
        from services.v2_strategy_watch import reset_watch_placement_counters

        reset_watch_placement_counters()
    elif seg == "commodity":
        from services.commodity_strategy_watch import reset_commodity_watch_placement_counters

        reset_commodity_watch_placement_counters()


def on_segment_paper_mode_changed(segment: str, was_paper: bool, is_paper: bool) -> None:
    """When user toggles paper/live, clear watch placement state from the prior mode."""
    if was_paper == is_paper:
        return
    seg = normalize_segment(segment)
    try:
        if seg == "commodity":
            from services.commodity_strategy_watch import on_commodity_trading_mode_changed

            on_commodity_trading_mode_changed(is_paper)
        elif seg == "nifty50":
            from services.v2_strategy_watch import reset_watch_placement_counters

            reset_watch_placement_counters()
            log_info(
                f"[V2Watch] Nifty50 {'paper' if is_paper else 'live'} — placement counters reset"
            )
        log_info(
            f"[Watch] segment {seg} mode {'paper' if is_paper else 'live'} "
            f"(was {'paper' if was_paper else 'live'}) — watch counters cleared"
        )
    except Exception as exc:
        log_warning(f"[Watch] mode-change reset failed for {seg}: {exc}")
