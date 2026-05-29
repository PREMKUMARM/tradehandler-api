"""
Segment trading agents — reusable logic shared by Nifty50, Commodity, Crypto.

Not to be confused with ``agent/`` (LangGraph LLM chat).
"""
from services.trading_agents.execution_agent import is_paper_trading, resolve_can_execute
from services.trading_agents.exit_agent import (
    broker_tp_for_trail,
    enforce_exits_on_plan,
    min_premium_exits,
    trail_after_target,
)
from services.trading_agents.gtt_agent import gtt_triggers_from_plan
from services.trading_agents.guard_agent import (
    COMMODITY_GUARD,
    GuardAgentConfig,
    NIFTY_GUARD,
    autonomous_place_allowed,
    entry_quality_for_autonomous,
    has_exchange_position,
    has_pending_exchange_order,
    min_entry_confirmation_score,
)
from services.trading_agents.invalidation_agent import (
    is_filled_order_status,
    is_open_order_status,
    pending_entry_invalidated,
    spot_from_plan,
)
from services.trading_agents.placement_agent import (
    on_segment_paper_mode_changed,
    reset_watch_placement_for_segment,
)
from services.trading_agents.readiness_agent import (
    build_readiness_payload,
    describe_autonomous_setup,
)
from services.trading_agents.reconcile_agent import gtt_exists_on_broker, reconcile_pending_watch
from services.trading_agents.segment_registry import (
    AGENT_CATALOG,
    SEGMENTS,
    get_segment,
    list_segments,
    segment_registry_payload,
)

__all__ = [
    "AGENT_CATALOG",
    "COMMODITY_GUARD",
    "GuardAgentConfig",
    "NIFTY_GUARD",
    "SEGMENTS",
    "autonomous_place_allowed",
    "broker_tp_for_trail",
    "build_readiness_payload",
    "describe_autonomous_setup",
    "enforce_exits_on_plan",
    "entry_quality_for_autonomous",
    "get_segment",
    "gtt_exists_on_broker",
    "gtt_triggers_from_plan",
    "has_exchange_position",
    "has_pending_exchange_order",
    "is_filled_order_status",
    "is_open_order_status",
    "is_paper_trading",
    "list_segments",
    "min_entry_confirmation_score",
    "min_premium_exits",
    "on_segment_paper_mode_changed",
    "pending_entry_invalidated",
    "reconcile_pending_watch",
    "reset_watch_placement_for_segment",
    "resolve_can_execute",
    "segment_registry_payload",
    "spot_from_plan",
    "trail_after_target",
]
