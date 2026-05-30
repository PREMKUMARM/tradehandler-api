"""Segment metadata and which trading agents apply to each segment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional


@dataclass(frozen=True)
class SegmentAgentProfile:
    """Configuration for a tradable segment (nifty50, commodity, crypto)."""

    id: str
    label: str
    exchange: str
    api_prefix: str
    checklist_steps: int
    kill_switch_key: str
    watch_state_file: str
    agents: FrozenSet[str]
    supports_gtt: bool = True
    supports_pending_invalidation: bool = True
    supports_momentum_trail: bool = True
    supports_eod_flatten: bool = False
    session_label: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


AGENT_CATALOG: Dict[str, str] = {
    "GuardAgent": "Entry quality, duplicate order/position checks",
    "InvalidationAgent": "Cancel pending LIMIT when setup invalidates",
    "ReconcileAgent": "Sync pending entry/GTT with broker state",
    "ExecutionAgent": "Paper vs live place gates (can_execute)",
    "ExitAgent": "Minimum premium SL/TP + momentum trailing",
    "GttAgent": "OCO trigger prices for Zerodha GTT",
    "ReadinessAgent": "Watch status gates and setup phase for UI",
    "PlacementAgent": "Reset watch counters on paper mode toggle",
}

_KITE_OPTION_AGENTS = frozenset(
    {
        "GuardAgent",
        "InvalidationAgent",
        "ReconcileAgent",
        "ExecutionAgent",
        "ExitAgent",
        "GttAgent",
        "ReadinessAgent",
        "PlacementAgent",
    }
)

SEGMENTS: Dict[str, SegmentAgentProfile] = {
    "nifty50": SegmentAgentProfile(
        id="nifty50",
        label="Nifty 50",
        exchange="NFO",
        api_prefix="/api/v1/v2/trade",
        checklist_steps=12,
        kill_switch_key="nifty",
        watch_state_file="data/v2_strategy_watch.json",
        agents=_KITE_OPTION_AGENTS,
        supports_gtt=True,
        supports_pending_invalidation=True,
        supports_momentum_trail=True,
        session_label="NSE 9:15–15:30 IST",
        extra={"min_score_env": "NIFTY_AUTO_MIN_ENTRY_SCORE"},
    ),
    "commodity": SegmentAgentProfile(
        id="commodity",
        label="Commodity (MCX)",
        exchange="MCX",
        api_prefix="/api/v1/commodity/trade",
        checklist_steps=11,
        kill_switch_key="commodity",
        watch_state_file="data/commodity_strategy_watch.json",
        agents=_KITE_OPTION_AGENTS,
        supports_gtt=True,
        supports_pending_invalidation=True,
        supports_momentum_trail=True,
        supports_eod_flatten=True,
        session_label="MCX 9:00–23:30 IST (cutoff 23:15)",
        extra={
            "min_score_env": "COMMODITY_AUTO_MIN_ENTRY_SCORE",
            "max_trades_env": "COMMODITY_WATCH_MAX_TRADES_PER_DAY",
        },
    ),
    "crypto": SegmentAgentProfile(
        id="crypto",
        label="Crypto (Binance)",
        exchange="BINANCE",
        api_prefix="/api/v1/crypto/trade",
        checklist_steps=8,
        kill_switch_key="crypto",
        watch_state_file="data/crypto_strategy_watch.json",
        agents=frozenset({"GuardAgent", "ExecutionAgent", "ReadinessAgent", "PlacementAgent"}),
        supports_gtt=False,
        supports_pending_invalidation=False,
        supports_momentum_trail=False,
        session_label="24/7",
    ),
}


def get_segment(segment_id: str) -> Optional[SegmentAgentProfile]:
    key = (segment_id or "").strip().lower()
    if key in ("nifty", "v2"):
        key = "nifty50"
    return SEGMENTS.get(key)


def list_segments() -> List[SegmentAgentProfile]:
    return list(SEGMENTS.values())


def segment_registry_payload() -> Dict[str, Any]:
    return {
        "agents": AGENT_CATALOG,
        "segments": [
            {
                "id": s.id,
                "label": s.label,
                "exchange": s.exchange,
                "api_prefix": s.api_prefix,
                "checklist_steps": s.checklist_steps,
                "agents": sorted(s.agents),
                "supports_gtt": s.supports_gtt,
                "supports_pending_invalidation": s.supports_pending_invalidation,
                "supports_momentum_trail": s.supports_momentum_trail,
                "supports_eod_flatten": s.supports_eod_flatten,
                "session_label": s.session_label,
                "extra": s.extra,
            }
            for s in list_segments()
        ],
    }
