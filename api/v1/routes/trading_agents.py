"""Trading segment registry and shared agent metadata."""
from __future__ import annotations

from fastapi import APIRouter

from services.trading_agents import get_segment, segment_registry_payload

router = APIRouter(prefix="/trading", tags=["Trading Agents"])


@router.get("/segments")
def list_trading_segments():
    """Segments (nifty50, commodity, crypto) and which trading agents each uses."""
    return {"data": segment_registry_payload()}


@router.get("/segments/{segment_id}")
def get_trading_segment(segment_id: str):
    profile = get_segment(segment_id)
    if not profile:
        return {"data": None, "error": "Unknown segment"}
    payload = segment_registry_payload()
    match = next((s for s in payload["segments"] if s["id"] == profile.id), None)
    return {"data": match}
