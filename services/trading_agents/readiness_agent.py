"""ReadinessAgent — watch status gates and setup phase for segment UI."""
from __future__ import annotations

from services.watch_readiness import build_readiness_payload
from services.watch_setup_status import describe_autonomous_setup

__all__ = ["build_readiness_payload", "describe_autonomous_setup"]
