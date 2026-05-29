"""Shared types for segment trading agents."""
from __future__ import annotations

from typing import FrozenSet

OPEN_ORDER_STATUSES: FrozenSet[str] = frozenset(
    {
        "OPEN",
        "TRIGGER PENDING",
        "AMO REQ RECEIVED",
        "VALIDATION PENDING",
        "PUT ORDER REQ RECEIVED",
        "OPEN PENDING",
        "OPEN QUEUED",
        "PENDING",
    }
)

FILLED_ORDER_STATUSES: FrozenSet[str] = frozenset({"COMPLETE", "EXECUTED"})

TERMINAL_ENTRY_STATUSES: FrozenSet[str] = frozenset({"CANCELLED", "REJECTED"})


def is_open_order_status(status: str | None) -> bool:
    return (status or "").upper() in OPEN_ORDER_STATUSES


def is_filled_order_status(status: str | None) -> bool:
    return (status or "").upper() in FILLED_ORDER_STATUSES
