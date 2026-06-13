"""Shared helpers for checklist step parsing (watch + trade services)."""
from __future__ import annotations

from typing import Any, List, Type, TypeVar

T = TypeVar("T")


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def parse_checklist_step(raw: Any, model_cls: Type[T]) -> T:
    """Parse dict/model to ChecklistStepStatus, coercing null completed/server_ok to False."""
    if isinstance(raw, model_cls):
        return raw
    if isinstance(raw, dict):
        data = dict(raw)
        data["completed"] = coerce_bool(data.get("completed"))
        data["server_ok"] = coerce_bool(data.get("server_ok"))
        return model_cls(**data)
    raise TypeError(f"Expected dict or {model_cls.__name__}, got {type(raw)!r}")


def parse_checklist_steps(raw_list: List[Any], model_cls: Type[T]) -> List[T]:
    return [parse_checklist_step(item, model_cls) for item in (raw_list or [])]


def apply_market_closed_gate(
    statuses: List[T],
    *,
    market_open: bool,
    allow_offhours: bool,
    gated_indices: List[int],
    closed_message: str = "Market closed — waits for session",
) -> List[T]:
    """
    When the exchange session is closed, do not mark session-dependent steps as server_ok.
    Keeps preview/connectivity steps (e.g. 0–1) intact.
    """
    if market_open or allow_offhours or not statuses:
        return statuses
    gated = set(gated_indices)
    out: List[T] = []
    for st in statuses:
        idx = getattr(st, "index", None)
        if idx in gated and getattr(st, "server_ok", False):
            if hasattr(st, "model_copy"):
                out.append(
                    st.model_copy(
                        update={
                            "server_ok": False,
                            "completed": False,
                            "message": closed_message,
                        }
                    )
                )
            else:
                out.append(st)
        else:
            out.append(st)
    return out
