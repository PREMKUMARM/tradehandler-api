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
