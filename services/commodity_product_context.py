"""Temporary active MCX product for a request (scan / preview / place)."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from services.commodity_products import McxProduct, resolve_product

_active: ContextVar[Optional[McxProduct]] = ContextVar("commodity_active_product", default=None)


def get_active_product() -> McxProduct:
    prod = _active.get()
    if prod is not None:
        return prod
    return resolve_product(None)


@contextmanager
def use_commodity_product(future_symbol: str) -> Iterator[McxProduct]:
    prod = resolve_product(future_symbol)
    token = _active.set(prod)
    try:
        yield prod
    finally:
        _active.reset(token)
