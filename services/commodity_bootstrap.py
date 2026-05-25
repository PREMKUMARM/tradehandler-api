"""Resolve nearest CRUDEOILM FUT on startup and log active contract."""
from __future__ import annotations

from services.commodity_product_context import use_commodity_product
from services.commodity_products import DEFAULT_FUTURE_SYMBOL, nearest_listed_future, resolve_product
from utils.kite_utils import get_kite_instance
from utils.logger import log_info, log_warning


def bootstrap_crude_mini_product() -> str:
    """Pick listed CRUDEOILM month; returns active future_symbol."""
    try:
        rows = get_kite_instance().instruments("MCX")
        listed = nearest_listed_future(rows)
        sym = listed or DEFAULT_FUTURE_SYMBOL
        with use_commodity_product(sym):
            prod = resolve_product(sym)
            log_info(
                f"[Commodity] Active product CRUDEOILM only: {prod.future_symbol} "
                f"prefix={prod.option_prefix} units/lot={prod.units_per_lot}"
            )
        return sym
    except Exception as exc:
        log_warning(f"[Commodity] bootstrap CRUDEOILM: {exc}")
        return DEFAULT_FUTURE_SYMBOL
