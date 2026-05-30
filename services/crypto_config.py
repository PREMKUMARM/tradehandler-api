"""Binance BTCUSDT perpetual wizard constants."""
from __future__ import annotations

import os

EXCHANGE = "BINANCE"
SYMBOL = os.getenv("CRYPTO_SYMBOL", "BTCUSDT").strip().upper() or "BTCUSDT"
DEFAULT_LEVERAGE = int(os.getenv("CRYPTO_LEVERAGE", "2") or 2)
DEFAULT_LEVERAGE = max(1, min(125, DEFAULT_LEVERAGE))

# Minimum BTC qty for BTCUSDT perp (exchange may enforce higher via LOT_SIZE)
DEFAULT_QUANTITY_BTC = float(os.getenv("CRYPTO_DEFAULT_QTY_BTC", "0.001") or 0.001)

# Risk defaults (% of allocated margin notional)
DEFAULT_RISK_PCT = float(os.getenv("CRYPTO_RISK_PCT", "1.0") or 1.0)
DEFAULT_REWARD_RATIO = float(os.getenv("CRYPTO_REWARD_RATIO", "2.0") or 2.0)

# Paper: size as % of paper fund. Live: default uses full available USDT as margin.
MAX_NOTIONAL_PCT_OF_USDT = float(os.getenv("CRYPTO_MAX_NOTIONAL_PCT", "10") or 10)
MAX_NOTIONAL_PCT_OF_USDT = max(0.0, min(100.0, MAX_NOTIONAL_PCT_OF_USDT))

# Live only — % of available USDT to commit as margin (100 = use all, e.g. $10 @ 50x → $500 notional).
LIVE_MARGIN_USE_PCT = float(os.getenv("CRYPTO_MARGIN_USE_PCT", "100") or 100)
LIVE_MARGIN_USE_PCT = max(1.0, min(100.0, LIVE_MARGIN_USE_PCT))

MIN_LIVE_USDT_BALANCE = float(os.getenv("CRYPTO_MIN_LIVE_USDT", "15") or 15)

# Live: fixed USDT margin per trade (e.g. $15 @ 50x → $750 notional). 0 = use LIVE_MARGIN_USE_PCT.
LIVE_MARGIN_USDT = float(os.getenv("CRYPTO_LIVE_MARGIN_USDT", "15") or 15)
LIVE_MARGIN_USDT = max(0.0, LIVE_MARGIN_USDT)

# Paper: fixed margin per trade; 0 = use MAX_NOTIONAL_PCT_OF_USDT.
PAPER_MARGIN_USDT = float(os.getenv("CRYPTO_PAPER_MARGIN_USDT", "0") or 0)
PAPER_MARGIN_USDT = max(0.0, PAPER_MARGIN_USDT)

# Autonomous watch — max entries per IST calendar day (paper + live).
CRYPTO_WATCH_MAX_TRADES_PER_DAY = max(
    1, min(100, int(os.getenv("CRYPTO_WATCH_MAX_TRADES_PER_DAY", "20") or 20))
)


def is_crypto_session_open() -> bool:
    """USDT-M perps trade 24/7."""
    return True


def allow_offhours_crypto_place() -> bool:
    return os.getenv("CRYPTO_ALLOW_OFFHOURS_PLACE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _lot_size_min(symbol: str) -> float:
    from utils.binance_order_utils import get_exchange_info_symbol

    info = get_exchange_info_symbol(symbol)
    for f in info.get("filters") or []:
        if f.get("filterType") == "LOT_SIZE":
            try:
                return max(0.0, float(f.get("minQty") or 0))
            except (TypeError, ValueError):
                pass
    return DEFAULT_QUANTITY_BTC


def compute_crypto_quantity(
    usdt_available: float,
    spot: float,
    *,
    paper: bool,
    symbol: str = SYMBOL,
) -> tuple[float, list[str]]:
    """
    Size BTC qty from USDT balance and leverage.
    Live: notional = (available × LIVE_MARGIN_USE_PCT) × leverage (e.g. $10 @ 50x → $500).
    """
    from utils.binance_order_utils import round_quantity

    messages: list[str] = []
    usdt = max(0.0, float(usdt_available or 0))
    lev = DEFAULT_LEVERAGE
    min_qty = _lot_size_min(symbol)

    if paper:
        if PAPER_MARGIN_USDT > 0:
            margin = min(usdt, PAPER_MARGIN_USDT)
            messages.append(
                f"Paper sizing: ${margin:,.2f} margin (target ${PAPER_MARGIN_USDT:,.2f} of ${usdt:,.2f}) @ {lev}x"
            )
        else:
            margin = usdt * (MAX_NOTIONAL_PCT_OF_USDT / 100.0)
            messages.append(
                f"Paper sizing: {MAX_NOTIONAL_PCT_OF_USDT:.0f}% of ${usdt:,.2f} @ {lev}x"
            )
        notional = margin * lev
    else:
        if LIVE_MARGIN_USDT > 0:
            margin = min(usdt, LIVE_MARGIN_USDT)
            messages.append(
                f"Live sizing: ${margin:,.2f} margin (target ${LIVE_MARGIN_USDT:,.2f}, avail ${usdt:,.2f}) "
                f"@ {lev}x → ~${margin * lev:,.2f} notional"
            )
        else:
            margin = usdt * (LIVE_MARGIN_USE_PCT / 100.0)
            messages.append(
                f"Live sizing: ${margin:,.2f} margin ({LIVE_MARGIN_USE_PCT:.0f}% of ${usdt:,.2f}) "
                f"@ {lev}x → ~${margin * lev:,.2f} notional"
            )
        notional = margin * lev

    if spot <= 0 or notional <= 0:
        return 0.0, messages + ["No USDT or spot price for sizing"]

    raw_qty = notional / spot
    qty = round_quantity(symbol, raw_qty)
    if qty <= 0 and raw_qty > 0:
        # round_quantity floors to step — try exchange min lot if margin covers it
        need_margin = (min_qty * spot) / max(1, lev)
        if margin + 0.01 >= need_margin:
            qty = round_quantity(symbol, min_qty)
            messages.append(
                f"Using min lot {qty} BTC (~${qty * spot:,.2f} notional, "
                f"needs ~${need_margin:,.2f} margin @ {lev}x)"
            )
        else:
            messages.append(
                f"Min BTCUSDT lot {min_qty} needs ~${need_margin:,.2f} margin @ {lev}x "
                f"(have ${margin:,.2f} → ~${notional:,.2f} notional)"
            )
            return 0.0, messages

    if qty > 0:
        messages.append(f"Qty {qty} BTC · notional ~${qty * spot:,.2f}")
    return qty, messages
