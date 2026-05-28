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

# If quantity not specified, size notional as % of available USDT margin.
MAX_NOTIONAL_PCT_OF_USDT = float(os.getenv("CRYPTO_MAX_NOTIONAL_PCT", "10") or 10)
MAX_NOTIONAL_PCT_OF_USDT = max(0.0, min(100.0, MAX_NOTIONAL_PCT_OF_USDT))


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
