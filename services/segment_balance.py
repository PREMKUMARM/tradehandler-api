"""Live balance snapshots pushed over segment WebSocket."""
from __future__ import annotations

from typing import Any, Dict


def get_crypto_balance_payload() -> Dict[str, Any]:
    from services.crypto_config import SYMBOL
    from services.crypto_live_indicators import recalculate_from_ticker
    from utils.binance_order_utils import get_binance_credentials, get_usdt_balance

    live: Dict[str, Any] = {}
    try:
        live = recalculate_from_ticker()
    except Exception:
        pass
    spot = float(live.get("btc_spot") or 0)

    try:
        get_binance_credentials()
        balance = float(get_usdt_balance())
        return {
            "broker": "binance",
            "currency": "USDT",
            "connected": True,
            "usdt_available": balance,
            "btc_spot": spot,
            "symbol": SYMBOL,
        }
    except Exception as exc:
        return {
            "broker": "binance",
            "currency": "USDT",
            "connected": False,
            "usdt_available": 0.0,
            "btc_spot": spot,
            "symbol": SYMBOL,
            "message": str(exc)[:200],
        }


def get_kite_balance_payload() -> Dict[str, Any]:
    try:
        from utils.kite_utils import get_kite_instance
        from utils.margin_utils import parse_equity_margins

        kite = get_kite_instance(user_id="default")
        margins = kite.margins()
        equity = margins.get("equity", {}) or {}
        available, utilised, total = parse_equity_margins(equity)
        return {
            "broker": "kite",
            "currency": "INR",
            "connected": True,
            "available_margin": float(available or 0),
            "utilised_margin": float(utilised or 0),
            "total_margin": float(total or 0),
        }
    except Exception as exc:
        return {
            "broker": "kite",
            "currency": "INR",
            "connected": False,
            "available_margin": 0.0,
            "utilised_margin": 0.0,
            "total_margin": 0.0,
            "message": str(exc)[:200],
        }


def get_segment_balance(segment: str) -> Dict[str, Any]:
    seg = (segment or "").strip().lower()
    if seg in ("crypto", "binance", "btc"):
        return get_crypto_balance_payload()
    return get_kite_balance_payload()


def is_kite_broker_connected() -> bool:
    """True when Kite token + margins API succeed (same signal as header balance)."""
    return bool(get_kite_balance_payload().get("connected"))
