"""
Auto-trader for push-only strategy signals (ORB, 9-EMA Pullback, PDH/PDL Breakout, ...).

When enabled, each fired strategy signal is translated into a real order:

  * **Paper mode** (``services.paper_trading.is_paper_mode() == True``) → log a synthetic
    paper order via :func:`paper_place_order`, with the SL / target / trailing levels
    attached (the existing :mod:`paper_order_monitor` handles the simulated exit).
  * **Live mode** (paper mode OFF) → place a real Kite REGULAR MARKET BUY for the
    resolved ATM CE/PE contract and, optionally, an SL-M sell + LIMIT sell to bracket
    the entry. All orders go through the existing :func:`risk_gate.check_order_allowed`
    so kill-switch / session-window / daily-cap limits apply.

The push notification still goes out either way; this module is *additive*.

Config (persisted to ``data/strategy_auto_trade.json`` and editable from the UI):

.. code-block:: json

    {
        "enabled": false,
        "place_sl_order": false,
        "place_target_order": false,
        "product": "MIS",
        "strategies": {
            "nifty_15m_orb": true,
            "nifty_9ema_pullback": true,
            "nifty_pdh_pdl_break": true,
            "market_open_gap": false
        }
    }

Master ``enabled`` defaults to **False** for safety — the user must explicitly
opt in from the Operations → Risk Control screen. Even when enabled,
``place_sl_order`` / ``place_target_order`` default to False (entry only).
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

from utils.logger import log_error, log_info, log_warning


_STATE_PATH = Path(os.getenv("STRATEGY_AUTO_TRADE_FILE", "data/strategy_auto_trade.json"))
_lock = RLock()

# All push-only strategies that can be auto-traded; market_open_gap is informational
# (no CE/PE entry levels are computed) so it stays off by default.
_KNOWN_STRATEGIES = (
    "nifty_15m_orb",
    "nifty_9ema_pullback",
    "nifty_pdh_pdl_break",
)

_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "place_sl_order": False,
    "place_target_order": False,
    "product": "MIS",  # MIS / NRML (intraday vs carry); Kite NFO supports both
    "strategies": {s: True for s in _KNOWN_STRATEGIES},
}


# ----------------------------- config persistence -----------------------------


def _env_overrides() -> Dict[str, Any]:
    """Optional env-driven overrides for ops/admin (parallel to PAPER_TRADING_MODE)."""
    out: Dict[str, Any] = {}
    val = os.getenv("STRATEGY_AUTO_TRADE_ENABLED", "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        out["enabled"] = True
    elif val in ("0", "false", "no", "off"):
        out["enabled"] = False
    return out


def _read_state_file() -> Dict[str, Any]:
    if not _STATE_PATH.exists():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        log_warning(f"[AutoTrader] state file read failed: {e}")
        return {}


def get_config() -> Dict[str, Any]:
    """Return the merged config — defaults < persisted < env."""
    with _lock:
        cfg = copy.deepcopy(_DEFAULTS)
        persisted = _read_state_file()
        if isinstance(persisted, dict):
            for k, v in persisted.items():
                if k == "strategies" and isinstance(v, dict):
                    merged = dict(cfg["strategies"])
                    merged.update({sk: bool(sv) for sk, sv in v.items()})
                    cfg["strategies"] = merged
                else:
                    cfg[k] = v
        cfg.update(_env_overrides())
        cfg["env_locks_enabled"] = "enabled" in _env_overrides()
        return cfg


def update_config(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a partial update; returns the new merged config."""
    with _lock:
        existing = _read_state_file() if _STATE_PATH.exists() else {}
        if not isinstance(existing, dict):
            existing = {}
        for k, v in patch.items():
            if k == "strategies" and isinstance(v, dict):
                merged = dict(existing.get("strategies") or {})
                for sk, sv in v.items():
                    merged[str(sk)] = bool(sv)
                existing["strategies"] = merged
            elif k in {"enabled", "place_sl_order", "place_target_order"}:
                existing[k] = bool(v)
            elif k == "product":
                existing[k] = str(v or "MIS").upper()
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        log_info(f"[AutoTrader] config updated: {existing}")
    return get_config()


def is_auto_trade_enabled() -> bool:
    return bool(get_config().get("enabled"))


def is_strategy_enabled(strategy_id: str) -> bool:
    cfg = get_config()
    if not cfg.get("enabled"):
        return False
    return bool((cfg.get("strategies") or {}).get(strategy_id, True))


# ----------------------------- order placement -----------------------------


def _coerce_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if v in (None, ""):
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _coerce_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v in (None, ""):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _option_entry_params(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate a strategy push-data payload into the basic Kite order params."""
    tsym = payload.get("tradingsymbol")
    if not tsym:
        return None
    lot_size = _coerce_int(payload.get("lot_size"), 0) or 0
    num_lots = _coerce_int(payload.get("num_lots"), 0) or 0
    quantity = lot_size * num_lots
    if quantity <= 0:
        return None
    return {
        "exchange": "NFO",
        "tradingsymbol": str(tsym),
        "transaction_type": "BUY",
        "quantity": quantity,
        "lot_size": lot_size,
        "num_lots": num_lots,
        "entry_premium": _coerce_float(payload.get("entry_premium")),
        "sl_premium": _coerce_float(payload.get("sl_premium")),
        "target_premium": _coerce_float(payload.get("target_premium")),
    }


def _place_paper(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    from services.execution_audit import log_execution_audit
    from services.paper_trading import paper_place_order

    p = _option_entry_params(payload)
    if not p:
        return {"placed": False, "skipped_reason": "missing_tradingsymbol_or_qty"}

    paper_payload = {
        "tradingsymbol": p["tradingsymbol"],
        "exchange": "NFO",
        "transaction_type": "BUY",
        "quantity": p["quantity"],
        "order_type": "LIMIT" if p["entry_premium"] else "MARKET",
        "product": cfg.get("product") or "MIS",
        "price": p["entry_premium"],
        "stoploss": p["sl_premium"],
        "target": p["target_premium"],
        "strategy": str(payload.get("strategy") or "unknown"),
        "direction": str(payload.get("direction") or ""),
        "tag": "strategy_auto",
    }
    try:
        order_id = paper_place_order(paper_payload)
        log_execution_audit(
            "STRATEGY_AUTO_PAPER",
            actor="strategy_auto_trader",
            exchange="NFO",
            tradingsymbol=p["tradingsymbol"],
            payload=paper_payload,
            result={"order_id": order_id},
            paper=True,
        )
        log_info(
            f"[AutoTrader] paper order placed: {order_id} {p['tradingsymbol']} "
            f"qty={p['quantity']} entry={p['entry_premium']} sl={p['sl_premium']} "
            f"tgt={p['target_premium']}"
        )
        return {
            "placed": True,
            "mode": "paper",
            "order_id": order_id,
            "tradingsymbol": p["tradingsymbol"],
            "quantity": p["quantity"],
        }
    except Exception as e:  # noqa: BLE001
        log_error(f"[AutoTrader] paper_place_order failed: {e}")
        return {"placed": False, "error": str(e), "mode": "paper"}


def _place_live(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    from kiteconnect.exceptions import KiteException

    from services.execution_audit import log_execution_audit
    from services.risk_gate import check_order_allowed, record_order_placed
    from utils.kite_utils import get_kite_instance

    p = _option_entry_params(payload)
    if not p:
        return {"placed": False, "skipped_reason": "missing_tradingsymbol_or_qty"}

    skip_sess = os.getenv("SKIP_SESSION_CHECK_ON_REST", "").lower() in ("1", "true", "yes")
    invest_inr = (p["entry_premium"] or 0.0) * p["quantity"]
    ok, msg = check_order_allowed(
        "NFO", p["tradingsymbol"], p["quantity"], "BUY", invest_inr, skip_session_check=skip_sess
    )
    if not ok:
        log_warning(f"[AutoTrader] live order blocked by risk gate: {msg}")
        return {"placed": False, "skipped_reason": msg, "mode": "live"}

    try:
        kite = get_kite_instance()
    except Exception as e:  # noqa: BLE001
        log_error(f"[AutoTrader] kite session unavailable: {e}")
        return {"placed": False, "error": f"kite_unavailable: {e}", "mode": "live"}

    product = cfg.get("product") or "MIS"
    entry_params: Dict[str, Any] = {
        "variety": getattr(kite, "VARIETY_REGULAR", "regular"),
        "exchange": "NFO",
        "tradingsymbol": p["tradingsymbol"],
        "transaction_type": "BUY",
        "quantity": p["quantity"],
        "product": product,
        "order_type": getattr(kite, "ORDER_TYPE_MARKET", "MARKET"),
        "validity": getattr(kite, "VALIDITY_DAY", "DAY"),
        "tag": "strat_auto",
    }
    try:
        entry_order_id = kite.place_order(**entry_params)
    except KiteException as e:
        log_error(f"[AutoTrader] Kite entry place_order failed: {e}")
        log_execution_audit(
            "STRATEGY_AUTO_LIVE",
            actor="strategy_auto_trader",
            exchange="NFO",
            tradingsymbol=p["tradingsymbol"],
            payload=entry_params,
            result={"error": str(e)},
            paper=False,
        )
        return {"placed": False, "error": f"kite_error: {e}", "mode": "live"}
    except Exception as e:  # noqa: BLE001
        log_error(f"[AutoTrader] unexpected entry place_order failure: {e}")
        return {"placed": False, "error": str(e), "mode": "live"}

    record_order_placed(invest_inr)
    log_execution_audit(
        "STRATEGY_AUTO_LIVE",
        actor="strategy_auto_trader",
        exchange="NFO",
        tradingsymbol=p["tradingsymbol"],
        payload=entry_params,
        result={"order_id": str(entry_order_id)},
        paper=False,
    )
    log_info(
        f"[AutoTrader] live entry placed: order_id={entry_order_id} "
        f"{p['tradingsymbol']} qty={p['quantity']}"
    )

    out: Dict[str, Any] = {
        "placed": True,
        "mode": "live",
        "order_id": str(entry_order_id),
        "tradingsymbol": p["tradingsymbol"],
        "quantity": p["quantity"],
    }

    # Optional SL leg: SL-M SELL at sl_premium
    if cfg.get("place_sl_order") and p["sl_premium"]:
        sl_params = {
            "variety": getattr(kite, "VARIETY_REGULAR", "regular"),
            "exchange": "NFO",
            "tradingsymbol": p["tradingsymbol"],
            "transaction_type": "SELL",
            "quantity": p["quantity"],
            "product": product,
            "order_type": getattr(kite, "ORDER_TYPE_SLM", "SL-M"),
            "trigger_price": p["sl_premium"],
            "validity": getattr(kite, "VALIDITY_DAY", "DAY"),
            "tag": "strat_sl",
        }
        try:
            sl_oid = kite.place_order(**sl_params)
            out["sl_order_id"] = str(sl_oid)
            log_info(f"[AutoTrader] live SL placed: {sl_oid} trig={p['sl_premium']}")
        except Exception as e:  # noqa: BLE001
            out["sl_error"] = str(e)
            log_warning(f"[AutoTrader] SL placement failed: {e}")

    # Optional target leg: LIMIT SELL at target_premium
    if cfg.get("place_target_order") and p["target_premium"]:
        tgt_params = {
            "variety": getattr(kite, "VARIETY_REGULAR", "regular"),
            "exchange": "NFO",
            "tradingsymbol": p["tradingsymbol"],
            "transaction_type": "SELL",
            "quantity": p["quantity"],
            "product": product,
            "order_type": getattr(kite, "ORDER_TYPE_LIMIT", "LIMIT"),
            "price": p["target_premium"],
            "validity": getattr(kite, "VALIDITY_DAY", "DAY"),
            "tag": "strat_tgt",
        }
        try:
            tgt_oid = kite.place_order(**tgt_params)
            out["target_order_id"] = str(tgt_oid)
            log_info(f"[AutoTrader] live target placed: {tgt_oid} price={p['target_premium']}")
        except Exception as e:  # noqa: BLE001
            out["target_error"] = str(e)
            log_warning(f"[AutoTrader] target placement failed: {e}")

    return out


def _place_sync(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous core; safe to run in :func:`asyncio.to_thread`."""
    from services.paper_trading import is_paper_mode

    strategy = str(payload.get("strategy") or "unknown")
    if not is_strategy_enabled(strategy):
        return {"placed": False, "skipped_reason": "auto_trade_disabled"}

    cfg = get_config()
    if is_paper_mode():
        return _place_paper(payload, cfg)
    return _place_live(payload, cfg)


async def place_strategy_order(
    payload: Dict[str, Any], *, is_test: bool = False
) -> Dict[str, Any]:
    """
    Async-friendly entry point used by each strategy's ``_dispatch_push``.

    Never raises — always returns a dict the caller can stash on the audit row.
    """
    if is_test:
        return {"placed": False, "skipped_reason": "test_push"}
    try:
        return await asyncio.to_thread(_place_sync, payload)
    except Exception as e:  # noqa: BLE001
        log_error(f"[AutoTrader] unexpected place_strategy_order failure: {e}")
        return {"placed": False, "error": str(e)}
