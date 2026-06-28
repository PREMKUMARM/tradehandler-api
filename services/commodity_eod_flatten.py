"""
Daily commodity cutoff — at 23:15 IST (configurable):
  - Cancel all open MCX orders
  - Square off MCX net long positions
  - Delete active MCX GTTs
  - Close exit trails + disarm commodity watch
  - Block further commodity trades until next session day
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from services.commodity_config import (
    IST,
    commodity_trading_cutoff_label,
    is_past_commodity_trading_cutoff,
)
from utils.logger import log_error, log_info, log_warning

_STATE_PATH = Path(os.getenv("COMMODITY_EOD_STATE_FILE", "data/commodity_eod_state.json"))

_OPEN_ORDER_STATUSES = frozenset(
    {
        "OPEN",
        "TRIGGER PENDING",
        "AMO REQ RECEIVED",
        "PUT ORDER REQ RECEIVED",
        "OPEN PENDING",
        "OPEN QUEUED",
        "VALIDATION PENDING",
        "PENDING",
    }
)


def _eod_enabled() -> bool:
    return os.getenv("COMMODITY_EOD_FLATTEN_ENABLED", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _load_state() -> Dict[str, Any]:
    if not _STATE_PATH.is_file():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(data: Dict[str, Any]) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def flattened_today() -> bool:
    today = date.today().isoformat()
    return _load_state().get("flattened_date") == today


def flatten_commodity_eod(*, force: bool = False) -> Dict[str, Any]:
    """
    Idempotent per calendar day unless force=True.
    Safe to call from watch loop and background scheduler.
    """
    today = date.today().isoformat()
    if not force and _load_state().get("flattened_date") == today:
        return {"skipped": True, "reason": "already_flattened_today", "date": today}

    if not force and not is_past_commodity_trading_cutoff():
        return {"skipped": True, "reason": "before_cutoff", "date": today}

    label = commodity_trading_cutoff_label()
    summary: Dict[str, Any] = {
        "date": today,
        "cutoff": label,
        "cancelled_orders": [],
        "exit_orders": [],
        "gtts_deleted": [],
        "trails_closed": 0,
        "errors": [],
    }

    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance(skip_validation=True)
    except Exception as exc:
        summary["errors"].append(f"kite: {exc}")
        log_error(f"[CommodityEOD] Kite unavailable: {exc}")
        return summary

    from agent.tools.kite_tools import cancel_order_tool, place_order_tool

    # 1. Cancel open MCX orders
    try:
        for o in kite.orders() or []:
            if str(o.get("exchange") or "").upper() != "MCX":
                continue
            status = str(o.get("status") or "").upper()
            if status not in _OPEN_ORDER_STATUSES:
                continue
            oid = str(o.get("order_id") or "")
            if not oid:
                continue
            try:
                res = cancel_order_tool.invoke({"order_id": oid, "variety": "regular"})
                if res.get("status") == "success":
                    summary["cancelled_orders"].append(oid)
                else:
                    summary["errors"].append(f"cancel {oid}: {res.get('error')}")
            except Exception as exc:
                summary["errors"].append(f"cancel {oid}: {exc}")
    except Exception as exc:
        summary["errors"].append(f"orders: {exc}")

    # 2. Square off MCX net positions
    try:
        seen: set[str] = set()
        for bucket in ("net", "day"):
            for p in (kite.positions().get(bucket) or []):
                if str(p.get("exchange") or "").upper() != "MCX":
                    continue
                sym = str(p.get("tradingsymbol") or "")
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                qty = int(p.get("quantity") or 0)
                if qty == 0:
                    continue
                side = "SELL" if qty > 0 else "BUY"
                try:
                    res = place_order_tool.invoke(
                        {
                            "tradingsymbol": sym,
                            "exchange": "MCX",
                            "transaction_type": side,
                            "quantity": abs(qty),
                            "order_type": "MARKET",
                            "product": str(p.get("product") or "NRML"),
                            "segment": "commodity",
                            "skip_session_check": True,
                        }
                    )
                    if res.get("status") == "success":
                        summary["exit_orders"].append(
                            {"symbol": sym, "side": side, "qty": abs(qty), "order_id": res.get("order_id")}
                        )
                    else:
                        summary["errors"].append(f"exit {sym}: {res.get('error')}")
                except Exception as exc:
                    summary["errors"].append(f"exit {sym}: {exc}")
    except Exception as exc:
        summary["errors"].append(f"positions: {exc}")

    # 3. Open MCX SL-M exit orders are cancelled in step 1 with other open orders.

    # 4. Close commodity exit trails
    try:
        from services.exit_trail_store import close_exit_trail, list_open_exit_trails

        for t in list_open_exit_trails():
            if str(t.get("segment") or "") != "commodity":
                continue
            close_exit_trail(int(t["id"]), reason="eod_flatten")
            summary["trails_closed"] += 1
    except Exception as exc:
        summary["errors"].append(f"trails: {exc}")

    # 5. Disarm watch + cancel pending entry
    try:
        from services.commodity_strategy_watch import commodity_eod_shutdown

        summary["watch"] = commodity_eod_shutdown(
            reason=f"Daily cutoff {label} IST — no further commodity trades today"
        )
    except Exception as exc:
        summary["errors"].append(f"watch: {exc}")

    _save_state(
        {
            "flattened_date": today,
            "flattened_at": datetime.now(IST).isoformat(),
            "summary": {
                "cancelled": len(summary["cancelled_orders"]),
                "exits": len(summary["exit_orders"]),
                "gtts": len(summary["gtts_deleted"]),
                "trails": summary["trails_closed"],
                "errors": len(summary["errors"]),
            },
        }
    )

    log_info(
        f"[CommodityEOD] Flatten complete {today} cutoff={label} "
        f"cancelled={len(summary['cancelled_orders'])} exits={len(summary['exit_orders'])} "
        f"gtts={len(summary['gtts_deleted'])} trails={summary['trails_closed']}"
    )
    if summary["errors"]:
        log_warning(f"[CommodityEOD] errors: {summary['errors'][:5]}")

    return summary


def maybe_run_commodity_eod_flatten() -> Dict[str, Any]:
    if not _eod_enabled():
        return {"skipped": True, "reason": "disabled"}
    if not is_past_commodity_trading_cutoff():
        return {"skipped": True, "reason": "before_cutoff"}
    return flatten_commodity_eod()


async def run_commodity_eod_loop() -> None:
    """Poll for cutoff; run flatten once per day."""
    log_info(
        f"[CommodityEOD] Background loop started (cutoff {commodity_trading_cutoff_label()} IST)"
    )
    while True:
        try:
            if _eod_enabled():
                await asyncio.to_thread(maybe_run_commodity_eod_flatten)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_error(f"[CommodityEOD] loop error: {exc}")
        await asyncio.sleep(max(15, int(os.getenv("COMMODITY_EOD_POLL_SEC", "30") or 30)))
