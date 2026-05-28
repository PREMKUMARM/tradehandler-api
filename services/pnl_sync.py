"""Sync daily net P&L from Kite positions into trade_limits (INR loss cap)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from zoneinfo import ZoneInfo

from utils.logger import log_warning
from utils.trade_limits import trade_limits

IST = ZoneInfo("Asia/Kolkata")
_last_sync_at: Optional[datetime] = None


def sync_daily_pnl_from_kite(
    *,
    force: bool = False,
    min_interval_sec: float = 60.0,
) -> Dict[str, Any]:
    """
    Overwrite pnl_inr_today from all Kite net positions (source of truth).
    Always sums every exchange so Nifty + commodity watches do not overwrite each other.
    """
    global _last_sync_at
    now = datetime.now(IST)
    if not force and _last_sync_at is not None:
        elapsed = (now - _last_sync_at).total_seconds()
        if elapsed < max(15.0, min_interval_sec):
            return {
                "ok": True,
                "skipped": True,
                "pnl_inr_today": float(trade_limits.limits.get("pnl_inr_today") or 0),
            }

    try:
        from utils.kite_utils import get_kite_instance

        kite = get_kite_instance()
        positions = kite.positions() or {}
        net = positions.get("net") or []
        total = 0.0
        for p in net:
            if not isinstance(p, dict):
                continue
            total += float(p.get("pnl") or 0)

        trade_limits.set_pnl_inr_today(total)
        _last_sync_at = now
        return {"ok": True, "pnl_inr_today": total, "legs": len(net)}
    except Exception as exc:
        log_warning(f"[PnlSync] failed: {exc}")
        return {"ok": False, "error": str(exc)}


def maybe_sync_pnl_for_watch(_segment: str = "") -> None:
    """Throttled sync used by strategy watch loops (segment ignored — always full book)."""
    sync_daily_pnl_from_kite()
