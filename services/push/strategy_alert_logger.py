"""
Tiny helper used by every push-only strategy (`nifty_orb_signal`,
`nifty_ema_pullback_signal`, `nifty_pdh_pdl_signal`, ...).

The strategy modules don't need to know about the database schema — they just
hand over the composed message ``{title, body, data}`` and the dispatch
result ``{sent, failed, reason, per_user}`` and we persist a single row in
the ``strategy_alerts`` table for tracking in the UI.

Save failures are swallowed: a missing audit row must never break the actual
push dispatch.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from database.repositories import get_strategy_alert_repository
from utils.logger import log_warning


def save_strategy_alert(
    payload: Dict[str, Any],
    dispatch_result: Optional[Dict[str, Any]] = None,
    *,
    is_test: bool = False,
) -> Optional[int]:
    """Persist one alert and return the row id (or None on failure).

    Args:
        payload: ``{"title": str, "body": str, "data": dict, "signal"?: any}`` -
            the same dict the strategy returns from ``_compose_push``.
        dispatch_result: ``{"sent": int, "failed": int, "reason": str?}`` - the
            value returned by the strategy's ``_dispatch_push`` after sending.
            May be ``None`` if the alert was composed but never dispatched.
        is_test: ``True`` when the alert was triggered from the UI's
            "Send test push" button so it can be filtered out of live history.
    """

    try:
        if not isinstance(payload, dict):
            return None
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            data = {}

        result = dispatch_result or {}
        sent = int(result.get("sent") or 0)
        failed = int(result.get("failed") or 0)
        reason = result.get("reason")
        if reason is not None:
            reason = str(reason)

        repo = get_strategy_alert_repository()
        return repo.save(
            title=str(payload.get("title") or ""),
            body=str(payload.get("body") or ""),
            payload=data,
            push_sent=sent,
            push_failed=failed,
            push_reason=reason,
            is_test=is_test,
        )
    except Exception as e:  # noqa: BLE001
        log_warning(f"[StrategyAlertLogger] save failed: {e}")
        return None
