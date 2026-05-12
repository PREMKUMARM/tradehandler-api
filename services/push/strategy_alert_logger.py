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


def augment_payload_with_order(
    payload: Dict[str, Any],
    order_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Append broker order info to a composed push payload (``title/body/data``).

    The strategy modules call this *after* :func:`place_strategy_order` so the
    push message visible on the user's phone includes the broker order id and
    mode (paper / live), and so the same fields end up on the audit row.
    """
    if not order_result or not isinstance(payload, dict):
        return payload
    if not order_result.get("placed"):
        # Even when no order was placed, surface skipped_reason in the payload so
        # the UI can show why (kill switch, auto_trade_disabled, etc.).
        reason = order_result.get("skipped_reason") or order_result.get("error")
        if reason:
            new = dict(payload)
            new_data = dict(new.get("data") or {})
            new_data["auto_order_skipped"] = str(reason)
            new["data"] = new_data
            return new
        return payload

    new = dict(payload)
    new_data = dict(new.get("data") or {})
    new_data["order_id"] = str(order_result.get("order_id") or "")
    new_data["order_mode"] = str(order_result.get("mode") or "")
    if order_result.get("sl_order_id"):
        new_data["sl_order_id"] = str(order_result["sl_order_id"])
    if order_result.get("target_order_id"):
        new_data["target_order_id"] = str(order_result["target_order_id"])
    new["data"] = new_data

    # Append a one-liner to the human-visible body so the push itself shows
    # the broker order id.
    suffix = f"\nOrder: {order_result.get('order_id')} ({order_result.get('mode')})"
    new["body"] = (new.get("body") or "") + suffix
    return new


def save_strategy_alert(
    payload: Dict[str, Any],
    dispatch_result: Optional[Dict[str, Any]] = None,
    *,
    is_test: bool = False,
    order_result: Optional[Dict[str, Any]] = None,
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
        order_result: optional output from
            :func:`services.strategy_auto_trader.place_strategy_order` —
            ``{placed, mode, order_id, sl_order_id, target_order_id, error,
            skipped_reason}``. When provided it is persisted alongside the push
            so the UI can show what (if anything) was actually traded.
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
            order_result=order_result,
        )
    except Exception as e:  # noqa: BLE001
        log_warning(f"[StrategyAlertLogger] save failed: {e}")
        return None
