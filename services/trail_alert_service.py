"""Sync alerts for trail/GTT failures (Telegram + FCM push)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from utils.logger import log_error, log_info, log_warning

IST = ZoneInfo("Asia/Kolkata")
_last_alert_key_at: dict[str, float] = {}


def _rate_limit_ok(key: str, cooldown_sec: float) -> bool:
    import time

    now = time.monotonic()
    last = _last_alert_key_at.get(key, 0.0)
    if now - last < cooldown_sec:
        return False
    _last_alert_key_at[key] = now
    return True


def _send_telegram_sync(title: str, body: str) -> bool:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        return False
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    text = f"*{title}*\n\n{body}\n\n_{ts} · vibeFnO trail_"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log_warning(f"[TrailAlert] Telegram failed: {exc}")
        return False


def _send_push_sync(title: str, body: str) -> bool:
    try:
        import asyncio

        from services.push.push_service import push_service

        async def _go() -> bool:
            r = await push_service.send_to_user(
                user_id="default",
                title=title[:120],
                body=body[:240],
                data={"type": "trail_alert"},
            )
            return bool(r)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            loop.create_task(_go())
            return True
        return asyncio.run(_go())
    except Exception as exc:
        log_warning(f"[TrailAlert] push failed: {exc}")
        return False


def alert_trail_issue(
    *,
    title: str,
    message: str,
    dedupe_key: Optional[str] = None,
    cooldown_sec: float = 300.0,
    priority: str = "high",
) -> None:
    """Fire Telegram + push with rate limiting per dedupe_key."""
    key = dedupe_key or title
    if not _rate_limit_ok(key, cooldown_sec):
        return
    log_info(f"[TrailAlert] {title}: {message}")
    tg = _send_telegram_sync(title, message)
    push = _send_push_sync(title, message)
    if not tg and not push:
        log_warning(f"[TrailAlert] no channel delivered for {title}")


def alert_gtt_sync_failed(symbol: str, gtt_id: str, fail_count: int) -> None:
    alert_trail_issue(
        title="GTT trail sync failed",
        message=(
            f"{symbol}: could not update GTT `{gtt_id}` after {fail_count} attempts. "
            "Check Kite token and manage exit manually if needed."
        ),
        dedupe_key=f"gtt_fail:{symbol}:{gtt_id}",
        cooldown_sec=600.0,
    )


def alert_stale_trail(symbol: str, minutes_stale: float) -> None:
    alert_trail_issue(
        title="Trail monitor stale",
        message=(
            f"{symbol}: open trail not updated for {minutes_stale:.0f} min "
            "while position may still be open."
        ),
        dedupe_key=f"stale:{symbol}",
        cooldown_sec=900.0,
    )


def alert_time_stop(symbol: str, reason: str) -> None:
    alert_trail_issue(
        title="Time stop exit",
        message=f"{symbol}: {reason}",
        dedupe_key=f"time_stop:{symbol}",
        cooldown_sec=3600.0,
    )
