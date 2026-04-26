"""
Weekday Kite / Zerodha login reminder via FCM (AlgoFeast app).

Fires every weekday at a configurable local time (default 08:00 Asia/Kolkata).
Sends to every user_id that has at least one row in push_devices.
"""
from __future__ import annotations

import asyncio
import os
from datetime import date, datetime, timedelta, time
from typing import List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from database.repositories import get_push_device_repository
from services.push.push_service import push_service
from utils.logger import log_info, log_warning, log_error

_DEFAULT_TZ = "Asia/Kolkata"


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5  # Mon=0 ... Fri=4


def _parse_time() -> time:
    h = int(os.getenv("KITE_PUSH_REMINDER_HOUR", "8"))
    m = int(os.getenv("KITE_PUSH_REMINDER_MINUTE", "0"))
    return time(hour=max(0, min(23, h)), minute=max(0, min(59, m)))


def _next_scheduled_run(now_aware: datetime, t: time, zone: ZoneInfo) -> datetime:
    """Next datetime strictly after `now_aware` in `zone` on a Mon–Fri at clock time `t`."""
    d = now_aware.astimezone(zone).date()
    for i in range(0, 14):
        day = d + timedelta(days=i)
        if not _is_weekday(day):
            continue
        slot = datetime.combine(day, t, tzinfo=zone)
        if slot > now_aware:
            return slot
    return now_aware + timedelta(days=1)  # fallback


async def _send_kite_reminder_for_all_users() -> None:
    title = os.getenv("KITE_PUSH_REMINDER_TITLE", "Zerodha (Kite) login").strip()
    body = os.getenv(
        "KITE_PUSH_REMINDER_BODY",
        "Log in to Kite Connect in AlgoFeast so trading stays connected today.",
    ).strip()
    # FCM "data" values must be strings
    data = {
        "type": "kite_login_reminder",
        "source": "algofeast_backend",
    }
    if not push_service.configured():
        log_warning("Kite push reminder: FCM not configured; skipping send.")
        return
    repo = get_push_device_repository()
    user_ids: List[str] = repo.list_distinct_user_ids()
    if not user_ids:
        log_info("Kite push reminder: no registered push devices; nothing to send.")
        return
    for uid in user_ids:
        try:
            result = await push_service.send_to_user(
                user_id=uid, title=title, body=body, data=data
            )
            log_info(
                f"Kite push reminder: user_id={uid!r} sent={result.get('sent')} failed={result.get('failed')}"
            )
        except Exception as e:  # noqa: BLE001
            log_error(f"Kite push reminder failed for user_id={uid!r}: {e}")


async def run_kite_login_reminder_loop() -> None:
    if not _env_bool("KITE_PUSH_REMINDER_ENABLED", True):
        log_info("Kite push reminder loop disabled (KITE_PUSH_REMINDER_ENABLED=0).")
        return
    tz_name = os.getenv("KITE_PUSH_REMINDER_TZ", _DEFAULT_TZ).strip() or _DEFAULT_TZ
    try:
        zone = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        log_error(f"Kite push reminder: invalid KITE_PUSH_REMINDER_TZ={tz_name!r}, using {_DEFAULT_TZ!r}.")
        zone = ZoneInfo(_DEFAULT_TZ)
    clock = _parse_time()
    last_sent: Optional[date] = None
    log_info(f"Kite push reminder: weekdays at {clock.strftime('%H:%M')} {tz_name} (all push user_ids).")
    while True:
        try:
            now = datetime.now(zone)
            nxt = _next_scheduled_run(now, clock, zone)
            wait = max(1.0, (nxt - now).total_seconds())
            await asyncio.sleep(wait)
            now2 = datetime.now(zone)
            d = now2.date()
            if d != nxt.date():
                # DST / clock change edge case: realign on next loop
                continue
            if not _is_weekday(d):
                continue
            if last_sent == d:
                continue
            last_sent = d
            await _send_kite_reminder_for_all_users()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log_error(f"Kite push reminder loop error: {e}")
            await asyncio.sleep(60)
