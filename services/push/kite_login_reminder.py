"""
Weekday Kite / Zerodha login reminder via FCM (AlgoFeast app).

Schedule is read from SQLite (`kite_push_reminder_settings`) when present, else from env.
See `services/push/kite_reminder_config.py`.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, time
from typing import List, Optional
from zoneinfo import ZoneInfo

from database.repositories import get_push_device_repository
from services.push.kite_reminder_config import (
    KiteReminderConfig,
    get_merged_config,
    next_scheduled_after,
    resolve_zone,
)
from services.push.push_service import push_service
from utils.logger import log_info, log_warning, log_error


async def send_kite_reminder_for_all_users(cfg: Optional[KiteReminderConfig] = None) -> None:
    """Send the Kite login reminder to every user_id with registered devices."""
    cfg = cfg or get_merged_config()
    title = cfg.title
    body = cfg.body
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
    log_info("Kite push reminder: background loop started (reads DB/env on each cycle).")
    last_sent: Optional[date] = None
    while True:
        try:
            cfg = get_merged_config()
            if not cfg.enabled:
                await asyncio.sleep(30)
                continue

            zone, _tz_name = resolve_zone(cfg)
            t = time(hour=cfg.hour, minute=cfg.minute)
            now = datetime.now(zone)
            nxt = next_scheduled_after(now, t, zone)
            wait = max(1.0, (nxt - now).total_seconds())
            await asyncio.sleep(wait)

            now2 = datetime.now(zone)
            d = now2.date()
            if d != nxt.date():
                continue
            if d.weekday() >= 5:
                continue
            if last_sent == d:
                continue
            last_sent = d
            await send_kite_reminder_for_all_users(get_merged_config())
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log_error(f"Kite push reminder loop error: {e}")
            await asyncio.sleep(60)
