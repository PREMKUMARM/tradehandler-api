"""
Merged config for weekday Kite login FCM reminders.

If a row exists in `kite_push_reminder_settings` (id=1), it wins.
Otherwise defaults come from environment variables (see kite_login_reminder).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from database.repositories import get_kite_push_reminder_repository

_DEFAULT_TZ = "Asia/Kolkata"


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        x = int(os.getenv(name, str(default)))
        return max(lo, min(hi, x))
    except ValueError:
        return default


@dataclass(frozen=True)
class KiteReminderConfig:
    enabled: bool
    tz: str
    hour: int
    minute: int
    title: str
    body: str
    from_database: bool


def _defaults_from_env() -> KiteReminderConfig:
    return KiteReminderConfig(
        enabled=_env_bool("KITE_PUSH_REMINDER_ENABLED", True),
        tz=(os.getenv("KITE_PUSH_REMINDER_TZ", _DEFAULT_TZ).strip() or _DEFAULT_TZ),
        hour=_env_int("KITE_PUSH_REMINDER_HOUR", 8, 0, 23),
        minute=_env_int("KITE_PUSH_REMINDER_MINUTE", 0, 0, 59),
        title=os.getenv("KITE_PUSH_REMINDER_TITLE", "Zerodha (Kite) login").strip(),
        body=os.getenv(
            "KITE_PUSH_REMINDER_BODY",
            "Log in to Kite Connect in AlgoFeast so trading stays connected today.",
        ).strip(),
        from_database=False,
    )


def _row_to_config(row: Dict[str, Any]) -> KiteReminderConfig:
    return KiteReminderConfig(
        enabled=bool(row.get("enabled")),
        tz=str(row.get("tz") or _DEFAULT_TZ).strip() or _DEFAULT_TZ,
        hour=max(0, min(23, int(row.get("hour", 8)))),
        minute=max(0, min(59, int(row.get("minute", 0)))),
        title=str(row.get("title") or "Zerodha (Kite) login").strip(),
        body=str(
            row.get("body")
            or "Log in to Kite Connect in AlgoFeast so trading stays connected today."
        ).strip(),
        from_database=True,
    )


def get_merged_config() -> KiteReminderConfig:
    repo = get_kite_push_reminder_repository()
    row = repo.get()
    if row:
        return _row_to_config(row)
    return _defaults_from_env()


def resolve_zone(cfg: KiteReminderConfig) -> Tuple[ZoneInfo, str]:
    try:
        z = ZoneInfo(cfg.tz)
        return z, cfg.tz
    except ZoneInfoNotFoundError:
        z = ZoneInfo(_DEFAULT_TZ)
        return z, _DEFAULT_TZ


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5


def next_scheduled_after(now_aware: datetime, t: time, zone: ZoneInfo) -> datetime:
    """Next Mon–Fri at local time `t` in `zone`, strictly after `now_aware` (any tz)."""
    d = now_aware.astimezone(zone).date()
    for i in range(0, 14):
        day = d + timedelta(days=i)
        if not _is_weekday(day):
            continue
        slot = datetime.combine(day, t, tzinfo=zone)
        if slot > now_aware:
            return slot
    return now_aware + timedelta(days=1)  # fallback


def next_weekday_run(cfg: KiteReminderConfig, now_utc: Optional[datetime] = None) -> Optional[datetime]:
    """
    Next Mon–Fri occurrence of cfg's local time, as aware datetime in that zone.
    Returns None if cfg is disabled.
    """
    if not cfg.enabled:
        return None
    zone, _ = resolve_zone(cfg)
    t = time(hour=cfg.hour, minute=cfg.minute)
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return next_scheduled_after(now, t, zone)


def next_run_payload(cfg: KiteReminderConfig) -> Dict[str, Any]:
    """For API: ISO instants and a human label."""
    nxt = next_weekday_run(cfg)
    if not nxt:
        return {
            "next_run_iso": None,
            "next_run_local_label": None,
            "timezone_effective": cfg.tz,
        }
    zone, effective_tz = resolve_zone(cfg)
    return {
        "next_run_iso": nxt.astimezone(zone).isoformat(),
        "next_run_utc_iso": nxt.astimezone(timezone.utc).isoformat(),
        "next_run_local_label": nxt.strftime("%A %Y-%m-%d %H:%M"),
        "timezone_effective": effective_tz,
    }
