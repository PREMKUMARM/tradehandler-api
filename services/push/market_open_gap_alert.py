"""
NIFTY 50 market-open GAP alert via FCM push.

At market open on every weekday (default 09:15:30 IST), this service:
1. Fetches NIFTY 50 quote (today's open + previous trading day's close).
2. Computes gap in points and percentage.
3. Pulls last few daily candles to derive a 5-day ATR and use it as a
   reasonable day-range estimate.
4. Classifies the gap (flat / mild / moderate / strong) and composes a short
   "possible movement" heuristic line.
5. Fans the resulting push out to every user with a registered FCM device.

Knobs (env vars, all optional):
  MARKET_OPEN_GAP_ALERT_ENABLED          (default "1")
  MARKET_OPEN_GAP_ALERT_TZ               (default "Asia/Kolkata")
  MARKET_OPEN_GAP_ALERT_HOUR             (default 9)
  MARKET_OPEN_GAP_ALERT_MINUTE           (default 15)
  MARKET_OPEN_GAP_ALERT_DELAY_SEC        (default 30)   wait after open for first ticks
  MARKET_OPEN_GAP_ALERT_INSTRUMENT_TOKEN (default 256265, NIFTY 50)
  MARKET_OPEN_GAP_ALERT_SYMBOL           (default "NSE:NIFTY 50")
  MARKET_OPEN_GAP_ALERT_FLAT_PCT         (default 0.10)
  MARKET_OPEN_GAP_ALERT_MILD_PCT         (default 0.30)
  MARKET_OPEN_GAP_ALERT_MODERATE_PCT     (default 0.70)
  MARKET_OPEN_GAP_ALERT_ATR_DAYS         (default 5)
  MARKET_OPEN_GAP_ALERT_FETCH_RETRY      (default 6)    each retry sleeps 5s
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from database.repositories import get_push_device_repository
from services.push.kite_reminder_config import next_scheduled_after
from services.push.push_service import push_service
from utils.kite_utils import get_kite_instance
from utils.logger import log_error, log_info, log_warning


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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _resolve_zone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo(_DEFAULT_TZ)


def _fmt_pts(x: float) -> str:
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.1f}"


def _fmt_pct(x: float) -> str:
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2f}%"


def _fmt_level(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.2f}"


@dataclass(frozen=True)
class GapResult:
    instrument_symbol: str
    open_price: float
    prev_close: float
    last_price: float
    gap_points: float
    gap_pct: float
    direction: str  # "GAP UP" | "GAP DOWN" | "FLAT"
    severity: str   # "flat" | "mild" | "moderate" | "strong"
    atr_value: float
    atr_days: int
    estimated_low: float
    estimated_high: float
    title: str
    body: str
    data: Dict[str, str]


def _classify_severity(gap_pct: float) -> str:
    flat = _env_float("MARKET_OPEN_GAP_ALERT_FLAT_PCT", 0.10)
    mild = _env_float("MARKET_OPEN_GAP_ALERT_MILD_PCT", 0.30)
    moderate = _env_float("MARKET_OPEN_GAP_ALERT_MODERATE_PCT", 0.70)
    a = abs(gap_pct)
    if a < flat:
        return "flat"
    if a < mild:
        return "mild"
    if a < moderate:
        return "moderate"
    return "strong"


def _movement_hint(severity: str, direction: str, prev_close: float) -> str:
    """
    Heuristic prose for the "possible movement" line. Intentionally cautious —
    this is pattern guidance, not a prediction.
    """
    if severity == "flat":
        return "Flat open — likely range-bound for the first 15-30 min; wait for direction."
    if severity == "mild":
        return (
            f"Mild gap — often fills early; watch for retest of prev close "
            f"({_fmt_level(prev_close)})."
        )
    if severity == "moderate":
        if direction == "GAP UP":
            return (
                "Moderate gap up — watch the first 15-30 min: hold above open ⇒ "
                "trend continuation; failure to hold ⇒ gap-fill toward prev close."
            )
        if direction == "GAP DOWN":
            return (
                "Moderate gap down — watch the first 15-30 min: hold below open ⇒ "
                "trend continuation; reclaim of open ⇒ gap-fill bounce toward prev close."
            )
        return "Moderate gap — watch the first 15-30 min for direction."
    # strong
    if direction == "GAP UP":
        return (
            "Strong gap up — trend-day risk on the upside; sharp reversal possible "
            "if open isn't held in the first 30 min."
        )
    if direction == "GAP DOWN":
        return (
            "Strong gap down — trend-day risk on the downside; squeeze possible "
            "if prev close is reclaimed in the first 30 min."
        )
    return "Strong gap — expect elevated volatility through the session."


def _compute_atr(daily_candles: List[Dict[str, Any]], days: int) -> float:
    """
    Simple ATR (average true range) over the most recent `days` *completed* daily
    candles. We exclude today's bar (it's still forming at 9:15).
    """
    if not daily_candles:
        return 0.0
    today = datetime.now(_resolve_zone(os.getenv("MARKET_OPEN_GAP_ALERT_TZ", _DEFAULT_TZ))).date()
    completed: List[Dict[str, Any]] = []
    for c in daily_candles:
        d = c.get("date")
        if isinstance(d, datetime):
            cd = d.date()
        elif isinstance(d, date):
            cd = d
        else:
            cd = None
        if cd is None or cd >= today:
            continue
        completed.append(c)
    if not completed:
        return 0.0
    completed.sort(key=lambda c: c["date"])
    last = completed[-days:]
    trs: List[float] = []
    prev_close: Optional[float] = None
    for c in last:
        h = float(c.get("high") or 0.0)
        lo = float(c.get("low") or 0.0)
        cl = float(c.get("close") or 0.0)
        if prev_close is None:
            tr = h - lo
        else:
            tr = max(h - lo, abs(h - prev_close), abs(lo - prev_close))
        if tr > 0:
            trs.append(tr)
        prev_close = cl
    if not trs:
        return 0.0
    return sum(trs) / len(trs)


def _build_message(g: Dict[str, Any]) -> GapResult:
    """Compose a `GapResult` from the raw numeric inputs in `g`."""
    open_price = float(g["open"])
    prev_close = float(g["prev_close"])
    last_price = float(g.get("last_price") or open_price)
    atr = float(g.get("atr") or 0.0)
    atr_days = int(g.get("atr_days") or 0)
    symbol = str(g.get("symbol") or "NSE:NIFTY 50")

    gap_pts = open_price - prev_close
    gap_pct = (gap_pts / prev_close * 100.0) if prev_close > 0 else 0.0
    direction = "GAP UP" if gap_pts > 0 else "GAP DOWN" if gap_pts < 0 else "FLAT"
    severity = _classify_severity(gap_pct)

    if atr > 0:
        est_high = open_price + atr
        est_low = open_price - atr
    else:
        est_high = open_price
        est_low = open_price

    short_sym = symbol.split(":", 1)[-1].strip() or symbol
    title = f"{short_sym} {direction}: {_fmt_pts(gap_pts)} pts ({_fmt_pct(gap_pct)})"

    lines = [
        f"Open {_fmt_level(open_price)} | Prev close {_fmt_level(prev_close)}",
        _movement_hint(severity, direction, prev_close),
    ]
    if atr > 0:
        lines.append(
            f"Day range est: {_fmt_level(est_low)} – {_fmt_level(est_high)} "
            f"(ATR{atr_days}={atr:.0f})"
        )
    body = "\n".join(lines)

    data: Dict[str, str] = {
        "type": "market_open_gap_alert",
        "source": "algofeast_backend",
        "instrument": short_sym,
        "direction": direction,
        "severity": severity,
        "open": f"{open_price:.2f}",
        "prev_close": f"{prev_close:.2f}",
        "last_price": f"{last_price:.2f}",
        "gap_points": f"{gap_pts:.2f}",
        "gap_pct": f"{gap_pct:.4f}",
        "atr": f"{atr:.2f}",
        "atr_days": str(atr_days),
        "estimated_low": f"{est_low:.2f}",
        "estimated_high": f"{est_high:.2f}",
    }

    return GapResult(
        instrument_symbol=symbol,
        open_price=open_price,
        prev_close=prev_close,
        last_price=last_price,
        gap_points=gap_pts,
        gap_pct=gap_pct,
        direction=direction,
        severity=severity,
        atr_value=atr,
        atr_days=atr_days,
        estimated_low=est_low,
        estimated_high=est_high,
        title=title,
        body=body,
        data=data,
    )


def _fetch_gap_inputs() -> Optional[Dict[str, Any]]:
    """
    Pull today's open + prev close + last price via `kite.quote(symbol)`, and
    fetch the last few daily candles for ATR. Returns None if the open hasn't
    printed yet (e.g. exchange holiday or quote not ready).

    NOTE: kite.quote()'s `ohlc.close` is the *previous trading day's* close,
    which is exactly what we want here.
    """
    symbol = os.getenv("MARKET_OPEN_GAP_ALERT_SYMBOL", "NSE:NIFTY 50").strip() or "NSE:NIFTY 50"
    token = int(os.getenv("MARKET_OPEN_GAP_ALERT_INSTRUMENT_TOKEN", "256265"))
    atr_days = _env_int("MARKET_OPEN_GAP_ALERT_ATR_DAYS", 5, 1, 30)

    try:
        kite = get_kite_instance()
    except Exception as e:  # noqa: BLE001
        log_warning(f"[GapAlert] Kite client unavailable: {e}")
        return None

    try:
        quotes = kite.quote(symbol) or {}
        q = quotes.get(symbol) or {}
    except Exception as e:  # noqa: BLE001
        log_warning(f"[GapAlert] kite.quote('{symbol}') failed: {e}")
        return None

    ohlc = q.get("ohlc") or {}
    try:
        open_price = float(ohlc.get("open") or 0.0)
        prev_close = float(ohlc.get("close") or 0.0)
        last_price = float(q.get("last_price") or 0.0)
    except (TypeError, ValueError):
        return None

    if open_price <= 0 or prev_close <= 0:
        log_warning(
            f"[GapAlert] quote payload missing open/prev-close (open={open_price}, "
            f"prev_close={prev_close}); likely a holiday or quote not yet ready."
        )
        return None

    atr = 0.0
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=max(atr_days * 3, 10))
        bars = kite.historical_data(
            instrument_token=token,
            from_date=start_date,
            to_date=end_date,
            interval="day",
        )
        atr = _compute_atr(bars or [], atr_days)
    except Exception as e:  # noqa: BLE001
        log_warning(f"[GapAlert] historical_data for ATR failed (using 0): {e}")

    return {
        "symbol": symbol,
        "open": open_price,
        "prev_close": prev_close,
        "last_price": last_price,
        "atr": atr,
        "atr_days": atr_days,
    }


async def compute_gap_alert(*, retries: Optional[int] = None) -> Optional[GapResult]:
    """
    Resolve the current NIFTY gap and return a `GapResult` ready to be sent.
    Retries a few times to give the exchange feed a moment to publish the open.
    Returns None on holidays / when the open is unavailable.
    """
    n = retries if retries is not None else _env_int("MARKET_OPEN_GAP_ALERT_FETCH_RETRY", 6, 1, 30)
    for attempt in range(n):
        inputs = await asyncio.to_thread(_fetch_gap_inputs)
        if inputs:
            return _build_message(inputs)
        await asyncio.sleep(5)
    return None


async def send_market_open_gap_alert(*, retries: Optional[int] = None) -> Dict[str, Any]:
    """
    Compose + fan-out the market-open gap push to every registered user.
    Safe to call manually (e.g. from the /test endpoint).
    """
    if not push_service.configured():
        log_warning("[GapAlert] FCM not configured; skipping send.")
        return {"sent": 0, "failed": 0, "reason": "fcm_not_configured"}

    result = await compute_gap_alert(retries=retries)
    if result is None:
        log_info("[GapAlert] No gap result available (likely holiday / no quote).")
        return {"sent": 0, "failed": 0, "reason": "no_quote"}

    repo = get_push_device_repository()
    user_ids: List[str] = repo.list_distinct_user_ids()
    if not user_ids:
        log_info("[GapAlert] No registered push devices; nothing to send.")
        return {"sent": 0, "failed": 0, "reason": "no_devices", "result": result.data}

    totals: Dict[str, Any] = {"sent": 0, "failed": 0, "per_user": {}, "result": result.data}
    for uid in user_ids:
        try:
            r = await push_service.send_to_user(
                user_id=uid,
                title=result.title,
                body=result.body,
                data=result.data,
            )
            totals["per_user"][uid] = r
            totals["sent"] += int(r.get("sent", 0))
            totals["failed"] += int(r.get("failed", 0))
        except Exception as e:  # noqa: BLE001
            log_error(f"[GapAlert] send failed for user_id={uid!r}: {e}")
            totals["failed"] += 1
            totals["per_user"][uid] = {"sent": 0, "failed": 1, "errors": [str(e)]}
    log_info(
        f"[GapAlert] dispatched: users={len(user_ids)} sent={totals['sent']} "
        f"failed={totals['failed']} title={result.title!r}"
    )
    return totals


def _schedule_config() -> Dict[str, Any]:
    return {
        "enabled": _env_bool("MARKET_OPEN_GAP_ALERT_ENABLED", True),
        "tz": (os.getenv("MARKET_OPEN_GAP_ALERT_TZ", _DEFAULT_TZ).strip() or _DEFAULT_TZ),
        "hour": _env_int("MARKET_OPEN_GAP_ALERT_HOUR", 9, 0, 23),
        "minute": _env_int("MARKET_OPEN_GAP_ALERT_MINUTE", 15, 0, 59),
        "delay_sec": _env_int("MARKET_OPEN_GAP_ALERT_DELAY_SEC", 30, 0, 600),
    }


def next_run_info() -> Dict[str, Any]:
    cfg = _schedule_config()
    zone = _resolve_zone(cfg["tz"])
    t = time(hour=cfg["hour"], minute=cfg["minute"])
    now = datetime.now(zone)
    nxt = next_scheduled_after(now, t, zone)
    return {
        "enabled": cfg["enabled"],
        "tz": cfg["tz"],
        "scheduled_local": f"{cfg['hour']:02d}:{cfg['minute']:02d}",
        "delay_sec": cfg["delay_sec"],
        "next_run_iso": nxt.isoformat(),
        "next_run_utc_iso": nxt.astimezone(timezone.utc).isoformat(),
        "next_run_local_label": nxt.strftime("%A %Y-%m-%d %H:%M"),
        "fcm_configured": push_service.configured(),
    }


async def run_market_open_gap_alert_loop() -> None:
    """
    Background loop: sleeps until the next weekday open and fires once per day.
    """
    log_info("[GapAlert] market-open gap alert loop started.")
    last_sent: Optional[date] = None
    while True:
        try:
            cfg = _schedule_config()
            if not cfg["enabled"]:
                await asyncio.sleep(30)
                continue

            zone = _resolve_zone(cfg["tz"])
            t = time(hour=cfg["hour"], minute=cfg["minute"])
            now = datetime.now(zone)
            nxt = next_scheduled_after(now, t, zone)
            wait_sec = max(1.0, (nxt - now).total_seconds()) + max(0, cfg["delay_sec"])
            await asyncio.sleep(wait_sec)

            now2 = datetime.now(zone)
            d = now2.date()
            # only weekdays; once per day
            if d.weekday() >= 5:
                continue
            if last_sent == d:
                continue
            last_sent = d

            await send_market_open_gap_alert()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log_error(f"[GapAlert] loop error: {e}")
            await asyncio.sleep(60)
