"""
NIFTY 50 — Previous-Day High / Previous-Day Low (PDH/PDL) breakout signal.

Strategy:
  * On each new trading day, look up yesterday's daily candle and store PDH/PDL.
  * After warm-up (default 09:30 IST), on every 5m candle close:
      LONG : close > PDH AND body/range >= NIFTY_PDH_PDL_MIN_BODY_RATIO
      SHORT: close < PDL AND body/range >= NIFTY_PDH_PDL_MIN_BODY_RATIO
  * SL: PDH (long) / PDL (short) ± buffer.
  * Target: RR × |entry - SL| (default 1.5).
  * Once per direction per day (so you can still get a SHORT later in the day
    if PDH bounce fails — but each direction only fires once).
  * Cutoff (default 14:30 IST).

The push includes a real ATM CE/PE contract and live entry/SL/target premiums.
No order is placed.
"""
from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Set

from zoneinfo import ZoneInfo

from database.repositories import get_push_device_repository
from services.push.option_contract_resolver import (
    OptionLegEstimate,
    estimate_option_leg,
)
from services.push.push_service import push_service
from services.push.strategy_alert_logger import save_strategy_alert
from utils.kite_utils import get_kite_instance
from utils.logger import log_error, log_info, log_warning


IST = ZoneInfo("Asia/Kolkata")


# ----------------------------- env helpers -----------------------------


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        x = int(os.getenv(name, str(default)))
        return max(lo, min(hi, x))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ----------------------------- config -----------------------------


@dataclass(frozen=True)
class PdhPdlConfig:
    enabled: bool
    warmup: time
    cutoff: time
    rr_ratio: float
    sl_buffer_pts: float
    min_body_ratio: float
    atm_step: int
    atm_delta: float
    lot_size: int
    num_lots: int
    instrument_token: int
    instrument_symbol: str
    user_id_override: Optional[str]


def _load_config() -> PdhPdlConfig:
    return PdhPdlConfig(
        enabled=_env_bool("NIFTY_PDH_PDL_SIGNAL_ENABLED", True),
        warmup=time(
            hour=_env_int("NIFTY_PDH_PDL_WARMUP_HOUR", 9, 0, 23),
            minute=_env_int("NIFTY_PDH_PDL_WARMUP_MINUTE", 30, 0, 59),
        ),
        cutoff=time(
            hour=_env_int("NIFTY_PDH_PDL_CUTOFF_HOUR", 14, 0, 23),
            minute=_env_int("NIFTY_PDH_PDL_CUTOFF_MINUTE", 30, 0, 59),
        ),
        rr_ratio=max(0.5, _env_float("NIFTY_PDH_PDL_RR_RATIO", 1.5)),
        sl_buffer_pts=max(0.0, _env_float("NIFTY_PDH_PDL_SL_BUFFER_POINTS", 3.0)),
        min_body_ratio=min(0.95, max(0.0, _env_float("NIFTY_PDH_PDL_MIN_BODY_RATIO", 0.4))),
        atm_step=_env_int("NIFTY_PDH_PDL_ATM_STEP", 50, 5, 500),
        atm_delta=min(0.99, max(0.05, _env_float("NIFTY_PDH_PDL_ATM_DELTA", 0.5))),
        lot_size=_env_int("NIFTY_PDH_PDL_LOT_SIZE", 75, 1, 10000),
        num_lots=_env_int("NIFTY_PDH_PDL_NUM_LOTS", 1, 1, 1000),
        instrument_token=_env_int(
            "NIFTY_PDH_PDL_INSTRUMENT_TOKEN", 256265, 1, 2_000_000_000
        ),
        instrument_symbol=(
            os.getenv("NIFTY_PDH_PDL_INSTRUMENT_SYMBOL", "NSE:NIFTY 50").strip()
            or "NSE:NIFTY 50"
        ),
        user_id_override=(os.getenv("NIFTY_PDH_PDL_USER_ID") or "").strip() or None,
    )


# ----------------------------- formatting helpers -----------------------------


def _fmt_level(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.2f}"


def _fmt_inr(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}₹{abs(x):,.0f}"


def _floor_5m(dt: datetime) -> datetime:
    dt = dt.astimezone(IST)
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def _tick_time(t: Dict[str, Any]) -> Optional[datetime]:
    ts = t.get("exchange_timestamp") or t.get("last_trade_time")
    if hasattr(ts, "astimezone"):
        return ts.astimezone(IST)
    return None


# ----------------------------- core state -----------------------------


@dataclass
class Candle5m:
    start: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class DayState:
    day: Optional[date] = None
    pdh: Optional[float] = None
    pdl: Optional[float] = None
    pdh_pdl_loaded: bool = False
    pdh_pdl_loading: bool = False
    pdh_pdl_error: Optional[str] = None
    cur_candle: Optional[Candle5m] = None
    closed_candles: int = 0
    last_price: Optional[float] = None
    fired_directions: Set[str] = field(default_factory=set)


@dataclass
class PdhPdlSignal:
    direction: str
    entry: float
    stop_loss: float
    target: float
    risk_points: float
    reward_points: float
    rr_ratio: float
    pdh: float
    pdl: float
    breakout_candle_start: datetime
    breakout_candle_close: float
    body_ratio: float
    atm_strike: int
    option_kind: str
    option_leg: OptionLegEstimate
    lot_size: int
    num_lots: int


_state = DayState()
_state_lock = threading.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None
_registered = False


# ----------------------------- helpers -----------------------------


def _ensure_day_state(now_ist: datetime) -> bool:
    today = now_ist.astimezone(IST).date()
    new_day = _state.day != today
    if new_day:
        _state.day = today
        _state.pdh = None
        _state.pdl = None
        _state.pdh_pdl_loaded = False
        _state.pdh_pdl_loading = False
        _state.pdh_pdl_error = None
        _state.cur_candle = None
        _state.closed_candles = 0
        _state.last_price = None
        _state.fired_directions = set()
    return new_day


def _round_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)


def _fetch_pdh_pdl_sync(cfg: PdhPdlConfig) -> Optional[Dict[str, float]]:
    """
    Pull yesterday's high/low via kite.historical_data() daily candles.
    """
    try:
        kite = get_kite_instance()
        end_date = date.today()
        start_date = end_date - timedelta(days=10)
        bars = kite.historical_data(
            instrument_token=cfg.instrument_token,
            from_date=start_date,
            to_date=end_date,
            interval="day",
        ) or []
    except Exception as e:  # noqa: BLE001
        log_warning(f"[PDH/PDL] historical_data failed: {e}")
        return None

    today = date.today()
    completed = []
    for b in bars:
        d = b.get("date")
        if isinstance(d, datetime):
            bd = d.date()
        elif isinstance(d, date):
            bd = d
        else:
            continue
        if bd < today:
            completed.append((bd, b))
    if not completed:
        return None
    completed.sort(key=lambda t: t[0])
    last_bar = completed[-1][1]
    try:
        return {
            "pdh": float(last_bar.get("high") or 0.0),
            "pdl": float(last_bar.get("low") or 0.0),
            "prev_close": float(last_bar.get("close") or 0.0),
            "prev_date": completed[-1][0].isoformat(),
        }
    except (TypeError, ValueError):
        return None


async def _load_pdh_pdl_async(cfg: PdhPdlConfig) -> None:
    res = await asyncio.to_thread(_fetch_pdh_pdl_sync, cfg)
    with _state_lock:
        if res is None:
            _state.pdh_pdl_loaded = False
            _state.pdh_pdl_loading = False
            _state.pdh_pdl_error = "yesterday's bar unavailable"
            log_warning("[PDH/PDL] couldn't load PDH/PDL for today")
            return
        _state.pdh = res["pdh"]
        _state.pdl = res["pdl"]
        _state.pdh_pdl_loaded = True
        _state.pdh_pdl_loading = False
        _state.pdh_pdl_error = None
        log_info(
            f"[PDH/PDL] loaded for {_state.day}: PDH={_state.pdh:.2f} PDL={_state.pdl:.2f}"
        )


def _kick_pdh_pdl_load(cfg: PdhPdlConfig) -> None:
    """Schedule the PDH/PDL fetch on the main loop (idempotent per day)."""
    if _state.pdh_pdl_loaded or _state.pdh_pdl_loading:
        return
    if _loop is None:
        return
    _state.pdh_pdl_loading = True
    try:
        asyncio.run_coroutine_threadsafe(_load_pdh_pdl_async(cfg), _loop)
    except Exception as e:  # noqa: BLE001
        _state.pdh_pdl_loading = False
        log_error(f"[PDH/PDL] failed to schedule load: {e}")


def _build_signal(
    *,
    cfg: PdhPdlConfig,
    direction: str,
    pdh: float,
    pdl: float,
    breakout_candle: Candle5m,
) -> PdhPdlSignal:
    entry = breakout_candle.close
    if direction == "LONG":
        stop_loss = pdh - cfg.sl_buffer_pts
        risk_pts = max(1e-6, entry - stop_loss)
        target = entry + cfg.rr_ratio * risk_pts
        option_kind = "CE"
    else:
        stop_loss = pdl + cfg.sl_buffer_pts
        risk_pts = max(1e-6, stop_loss - entry)
        target = entry - cfg.rr_ratio * risk_pts
        option_kind = "PE"

    rng = max(1e-6, breakout_candle.high - breakout_candle.low)
    body = abs(breakout_candle.close - breakout_candle.open)
    body_ratio = body / rng

    reward_pts = abs(target - entry)
    atm_strike = _round_to_step(entry, cfg.atm_step)

    option_leg = estimate_option_leg(
        spot_entry=entry,
        spot_stop_loss=stop_loss,
        spot_target=target,
        kind=option_kind,
        delta=cfg.atm_delta,
        lot_size=cfg.lot_size,
        num_lots=cfg.num_lots,
        atm_step=cfg.atm_step,
        name="NIFTY",
    )

    return PdhPdlSignal(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        target=target,
        risk_points=risk_pts,
        reward_points=reward_pts,
        rr_ratio=cfg.rr_ratio,
        pdh=pdh,
        pdl=pdl,
        breakout_candle_start=breakout_candle.start,
        breakout_candle_close=breakout_candle.close,
        body_ratio=body_ratio,
        atm_strike=atm_strike,
        option_kind=option_kind,
        option_leg=option_leg,
        lot_size=cfg.lot_size,
        num_lots=cfg.num_lots,
    )


def _compose_push(sig: PdhPdlSignal) -> Dict[str, Any]:
    arrow = "▲" if sig.direction == "LONG" else "▼"
    title = (
        f"NIFTY {('PDH' if sig.direction == 'LONG' else 'PDL')} break {sig.direction} {arrow} · "
        f"R:R 1:{sig.rr_ratio:g}"
    )
    leg = sig.option_leg
    qty = sig.lot_size * sig.num_lots
    option_label = (
        leg.contract.tradingsymbol
        if leg.contract
        else f"NIFTY {sig.atm_strike} {sig.option_kind} (ATM)"
    )
    est_tag = " (est.)" if leg.estimated else ""
    risk_inr = leg.risk_inr if leg.risk_inr is not None else 0.0
    reward_inr = leg.reward_inr if leg.reward_inr is not None else 0.0

    lines = [
        f"Spot: Entry {_fmt_level(sig.entry)} · SL {_fmt_level(sig.stop_loss)} · Tgt {_fmt_level(sig.target)}",
        f"Spot risk {sig.risk_points:.0f} pts | reward {sig.reward_points:.0f} pts",
        f"PDH {_fmt_level(sig.pdh)} · PDL {_fmt_level(sig.pdl)} · body {sig.body_ratio:.0%}",
        f"Option leg (not placed): {option_label}",
        (
            f"Premium{est_tag}: Entry ₹{leg.entry_premium:.1f} · "
            f"SL ₹{leg.sl_premium:.1f} · Tgt ₹{leg.target_premium:.1f}"
        ),
        (
            f"{sig.num_lots} lot × {sig.lot_size} = {qty} qty → "
            f"Risk {_fmt_inr(risk_inr)} | Reward {_fmt_inr(reward_inr)}"
        ),
        "Signal only — no order placed.",
    ]
    body = "\n".join(lines)

    data: Dict[str, str] = {
        "type": "nifty_pdh_pdl_signal",
        "source": "algofeast_backend",
        "strategy": "nifty_pdh_pdl_break",
        "direction": sig.direction,
        "rr_ratio": f"{sig.rr_ratio:.2f}",
        "entry": f"{sig.entry:.2f}",
        "stop_loss": f"{sig.stop_loss:.2f}",
        "target": f"{sig.target:.2f}",
        "risk_points": f"{sig.risk_points:.2f}",
        "reward_points": f"{sig.reward_points:.2f}",
        "pdh": f"{sig.pdh:.2f}",
        "pdl": f"{sig.pdl:.2f}",
        "body_ratio": f"{sig.body_ratio:.4f}",
        "breakout_close": f"{sig.breakout_candle_close:.2f}",
        "atm_strike": str(sig.atm_strike),
        "option_kind": sig.option_kind,
        "lot_size": str(sig.lot_size),
        "num_lots": str(sig.num_lots),
    }
    data.update(leg.to_payload())
    return {"title": title, "body": body, "data": data, "signal": sig}


# ----------------------------- tick processing -----------------------------


def _process_closed_candle(
    closed: Candle5m, cfg: PdhPdlConfig
) -> Optional[Dict[str, Any]]:
    cs = closed.start.astimezone(IST)
    today = cs.date()
    warmup = datetime.combine(today, cfg.warmup, tzinfo=IST)
    cutoff = datetime.combine(today, cfg.cutoff, tzinfo=IST)
    _state.closed_candles += 1

    if cs < warmup or cs >= cutoff:
        return None
    if not _state.pdh_pdl_loaded or _state.pdh is None or _state.pdl is None:
        return None

    rng = max(1e-6, closed.high - closed.low)
    body = abs(closed.close - closed.open)
    body_ratio = body / rng
    if body_ratio < cfg.min_body_ratio:
        return None

    direction: Optional[str] = None
    if closed.close > _state.pdh and "LONG" not in _state.fired_directions:
        direction = "LONG"
    elif closed.close < _state.pdl and "SHORT" not in _state.fired_directions:
        direction = "SHORT"

    if direction is None:
        return None

    sig = _build_signal(
        cfg=cfg,
        direction=direction,
        pdh=float(_state.pdh or 0.0),
        pdl=float(_state.pdl or 0.0),
        breakout_candle=closed,
    )
    _state.fired_directions.add(direction)
    return _compose_push(sig)


def _update_candle(tick: Dict[str, Any], cfg: PdhPdlConfig) -> Optional[Candle5m]:
    ts = _tick_time(tick)
    lp = tick.get("last_price")
    if ts is None or lp is None:
        return None
    try:
        price = float(lp)
    except (TypeError, ValueError):
        return None

    _state.last_price = price
    start = _floor_5m(ts)
    if _state.cur_candle is None:
        _state.cur_candle = Candle5m(start=start, open=price, high=price, low=price, close=price)
        return None

    if start == _state.cur_candle.start:
        c = _state.cur_candle
        c.high = max(c.high, price)
        c.low = min(c.low, price)
        c.close = price
        return None

    closed = _state.cur_candle
    _state.cur_candle = Candle5m(start=start, open=price, high=price, low=price, close=price)
    return closed


def _on_ticks_batch(ticks: List[Dict]) -> None:
    if _loop is None:
        return
    cfg = _load_config()
    if not cfg.enabled:
        return

    pushes: List[Dict[str, Any]] = []
    with _state_lock:
        now_ist = datetime.now(IST)
        _ensure_day_state(now_ist)
        # Kick the PDH/PDL fetch ASAP on a new day.
        if not _state.pdh_pdl_loaded and not _state.pdh_pdl_loading:
            _kick_pdh_pdl_load(cfg)

        for t in ticks:
            try:
                if int(t.get("instrument_token") or 0) != cfg.instrument_token:
                    continue
            except (TypeError, ValueError):
                continue
            closed = _update_candle(t, cfg)
            if closed is None:
                continue
            push = _process_closed_candle(closed, cfg)
            if push is not None:
                pushes.append(push)

    for p in pushes:
        try:
            fut = asyncio.run_coroutine_threadsafe(_dispatch_push(p), _loop)
            fut.add_done_callback(lambda f: f.exception())
        except Exception as e:  # noqa: BLE001
            log_error(f"[PDH/PDL] failed to dispatch push: {e}")


# ----------------------------- push dispatch -----------------------------


async def _dispatch_push(p: Dict[str, Any], *, is_test: bool = False) -> Dict[str, Any]:
    if not push_service.configured():
        log_warning("[PDH/PDL] FCM not configured; skipping send.")
        result = {"sent": 0, "failed": 0, "reason": "fcm_not_configured"}
        alert_id = save_strategy_alert(p, result, is_test=is_test)
        if alert_id is not None:
            result["alert_id"] = alert_id
        return result

    cfg = _load_config()
    if cfg.user_id_override:
        user_ids = [cfg.user_id_override]
    else:
        repo = get_push_device_repository()
        user_ids = repo.list_distinct_user_ids() or []
    if not user_ids:
        log_info("[PDH/PDL] No registered devices.")
        result = {"sent": 0, "failed": 0, "reason": "no_devices", "payload": p["data"]}
        alert_id = save_strategy_alert(p, result, is_test=is_test)
        if alert_id is not None:
            result["alert_id"] = alert_id
        return result

    totals: Dict[str, Any] = {"sent": 0, "failed": 0, "per_user": {}, "payload": p["data"]}
    for uid in user_ids:
        try:
            r = await push_service.send_to_user(
                user_id=uid, title=p["title"], body=p["body"], data=p["data"]
            )
            totals["per_user"][uid] = r
            totals["sent"] += int(r.get("sent", 0))
            totals["failed"] += int(r.get("failed", 0))
        except Exception as e:  # noqa: BLE001
            log_error(f"[PDH/PDL] send failed for user_id={uid!r}: {e}")
            totals["failed"] += 1
            totals["per_user"][uid] = {"sent": 0, "failed": 1, "errors": [str(e)]}
    log_info(
        f"[PDH/PDL] dispatched: users={len(user_ids)} sent={totals['sent']} "
        f"failed={totals['failed']} title={p['title']!r}"
    )
    alert_id = save_strategy_alert(p, totals, is_test=is_test)
    if alert_id is not None:
        totals["alert_id"] = alert_id
    return totals


# ----------------------------- public API -----------------------------


def register_nifty_pdh_pdl_signal(loop: asyncio.AbstractEventLoop) -> None:
    global _registered, _loop
    if _registered:
        return
    cfg = _load_config()
    if not cfg.enabled:
        log_info("[PDH/PDL] disabled via NIFTY_PDH_PDL_SIGNAL_ENABLED=0")
        _registered = True
        return
    _loop = loop
    from utils.kite_websocket_ticker import register_tick_callback

    register_tick_callback(_on_ticks_batch)
    _registered = True
    # Prime today's PDH/PDL eagerly so the first tick can act if it's a breakout.
    try:
        with _state_lock:
            _ensure_day_state(datetime.now(IST))
            _kick_pdh_pdl_load(cfg)
    except Exception as e:  # noqa: BLE001
        log_warning(f"[PDH/PDL] initial PDH/PDL load kick failed: {e}")
    log_info(
        f"[PDH/PDL] registered tick handler for token={cfg.instrument_token} "
        f"warmup={cfg.warmup.isoformat()} cutoff={cfg.cutoff.isoformat()}"
    )


def get_state_snapshot() -> Dict[str, Any]:
    cfg = _load_config()
    with _state_lock:
        s = _state
        return {
            "enabled": cfg.enabled,
            "config": {
                "warmup": cfg.warmup.isoformat(),
                "cutoff": cfg.cutoff.isoformat(),
                "rr_ratio": cfg.rr_ratio,
                "sl_buffer_pts": cfg.sl_buffer_pts,
                "min_body_ratio": cfg.min_body_ratio,
                "atm_step": cfg.atm_step,
                "atm_delta": cfg.atm_delta,
                "lot_size": cfg.lot_size,
                "num_lots": cfg.num_lots,
                "instrument_token": cfg.instrument_token,
                "instrument_symbol": cfg.instrument_symbol,
            },
            "day": s.day.isoformat() if s.day else None,
            "pdh": s.pdh,
            "pdl": s.pdl,
            "pdh_pdl_loaded": s.pdh_pdl_loaded,
            "pdh_pdl_error": s.pdh_pdl_error,
            "fired_directions": sorted(list(s.fired_directions)),
            "last_price": s.last_price,
            "closed_candles_today": s.closed_candles,
            "fcm_configured": push_service.configured(),
        }


def _fetch_current_spot() -> Optional[float]:
    cfg = _load_config()
    try:
        kite = get_kite_instance()
        q = kite.quote(cfg.instrument_symbol) or {}
        row = q.get(cfg.instrument_symbol) or {}
        lp = row.get("last_price")
        if lp is None:
            ohlc = row.get("ohlc") or {}
            lp = ohlc.get("open") or ohlc.get("close")
        return float(lp) if lp is not None else None
    except Exception as e:  # noqa: BLE001
        log_warning(f"[PDH/PDL] _fetch_current_spot failed: {e}")
        return None


def preview_signal(*, direction_override: Optional[str] = None) -> Dict[str, Any]:
    cfg = _load_config()
    with _state_lock:
        last_price = _state.last_price
        pdh = _state.pdh
        pdl = _state.pdl
    if last_price is None:
        last_price = _fetch_current_spot()
    if last_price is None:
        return {"available": False, "reason": "spot price unavailable"}

    if pdh is None or pdl is None:
        # Try to fetch synchronously; tolerate failure with a synthetic context.
        snap = _fetch_pdh_pdl_sync(cfg)
        if snap:
            pdh, pdl = snap["pdh"], snap["pdl"]
            synthesized = False
        else:
            pdh = last_price + 60.0
            pdl = last_price - 60.0
            synthesized = True
    else:
        synthesized = False

    direction = (direction_override or "LONG").upper()
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"

    if direction == "LONG":
        close = max(last_price, pdh + 10.0)
    else:
        close = min(last_price, pdl - 10.0)
    open_ = close - 6.0 if direction == "LONG" else close + 6.0
    high = max(open_, close) + 2.0
    low = min(open_, close) - 2.0
    breakout = Candle5m(
        start=_floor_5m(datetime.now(IST)),
        open=open_,
        high=high,
        low=low,
        close=close,
    )

    sig = _build_signal(
        cfg=cfg,
        direction=direction,
        pdh=float(pdh),
        pdl=float(pdl),
        breakout_candle=breakout,
    )
    p = _compose_push(sig)
    return {
        "available": True,
        "synthesized": synthesized,
        "title": p["title"],
        "body": p["body"],
        "payload": p["data"],
    }


async def force_test_send(*, direction: str = "LONG") -> Dict[str, Any]:
    if not push_service.configured():
        return {"sent": 0, "failed": 0, "reason": "fcm_not_configured"}
    prev = preview_signal(direction_override=direction)
    if not prev.get("available"):
        return {"sent": 0, "failed": 0, "reason": prev.get("reason", "preview_unavailable")}
    return await _dispatch_push(
        {"title": prev["title"], "body": prev["body"], "data": prev["payload"]},
        is_test=True,
    )


def reset_day_state() -> Dict[str, Any]:
    with _state_lock:
        _state.day = None
        _state.pdh = None
        _state.pdl = None
        _state.pdh_pdl_loaded = False
        _state.pdh_pdl_loading = False
        _state.pdh_pdl_error = None
        _state.cur_candle = None
        _state.closed_candles = 0
        _state.last_price = None
        _state.fired_directions = set()
    log_info("[PDH/PDL] day state reset")
    return {"ok": True}
