"""Trail operations: activation hold, partial exit, time stop, regime config."""
from __future__ import annotations

import os
from datetime import datetime, time as dt_time
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from services.momentum_trail import (
    MomentumTrailConfig,
    get_momentum_trail_config,
    get_regime_for_strategy,
)
from utils.logger import log_info, log_warning

IST = ZoneInfo("Asia/Kolkata")


def resolve_trail_config(strategy_id: Optional[str] = None) -> MomentumTrailConfig:
    """Regime-aware overrides on top of env defaults."""
    cfg = get_momentum_trail_config()
    regime = get_regime_for_strategy(strategy_id)
    if regime == "mean_reversion":
        return MomentumTrailConfig(
            enabled=cfg.enabled,
            breakeven_buffer=cfg.breakeven_buffer,
            breakeven_pct=cfg.breakeven_pct,
            breakeven_r_fraction=min(0.35, cfg.breakeven_r_fraction + 0.1),
            extend_gain_ratio=max(0.35, cfg.extend_gain_ratio - 0.1),
            lock_gain_ratio=min(0.5, cfg.lock_gain_ratio + 0.1),
            min_level_update=cfg.min_level_update,
            stepped_rr=cfg.stepped_rr,
            partial_exit_enabled=cfg.partial_exit_enabled,
            partial_exit_pct=min(0.55, cfg.partial_exit_pct + 0.15),
            activation_hold_sec=cfg.activation_hold_sec,
            require_5m_close=cfg.require_5m_close,
            time_stop_enabled=cfg.time_stop_enabled,
            time_stop_minutes=max(45, cfg.time_stop_minutes - 30),
            time_stop_before_ist=cfg.time_stop_before_ist,
            time_stop_only_without_1r=cfg.time_stop_only_without_1r,
            trail_stale_alert_min=cfg.trail_stale_alert_min,
            gtt_fail_alert_threshold=cfg.gtt_fail_alert_threshold,
        )
    if regime == "trend":
        return MomentumTrailConfig(
            enabled=cfg.enabled,
            breakeven_buffer=cfg.breakeven_buffer,
            breakeven_pct=cfg.breakeven_pct,
            breakeven_r_fraction=cfg.breakeven_r_fraction,
            extend_gain_ratio=min(0.65, cfg.extend_gain_ratio + 0.1),
            lock_gain_ratio=max(0.25, cfg.lock_gain_ratio - 0.05),
            min_level_update=cfg.min_level_update,
            stepped_rr=cfg.stepped_rr,
            partial_exit_enabled=cfg.partial_exit_enabled,
            partial_exit_pct=max(0.25, cfg.partial_exit_pct - 0.1),
            activation_hold_sec=cfg.activation_hold_sec,
            require_5m_close=cfg.require_5m_close,
            time_stop_enabled=cfg.time_stop_enabled,
            time_stop_minutes=cfg.time_stop_minutes,
            time_stop_before_ist=cfg.time_stop_before_ist,
            time_stop_only_without_1r=cfg.time_stop_only_without_1r,
            trail_stale_alert_min=cfg.trail_stale_alert_min,
            gtt_fail_alert_threshold=cfg.gtt_fail_alert_threshold,
        )
    return cfg


def get_exit_policy_summary(
    strategy_id: Optional[str] = None,
    *,
    quantity: Optional[int] = None,
) -> Dict[str, Any]:
    from services.premium_exit_policy import entry_initial_rr, entry_validation_skips_reward

    cfg = resolve_trail_config(strategy_id)
    initial_rr = entry_initial_rr()
    qty = int(quantity or 0)
    if qty <= 1:
        partial = "off (single lot → breakeven trail at 1R)"
    elif cfg.partial_exit_enabled:
        partial = f"{int(cfg.partial_exit_pct * 100)}%"
    else:
        partial = "off"
    lines = [
        f"Entry target {initial_rr:g}:1 · trail extends to 2R, 3R…",
        f"At 1R: book {partial}, SL→breakeven, TP→next R step",
        f"Activation hold {cfg.activation_hold_sec}s"
        + (" + 5m close" if cfg.require_5m_close else ""),
    ]
    if cfg.time_stop_enabled:
        lines.append(
            f"Time stop: {cfg.time_stop_minutes}m without 1R or before {cfg.time_stop_before_ist} IST"
        )
    if entry_validation_skips_reward():
        lines.append("Entry blocked only on risk cap (not reward ratio)")
    return {
        "entry_rr": initial_rr,
        "partial_at_1r_pct": round(cfg.partial_exit_pct * 100, 0),
        "activation_hold_sec": cfg.activation_hold_sec,
        "require_5m_close": cfg.require_5m_close,
        "time_stop_minutes": cfg.time_stop_minutes,
        "regime": get_regime_for_strategy(strategy_id),
        "summary_lines": lines,
        "summary": " · ".join(lines[:2]),
    }


def _parse_ist_hhmm(value: str) -> dt_time:
    parts = (value or "15:10").strip().split(":")
    h = int(parts[0]) if parts else 15
    m = int(parts[1]) if len(parts) > 1 else 10
    return dt_time(max(0, min(23, h)), max(0, min(59, m)))


def last_5m_close_at_or_above(exchange: str, symbol: str, level: float) -> bool:
    if level <= 0 or not symbol:
        return True
    try:
        from services.kite_live_indicators import get_option_bollinger_snapshot

        bb = get_option_bollinger_snapshot(symbol, exchange.upper())
        close = bb.get("last_5m_close")
        if close is None:
            return True
        return float(close) >= level
    except Exception:
        return True


def activation_ready(
    trail: Dict[str, Any],
    *,
    ltp: float,
    activation_target: float,
    now: datetime,
    cfg: MomentumTrailConfig,
) -> Tuple[bool, Optional[str]]:
    """Returns (ready, target_touch_since_iso to persist)."""
    if bool(trail.get("trail_active")):
        return True, None
    if activation_target <= 0 or ltp < activation_target:
        return False, None

    touch_since = trail.get("target_touch_since")
    if not touch_since:
        return False, now.isoformat()

    try:
        started = datetime.fromisoformat(str(touch_since))
        if started.tzinfo is None:
            started = started.replace(tzinfo=IST)
        held = (now.astimezone(IST) - started.astimezone(IST)).total_seconds()
    except Exception:
        held = 0.0

    if held < cfg.activation_hold_sec:
        return False, None

    if cfg.require_5m_close:
        ex = str(trail.get("exchange") or "NFO")
        sym = str(trail.get("tradingsymbol") or "")
        if not last_5m_close_at_or_above(ex, sym, activation_target):
            return False, None

    return True, None


def check_time_stop(
    trail: Dict[str, Any],
    *,
    now: datetime,
    trail_active: bool,
    cfg: MomentumTrailConfig,
) -> Optional[str]:
    if not cfg.time_stop_enabled:
        return None
    if cfg.time_stop_only_without_1r and trail_active:
        return None

    created = trail.get("created_at") or trail.get("updated_at")
    if not created:
        return None
    try:
        placed = datetime.fromisoformat(str(created))
        if placed.tzinfo is None:
            placed = placed.replace(tzinfo=IST)
        age_min = (now.astimezone(IST) - placed.astimezone(IST)).total_seconds() / 60.0
    except Exception:
        age_min = 0.0

    if not trail_active and age_min >= cfg.time_stop_minutes:
        return f"No 1R after {cfg.time_stop_minutes:.0f} min — theta/time stop"

    cutoff = _parse_ist_hhmm(cfg.time_stop_before_ist)
    now_t = now.astimezone(IST).time()
    if not trail_active and now_t >= cutoff:
        return f"Session cutoff {cfg.time_stop_before_ist} IST without 1R — time stop"

    return None


def partial_exit_qty(total_qty: int, cfg: MomentumTrailConfig) -> int:
    if not cfg.partial_exit_enabled or total_qty <= 1:
        return 0
    pct = max(0.1, min(0.7, cfg.partial_exit_pct))
    raw = int(round(total_qty * pct))
    return max(1, min(total_qty - 1, raw))


def execute_live_partial_exit(trail: Dict[str, Any], qty: int, ltp: float) -> bool:
    sym = str(trail.get("tradingsymbol") or "")
    if not sym or qty <= 0:
        return False
    try:
        from agent.tools.kite_tools import place_order_tool

        res = place_order_tool.invoke(
            {
                "tradingsymbol": sym,
                "exchange": trail.get("exchange") or "NFO",
                "transaction_type": "SELL",
                "quantity": qty,
                "order_type": "MARKET",
                "product": trail.get("product") or "NRML",
                "segment": str(trail.get("segment") or "nifty50"),
            }
        )
        if res.get("status") == "success":
            log_info(
                f"[TrailOps] partial exit {sym} qty={qty} @ ~{ltp:.2f} order={res.get('order_id')}"
            )
            return True
        log_warning(f"[TrailOps] partial exit failed {sym}: {res.get('error') or res}")
    except Exception as exc:
        log_warning(f"[TrailOps] partial exit error {sym}: {exc}")
    return False


def execute_paper_partial_exit(trail: Dict[str, Any], qty: int, ltp: float) -> bool:
    from database.connection import get_database
    from services.paper_trading import paper_place_order

    paper_oid = str(trail.get("paper_order_id") or trail.get("entry_order_id") or "")
    if not paper_oid or qty <= 0:
        return False
    db = get_database()
    conn = db.get_connection()
    cur = conn.execute(
        "SELECT id, payload, quantity FROM paper_orders WHERE order_id = ? AND exit_reason IS NULL",
        (paper_oid,),
    )
    row = cur.fetchone()
    if not row:
        return False
    import json

    payload = row["payload"]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    exit_payload = {
        "paper_exit_leg": True,
        "paper_partial_exit": True,
        "parent_order_id": paper_oid,
        "exit_reason": "partial_1r",
        "tradingsymbol": payload.get("tradingsymbol"),
        "exchange": payload.get("exchange"),
        "transaction_type": "SELL",
        "quantity": qty,
        "order_type": "MARKET",
        "product": payload.get("product"),
        "paper_fill_price": ltp,
    }
    try:
        exit_oid = paper_place_order(exit_payload)
        remaining = max(1, int(row["quantity"] or qty) - qty)
        conn.execute(
            "UPDATE paper_orders SET quantity = ? WHERE id = ?",
            (remaining, int(row["id"])),
        )
        conn.commit()
        log_info(f"[TrailOps] paper partial {paper_oid} qty={qty} rem={remaining} exit={exit_oid}")
        return True
    except Exception as exc:
        log_warning(f"[TrailOps] paper partial failed: {exc}")
        return False
