#!/usr/bin/env python3
"""Backtest BTCUSDT 5m BB mean-reversion strategy (matches live crypto_indicator_plan)."""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.crypto_indicator_plan import (
    _bb_entry_analysis,
    _bb_exit_levels,
    _resolve_side,
    daily_candle_bias,
    side_aligns_with_daily_candle,
)
from services.kite_live_indicators import compute_bollinger_bands
from utils.binance_historical import fetch_historical_klines

IST = ZoneInfo("Asia/Kolkata")
MARGIN_USDT = float(os.getenv("CRYPTO_LIVE_MARGIN_USDT", "15") or 15)
LEVERAGE = int(os.getenv("CRYPTO_LEVERAGE", "50") or 50)
MAX_TRADES = int(os.getenv("CRYPTO_WATCH_MAX_TRADES_PER_DAY", "20") or 20)
QTY_STEP = 0.001
COMMISSION_RATE = 0.0004  # ~0.04% per side taker estimate


def _round_qty(raw: float) -> float:
    if raw <= 0:
        return 0.0
    q = math.floor(raw / QTY_STEP) * QTY_STEP
    return round(q, 3)


def qty_from_margin(spot: float) -> float:
    notional = MARGIN_USDT * LEVERAGE
    return _round_qty(notional / spot)


def _parse_ts(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        ts = raw
    else:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _build_day_opens(klines: List[Dict[str, Any]]) -> Dict[Any, float]:
    """UTC calendar day → first 5m bar open (proxy for Binance 1d open)."""
    out: Dict[Any, float] = {}
    for k in klines:
        day = _parse_ts(k["timestamp"]).date()
        if day not in out:
            out[day] = float(k["open"])
    return out


def _live_snapshot(
    closes: List[float], spot: float, ts: datetime, day_opens: Dict[Any, float]
) -> Dict[str, Any]:
    bb_mid, bb_upper, bb_lower = compute_bollinger_bands(closes, period=20)
    day_open = day_opens.get(ts.date(), spot)
    return {
        "connected": bb_mid is not None,
        "btc_spot": spot,
        "bb_middle": bb_mid,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "prev_close": closes[-2] if len(closes) >= 2 else spot,
        "day_open": day_open,
        "day_candle_green": spot > day_open,
        "day_candle_red": spot < day_open,
    }


def _try_fill_limit(side: str, limit_px: float, bar: Dict[str, float]) -> Optional[float]:
    """LIMIT fill if price trades through limit on this 5m bar."""
    if side == "LONG":
        if bar["low"] <= limit_px:
            return limit_px
    else:
        if bar["high"] >= limit_px:
            return limit_px
    return None


def _check_exit(
    side: str, sl: float, tp: float, entry: float, bar: Dict[str, float]
) -> tuple[Optional[str], Optional[float]]:
    """Return (exit_reason, exit_price) if SL or TP hit on bar."""
    if side == "LONG":
        if sl >= entry or tp <= entry:
            return None, None
        sl_hit = bar["low"] <= sl
        tp_hit = bar["high"] >= tp
        if sl_hit and tp_hit:
            return ("SL", sl) if bar["open"] >= entry else ("TP", tp)
        if sl_hit:
            return "SL", sl
        if tp_hit:
            return "TP", tp
    else:
        if sl <= entry or tp >= entry:
            return None, None
        sl_hit = bar["high"] >= sl
        tp_hit = bar["low"] <= tp
        if sl_hit and tp_hit:
            return ("SL", sl) if bar["open"] <= entry else ("TP", tp)
        if sl_hit:
            return "SL", sl
        if tp_hit:
            return "TP", tp
    return None, None


async def run_backtest(hours: float = 24.0) -> Dict[str, Any]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    klines = await fetch_historical_klines("BTCUSDT", "5m", start, end)
    if len(klines) < 25:
        return {"error": "Insufficient klines", "orders": []}

    day_opens = _build_day_opens(klines)
    orders: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None
    open_pos: Optional[Dict[str, Any]] = None
    trades_today = 0
    skipped_daily = 0
    cooldown_until: Optional[datetime] = None
    closes: List[float] = []

    for i, k in enumerate(klines):
        closes.append(float(k["close"]))
        if len(closes) < 20:
            continue

        ts = _parse_ts(k["timestamp"])

        bar = {
            "open": float(k["open"]),
            "high": float(k["high"]),
            "low": float(k["low"]),
            "close": float(k["close"]),
        }
        spot = bar["close"]
        live = _live_snapshot(closes, spot, ts, day_opens)
        daily_bias, _ = daily_candle_bias(live)

        # Manage open position exits
        if open_pos:
            reason, exit_px = _check_exit(
                open_pos["side"], open_pos["sl"], open_pos["tp"], open_pos["entry_price"], bar
            )
            if reason:
                qty = open_pos["qty"]
                side = open_pos["side"]
                entry = open_pos["entry_price"]
                if side == "LONG":
                    gross = (exit_px - entry) * qty
                else:
                    gross = (entry - exit_px) * qty
                fees = (entry + exit_px) * qty * COMMISSION_RATE
                pnl = gross - fees
                orders.append(
                    {
                        "trade_num": len(orders) + 1,
                        "side": side,
                        "entry_time_ist": open_pos["entry_time"].astimezone(IST).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "exit_time_ist": ts.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": round(entry, 2),
                        "exit_price": round(exit_px, 2),
                        "qty_btc": qty,
                        "margin_usdt": round(entry * qty / LEVERAGE, 2),
                        "notional_usdt": round(entry * qty, 2),
                        "sl_price": round(open_pos["sl"], 2),
                        "tp_price": round(open_pos["tp"], 2),
                        "exit_reason": reason,
                        "daily_bias": open_pos.get("daily_bias"),
                        "day_open": open_pos.get("day_open"),
                        "bb_zone": open_pos.get("bb_zone"),
                        "entry_style": open_pos.get("entry_style"),
                        "score": open_pos.get("score"),
                        "gross_pnl_usdt": round(gross, 4),
                        "fees_usdt": round(fees, 4),
                        "pnl_usdt": round(pnl, 4),
                    }
                )
                open_pos = None
                cooldown_until = ts + timedelta(seconds=90)
            continue

        # Pending limit entry fill (max 6 bars = 30 min)
        if pending:
            pending["bars_waited"] = pending.get("bars_waited", 0) + 1
            fill = _try_fill_limit(pending["side"], pending["entry_limit"], bar)
            if fill:
                live_fill = _live_snapshot(closes, fill, ts, day_opens)
                _, sl, tp = _bb_exit_levels(
                    pending["side"], fill, live_fill, bb_zone=pending.get("bb_zone")
                )
                open_pos = {
                    "side": pending["side"],
                    "entry_price": fill,
                    "entry_time": ts,
                    "sl": sl,
                    "tp": tp,
                    "qty": pending["qty"],
                    "bb_zone": pending.get("bb_zone"),
                    "entry_style": pending.get("entry_style"),
                    "score": pending.get("score"),
                    "daily_bias": pending.get("daily_bias"),
                    "day_open": pending.get("day_open"),
                }
                pending = None
            elif pending["bars_waited"] >= 6:
                pending = None
            continue

        if trades_today >= MAX_TRADES:
            continue
        if cooldown_until and ts < cooldown_until:
            continue

        side = _resolve_side("AUTO", live)
        entry_ready, block, score, entry_style, entry_limit, bb_zone = _bb_entry_analysis(
            side, spot, live
        )
        if not entry_ready:
            continue

        ok_day, day_msg = side_aligns_with_daily_candle(side, live)
        if not ok_day:
            skipped_daily += 1
            continue

        qty = qty_from_margin(entry_limit)
        if qty < QTY_STEP:
            continue

        need_margin = entry_limit * qty / LEVERAGE
        if need_margin > MARGIN_USDT + 0.05:
            continue

        _, sl, tp = _bb_exit_levels(side, entry_limit, live, bb_zone=bb_zone)
        plan = {
            "side": side,
            "entry_limit_price": entry_limit,
            "stop_loss_premium": sl,
            "target_premium": tp,
            "bb_zone": bb_zone,
            "entry_style": entry_style,
            "entry_confirmation_score": score,
        }

        # Try same-bar fill
        fill = _try_fill_limit(side, entry_limit, bar)
        if fill:
            live_fill = _live_snapshot(closes, fill, ts, day_opens)
            _, sl, tp = _bb_exit_levels(side, fill, live_fill, bb_zone=bb_zone)
            open_pos = {
                "side": side,
                "entry_price": fill,
                "entry_time": ts,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "bb_zone": bb_zone,
                "entry_style": entry_style,
                "score": score,
                "daily_bias": daily_bias,
                "day_open": live.get("day_open"),
            }
            trades_today += 1
        else:
            pending = {
                "side": side,
                "entry_limit": entry_limit,
                "qty": qty,
                "plan": plan,
                "bb_zone": bb_zone,
                "entry_style": entry_style,
                "score": score,
                "daily_bias": daily_bias,
                "day_open": live.get("day_open"),
                "bars_waited": 0,
            }
            trades_today += 1

    # Close any open position at last close
    if open_pos:
        last = klines[-1]
        ts = last["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        exit_px = float(last["close"])
        side = open_pos["side"]
        entry = open_pos["entry_price"]
        qty = open_pos["qty"]
        gross = (exit_px - entry) * qty if side == "LONG" else (entry - exit_px) * qty
        fees = (entry + exit_px) * qty * COMMISSION_RATE
        orders.append(
            {
                "trade_num": len(orders) + 1,
                "side": side,
                "entry_time_ist": open_pos["entry_time"].astimezone(IST).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "exit_time_ist": ts.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S"),
                "entry_price": round(entry, 2),
                "exit_price": round(exit_px, 2),
                "qty_btc": qty,
                "margin_usdt": round(entry * qty / LEVERAGE, 2),
                "notional_usdt": round(entry * qty, 2),
                "exit_reason": "EOD",
                "bb_zone": open_pos.get("bb_zone"),
                "entry_style": open_pos.get("entry_style"),
                "score": open_pos.get("score"),
                "gross_pnl_usdt": round(gross, 4),
                "fees_usdt": round(fees, 4),
                "pnl_usdt": round(gross - fees, 4),
            }
        )

    total_pnl = sum(o["pnl_usdt"] for o in orders)
    wins = sum(1 for o in orders if o["pnl_usdt"] > 0)
    losses = sum(1 for o in orders if o["pnl_usdt"] < 0)

    return {
        "symbol": "BTCUSDT",
        "strategy": "bb_5m_mean_reversion + 1d_candle_filter",
        "filters": {
            "daily_candle": "LONG if 1d green (spot > day open), SHORT if 1d red",
        },
        "margin_usdt": MARGIN_USDT,
        "leverage": LEVERAGE,
        "skipped_against_daily": skipped_daily,
        "period_utc": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "candles": len(klines),
        "total_trades": len(orders),
        "wins": wins,
        "losses": losses,
        "total_pnl_usdt": round(total_pnl, 4),
        "orders": orders,
    }


def main() -> None:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
    result = asyncio.run(run_backtest(hours=hours))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
