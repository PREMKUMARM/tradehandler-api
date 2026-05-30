#!/usr/bin/env python3
"""Backtest BTCUSDT 5m BB mean-reversion v3 (matches live crypto_indicator_plan)."""
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
    bb_reentry_reset_zone,
    passes_min_rr,
    passes_min_tp_reward,
    side_aligns_with_daily_candle,
)
from services.crypto_ta import (
    bb_bandwidth_pct,
    compute_adx,
    compute_atr,
    compute_atr_series,
    compute_rsi,
    middle_band_slope,
)
from services.crypto_trail import compute_crypto_trail_sl, get_crypto_trail_config
from services.kite_live_indicators import compute_bollinger_bands
from utils.binance_historical import fetch_historical_klines

IST = ZoneInfo("Asia/Kolkata")
MARGIN_USDT = float(os.getenv("CRYPTO_LIVE_MARGIN_USDT", "15") or 15)
LEVERAGE = int(os.getenv("CRYPTO_LEVERAGE", "50") or 50)
MAX_TRADES = int(os.getenv("CRYPTO_WATCH_MAX_TRADES_PER_DAY", "20") or 20)
QTY_STEP = 0.001
MAX_HOLD_BARS = int(os.getenv("CRYPTO_BB_MAX_HOLD_BARS", "12") or 12)
COMMISSION_RATE = 0.0004  # taker exit (SL/TP market)
ENTRY_FEE_RATE = 0.0002  # limit entry (maker estimate)


def _trade_fees(entry: float, exit_px: float, qty: float) -> float:
    return entry * qty * ENTRY_FEE_RATE + exit_px * qty * COMMISSION_RATE


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
    out: Dict[Any, float] = {}
    for k in klines:
        day = _parse_ts(k["timestamp"]).date()
        if day not in out:
            out[day] = float(k["open"])
    return out


def _bar(k: Dict[str, Any]) -> Dict[str, float]:
    return {
        "open": float(k["open"]),
        "high": float(k["high"]),
        "low": float(k["low"]),
        "close": float(k["close"]),
        "volume": float(k.get("volume", 0)),
    }


def _live_snapshot(
    klines: List[Dict[str, Any]],
    i: int,
    day_opens: Dict[Any, float],
) -> Dict[str, Any]:
    """Indicators at close of bar i (signal bar = klines[i])."""
    bars = [_bar(k) for k in klines[: i + 1]]
    closes = [b["close"] for b in bars]
    ts = _parse_ts(klines[i]["timestamp"])
    sig = bars[-1]
    prev = bars[-2] if len(bars) >= 2 else sig
    spot = float(sig["close"])

    bb_mid, bb_upper, bb_lower = compute_bollinger_bands(closes, period=20)
    rsi14 = compute_rsi(closes, period=14)
    adx14 = compute_adx(bars, period=14)
    atr14 = compute_atr(bars, period=14)
    atr_series = compute_atr_series(bars, period=14)
    recent_atr = [a for a in atr_series[-20:] if a is not None]
    atr_sma20 = sum(recent_atr) / len(recent_atr) if recent_atr else None
    atr_ratio = (atr14 / atr_sma20) if atr14 and atr_sma20 and atr_sma20 > 0 else None
    bb_bw = (
        bb_bandwidth_pct(float(bb_upper), float(bb_lower), float(bb_mid))
        if bb_mid and bb_upper and bb_lower
        else None
    )
    mid_slope = middle_band_slope(closes, period=20, lookback=3)
    day_open = day_opens.get(ts.date(), spot)
    vols = [b.get("volume", 0) for b in bars]
    vol_avg20 = sum(vols[-20:]) / min(20, len(vols)) if vols else None

    return {
        "connected": bb_mid is not None and adx14 is not None,
        "btc_spot": spot,
        "signal_spot": spot,
        "bb_middle": bb_mid,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_bandwidth_pct": bb_bw,
        "rsi14": rsi14,
        "adx14": adx14,
        "atr14": atr14,
        "atr_sma20": atr_sma20,
        "atr_ratio": atr_ratio,
        "bb_middle_slope": mid_slope,
        "volume_avg20": vol_avg20,
        "prev_5m_bar": prev,
        "last_5m_bar": sig,
        "prev_close": closes[-2] if len(closes) >= 2 else spot,
        "day_open": day_open,
        "day_candle_green": spot > day_open,
        "day_candle_red": spot < day_open,
    }


def _update_trail(open_pos: Dict[str, Any], bar: Dict[str, float]) -> None:
    cfg = get_crypto_trail_config()
    if not cfg.enabled:
        return
    side = open_pos["side"]
    ltp = float(bar["high"]) if side == "LONG" else float(bar["low"])
    peak = float(open_pos.get("peak") or open_pos["entry_price"])
    new_sl, new_peak, activated, _ = compute_crypto_trail_sl(
        side=side,
        entry=float(open_pos["entry_price"]),
        initial_sl=float(open_pos["initial_sl"]),
        min_tp=float(open_pos["tp"]),
        peak=peak,
        ltp=ltp,
        current_sl=float(open_pos["sl"]),
        trail_active=bool(open_pos.get("trail_active")),
        symbol="BTCUSDT",
        cfg=cfg,
    )
    open_pos["sl"] = new_sl
    open_pos["peak"] = new_peak
    open_pos["trail_active"] = activated or open_pos.get("trail_active")


def _check_exit(open_pos: Dict[str, Any], bar: Dict[str, float]) -> tuple[Optional[str], Optional[float]]:
    side = open_pos["side"]
    sl = float(open_pos["sl"])
    entry = float(open_pos["entry_price"])
    _update_trail(open_pos, bar)
    sl = float(open_pos["sl"])

    if side == "LONG":
        if sl >= entry:
            return None, None
        if bar["low"] <= sl:
            return ("TRAIL" if open_pos.get("trail_active") else "SL"), sl
    else:
        if sl <= entry:
            return None, None
        if bar["high"] >= sl:
            return ("TRAIL" if open_pos.get("trail_active") else "SL"), sl
    return None, None


async def run_backtest(hours: float = 24.0) -> Dict[str, Any]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    klines = await fetch_historical_klines("BTCUSDT", "5m", start, end)
    if len(klines) < 50:
        return {"error": "Insufficient klines", "orders": []}

    day_opens = _build_day_opens(klines)
    orders: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None
    trades_today = 0
    session_day: Optional[Any] = None
    skipped_daily = 0
    skipped_reward = 0
    skipped_rr = 0
    skipped_cycle = 0
    skipped_no_signal = 0
    cooldown_until: Optional[datetime] = None
    bb_cycle_armed = True
    last_signal_bar: Optional[int] = None

    for i in range(40, len(klines)):
        k = klines[i]
        ts = _parse_ts(k["timestamp"])
        bar = _bar(k)
        day = ts.date()
        if session_day != day:
            session_day = day
            trades_today = 0
        live = _live_snapshot(klines, i, day_opens)

        if open_pos:
            open_pos["bars_held"] = int(open_pos.get("bars_held", 0)) + 1
            reason, exit_px = _check_exit(open_pos, bar)
            if not reason and open_pos["bars_held"] >= MAX_HOLD_BARS:
                side = open_pos["side"]
                entry = open_pos["entry_price"]
                close = float(bar["close"])
                unreal = (close - entry) * open_pos["qty"] if side == "LONG" else (entry - close) * open_pos["qty"]
                if unreal <= 0:
                    reason, exit_px = "TIME", close
            if reason:
                qty = open_pos["qty"]
                side = open_pos["side"]
                entry = open_pos["entry_price"]
                gross = (exit_px - entry) * qty if side == "LONG" else (entry - exit_px) * qty
                fees = _trade_fees(entry, exit_px, qty)
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
                        "sl_price": round(open_pos["sl"], 2),
                        "tp_price": round(open_pos["tp"], 2),
                        "exit_reason": reason,
                        "bb_zone": open_pos.get("bb_zone"),
                        "entry_style": open_pos.get("entry_style"),
                        "score": open_pos.get("score"),
                        "adx": open_pos.get("adx"),
                        "rsi": open_pos.get("rsi"),
                        "gross_pnl_usdt": round(gross, 4),
                        "fees_usdt": round(fees, 4),
                        "pnl_usdt": round(gross - fees, 4),
                    }
                )
                open_pos = None
                cooldown_until = ts + timedelta(seconds=90)
                bb_cycle_armed = False
            continue

        if not bb_cycle_armed and bb_reentry_reset_zone(live):
            bb_cycle_armed = True

        if trades_today >= MAX_TRADES:
            continue
        if cooldown_until and ts < cooldown_until:
            continue
        if not bb_cycle_armed:
            skipped_cycle += 1
            continue
        if last_signal_bar == i:
            continue

        side = _resolve_side("AUTO", live)
        spot = float(live.get("signal_spot") or 0)
        entry_ready, block, score, entry_style, entry_limit, bb_zone = _bb_entry_analysis(
            side, spot, live
        )
        if not entry_ready:
            skipped_no_signal += 1
            continue

        ok_day, _ = side_aligns_with_daily_candle(side, live)
        if not ok_day:
            skipped_daily += 1
            continue

        qty = qty_from_margin(entry_limit)
        if qty < QTY_STEP:
            continue

        _, sl, tp = _bb_exit_levels(side, entry_limit, live, bb_zone=bb_zone)
        ok_rw, _ = passes_min_tp_reward(entry_limit, tp, side, qty)
        if not ok_rw:
            skipped_reward += 1
            continue
        ok_rr, _ = passes_min_rr(entry_limit, sl, tp, side)
        if not ok_rr:
            skipped_rr += 1
            continue

        open_pos = {
            "side": side,
            "entry_price": entry_limit,
            "entry_time": ts,
            "initial_sl": sl,
            "sl": sl,
            "tp": tp,
            "peak": entry_limit,
            "trail_active": False,
            "qty": qty,
            "bb_zone": bb_zone,
            "entry_style": entry_style,
            "score": score,
            "adx": live.get("adx14"),
            "rsi": live.get("rsi14"),
        }
        trades_today += 1
        last_signal_bar = i

    if open_pos:
        last = klines[-1]
        ts = _parse_ts(last["timestamp"])
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
                "exit_reason": "EOD",
                "bb_zone": open_pos.get("bb_zone"),
                "pnl_usdt": round(gross - fees, 4),
            }
        )

    total_pnl = sum(o.get("pnl_usdt", 0) for o in orders)
    wins = sum(1 for o in orders if o.get("pnl_usdt", 0) > 0)
    losses = sum(1 for o in orders if o.get("pnl_usdt", 0) < 0)

    return {
        "symbol": "BTCUSDT",
        "strategy": "bb_5m_reversal_v3 (close-back-inside + ADX + RSI)",
        "filters": {
            "entry": "Pierce band then close back inside (reversal confirm)",
            "regime": f"ADX ≤ {os.getenv('CRYPTO_BB_ADX_MAX', '25')}, BB width band, ATR ratio cap",
            "rsi": f"LONG ≤ {os.getenv('CRYPTO_BB_RSI_LONG_MAX', '38')}, SHORT ≥ {os.getenv('CRYPTO_BB_RSI_SHORT_MIN', '62')}",
            "daily_candle": os.getenv("CRYPTO_DAILY_CANDLE_FILTER", "0"),
            "min_tp_reward_usdt": float(os.getenv("CRYPTO_MIN_TP_REWARD_USDT", "0.75") or 0.75),
            "min_rr": float(os.getenv("CRYPTO_MIN_RR", "1.5") or 1.5),
            "trail": "SL ratchets after 1R or min-TP; no fixed TP cap",
        },
        "margin_usdt": MARGIN_USDT,
        "leverage": LEVERAGE,
        "skipped_no_signal": skipped_no_signal,
        "skipped_against_daily": skipped_daily,
        "skipped_low_reward": skipped_reward,
        "skipped_low_rr": skipped_rr,
        "skipped_bb_cycle": skipped_cycle,
        "period_utc": {"start": start.isoformat(), "end": end.isoformat()},
        "candles": len(klines),
        "total_trades": len(orders),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(100.0 * wins / len(orders), 1) if orders else 0.0,
        "total_pnl_usdt": round(total_pnl, 4),
        "orders": orders,
    }


def main() -> None:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
    result = asyncio.run(run_backtest(hours=hours))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
