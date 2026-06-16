#!/usr/bin/env python3
"""Replay live strategy rules on a single FnO session day (Kite historical)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

IST = ZoneInfo("Asia/Kolkata")
MIN_SCORE = 65
SENSEX_BAND = (17.0, 23.0)
SENSEX_SL = 9.0
SENSEX_SCAN_START = 9 * 60 + 20
SENSEX_SCAN_END = 15 * 60


def _session_bars(kite, token: int, d: date, *, open_min: int, close_min: int) -> List[Dict[str, Any]]:
    raw = kite.historical_data(token, d, d + timedelta(days=1), "5minute")
    out: List[Dict[str, Any]] = []
    for c in raw or []:
        t = c["date"].astimezone(IST)
        hm = t.hour * 60 + t.minute
        if open_min <= hm < close_min:
            out.append(
                {
                    "ist": t,
                    "hm": hm,
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                }
            )
    return out


def _prev_close(kite, token: int, d: date) -> float:
    prev = d - timedelta(days=1)
    while prev.weekday() > 4:
        prev -= timedelta(days=1)
    raw = kite.historical_data(token, prev, d, "day")
    if not raw:
        return 0.0
    return float(raw[-1]["close"])


def _bb(closes: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(closes) < 20:
        return None, None, None
    window = closes[-20:]
    mid = sum(window) / 20.0
    var = sum((x - mid) ** 2 for x in window) / 20.0
    std = var**0.5
    return mid - 2 * std, mid, mid + 2 * std


def replay_sensex(kite, d: date) -> Dict[str, Any]:
    from services.sensex_constants import sensex_gap_direction_kind

    bse = kite.instruments("BSE")
    bfo = kite.instruments("BFO")
    idx = next(i for i in bse if i.get("tradingsymbol") == "SENSEX")
    idx_bars = _session_bars(kite, idx["instrument_token"], d, open_min=9 * 60 + 15, close_min=15 * 60 + 30)
    if not idx_bars:
        return {"error": "no sensex index bars", "valid_trades": []}

    prev = _prev_close(kite, idx["instrument_token"], d)
    day_open = idx_bars[0]["open"]
    gap_pct = ((day_open - prev) / prev * 100.0) if prev > 0 else 0.0
    kind = sensex_gap_direction_kind(day_open, prev)

    opts = [
        i
        for i in bfo
        if (i.get("name") or "").upper() == "SENSEX"
        and i.get("instrument_type") in ("CE", "PE")
        and i.get("expiry") and i["expiry"] >= d
    ]
    if not opts:
        return {"error": "no BFO sensex options", "valid_trades": []}
    expiry = min({i["expiry"] for i in opts})
    spot = idx_bars[0]["open"]
    atm = int(round(spot / 100.0) * 100)
    band_lo, band_hi = SENSEX_BAND
    candidates = [
        i
        for i in opts
        if i["expiry"] == expiry
        and i.get("instrument_type") == kind
        and abs(int(i.get("strike") or 0) - atm) <= 1000
    ]
    valid: List[Dict[str, Any]] = []
    for inst in candidates[:12]:
        sym = inst["tradingsymbol"]
        bars = _session_bars(kite, inst["instrument_token"], d, open_min=SENSEX_SCAN_START, close_min=SENSEX_SCAN_END)
        for b in bars:
            close = b["close"]
            if not (band_lo <= close <= band_hi):
                continue
            entry = round(close, 2)
            sl = SENSEX_SL
            if sl >= entry:
                continue
            risk = entry - sl
            target = entry + risk
            exit_px = entry
            reason = "eod"
            for fb in bars:
                if fb["hm"] <= b["hm"]:
                    continue
                if fb["low"] <= sl:
                    exit_px, reason = sl, "stop_loss"
                    break
                if fb["high"] >= target:
                    exit_px, reason = target, "target_1r"
                    break
                exit_px = fb["close"]
            valid.append(
                {
                    "segment": "sensex",
                    "time_ist": b["ist"].strftime("%H:%M"),
                    "symbol": sym,
                    "kind": kind,
                    "strike": int(inst.get("strike") or 0),
                    "entry": entry,
                    "exit": round(exit_px, 2),
                    "exit_reason": reason,
                    "gap_pct": round(gap_pct, 2),
                }
            )
            break
        if valid:
            break
    return {
        "gap_pct": round(gap_pct, 2),
        "auto_kind": kind,
        "expiry": str(expiry),
        "valid_trades": valid,
    }


def replay_nifty_bb(kite, d: date) -> Dict[str, Any]:
    from services.v2_entry_pricing import compute_strategy_entry
    from services.v2_order_guard import min_entry_confirmation_score

    nse = kite.instruments("NSE")
    nfo = kite.instruments("NFO")
    idx = next(i for i in nse if i.get("tradingsymbol") == "NIFTY 50")
    bars = _session_bars(kite, idx["instrument_token"], d, open_min=9 * 60 + 15, close_min=15 * 60 + 30)
    if len(bars) < 25:
        return {"error": "insufficient nifty bars", "valid_signals": []}

    prev = _prev_close(kite, idx["instrument_token"], d)
    min_score = min_entry_confirmation_score()
    signals: List[Dict[str, Any]] = []

    for i in range(20, len(bars)):
        b = bars[i]
        if b["hm"] < 9 * 60 + 30:
            continue
        closes = [x["close"] for x in bars[: i + 1]]
        lower, mid, upper = _bb(closes)
        spot = b["close"]
        intra: Dict[str, Any] = {
            "nifty_spot": spot,
            "prev_close": prev,
            "day_open": bars[0]["open"],
            "bb_lower": lower,
            "bb_middle": mid,
            "bb_upper": upper,
            "last_5m_close": closes[-1],
        }
        for kind in ("CE", "PE"):
            strike = int(round(spot / 50.0) * 50)
            entry = compute_strategy_entry(
                strategy_id="bb_5m_mean_reversion",
                option_kind=kind,
                quote={"ltp": spot * 0.008, "bid": spot * 0.0079, "ask": spot * 0.0081},
                spot=spot,
                strike=strike,
                delta=0.45,
                intra=intra,
                prev_close=prev,
            )
            if entry.entry_ready and entry.confirmation_score >= min_score:
                signals.append(
                    {
                        "segment": "nifty50",
                        "time_ist": b["ist"].strftime("%H:%M"),
                        "kind": kind,
                        "score": entry.confirmation_score,
                        "limit": entry.entry_limit_price,
                        "spot": round(spot, 2),
                        "bb_zone": intra,
                    }
                )
    # dedupe consecutive same kind
    deduped: List[Dict[str, Any]] = []
    last = None
    for s in signals:
        key = (s["kind"], s["time_ist"][:4])
        if key == last:
            continue
        deduped.append(s)
        last = key
    return {"min_score": min_score, "valid_signals": deduped[:10]}


def replay_commodity(kite, d: date) -> Dict[str, Any]:
    from services.commodity_entry_pricing import compute_strategy_entry
    from services.commodity_order_guard import min_entry_confirmation_score

    mcx = kite.instruments("MCX")
    fut = sorted(
        [i for i in mcx if "CRUDEOIL" in (i.get("name") or "") and i.get("instrument_type") == "FUT" and i.get("expiry")],
        key=lambda x: x["expiry"],
    )
    fut = [i for i in fut if i["expiry"] >= d]
    if not fut:
        return {"error": "no crude fut", "valid_signals": []}
    fut_inst = fut[0]
    bars = _session_bars(kite, fut_inst["instrument_token"], d, open_min=9 * 60, close_min=23 * 60 + 30)
    if len(bars) < 25:
        return {"error": "insufficient crude bars", "valid_signals": []}

    min_score = min_entry_confirmation_score()
    signals: List[Dict[str, Any]] = []
    prefix = fut_inst["tradingsymbol"][:11]  # e.g. CRUDEOILM26
    opts = [i for i in mcx if (i.get("tradingsymbol") or "").startswith(prefix.replace("FUT", "")) or prefix in (i.get("tradingsymbol") or "")]
    opts = [i for i in mcx if "CRUDEOILM26JUN" in (i.get("tradingsymbol") or "") and i.get("instrument_type") in ("CE", "PE")]

    for i in range(20, len(bars)):
        b = bars[i]
        if b["hm"] < 9 * 60 + 30:
            continue
        spot = b["close"]
        strike = int(round(spot / 50.0) * 50)
        intra = {
            "crude_spot": spot,
            "day_open": bars[0]["open"],
            "or_high": max(x["high"] for x in bars if x["hm"] <= 9 * 60 + 30),
            "or_low": min(x["low"] for x in bars if x["hm"] <= 9 * 60 + 30),
            "last_5m_close": spot,
        }
        for kind in ("CE", "PE"):
            entry = compute_strategy_entry(
                strategy_id="orb_15m_breakout",
                option_kind=kind,
                quote={"ltp": 80.0, "bid": 79.5, "ask": 80.5},
                spot=spot,
                strike=strike,
                delta=0.4,
                intra=intra,
                prev_close=bars[0]["open"],
            )
            if entry.entry_ready and entry.confirmation_score >= min_score:
                sym = next((o["tradingsymbol"] for o in opts if o.get("strike") == strike and o.get("instrument_type") == kind), f"CRUDE-{strike}-{kind}")
                signals.append(
                    {
                        "segment": "commodity",
                        "time_ist": b["ist"].strftime("%H:%M"),
                        "symbol": sym,
                        "kind": kind,
                        "score": entry.confirmation_score,
                        "spot": round(spot, 2),
                    }
                )
    return {"min_score": min_score, "valid_signals": signals[:10]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-06-15")
    args = parser.parse_args()
    d = date.fromisoformat(args.date)

    from utils.kite_utils import get_kite_instance

    kite = get_kite_instance(skip_validation=False)
    out = {
        "date": args.date,
        "weekday": d.strftime("%A"),
        "sensex": replay_sensex(kite, d),
        "nifty50": replay_nifty_bb(kite, d),
        "commodity": replay_commodity(kite, d),
    }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
