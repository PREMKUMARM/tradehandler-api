#!/usr/bin/env python3
"""Debug Sensex backtest entry/strike selection per session."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.dhan_data_client import OptionSeries, load_cached_session
from services.sensex_dhan_backtest import (
    BacktestParams,
    _estimate_entry,
    _max_oi_at_bar,
    _pick_entry_auto,
    _run_day,
    _simulate_from_entry,
)

IST = ZoneInfo("Asia/Kolkata")
CACHE = ROOT / "data" / "sensex" / "dhan_intraday"
OHLC = ROOT / "data" / "sensex" / "weekly_expiry_day_ohlc.csv"
BAND = (17.0, 23.0)
SL = 10.0
MIN_TGT = 34.0


def _load_rows() -> List[Dict[str, str]]:
    with OHLC.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bar_info(series: OptionSeries, idx: int) -> Dict[str, Any]:
    ts = series.timestamps[idx]
    dt = datetime.fromtimestamp(ts, tz=IST)
    return {
        "time": dt.strftime("%H:%M"),
        "open": series.open[idx],
        "high": series.high[idx],
        "low": series.low[idx],
        "close": series.close[idx],
        "oi": series.oi[idx],
        "spot": series.spot[idx],
        "strike": int(series.strike[idx]),
        "offset": series.offset,
        "kind": series.kind,
    }


def _candidates_at_bar(session: Dict[str, Dict[str, OptionSeries]], idx: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for kind in ("CE", "PE"):
        for offset, series in (session.get(kind) or {}).items():
            if idx >= len(series.oi):
                continue
            lo, hi = series.low[idx], series.high[idx]
            touched = lo <= BAND[1] and hi >= BAND[0]
            entry_est = _estimate_entry(series.open[idx], hi, lo, *BAND)
            rows.append(
                {
                    "kind": kind,
                    "offset": offset,
                    "strike": int(series.strike[idx]),
                    "oi": series.oi[idx],
                    "open": series.open[idx],
                    "high": hi,
                    "low": lo,
                    "close": series.close[idx],
                    "band_touched": touched,
                    "entry_est": entry_est,
                    "in_band_close": BAND[0] <= series.close[idx] <= BAND[1],
                }
            )
    rows.sort(key=lambda r: (-r["oi"], r["kind"], r["offset"]))
    return rows


def _first_band_times(session, ref_idx_limit: int) -> List[Dict[str, Any]]:
    """When does each offset first touch ₹17–23 band (any bar)?"""
    ref = session.get("CE", {}).get("ATM") or session.get("PE", {}).get("ATM")
    if not ref:
        return []
    out = []
    for idx in range(min(ref_idx_limit + 1, len(ref.timestamps))):
        dt = datetime.fromtimestamp(ref.timestamps[idx], tz=IST)
        if dt.hour * 60 + dt.minute >= 15 * 60:
            break
        for kind in ("CE", "PE"):
            for offset, series in (session.get(kind) or {}).items():
                if idx >= len(series.low):
                    continue
                if series.low[idx] <= BAND[1] and series.high[idx] >= BAND[0]:
                    out.append(
                        {
                            "time": dt.strftime("%H:%M"),
                            "idx": idx,
                            "kind": kind,
                            "offset": offset,
                            "strike": int(series.strike[idx]),
                            "oi": series.oi[idx],
                            "low": series.low[idx],
                            "high": series.high[idx],
                            "close": series.close[idx],
                        }
                    )
    return out


def _alt_outcomes(session, bar_idx: int, entry: float, series: OptionSeries) -> Dict[str, Any]:
    """What if we used close as entry instead of estimate?"""
    close_entry = round(series.close[bar_idx], 2)
    for label, px in [("estimated", entry), ("bar_close", close_entry), ("bar_open", round(series.open[bar_idx], 2))]:
        ex, reason, _ = _simulate_from_entry(px, series, bar_idx, "conservative", SL, MIN_TGT, MIN_TGT)
        yield label, px, ex, reason


def analyze_day(row: Dict[str, str]) -> Dict[str, Any]:
    expiry = row["expiry_date"]
    session = load_cached_session(CACHE, expiry)
    if not session:
        return {"expiry": expiry, "error": "no cache"}

    params = BacktestParams(direction="AUTO")
    trade = _run_day(expiry, float(row["open"]), float(row["prev_close"]), session, params, "conservative")
    if not trade:
        return {"expiry": expiry, "error": "no trade picked"}

    picked = _pick_entry_auto(session, *BAND, 15 * 60)
    if not picked:
        return {"expiry": expiry, "error": "pick failed"}
    bar, source, series = picked
    idx = bar.idx

    max_oi = _max_oi_at_bar(session, idx)
    all_at_entry = _candidates_at_bar(session, idx)

    # ATM alternatives at same bar
    spot = series.spot[idx]
    atm_strike = round(spot / 100) * 100
    atm_ce = session.get("CE", {}).get("ATM")
    atm_pe = session.get("PE", {}).get("ATM")

    def _atm_at(kind: str) -> Optional[Dict]:
        s = session.get(kind, {}).get("ATM")
        if not s or idx >= len(s.close):
            return None
        return _bar_info(s, idx)

    # Distance from spot
    strike = trade.strike
    moneyness_pts = strike - atm_strike
    if trade.direction == "CE":
        moneyness = f"{'ITM' if strike < atm_strike else 'OTM' if strike > atm_strike else 'ATM'} {abs(moneyness_pts)}pts"
    else:
        moneyness = f"{'ITM' if strike > atm_strike else 'OTM' if strike < atm_strike else 'ATM'} {abs(moneyness_pts)}pts"

    # Index day move
    index_open = float(row["open"])
    index_close = float(row["close"])
    index_chg = index_close - index_open

    # Premium path after entry (next 6 bars = 30 min)
    path = []
    for j in range(idx, min(idx + 7, len(series.timestamps))):
        path.append(
            {
                "time": datetime.fromtimestamp(series.timestamps[j], tz=IST).strftime("%H:%M"),
                "low": series.low[j],
                "high": series.high[j],
                "close": series.close[j],
            }
        )

    # Did SL hit same bar as entry?
    same_bar_sl = series.low[idx] <= trade.entry - SL

    # Band touch quality: was close in band or only wick?
    close_in_band = BAND[0] <= series.close[idx] <= BAND[1]

    alts = list(_alt_outcomes(session, idx, trade.entry, series))

    # Compare: if we picked ATM on same side when in band
    atm_same = _atm_at(trade.direction)
    atm_alt = None
    if atm_same and atm_same["low"] <= BAND[1] and atm_same["high"] >= BAND[0]:
        ae = _estimate_entry(atm_same["open"], atm_same["high"], atm_same["low"], *BAND)
        if ae:
            ex, reason, _ = _simulate_from_entry(ae, session[trade.direction]["ATM"], idx, "conservative", SL, MIN_TGT, MIN_TGT)
            atm_alt = {"entry": ae, "exit": ex, "reason": reason}

    # Top OI at entry vs band-in-band candidates
    in_band_candidates = [c for c in all_at_entry if c["band_touched"]]
    max_oi_in_band = max(in_band_candidates, key=lambda c: c["oi"]) if in_band_candidates else None

    return {
        "expiry": expiry,
        "index_open": index_open,
        "index_close": index_close,
        "index_chg_pts": round(index_chg, 1),
        "index_chg_pct": round(index_chg / index_open * 100, 2),
        "trade": {
            "direction": trade.direction,
            "strike": trade.strike,
            "source": trade.strike_source,
            "entry": trade.entry,
            "exit": trade.exit,
            "r": trade.r_multiple,
            "reason": trade.exit_reason,
            "entry_time": trade.entry_datetime_ist.split(" ")[1][:5],
            "exit_time": trade.exit_datetime_ist.split(" ")[1][:5] if trade.exit_datetime_ist else "",
        },
        "entry_bar": _bar_info(series, idx),
        "spot_at_entry": round(spot, 1),
        "atm_strike": atm_strike,
        "moneyness": moneyness,
        "max_oi_pick": {"kind": max_oi[0], "offset": max_oi[1], "oi": series.oi[idx]} if max_oi else None,
        "close_in_band": close_in_band,
        "same_bar_sl": same_bar_sl,
        "bar_range": round(series.high[idx] - series.low[idx], 2),
        "premium_path_30m": path,
        "entry_alternatives": [{"label": a[0], "entry": a[1], "exit": a[2], "reason": a[3]} for a in alts],
        "atm_same_side_alt": atm_alt,
        "max_oi_in_band_candidate": max_oi_in_band,
        "top3_oi_at_entry": all_at_entry[:3],
        "first_band_touch_count": len(_first_band_times(session, idx)),
    }


def main() -> None:
    rows = _load_rows()
    reports = []
    for row in rows:
        reports.append(analyze_day(row))

    print("=" * 80)
    print("SENSEX BACKTEST ENTRY DEBUG — AUTO (highest OI)")
    print("=" * 80)

    issues = {
        "wrong_side_vs_index": 0,
        "far_otm": 0,
        "wick_only_entry": 0,
        "same_bar_sl": 0,
        "entry_at_band_edge_23": 0,
        "sl_before_trail_possible": 0,
    }

    for r in reports:
        if r.get("error"):
            print(f"\n{r['expiry']}: ERROR — {r['error']}")
            continue

        t = r["trade"]
        eb = r["entry_bar"]
        print(f"\n{'─' * 80}")
        print(f"{r['expiry']}  |  Index {r['index_open']:.0f} → {r['index_close']:.0f} ({r['index_chg_pts']:+.0f} pts, {r['index_chg_pct']:+.2f}%)")
        print(f"PICK: {t['direction']} {t['strike']} {t['source']} @ ₹{t['entry']:.2f} ({t['entry_time']}) → ₹{t['exit']:.2f} ({t['r']:+.0f}R {t['reason']})")
        print(f"Spot {r['spot_at_entry']:.0f} · ATM {r['atm_strike']} · Moneyness: {r['moneyness']}")
        print(
            f"Entry bar O/H/L/C: ₹{eb['open']:.2f}/₹{eb['high']:.2f}/₹{eb['low']:.2f}/₹{eb['close']:.2f} · "
            f"OI {eb['oi']:,.0f} · range ₹{r['bar_range']:.2f}"
        )
        print(f"Close in band: {r['close_in_band']} · Same-bar SL possible: {r['same_bar_sl']}")

        # Wrong side vs index move
        idx_up = r["index_chg_pts"] > 0
        bought_ce = t["direction"] == "CE"
        aligned = (idx_up and bought_ce) or (not idx_up and not bought_ce)
        if not aligned:
            issues["wrong_side_vs_index"] += 1
            print(f"⚠️  DIRECTION MISMATCH: index moved {'UP' if idx_up else 'DOWN'} but bought {t['direction']}")

        if "OTM" in r["moneyness"] and int(r["moneyness"].split()[-1].replace("pts", "")) >= 500:
            issues["far_otm"] += 1
            print(f"⚠️  FAR OTM strike ({r['moneyness']}) — MAX_OI often = deep OTM lottery ticket")

        if not r["close_in_band"]:
            issues["wick_only_entry"] += 1
            print(f"⚠️  WICK-ONLY ENTRY: bar touched band but close ₹{eb['close']:.2f} outside ₹17–23")

        if r["same_bar_sl"]:
            issues["same_bar_sl"] += 1
            print(f"⚠️  SAME-BAR SL: low ₹{eb['low']:.2f} already below SL ₹{t['entry'] - SL:.2f}")

        if t["entry"] >= 22.5:
            issues["entry_at_band_edge_23"] += 1
            print(f"⚠️  ENTRY AT TOP OF BAND (₹{t['entry']:.2f}) — worst R:R for ₹10 SL")

        if t["reason"] == "stop_loss":
            issues["sl_before_trail_possible"] += 1

        print("Top OI at entry bar:")
        for c in r["top3_oi_at_entry"]:
            flag = " ← PICKED" if c["strike"] == t["strike"] and c["kind"] == t["direction"] else ""
            band = "IN_BAND" if c["band_touched"] else "no_band"
            print(
                f"  {c['kind']} {c['offset']} strike {c['strike']} OI {c['oi']:,.0f} "
                f"L/H ₹{c['low']:.1f}/₹{c['high']:.1f} close ₹{c['close']:.1f} [{band}]{flag}"
            )

        if r["atm_same_side_alt"]:
            a = r["atm_same_side_alt"]
            print(f"ATM {t['direction']} alt @ ₹{a['entry']:.2f} → ₹{a['exit']:.2f} ({a['reason']})")

        print("Entry price sensitivity:")
        for a in r["entry_alternatives"]:
            print(f"  {a['label']:12} ₹{a['entry']:.2f} → ₹{a['exit']:.2f} ({a['reason']}, {((a['exit']-a['entry'])/SL):+.1f}R)")

        print("Premium next 30m:", " → ".join(f"{p['time']} L{p['low']:.0f}/H{p['high']:.0f}" for p in r["premium_path_30m"][:4]))

    print(f"\n{'=' * 80}")
    print("SUMMARY OF SYSTEMIC ISSUES (10 sessions)")
    print(f"  Direction vs index day move mismatch: {issues['wrong_side_vs_index']}/10")
    print(f"  Wick-only entry (close outside band):  {issues['wick_only_entry']}/10")
    print(f"  Entry at band top (~₹23):             {issues['entry_at_band_edge_23']}/10")
    print(f"  Same-bar SL possible on entry bar:     {issues['same_bar_sl']}/10")
    print(f"  Far OTM (≥500 pts from ATM):           {issues['far_otm']}/10")
    print(f"  Stop loss exits:                       {issues['sl_before_trail_possible']}/10")
    print("=" * 80)

    out = ROOT / "data" / "sensex" / "entry_debug_report.json"
    out.write_text(json.dumps(reports, indent=2, default=str))
    print(f"Full JSON: {out}")


if __name__ == "__main__":
    main()
