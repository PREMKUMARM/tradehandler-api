#!/usr/bin/env python3
"""Grid-search patterns on cached Dhan data to find win-rate improvements."""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.dhan_data_client import OptionSeries, load_cached_session
from services.sensex_dhan_backtest import (
    BacktestParams,
    CACHE_DIR,
    _estimate_entry,
    _load_all_sessions,
    _ref_series,
    _run_day,
    _simulate_from_entry,
)
from services.sensex_strike_selection import pick_smart_at_bar, _direction_bias_kind, _true_atm

IST = ZoneInfo("Asia/Kolkata")
SL = 10.0
MIN_TGT = 34.0


@dataclass
class FilterConfig:
    name: str
    min_entry_minutes: int = 9 * 60 + 15  # 09:15
    max_entry_minutes: int = 15 * 60
    force_kind: Optional[str] = None  # None=AUTO both, or CE/PE
    require_aligned: bool = False  # spot vs prev_close matches leg
    skip_first_n_bars: int = 0
    min_dist: int = 0
    max_dist: int = 1000
    entry_band_low: float = 17.0
    entry_band_high: float = 23.0
    prefer_mid_band: bool = False  # score lower entry prices higher
    require_close_in_band: bool = False
    min_gap_pct: Optional[float] = None  # abs gap filter
    aligned_with_day: bool = False  # leg matches index open->close direction


def _series_bar(series: OptionSeries, idx: int) -> Dict[str, Any]:
    ts = series.timestamps[idx]
    dt = datetime.fromtimestamp(ts, tz=IST)
    return {
        "idx": idx,
        "time": dt.strftime("%H:%M"),
        "minutes": dt.hour * 60 + dt.minute,
        "open": series.open[idx],
        "high": series.high[idx],
        "low": series.low[idx],
        "close": series.close[idx],
        "spot": series.spot[idx],
        "strike": int(series.strike[idx]),
    }


def run_with_filter(
    session: Dict[str, Dict[str, OptionSeries]],
    row: Dict[str, str],
    cfg: FilterConfig,
) -> Optional[Dict[str, Any]]:
    ref = _ref_series(session)
    if not ref:
        return None
    prev_close = float(row["prev_close"])
    index_open = float(row["open"])
    index_close = float(row["close"])
    day_up = index_close > index_open

    kinds: Tuple[str, ...]
    if cfg.force_kind:
        kinds = (cfg.force_kind.upper(),)
    else:
        kinds = ("CE", "PE")

    start_idx = cfg.skip_first_n_bars
    for idx in range(start_idx, len(ref.timestamps)):
        ts = ref.timestamps[idx]
        dt = datetime.fromtimestamp(ts, tz=IST)
        mins = dt.hour * 60 + dt.minute
        if mins < cfg.min_entry_minutes:
            continue
        if mins >= cfg.max_entry_minutes:
            break

        picked = pick_smart_at_bar(
            session,
            idx,
            kinds=kinds,
            band_low=cfg.entry_band_low,
            band_high=cfg.entry_band_high,
            prev_close=prev_close,
        )
        if not picked:
            continue
        candidate, series = picked
        bar = _series_bar(series, idx)

        if cfg.require_close_in_band:
            if not (cfg.entry_band_low <= bar["close"] <= cfg.entry_band_high):
                continue

        atm = _true_atm(bar["spot"])
        dist = abs(bar["strike"] - atm)
        if dist < cfg.min_dist or dist > cfg.max_dist:
            continue

        bias = _direction_bias_kind(bar["spot"], prev_close)
        if cfg.require_aligned and bias and candidate.kind != bias:
            continue

        if cfg.aligned_with_day:
            if day_up and candidate.kind != "CE":
                continue
            if not day_up and candidate.kind != "PE":
                continue

        entry = _estimate_entry(bar["open"], bar["high"], bar["low"], cfg.entry_band_low, cfg.entry_band_high)
        if entry is None:
            continue

        if cfg.prefer_mid_band and entry >= cfg.entry_band_high - 0.5:
            continue  # skip top-of-band entries

        exit_px, reason, exit_idx = _simulate_from_entry(
            entry, series, idx, "conservative", SL, MIN_TGT, MIN_TGT
        )
        r = round((exit_px - entry) / SL, 2)
        return {
            "expiry": row["expiry_date"],
            "direction": candidate.kind,
            "strike": bar["strike"],
            "dist": dist,
            "entry": entry,
            "exit": exit_px,
            "r": r,
            "reason": reason,
            "entry_time": bar["time"],
            "win": r > 0,
        }
    return None


def evaluate(cfg: FilterConfig, sessions_data: Dict[str, Any]) -> Dict[str, Any]:
    rows = _load_all_sessions()
    trades: List[Dict[str, Any]] = []
    skipped = 0
    for row in rows:
        exp = row["expiry_date"]
        session = sessions_data.get(exp)
        if not session:
            skipped += 1
            continue
        t = run_with_filter(session, row, cfg)
        if t:
            trades.append(t)
        else:
            skipped += 1
    wins = sum(1 for t in trades if t["win"])
    n = len(trades)
    total_r = sum(t["r"] for t in trades)
    return {
        "name": cfg.name,
        "trades": n,
        "skipped": skipped,
        "wins": wins,
        "losses": n - wins,
        "win_rate": round(wins / n * 100, 1) if n else 0,
        "total_r": round(total_r, 1),
        "avg_r": round(total_r / n, 2) if n else 0,
        "trade_list": trades,
    }


def main() -> None:
    rows = _load_all_sessions()
    data = {}
    for row in rows:
        exp = row["expiry_date"]
        cached = load_cached_session(CACHE_DIR, exp)
        if cached:
            data[exp] = cached

    configs = [
        FilterConfig("baseline (current smart)"),
        FilterConfig("skip 09:15, enter from 09:30", min_entry_minutes=9 * 60 + 30),
        FilterConfig("enter from 10:00 only", min_entry_minutes=10 * 60),
        FilterConfig("enter from 10:30 only", min_entry_minutes=10 * 60 + 30),
        FilterConfig("force aligned CE/PE only", require_aligned=True),
        FilterConfig("aligned + from 09:30", require_aligned=True, min_entry_minutes=9 * 60 + 30),
        FilterConfig("aligned + from 10:00", require_aligned=True, min_entry_minutes=10 * 60),
        FilterConfig("skip top-of-band (entry < 22.5)", prefer_mid_band=True),
        FilterConfig("close must be in band", require_close_in_band=True),
        FilterConfig("close in band + 09:30", require_close_in_band=True, min_entry_minutes=9 * 60 + 30),
        FilterConfig("max dist 800 pts", max_dist=800),
        FilterConfig("max dist 700 pts", max_dist=700),
        FilterConfig("dist 600-900", min_dist=600, max_dist=900),
        FilterConfig("entry band 17-20 only", entry_band_high=20.0),
        FilterConfig("entry band 20-23 only", entry_band_low=20.0),
        FilterConfig("CE only", force_kind="CE"),
        FilterConfig("PE only", force_kind="PE"),
        FilterConfig("aligned with day close", aligned_with_day=True),
        FilterConfig("PE + 09:30", force_kind="PE", min_entry_minutes=9 * 60 + 30),
        FilterConfig("CE + 09:30", force_kind="CE", min_entry_minutes=9 * 60 + 30),
        FilterConfig("aligned + close in band + 10:00", require_aligned=True, require_close_in_band=True, min_entry_minutes=10 * 60),
        FilterConfig("skip 1st bar + aligned", skip_first_n_bars=1, require_aligned=True),
        FilterConfig("skip 2 bars + aligned", skip_first_n_bars=2, require_aligned=True),
    ]

    results = [evaluate(c, data) for c in configs]
    results.sort(key=lambda r: (-r["win_rate"], -r["total_r"], -r["trades"]))

    print("=" * 90)
    print("PATTERN SEARCH — sorted by win rate (cached 10 sessions, smart strike base)")
    print("=" * 90)
    print(f"{'pattern':<42} {'trades':>6} {'W/L':>8} {'win%':>6} {'totR':>6} {'avgR':>6}")
    print("-" * 90)
    for r in results:
        wl = f"{r['wins']}/{r['losses']}"
        print(f"{r['name']:<42} {r['trades']:>6} {wl:>8} {r['win_rate']:>5.1f}% {r['total_r']:>+6.1f} {r['avg_r']:>+6.2f}")

    best = results[0]
    print("\n" + "=" * 90)
    print(f"BEST PATTERN: {best['name']}")
    print(f"  Win rate: {best['win_rate']}% ({best['wins']}W / {best['losses']}L on {best['trades']} trades)")
    print(f"  Total R: {best['total_r']:+.1f}  Avg R: {best['avg_r']:+.2f}")
    if best.get("trade_list"):
        print("  Trades:")
        for t in best["trade_list"]:
            tag = "WIN " if t["win"] else "LOSS"
            print(f"    {t['expiry']} {tag} {t['direction']} {t['strike']} dist={t['dist']} entry={t['entry']} R={t['r']:+.0f} @ {t['entry_time']}")

    # Win/loss feature analysis on baseline trades
    print("\n" + "=" * 90)
    print("WIN vs LOSS FEATURE PROFILE (baseline trades)")
    baseline = next(r for r in results if r["name"].startswith("baseline"))
    wins = [t for t in baseline["trade_list"] if t["win"]]
    losses = [t for t in baseline["trade_list"] if not t["win"]]
    for label, group in [("WINNERS", wins), ("LOSERS", losses)]:
        if not group:
            continue
        avg_dist = sum(t["dist"] for t in group) / len(group)
        avg_entry = sum(t["entry"] for t in group) / len(group)
        ce = sum(1 for t in group if t["direction"] == "CE")
        print(f"  {label} (n={len(group)}): avg_dist={avg_dist:.0f} avg_entry={avg_entry:.1f} CE={ce} PE={len(group)-ce}")

    out = ROOT / "data" / "sensex" / "winrate_pattern_search.json"
    out.write_text(json.dumps([{k: v for k, v in r.items() if k != "trade_list"} for r in results], indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
