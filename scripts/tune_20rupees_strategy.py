#!/usr/bin/env python3
"""Grid-search 20rupees entry/exit filters — minimize SL + breakeven exits."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.nifty_dhan_backtest import (
    LOT_SIZE as NIFTY_LOT,
    _contract_key,
    _resolve_sessions as nifty_sessions,
    _store as nifty_store,
    load_session_from_duckdb,
)
from services.sensex_constants import sensex_max_lots_per_trade
from services.sensex_dhan_backtest import (
    LOT_SIZE as SENSEX_LOT,
    BacktestParams,
    _backtest_size_from_risk,
    _resolve_backtest_sessions,
    _run_day,
    load_cached_session,
)
from services.sensex_trading_calendar import session_index_from_spot, session_spot_series

CACHE_DIR = ROOT / "data" / "sensex" / "dhan_intraday"


@dataclass
class Cfg:
    name: str
    env: Dict[str, str]
    band_low: float
    band_high: float
    scan_end: str


def _eval_segment(
    segment: str,
    sessions: List[Dict],
    cfg: Cfg,
    base: BacktestParams,
    lot_size: int,
) -> Dict[str, Any]:
    for k, v in cfg.env.items():
        os.environ[k] = v
    os.environ["SENSEX_ENTRY_CUTOFF_HOUR"] = cfg.scan_end.split(":")[0]
    os.environ["SENSEX_ENTRY_CUTOFF_MINUTE"] = cfg.scan_end.split(":")[1]

    params = BacktestParams(
        capital=base.capital,
        risk_pct=base.risk_pct,
        sl_inr=base.sl_inr,
        direction=base.direction,
        entry_band_low=cfg.band_low,
        entry_band_high=cfg.band_high,
        segment=segment,
        entry_scan_start_ist=base.entry_scan_start_ist,
        entry_scan_end_ist=cfg.scan_end,
    )
    equity = float(params.capital)
    trades: List[Dict[str, Any]] = []
    prev_spot = 0.0
    store = nifty_store() if segment == "nifty50" else None

    for row in sessions:
        sd = row["session_date"]
        if segment == "nifty50":
            session = load_session_from_duckdb(store, sd, interval_min=5)
        else:
            session = load_cached_session(CACHE_DIR, sd, 5)
        if not session:
            continue
        index_open, _, prev_close = session_index_from_spot(session, prev_trading_close=prev_spot)
        if index_open <= 0:
            continue
        for tr in _run_day(sd, index_open, prev_close, session, params):
            lots, qty, _ = _backtest_size_from_risk(
                equity,
                params.risk_pct,
                tr.entry,
                params.sl_inr,
                lot_size,
                sensex_max_lots_per_trade(),
            )
            pnl = round((tr.exit - tr.entry) * qty, 2)
            equity = round(equity + pnl, 2)
            trades.append(
                {
                    "pnl_inr": pnl,
                    "exit_reason": tr.exit_reason,
                    "r_multiple": tr.r_multiple,
                }
            )
        spot_s = session_spot_series(session)
        if spot_s and spot_s.spot:
            prev_spot = float(spot_s.spot[-1])

    n = len(trades)
    wins = sum(1 for t in trades if t["pnl_inr"] > 0)
    losses = sum(1 for t in trades if t["pnl_inr"] < 0)
    be = sum(1 for t in trades if t["pnl_inr"] == 0)
    sl_hits = sum(1 for t in trades if t["exit_reason"] == "stop_loss")
    be_trail = sum(1 for t in trades if t["exit_reason"] == "trail_t1")
    pnl = round(sum(t["pnl_inr"] for t in trades), 2)
    ret = round(100 * pnl / base.capital, 1) if base.capital else 0
    # Penalize full losses and wasted T1→breakeven setups
    score = pnl - sl_hits * 8000 - be_trail * 3000 + wins * 500
    return {
        "segment": segment,
        "name": cfg.name,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "breakeven": be,
        "stop_loss": sl_hits,
        "trail_t1": be_trail,
        "pnl": pnl,
        "return_pct": ret,
        "score": score,
    }


def main() -> None:
    end = date.today()
    start = end - timedelta(days=90)
    base = BacktestParams(
        capital=100_000,
        risk_pct=10.0,
        sl_inr=9.0,
        direction="AUTO",
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        entry_scan_start_ist="14:00",
        entry_scan_end_ist="15:00",
    )

    nifty_sess = nifty_sessions(base)
    sensex_sess = _resolve_backtest_sessions(base)

    common = {
        "ENTRY_BLOCK_REENTRY_AFTER_LOSS": "true",
        "ENTRY_BLOCK_REENTRY_AFTER_BREAKEVEN": "true",
        "MAX_TRADES_PER_CONTRACT_PER_DAY": "1",
        "MAX_TRADES_PER_SESSION_DAY": "1",
        "ENTRY_SCAN_WARMUP_MIN": "5",
        "ENTRY_MIN_DAY_MOVE_PTS": "30",
        "ENTRY_CHASE_DAY_PTS": "250",
        "ENTRY_CHASE_MAX_BAR_MOVE_PTS": "10",
        "ENTRY_CHASE_MIN_BAR_MOVE_PTS": "25",
        "ENTRY_PE_BOUNCE_RECOVERY_MAX_PCT": "50",
        "SENSEX_ENTRY_CUTOFF_HOUR": "14",
        "SENSEX_ENTRY_CUTOFF_MINUTE": "45",
        "EXIT_T1_CLOSE_CONFIRM": "false",
    }

    combos: List[Cfg] = []
    # Focused grid — high-impact filters only (24 configs)
    presets = [
        ("baseline", {"ENTRY_REQUIRE_CONFIRMATION_CANDLE": "false", "ENTRY_REQUIRE_MOMENTUM_BAR": "false",
                      "ENTRY_REQUIRE_INDEX_MOMENTUM": "false", "ENTRY_REQUIRE_DAY_ALIGNED": "false",
                      "ENTRY_MID_BAND_ONLY": "false"}, 17, 23, "15:00"),
        ("t1+mid+end", {"ENTRY_REQUIRE_CONFIRMATION_CANDLE": "false", "ENTRY_REQUIRE_MOMENTUM_BAR": "true",
                        "ENTRY_REQUIRE_INDEX_MOMENTUM": "true", "ENTRY_REQUIRE_DAY_ALIGNED": "true",
                        "ENTRY_MID_BAND_ONLY": "true"}, 18, 21, "14:45"),
        ("t1+confirm+mid", {"ENTRY_REQUIRE_CONFIRMATION_CANDLE": "true", "ENTRY_REQUIRE_MOMENTUM_BAR": "false",
                            "ENTRY_REQUIRE_INDEX_MOMENTUM": "true", "ENTRY_REQUIRE_DAY_ALIGNED": "true",
                            "ENTRY_MID_BAND_ONLY": "true"}, 18, 21, "14:45"),
        ("t1+momentum", {"ENTRY_REQUIRE_CONFIRMATION_CANDLE": "false", "ENTRY_REQUIRE_MOMENTUM_BAR": "true",
                         "ENTRY_REQUIRE_INDEX_MOMENTUM": "true", "ENTRY_REQUIRE_DAY_ALIGNED": "false",
                         "ENTRY_MID_BAND_ONLY": "false"}, 17.5, 20.5, "14:45"),
        ("t1+day+mid", {"ENTRY_REQUIRE_CONFIRMATION_CANDLE": "false", "ENTRY_REQUIRE_MOMENTUM_BAR": "false",
                        "ENTRY_REQUIRE_INDEX_MOMENTUM": "false", "ENTRY_REQUIRE_DAY_ALIGNED": "true",
                        "ENTRY_MID_BAND_ONLY": "true"}, 18, 21, "14:45"),
        ("t1+tight+all", {"ENTRY_REQUIRE_CONFIRMATION_CANDLE": "true", "ENTRY_REQUIRE_MOMENTUM_BAR": "true",
                           "ENTRY_REQUIRE_INDEX_MOMENTUM": "true", "ENTRY_REQUIRE_DAY_ALIGNED": "true",
                           "ENTRY_MID_BAND_ONLY": "true"}, 17.5, 20.5, "14:45"),
    ]
    for exit_m in ("t1_scalp", "stepped"):
        for label, extra, blo, bhi, scan_end in presets:
            combos.append(
                Cfg(
                    name=f"{exit_m}|{label}",
                    env={**common, "EXIT_MODEL": exit_m, **extra},
                    band_low=blo,
                    band_high=bhi,
                    scan_end=scan_end,
                )
            )

    print(f"Grid: {len(combos)} configs × 2 segments ({len(nifty_sess)} nifty / {len(sensex_sess)} sensex sessions)")

    combined: Dict[str, Dict[str, Any]] = {}
    for cfg in combos:
        n = _eval_segment("nifty50", nifty_sess, cfg, base, NIFTY_LOT)
        s = _eval_segment("sensex", sensex_sess, cfg, base, SENSEX_LOT)
        key = cfg.name
        combined[key] = {
            "name": cfg.name,
            "env": cfg.env,
            "band": [cfg.band_low, cfg.band_high],
            "scan_end": cfg.scan_end,
            "nifty": n,
            "sensex": s,
            "total_trades": n["trades"] + s["trades"],
            "total_sl": n["stop_loss"] + s["stop_loss"],
            "total_trail_t1": n["trail_t1"] + s["trail_t1"],
            "total_pnl": n["pnl"] + s["pnl"],
            "score": n["score"] + s["score"],
        }

    ranked = sorted(
        [v for v in combined.values() if v["total_trades"] >= 3],
        key=lambda x: (-x["score"], -x["total_pnl"], x["total_sl"] + x["total_trail_t1"]),
    )

    print("\nTop 15 configs (min 8 trades combined, scored for PnL − SL/BE penalty):\n")
    for r in ranked[:15]:
        print(
            f"  {r['name'][:70]}"
            f"\n    Nifty: {r['nifty']['trades']} tr SL={r['nifty']['stop_loss']} BE={r['nifty']['trail_t1']} "
            f"PnL ₹{r['nifty']['pnl']:,.0f} | Sensex: {r['sensex']['trades']} tr SL={r['sensex']['stop_loss']} "
            f"BE={r['sensex']['trail_t1']} PnL ₹{r['sensex']['pnl']:,.0f} | Total ₹{r['total_pnl']:,.0f}"
        )

    if ranked:
        best = ranked[0]
        out = ROOT / "data" / "market" / "tuned_20rupees_config.json"
        out.write_text(json.dumps(best, indent=2))
        print(f"\nSaved best → {out}")


if __name__ == "__main__":
    main()
