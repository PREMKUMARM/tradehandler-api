#!/usr/bin/env python3
"""Grid-search Nifty50 Dhan entries + exits for high win-rate configs."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from services.nifty_dhan_backtest import _resolve_sessions, _run_backtest_for_timeframe, _store
from services.sensex_dhan_backtest import BacktestParams


@dataclass
class Cfg:
    name: str
    env: Dict[str, str]
    band_low: float = 17.0
    band_high: float = 23.0


def _run(cfg: Cfg, base: BacktestParams) -> Dict[str, Any]:
    for k, v in cfg.env.items():
        os.environ[k] = v
    params = BacktestParams(
        capital=base.capital,
        sl_inr=base.sl_inr,
        direction=base.direction,
        start_date=base.start_date,
        end_date=base.end_date,
        segment=base.segment,
        entry_scan_start_ist=base.entry_scan_start_ist,
        entry_scan_end_ist=base.entry_scan_end_ist,
        entry_band_low=cfg.band_low,
        entry_band_high=cfg.band_high,
        refresh_dhan=False,
    )
    sessions = _resolve_sessions(params)
    report, _ = _run_backtest_for_timeframe(params, sessions, 5, store=_store())
    s = report["summary"]
    trades = report["trades"]
    wins = len([t for t in trades if t["pnl_inr"] > 0])
    losses = len([t for t in trades if t["pnl_inr"] < 0])
    be = len([t for t in trades if t["pnl_inr"] == 0])
    n = len(trades)
    return {
        "name": cfg.name,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "breakeven": be,
        "win_rate": round(100.0 * wins / n, 1) if n else 0.0,
        "non_loss_rate": round(100.0 * (wins + be) / n, 1) if n else 0.0,
        "pnl": s["total_pnl_inr"],
        "avg_r": s["avg_r"],
        "trade_list": trades,
    }


def main() -> None:
    base = BacktestParams(
        capital=100_000,
        sl_inr=9.0,
        direction="AUTO",
        start_date="2026-05-22",
        end_date="2026-06-19",
        segment="nifty50",
        entry_scan_start_ist="14:00",
        entry_scan_end_ist="15:00",
    )

    common = {
        "ENTRY_BLOCK_REENTRY_AFTER_LOSS": "true",
        "ENTRY_BLOCK_REENTRY_AFTER_BREAKEVEN": "true",
        "MAX_TRADES_PER_CONTRACT_PER_DAY": "1",
        "EXIT_T1_CLOSE_CONFIRM": "false",
    }

    configs: List[Cfg] = []
    combos = [
        ("t1_scalp", "true", "true", "true", "false", 18, 21, "mid"),
        ("t1_scalp", "true", "true", "false", "false", 18, 21, "mid"),
        ("t1_scalp", "true", "false", "true", "false", 17, 23, "full"),
        ("t1_scalp", "true", "false", "false", "false", 17, 23, "full"),
        ("t1_scalp", "false", "true", "false", "false", 17, 23, "full"),
        ("t1_scalp", "false", "false", "false", "false", 17, 23, "full"),
        ("t1_scalp", "true", "true", "true", "true", 18, 21, "mid"),
        ("t1_scalp", "true", "true", "false", "true", 18, 21, "mid"),
        ("stepped", "true", "true", "false", "false", 18, 21, "mid"),
        ("stepped", "true", "false", "false", "false", 17, 23, "full"),
        ("stepped", "false", "false", "false", "false", 17, 23, "full"),
        ("t1_scalp", "true", "true", "true", "false", 17.5, 20.5, "tight"),
        ("t1_scalp", "true", "false", "true", "true", 18, 21, "mid"),
        ("t1_scalp", "false", "true", "true", "false", 18, 21, "mid"),
        ("t1_scalp", "true", "true", "false", "false", 17.5, 20.5, "tight"),
        ("t1_scalp", "true", "false", "false", "true", 17, 23, "full"),
    ]
    for exit_m, conf, idx, direction, day, blo, bhi, bl in combos:
        configs.append(
            Cfg(
                name=f"{exit_m}|conf={conf}|idx={idx}|dir={direction}|day={day}|{bl}",
                env={
                    **common,
                    "EXIT_MODEL": exit_m,
                    "ENTRY_REQUIRE_CONFIRMATION_CANDLE": conf,
                    "ENTRY_REQUIRE_INDEX_MOMENTUM": idx,
                    "ENTRY_REQUIRE_DIRECTION_ALIGNED": direction,
                    "ENTRY_REQUIRE_DAY_ALIGNED": day,
                },
                band_low=blo,
                band_high=bhi,
            )
        )

    results = [_run(c, base) for c in configs]
    hits = [r for r in results if r["trades"] >= 1 and r["win_rate"] >= 90]
    hits.sort(key=lambda x: (-x["win_rate"], -x["trades"], -x["pnl"]))

    print("=" * 105)
    print("Nifty50 win-rate search (20 cached sessions, ₹1L)")
    print("=" * 105)
    if not hits:
        print("No config hit 90% win rate with trades>=1. Top by win_rate:")
        alt = sorted([r for r in results if r["trades"] >= 1], key=lambda x: (-x["win_rate"], -x["trades"]))[:20]
        hits = alt

    print(f"{'config':<55} {'tr':>3} {'W/L/BE':>9} {'win%':>6} {'nl%':>6} {'pnl':>8}")
    print("-" * 105)
    for r in hits[:25]:
        wl = f"{r['wins']}/{r['losses']}/{r['breakeven']}"
        print(
            f"{r['name']:<55} {r['trades']:>3} {wl:>9} {r['win_rate']:>5.1f}% "
            f"{r['non_loss_rate']:>5.1f}% {r['pnl']:>8.0f}"
        )

    if hits and hits[0]["win_rate"] >= 90:
        b = hits[0]
        print(f"\n*** TARGET: {b['name']} → {b['win_rate']}% ({b['trades']} trades, P&L ₹{b['pnl']:,.0f})")
        for t in b.get("trade_list", []):
            print(f"  {t['expiry_date']} {t['kind']} {t['strike']} ₹{t['entry']:.1f}→₹{t['exit']:.1f} {t['exit_reason']}")


if __name__ == "__main__":
    main()
