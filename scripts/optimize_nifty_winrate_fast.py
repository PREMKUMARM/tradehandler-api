#!/usr/bin/env python3
"""Fast win-rate grid on cached DuckDB only (no Dhan API)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.sensex_constants import sensex_max_lots_per_trade
from services.nifty_dhan_backtest import (
    LOT_SIZE,
    _contract_key,
    _resolve_sessions,
    _store,
    load_session_from_duckdb,
)
from services.sensex_dhan_backtest import (
    BacktestParams,
    _backtest_size_from_risk,
    _run_day,
    resolve_backtest_scan_window,
)
from services.sensex_trading_calendar import session_index_from_spot, session_spot_series


@dataclass
class Cfg:
    name: str
    env: Dict[str, str]
    band_low: float
    band_high: float


def _eval_cfg(cfg: Cfg, sessions: List[Dict], store, base: BacktestParams) -> Dict[str, Any]:
    for k, v in cfg.env.items():
        os.environ[k] = v
    params = BacktestParams(
        capital=base.capital,
        sl_inr=base.sl_inr,
        direction=base.direction,
        entry_band_low=cfg.band_low,
        entry_band_high=cfg.band_high,
        segment="nifty50",
        entry_scan_start_ist=base.entry_scan_start_ist,
        entry_scan_end_ist=base.entry_scan_end_ist,
    )
    equity = float(params.capital)
    trades: List[Dict[str, Any]] = []
    prev_spot = 0.0
    for row in sessions:
        sd = row["session_date"]
        session = load_session_from_duckdb(store, sd, interval_min=5)
        if not session:
            continue
        index_open, _, prev_close = session_index_from_spot(session, prev_trading_close=prev_spot)
        if index_open <= 0:
            continue
        for tr in _run_day(sd, index_open, prev_close, session, params):
            tr.symbol = _contract_key(tr.kind, tr.strike)
            lots, qty, _ = _backtest_size_from_risk(
                equity, params.risk_pct, tr.entry, params.sl_inr, LOT_SIZE, sensex_max_lots_per_trade()
            )
            pnl = round((tr.exit - tr.entry) * qty, 2)
            equity = round(equity + pnl, 2)
            trades.append(
                {
                    "expiry_date": sd,
                    "kind": tr.kind,
                    "strike": tr.strike,
                    "entry": tr.entry,
                    "exit": tr.exit,
                    "pnl_inr": pnl,
                    "exit_reason": tr.exit_reason,
                    "r_multiple": tr.r_multiple,
                }
            )
        spot_s = session_spot_series(session)
        if spot_s and spot_s.spot:
            prev_spot = float(spot_s.spot[-1])

    wins = [t for t in trades if t["pnl_inr"] > 0]
    losses = [t for t in trades if t["pnl_inr"] < 0]
    be = [t for t in trades if t["pnl_inr"] == 0]
    n = len(trades)
    return {
        "name": cfg.name,
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(be),
        "win_rate": round(100 * len(wins) / n, 1) if n else 0,
        "non_loss_rate": round(100 * (len(wins) + len(be)) / n, 1) if n else 0,
        "pnl": round(sum(t["pnl_inr"] for t in trades), 2),
        "trade_list": trades,
    }


def main() -> None:
    base = BacktestParams(
        capital=100_000,
        sl_inr=9.0,
        direction="AUTO",
        start_date="2026-05-22",
        end_date="2026-06-19",
        entry_scan_start_ist="14:00",
        entry_scan_end_ist="15:00",
    )
    store = _store()
    sessions = _resolve_sessions(base)

    common = {
        "ENTRY_BLOCK_REENTRY_AFTER_LOSS": "true",
        "ENTRY_BLOCK_REENTRY_AFTER_BREAKEVEN": "true",
        "MAX_TRADES_PER_CONTRACT_PER_DAY": "1",
        "EXIT_T1_CLOSE_CONFIRM": "false",
    }

    combos = []
    for exit_m in ("t1_scalp", "stepped"):
        for conf in ("true", "false"):
            for idx in ("true", "false"):
                for direction in ("true", "false"):
                    for day in ("true", "false"):
                        for blo, bhi, bl in ((17, 23, "full"), (18, 21, "mid"), (17.5, 20.5, "tight")):
                            combos.append(
                                (
                                    f"{exit_m}|c={conf}|i={idx}|d={direction}|day={day}|{bl}",
                                    {
                                        **common,
                                        "EXIT_MODEL": exit_m,
                                        "ENTRY_REQUIRE_CONFIRMATION_CANDLE": conf,
                                        "ENTRY_REQUIRE_INDEX_MOMENTUM": idx,
                                        "ENTRY_REQUIRE_DIRECTION_ALIGNED": direction,
                                        "ENTRY_REQUIRE_DAY_ALIGNED": day,
                                    },
                                    blo,
                                    bhi,
                                )
                            )

    results = [_eval_cfg(Cfg(n, e, blo, bhi), sessions, store, base) for n, e, blo, bhi in combos]
    hits = [r for r in results if r["trades"] >= 1 and r["win_rate"] >= 90]
    hits.sort(key=lambda x: (-x["win_rate"], -x["trades"], -x["pnl"]))

    print("=" * 100)
    print(f"Fast grid: {len(sessions)} sessions, {len(combos)} configs")
    if hits:
        print(f"Found {len(hits)} configs with win_rate >= 90%")
    else:
        print("No 90%+ configs — showing top 20:")
        hits = sorted([r for r in results if r["trades"] >= 1], key=lambda x: (-x["win_rate"], -x["trades"]))[:20]

    for r in hits[:20]:
        print(
            f"  {r['name']:<48} tr={r['trades']} W/L/BE={r['wins']}/{r['losses']}/{r['breakeven']} "
            f"win={r['win_rate']}% pnl=₹{r['pnl']:,.0f}"
        )
        for t in r.get("trade_list", []):
            tag = "W" if t["pnl_inr"] > 0 else ("BE" if t["pnl_inr"] == 0 else "L")
            print(f"    [{tag}] {t['expiry_date']} {t['kind']} {t['strike']} {t['exit_reason']} ₹{t['pnl_inr']:.0f}")

    if hits and hits[0]["win_rate"] >= 90:
        best = hits[0]
        out = ROOT / "data" / "market" / "nifty90_winrate_config.json"
        import json

        out.write_text(json.dumps(best, indent=2))
        print(f"\nSaved best config → {out}")


if __name__ == "__main__":
    main()
