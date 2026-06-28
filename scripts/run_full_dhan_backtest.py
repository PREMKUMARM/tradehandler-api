#!/usr/bin/env python3
"""Run full Nifty50 + Sensex Dhan backtest and write combined trade CSV."""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from services.dhan_backtest_export import trades_from_report, write_trades_csv
from services.entry_quality import entry_band_limits, exit_model
from services.nifty_dhan_backtest import LOT_SIZE as NIFTY_LOT, run_nifty_dhan_backtest
from services.sensex_dhan_backtest import BacktestParams, LOT_SIZE as SENSEX_LOT, run_sensex_dhan_backtest


def main() -> None:
    end = date.today()
    nifty_start = (end - timedelta(days=90)).isoformat()
    sensex_start = (end - timedelta(days=120)).isoformat()
    nifty_end = end.isoformat()
    sensex_end = end.isoformat()

    band_lo, band_hi = entry_band_limits()
    capital = 100_000.0
    risk = 10.0

    print(f"\n{'=' * 64}")
    print(f"  Full Dhan backtest · exit={exit_model()} · band ₹{band_lo:g}–₹{band_hi:g}")
    print(f"  Capital ₹{capital:,.0f} · risk {risk}% at SL")
    print(f"{'=' * 64}\n")

    nifty_params = BacktestParams(
        capital=capital,
        risk_pct=risk,
        sl_inr=9.0,
        direction="AUTO",
        start_date=nifty_start,
        end_date=nifty_end,
        segment="nifty50",
        entry_band_low=band_lo,
        entry_band_high=band_hi,
    )
    sensex_params = BacktestParams(
        capital=capital,
        risk_pct=risk,
        sl_inr=9.0,
        direction="AUTO",
        start_date=sensex_start,
        end_date=sensex_end,
        segment="sensex",
        entry_band_low=band_lo,
        entry_band_high=band_hi,
    )

    nifty = run_nifty_dhan_backtest(nifty_params)
    sensex = run_sensex_dhan_backtest(sensex_params)

    market_dir = ROOT / "data" / "market"
    sensex_dir = ROOT / "data" / "sensex"
    market_dir.mkdir(parents=True, exist_ok=True)
    sensex_dir.mkdir(parents=True, exist_ok=True)

    nifty_json = market_dir / "backtest_nifty50_dhan_results.json"
    sensex_json = sensex_dir / "backtest_20rupees_dhan_results.json"
    nifty_json.write_text(json.dumps(nifty, indent=2), encoding="utf-8")
    sensex_json.write_text(json.dumps(sensex, indent=2), encoding="utf-8")

    nifty_report = nifty["reports"]["5m"]
    sensex_report = sensex["reports"]["5m"]
    combined_rows = trades_from_report("nifty50", nifty_report, lot_size=NIFTY_LOT)
    combined_rows.extend(trades_from_report("sensex", sensex_report, lot_size=SENSEX_LOT))
    combined_csv = market_dir / "backtest_nifty50_sensex_trades.csv"
    write_trades_csv(combined_csv, combined_rows)

    def _print_seg(name: str, rep: dict) -> None:
        s = rep["summary"]
        print(f"  --- {name} ---")
        print(
            f"  Capital: ₹{s['starting_capital_inr']:,.0f} → ₹{s['ending_capital_inr']:,.0f} "
            f"({s['return_pct']:+.1f}%)  |  Max DD: ₹{s.get('max_drawdown_inr', 0):,.0f}"
        )
        print(
            f"  Trades: {s['trades']}  |  Win rate: {s['win_rate_pct']}%  |  "
            f"SL: {s.get('losses', 0)}  |  P&L: ₹{s['total_pnl_inr']:,.0f}  |  Avg: {s['avg_r']}R"
        )
        for t in rep["trades"]:
            print(
                f"    {t['expiry_date']} {t['kind']} {t['strike']} "
                f"₹{t['entry']:.1f}→₹{t['exit']:.1f}  P&L ₹{t['pnl_inr']:,.0f} "
                f"({t['r_multiple']}R)  {t['exit_reason']}"
            )
        print()

    _print_seg("NIFTY50", nifty_report)
    _print_seg("SENSEX", sensex_report)

    total_trades = nifty_report["summary"]["trades"] + sensex_report["summary"]["trades"]
    total_wins = nifty_report["summary"]["wins"] + sensex_report["summary"]["wins"]
    total_pnl = nifty_report["summary"]["total_pnl_inr"] + sensex_report["summary"]["total_pnl_inr"]
    comb_wr = round(100 * total_wins / total_trades, 1) if total_trades else 0.0
    print(
        f"  --- COMBINED ---\n"
        f"  Trades: {total_trades}  |  Win rate: {comb_wr}%  |  Total P&L: ₹{total_pnl:,.0f}\n"
        f"  JSON: {nifty_json}\n        {sensex_json}\n"
        f"  CSV:  {combined_csv} ({len(combined_rows)} rows)\n"
    )

    tuned = {
        "exit_model": exit_model(),
        "band": [band_lo, band_hi],
        "capital": capital,
        "risk_pct": risk,
        "nifty": nifty_report["summary"],
        "sensex": sensex_report["summary"],
        "combined_win_rate_pct": comb_wr,
        "combined_pnl_inr": total_pnl,
    }
    (market_dir / "tuned_20rupees_config.json").write_text(json.dumps(tuned, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
