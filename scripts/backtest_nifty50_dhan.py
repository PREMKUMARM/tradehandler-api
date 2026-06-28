#!/usr/bin/env python3
"""CLI — Nifty 50 Dhan backtest (last month by default)."""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from services.nifty_dhan_backtest import run_nifty_dhan_backtest
from services.sensex_dhan_backtest import BacktestParams
from services.dhan_backtest_export import export_backtest_result_csv, trades_from_report, write_trades_csv
from services.entry_quality import entry_band_limits, exit_model


def main() -> None:
    end = date.today()
    start = end - timedelta(days=90)
    parser = argparse.ArgumentParser(description="Backtest Nifty50 on Dhan FnO (stepped SL)")
    parser.add_argument("--start", default=start.isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--end", default=end.isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--risk-pct", type=float, default=10.0)
    parser.add_argument("--sl", type=float, default=9.0, help="Initial SL premium")
    parser.add_argument("--direction", default="AUTO", choices=["AUTO", "CE", "PE"])
    parser.add_argument("--refresh", action="store_true", help="Re-fetch from Dhan API")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "market" / "backtest_nifty50_dhan_results.json"),
    )
    parser.add_argument("--band-low", type=float, default=None, help="Entry band low (default env/17)")
    parser.add_argument("--band-high", type=float, default=None, help="Entry band high (default env/23)")
    args = parser.parse_args()

    band_lo, band_hi = entry_band_limits()
    if args.band_low is not None:
        band_lo = args.band_low
    if args.band_high is not None:
        band_hi = args.band_high

    params = BacktestParams(
        capital=args.capital,
        risk_pct=args.risk_pct,
        sl_inr=args.sl,
        direction=args.direction,
        start_date=args.start,
        end_date=args.end,
        refresh_dhan=args.refresh,
        timeframes_min=[5],
        segment="nifty50",
        entry_band_low=band_lo,
        entry_band_high=band_hi,
    )
    result = run_nifty_dhan_backtest(params)

    print(f"\n{'=' * 60}")
    print(f"  Nifty50 — Dhan backtest (exit: {exit_model()}, band ₹{band_lo:g}–₹{band_hi:g})")
    print(f"{'=' * 60}")
    print(f"  {result['note']}\n")
    for mode, block in result["reports"].items():
        s = block["summary"]
        print(f"  --- {mode.upper()} ---")
        print(
            f"  Capital: ₹{s['starting_capital_inr']:,.0f} → ₹{s['ending_capital_inr']:,.0f} "
            f"({s['return_pct']:+.1f}%)  |  Max DD: ₹{s['max_drawdown_inr']:,.0f}"
        )
        print(
            f"  Trades: {s['trades']}  |  Win rate: {s['win_rate_pct']}%  |  "
            f"P&L: ₹{s['total_pnl_inr']:,.0f}  |  Avg: {s['avg_r']}R"
        )
        for t in block["trades"][:12]:
            print(
                f"    {t['expiry_date']} {t['kind']} {t['strike']} "
                f"₹{t['entry']:.1f}→₹{t['exit']:.1f}  P&L ₹{t['pnl_inr']:,.0f} "
                f"({t['r_multiple']}R)  {t['exit_reason']}"
            )
        if len(block["trades"]) > 12:
            print(f"    ... +{len(block['trades']) - 12} more")
        print()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = out.with_name(out.stem + "_trades.csv")
    n = export_backtest_result_csv(result, csv_path, segment="nifty50", lot_size=75)
    print(f"  Full results → {out}")
    print(f"  Trades CSV  → {csv_path} ({n} rows)\n")


if __name__ == "__main__":
    main()
