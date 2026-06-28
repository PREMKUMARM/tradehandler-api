#!/usr/bin/env python3
"""CLI wrapper for Sensex Dhan backtest service."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from services.dhan_backtest_export import export_backtest_result_csv
from services.entry_quality import entry_band_limits, exit_model
from services.sensex_dhan_backtest import BacktestParams, run_sensex_dhan_backtest

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sensex"


def _print_report(result: dict) -> None:
    print(f"\n{'=' * 60}")
    print("  20rupees-strategy — Dhan intraday backtest")
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
        for t in block["trades"]:
            print(
                f"    {t['expiry_date']} {t['kind']} {t['strike']} ({t['strike_source']}) "
                f"{t['num_lots']} lots · ₹{t['entry']:.2f} → ₹{t['exit']:.2f}  "
                f"P&L ₹{t['pnl_inr']:,.0f} ({t['r_multiple']}R)  {t['exit_reason']}"
            )
        print()


def main() -> None:
    end_default = date.today()
    start_default = end_default - timedelta(days=90)
    parser = argparse.ArgumentParser(description="Backtest Sensex 20rupees-strategy on Dhan 5m data")
    parser.add_argument("--direction", default="AUTO", choices=["AUTO", "CE", "PE"])
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--risk-pct", type=float, default=10.0)
    parser.add_argument("--sl", type=float, default=9.0)
    parser.add_argument("--start", default=start_default.isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--end", default=end_default.isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--band-low", type=float, default=None)
    parser.add_argument("--band-high", type=float, default=None)
    parser.add_argument("--min-target", type=float, default=None, help="Trail trigger (default 10 = 1R)")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch missing sessions from Dhan API")
    parser.add_argument("--output", default=str(DATA_DIR / "backtest_20rupees_dhan_results.json"))
    args = parser.parse_args()

    band_lo, band_hi = entry_band_limits()
    if args.band_low is not None:
        band_lo = args.band_low
    if args.band_high is not None:
        band_hi = args.band_high

    mt = args.min_target if args.min_target is not None else BacktestParams().min_target_low
    params = BacktestParams(
        capital=args.capital,
        risk_pct=args.risk_pct,
        sl_inr=args.sl,
        min_target_low=mt,
        min_target_high=mt,
        direction=args.direction,
        start_date=args.start,
        end_date=args.end,
        entry_band_low=band_lo,
        entry_band_high=band_hi,
        refresh_dhan=args.refresh,
    )
    result = run_sensex_dhan_backtest(params)
    print(f"\n  Exit model: {exit_model()}  |  Band: ₹{band_lo:g}–₹{band_hi:g}  |  Range: {args.start} → {args.end}")
    _print_report(result)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = out.with_name(out.stem + "_trades.csv")
    n = export_backtest_result_csv(result, csv_path, segment="sensex", lot_size=20)
    print(f"  Full results → {out}")
    print(f"  Trades CSV  → {csv_path} ({n} rows)\n")


if __name__ == "__main__":
    main()
