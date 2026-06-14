#!/usr/bin/env python3
"""CLI wrapper for Sensex Dhan backtest service."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

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
    parser = argparse.ArgumentParser(description="Backtest Sensex 20rupees-strategy on Dhan 5m data")
    parser.add_argument("--direction", default="AUTO", choices=["AUTO", "CE", "PE"])
    parser.add_argument("--mode", default="both", choices=["both", "conservative", "optimistic"])
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--risk-pct", type=float, default=1.0)
    parser.add_argument("--sl", type=float, default=10.0)
    parser.add_argument("--reward", type=float, default=10.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", default=str(DATA_DIR / "backtest_20rupees_dhan_results.json"))
    args = parser.parse_args()

    params = BacktestParams(
        capital=args.capital,
        risk_pct=args.risk_pct,
        sl_inr=args.sl,
        reward_inr=args.reward,
        direction=args.direction,
        mode=args.mode,
        refresh_dhan=args.refresh,
    )
    result = run_sensex_dhan_backtest(params)
    _print_report(result)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  Full results → {out}\n")


if __name__ == "__main__":
    main()
