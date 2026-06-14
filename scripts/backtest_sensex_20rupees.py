#!/usr/bin/env python3
"""
Backtest Sensex 20rupees-strategy on weekly BSE expiry day OHLC.

Rules (matches live):
  • ATM or highest-OI strike on chosen leg (CE/PE)
  • Entry when premium touches ₹17–₹23 (session entries only before 15:00 IST)
  • Size to risk % of capital (default 1%), max 50 lots · ₹10 SL, 1:1 target, trail after target

Data: data/sensex/options/*_sensex_expiry_chain.csv (daily bar per contract).
Path ordering on a single daily bar is approximate — SL is checked before target (conservative).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.momentum_trail import breakeven_stop, get_momentum_trail_config
from services.sensex_constants import sensex_max_lots_per_trade
from services.sensex_indicator_plan import size_from_risk
from services.sensex_strategy_analysis import (
    FIXED_SL_INR,
    PREMIUM_BAND_HIGH,
    PREMIUM_BAND_LOW,
    STRATEGY_NAME,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sensex"
CHAIN_GLOB = "options/*_sensex_expiry_chain.csv"
OHLC_PATH = DATA_DIR / "weekly_expiry_day_ohlc.csv"
LOT_SIZE = 20
MAX_LOTS = sensex_max_lots_per_trade()
DEFAULT_RISK_PCT = 1.0


@dataclass
class ContractBar:
    strike: int
    kind: str
    symbol: str
    open_p: float
    high: float
    low: float
    close: float
    oi: float


@dataclass
class TradeResult:
    expiry_date: str
    direction: str
    strike: int
    kind: str
    strike_source: str
    symbol: str
    entry: float
    exit: float
    sl: float
    target: float
    pnl_inr: float
    r_multiple: float
    exit_reason: str
    index_open: float
    premium_open: float
    premium_high: float
    premium_low: float
    num_lots: int = 1
    entry_notional: float = 0.0
    risk_at_sl_inr: float = 0.0
    capital_before: float = 0.0
    capital_after: float = 0.0


def _f(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _in_band(px: float) -> bool:
    return PREMIUM_BAND_LOW <= px <= PREMIUM_BAND_HIGH


def _band_touched(low: float, high: float) -> bool:
    return low <= PREMIUM_BAND_HIGH and high >= PREMIUM_BAND_LOW


def _estimate_entry(open_p: float, high: float, low: float) -> Optional[float]:
    if not _band_touched(low, high):
        return None
    if _in_band(open_p):
        return round(open_p, 2)
    if open_p > PREMIUM_BAND_HIGH:
        return PREMIUM_BAND_HIGH
    if open_p < PREMIUM_BAND_LOW:
        return PREMIUM_BAND_LOW
    return round((PREMIUM_BAND_LOW + PREMIUM_BAND_HIGH) / 2.0, 2)


def _load_prev_closes() -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not OHLC_PATH.exists():
        return out
    with OHLC_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["expiry_date"]] = _f(row.get("prev_close"))
    return out


def _load_chain(path: Path) -> Tuple[str, float, List[ContractBar]]:
    bars: List[ContractBar] = []
    index_open = 0.0
    expiry_date = ""
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not expiry_date:
                expiry_date = row.get("session_date") or row.get("TradDt") or ""
                index_open = _f(row.get("index_open"))
            kind = (row.get("OptnTp") or "").upper()
            if kind not in ("CE", "PE"):
                continue
            bars.append(
                ContractBar(
                    strike=int(_f(row.get("StrkPric"))),
                    kind=kind,
                    symbol=row.get("FinInstrmNm") or row.get("TckrSymb") or "",
                    open_p=_f(row.get("OpnPric")),
                    high=_f(row.get("HghPric")),
                    low=_f(row.get("LwPric")),
                    close=_premium_close(row),
                    oi=_f(row.get("OpnIntrst")),
                )
            )
    return expiry_date, index_open, bars


def _resolve_kind(direction: str, index_open: float, prev_close: float) -> str:
    d = (direction or "AUTO").upper()
    if d in ("CE", "PE"):
        return d
    if prev_close > 0:
        return "CE" if index_open >= prev_close else "PE"
    return "CE"


def _pick_contract(
    bars: List[ContractBar],
    kind: str,
    index_open: float,
) -> Tuple[Optional[ContractBar], str]:
    leg = [b for b in bars if b.kind == kind]
    if not leg:
        return None, ""
    atm = int(round(index_open / 100) * 100)
    by_strike = {b.strike: b for b in leg}
    atm_bar = by_strike.get(atm)
    max_oi_bar = max(leg, key=lambda b: b.oi)

    candidates: List[Tuple[str, ContractBar]] = []
    if atm_bar:
        candidates.append(("ATM", atm_bar))
    if max_oi_bar.strike != atm:
        candidates.append(("MAX_OI", max_oi_bar))

    for source, bar in candidates:
        if _band_touched(bar.low, bar.high):
            return bar, source

    return (candidates[0][1], candidates[0][0]) if candidates else (None, "")


def _premium_close(row: Dict[str, str]) -> float:
    """BSE chain files sometimes set ClsPric to index settlement — prefer LastPric."""
    last = _f(row.get("LastPric"))
    cls = _f(row.get("ClsPric"))
    if 0 < last < 5000:
        return last
    if 0 < cls < 5000:
        return cls
    return _f(row.get("OpnPric"))


def _simulate_trail_exit(entry: float, bar: ContractBar) -> Tuple[float, str]:
    cfg = get_momentum_trail_config()
    r = FIXED_SL_INR
    peak = bar.high
    be = breakeven_stop(entry, r, cfg)
    step = max(1, int((peak - entry) / r))
    locked = round(entry + (step - 1) * r, 2) if step > 1 else be
    trail_sl = max(be, locked)
    if bar.low <= trail_sl and trail_sl > entry:
        return round(trail_sl, 2), "trail_stop"
    if bar.close > entry:
        return round(bar.close, 2), "eod_trail"
    return round(entry + r, 2), "target_trail"


def _simulate_exit_conservative(entry: float, bar: ContractBar) -> Tuple[float, str]:
    r = FIXED_SL_INR
    sl = round(entry - r, 2)
    target = round(entry + r, 2)
    if bar.low <= sl:
        return sl, "stop_loss"
    if bar.high >= target:
        return _simulate_trail_exit(entry, bar)
    return round(bar.close, 2), "eod"


def _simulate_exit_optimistic(entry: float, bar: ContractBar) -> Tuple[float, str]:
    r = FIXED_SL_INR
    sl = round(entry - r, 2)
    target = round(entry + r, 2)
    hit_sl = bar.low <= sl
    hit_tgt = bar.high >= target
    if hit_sl and hit_tgt:
        return _simulate_trail_exit(entry, bar)
    if hit_sl:
        return sl, "stop_loss"
    if hit_tgt:
        return _simulate_trail_exit(entry, bar)
    return round(bar.close, 2), "eod"


def _simulate_exit(entry: float, bar: ContractBar, mode: str) -> Tuple[float, str]:
    if mode == "conservative":
        return _simulate_exit_conservative(entry, bar)
    return _simulate_exit_optimistic(entry, bar)


def _run_day(
    expiry_date: str,
    index_open: float,
    bars: List[ContractBar],
    prev_close: float,
    direction: str,
    mode: str,
) -> Optional[TradeResult]:
    kind = _resolve_kind(direction, index_open, prev_close)
    bar, source = _pick_contract(bars, kind, index_open)
    if bar is None:
        return None

    entry = _estimate_entry(bar.open_p, bar.high, bar.low)
    if entry is None:
        return None

    exit_px, reason = _simulate_exit(entry, bar, mode)
    pnl_per_unit = exit_px - entry
    r_mult = round(pnl_per_unit / FIXED_SL_INR, 2) if FIXED_SL_INR else 0.0

    return TradeResult(
        expiry_date=expiry_date,
        direction=kind,
        strike=bar.strike,
        kind=bar.kind,
        strike_source=source,
        symbol=bar.symbol,
        entry=entry,
        exit=exit_px,
        sl=round(entry - FIXED_SL_INR, 2),
        target=round(entry + FIXED_SL_INR, 2),
        pnl_inr=0.0,
        r_multiple=r_mult,
        exit_reason=reason,
        index_open=index_open,
        premium_open=bar.open_p,
        premium_high=bar.high,
        premium_low=bar.low,
    )


def run_backtest(
    direction: str = "AUTO",
    mode: str = "both",
    capital: float = 10000.0,
    risk_pct: float = DEFAULT_RISK_PCT,
) -> Dict[str, Any]:
    prev_closes = _load_prev_closes()
    chain_files = sorted(DATA_DIR.glob(CHAIN_GLOB))
    modes = ["conservative", "optimistic"] if mode == "both" else [mode]
    reports: Dict[str, Any] = {}
    start_capital = max(1000.0, float(capital or 10000.0))

    for m in modes:
        trades: List[TradeResult] = []
        skipped: List[Dict[str, str]] = []
        equity = start_capital
        peak_equity = start_capital
        max_drawdown_inr = 0.0

        for path in chain_files:
            expiry_date, index_open, bars = _load_chain(path)
            prev = prev_closes.get(expiry_date, index_open)
            tr = _run_day(expiry_date, index_open, bars, prev, direction, m)
            if tr:
                sl_prem = round(tr.entry - FIXED_SL_INR, 2)
                lots, qty, risk_inr = size_from_risk(
                    equity,
                    risk_pct,
                    tr.entry,
                    sl_prem,
                    LOT_SIZE,
                    MAX_LOTS,
                )
                notional = tr.entry * qty
                if notional > equity:
                    skipped.append(
                        {
                            "expiry_date": expiry_date,
                            "reason": f"insufficient capital (need ₹{notional:.0f}, have ₹{equity:.0f})",
                        }
                    )
                    continue
                cap_before = equity
                pnl_inr = round((tr.exit - tr.entry) * qty, 2)
                equity = round(equity + pnl_inr, 2)
                peak_equity = max(peak_equity, equity)
                max_drawdown_inr = max(max_drawdown_inr, peak_equity - equity)
                tr.num_lots = lots
                tr.pnl_inr = pnl_inr
                tr.entry_notional = round(notional, 2)
                tr.risk_at_sl_inr = round(risk_inr, 2)
                tr.capital_before = round(cap_before, 2)
                tr.capital_after = equity
                trades.append(tr)
            else:
                skipped.append({"expiry_date": expiry_date, "reason": "premium never in ₹17–₹23 band"})

        wins = [t for t in trades if t.pnl_inr > 0]
        total_pnl = sum(t.pnl_inr for t in trades)
        ending = round(start_capital + total_pnl, 2)
        avg_lots = round(sum(t.num_lots for t in trades) / len(trades), 1) if trades else 0.0
        avg_risk = round(sum(t.risk_at_sl_inr for t in trades) / len(trades), 2) if trades else 0.0
        reports[m] = {
            "summary": {
                "strategy": STRATEGY_NAME,
                "mode": m,
                "direction_mode": direction,
                "starting_capital_inr": round(start_capital, 2),
                "ending_capital_inr": ending,
                "return_pct": round((ending - start_capital) / start_capital * 100.0, 2)
                if start_capital > 0
                else 0.0,
                "max_drawdown_inr": round(max_drawdown_inr, 2),
                "sessions": len(chain_files),
                "trades": len(trades),
                "skipped": len(skipped),
                "wins": len(wins),
                "losses": len([t for t in trades if t.pnl_inr < 0]),
                "win_rate_pct": round(100.0 * len(wins) / len(trades), 1) if trades else 0.0,
                "total_pnl_inr": round(total_pnl, 2),
                "avg_pnl_inr": round(total_pnl / len(trades), 2) if trades else 0.0,
                "avg_r": round(sum(t.r_multiple for t in trades) / len(trades), 2) if trades else 0.0,
                "lot_size": LOT_SIZE,
                "risk_pct": risk_pct,
                "max_lots_cap": MAX_LOTS,
                "avg_lots_per_trade": avg_lots,
                "max_risk_per_trade_inr": avg_risk,
                "typical_entry_notional_inr": round(23.0 * LOT_SIZE * avg_lots, 2) if avg_lots else 0.0,
                "sl_inr": FIXED_SL_INR,
                "premium_band": [PREMIUM_BAND_LOW, PREMIUM_BAND_HIGH],
            },
            "trades": [asdict(t) for t in trades],
            "skipped": skipped,
        }

    note = (
        f"Starting capital ₹{start_capital:,.0f} · size to {risk_pct:g}% risk (max {MAX_LOTS} lots × "
        f"{LOT_SIZE} qty). Daily OHLC on 10 weekly BSE expiry sessions (Apr–Jun 2026). "
        "Conservative = SL before target when both touched; "
        "Optimistic = trail path when both touched."
    )
    return {
        "note": note,
        "starting_capital_inr": round(start_capital, 2),
        "risk_pct": risk_pct,
        "reports": reports,
        "generated_at": datetime.now().isoformat(),
    }


def _print_report(result: Dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {STRATEGY_NAME} — backtest report")
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
            f"P&L: ₹{s['total_pnl_inr']:,.0f}  |  Avg: {s['avg_r']}R  |  "
            f"Avg lots: {s['avg_lots_per_trade']}  |  Avg risk/trade: ₹{s['max_risk_per_trade_inr']:,.0f}"
        )
        for t in block["trades"]:
            print(
                f"    {t['expiry_date']} {t['kind']} {t['strike']} ({t['strike_source']}) "
                f"{t['num_lots']} lots · entry ₹{t['entry']:.2f} → ₹{t['exit']:.2f}  "
                f"P&L ₹{t['pnl_inr']:,.0f} ({t['r_multiple']}R)  "
                f"risk ₹{t['risk_at_sl_inr']:,.0f}  "
                f"cap ₹{t['capital_before']:,.0f}→₹{t['capital_after']:,.0f}  {t['exit_reason']}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Sensex 20rupees-strategy")
    parser.add_argument("--direction", default="AUTO", choices=["AUTO", "CE", "PE"])
    parser.add_argument("--mode", default="both", choices=["both", "conservative", "optimistic"])
    parser.add_argument("--capital", type=float, default=10000.0, help="Starting capital in INR")
    parser.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT, help="Risk %% of capital per trade")
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "backtest_20rupees_results.json"),
        help="JSON output path",
    )
    args = parser.parse_args()

    result = run_backtest(direction=args.direction, mode=args.mode, capital=args.capital, risk_pct=args.risk_pct)
    _print_report(result)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  Full results → {out}\n")


if __name__ == "__main__":
    main()
