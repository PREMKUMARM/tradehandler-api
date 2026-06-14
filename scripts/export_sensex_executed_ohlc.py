#!/usr/bin/env python3
"""Export 5m OHLC for picked contracts on backtest-executed expiry days."""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.dhan_data_client import build_fixed_strike_series, load_cached_session
from services.sensex_constants import sensex_entry_cutoff_minutes
from services.sensex_dhan_backtest import (
    CACHE_DIR,
    BacktestParams,
    TradeResult,
    _load_all_sessions,
    _pick_entry,
    _pick_entry_auto,
    _resolve_exit_bar_index,
    _run_day,
    _simulate_from_entry,
    sensex_entry_cutoff_minutes,
)

DEFAULT_OUT_DIR = ROOT / "data" / "sensex" / "executed_trades_ohlc"
DEFAULT_COMBINED = ROOT / "data" / "sensex" / "executed_trades_ohlc_all.csv"
IST = ZoneInfo("Asia/Kolkata")

COLUMNS = [
    "expiry_date",
    "index_open",
    "prev_close",
    "symbol",
    "kind",
    "entry_strike",
    "bar_strike",
    "strike_offset",
    "strike_source",
    "trade_entry",
    "trade_exit",
    "trade_sl",
    "trade_target",
    "exit_reason",
    "r_multiple",
    "bar_index",
    "datetime_ist",
    "open",
    "high",
    "low",
    "close",
    "oi",
    "spot",
    "is_entry_bar",
    "is_exit_bar",
    "in_entry_scan_window",
]


@dataclass
class ExecutedContract:
    expiry_date: str
    index_open: float
    prev_close: float
    strike_source: str
    entry_bar_idx: int
    display_exit_idx: int
    trade: TradeResult
    series: object


def _f(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _contract_filename(contract: ExecutedContract) -> str:
    expiry = contract.expiry_date.replace("-", "")
    strike = int(contract.trade.strike)
    offset = contract.series.offset.replace("/", "-")
    return f"{expiry}_{contract.trade.kind}_{strike}_{offset}_5m.csv"


def _iter_executed_contracts(params: BacktestParams) -> Iterator[ExecutedContract]:
    sessions = _load_all_sessions()
    if params.expiry_dates:
        wanted = set(params.expiry_dates)
        sessions = [r for r in sessions if r.get("expiry_date") in wanted]

    mode = params.mode if params.mode in ("conservative", "optimistic") else "conservative"
    cutoff = sensex_entry_cutoff_minutes()

    for row in sessions:
        expiry_date = str(row.get("expiry_date") or "")
        index_open = _f(row.get("open"))
        prev_close = _f(row.get("prev_close"))
        session = load_cached_session(CACHE_DIR, expiry_date)
        if not session:
            continue

        direction = (params.direction or "AUTO").upper()
        if direction == "AUTO":
            picked = _pick_entry_auto(
                session,
                params.entry_band_low,
                params.entry_band_high,
                cutoff,
                index_open=index_open,
                prev_close=prev_close,
            )
        else:
            picked = _pick_entry(
                session,
                direction,
                params.entry_band_low,
                params.entry_band_high,
                cutoff,
                prev_close=prev_close,
            )
        if not picked:
            continue

        entry_bar, strike_source, series = picked
        trade = _run_day(expiry_date, index_open, prev_close, session, params, mode)
        if not trade:
            continue

        entry_strike = int(trade.strike)
        entry_ts = series.timestamps[entry_bar.idx]
        export_series = series
        export_entry_idx = entry_bar.idx
        locked = build_fixed_strike_series(
            session,
            kind=trade.kind,
            strike=entry_strike,
            session_date=expiry_date,
        )
        if locked and entry_ts in locked.timestamps:
            export_series = locked
            export_entry_idx = locked.timestamps.index(entry_ts)

        _, reason, sim_exit_idx = _simulate_from_entry(
            trade.entry,
            export_series,
            export_entry_idx,
            mode,
            params.sl_inr,
            params.min_target_low,
            params.min_target_high,
        )
        display_exit_idx = _resolve_exit_bar_index(
            trade.entry,
            export_series,
            export_entry_idx,
            trade.exit,
            reason,
            sim_exit_idx,
            params.sl_inr,
        )
        yield ExecutedContract(
            expiry_date=expiry_date,
            index_open=index_open,
            prev_close=prev_close,
            strike_source=strike_source,
            entry_bar_idx=export_entry_idx,
            display_exit_idx=display_exit_idx,
            trade=trade,
            series=export_series,
        )


def _bar_rows(contract: ExecutedContract) -> List[Dict[str, object]]:
    scan_start = 14 * 60
    cutoff = sensex_entry_cutoff_minutes()
    series = contract.series
    trade = contract.trade
    entry_strike = int(trade.strike)
    rows: List[Dict[str, object]] = []

    for idx in range(len(series.timestamps)):
        dt = datetime.fromtimestamp(series.timestamps[idx], tz=IST)
        ist_label = dt.strftime("%Y-%m-%d %H:%M:%S")
        bar_minutes = dt.hour * 60 + dt.minute
        rows.append(
            {
                "expiry_date": contract.expiry_date,
                "index_open": contract.index_open,
                "prev_close": contract.prev_close,
                "symbol": trade.symbol,
                "kind": series.kind,
                "entry_strike": entry_strike,
                "bar_strike": int(series.strike[idx]),
                "strike_offset": series.offset,
                "strike_source": contract.strike_source,
                "trade_entry": trade.entry,
                "trade_exit": trade.exit,
                "trade_sl": trade.sl,
                "trade_target": trade.target,
                "exit_reason": trade.exit_reason,
                "r_multiple": trade.r_multiple,
                "bar_index": idx,
                "datetime_ist": ist_label,
                "open": round(series.open[idx], 2),
                "high": round(series.high[idx], 2),
                "low": round(series.low[idx], 2),
                "close": round(series.close[idx], 2),
                "oi": round(series.oi[idx], 2),
                "spot": round(series.spot[idx], 2),
                "is_entry_bar": idx == contract.entry_bar_idx,
                "is_exit_bar": idx == contract.display_exit_idx,
                "in_entry_scan_window": scan_start <= bar_minutes < cutoff,
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def export_executed_ohlc(
    params: BacktestParams,
    out_dir: Path,
    *,
    combined_path: Optional[Path] = None,
) -> Tuple[List[Path], int]:
    written: List[Path] = []
    combined_rows: List[Dict[str, object]] = []
    total_bars = 0

    for contract in _iter_executed_contracts(params):
        rows = _bar_rows(contract)
        if not rows:
            continue
        out_path = out_dir / _contract_filename(contract)
        _write_csv(out_path, rows)
        written.append(out_path)
        total_bars += len(rows)
        if combined_path is not None:
            combined_rows.extend(rows)

    if combined_path is not None and combined_rows:
        _write_csv(combined_path, combined_rows)

    return written, total_bars


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OHLC for executed backtest contracts")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for one CSV per executed contract",
    )
    parser.add_argument(
        "--combined",
        type=Path,
        nargs="?",
        const=DEFAULT_COMBINED,
        default=None,
        help="Also write a merged CSV (optional path; default: executed_trades_ohlc_all.csv)",
    )
    parser.add_argument("--mode", default="conservative", choices=["conservative", "optimistic"])
    parser.add_argument("--direction", default="AUTO", choices=["AUTO", "CE", "PE"])
    args = parser.parse_args()

    params = BacktestParams(mode=args.mode, direction=args.direction)
    files, bars = export_executed_ohlc(params, args.out_dir, combined_path=args.combined)
    if not files:
        print("No executed trades found.")
        return

    print(f"Wrote {len(files)} contract CSV(s), {bars} bars total → {args.out_dir}/")
    for path in files:
        print(f"  • {path.name}")
    if args.combined is not None:
        print(f"Combined file → {args.combined}")


if __name__ == "__main__":
    main()
