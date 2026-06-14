"""Trading-day calendar and index context for Sensex Dhan backtest."""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.dhan_data_client import OptionSeries

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sensex"
OHLC_PATH = DATA_DIR / "weekly_expiry_day_ohlc.csv"


def _f(val: Any, default: float = 0.0) -> float:
    try:
        return float(val or 0)
    except (TypeError, ValueError):
        return default


def _load_expiry_rows() -> List[Dict[str, str]]:
    if not OHLC_PATH.exists():
        return []
    with OHLC_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def backtest_calendar_bounds(*, lookback_days: int = 28) -> Tuple[date, date]:
    """Default backtest window: lookback before first expiry through last expiry."""
    dates: List[date] = []
    for row in _load_expiry_rows():
        raw = row.get("expiry_date") or ""
        if raw:
            dates.append(date.fromisoformat(raw))
    if not dates:
        today = date.today()
        return today - timedelta(days=lookback_days), today
    return min(dates) - timedelta(days=lookback_days), max(dates)


def iter_trading_days(start: date, end: date) -> List[str]:
    out: List[str] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def list_trading_days(
    *,
    start: Optional[date] = None,
    end: Optional[date] = None,
    only: Optional[List[str]] = None,
) -> List[str]:
    if only:
        return sorted(set(only))
    lo, hi = backtest_calendar_bounds()
    if start:
        lo = start
    if end:
        hi = end
    return iter_trading_days(lo, hi)


def expiry_index_by_date() -> Dict[str, Dict[str, str]]:
    return {str(r.get("expiry_date") or ""): r for r in _load_expiry_rows()}


def session_spot_series(session: Dict[str, Dict[str, OptionSeries]]) -> Optional[OptionSeries]:
    for kind in ("CE", "PE"):
        atm = (session.get(kind) or {}).get("ATM")
        if atm and atm.spot:
            return atm
    return None


def session_index_from_spot(
    session: Dict[str, Dict[str, OptionSeries]],
    *,
    prev_trading_close: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Index OHLC from Sensex spot LTP embedded in Dhan option series (not day-open guess).
    Returns (index_open, index_close, prev_close_for_gap).
    """
    spot_s = session_spot_series(session)
    if not spot_s or not spot_s.spot:
        return 0.0, 0.0, prev_trading_close
    index_open = float(spot_s.spot[0])
    index_close = float(spot_s.spot[-1])
    prev_close = prev_trading_close if prev_trading_close > 0 else index_open
    return index_open, index_close, prev_close
