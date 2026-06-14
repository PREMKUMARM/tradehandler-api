"""
Sensex 20rupees-strategy backtest on Dhan rolling options data (configurable bar interval).
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

from services.dhan_data_client import (
    DhanDataClient,
    OptionSeries,
    SUPPORTED_INTERVALS_MIN,
    build_entry_offset_exit_series,
    cache_path,
    interval_to_str,
    load_cached_session,
    save_cached_session,
    ts_to_ist_label,
)
from services.sensex_constants import (
    sensex_atm_near_offsets,
    sensex_atm_near_steps,
    sensex_entry_cutoff_minutes,
    sensex_entry_scan_start_minutes,
    sensex_default_min_target_inr,
    sensex_gap_direction_kind,
    sensex_is_bad_option_bar,
    sensex_backtest_max_trades_per_contract_per_day,
    sensex_max_lots_per_trade,
    sensex_premium_in_band,
)
from services.sensex_strike_selection import pick_smart_at_bar, strike_source_label
from services.sensex_strategy_analysis import STRATEGY_NAME
from services.sensex_trading_calendar import (
    expiry_index_by_date,
    list_trading_days,
    session_index_from_spot,
    session_spot_series,
)

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sensex"
CACHE_DIR = DATA_DIR / "dhan_intraday"
OHLC_PATH = DATA_DIR / "weekly_expiry_day_ohlc.csv"
LOT_SIZE = 20
MAX_LOTS = sensex_max_lots_per_trade()
MAX_TRADES_PER_CONTRACT_PER_DAY = sensex_backtest_max_trades_per_contract_per_day()
DEFAULT_RISK_PCT = 1.0
DEFAULT_SL_INR = 9.0  # fixed initial stop-loss premium price (API field: sl_inr)
DEFAULT_ENTRY_LOW = 17.0
DEFAULT_ENTRY_HIGH = 23.0
DEFAULT_MIN_TARGET_LOW = sensex_default_min_target_inr()
DEFAULT_MIN_TARGET_HIGH = sensex_default_min_target_inr()
DEFAULT_TIMEFRAMES_MIN: Tuple[int, ...] = (5,)


def _normalize_timeframes(raw: Optional[List[int]]) -> List[int]:
    if not raw:
        return list(DEFAULT_TIMEFRAMES_MIN)
    out: List[int] = []
    for iv in raw:
        n = int(iv)
        if n not in SUPPORTED_INTERVALS_MIN:
            raise ValueError(f"Unsupported timeframe {n}m — choose from {list(SUPPORTED_INTERVALS_MIN)}")
        if n not in out:
            out.append(n)
    if not out:
        raise ValueError("Select at least one timeframe")
    return sorted(out)


def _timeframe_key(interval_min: int) -> str:
    return f"{int(interval_min)}m"


def _interval_label(interval_min: int) -> str:
    iv = int(interval_min)
    return f"{iv}m" if iv < 60 else f"{iv // 60}h"


def _session_has_cached_interval(session_date: str, interval_min: int) -> bool:
    return load_cached_session(CACHE_DIR, session_date, interval_min) is not None


def _cached_intervals_for_session(session_date: str) -> List[int]:
    return [iv for iv in SUPPORTED_INTERVALS_MIN if _session_has_cached_interval(session_date, iv)]


@dataclass
class BacktestParams:
    capital: float = 1_000_000.0
    risk_pct: float = DEFAULT_RISK_PCT
    sl_inr: float = DEFAULT_SL_INR
    entry_band_low: float = DEFAULT_ENTRY_LOW
    entry_band_high: float = DEFAULT_ENTRY_HIGH
    min_target_low: float = DEFAULT_MIN_TARGET_LOW
    min_target_high: float = DEFAULT_MIN_TARGET_HIGH
    direction: str = "AUTO"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    expiry_dates: Optional[List[str]] = None
    refresh_dhan: bool = False
    timeframes_min: Optional[List[int]] = None
    entry_scan_start_ist: Optional[str] = None
    entry_scan_end_ist: Optional[str] = None


def _format_ist_minutes(minutes: int) -> str:
    m = int(minutes)
    return f"{m // 60:02d}:{m % 60:02d}"


def _parse_ist_hhmm(value: str) -> int:
    raw = (value or "").strip()
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid IST time {value!r} — use HH:MM")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid IST time {value!r}")
    return hour * 60 + minute


def resolve_backtest_scan_window(params: BacktestParams) -> Tuple[int, int, str, str]:
    """Return (start_min, end_min_exclusive, start_label, end_label) for entry scan."""
    from services.sensex_constants import SENSEX_SESSION_CLOSE_MINUTES, SENSEX_SESSION_OPEN_MINUTES

    start = sensex_entry_scan_start_minutes()
    end = sensex_entry_cutoff_minutes()
    if params.entry_scan_start_ist:
        start = _parse_ist_hhmm(params.entry_scan_start_ist)
    if params.entry_scan_end_ist:
        end = _parse_ist_hhmm(params.entry_scan_end_ist)
    if end <= start:
        raise ValueError("entry_scan_end_ist must be after entry_scan_start_ist")
    if start < SENSEX_SESSION_OPEN_MINUTES:
        raise ValueError(
            f"entry_scan_start_ist must be on or after {_format_ist_minutes(SENSEX_SESSION_OPEN_MINUTES)} IST"
        )
    if end > SENSEX_SESSION_CLOSE_MINUTES:
        raise ValueError(
            f"entry_scan_end_ist must be on or before {_format_ist_minutes(SENSEX_SESSION_CLOSE_MINUTES)} IST"
        )
    return start, end, _format_ist_minutes(start), _format_ist_minutes(end)


@dataclass
class MinuteBar:
    idx: int
    ts: int
    ist_time: str
    ist_minutes: int
    open: float
    high: float
    low: float
    close: float
    oi: float
    spot: float
    strike: int
    offset: str


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
    entry_datetime_ist: str = ""
    exit_datetime_ist: str = ""


@dataclass
class BacktestProgress:
    phase: str = ""
    current: int = 0
    total: int = 0
    expiry_date: str = ""
    message: str = ""


def _f(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _load_all_sessions() -> List[Dict[str, Any]]:
    """All weekday sessions in the backtest window (not expiry-only)."""
    expiry_idx = expiry_index_by_date()
    out: List[Dict[str, Any]] = []
    for trade_date in list_trading_days():
        row = expiry_idx.get(trade_date) or {}
        out.append(
            {
                "expiry_date": trade_date,
                "session_date": trade_date,
                "is_weekly_expiry": trade_date in expiry_idx,
                "expiry_weekday": row.get("expiry_weekday") or "",
                "holiday_adjusted": row.get("holiday_adjusted") == "True",
                "index_open": _f(row.get("open")),
                "index_close": _f(row.get("close")),
                "prev_close": _f(row.get("prev_close")),
            }
        )
    return out


def list_available_sessions() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in _load_all_sessions():
        trade_date = row.get("expiry_date") or row.get("session_date") or ""
        cached_intervals = _cached_intervals_for_session(trade_date)
        out.append(
            {
                "expiry_date": trade_date,
                "session_date": trade_date,
                "is_weekly_expiry": bool(row.get("is_weekly_expiry")),
                "expiry_weekday": row.get("expiry_weekday"),
                "holiday_adjusted": row.get("holiday_adjusted"),
                "index_open": _f(row.get("index_open") or row.get("open")),
                "index_close": _f(row.get("index_close") or row.get("close")),
                "prev_close": _f(row.get("prev_close")),
                "dhan_cached": 5 in cached_intervals,
                "dhan_cached_intervals": cached_intervals,
            }
        )
    return out


def backtest_session_calendar() -> Dict[str, Any]:
    """Trading-day bounds and cache summary for the date-range picker."""
    sessions = list_available_sessions()
    dates = [s["expiry_date"] for s in sessions if s.get("expiry_date")]
    cached = sum(1 for s in sessions if s.get("dhan_cached"))
    scan_start = sensex_entry_scan_start_minutes()
    scan_end = sensex_entry_cutoff_minutes()
    return {
        "start_date": min(dates) if dates else None,
        "end_date": max(dates) if dates else None,
        "trading_days": len(dates),
        "cached_count": cached,
        "default_entry_scan_start_ist": _format_ist_minutes(scan_start),
        "default_entry_scan_end_ist": _format_ist_minutes(scan_end),
    }


def _resolve_backtest_sessions(params: BacktestParams) -> List[Dict[str, Any]]:
    """Pick sessions from explicit dates, inclusive date range, or full calendar."""
    all_sessions = _load_all_sessions()
    by_date = {r["expiry_date"]: r for r in all_sessions}

    if params.expiry_dates:
        wanted = sorted(set(params.expiry_dates))
        sessions = [by_date[d] for d in wanted if d in by_date]
        if not sessions:
            raise ValueError("No matching trading sessions for selected dates")
        return sessions

    start_raw = (params.start_date or "").strip()
    end_raw = (params.end_date or "").strip()
    if start_raw or end_raw:
        if not start_raw or not end_raw:
            raise ValueError("Both start_date and end_date are required")
        start = date.fromisoformat(start_raw)
        end = date.fromisoformat(end_raw)
        if end < start:
            raise ValueError("end_date must be on or after start_date")
        wanted = set(list_trading_days(start=start, end=end))
        sessions = [r for r in all_sessions if r.get("expiry_date") in wanted]
        if not sessions:
            raise ValueError(f"No trading sessions between {start_raw} and {end_raw}")
        return sessions

    return all_sessions


def check_dhan_status() -> Dict[str, Any]:
    client = DhanDataClient()
    prof = client.profile()
    return {
        "data_plan": prof.get("dataPlan"),
        "data_validity": prof.get("dataValidity"),
        "token_validity": prof.get("tokenValidity"),
        "active": str(prof.get("dataPlan") or "").lower() == "active",
    }


def _in_band(px: float, low: float, high: float) -> bool:
    return low <= px <= high


def _band_touched(bar_low: float, bar_high: float, band_low: float, band_high: float) -> bool:
    return bar_low <= band_high and bar_high >= band_low


def _estimate_entry(
    open_p: float,
    high: float,
    low: float,
    band_low: float,
    band_high: float,
) -> Optional[float]:
    """Legacy wick-fill estimate (debug scripts only). Backtest uses bar close in band."""
    if not _band_touched(low, high, band_low, band_high):
        return None
    if _in_band(open_p, band_low, band_high):
        return round(open_p, 2)
    if open_p > band_high:
        return band_high
    if open_p < band_low:
        return band_low
    return round((band_low + band_high) / 2.0, 2)


def _entry_from_bar_close(close: float, band_low: float, band_high: float) -> Optional[float]:
    """Entry at 5m close when premium closes inside the band (no wick-fill at band top)."""
    if close <= 0 or not sensex_premium_in_band(close, band_low, band_high):
        return None
    return round(close, 2)



def _series_bar(series: OptionSeries, idx: int) -> MinuteBar:
    ts = series.timestamps[idx]
    dt = datetime.fromtimestamp(ts, tz=IST)
    return MinuteBar(
        idx=idx,
        ts=ts,
        ist_time=dt.strftime("%Y-%m-%d %H:%M:%S"),
        ist_minutes=dt.hour * 60 + dt.minute,
        open=series.open[idx],
        high=series.high[idx],
        low=series.low[idx],
        close=series.close[idx],
        oi=series.oi[idx],
        spot=series.spot[idx],
        strike=int(series.strike[idx]),
        offset=series.offset,
    )


def _ref_series(session: Dict[str, Dict[str, OptionSeries]]) -> Optional[OptionSeries]:
    for kind in ("CE", "PE"):
        ref = (session.get(kind) or {}).get("ATM")
        if ref and ref.timestamps:
            return ref
    return None


def _contract_key(kind: str, strike: int) -> str:
    return f"SENSEX-{int(strike)}-{str(kind).upper()}"


def _ref_idx_after_ts(ref: OptionSeries, ts: int) -> int:
    """First ref bar strictly after `ts` (resume scan after exit)."""
    for idx, bar_ts in enumerate(ref.timestamps):
        if int(bar_ts) > int(ts):
            return idx
    return len(ref.timestamps)


def _pick_entry_auto(
    session: Dict[str, Dict[str, OptionSeries]],
    band_low: float,
    band_high: float,
    cutoff_minutes: int,
    index_open: float = 0.0,
    prev_close: float = 0.0,
    *,
    scan_start_min: Optional[int] = None,
    start_idx: int = 0,
    contract_trade_counts: Optional[Dict[str, int]] = None,
    max_trades_per_contract: int = MAX_TRADES_PER_CONTRACT_PER_DAY,
) -> Optional[Tuple[MinuteBar, str, OptionSeries]]:
    """AUTO: gap direction (PE gap-down, CE gap-up/flat) + smart OI strike in band."""
    kind = sensex_gap_direction_kind(index_open, prev_close)
    return _pick_entry(
        session,
        kind,
        band_low,
        band_high,
        cutoff_minutes,
        prev_close=prev_close,
        scan_start_min=scan_start_min,
        start_idx=start_idx,
        contract_trade_counts=contract_trade_counts,
        max_trades_per_contract=max_trades_per_contract,
    )


def _pick_entry(
    session: Dict[str, Dict[str, OptionSeries]],
    kind: str,
    band_low: float,
    band_high: float,
    cutoff_minutes: int,
    prev_close: float = 0.0,
    *,
    scan_start_min: Optional[int] = None,
    start_idx: int = 0,
    contract_trade_counts: Optional[Dict[str, int]] = None,
    max_trades_per_contract: int = MAX_TRADES_PER_CONTRACT_PER_DAY,
) -> Optional[Tuple[MinuteBar, str, OptionSeries]]:
    """CE/PE: smart strike within the chosen leg when premium is in band."""
    leg = session.get(kind) or {}
    ref = leg.get("ATM")
    if not ref or not ref.timestamps:
        return None

    counts = contract_trade_counts or {}
    scan_start = int(scan_start_min if scan_start_min is not None else sensex_entry_scan_start_minutes())
    for idx in range(max(0, int(start_idx)), len(ref.timestamps)):
        ts = ref.timestamps[idx]
        dt = datetime.fromtimestamp(ts, tz=IST)
        bar_minutes = dt.hour * 60 + dt.minute
        if bar_minutes < scan_start:
            continue
        if bar_minutes >= cutoff_minutes:
            break

        picked = pick_smart_at_bar(
            session,
            idx,
            kinds=(kind.upper(),),
            band_low=band_low,
            band_high=band_high,
            prev_close=prev_close,
        )
        if not picked:
            continue
        candidate, series = picked
        bar = _series_bar(series, idx)
        if sensex_is_bad_option_bar(bar.open, bar.high, bar.close):
            continue
        contract = _contract_key(series.kind, bar.strike)
        if counts.get(contract, 0) >= max(1, int(max_trades_per_contract)):
            continue
        source = strike_source_label(candidate.offset)
        entry = _entry_from_bar_close(bar.close, band_low, band_high)
        if entry is not None:
            return bar, source, series

    return None


def _r_unit(entry: float, fixed_sl_premium: float) -> float:
    """1R = entry premium minus fixed initial SL premium."""
    return max(0.05, float(entry) - max(0.05, float(fixed_sl_premium)))


def _stepped_trail_sl(entry: float, peak: float, r_inr: float) -> float:
    """
    Stepped-R trail after 1R target (aligned with live momentum_trail stepped_rr).

    Example entry=23, R=10: target1=33 → SL moves to 23; at peak 43 SL=33; at peak 53 SL=43.
    """
    r = max(0.05, float(r_inr))
    step = max(1, int((peak - entry) / r))
    locked_step = max(0, step - 1)
    if locked_step >= 1:
        return round(entry + locked_step * r, 2)
    return round(entry, 2)


def _backtest_size_from_allocation(
    capital: float,
    risk_pct: float,
    entry_premium: float,
    lot_size: int,
) -> Tuple[int, int, float]:
    """Lots from risk% capital allocation ÷ entry premium (not SL distance)."""
    alloc_inr = capital * (risk_pct / 100.0)
    if entry_premium <= 0:
        return 1, lot_size, alloc_inr
    lots = int(alloc_inr / (entry_premium * lot_size))
    lots = max(1, lots)
    quantity = lots * lot_size
    return lots, quantity, round(alloc_inr, 2)


def _simulate_from_entry(
    entry: float,
    series: OptionSeries,
    entry_bar_idx: int,
    fixed_sl_premium: float,
    min_target_high: float,
) -> Tuple[float, str, int]:
    """
    Stepped trailing stop on 5m bars after fill at entry close.

    fixed_sl_premium: absolute initial SL option price (e.g. ₹9).
    R = entry − fixed_sl; 1R target = entry + R; trail steps every R.
    Trail SL is not checked on the bar where 1R first activates.
    """
    initial_sl = round(max(0.05, float(fixed_sl_premium)), 2)
    r = _r_unit(entry, initial_sl)
    first_target = round(entry + r, 2)
    max_target = round(entry + max(r, float(min_target_high)), 2)

    trailing = False
    activation_idx: Optional[int] = None
    peak = entry
    first_idx = entry_bar_idx + 1
    if first_idx >= len(series.timestamps):
        last_close = round(series.close[entry_bar_idx], 2)
        return last_close, "eod", entry_bar_idx

    exit_idx = first_idx
    for idx in range(first_idx, len(series.timestamps)):
        low = series.low[idx]
        high = series.high[idx]
        exit_idx = idx

        if not trailing and high >= first_target:
            trailing = True
            activation_idx = idx
            peak = high
        elif trailing:
            peak = max(peak, high)

        if trailing:
            sl = _stepped_trail_sl(entry, peak, r)
        else:
            sl = initial_sl
            if high >= max_target and max_target > first_target:
                return max_target, "target_max", idx

        skip_trail_sl = trailing and activation_idx == idx
        if not skip_trail_sl and low <= sl:
            reason = "trail_stop" if trailing else "stop_loss"
            return round(sl, 2), reason, idx

    last_close = round(series.close[exit_idx], 2)
    if trailing:
        return last_close, "eod_trail", exit_idx
    return last_close, "eod", exit_idx


def _resolve_exit_bar_index(
    entry: float,
    series: OptionSeries,
    start_idx: int,
    exit_px: float,
    reason: str,
    sim_exit_idx: int,
    fixed_sl_premium: float,
) -> int:
    """Map simulated exit to a display bar — profit exits never show the entry bar."""
    sl = round(max(0.05, float(fixed_sl_premium)), 2)
    if reason == "stop_loss":
        if series.low[start_idx] <= sl:
            return start_idx
        for idx in range(start_idx + 1, len(series.timestamps)):
            if series.low[idx] <= sl:
                return idx
        return sim_exit_idx

    if sim_exit_idx == start_idx and start_idx + 1 < len(series.timestamps):
        return start_idx + 1
    return sim_exit_idx


def _simulate_trade(
    expiry_date: str,
    index_open: float,
    bar: MinuteBar,
    source: str,
    series: OptionSeries,
    session: Dict[str, Dict[str, OptionSeries]],
    params: BacktestParams,
    ref: OptionSeries,
) -> Tuple[Optional[TradeResult], int]:
    """Simulate one round-trip; returns (trade, next ref scan index after exit)."""
    kind = series.kind
    entry = _entry_from_bar_close(bar.close, params.entry_band_low, params.entry_band_high)
    if entry is None:
        return None, bar.idx + 1

    fixed_sl = round(max(0.05, float(params.sl_inr)), 2)
    if fixed_sl >= entry:
        return None, bar.idx + 1

    entry_strike = int(bar.strike)
    entry_ts = series.timestamps[bar.idx]
    exit_bundle = build_entry_offset_exit_series(
        session,
        kind=kind,
        entry_offset=series.offset,
        entry_ts=entry_ts,
        entry_strike=entry_strike,
        session_date=expiry_date,
    )
    sim_series = series
    sim_entry_idx = bar.idx
    if exit_bundle:
        sim_series, sim_entry_idx = exit_bundle

    exit_px, reason, exit_idx = _simulate_from_entry(
        entry,
        sim_series,
        sim_entry_idx,
        fixed_sl,
        params.min_target_high,
    )
    display_exit_idx = _resolve_exit_bar_index(
        entry, sim_series, sim_entry_idx, exit_px, reason, exit_idx, fixed_sl
    )
    r_unit = _r_unit(entry, fixed_sl)
    pnl_per_unit = exit_px - entry
    r_mult = round(pnl_per_unit / r_unit, 2)
    exit_ts = int(sim_series.timestamps[display_exit_idx])
    next_idx = _ref_idx_after_ts(ref, exit_ts)

    return (
        TradeResult(
            expiry_date=expiry_date,
            direction=kind,
            strike=entry_strike,
            kind=kind,
            strike_source=source,
            symbol=_contract_key(kind, entry_strike),
            entry=entry,
            exit=exit_px,
            sl=fixed_sl,
            target=round(entry + r_unit, 2),
            pnl_inr=0.0,
            r_multiple=r_mult,
            exit_reason=reason,
            index_open=index_open,
            premium_open=bar.open,
            premium_high=bar.high,
            premium_low=bar.low,
            entry_datetime_ist=bar.ist_time,
            exit_datetime_ist=ts_to_ist_label(exit_ts),
        ),
        next_idx,
    )


def _run_day(
    expiry_date: str,
    index_open: float,
    prev_close: float,
    session: Dict[str, Dict[str, OptionSeries]],
    params: BacktestParams,
) -> List[TradeResult]:
    """Up to N round-trips per contract per session day; re-entries after exit."""
    scan_start, cutoff, _, _ = resolve_backtest_scan_window(params)
    direction = (params.direction or "AUTO").upper()
    ref = _ref_series(session)
    if not ref:
        return []

    contract_counts: Dict[str, int] = {}
    trades: List[TradeResult] = []
    start_idx = 0

    while start_idx < len(ref.timestamps):
        if direction == "AUTO":
            picked = _pick_entry_auto(
                session,
                params.entry_band_low,
                params.entry_band_high,
                cutoff,
                index_open=index_open,
                prev_close=prev_close,
                scan_start_min=scan_start,
                start_idx=start_idx,
                contract_trade_counts=contract_counts,
            )
        else:
            picked = _pick_entry(
                session,
                direction,
                params.entry_band_low,
                params.entry_band_high,
                cutoff,
                prev_close=prev_close,
                scan_start_min=scan_start,
                start_idx=start_idx,
                contract_trade_counts=contract_counts,
            )
        if not picked:
            break

        bar, source, series = picked
        trade, next_idx = _simulate_trade(
            expiry_date,
            index_open,
            bar,
            source,
            series,
            session,
            params,
            ref,
        )
        if next_idx <= start_idx:
            start_idx += 1
        else:
            start_idx = next_idx
        if trade is None:
            continue

        contract = trade.symbol
        contract_counts[contract] = contract_counts.get(contract, 0) + 1
        trades.append(trade)

    return trades


def _missing_session_offsets(
    session: Dict[str, Dict[str, OptionSeries]],
) -> List[str]:
    """Offsets required for current band but absent or empty in a cached session."""
    required = sensex_atm_near_offsets()
    missing: List[str] = []
    for kind in ("CE", "PE"):
        leg = session.get(kind) or {}
        for off in required:
            series = leg.get(off)
            if not series or not series.timestamps:
                if off not in missing:
                    missing.append(off)
    return missing


def _merge_session_series(
    base: Dict[str, Dict[str, OptionSeries]],
    extra: Dict[str, Dict[str, OptionSeries]],
) -> Dict[str, Dict[str, OptionSeries]]:
    out: Dict[str, Dict[str, OptionSeries]] = {
        kind: dict(offsets) for kind, offsets in base.items()
    }
    for kind, offsets in extra.items():
        out.setdefault(kind, {})
        out[kind].update(offsets)
    return out


def fetch_sessions_data(
    sessions: List[Dict[str, Any]],
    *,
    interval_min: int = 5,
    refresh: bool = False,
    progress_cb: Optional[Any] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, OptionSeries]]], Dict[str, int]]:
    loaded: Dict[str, Dict[str, Dict[str, OptionSeries]]] = {}
    stats = {"cached": 0, "fetched": 0}
    pending: List[str] = []
    pending_topup: List[Tuple[str, List[str]]] = []
    total = len(sessions)
    iv_label = _interval_label(interval_min)

    for i, row in enumerate(sessions, start=1):
        expiry = row["expiry_date"]
        if progress_cb:
            progress_cb(
                BacktestProgress(
                    phase="fetch",
                    current=i,
                    total=total,
                    expiry_date=expiry,
                    message=f"Loading Dhan {iv_label} data for {expiry}",
                )
            )
        if refresh:
            pending.append(expiry)
            continue
        cached = load_cached_session(CACHE_DIR, expiry, interval_min)
        if not cached:
            pending.append(expiry)
            continue
        missing = _missing_session_offsets(cached)
        loaded[expiry] = cached
        stats["cached"] += 1
        if missing:
            pending_topup.append((expiry, missing))

    if pending or pending_topup:
        client = DhanDataClient()
        prof = client.profile()
        if str(prof.get("dataPlan") or "").lower() != "active":
            raise RuntimeError("Dhan Data API is not active. Subscribe at dhan.co and set DHAN_ACCESS_TOKEN.")
        interval_str = interval_to_str(interval_min)
        for expiry in pending:
            series = client.fetch_sensex_session(
                expiry,
                offsets=sensex_atm_near_offsets(),
                interval=interval_str,
            )
            save_cached_session(
                CACHE_DIR,
                expiry,
                series,
                interval_min=interval_min,
                meta={"profile_dataPlan": prof.get("dataPlan"), "interval_min": interval_min},
            )
            loaded[expiry] = series
            stats["fetched"] += 1
        for expiry, missing_offsets in pending_topup:
            extra = client.fetch_sensex_session(
                expiry,
                offsets=missing_offsets,
                interval=interval_str,
            )
            merged = _merge_session_series(loaded[expiry], extra)
            save_cached_session(
                CACHE_DIR,
                expiry,
                merged,
                interval_min=interval_min,
                meta={
                    "profile_dataPlan": prof.get("dataPlan"),
                    "interval_min": interval_min,
                    "topped_up_offsets": missing_offsets,
                },
            )
            loaded[expiry] = merged
            stats["fetched"] += 1

    return loaded, stats


def _run_backtest_for_timeframe(
    params: BacktestParams,
    sessions: List[Dict[str, Any]],
    interval_min: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Run the strategy on one bar interval; returns (report_block, fetch_stats)."""
    data, fetch_stats = fetch_sessions_data(
        sessions,
        interval_min=interval_min,
        refresh=params.refresh_dhan,
    )
    iv_label = _interval_label(interval_min)
    start_capital = max(1000.0, float(params.capital))
    scan_start, cutoff, scan_start_label, scan_end_label = resolve_backtest_scan_window(params)
    trades: List[TradeResult] = []
    skipped: List[Dict[str, str]] = []
    equity = start_capital
    peak_equity = start_capital
    max_drawdown_inr = 0.0

    prev_spot_close = 0.0
    for row in sessions:
        session_date = row["expiry_date"]
        session = data.get(session_date)
        if not session:
            skipped.append({"expiry_date": session_date, "reason": f"no Dhan {iv_label} data"})
            continue

        index_open, _, prev_close = session_index_from_spot(
            session,
            prev_trading_close=prev_spot_close,
        )
        if index_open <= 0:
            skipped.append({"expiry_date": session_date, "reason": "no Sensex spot in Dhan data"})
            continue
        if prev_spot_close <= 0 and _f(row.get("prev_close")) > 0:
            prev_close = _f(row.get("prev_close"))

        tr_list = _run_day(session_date, index_open, prev_close, session, params)
        spot_s = session_spot_series(session)
        if spot_s and spot_s.spot:
            prev_spot_close = float(spot_s.spot[-1])
        if not tr_list:
            skipped.append(
                {
                    "expiry_date": session_date,
                    "reason": (
                        f"no {iv_label} close in ₹{params.entry_band_low:g}–₹{params.entry_band_high:g} "
                        f"from {scan_start_label} to {scan_end_label} IST"
                    ),
                }
            )
            continue

        for tr in tr_list:
            lots, qty, alloc_inr = _backtest_size_from_allocation(
                equity,
                params.risk_pct,
                tr.entry,
                LOT_SIZE,
            )
            notional = tr.entry * qty
            if notional > equity:
                skipped.append(
                    {
                        "expiry_date": session_date,
                        "reason": (
                            f"{tr.symbol} insufficient capital "
                            f"(need ₹{notional:.0f}, have ₹{equity:.0f})"
                        ),
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
            tr.risk_at_sl_inr = round(alloc_inr, 2)
            tr.capital_before = round(cap_before, 2)
            tr.capital_after = equity
            trades.append(tr)

    wins = [t for t in trades if t.pnl_inr > 0]
    total_pnl = sum(t.pnl_inr for t in trades)
    ending = round(start_capital + total_pnl, 2)
    avg_lots = round(sum(t.num_lots for t in trades) / len(trades), 1) if trades else 0.0
    avg_risk = round(sum(t.risk_at_sl_inr for t in trades) / len(trades), 2) if trades else 0.0
    report = {
        "summary": {
            "strategy": STRATEGY_NAME,
            "timeframe": iv_label,
            "interval_min": interval_min,
            "direction_mode": params.direction.upper(),
            "data_source": f"Dhan rollingoption {iv_label} (BSE_FNO securityId 51)",
            "starting_capital_inr": round(start_capital, 2),
            "ending_capital_inr": ending,
            "return_pct": round((ending - start_capital) / start_capital * 100.0, 2)
            if start_capital > 0
            else 0.0,
            "max_drawdown_inr": round(max_drawdown_inr, 2),
            "sessions": len(sessions),
            "trades": len(trades),
            "skipped": len(skipped),
            "wins": len(wins),
            "losses": len([t for t in trades if t.pnl_inr < 0]),
            "win_rate_pct": round(100.0 * len(wins) / len(trades), 1) if trades else 0.0,
            "total_pnl_inr": round(total_pnl, 2),
            "avg_pnl_inr": round(total_pnl / len(trades), 2) if trades else 0.0,
            "avg_r": round(sum(t.r_multiple for t in trades) / len(trades), 2) if trades else 0.0,
            "lot_size": LOT_SIZE,
            "risk_pct": params.risk_pct,
            "max_lots_cap": None,
            "live_max_lots_cap": MAX_LOTS,
            "avg_lots_per_trade": avg_lots,
            "max_risk_per_trade_inr": avg_risk,
            "entry_scan_start_ist": scan_start_label,
            "entry_scan_end_ist": scan_end_label,
            "entry_cutoff_ist": scan_end_label,
            "sl_inr": params.sl_inr,
            "entry_band": [params.entry_band_low, params.entry_band_high],
            "min_target_band": [params.min_target_low, params.min_target_high],
            "max_trades_per_contract_per_day": MAX_TRADES_PER_CONTRACT_PER_DAY,
        },
        "trades": [asdict(t) for t in trades],
        "skipped": skipped,
    }
    return report, fetch_stats


def run_sensex_dhan_backtest(params: BacktestParams) -> Dict[str, Any]:
    sessions = _resolve_backtest_sessions(params)
    timeframes = _normalize_timeframes(params.timeframes_min)
    start_capital = max(1000.0, float(params.capital))
    _, _, scan_start_label, scan_end_label = resolve_backtest_scan_window(params)

    reports: Dict[str, Any] = {}
    fetch_stats_by_tf: Dict[str, Dict[str, int]] = {}
    for interval_min in timeframes:
        key = _timeframe_key(interval_min)
        report, fetch_stats = _run_backtest_for_timeframe(params, sessions, interval_min)
        reports[key] = report
        fetch_stats_by_tf[key] = fetch_stats

    primary_key = _timeframe_key(timeframes[0])
    if len(timeframes) == 1:
        reports["trail"] = reports[primary_key]

    tf_labels = ", ".join(_interval_label(iv) for iv in timeframes)
    total_cached = sum(s.get("cached", 0) for s in fetch_stats_by_tf.values())
    total_fetched = sum(s.get("fetched", 0) for s in fetch_stats_by_tf.values())

    note = (
        f"Starting capital ₹{start_capital:,.0f} · {params.risk_pct:g}% allocation per trade · "
        f"fixed initial SL premium ₹{params.sl_inr:g} · 1R = entry − SL · 1R target = entry + 1R · "
        f"stepped trail every 1R · entry ₹{params.entry_band_low:g}–₹{params.entry_band_high:g}. "
        f"Timeframes: {tf_labels}. "
        f"Lots = floor(capital × risk% ÷ (entry × {LOT_SIZE})) — not sized from SL distance. "
        f"AUTO picks PE on gap-down and CE on gap-up/flat; monitors ATM±{sensex_atm_near_steps()} per leg for band close. "
        f"Skips bad option ticks (open/high > 3× close). "
        f"Entry scan {scan_start_label}–{scan_end_label} IST when bar close is in band. "
        f"Exit simulation starts on the bar after entry (fill at close). "
        f"Trail SL deferred on the 1R activation bar (no same-bar low wick stop). "
        f"Exits stay on the entry offset series from the entry bar onward. "
        f"Up to {MAX_TRADES_PER_CONTRACT_PER_DAY} round-trips per contract per session day (re-entry after exit). "
        f"All trading days in window ({len(sessions)} sessions). "
        f"Dhan fetch totals: {total_cached} cached, {total_fetched} fetched across selected timeframes."
    )
    return {
        "note": note,
        "starting_capital_inr": round(start_capital, 2),
        "risk_pct": params.risk_pct,
        "sl_inr": params.sl_inr,
        "entry_band": [params.entry_band_low, params.entry_band_high],
        "min_target_band": [params.min_target_low, params.min_target_high],
        "entry_scan_start_ist": scan_start_label,
        "entry_scan_end_ist": scan_end_label,
        "timeframes_min": timeframes,
        "data_source": "dhan_rolling",
        "dhan_status": check_dhan_status(),
        "selected_expiry_dates": [r["expiry_date"] for r in sessions],
        "start_date": params.start_date,
        "end_date": params.end_date,
        "dhan_fetch_stats": fetch_stats_by_tf.get(primary_key, {"cached": 0, "fetched": 0}),
        "dhan_fetch_stats_by_timeframe": fetch_stats_by_tf,
        "reports": reports,
        "generated_at": datetime.now(IST).isoformat(),
    }
