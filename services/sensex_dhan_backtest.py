"""
Sensex 20rupees-strategy backtest on Dhan 5-minute rolling options data.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

from services.dhan_data_client import (
    DhanDataClient,
    OptionSeries,
    cache_path,
    load_cached_session,
    save_cached_session,
    ts_to_ist_label,
)
from services.momentum_trail import breakeven_stop, get_momentum_trail_config
from services.sensex_constants import (
    sensex_entry_cutoff_minutes,
    sensex_entry_scan_start_minutes,
    sensex_default_min_target_inr,
    sensex_is_bad_option_bar,
    sensex_is_gap_up_session,
    sensex_max_lots_per_trade,
    sensex_premium_in_band,
)
from services.sensex_indicator_plan import size_from_risk
from services.sensex_strike_selection import pick_smart_at_bar, strike_source_label
from services.sensex_strategy_analysis import STRATEGY_NAME

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sensex"
CACHE_DIR = DATA_DIR / "dhan_intraday"
OHLC_PATH = DATA_DIR / "weekly_expiry_day_ohlc.csv"
LOT_SIZE = 20
MAX_LOTS = sensex_max_lots_per_trade()
DEFAULT_RISK_PCT = 1.0
DEFAULT_SL_INR = 10.0
DEFAULT_ENTRY_LOW = 17.0
DEFAULT_ENTRY_HIGH = 23.0
DEFAULT_MIN_TARGET_LOW = sensex_default_min_target_inr()
DEFAULT_MIN_TARGET_HIGH = sensex_default_min_target_inr()


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
    mode: str = "conservative"
    expiry_dates: Optional[List[str]] = None
    refresh_dhan: bool = False


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
    rows: List[Dict[str, Any]] = []
    with OHLC_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    rows.sort(key=lambda r: r.get("expiry_date") or "")
    return rows


def list_available_sessions() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in _load_all_sessions():
        expiry = row.get("expiry_date") or ""
        cached = cache_path(CACHE_DIR, expiry).exists() if expiry else False
        out.append(
            {
                "expiry_date": expiry,
                "expiry_weekday": row.get("expiry_weekday"),
                "holiday_adjusted": row.get("holiday_adjusted") == "True",
                "index_open": _f(row.get("open")),
                "index_close": _f(row.get("close")),
                "prev_close": _f(row.get("prev_close")),
                "dhan_cached": cached,
            }
        )
    return out


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


def _pick_entry_auto(
    session: Dict[str, Dict[str, OptionSeries]],
    band_low: float,
    band_high: float,
    cutoff_minutes: int,
    prev_close: float = 0.0,
) -> Optional[Tuple[MinuteBar, str, OptionSeries]]:
    """AUTO: smart OI + proximity strike when premium band is touched."""
    ref = _ref_series(session)
    if not ref:
        return None

    scan_start = sensex_entry_scan_start_minutes()
    for idx in range(len(ref.timestamps)):
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
            kinds=("CE", "PE"),
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
        source = strike_source_label(candidate.offset)
        entry = _entry_from_bar_close(bar.close, band_low, band_high)
        if entry is not None:
            return bar, source, series

    return None


def _pick_entry(
    session: Dict[str, Dict[str, OptionSeries]],
    kind: str,
    band_low: float,
    band_high: float,
    cutoff_minutes: int,
    prev_close: float = 0.0,
) -> Optional[Tuple[MinuteBar, str, OptionSeries]]:
    """CE/PE: smart strike within the chosen leg when premium is in band."""
    leg = session.get(kind) or {}
    ref = leg.get("ATM")
    if not ref or not ref.timestamps:
        return None

    scan_start = sensex_entry_scan_start_minutes()
    for idx in range(len(ref.timestamps)):
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
        source = strike_source_label(candidate.offset)
        entry = _entry_from_bar_close(bar.close, band_low, band_high)
        if entry is not None:
            return bar, source, series

    return None


def _trail_stop(entry: float, peak: float, sl_inr: float, *, trail_active: bool) -> float:
    """Stepped-R trail aligned with live momentum_trail.compute_trailed_levels."""
    r = max(0.05, float(sl_inr))
    cfg = get_momentum_trail_config()
    be = breakeven_stop(entry, r, cfg)
    if not trail_active:
        return be
    step = max(1, int((peak - entry) / r))
    locked_step = max(0, step - 1)
    locked = round(entry + locked_step * r, 2) if locked_step >= 1 else be
    return round(max(be, locked), 2)


def _simulate_from_entry(
    entry: float,
    series: OptionSeries,
    entry_bar_idx: int,
    mode: str,
    sl_inr: float,
    min_target_low: float,
    min_target_high: float,
) -> Tuple[float, str, int]:
    """
    Simulate exits after a fill at the entry bar's close.

    OHLC on the entry bar includes price action before the fill, so exit logic
    starts on the next 5m bar. Conservative mode also skips trail SL checks on
    the bar where min-target is first touched (SL-before-target on same bar).
    """
    r = max(0.05, float(sl_inr))
    tgt_low = max(r, float(min_target_low))
    tgt_high = max(tgt_low, float(min_target_high))
    sl = round(entry - r, 2)
    trail_trigger = round(entry + tgt_low, 2)
    trailing = False
    defer_trail_sl = False
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

        if not trailing:
            hit_sl = low <= sl
            hit_min_tgt = high >= trail_trigger
            hit_max_tgt = high >= round(entry + tgt_high, 2)
            if mode == "conservative":
                if hit_sl:
                    return sl, "stop_loss", idx
                if hit_min_tgt:
                    trailing = True
                    peak = high
                    defer_trail_sl = True
            else:
                if hit_sl and hit_min_tgt:
                    trailing = True
                    peak = high
                elif hit_sl:
                    return sl, "stop_loss", idx
                elif hit_min_tgt:
                    trailing = True
                    peak = high
            if not trailing and hit_max_tgt and tgt_high > tgt_low:
                return round(entry + tgt_high, 2), "target_max", idx

        if trailing:
            peak = max(peak, high)
            if defer_trail_sl:
                defer_trail_sl = False
            else:
                trail_sl = _trail_stop(entry, peak, r, trail_active=True)
                if low <= trail_sl:
                    return round(trail_sl, 2), "trail_stop", idx

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
    sl_inr: float,
) -> int:
    """Map simulated exit to a display bar — profit exits never show the entry bar."""
    sl = round(entry - sl_inr, 2)
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


def _run_day(
    expiry_date: str,
    index_open: float,
    prev_close: float,
    session: Dict[str, Dict[str, OptionSeries]],
    params: BacktestParams,
    mode: str,
) -> Optional[TradeResult]:
    cutoff = sensex_entry_cutoff_minutes()
    direction = (params.direction or "AUTO").upper()
    if direction == "AUTO":
        picked = _pick_entry_auto(
            session,
            params.entry_band_low,
            params.entry_band_high,
            cutoff,
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
        return None

    bar, source, series = picked
    kind = series.kind
    entry = _entry_from_bar_close(bar.close, params.entry_band_low, params.entry_band_high)
    if entry is None:
        return None

    exit_px, reason, exit_idx = _simulate_from_entry(
        entry,
        series,
        bar.idx,
        mode,
        params.sl_inr,
        params.min_target_low,
        params.min_target_high,
    )
    display_exit_idx = _resolve_exit_bar_index(
        entry, series, bar.idx, exit_px, reason, exit_idx, params.sl_inr
    )
    r_unit = max(0.05, float(params.sl_inr))
    pnl_per_unit = exit_px - entry
    r_mult = round(pnl_per_unit / r_unit, 2)

    return TradeResult(
        expiry_date=expiry_date,
        direction=kind,
        strike=bar.strike,
        kind=kind,
        strike_source=source,
        symbol=f"SENSEX-{bar.strike}-{kind}",
        entry=entry,
        exit=exit_px,
        sl=round(entry - params.sl_inr, 2),
        target=round(entry + params.min_target_low, 2),
        pnl_inr=0.0,
        r_multiple=r_mult,
        exit_reason=reason,
        index_open=index_open,
        premium_open=bar.open,
        premium_high=bar.high,
        premium_low=bar.low,
        entry_datetime_ist=bar.ist_time,
        exit_datetime_ist=ts_to_ist_label(series.timestamps[display_exit_idx]),
    )


def fetch_sessions_data(
    sessions: List[Dict[str, Any]],
    *,
    refresh: bool = False,
    progress_cb: Optional[Any] = None,
) -> Dict[str, Dict[str, Dict[str, OptionSeries]]]:
    client = DhanDataClient()
    prof = client.profile()
    if str(prof.get("dataPlan") or "").lower() != "active":
        raise RuntimeError("Dhan Data API is not active. Subscribe at dhan.co and set DHAN_ACCESS_TOKEN.")

    loaded: Dict[str, Dict[str, Dict[str, OptionSeries]]] = {}
    total = len(sessions)
    for i, row in enumerate(sessions, start=1):
        expiry = row["expiry_date"]
        if progress_cb:
            progress_cb(
                BacktestProgress(
                    phase="fetch",
                    current=i,
                    total=total,
                    expiry_date=expiry,
                    message=f"Loading Dhan data for {expiry}",
                )
            )
        if not refresh:
            cached = load_cached_session(CACHE_DIR, expiry)
            if cached:
                loaded[expiry] = cached
                continue
        series = client.fetch_sensex_session(expiry)
        save_cached_session(CACHE_DIR, expiry, series, meta={"profile_dataPlan": prof.get("dataPlan")})
        loaded[expiry] = series
    return loaded


def run_sensex_dhan_backtest(params: BacktestParams) -> Dict[str, Any]:
    all_sessions = _load_all_sessions()
    if params.expiry_dates:
        wanted = set(params.expiry_dates)
        sessions = [r for r in all_sessions if r.get("expiry_date") in wanted]
        if not sessions:
            raise ValueError("No matching expiry sessions for selected dates")
    else:
        sessions = all_sessions

    data = fetch_sessions_data(sessions, refresh=params.refresh_dhan)
    modes = ["conservative", "optimistic"] if params.mode == "both" else [params.mode]
    if params.mode not in ("conservative", "optimistic", "both"):
        raise ValueError("mode must be conservative, optimistic, or both")

    start_capital = max(1000.0, float(params.capital))
    cutoff = sensex_entry_cutoff_minutes()
    reports: Dict[str, Any] = {}

    for m in modes:
        trades: List[TradeResult] = []
        skipped: List[Dict[str, str]] = []
        equity = start_capital
        peak_equity = start_capital
        max_drawdown_inr = 0.0

        for row in sessions:
            expiry_date = row["expiry_date"]
            index_open = _f(row.get("open"))
            prev_close = _f(row.get("prev_close"))
            session = data.get(expiry_date)
            if not session:
                skipped.append({"expiry_date": expiry_date, "reason": "no Dhan data"})
                continue

            if sensex_is_gap_up_session(index_open, prev_close):
                skipped.append(
                    {
                        "expiry_date": expiry_date,
                        "reason": "gap-up session skipped (index open > prev close)",
                    }
                )
                continue

            tr = _run_day(expiry_date, index_open, prev_close, session, params, m)
            if tr:
                sl_prem = round(tr.entry - params.sl_inr, 2)
                lots, qty, risk_inr = size_from_risk(
                    equity,
                    params.risk_pct,
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
                skipped.append(
                    {
                        "expiry_date": expiry_date,
                        "reason": (
                            f"no 5m close in ₹{params.entry_band_low:g}–₹{params.entry_band_high:g} "
                            f"from {sensex_entry_scan_start_minutes() // 60:02d}:{sensex_entry_scan_start_minutes() % 60:02d} "
                            f"to {cutoff // 60:02d}:{cutoff % 60:02d} IST"
                        ),
                    }
                )

        wins = [t for t in trades if t.pnl_inr > 0]
        total_pnl = sum(t.pnl_inr for t in trades)
        ending = round(start_capital + total_pnl, 2)
        avg_lots = round(sum(t.num_lots for t in trades) / len(trades), 1) if trades else 0.0
        avg_risk = round(sum(t.risk_at_sl_inr for t in trades) / len(trades), 2) if trades else 0.0
        reports[m] = {
            "summary": {
                "strategy": STRATEGY_NAME,
                "mode": m,
                "direction_mode": params.direction.upper(),
                "data_source": "Dhan rollingoption 5m (BSE_FNO securityId 51)",
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
                "max_lots_cap": MAX_LOTS,
                "avg_lots_per_trade": avg_lots,
                "max_risk_per_trade_inr": avg_risk,
                "entry_cutoff_ist": f"{cutoff // 60:02d}:{cutoff % 60:02d}",
                "sl_inr": params.sl_inr,
                "entry_band": [params.entry_band_low, params.entry_band_high],
                "min_target_band": [params.min_target_low, params.min_target_high],
            },
            "trades": [asdict(t) for t in trades],
            "skipped": skipped,
        }

    tgt_label = (
        f"₹{params.min_target_low:g}"
        if params.min_target_low == params.min_target_high
        else f"₹{params.min_target_low:g}–₹{params.min_target_high:g}"
    )
    note = (
        f"Starting capital ₹{start_capital:,.0f} · {params.risk_pct:g}% risk per trade · "
        f"₹{params.sl_inr:g} SL · min-target {tgt_label} (trail in ₹{params.sl_inr:g} steps after min-target) · "
        f"entry ₹{params.entry_band_low:g}–₹{params.entry_band_high:g}. "
        f"Skips gap-up sessions (open > prev close) and bad option ticks (open/high > 3× close). "
        f"Entry from {sensex_entry_scan_start_minutes() // 60:02d}:{sensex_entry_scan_start_minutes() % 60:02d} "
        f"when 5m close is in band · trail at entry + ₹{params.min_target_low:g} (1R). "
        f"Exit simulation starts on the bar after entry (fill at close). "
        f"Conservative defers trail SL on the activation bar; optimistic vs conservative only "
        f"differs when SL and 1R target both touch one bar. "
        f"Dhan 5m data on {len(sessions)} expiry session(s). "
        f"Entries before {cutoff // 60:02d}:{cutoff % 60:02d} IST."
    )
    return {
        "note": note,
        "starting_capital_inr": round(start_capital, 2),
        "risk_pct": params.risk_pct,
        "sl_inr": params.sl_inr,
        "entry_band": [params.entry_band_low, params.entry_band_high],
        "min_target_band": [params.min_target_low, params.min_target_high],
        "data_source": "dhan_5m_rolling",
        "dhan_status": check_dhan_status(),
        "selected_expiry_dates": [r["expiry_date"] for r in sessions],
        "reports": reports,
        "generated_at": datetime.now(IST).isoformat(),
    }
