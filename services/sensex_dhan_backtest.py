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
from services.sensex_constants import sensex_entry_cutoff_minutes, sensex_max_lots_per_trade
from services.sensex_indicator_plan import size_from_risk
from services.sensex_strategy_analysis import STRATEGY_NAME

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sensex"
CACHE_DIR = DATA_DIR / "dhan_intraday"
OHLC_PATH = DATA_DIR / "weekly_expiry_day_ohlc.csv"
LOT_SIZE = 20
MAX_LOTS = sensex_max_lots_per_trade()
DEFAULT_RISK_PCT = 1.0
DEFAULT_SL_INR = 10.0
DEFAULT_REWARD_INR = 10.0
DEFAULT_PREMIUM_LOW = 17.0
DEFAULT_PREMIUM_HIGH = 23.0


@dataclass
class BacktestParams:
    capital: float = 1_000_000.0
    risk_pct: float = DEFAULT_RISK_PCT
    sl_inr: float = DEFAULT_SL_INR
    reward_inr: float = DEFAULT_REWARD_INR
    premium_band_low: float = DEFAULT_PREMIUM_LOW
    premium_band_high: float = DEFAULT_PREMIUM_HIGH
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
    if not _band_touched(low, high, band_low, band_high):
        return None
    if _in_band(open_p, band_low, band_high):
        return round(open_p, 2)
    if open_p > band_high:
        return band_high
    if open_p < band_low:
        return band_low
    return round((band_low + band_high) / 2.0, 2)


def _resolve_kind(direction: str, index_open: float, prev_close: float) -> str:
    d = (direction or "AUTO").upper()
    if d in ("CE", "PE"):
        return d
    if prev_close > 0:
        return "CE" if index_open >= prev_close else "PE"
    return "CE"


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


def _pick_entry(
    session: Dict[str, Dict[str, OptionSeries]],
    kind: str,
    band_low: float,
    band_high: float,
    cutoff_minutes: int,
) -> Optional[Tuple[MinuteBar, str, OptionSeries]]:
    leg = session.get(kind) or {}
    ref = leg.get("ATM")
    if not ref or not ref.timestamps:
        return None

    for idx in range(len(ref.timestamps)):
        ts = ref.timestamps[idx]
        dt = datetime.fromtimestamp(ts, tz=IST)
        if dt.hour * 60 + dt.minute >= cutoff_minutes:
            break

        atm_series = leg.get("ATM")
        atm_bar = _series_bar(atm_series, idx) if atm_series else None

        max_oi_offset = None
        max_oi = -1.0
        for offset, series in leg.items():
            if idx >= len(series.oi):
                continue
            oi = series.oi[idx]
            if oi > max_oi:
                max_oi = oi
                max_oi_offset = offset
        max_oi_bar = _series_bar(leg[max_oi_offset], idx) if max_oi_offset else None

        candidates: List[Tuple[str, MinuteBar, OptionSeries]] = []
        if atm_bar:
            candidates.append(("ATM", atm_bar, atm_series))
        if max_oi_bar and max_oi_offset != "ATM":
            candidates.append(("MAX_OI", max_oi_bar, leg[max_oi_offset]))

        for source, bar, series in candidates:
            entry = _estimate_entry(bar.open, bar.high, bar.low, band_low, band_high)
            if entry is not None:
                return bar, source, series

    return None


def _trail_stop(entry: float, peak: float, r: float) -> float:
    cfg = get_momentum_trail_config()
    be = breakeven_stop(entry, r, cfg)
    step = max(1, int((peak - entry) / r))
    locked = round(entry + (step - 1) * r, 2) if step > 1 else be
    return max(be, locked)


def _simulate_from_entry(
    entry: float,
    series: OptionSeries,
    start_idx: int,
    mode: str,
    sl_inr: float,
    reward_inr: float,
) -> Tuple[float, str, int]:
    r = max(0.05, float(sl_inr))
    reward = max(0.05, float(reward_inr))
    sl = round(entry - r, 2)
    target = round(entry + reward, 2)
    trailing = False
    peak = entry
    exit_idx = start_idx

    for idx in range(start_idx, len(series.timestamps)):
        low = series.low[idx]
        high = series.high[idx]
        exit_idx = idx

        if not trailing:
            hit_sl = low <= sl
            hit_tgt = high >= target
            if mode == "conservative":
                if hit_sl:
                    return sl, "stop_loss", idx
                if hit_tgt:
                    trailing = True
                    peak = high
            else:
                if hit_sl and hit_tgt:
                    trailing = True
                    peak = high
                elif hit_sl:
                    return sl, "stop_loss", idx
                elif hit_tgt:
                    trailing = True
                    peak = high

        if trailing:
            peak = max(peak, high)
            trail_sl = _trail_stop(entry, peak, r)
            if low <= trail_sl and trail_sl >= entry:
                return round(trail_sl, 2), "trail_stop", idx

    last_close = round(series.close[exit_idx], 2)
    if trailing:
        return last_close, "eod_trail", exit_idx
    return last_close, "eod", exit_idx


def _run_day(
    expiry_date: str,
    index_open: float,
    prev_close: float,
    session: Dict[str, Dict[str, OptionSeries]],
    params: BacktestParams,
    mode: str,
) -> Optional[TradeResult]:
    kind = _resolve_kind(params.direction, index_open, prev_close)
    cutoff = sensex_entry_cutoff_minutes()
    picked = _pick_entry(
        session,
        kind,
        params.premium_band_low,
        params.premium_band_high,
        cutoff,
    )
    if not picked:
        return None

    bar, source, series = picked
    entry = _estimate_entry(bar.open, bar.high, bar.low, params.premium_band_low, params.premium_band_high)
    if entry is None:
        return None

    exit_px, reason, exit_idx = _simulate_from_entry(
        entry,
        series,
        bar.idx,
        mode,
        params.sl_inr,
        params.reward_inr,
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
        target=round(entry + params.reward_inr, 2),
        pnl_inr=0.0,
        r_multiple=r_mult,
        exit_reason=reason,
        index_open=index_open,
        premium_open=bar.open,
        premium_high=bar.high,
        premium_low=bar.low,
        entry_datetime_ist=bar.ist_time,
        exit_datetime_ist=ts_to_ist_label(series.timestamps[exit_idx]),
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
                            f"premium never in ₹{params.premium_band_low:g}–₹{params.premium_band_high:g} "
                            f"band before {cutoff // 60:02d}:{cutoff % 60:02d} IST"
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
                "reward_inr": params.reward_inr,
                "premium_band": [params.premium_band_low, params.premium_band_high],
            },
            "trades": [asdict(t) for t in trades],
            "skipped": skipped,
        }

    rr = round(params.reward_inr / params.sl_inr, 2) if params.sl_inr > 0 else 0.0
    note = (
        f"Starting capital ₹{start_capital:,.0f} · {params.risk_pct:g}% risk per trade · "
        f"₹{params.sl_inr:g} SL / ₹{params.reward_inr:g} target ({rr:g}:1 R:R) · "
        f"premium ₹{params.premium_band_low:g}–₹{params.premium_band_high:g}. "
        f"Dhan 5m data on {len(sessions)} expiry session(s). "
        f"Entries before {cutoff // 60:02d}:{cutoff % 60:02d} IST."
    )
    return {
        "note": note,
        "starting_capital_inr": round(start_capital, 2),
        "risk_pct": params.risk_pct,
        "sl_inr": params.sl_inr,
        "reward_inr": params.reward_inr,
        "premium_band": [params.premium_band_low, params.premium_band_high],
        "data_source": "dhan_5m_rolling",
        "dhan_status": check_dhan_status(),
        "selected_expiry_dates": [r["expiry_date"] for r in sessions],
        "reports": reports,
        "generated_at": datetime.now(IST).isoformat(),
    }
