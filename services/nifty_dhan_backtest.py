"""
Nifty 50 Dhan backtest — DuckDB FnO cache + live Dhan fetch, stepped SL exit (T1/T2).

Reuses Sensex 20rupees entry scan and SL ratchet simulation from sensex_dhan_backtest.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.dhan_data_client import DhanDataClient, OptionSeries, interval_to_str
from services.entry_quality import exit_model
from services.nifty_duckdb_store import NIFTY_STRIKE_OFFSETS, STAGING_DB_PATH, NiftyDuckDBStore
from services.sensex_constants import sensex_atm_near_offsets, sensex_max_lots_per_trade
from services.sensex_dhan_backtest import (
    BacktestParams,
    TradeResult,
    _backtest_size_from_risk,
    _f,
    _normalize_timeframes,
    _run_day,
    check_dhan_status,
    resolve_backtest_scan_window,
)
from services.sensex_trading_calendar import iter_trading_days, session_index_from_spot, session_spot_series

LOT_SIZE = 75
MAX_LOTS = sensex_max_lots_per_trade()
MAX_TRADES_PER_CONTRACT_PER_DAY = 5
STRATEGY_NAME = "nifty50_dhan_20rupees"


def _default_db_path() -> Path:
    raw = (os.getenv("NIFTY_DUCKDB_PATH") or "").strip()
    if raw:
        return Path(raw)
    main = Path(__file__).resolve().parent.parent / "data" / "market" / "nifty50.duckdb"
    if main.exists():
        try:
            store = NiftyDuckDBStore(main)
            store.connect().close()
            return main
        except Exception:
            pass
    return STAGING_DB_PATH


def _store() -> NiftyDuckDBStore:
    return NiftyDuckDBStore(_default_db_path())


def _contract_key(kind: str, strike: int) -> str:
    return f"NIFTY-{int(strike)}-{str(kind).upper()}"


def _rows_to_series(kind: str, offset: str, rows: List[tuple]) -> OptionSeries:
    return OptionSeries(
        kind=kind.upper(),
        offset=offset,
        timestamps=[int(r[0]) for r in rows],
        open=[float(r[1]) for r in rows],
        high=[float(r[2]) for r in rows],
        low=[float(r[3]) for r in rows],
        close=[float(r[4]) for r in rows],
        oi=[float(r[5]) for r in rows],
        spot=[float(r[6]) for r in rows],
        strike=[float(r[7]) for r in rows],
    )


def load_session_from_duckdb(
    store: NiftyDuckDBStore,
    session_date: str,
    *,
    interval_min: int = 5,
    offsets: Optional[List[str]] = None,
    expiry_code: int = 1,
) -> Dict[str, Dict[str, OptionSeries]]:
    offsets = offsets or list(sensex_atm_near_offsets())
    session: Dict[str, Dict[str, OptionSeries]] = {"CE": {}, "PE": {}}
    conn = store.connect()
    try:
        for kind in ("CE", "PE"):
            for offset in offsets:
                rows = conn.execute(
                    """
                    SELECT ts, open, high, low, close, oi, spot, strike
                    FROM nifty50_fno_bars
                    WHERE session_date = ? AND kind = ? AND strike_offset = ?
                      AND interval_min = ? AND expiry_code = ?
                    ORDER BY ts
                    """,
                    [session_date, kind, offset, interval_min, expiry_code],
                ).fetchall()
                if rows:
                    session[kind][offset] = _rows_to_series(kind, offset, rows)
    finally:
        conn.close()
    return session


def _session_has_data(session: Dict[str, Dict[str, OptionSeries]]) -> bool:
    for kind in ("CE", "PE"):
        atm = (session.get(kind) or {}).get("ATM")
        if atm and atm.timestamps:
            return True
    return False


def list_cached_session_dates(
    store: NiftyDuckDBStore,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[str]:
    conn = store.connect()
    try:
        clauses = ["interval_min = 5", "strike_offset = 'ATM'", "kind = 'CE'"]
        params: List[Any] = []
        if start_date:
            clauses.append("session_date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("session_date <= ?")
            params.append(end_date)
        where = " AND ".join(clauses)
        cur = conn.execute(
            f"SELECT DISTINCT session_date FROM nifty50_fno_bars WHERE {where} ORDER BY session_date",
            params,
        )
        return [str(r[0]) for r in cur.fetchall()]
    finally:
        conn.close()


def _resolve_sessions(params: BacktestParams) -> List[Dict[str, Any]]:
    if params.expiry_dates:
        return [{"expiry_date": d, "session_date": d} for d in sorted(set(params.expiry_dates))]

    end = date.today()
    if params.end_date:
        end = date.fromisoformat(params.end_date.strip())
    start = end - timedelta(days=30)
    if params.start_date:
        start = date.fromisoformat(params.start_date.strip())
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    days = iter_trading_days(start, end)
    return [{"expiry_date": d, "session_date": d} for d in days]


def fetch_nifty_sessions_data(
    sessions: List[Dict[str, Any]],
    *,
    interval_min: int = 5,
    refresh: bool = False,
    store: Optional[NiftyDuckDBStore] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, OptionSeries]]], Dict[str, int]]:
    store = store or _store()
    offsets = list(sensex_atm_near_offsets())
    loaded: Dict[str, Dict[str, Dict[str, OptionSeries]]] = {}
    stats = {"cached": 0, "fetched": 0}
    pending: List[str] = []

    for row in sessions:
        session_date = row["session_date"]
        if refresh:
            pending.append(session_date)
            continue
        session = load_session_from_duckdb(store, session_date, interval_min=interval_min, offsets=offsets)
        if _session_has_data(session):
            loaded[session_date] = session
            stats["cached"] += 1
        else:
            pending.append(session_date)

    if pending:
        client = DhanDataClient()
        prof = client.profile()
        if str(prof.get("dataPlan") or "").lower() != "active":
            raise RuntimeError("Dhan Data API is not active. Set DHAN_ACCESS_TOKEN in .env")
        interval_str = interval_to_str(interval_min)
        from services.nifty_duckdb_store import parse_rolling_option_leg

        conn = store.connect()
        try:
            store.ensure_schema(conn)
            for session_date in pending:
                series = client.fetch_nifty_session(
                    session_date,
                    offsets=offsets,
                    interval=interval_str,
                )
                loaded[session_date] = series
                stats["fetched"] += 1
                rows: List[Dict[str, Any]] = []
                for kind, leg_map in series.items():
                    for offset, opt in leg_map.items():
                        if not opt.timestamps:
                            continue
                        leg_dict = opt.to_dict()
                        rows.extend(
                            parse_rolling_option_leg(
                                leg_dict,
                                kind=kind,
                                strike_offset=offset,
                                expiry_code=1,
                                expiry_flag="WEEK",
                                interval_min=interval_min,
                            )
                        )
                if rows:
                    store.upsert_fno_bars(rows, conn=conn)
        finally:
            conn.close()

    return loaded, stats


def _run_backtest_for_timeframe(
    params: BacktestParams,
    sessions: List[Dict[str, Any]],
    interval_min: int,
    *,
    store: Optional[NiftyDuckDBStore] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    data, fetch_stats = fetch_nifty_sessions_data(
        sessions,
        interval_min=interval_min,
        refresh=params.refresh_dhan,
        store=store,
    )
    iv = f"{interval_min}m"
    start_capital = max(1000.0, float(params.capital))
    scan_start, cutoff, scan_start_label, scan_end_label = resolve_backtest_scan_window(params)
    trades: List[TradeResult] = []
    skipped: List[Dict[str, str]] = []
    equity = start_capital
    peak_equity = start_capital
    max_drawdown_inr = 0.0
    prev_spot_close = 0.0

    for row in sessions:
        session_date = row["session_date"]
        session = data.get(session_date)
        if not session:
            skipped.append({"expiry_date": session_date, "reason": f"no Dhan {iv} data"})
            continue

        index_open, _, prev_close = session_index_from_spot(session, prev_trading_close=prev_spot_close)
        if index_open <= 0:
            skipped.append({"expiry_date": session_date, "reason": "no Nifty spot in Dhan data"})
            continue

        tr_list = _run_day(session_date, index_open, prev_close, session, params)
        spot_s = session_spot_series(session)
        if spot_s and spot_s.spot:
            prev_spot_close = float(spot_s.spot[-1])
        if not tr_list:
            skipped.append(
                {
                    "expiry_date": session_date,
                    "reason": (
                        f"no {iv} close in ₹{params.entry_band_low:g}–₹{params.entry_band_high:g} "
                        f"from {scan_start_label} to {scan_end_label} IST"
                    ),
                }
            )
            continue

        for tr in tr_list:
            tr.symbol = _contract_key(tr.kind, tr.strike)
            lots, qty, risk_inr = _backtest_size_from_risk(
                equity, params.risk_pct, tr.entry, params.sl_inr, LOT_SIZE, MAX_LOTS
            )
            notional = tr.entry * qty
            if notional > equity:
                skipped.append(
                    {
                        "expiry_date": session_date,
                        "reason": f"{tr.symbol} insufficient capital (need ₹{notional:.0f}, have ₹{equity:.0f})",
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

    wins = [t for t in trades if t.pnl_inr > 0]
    total_pnl = sum(t.pnl_inr for t in trades)
    ending = round(start_capital + total_pnl, 2)
    report = {
        "summary": {
            "strategy": STRATEGY_NAME,
            "timeframe": iv,
            "interval_min": interval_min,
            "direction_mode": params.direction.upper(),
            "data_source": f"Dhan rollingoption {iv} (NSE_FNO securityId 13) + DuckDB",
            "duckdb_path": str((store or _store()).db_path),
            "starting_capital_inr": round(start_capital, 2),
            "ending_capital_inr": ending,
            "return_pct": round((ending - start_capital) / start_capital * 100.0, 2) if start_capital else 0.0,
            "max_drawdown_inr": round(max_drawdown_inr, 2),
            "sessions": len(sessions),
            "trades": len(trades),
            "skipped": len(skipped),
            "wins": len(wins),
            "losses": len([t for t in trades if t.pnl_inr < 0]),
            "win_rate_pct": round(100.0 * len(wins) / len(trades), 1) if trades else 0.0,
            "total_pnl_inr": round(total_pnl, 2),
            "avg_r": round(sum(t.r_multiple for t in trades) / len(trades), 2) if trades else 0.0,
            "lot_size": LOT_SIZE,
            "sl_inr": params.sl_inr,
            "entry_band": [params.entry_band_low, params.entry_band_high],
            "exit_model": f"SL {exit_model()}",
            "entry_model": "band + optional 2-candle confirmation",
        },
        "trades": [asdict(t) for t in trades],
        "skipped": skipped,
    }
    return report, fetch_stats


def run_nifty_dhan_backtest(params: BacktestParams) -> Dict[str, Any]:
    store = _store()
    sessions = _resolve_sessions(params)
    timeframes = _normalize_timeframes(params.timeframes_min)
    reports: Dict[str, Any] = {}
    total_fetch = {"cached": 0, "fetched": 0}

    for interval_min in timeframes:
        key = f"{interval_min}m"
        report, fetch_stats = _run_backtest_for_timeframe(params, sessions, interval_min, store=store)
        reports[key] = report
        total_fetch["cached"] += fetch_stats.get("cached", 0)
        total_fetch["fetched"] += fetch_stats.get("fetched", 0)

    fno = store.fno_summary()
    note = (
        f"Nifty50 Dhan backtest · capital ₹{params.capital:,.0f} · "
        f"exit {exit_model()} · entry ₹{params.entry_band_low:g}–₹{params.entry_band_high:g} · "
        f"{len(sessions)} sessions ({sessions[0]['session_date']} → {sessions[-1]['session_date']}). "
        f"DuckDB: {fno.get('trading_days', 0)} days cached · fetch {total_fetch['cached']} cached / "
        f"{total_fetch['fetched']} from API."
    )
    return {
        "note": note,
        "reports": reports,
        "dhan_status": check_dhan_status(),
        "duckdb": fno,
        "sessions": len(sessions),
    }
