"""Unified Dhan rolling-option viewer — Sensex cache + Nifty DuckDB."""
from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

from services.dhan_data_client import (
    DhanDataClient,
    OptionSeries,
    load_cached_session,
    save_cached_session,
    ts_to_ist_label,
)
from services.entry_quality import entry_band_limits, exit_model
from services.sensex_dhan_backtest import DEFAULT_SL_INR
from services.nifty_dhan_backtest import _store as nifty_store
from services.nifty_dhan_backtest import list_cached_session_dates, load_session_from_duckdb
from services.sensex_constants import normalize_rolling_offset, sensex_atm_near_offsets, sensex_entry_scan_start_minutes
from services.sensex_dhan_backtest import CACHE_DIR as SENSEX_CACHE_DIR, list_available_sessions
from services.sensex_dhan_viewer import (
    offset_moneyness_bucket,
    offset_moneyness_label,
)

IST = ZoneInfo("Asia/Kolkata")
API_ROOT = Path(__file__).resolve().parent.parent


def _reference_bar_idx(series: OptionSeries) -> int:
    scan = sensex_entry_scan_start_minutes()
    for idx, ts in enumerate(series.timestamps):
        dt = datetime.fromtimestamp(ts, tz=IST)
        if dt.hour * 60 + dt.minute >= scan:
            return idx
    return max(0, len(series.timestamps) // 2)


def _contract_symbol(segment: str, strike: int, kind: str) -> str:
    seg = (segment or "sensex").lower()
    if seg in ("nifty50", "nifty"):
        return f"NIFTY-{int(strike)}-{kind.upper()}"
    return f"SENSEX-{int(strike)}-{kind.upper()}"


def list_dhan_sessions(segment: str = "sensex") -> List[Dict[str, Any]]:
    seg = (segment or "sensex").lower()
    if seg in ("nifty50", "nifty"):
        store = nifty_store()
        dates = list_cached_session_dates(store)
        return [
            {
                "session_date": d,
                "expiry_date": d,
                "dhan_cached": True,
                "segment": "nifty50",
            }
            for d in dates
        ]
    return [{**row, "session_date": row.get("expiry_date"), "segment": "sensex"} for row in list_available_sessions()]


def _load_sensex_session(session_date: str, *, refresh: bool = False) -> Optional[Dict[str, Dict[str, OptionSeries]]]:
    if not refresh:
        cached = load_cached_session(SENSEX_CACHE_DIR, session_date)
        if cached:
            return cached
    client = DhanDataClient()
    prof = client.profile()
    if str(prof.get("dataPlan") or "").lower() != "active":
        raise RuntimeError("Dhan Data API is not active. Subscribe at dhan.co and set DHAN_ACCESS_TOKEN.")
    series = client.fetch_sensex_session(session_date)
    save_cached_session(SENSEX_CACHE_DIR, session_date, series, meta={"profile_dataPlan": prof.get("dataPlan")})
    return series


def _load_nifty_session(session_date: str) -> Optional[Dict[str, Dict[str, OptionSeries]]]:
    store = nifty_store()
    session = load_session_from_duckdb(store, session_date, interval_min=5)
    if not session.get("CE") and not session.get("PE"):
        return None
    return session


def load_dhan_session(segment: str, session_date: str, *, refresh: bool = False) -> Optional[Dict[str, Dict[str, OptionSeries]]]:
    seg = (segment or "sensex").lower()
    if seg in ("nifty50", "nifty"):
        return _load_nifty_session(session_date)
    return _load_sensex_session(session_date, refresh=refresh)


def list_dhan_contracts(
    segment: str,
    session_date: str,
    *,
    kind: Optional[str] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    session = load_dhan_session(segment, session_date, refresh=refresh)
    if not session:
        return {"segment": segment, "session_date": session_date, "cached": False, "contracts": [], "count": 0}

    kinds = [kind.upper()] if kind else ["CE", "PE"]
    contracts: List[Dict[str, Any]] = []
    for leg in kinds:
        offsets = session.get(leg) or {}
        for offset in sensex_atm_near_offsets():
            series = offsets.get(offset)
            if not series or not series.timestamps:
                continue
            ref_idx = _reference_bar_idx(series)
            strike = int(round(float(series.strike[ref_idx])))
            bucket = offset_moneyness_bucket(leg, offset)
            contracts.append(
                {
                    "kind": leg,
                    "offset": offset,
                    "moneyness_bucket": bucket,
                    "moneyness_label": offset_moneyness_label(leg, offset),
                    "strike": strike,
                    "symbol": _contract_symbol(segment, strike, leg),
                    "bars": len(series.timestamps),
                    "ref_close": round(float(series.close[ref_idx]), 2),
                    "ref_spot": round(float(series.spot[ref_idx]), 2),
                }
            )

    order = {"ATM": 0, "OTM": 1, "ITM": 2}

    def _sort_key(row: Dict[str, Any]) -> tuple:
        off = str(row.get("offset") or "ATM")
        step = 0
        if off.startswith("ATM+"):
            step = int(off[4:] or 0)
        elif off.startswith("ATM-"):
            step = -int(off[4:] or 0)
        return (row.get("kind") or "", order.get(row.get("moneyness_bucket") or "ATM", 9), step)

    contracts.sort(key=_sort_key)
    return {
        "segment": segment,
        "session_date": session_date,
        "cached": True,
        "contracts": contracts,
        "count": len(contracts),
    }


def _parse_hm(value: str) -> Optional[int]:
    if not value:
        return None
    raw = str(value).strip()
    if " " in raw:
        raw = raw.split(" ", 1)[1]
    parts = raw.split(":")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except (TypeError, ValueError):
        return None


def _find_bar_index(rows: List[Dict[str, Any]], hm: Optional[int]) -> Optional[int]:
    if hm is None:
        return None
    for row in rows:
        bar_hm = _parse_hm(str(row.get("datetime_ist") or ""))
        if bar_hm is not None and bar_hm >= hm:
            return int(row.get("bar_index", 0))
    return None


def get_dhan_ohlc(
    segment: str,
    session_date: str,
    *,
    kind: str,
    offset: str,
    refresh: bool = False,
    trade: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    leg = (kind or "").upper()
    off = normalize_rolling_offset(offset)
    if leg not in ("CE", "PE"):
        raise ValueError("kind must be CE or PE")
    if off not in sensex_atm_near_offsets():
        raise ValueError(f"offset must be one of: {', '.join(sensex_atm_near_offsets())}")

    session = load_dhan_session(segment, session_date, refresh=refresh)
    if not session:
        raise ValueError(f"No Dhan data for {session_date}. Run backtest or refresh to fetch.")

    series = (session.get(leg) or {}).get(off)
    if not series or not series.timestamps:
        raise ValueError(f"No OHLC for {leg} {off} on {session_date}")

    session_day = date.fromisoformat(session_date)
    ref_idx = _reference_bar_idx(series)
    strike = int(round(float(series.strike[ref_idx])))
    rows: List[Dict[str, Any]] = []
    index_open = 0.0
    for idx in range(len(series.timestamps)):
        dt = datetime.fromtimestamp(series.timestamps[idx], tz=IST)
        if dt.date() != session_day:
            continue
        if index_open <= 0 and series.spot:
            index_open = float(series.spot[0])
        spot = round(float(series.spot[idx]), 2) if series.spot else 0.0
        rows.append(
            {
                "bar_index": idx,
                "datetime_ist": ts_to_ist_label(series.timestamps[idx]),
                "timestamp": series.timestamps[idx],
                "open": round(float(series.open[idx]), 2),
                "high": round(float(series.high[idx]), 2),
                "low": round(float(series.low[idx]), 2),
                "close": round(float(series.close[idx]), 2),
                "oi": round(float(series.oi[idx]), 0),
                "spot": spot,
                "strike": int(round(float(series.strike[idx]))),
            }
        )

    band_lo, band_hi = entry_band_limits()
    try:
        sl_inr = float(os.getenv("SL_INR", str(DEFAULT_SL_INR)) or DEFAULT_SL_INR)
    except (TypeError, ValueError):
        sl_inr = DEFAULT_SL_INR
    scan_start = sensex_entry_scan_start_minutes()
    scan_end_h, scan_end_m = 14, 45
    payload: Dict[str, Any] = {
        "segment": segment,
        "session_date": session_date,
        "kind": leg,
        "offset": off,
        "moneyness_bucket": offset_moneyness_bucket(leg, off),
        "moneyness_label": offset_moneyness_label(leg, off),
        "symbol": _contract_symbol(segment, strike, leg),
        "strike": strike,
        "interval_min": 5,
        "data_source": "dhan_rollingoption_5m",
        "bars": len(rows),
        "rows": rows,
        "index_open": round(index_open, 2) if index_open > 0 else None,
        "entry_band": [band_lo, band_hi],
        "scan_window": {
            "start": f"{scan_start // 60:02d}:{scan_start % 60:02d}",
            "end": f"{scan_end_h:02d}:{scan_end_m:02d}",
        },
        "strategy": {
            "sl_inr": sl_inr,
            "exit_model": exit_model(),
            "entry_band": [band_lo, band_hi],
            "scan_window": {
                "start": f"{scan_start // 60:02d}:{scan_start % 60:02d}",
                "end": f"{scan_end_h:02d}:{scan_end_m:02d}",
            },
        },
    }

    if trade:
        entry_hm = _parse_hm(str(trade.get("entry_datetime_ist") or trade.get("entry_time") or ""))
        exit_hm = _parse_hm(str(trade.get("exit_datetime_ist") or trade.get("exit_time") or ""))
        entry_idx = _find_bar_index(rows, entry_hm)
        exit_idx = _find_bar_index(rows, exit_hm)
        payload["trade"] = {
            "entry": float(trade.get("entry") or 0),
            "exit": float(trade.get("exit") or 0),
            "sl": float(trade.get("sl") or 9),
            "target": float(trade.get("target") or 0),
            "entry_datetime_ist": trade.get("entry_datetime_ist") or "",
            "exit_datetime_ist": trade.get("exit_datetime_ist") or "",
            "exit_reason": trade.get("exit_reason") or "",
            "pnl_inr": trade.get("pnl_inr"),
            "entry_bar_index": entry_idx,
            "exit_bar_index": exit_idx,
        }
    return payload


def load_cached_backtest_bundle() -> Dict[str, Any]:
    """Load latest on-disk backtest JSON + combined trade rows."""
    nifty_path = API_ROOT / "data" / "market" / "backtest_nifty50_dhan_results.json"
    sensex_path = API_ROOT / "data" / "sensex" / "backtest_20rupees_dhan_results.json"
    csv_path = API_ROOT / "data" / "market" / "backtest_nifty50_sensex_trades.csv"
    tuned_path = API_ROOT / "data" / "market" / "tuned_20rupees_config.json"

    out: Dict[str, Any] = {"nifty": None, "sensex": None, "combined_trades": [], "tuned_config": None}
    if nifty_path.exists():
        out["nifty"] = json.loads(nifty_path.read_text(encoding="utf-8"))
    if sensex_path.exists():
        out["sensex"] = json.loads(sensex_path.read_text(encoding="utf-8"))
    if tuned_path.exists():
        out["tuned_config"] = json.loads(tuned_path.read_text(encoding="utf-8"))
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            out["combined_trades"] = list(csv.DictReader(fh))
    return out
