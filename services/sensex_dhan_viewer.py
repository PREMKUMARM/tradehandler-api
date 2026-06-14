"""Browse cached Dhan 5m Sensex rolling-option OHLC by session date."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

from services.dhan_data_client import (
    DhanDataClient,
    OptionSeries,
    STRIKE_OFFSETS,
    load_cached_session,
    save_cached_session,
    ts_to_ist_label,
)
from services.sensex_constants import sensex_entry_scan_start_minutes
from services.sensex_dhan_backtest import CACHE_DIR as BACKTEST_CACHE_DIR, list_available_sessions

IST = ZoneInfo("Asia/Kolkata")


def _reference_bar_idx(series: OptionSeries) -> int:
    scan = sensex_entry_scan_start_minutes()
    for idx, ts in enumerate(series.timestamps):
        dt = datetime.fromtimestamp(ts, tz=IST)
        if dt.hour * 60 + dt.minute >= scan:
            return idx
    return max(0, len(series.timestamps) // 2)


def offset_moneyness_bucket(kind: str, offset: str) -> str:
    """ATM | OTM | ITM for CE/PE rolling offsets."""
    k = (kind or "").upper()
    off = (offset or "ATM").upper()
    if off == "ATM":
        return "ATM"
    if off.startswith("ATM+"):
        return "OTM" if k == "CE" else "ITM"
    if off.startswith("ATM-"):
        return "ITM" if k == "CE" else "OTM"
    return "ATM"


def offset_moneyness_label(kind: str, offset: str) -> str:
    bucket = offset_moneyness_bucket(kind, offset)
    off = (offset or "ATM").upper()
    if off == "ATM":
        return "ATM"
    step = off.replace("ATM+", "+").replace("ATM-", "-")
    return f"{bucket} {step}"


def _load_session(session_date: str, *, refresh: bool = False) -> Optional[Dict[str, Dict[str, OptionSeries]]]:
    cache_dir = BACKTEST_CACHE_DIR
    if not refresh:
        cached = load_cached_session(cache_dir, session_date)
        if cached:
            return cached
    client = DhanDataClient()
    prof = client.profile()
    if str(prof.get("dataPlan") or "").lower() != "active":
        raise RuntimeError("Dhan Data API is not active. Subscribe at dhan.co and set DHAN_ACCESS_TOKEN.")
    series = client.fetch_sensex_session(session_date)
    save_cached_session(cache_dir, session_date, series, meta={"profile_dataPlan": prof.get("dataPlan")})
    return series


def list_dhan_contracts(
    session_date: str,
    *,
    kind: Optional[str] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    session = _load_session(session_date, refresh=refresh)
    if not session:
        return {
            "session_date": session_date,
            "cached": False,
            "contracts": [],
            "count": 0,
        }

    kinds = [kind.upper()] if kind else ["CE", "PE"]
    contracts: List[Dict[str, Any]] = []
    for leg in kinds:
        offsets = session.get(leg) or {}
        for offset in STRIKE_OFFSETS:
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
                    "symbol": f"SENSEX-{strike}-{leg}",
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
        "session_date": session_date,
        "cached": True,
        "contracts": contracts,
        "count": len(contracts),
    }


def get_dhan_ohlc(
    session_date: str,
    *,
    kind: str,
    offset: str,
    refresh: bool = False,
) -> Dict[str, Any]:
    leg = (kind or "").upper()
    off = (offset or "ATM").upper()
    if leg not in ("CE", "PE"):
        raise ValueError("kind must be CE or PE")
    if off not in STRIKE_OFFSETS:
        raise ValueError(f"offset must be one of: {', '.join(STRIKE_OFFSETS)}")

    session = _load_session(session_date, refresh=refresh)
    if not session:
        raise ValueError(f"No Dhan data for {session_date}. Run backtest or refresh to fetch.")

    series = (session.get(leg) or {}).get(off)
    if not series or not series.timestamps:
        raise ValueError(f"No OHLC for {leg} {off} on {session_date}")

    ref_idx = _reference_bar_idx(series)
    strike = int(round(float(series.strike[ref_idx])))
    rows: List[Dict[str, Any]] = []
    for idx in range(len(series.timestamps)):
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
                "spot": round(float(series.spot[idx]), 2),
                "strike": int(round(float(series.strike[idx]))),
            }
        )

    return {
        "session_date": session_date,
        "kind": leg,
        "offset": off,
        "moneyness_bucket": offset_moneyness_bucket(leg, off),
        "moneyness_label": offset_moneyness_label(leg, off),
        "symbol": f"SENSEX-{strike}-{leg}",
        "strike": strike,
        "interval_min": 5,
        "data_source": "dhan_rollingoption_5m",
        "bars": len(rows),
        "rows": rows,
    }
