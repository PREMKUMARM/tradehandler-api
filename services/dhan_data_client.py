"""
DhanHQ Data API client — rolling expired options (minute OHLC + OI + spot).

Uses DHAN_ACCESS_TOKEN from env; client-id is decoded from JWT when DHAN_CLIENT_ID is unset.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
API_BASE = "https://api.dhan.co/v2"
SENSEX_SECURITY_ID = "51"
NIFTY_SECURITY_ID = "13"
INTRADAY_MAX_DAYS = 90
DEFAULT_INTERVAL = "5"
DEFAULT_SLEEP_SEC = 1.25
SUPPORTED_INTERVALS_MIN: Tuple[int, ...] = (1, 5, 15, 25, 60)

STRIKE_OFFSETS: List[str] = ["ATM"] + [f"ATM+{i}" for i in range(1, 11)] + [f"ATM-{i}" for i in range(1, 11)]


@dataclass
class OptionSeries:
    kind: str
    offset: str
    timestamps: List[int]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    oi: List[float]
    spot: List[float]
    strike: List[float]

    @classmethod
    def from_api(cls, kind: str, offset: str, leg: Dict[str, Any]) -> "OptionSeries":
        def _arr(key: str) -> List[float]:
            raw = leg.get(key) or []
            return [float(x) for x in raw]

        return cls(
            kind=kind.upper(),
            offset=offset,
            timestamps=[int(x) for x in (leg.get("timestamp") or [])],
            open=_arr("open"),
            high=_arr("high"),
            low=_arr("low"),
            close=_arr("close"),
            oi=_arr("oi"),
            spot=_arr("spot"),
            strike=_arr("strike"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "offset": self.offset,
            "timestamps": self.timestamps,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "oi": self.oi,
            "spot": self.spot,
            "strike": self.strike,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptionSeries":
        return cls(
            kind=str(data.get("kind") or ""),
            offset=str(data.get("offset") or ""),
            timestamps=[int(x) for x in (data.get("timestamps") or [])],
            open=[float(x) for x in (data.get("open") or [])],
            high=[float(x) for x in (data.get("high") or [])],
            low=[float(x) for x in (data.get("low") or [])],
            close=[float(x) for x in (data.get("close") or [])],
            oi=[float(x) for x in (data.get("oi") or [])],
            spot=[float(x) for x in (data.get("spot") or [])],
            strike=[float(x) for x in (data.get("strike") or [])],
        )


class DhanDataClient:
    def __init__(
        self,
        access_token: Optional[str] = None,
        client_id: Optional[str] = None,
        sleep_sec: float = DEFAULT_SLEEP_SEC,
    ) -> None:
        self.access_token = (access_token or os.getenv("DHAN_ACCESS_TOKEN") or "").strip()
        if not self.access_token:
            raise ValueError("DHAN_ACCESS_TOKEN is required")
        self.client_id = (client_id or os.getenv("DHAN_CLIENT_ID") or self._client_id_from_jwt(self.access_token)).strip()
        self.sleep_sec = max(0.5, float(sleep_sec or DEFAULT_SLEEP_SEC))
        self._last_call = 0.0

    @staticmethod
    def _client_id_from_jwt(token: str) -> str:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        return str(data.get("dhanClientId") or "")

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": self.access_token,
            "client-id": self.client_id,
        }

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.sleep_sec:
            time.sleep(self.sleep_sec - elapsed)

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._throttle()
        req = urllib.request.Request(
            f"{API_BASE}{path}",
            method=method,
            headers=self._headers(),
        )
        if body is not None:
            req.data = json.dumps(body).encode()
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                self._last_call = time.time()
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            self._last_call = time.time()
            raw = exc.read().decode()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"errorMessage": raw[:400], "httpStatus": exc.code}
            data["_httpStatus"] = exc.code
            return data

    def profile(self) -> Dict[str, Any]:
        return self._request("GET", "/profile")

    def intraday_chart(
        self,
        *,
        security_id: str,
        exchange_segment: str = "BSE_FNO",
        instrument: str = "OPTIDX",
        interval: str = DEFAULT_INTERVAL,
        from_datetime: str,
        to_datetime: str,
        oi: bool = True,
        expiry_code: int = 0,
    ) -> Dict[str, Any]:
        """Active-contract intraday OHLC via POST /charts/intraday."""
        body = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": str(interval),
            "oi": bool(oi),
            "expiryCode": int(expiry_code),
            "fromDate": from_datetime,
            "toDate": to_datetime,
        }
        resp = self._request("POST", "/charts/intraday", body)
        if resp.get("errorCode") or resp.get("status") == "failed":
            msg = resp.get("errorMessage") or resp.get("remarks") or resp.get("data")
            raise RuntimeError(f"Dhan intraday failed ({security_id}): {msg}")
        return resp

    def nifty_index_intraday(
        self,
        *,
        from_date: str,
        to_date: str,
        interval: str = DEFAULT_INTERVAL,
    ) -> Dict[str, Any]:
        """Nifty 50 index intraday OHLC via POST /charts/intraday (IDX_I, max 90 days per call)."""
        return self.intraday_chart(
            security_id=NIFTY_SECURITY_ID,
            exchange_segment="IDX_I",
            instrument="INDEX",
            interval=interval,
            from_datetime=from_date,
            to_datetime=to_date,
            oi=False,
            expiry_code=0,
        )

    def rolling_option_raw(
        self,
        *,
        session_date: str,
        kind: str,
        offset: str = "ATM",
        security_id: str = SENSEX_SECURITY_ID,
        exchange_segment: str = "BSE_FNO",
        interval: str = DEFAULT_INTERVAL,
        expiry_code: int = 1,
        expiry_flag: str = "WEEK",
    ) -> Dict[str, Any]:
        """Raw PE/CE leg dict from POST /charts/rollingoption (expired contracts)."""
        next_day = (date.fromisoformat(session_date) + timedelta(days=1)).isoformat()
        return self.rolling_option_range(
            from_date=session_date,
            to_date=next_day,
            kind=kind,
            offset=offset,
            security_id=security_id,
            exchange_segment=exchange_segment,
            interval=interval,
            expiry_code=expiry_code,
            expiry_flag=expiry_flag,
        )

    def rolling_option_range(
        self,
        *,
        from_date: str,
        to_date: str,
        kind: str,
        offset: str = "ATM",
        security_id: str = SENSEX_SECURITY_ID,
        exchange_segment: str = "BSE_FNO",
        interval: str = DEFAULT_INTERVAL,
        expiry_code: int = 1,
        expiry_flag: str = "WEEK",
    ) -> Dict[str, Any]:
        """Raw PE/CE leg dict from POST /charts/rollingoption for a date range (up to ~30 days)."""
        body = {
            "exchangeSegment": exchange_segment,
            "interval": str(interval),
            "securityId": str(security_id),
            "instrument": "OPTIDX",
            "expiryFlag": expiry_flag,
            "expiryCode": int(expiry_code),
            "strike": offset,
            "drvOptionType": "CALL" if kind.upper() == "CE" else "PUT",
            "requiredData": ["open", "high", "low", "close", "oi", "spot", "strike", "volume"],
            "fromDate": from_date,
            "toDate": to_date,
        }
        resp = self._request("POST", "/charts/rollingoption", body)
        if resp.get("errorCode") or resp.get("status") == "failed":
            msg = resp.get("errorMessage") or resp.get("remarks") or resp.get("data")
            raise RuntimeError(
                f"Dhan rollingoption failed ({exchange_segment} {offset} {kind} {from_date}->{to_date}): {msg}"
            )
        key = "ce" if kind.upper() == "CE" else "pe"
        return resp.get("data", {}).get(key) or {}

    def nifty_rolling_option_range(
        self,
        *,
        from_date: str,
        to_date: str,
        kind: str,
        offset: str = "ATM",
        interval: str = DEFAULT_INTERVAL,
        expiry_code: int = 1,
        expiry_flag: str = "WEEK",
    ) -> Dict[str, Any]:
        """Nifty 50 rolling expired options OHLC via NSE_FNO."""
        return self.rolling_option_range(
            from_date=from_date,
            to_date=to_date,
            kind=kind,
            offset=offset,
            security_id=NIFTY_SECURITY_ID,
            exchange_segment="NSE_FNO",
            interval=interval,
            expiry_code=expiry_code,
            expiry_flag=expiry_flag,
        )

    def fetch_expired_fixed_strike_bars(
        self,
        *,
        session_date: str,
        strike: int,
        kind: str = "PE",
        interval: str = DEFAULT_INTERVAL,
        expiry_code: int = 1,
        max_offset: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct fixed-strike expired option 5m bars from Dhan rollingoption.

        Dhan stores expired options by ATM offset, not absolute strike. We scan
        ATM±max_offset and keep the bar whenever rolling strike equals `strike`.
        """
        offsets = ["ATM"] + [f"ATM+{i}" for i in range(1, max_offset + 1)]
        offsets += [f"ATM-{i}" for i in range(1, max_offset + 1)]
        merged: Dict[int, Dict[str, Any]] = {}
        target = int(strike)
        session_day = date.fromisoformat(session_date)

        for offset in offsets:
            leg = self.rolling_option_raw(
                session_date=session_date,
                kind=kind,
                offset=offset,
                interval=interval,
                expiry_code=expiry_code,
            )
            timestamps = leg.get("timestamp") or []
            volumes = leg.get("volume") or [0] * len(timestamps)
            for idx, ts in enumerate(timestamps):
                bar_strike = int(round(float((leg.get("strike") or [0])[idx])))
                if bar_strike != target:
                    continue
                dt = datetime.fromtimestamp(int(ts), tz=IST)
                if dt.date() != session_day:
                    continue
                if int(ts) in merged:
                    continue
                merged[int(ts)] = {
                    "datetime_ist": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp": int(ts),
                    "strike": target,
                    "open": round(float(leg["open"][idx]), 2),
                    "high": round(float(leg["high"][idx]), 2),
                    "low": round(float(leg["low"][idx]), 2),
                    "close": round(float(leg["close"][idx]), 2),
                    "volume": int(volumes[idx] if idx < len(volumes) else 0),
                    "oi": round(float(leg["oi"][idx]), 2),
                    "spot": round(float(leg["spot"][idx]), 2),
                    "rolling_offset": offset,
                }
        return [merged[ts] for ts in sorted(merged)]

    def rolling_option(
        self,
        *,
        session_date: str,
        kind: str,
        offset: str = "ATM",
        security_id: str = SENSEX_SECURITY_ID,
        interval: str = DEFAULT_INTERVAL,
    ) -> OptionSeries:
        next_day = (date.fromisoformat(session_date) + timedelta(days=1)).isoformat()
        body = {
            "exchangeSegment": "BSE_FNO",
            "interval": str(interval),
            "securityId": str(security_id),
            "instrument": "OPTIDX",
            "expiryFlag": "WEEK",
            "expiryCode": 1,
            "strike": offset,
            "drvOptionType": "CALL" if kind.upper() == "CE" else "PUT",
            "requiredData": ["open", "high", "low", "close", "oi", "spot", "strike", "volume"],
            "fromDate": session_date,
            "toDate": next_day,
        }
        resp = self._request("POST", "/charts/rollingoption", body)
        if resp.get("errorCode") or resp.get("status") == "failed":
            msg = resp.get("errorMessage") or resp.get("remarks") or resp.get("data")
            raise RuntimeError(f"Dhan rollingoption failed ({offset} {kind} {session_date}): {msg}")
        data = resp.get("data") or {}
        key = "ce" if kind.upper() == "CE" else "pe"
        leg = data.get(key) or {}
        return OptionSeries.from_api(kind, offset, leg)

    def fetch_sensex_session(
        self,
        session_date: str,
        *,
        kinds: Optional[List[str]] = None,
        offsets: Optional[List[str]] = None,
        interval: str = DEFAULT_INTERVAL,
    ) -> Dict[str, Dict[str, OptionSeries]]:
        kinds = kinds or ["CE", "PE"]
        offsets = offsets or STRIKE_OFFSETS
        out: Dict[str, Dict[str, OptionSeries]] = {}
        for kind in kinds:
            out[kind] = {}
            for offset in offsets:
                out[kind][offset] = self.rolling_option(
                    session_date=session_date,
                    kind=kind,
                    offset=offset,
                    interval=str(interval),
                )
        return out


def interval_to_str(interval_min: int) -> str:
    iv = int(interval_min)
    if iv not in SUPPORTED_INTERVALS_MIN:
        raise ValueError(f"Unsupported interval {iv}m — use {list(SUPPORTED_INTERVALS_MIN)}")
    return str(iv)


def cache_path(cache_dir: Path, session_date: str, interval_min: int = 5) -> Path:
    iv = int(interval_min)
    return cache_dir / f"{session_date.replace('-', '')}_sensex_{iv}m.json"


def _legacy_cache_path(cache_dir: Path, session_date: str) -> Path:
    return cache_dir / f"{session_date.replace('-', '')}_sensex_5m.json"


def load_cached_session(
    cache_dir: Path,
    session_date: str,
    interval_min: int = 5,
) -> Optional[Dict[str, Dict[str, OptionSeries]]]:
    path = cache_path(cache_dir, session_date, interval_min)
    if not path.exists() and int(interval_min) == 5:
        path = _legacy_cache_path(cache_dir, session_date)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, OptionSeries]] = {}
    for kind, offsets in (raw.get("series") or {}).items():
        out[kind] = {offset: OptionSeries.from_dict(payload) for offset, payload in offsets.items()}
    return out


def save_cached_session(
    cache_dir: Path,
    session_date: str,
    series: Dict[str, Dict[str, OptionSeries]],
    *,
    meta: Optional[Dict[str, Any]] = None,
    interval_min: int = 5,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, session_date, interval_min)
    payload = {
        "session_date": session_date,
        "interval_min": int(interval_min),
        "fetched_at": datetime.now(IST).isoformat(),
        "meta": meta or {},
        "series": {
            kind: {offset: s.to_dict() for offset, s in offsets.items()}
            for kind, offsets in series.items()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _session_day_indices(
    series: OptionSeries,
    *,
    session_day: Optional[date],
    start_idx: int = 0,
) -> List[int]:
    """Bar indices on `series` belonging to `session_day` (or all bars if unset)."""
    if session_day is None:
        return list(range(start_idx, len(series.timestamps)))
    out: List[int] = []
    for idx in range(start_idx, len(series.timestamps)):
        dt = datetime.fromtimestamp(int(series.timestamps[idx]), tz=IST)
        if dt.date() == session_day:
            out.append(idx)
    return out


def build_entry_offset_exit_series(
    session: Dict[str, Dict[str, OptionSeries]],
    *,
    kind: str,
    entry_offset: str,
    entry_ts: int,
    entry_strike: int,
    session_date: Optional[str] = None,
) -> Optional[Tuple[OptionSeries, int]]:
    """
    Exit OHLC on the same rolling-offset slice used for entry, from the entry bar onward.

    Dhan stores one series per ATM offset; merging all offsets that share an absolute
    strike reuses stale labels from other offsets and inflates trail exits. Exits stay
    on the entry offset path instead.
    """
    kind_key = kind.upper()
    series = (session.get(kind_key) or {}).get(entry_offset)
    if not series or not series.timestamps:
        return None

    session_day = date.fromisoformat(session_date) if session_date else None
    entry_idx: Optional[int] = None
    for idx, ts in enumerate(series.timestamps):
        if int(ts) != int(entry_ts):
            continue
        if session_day is not None:
            dt = datetime.fromtimestamp(int(ts), tz=IST)
            if dt.date() != session_day:
                continue
        entry_idx = idx
        break
    if entry_idx is None:
        return None

    idxs = _session_day_indices(series, session_day=session_day, start_idx=entry_idx)
    if not idxs:
        return None

    target = int(entry_strike)
    return (
        OptionSeries(
            kind=kind_key,
            offset=f"{entry_offset}@{target}",
            timestamps=[series.timestamps[i] for i in idxs],
            open=[series.open[i] for i in idxs],
            high=[series.high[i] for i in idxs],
            low=[series.low[i] for i in idxs],
            close=[series.close[i] for i in idxs],
            oi=[series.oi[i] for i in idxs],
            spot=[series.spot[i] for i in idxs],
            strike=[series.strike[i] for i in idxs],
        ),
        0,
    )


def build_fixed_strike_series(
    session: Dict[str, Dict[str, OptionSeries]],
    *,
    kind: str,
    strike: int,
    session_date: Optional[str] = None,
    entry_offset: Optional[str] = None,
) -> Optional[OptionSeries]:
    """
    Merge cached rolling-offset series into one OHLC stream for an absolute strike.

    When `entry_offset` is set, only that offset slice is used so cross-offset
    strike reuse cannot contaminate exit bars.
    """
    kind_key = kind.upper()
    leg = session.get(kind_key) or {}
    if not leg:
        return None

    target = int(strike)
    session_day = date.fromisoformat(session_date) if session_date else None
    merged: Dict[int, Dict[str, float]] = {}
    offsets = [entry_offset] if entry_offset else STRIKE_OFFSETS

    for offset in offsets:
        if not offset:
            continue
        series = leg.get(offset)
        if not series or not series.timestamps:
            continue
        for idx, ts in enumerate(series.timestamps):
            bar_strike = int(round(float(series.strike[idx])))
            if bar_strike != target:
                continue
            if session_day is not None:
                dt = datetime.fromtimestamp(int(ts), tz=IST)
                if dt.date() != session_day:
                    continue
            ts_key = int(ts)
            if ts_key in merged:
                continue
            merged[ts_key] = {
                "open": float(series.open[idx]),
                "high": float(series.high[idx]),
                "low": float(series.low[idx]),
                "close": float(series.close[idx]),
                "oi": float(series.oi[idx]),
                "spot": float(series.spot[idx]),
            }

    if not merged:
        return None

    lock_label = f"{entry_offset}@{target}" if entry_offset else f"LOCK-{target}"
    timestamps = sorted(merged.keys())
    return OptionSeries(
        kind=kind_key,
        offset=lock_label,
        timestamps=timestamps,
        open=[merged[ts]["open"] for ts in timestamps],
        high=[merged[ts]["high"] for ts in timestamps],
        low=[merged[ts]["low"] for ts in timestamps],
        close=[merged[ts]["close"] for ts in timestamps],
        oi=[merged[ts]["oi"] for ts in timestamps],
        spot=[merged[ts]["spot"] for ts in timestamps],
        strike=[float(target)] * len(timestamps),
    )


def ts_to_ist_label(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d %H:%M:%S")
