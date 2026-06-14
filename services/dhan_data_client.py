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
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
API_BASE = "https://api.dhan.co/v2"
SENSEX_SECURITY_ID = "51"
DEFAULT_INTERVAL = "5"
DEFAULT_SLEEP_SEC = 1.25

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
    ) -> Dict[str, Dict[str, OptionSeries]]:
        kinds = kinds or ["CE", "PE"]
        offsets = offsets or STRIKE_OFFSETS
        out: Dict[str, Dict[str, OptionSeries]] = {}
        for kind in kinds:
            out[kind] = {}
            for offset in offsets:
                out[kind][offset] = self.rolling_option(session_date=session_date, kind=kind, offset=offset)
        return out


def cache_path(cache_dir: Path, session_date: str) -> Path:
    return cache_dir / f"{session_date.replace('-', '')}_sensex_5m.json"


def load_cached_session(cache_dir: Path, session_date: str) -> Optional[Dict[str, Dict[str, OptionSeries]]]:
    path = cache_path(cache_dir, session_date)
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
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, session_date)
    payload = {
        "session_date": session_date,
        "fetched_at": datetime.now(IST).isoformat(),
        "meta": meta or {},
        "series": {
            kind: {offset: s.to_dict() for offset, s in offsets.items()}
            for kind, offsets in series.items()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def ts_to_ist_label(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d %H:%M:%S")
