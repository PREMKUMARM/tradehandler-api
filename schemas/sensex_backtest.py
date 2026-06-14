"""Sensex Dhan backtest API schemas."""
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SensexBacktestRunRequest(BaseModel):
    start_date: Optional[str] = Field(
        default=None,
        description="Inclusive backtest start date YYYY-MM-DD (weekdays only)",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Inclusive backtest end date YYYY-MM-DD (weekdays only)",
    )
    expiry_dates: Optional[List[str]] = Field(
        default=None,
        description="Explicit session dates (legacy). Omit when using start_date/end_date.",
    )
    capital: float = Field(default=1_000_000.0, ge=10_000, le=100_000_000)
    risk_pct: float = Field(
        default=1.0,
        gt=0,
        le=100,
        description="Capital allocation % per trade (lots = allocation ÷ entry premium)",
    )
    sl_inr: float = Field(
        default=9.0,
        gt=0,
        le=100,
        description="Fixed initial stop-loss option premium price (₹), e.g. 9 — not distance from entry",
    )
    entry_band_low: float = Field(default=17.0, gt=0, le=500, description="Entry premium range low (₹)")
    entry_band_high: float = Field(default=23.0, gt=0, le=500, description="Entry premium range high (₹)")
    min_target_low: float = Field(
        default=10.0,
        gt=0,
        le=500,
        description="Min target premium points from entry — trail activates at entry + this (default 10 = 1R)",
    )
    min_target_high: float = Field(
        default=10.0,
        gt=0,
        le=500,
        description="Max target premium points (optional cap before trail)",
    )
    direction: str = Field(
        default="AUTO",
        description="AUTO (PE gap-down, CE gap-up/flat), CE, or PE",
    )
    refresh_dhan: bool = Field(default=False, description="Re-fetch from Dhan even if cached")
    timeframes_min: Optional[List[int]] = Field(
        default=None,
        description="Bar intervals in minutes (1, 5, 15, 25, 60). Default [5]. Runs one backtest per selection.",
    )
    entry_scan_start_ist: Optional[str] = Field(
        default=None,
        description="First bar to scan for entry, IST HH:MM (default 14:00)",
    )
    entry_scan_end_ist: Optional[str] = Field(
        default=None,
        description="Last minute to allow new entries (exclusive), IST HH:MM (default 15:00)",
    )

    @field_validator("entry_scan_start_ist", "entry_scan_end_ist")
    @classmethod
    def _scan_time(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        raw = str(value).strip()
        parts = raw.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid time {raw!r} — use HH:MM")
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"Invalid time {raw!r}")
        return f"{hour:02d}:{minute:02d}"

    @field_validator("entry_scan_end_ist")
    @classmethod
    def _scan_window(cls, value: Optional[str], info) -> Optional[str]:
        if value is None:
            return None
        start_raw = info.data.get("entry_scan_start_ist")
        if not start_raw:
            return value
        start_parts = str(start_raw).split(":")
        end_parts = str(value).split(":")
        start_min = int(start_parts[0]) * 60 + int(start_parts[1])
        end_min = int(end_parts[0]) * 60 + int(end_parts[1])
        if end_min <= start_min:
            raise ValueError("entry_scan_end_ist must be after entry_scan_start_ist")
        return value

    @field_validator("timeframes_min")
    @classmethod
    def _timeframes(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return None
        allowed = {1, 5, 15, 25, 60}
        out: List[int] = []
        for iv in value:
            n = int(iv)
            if n not in allowed:
                raise ValueError(f"timeframes_min must be subset of {sorted(allowed)}")
            if n not in out:
                out.append(n)
        if not out:
            raise ValueError("Select at least one timeframe")
        return sorted(out)

    @field_validator("direction")
    @classmethod
    def _direction(cls, value: str) -> str:
        v = (value or "AUTO").upper()
        if v not in ("AUTO", "CE", "PE"):
            raise ValueError("direction must be AUTO, CE, or PE")
        return v

    @field_validator("entry_band_high")
    @classmethod
    def _entry_band(cls, value: float, info) -> float:
        low = info.data.get("entry_band_low", 17.0)
        if value <= low:
            raise ValueError("entry_band_high must be greater than entry_band_low")
        return value

    @field_validator("min_target_high")
    @classmethod
    def _min_target_band(cls, value: float, info) -> float:
        low = info.data.get("min_target_low", 10.0)
        if value < low:
            raise ValueError("min_target_high must be >= min_target_low")
        return value
