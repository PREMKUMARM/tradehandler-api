"""Shared Sensex 20rupees run parameters (backtest, live, paper)."""
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SensexRunParamsIn(BaseModel):
    """Strategy run parameters — same as backtest except date range / timeframes."""

    capital: Optional[float] = Field(default=None, ge=10_000, le=100_000_000)
    risk_pct: Optional[float] = Field(
        default=None,
        gt=0,
        le=100,
        description="Capital allocation % per trade (lots = allocation ÷ entry premium)",
    )
    risk_percentage: Optional[float] = Field(default=None, gt=0, le=100)
    sl_inr: Optional[float] = Field(
        default=None,
        gt=0,
        le=100,
        description="Fixed initial stop-loss option premium (₹)",
    )
    entry_band_low: Optional[float] = Field(default=None, gt=0, le=500)
    entry_band_high: Optional[float] = Field(default=None, gt=0, le=500)
    min_target_low: Optional[float] = Field(default=None, gt=0, le=500)
    min_target_high: Optional[float] = Field(default=None, gt=0, le=500)
    entry_scan_start_ist: Optional[str] = Field(default=None, description="HH:MM IST")
    entry_scan_end_ist: Optional[str] = Field(default=None, description="HH:MM IST, exclusive")
    num_lots: Optional[int] = Field(default=None, ge=1, le=50, description="Max lots cap")

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

    def to_resolve_kwargs(self) -> dict:
        data = self.model_dump(exclude_none=True)
        if "risk_percentage" in data and "risk_pct" not in data:
            data["risk_pct"] = data.pop("risk_percentage")
        elif "risk_percentage" in data:
            data.pop("risk_percentage", None)
        return data
