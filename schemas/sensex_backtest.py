"""Sensex Dhan backtest API schemas."""
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SensexBacktestRunRequest(BaseModel):
    expiry_dates: Optional[List[str]] = Field(
        default=None,
        description="Weekly expiry session dates (YYYY-MM-DD). Omit for all available.",
    )
    capital: float = Field(default=1_000_000.0, ge=10_000, le=100_000_000)
    risk_pct: float = Field(default=1.0, gt=0, le=10, description="Risk % of capital per trade")
    sl_inr: float = Field(default=10.0, gt=0, le=100, description="Stop-loss in premium points (₹)")
    entry_band_low: float = Field(default=17.0, gt=0, le=500, description="Entry premium range low (₹)")
    entry_band_high: float = Field(default=23.0, gt=0, le=500, description="Entry premium range high (₹)")
    min_target_low: float = Field(
        default=34.0,
        gt=0,
        le=500,
        description="Min target premium points from entry — trail activates at entry + this",
    )
    min_target_high: float = Field(
        default=34.0,
        gt=0,
        le=500,
        description="Max target premium points (optional cap before trail)",
    )
    direction: str = Field(default="AUTO", description="AUTO (highest OI strike), CE, or PE")
    mode: str = Field(default="conservative", description="conservative, optimistic, or both")
    refresh_dhan: bool = Field(default=False, description="Re-fetch from Dhan even if cached")

    @field_validator("direction")
    @classmethod
    def _direction(cls, value: str) -> str:
        v = (value or "AUTO").upper()
        if v not in ("AUTO", "CE", "PE"):
            raise ValueError("direction must be AUTO, CE, or PE")
        return v

    @field_validator("mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        v = (value or "conservative").lower()
        if v not in ("conservative", "optimistic", "both"):
            raise ValueError("mode must be conservative, optimistic, or both")
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
        low = info.data.get("min_target_low", 34.0)
        if value < low:
            raise ValueError("min_target_high must be >= min_target_low")
        return value
