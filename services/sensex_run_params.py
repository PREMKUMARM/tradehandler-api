"""Resolve Sensex 20rupees run parameters for live, paper, and backtest."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from services.sensex_constants import (
    sensex_entry_cutoff_minutes,
    sensex_entry_scan_start_minutes,
    sensex_max_lots_per_trade,
)


def _ist_minutes(label: str) -> int:
    parts = str(label or "14:00").strip().split(":")
    hour = max(0, min(23, int(parts[0])))
    minute = max(0, min(59, int(parts[1] if len(parts) > 1 else 0)))
    return hour * 60 + minute


def _default_scan_start_ist() -> str:
    mins = sensex_entry_scan_start_minutes()
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _default_scan_end_ist() -> str:
    mins = sensex_entry_cutoff_minutes()
    return f"{mins // 60:02d}:{mins % 60:02d}"


@dataclass
class SensexRunParams:
    capital: float = 1_000_000.0
    risk_pct: float = 1.0
    sl_inr: float = 9.0
    entry_band_low: float = 17.0
    entry_band_high: float = 23.0
    min_target_low: float = 10.0
    min_target_high: float = 10.0
    direction: str = "AUTO"
    entry_scan_start_ist: str = "14:00"
    entry_scan_end_ist: str = "14:45"
    num_lots: int = 50

    @classmethod
    def defaults(cls) -> SensexRunParams:
        return cls(
            entry_scan_start_ist=_default_scan_start_ist(),
            entry_scan_end_ist=_default_scan_end_ist(),
            num_lots=sensex_max_lots_per_trade(),
        )

    @classmethod
    def from_mapping(
        cls,
        data: Optional[Mapping[str, Any]] = None,
        *,
        direction: Optional[str] = None,
        **kwargs: Any,
    ) -> SensexRunParams:
        base = cls.defaults()
        merged: Dict[str, Any] = asdict(base)
        if data:
            merged.update({k: v for k, v in dict(data).items() if v is not None})
        merged.update({k: v for k, v in kwargs.items() if v is not None})
        if merged.get("risk_percentage") is not None and merged.get("risk_pct") is None:
            merged["risk_pct"] = merged.pop("risk_percentage")
        else:
            merged.pop("risk_percentage", None)
        if direction:
            merged["direction"] = str(direction).upper()
        merged["direction"] = str(merged.get("direction") or "AUTO").upper()
        if merged["direction"] not in ("AUTO", "CE", "PE"):
            merged["direction"] = "AUTO"
        merged["num_lots"] = max(1, min(50, int(merged.get("num_lots") or base.num_lots)))
        merged["capital"] = float(merged.get("capital") or base.capital)
        merged["risk_pct"] = float(merged.get("risk_pct") or base.risk_pct)
        merged["sl_inr"] = float(merged.get("sl_inr") or base.sl_inr)
        merged["entry_band_low"] = float(merged.get("entry_band_low") or base.entry_band_low)
        merged["entry_band_high"] = float(merged.get("entry_band_high") or base.entry_band_high)
        merged["min_target_low"] = float(merged.get("min_target_low") or base.min_target_low)
        merged["min_target_high"] = float(merged.get("min_target_high") or base.min_target_high)
        merged["entry_scan_start_ist"] = str(
            merged.get("entry_scan_start_ist") or base.entry_scan_start_ist
        )
        merged["entry_scan_end_ist"] = str(
            merged.get("entry_scan_end_ist") or base.entry_scan_end_ist
        )
        return cls(**{k: merged[k] for k in asdict(base).keys()})

    def scan_start_minutes(self) -> int:
        return _ist_minutes(self.entry_scan_start_ist)

    def scan_end_minutes(self) -> int:
        return _ist_minutes(self.entry_scan_end_ist)

    def is_past_entry_cutoff(self, bar_minutes: int) -> bool:
        if bar_minutes <= 0:
            from services.sensex_constants import is_past_sensex_entry_cutoff

            return is_past_sensex_entry_cutoff()
        return bar_minutes >= self.scan_end_minutes()

    def entry_cutoff_label(self) -> str:
        return self.entry_scan_end_ist

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
