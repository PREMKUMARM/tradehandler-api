"""Index ATM strike rounding (Nifty 50-pt vs Sensex 100-pt)."""
from __future__ import annotations


def index_strike_step(segment: str = "sensex") -> int:
    seg = (segment or "sensex").strip().lower()
    if seg in ("nifty50", "nifty", "nfo"):
        return 50
    return 100


def true_atm_from_spot(spot: float, *, segment: str = "sensex") -> int:
    step = index_strike_step(segment)
    if spot <= 0:
        return 0
    return int(round(float(spot) / step) * step)
