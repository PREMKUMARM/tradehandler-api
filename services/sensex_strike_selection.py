"""
Smart strike selection for Sensex 20rupees-strategy.

Picks among premium-band (₹17–₹23) candidates using OI + proximity to ATM,
instead of raw global max-OI deep OTM lottery strikes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.sensex_constants import sensex_atm_near_offsets, sensex_atm_near_strike_points

# ₹17–₹23 band on Sensex weekly options typically sits ~700–1000 pts OTM.
MAX_OTM_POINTS = 1000
MAX_OTM_POINTS_RELAXED = 1200

OI_WEIGHT = 0.50
PROXIMITY_WEIGHT = 0.50
DIRECTION_BIAS_MULT = 1.15

PREMIUM_BAND_LOW = 17.0
PREMIUM_BAND_HIGH = 23.0


@dataclass(frozen=True)
class StrikeCandidate:
    kind: str
    strike: int
    offset: str
    oi: float
    ltp: float
    score: float
    dist_pts: int
    symbol: Optional[str] = None


def _in_band(ltp: float, band_low: float, band_high: float) -> bool:
    return band_low <= ltp <= band_high


def _band_touched(bar_low: float, bar_high: float, band_low: float, band_high: float) -> bool:
    return bar_low <= band_high and bar_high >= band_low


def _true_atm(spot: float) -> int:
    """ATM strike from Sensex index LTP (rounded to 100)."""
    return int(round(float(spot) / 100.0) * 100)


def true_atm_from_spot(spot: float) -> int:
    return _true_atm(spot)


def _valid_otm_side(kind: str, strike: int, atm: int) -> bool:
    k = (kind or "").upper()
    if k == "CE":
        return strike >= atm
    if k == "PE":
        return strike <= atm
    return True


def _proximity_factor(dist_pts: int) -> float:
    """Prefer nearer OTM (700 pts) over tail (1000 pts) within the band."""
    d = max(0, int(dist_pts))
    return 1.0 / (1.0 + max(0, d - 400) / 200.0)


def _offset_step_distance(offset: str) -> int:
    off = (offset or "ATM").upper()
    if off == "ATM":
        return 0
    if off.startswith("ATM+"):
        return int(off[4:] or 0)
    if off.startswith("ATM-"):
        return int(off[4:] or 0)
    return 99


def _near_atm_strike_ok(strike: int, atm: int) -> bool:
    return abs(int(strike) - int(atm)) <= sensex_atm_near_strike_points()


def _pick_best_near_atm(pool: List[StrikeCandidate]) -> Optional[StrikeCandidate]:
    if not pool:
        return None
    # Prefer strike nearest spot-derived ATM (LTP), then rolling offset, then OI.
    return min(pool, key=lambda c: (c.dist_pts, _offset_step_distance(c.offset), -c.oi))


def _direction_bias_kind(spot: float, prev_close: float) -> Optional[str]:
    if prev_close <= 0 or spot <= 0:
        return None
    return "CE" if spot >= prev_close else "PE"


def score_strike(
    *,
    oi: float,
    strike: int,
    atm: int,
    kind: str,
    max_oi: float,
    bias_kind: Optional[str] = None,
) -> float:
    dist = abs(int(strike) - int(atm))
    oi_norm = float(oi) / max_oi if max_oi > 0 else 0.0
    prox = _proximity_factor(dist)
    score = OI_WEIGHT * oi_norm + PROXIMITY_WEIGHT * prox
    if bias_kind and (kind or "").upper() == bias_kind:
        score *= DIRECTION_BIAS_MULT
    return score


def _pick_best(
    pool: List[StrikeCandidate],
) -> Optional[StrikeCandidate]:
    if not pool:
        return None
    return max(pool, key=lambda c: (c.score, c.oi, -c.dist_pts))


def _build_pool_from_rows(
    rows: List[Tuple[str, int, str, float, float, Optional[str]]],
    atm: int,
    max_dist: int,
    max_oi: float,
    bias_kind: Optional[str],
    *,
    require_band: bool,
    band_low: float,
    band_high: float,
    band_touched_only: bool = False,
    bar_low: float = 0.0,
    bar_high: float = 0.0,
) -> List[StrikeCandidate]:
    out: List[StrikeCandidate] = []
    for kind, strike, offset, oi, ltp, sym in rows:
        if oi <= 0:
            continue
        if not _valid_otm_side(kind, strike, atm):
            continue
        dist = abs(strike - atm)
        if dist > max_dist:
            continue
        if band_touched_only:
            if not _band_touched(bar_low, bar_high, band_low, band_high):
                continue
        elif require_band and not _in_band(ltp, band_low, band_high):
            continue
        sc = score_strike(
            oi=oi,
            strike=strike,
            atm=atm,
            kind=kind,
            max_oi=max_oi,
            bias_kind=bias_kind,
        )
        out.append(
            StrikeCandidate(
                kind=kind.upper(),
                strike=strike,
                offset=offset,
                oi=oi,
                ltp=ltp,
                score=sc,
                dist_pts=dist,
                symbol=sym,
            )
        )
    return out


def pick_smart_from_chain(
    chain_oi: Dict[str, Any],
    spot: float,
    *,
    kinds: Sequence[str] = ("CE", "PE"),
    band_low: float = PREMIUM_BAND_LOW,
    band_high: float = PREMIUM_BAND_HIGH,
    prev_close: float = 0.0,
) -> Optional[StrikeCandidate]:
    """Live chain: score in-band strikes from ranked OI rows."""
    atm = int(chain_oi.get("atm") or _true_atm(spot))
    bias = _direction_bias_kind(spot, prev_close)

    rows: List[Tuple[str, int, str, float, float, Optional[str]]] = []
    for kind in kinds:
        k = kind.upper()
        atm_row = (chain_oi.get("atm_ce") if k == "CE" else chain_oi.get("atm_pe")) or {}
        atm_ltp = float(atm_row.get("ltp") or 0)
        if atm_ltp > 0:
            rows.append((k, atm, "ATM", float(atm_row.get("oi") or 0), atm_ltp, None))
        ranked = chain_oi.get("ranked_ce_oi") if k == "CE" else chain_oi.get("ranked_pe_oi")
        for row in list(ranked or []):
            strike = int(row.get("strike") or 0)
            if strike <= 0 or not _near_atm_strike_ok(strike, atm):
                continue
            step = abs(strike - atm) // 100
            off = "ATM" if step == 0 else (f"ATM+{step}" if strike > atm else f"ATM-{step}")
            rows.append(
                (
                    k,
                    strike,
                    off,
                    float(row.get("oi") or 0),
                    float(row.get("ltp") or 0),
                    row.get("symbol"),
                )
            )

    if not rows:
        return None

    max_oi = max(r[3] for r in rows) or 1.0
    pool = _build_pool_from_rows(
        rows,
        atm,
        sensex_atm_near_strike_points(),
        max_oi,
        bias,
        require_band=True,
        band_low=band_low,
        band_high=band_high,
    )
    best = _pick_best_near_atm(pool)
    if best:
        return best

    fallback: List[StrikeCandidate] = []
    for kind, strike, offset, oi, ltp, sym in rows:
        if oi <= 0:
            continue
        dist = abs(strike - atm)
        sc = score_strike(oi=oi, strike=strike, atm=atm, kind=kind, max_oi=max_oi, bias_kind=bias)
        fallback.append(
            StrikeCandidate(kind.upper(), strike, offset, oi, ltp, sc, dist, sym)
        )
    return _pick_best_near_atm(fallback)


def strike_source_label(offset: str) -> str:
    off = (offset or "ATM").upper()
    if off == "ATM" or off.startswith("ATM+") or off.startswith("ATM-"):
        return off
    return "SMART_OI"


def moneyness_label(strike: int, atm: int, offset: str) -> str:
    if (offset or "").upper() == "ATM" or strike == atm:
        return "ATM"
    return "SMART_OI"


def pick_smart_at_bar(
    session: Dict[str, Dict[Any, Any]],
    idx: int,
    *,
    kinds: Sequence[str] = ("CE", "PE"),
    band_low: float = PREMIUM_BAND_LOW,
    band_high: float = PREMIUM_BAND_HIGH,
    prev_close: float = 0.0,
) -> Optional[Tuple[StrikeCandidate, Any]]:
    """
    Backtest: first band close among ATM±N monitored offsets at bar idx.
    Returns (candidate, OptionSeries).
    """
    spot = 0.0
    for kind in ("CE", "PE"):
        atm_s = (session.get(kind) or {}).get("ATM")
        if atm_s and idx < len(atm_s.spot):
            spot = float(atm_s.spot[idx])
            break
    if spot <= 0:
        return None
    atm = _true_atm(spot)

    pool: List[StrikeCandidate] = []
    series_map: Dict[Tuple[str, str], Any] = {}
    for kind in kinds:
        k = kind.upper()
        for offset in sensex_atm_near_offsets():
            series = (session.get(k) or {}).get(offset)
            if not series or idx >= len(series.close):
                continue
            series_map[(k, offset)] = series
            bar_close = float(series.close[idx])
            if bar_close <= 0 or not _in_band(bar_close, band_low, band_high):
                continue
            strike = int(series.strike[idx])
            if not _valid_otm_side(k, strike, atm):
                continue
            pool.append(
                StrikeCandidate(
                    kind=k,
                    strike=strike,
                    offset=str(offset),
                    oi=float(series.oi[idx]),
                    ltp=bar_close,
                    score=0.0,
                    dist_pts=abs(strike - atm),
                )
            )

    best = _pick_best_near_atm(pool)
    if not best:
        return None
    series = series_map.get((best.kind, best.offset))
    if series is None:
        return None
    return best, series
