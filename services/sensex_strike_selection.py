"""
Smart strike selection for Sensex 20rupees-strategy.

Picks among premium-band (₹17–₹23) candidates using OI + proximity to ATM,
instead of raw global max-OI deep OTM lottery strikes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.sensex_constants import (
    sensex_atm_near_offsets,
    sensex_atm_near_strike_points,
    sensex_premium_band_scan_points,
)

# ₹17–₹23 band on Sensex weekly options typically sits ~700–1000 pts OTM.
MAX_OTM_POINTS = 1000
MAX_OTM_POINTS_RELAXED = 1200

PREMIUM_BAND_LOW = 17.0
PREMIUM_BAND_HIGH = 23.0

OI_WEIGHT = 0.35
PROXIMITY_WEIGHT = 0.30
BAND_CENTER_WEIGHT = 0.20
BUILDUP_WEIGHT = 0.15
DIRECTION_BIAS_MULT = 1.15
MAX_SPREAD_PCT = 4.0
BAND_CENTER = (PREMIUM_BAND_LOW + PREMIUM_BAND_HIGH) / 2.0


@dataclass(frozen=True)
class ResolvedSensexStrike:
    strike: int
    kind: str
    symbol: Optional[str]
    ltp: float
    moneyness: str
    offset: str
    oi: float
    source: str
    reason: str


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


def _strike_near_atm(strike: int, atm: int, max_dist: Optional[int] = None) -> bool:
    if strike <= 0 or atm <= 0:
        return False
    limit = max_dist if max_dist is not None else sensex_premium_band_scan_points()
    return abs(int(strike) - int(atm)) <= int(limit)


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
    ltp: float = 0.0,
    spread_pct: float = 0.0,
    oi_change: float = 0.0,
    band_low: float = PREMIUM_BAND_LOW,
    band_high: float = PREMIUM_BAND_HIGH,
) -> float:
    dist = abs(int(strike) - int(atm))
    oi_norm = float(oi) / max_oi if max_oi > 0 else 0.0
    prox = _proximity_factor(dist)
    half_band = max(0.5, (band_high - band_low) / 2.0)
    band_score = (
        1.0 - min(1.0, abs(float(ltp) - BAND_CENTER) / half_band)
        if ltp > 0
        else 0.5
    )
    buildup = min(1.0, float(oi_change) / max_oi) if max_oi > 0 and oi_change > 0 else 0.0
    spread_penalty = (
        max(0.25, 1.0 - float(spread_pct) / MAX_SPREAD_PCT)
        if spread_pct > 0
        else 1.0
    )
    score = (
        OI_WEIGHT * oi_norm
        + PROXIMITY_WEIGHT * prox
        + BAND_CENTER_WEIGHT * band_score
        + BUILDUP_WEIGHT * buildup
    )
    if bias_kind and (kind or "").upper() == bias_kind:
        score *= DIRECTION_BIAS_MULT
    return score * spread_penalty


def _pick_best(
    pool: List[StrikeCandidate],
) -> Optional[StrikeCandidate]:
    if not pool:
        return None
    return max(pool, key=lambda c: (c.score, c.oi, -c.dist_pts))


def _build_pool_from_rows(
    rows: Sequence[Tuple],
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
    for row in rows:
        kind, strike, offset, oi, ltp, sym = row[:6]
        spread_pct = float(row[6]) if len(row) > 6 else 0.0
        oi_change = float(row[7]) if len(row) > 7 else 0.0
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
            ltp=ltp,
            spread_pct=spread_pct,
            oi_change=oi_change,
            band_low=band_low,
            band_high=band_high,
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
    segment: str = "sensex",
) -> Optional[StrikeCandidate]:
    """Live chain: score in-band strikes from ranked OI rows."""
    from services.index_atm import index_strike_step, true_atm_from_spot

    atm = int(chain_oi.get("atm") or true_atm_from_spot(spot, segment=segment))
    bias = _direction_bias_kind(spot, prev_close)
    scan_pts = sensex_premium_band_scan_points()
    strike_step = index_strike_step(segment)

    rows: List[Tuple] = []
    for kind in kinds:
        k = kind.upper()
        atm_row = (chain_oi.get("atm_ce") if k == "CE" else chain_oi.get("atm_pe")) or {}
        atm_ltp = float(atm_row.get("ltp") or 0)
        if atm_ltp > 0 and _in_band(atm_ltp, band_low, band_high):
            rows.append(
                (
                    k,
                    atm,
                    "ATM",
                    float(atm_row.get("oi") or 0),
                    atm_ltp,
                    None,
                    float(atm_row.get("spread_pct") or 0),
                    0.0,
                )
            )
        ranked = chain_oi.get("ranked_ce_oi") if k == "CE" else chain_oi.get("ranked_pe_oi")
        for row in list(ranked or []):
            strike = int(row.get("strike") or 0)
            if strike <= 0 or abs(strike - atm) > scan_pts:
                continue
            step = abs(strike - atm) // max(1, strike_step)
            off = "ATM" if step == 0 else (f"ATM+{step}" if strike > atm else f"ATM-{step}")
            rows.append(
                (
                    k,
                    strike,
                    off,
                    float(row.get("oi") or 0),
                    float(row.get("ltp") or 0),
                    row.get("symbol"),
                    float(row.get("spread_pct") or 0),
                    float(row.get("oi_change") or 0),
                )
            )

    if not rows:
        return None

    max_oi = max(r[3] for r in rows) or 1.0
    pool = _build_pool_from_rows(
        rows,
        atm,
        scan_pts,
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
    for row in rows:
        kind, strike, offset, oi, ltp, sym = row[:6]
        spread_pct = float(row[6]) if len(row) > 6 else 0.0
        oi_change = float(row[7]) if len(row) > 7 else 0.0
        if oi <= 0 or not _in_band(ltp, band_low, band_high):
            continue
        dist = abs(strike - atm)
        sc = score_strike(
            oi=oi,
            strike=strike,
            atm=atm,
            kind=kind,
            max_oi=max_oi,
            bias_kind=bias,
            ltp=ltp,
            spread_pct=spread_pct,
            oi_change=oi_change,
            band_low=band_low,
            band_high=band_high,
        )
        fallback.append(
            StrikeCandidate(kind.upper(), strike, offset, oi, ltp, sc, dist, sym)
        )
    return _pick_best(fallback) if fallback else None


def strike_source_label(offset: str) -> str:
    off = (offset or "ATM").upper()
    if off == "ATM" or off.startswith("ATM+") or off.startswith("ATM-"):
        return off
    return "SMART_OI"


def moneyness_label(strike: int, atm: int, offset: str) -> str:
    if (offset or "").upper() == "ATM" or strike == atm:
        return "ATM"
    return "SMART_OI"


def _chain_row_for_strike(
    chain_oi: Dict[str, Any], kind: str, strike: int
) -> Optional[Dict[str, Any]]:
    k = (kind or "CE").upper()
    atm = int(chain_oi.get("atm") or 0)
    if strike == atm:
        row = (chain_oi.get("atm_ce") if k == "CE" else chain_oi.get("atm_pe")) or {}
        if row:
            return {"strike": strike, **row}
    ranked = chain_oi.get("ranked_ce_oi") if k == "CE" else chain_oi.get("ranked_pe_oi")
    for row in list(ranked or []):
        if int(row.get("strike") or 0) == int(strike):
            return row
    return None


def _available_strikes(chain_oi: Dict[str, Any], kind: str) -> List[int]:
    k = (kind or "CE").upper()
    ranked = chain_oi.get("ranked_ce_oi") if k == "CE" else chain_oi.get("ranked_pe_oi")
    strikes = sorted({int(r.get("strike") or 0) for r in list(ranked or []) if int(r.get("strike") or 0) > 0})
    atm = int(chain_oi.get("atm") or 0)
    if atm > 0 and atm not in strikes:
        strikes.append(atm)
        strikes.sort()
    return strikes


def _strike_from_moneyness_label(
    moneyness: str,
    kind: str,
    spot: float,
    available: List[int],
) -> int:
    from services.push.option_contract_resolver import strike_for_moneyness

    target = strike_for_moneyness(spot, kind, moneyness, atm_step=100, available=available or None)
    if available:
        return min(available, key=lambda s: abs(s - target))
    return target


def _candidate_to_resolved(
    picked: StrikeCandidate, atm: int, *, source: str, reason: str
) -> ResolvedSensexStrike:
    return ResolvedSensexStrike(
        strike=int(picked.strike),
        kind=picked.kind.upper(),
        symbol=picked.symbol,
        ltp=float(picked.ltp),
        moneyness=moneyness_label(picked.strike, atm, picked.offset),
        offset=picked.offset,
        oi=float(picked.oi),
        source=source,
        reason=reason,
    )


def resolve_sensex_strike_for_plan(
    *,
    spot: float,
    option_kind: str,
    chain_oi: Optional[Dict[str, Any]],
    strategy_id: str,
    moneyness: str = "ATM",
    anchor_strike: Optional[int] = None,
    band_low: float = PREMIUM_BAND_LOW,
    band_high: float = PREMIUM_BAND_HIGH,
    prev_close: float = 0.0,
    day_open: float = 0.0,
    direction: str = "AUTO",
) -> ResolvedSensexStrike:
    """
    Pick the best Sensex strike for live order placement.

    Priority: validated strategy anchor → smart OI band pick → moneyness map → ATM.
    """
    chain = chain_oi or {}
    kind = (option_kind or "CE").upper()
    atm = int(chain.get("atm") or _true_atm(spot))
    sid = (strategy_id or "20rupees_strategy").strip()

    if sid == "green_bar_sentinel_2nd_oi" and chain:
        from services.sensex_oi_sentinel import pick_sentinel_anchor

        sk, anchor, meta = pick_sentinel_anchor(chain, direction_pref=direction)
        if anchor and int(anchor) > 0:
            row = _chain_row_for_strike(chain, sk, int(anchor)) or meta or {}
            return ResolvedSensexStrike(
                strike=int(anchor),
                kind=sk.upper(),
                symbol=row.get("symbol") or row.get("ce_symbol") or row.get("pe_symbol"),
                ltp=float(row.get("ltp") or row.get("ce_ltp") or row.get("pe_ltp") or 0),
                moneyness="ANCHOR",
                offset="OI_SENTINEL",
                oi=float(row.get("oi") or row.get("ce_oi") or row.get("pe_oi") or 0),
                source="oi_sentinel",
                reason="2nd OI-anchor strike from intraday buildup rank",
            )

    if anchor_strike and int(anchor_strike) > 0:
        row = _chain_row_for_strike(chain, kind, int(anchor_strike)) if chain else None
        ltp = float((row or {}).get("ltp") or 0)
        if (
            row
            and _valid_otm_side(kind, int(anchor_strike), atm)
            and _strike_near_atm(int(anchor_strike), atm)
            and _in_band(ltp, band_low, band_high)
            and float(row.get("spread_pct") or 0) <= MAX_SPREAD_PCT
        ):
            step = abs(int(anchor_strike) - atm) // 100
            off = "ATM" if step == 0 else (f"ATM+{step}" if int(anchor_strike) > atm else f"ATM-{step}")
            return ResolvedSensexStrike(
                strike=int(anchor_strike),
                kind=kind,
                symbol=row.get("symbol"),
                ltp=ltp,
                moneyness=moneyness_label(int(anchor_strike), atm, off),
                offset=off,
                oi=float(row.get("oi") or 0),
                source="anchor",
                reason=f"Strategy anchor {kind} {anchor_strike} in ₹{band_low:.0f}–₹{band_high:.0f} band",
            )

    if chain and sid in ("20rupees_strategy", "bb_5m_mean_reversion", "long_atm_directional", ""):
        from services.entry_quality import auto_entry_kind, entry_day_aligned_ok

        kinds = (kind,)
        if sid == "20rupees_strategy" and (direction or "AUTO").upper() == "AUTO":
            day_open = day_open or spot
            auto_kind = auto_entry_kind(day_open, spot, prev_close)
            if not auto_kind or not entry_day_aligned_ok(
                kind=auto_kind, index_open=day_open, spot=spot
            ):
                auto_kind = None
            kind = auto_kind or kind
            kinds = (kind,) if kind else ()
        picked = pick_smart_from_chain(
            chain,
            spot,
            kinds=kinds,
            band_low=band_low,
            band_high=band_high,
            prev_close=prev_close,
        )
        if picked:
            return _candidate_to_resolved(
                picked,
                atm,
                source="smart_oi",
                reason=(
                    f"Smart pick {picked.kind} {picked.strike} ({picked.offset}) "
                    f"₹{picked.ltp:.2f} · OI {picked.oi:,.0f} near ATM {atm}"
                ),
            )

    if chain and moneyness not in ("ATM", "SMART_OI", "ANCHOR", "20rupees"):
        available = _available_strikes(chain, kind)
        if available:
            target = _strike_from_moneyness_label(moneyness, kind, spot, available)
            if _strike_near_atm(target, atm):
                row = _chain_row_for_strike(chain, kind, target)
                return ResolvedSensexStrike(
                    strike=int(target),
                    kind=kind,
                    symbol=(row or {}).get("symbol"),
                    ltp=float((row or {}).get("ltp") or 0),
                    moneyness=moneyness,
                    offset=moneyness,
                    oi=float((row or {}).get("oi") or 0),
                    source="moneyness",
                    reason=f"{sid}: {moneyness} strike mapped on live chain",
                )

    atm_fallback = atm
    return ResolvedSensexStrike(
        strike=atm_fallback,
        kind=kind,
        symbol=None,
        ltp=float(((chain.get("atm_ce") if kind == "CE" else chain.get("atm_pe")) or {}).get("ltp") or 0),
        moneyness="ATM",
        offset="ATM",
        oi=0.0,
        source="atm_fallback",
        reason=f"Fallback ATM {kind} {atm} — no in-band smart candidate",
    )


def pick_smart_at_bar(
    session: Dict[str, Dict[Any, Any]],
    idx: int,
    *,
    kinds: Sequence[str] = ("CE", "PE"),
    band_low: float = PREMIUM_BAND_LOW,
    band_high: float = PREMIUM_BAND_HIGH,
    prev_close: float = 0.0,
    segment: str = "sensex",
) -> Optional[Tuple[StrikeCandidate, Any]]:
    """
    Backtest: band close among ATM±N offsets at bar idx; OI-weighted strike pick.
    Returns (candidate, OptionSeries).
    """
    from services.index_atm import true_atm_from_spot

    spot = 0.0
    for kind in ("CE", "PE"):
        atm_s = (session.get(kind) or {}).get("ATM")
        if atm_s and idx < len(atm_s.spot):
            spot = float(atm_s.spot[idx])
            break
    if spot <= 0:
        return None
    atm = true_atm_from_spot(spot, segment=segment)
    bias_kind = _direction_bias_kind(spot, prev_close) if prev_close > 0 else None

    pool: List[StrikeCandidate] = []
    series_map: Dict[Tuple[str, str], Any] = {}
    max_oi = 1.0
    raw_candidates: List[Tuple] = []
    for kind in kinds:
        k = kind.upper()
        for offset in sensex_atm_near_offsets():
            series = (session.get(k) or {}).get(offset)
            if not series or idx >= len(series.close):
                continue
            bar_close = float(series.close[idx])
            if bar_close <= 0 or not _in_band(bar_close, band_low, band_high):
                continue
            strike = int(series.strike[idx])
            if not _valid_otm_side(k, strike, atm):
                continue
            oi = float(series.oi[idx])
            max_oi = max(max_oi, oi)
            raw_candidates.append((k, strike, offset, oi, bar_close, series))

    for k, strike, offset, oi, bar_close, series in raw_candidates:
        dist = abs(strike - atm)
        sc = score_strike(
            oi=oi,
            strike=strike,
            atm=atm,
            kind=k,
            max_oi=max_oi,
            bias_kind=bias_kind,
            ltp=bar_close,
            band_low=band_low,
            band_high=band_high,
        )
        pool.append(
            StrikeCandidate(
                kind=k,
                strike=strike,
                offset=str(offset),
                oi=oi,
                ltp=bar_close,
                score=sc,
                dist_pts=dist,
            )
        )
        series_map[(k, offset)] = series

    best = _pick_best(pool)
    if not best:
        return None
    series = series_map.get((best.kind, best.offset))
    if series is None:
        return None
    return best, series
