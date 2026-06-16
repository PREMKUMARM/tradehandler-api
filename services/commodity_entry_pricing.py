"""
V2 — strategy-aware entry pricing (not blind ask chase).

Aligns with common Indian F&O intraday practice:
- ORB: enter after range breakout close, limit at mid / structure fair premium (Zerodha ITM ORB).
- PDH/PDL: enter after level break + buffer, prefer pullback limit at break retest.
- EMA pullback: enter only in EMA zone when trend aligns (do not chase extended moves).
- Directional: EMA + prior-close alignment; patient mid limit with chase cap.

References: Zerodha ITM ORB, OptionX ORB+EMA, BB mean-reversion pullback entries
on 5m Nifty (middle / lower for CE, middle / upper for PE).

Bollinger (20, 2σ on 5m) is well suited here: we buy options after spot pulls back
to the mean or band — not at band extension (reduces chase into expensive premium).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.kite_live_indicators import bollinger_zone
from utils.kite_order_utils import round_to_tick

ORB_BREAK_BUFFER = 5.0
PDH_PDL_BUFFER = 3.0
EMA_PULLBACK_BAND = 28.0
EMA_MAX_CHASE_PTS = 48.0
MAX_CHASE_ABOVE_FAIR_PCT = 2.0
WIDE_SPREAD_PCT = 2.5
TICK = 0.05


@dataclass
class EntryAnalysis:
    entry_ready: bool
    entry_limit_price: float
    fair_premium: float
    entry_style: str
    spot_trigger: Optional[float]
    confirmation_score: int
    notes: List[str] = field(default_factory=list)
    block_reason: Optional[str] = None


def _book(quote: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    bid = float(quote.get("bid") or 0)
    ask = float(quote.get("ask") or 0)
    ltp = float(quote.get("ltp") or 0)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else ltp
    spread_pct = ((ask - bid) / ltp * 100.0) if ltp > 0 and ask > bid else 0.0
    return bid, ask, ltp, mid, spread_pct


def _structure_fair_premium(
    ltp: float,
    spot: float,
    spot_trigger: float,
    kind: str,
    delta: float,
) -> float:
    """Fair option premium if entry were at the structural spot trigger (not current chase)."""
    if ltp <= 0:
        return max(TICK, 0.007 * spot)
    k = (kind or "CE").upper()
    if k == "CE":
        inflation = max(0.0, spot - spot_trigger) * delta
    else:
        inflation = max(0.0, spot_trigger - spot) * delta
    return round_to_tick(max(TICK, ltp - inflation))


def _patient_buy_limit(
    quote: Dict[str, float],
    fair: float,
    *,
    max_chase_pct: float = MAX_CHASE_ABOVE_FAIR_PCT,
) -> Tuple[float, str]:
    bid, ask, ltp, mid, spread_pct = _book(quote)
    ref = fair if fair > 0 else (ltp if ltp > 0 else 0)
    min_px = max(TICK, ref * 0.5) if ref >= 2.0 else max(TICK, ref * 0.85 if ref > 0 else TICK)
    cap = fair * (1.0 + max_chase_pct / 100.0) if fair > 0 else (ask or ltp)

    if spread_pct >= WIDE_SPREAD_PCT and bid >= min_px:
        base = round_to_tick(bid + TICK)
        style = "bid_plus_tick_wide_spread"
    elif mid >= min_px:
        base = round_to_tick(mid)
        style = "mid_patient"
    elif bid >= min_px:
        base = round_to_tick(bid + TICK)
        style = "bid_plus_tick"
    elif ltp >= min_px:
        base = round_to_tick(max(min_px, ltp * 0.995))
        style = "ltp_discount"
    elif ref >= min_px:
        base = round_to_tick(ref)
        style = "fair_ref"
    else:
        base = round_to_tick(min_px)
        style = "min_floor"

    limit = base
    if ask > 0:
        limit = min(limit, ask)
    if cap > 0:
        limit = min(limit, round_to_tick(cap))
    return round_to_tick(max(min_px, limit)), style


def _candle_confirms_break(
    level: float,
    kind: str,
    spot: float,
    last_5m_close: Optional[float],
) -> bool:
    """Breakout confirmation: last 5m close beyond level (ORB best practice)."""
    if last_5m_close is None or level <= 0:
        return False
    if kind == "CE":
        return last_5m_close > level and spot > level
    return last_5m_close < level and spot < level


def _analyze_orb(
    spot: float,
    kind: str,
    intra: Dict[str, Any],
    last_5m: Optional[float],
) -> Tuple[bool, Optional[float], int, List[str], Optional[str]]:
    or_h, or_l = intra.get("or_high"), intra.get("or_low")
    notes: List[str] = []
    if not or_h or not or_l:
        return False, None, 0, notes, "Opening range not formed yet (wait until 9:30 AM)"

    or_h_f, or_l_f = float(or_h), float(or_l)
    or_range = or_h_f - or_l_f
    if or_range < 25:
        return (
            False,
            None,
            20,
            notes,
            f"ORB: opening range too tight ({or_range:.0f} pts) — skip false breaks",
        )

    minutes = int(intra.get("session_minutes") or 0)
    if minutes >= 13 * 60 + 30:
        return False, None, 20, notes, "ORB entry window closed after 13:30 IST"

    day_open = float(intra.get("day_open") or 0)

    if kind == "CE":
        trigger = or_h_f
        inside = spot <= trigger
        if inside:
            return (
                False,
                trigger,
                15,
                notes,
                f"ORB CE: wait for close above OR high {trigger:.0f} (spot {spot:.0f})",
            )
        confirmed = _candle_confirms_break(trigger, "CE", spot, last_5m)
        fresh = spot > trigger + ORB_BREAK_BUFFER
        if not fresh:
            return (
                False,
                trigger,
                25,
                notes,
                f"ORB CE: need break above OR high +{ORB_BREAK_BUFFER:.0f} pts buffer",
            )
        if not confirmed:
            return (
                False,
                trigger,
                45,
                notes,
                f"ORB CE: above OR high but 5m close not confirmed — wait",
            )
        if day_open > 0 and spot < day_open - ORB_BREAK_BUFFER:
            return (
                False,
                trigger,
                30,
                notes,
                "ORB CE blocked: spot below day open — bearish session, not CE breakout",
            )
        notes.append(f"ORB CE breakout confirmed (5m close > {trigger:.0f})")
        return True, trigger, 85, notes, None

    trigger = or_l_f
    inside = spot >= trigger
    if inside:
        return (
            False,
            trigger,
            15,
            notes,
            f"ORB PE: wait for close below OR low {trigger:.0f} (spot {spot:.0f})",
        )
    confirmed = _candle_confirms_break(trigger, "PE", spot, last_5m)
    fresh = spot < trigger - ORB_BREAK_BUFFER
    if not fresh:
        return (
            False,
            trigger,
            25,
            notes,
            f"ORB PE: need break below OR low −{ORB_BREAK_BUFFER:.0f} pts buffer",
        )
    if not confirmed:
        return (
            False,
            trigger,
            45,
            notes,
            f"ORB PE: below OR low but 5m close not confirmed — wait",
        )
    if day_open > 0 and spot > day_open + ORB_BREAK_BUFFER:
        return (
            False,
            trigger,
            30,
            notes,
            "ORB PE blocked: spot above day open — bullish session, not PE breakdown",
        )
    notes.append(f"ORB PE breakdown confirmed (5m close < {trigger:.0f})")
    return True, trigger, 85, notes, None


def _analyze_pdh_pdl(
    spot: float,
    kind: str,
    intra: Dict[str, Any],
    last_5m: Optional[float],
) -> Tuple[bool, Optional[float], int, List[str], Optional[str]]:
    pdh, pdl = intra.get("pdh"), intra.get("pdl")
    notes: List[str] = []
    if kind == "CE":
        if not pdh:
            return False, None, 0, notes, "PDH unavailable"
        trigger = float(pdh)
        if spot <= trigger + PDH_PDL_BUFFER:
            return (
                False,
                trigger,
                20,
                notes,
                f"PDH CE: need break above PDH {trigger:.0f} (spot {spot:.0f})",
            )
        confirmed = _candle_confirms_break(trigger, "CE", spot, last_5m)
        score = 80 if confirmed else 55
        notes.append(
            f"PDH break {'confirmed' if confirmed else 'in progress'} above {trigger:.0f}"
        )
        return True, trigger, score, notes, None

    if not pdl:
        return False, None, 0, notes, "PDL unavailable"
    trigger = float(pdl)
    if spot >= trigger - PDH_PDL_BUFFER:
        return (
            False,
            trigger,
            20,
            notes,
            f"PDL PE: need break below PDL {trigger:.0f} (spot {spot:.0f})",
        )
    confirmed = _candle_confirms_break(trigger, "PE", spot, last_5m)
    score = 80 if confirmed else 55
    notes.append(
        f"PDL break {'confirmed' if confirmed else 'in progress'} below {trigger:.0f}"
    )
    return True, trigger, score, notes, None


def _analyze_ema_pullback(
    spot: float,
    kind: str,
    intra: Dict[str, Any],
    prev_close: float,
) -> Tuple[bool, Optional[float], int, List[str], Optional[str]]:
    ema9 = intra.get("ema9")
    notes: List[str] = []
    if ema9 is None:
        return False, None, 0, notes, "9 EMA not ready — wait for 5m bars"

    ema = float(ema9)
    dist = abs(spot - ema)
    if dist > EMA_MAX_CHASE_PTS:
        return (
            False,
            ema,
            10,
            notes,
            f"Too far from 9 EMA ({dist:.0f} pts) — do not chase; wait for pullback",
        )

    if kind == "CE":
        in_band = ema - 8 <= spot <= ema + EMA_PULLBACK_BAND
        trend_ok = spot >= ema and (prev_close <= 0 or spot >= prev_close * 0.999)
        if not in_band:
            return (
                False,
                ema,
                25,
                notes,
                f"EMA CE: wait for pullback to 9 EMA zone ({ema:.0f} ± band)",
            )
        if not trend_ok:
            return False, ema, 30, notes, "EMA CE: spot below EMA or weak vs prior close"
        notes.append(f"EMA pullback CE near 9 EMA {ema:.0f} (dist {dist:.0f})")
        return True, ema, 75 if dist <= 18 else 60, notes, None

    in_band = ema - EMA_PULLBACK_BAND <= spot <= ema + 8
    trend_ok = spot <= ema and (prev_close <= 0 or spot <= prev_close * 1.001)
    if not in_band:
        return (
            False,
            ema,
            25,
            notes,
            f"EMA PE: wait for pullback to 9 EMA zone ({ema:.0f} ± band)",
        )
    if not trend_ok:
        return False, ema, 30, notes, "EMA PE: spot above EMA or weak vs prior close"
    notes.append(f"EMA pullback PE near 9 EMA {ema:.0f} (dist {dist:.0f})")
    return True, ema, 75 if dist <= 18 else 60, notes, None


def _apply_bollinger_entry_gate(
    spot: float,
    kind: str,
    intra: Dict[str, Any],
    ready: bool,
    trigger: Optional[float],
    score: int,
    notes: List[str],
    block: Optional[str],
) -> Tuple[bool, Optional[float], int, List[str], Optional[str], str]:
    """
    Require BB middle or favorable band touch before entry (adaptable to all strategies).
    Returns (ready, trigger, score, notes, block, entry_style_suffix).
    """
    mid = intra.get("bb_middle")
    upper = intra.get("bb_upper")
    lower = intra.get("bb_lower")
    if mid is None or upper is None or lower is None:
        notes.append(
            "BB unavailable (need Kite 5m history) — strategy levels only until bands load"
        )
        return ready, trigger, score, notes, block, ""

    bb_src = (intra.get("indicator_sources") or {}).get("bb_middle", "")
    hist_pad = intra.get("hist_pad_bars")
    live_n = intra.get("live_session_bars")
    if hist_pad is not None and live_n is not None:
        notes.append(
            f"BB window: {hist_pad} hist pad + {live_n} live session bars"
            + (f" ({bb_src})" if bb_src else "")
        )

    bb = bollinger_zone(float(spot), float(mid), float(upper), float(lower), kind)
    zone = bb["zone"]
    style_suffix = f"bb_{zone}"

    if bb["extended"]:
        notes.append(
            f"BB {zone}: spot extended — prefer middle ({mid:.0f}) not band chase"
        )
        if ready:
            return False, bb["trigger"], score, notes, bb["wait_msg"], style_suffix
        return ready, trigger, score, notes, block or bb["wait_msg"], style_suffix

    if bb["preferred"]:
        notes.append(
            f"BB {zone} touch (mid {mid:.0f}, L {lower:.0f}, U {upper:.0f}) — preferred entry zone"
        )
        new_trigger = float(bb["trigger"])
        if ready:
            return (
                True,
                new_trigger,
                min(98, score + 14),
                notes,
                None,
                style_suffix,
            )
        return (
            False,
            new_trigger,
            max(score, 45),
            notes,
            block or bb["wait_msg"].replace("Wait for", "BB ready — need"),
            style_suffix,
        )

    wait = bb["wait_msg"]
    notes.append(wait)
    if ready:
        return False, bb["trigger"], score, notes, wait, style_suffix
    return ready, trigger, score, notes, block, style_suffix


def _analyze_directional(
    spot: float,
    kind: str,
    intra: Dict[str, Any],
    prev_close: float,
) -> Tuple[bool, Optional[float], int, List[str], Optional[str]]:
    ema9 = intra.get("ema9")
    notes: List[str] = []
    trigger = spot
    if kind == "CE":
        if prev_close > 0 and spot < prev_close:
            return (
                False,
                prev_close,
                35,
                notes,
                "Directional CE: spot below prior close — wait for bullish alignment",
            )
        if ema9 is not None and spot < float(ema9) - 15:
            return (
                False,
                float(ema9),
                40,
                notes,
                f"Directional CE: spot below 9 EMA {float(ema9):.0f}",
            )
        notes.append("Directional CE: spot vs prior close / EMA aligned")
        return True, trigger, 65, notes, None

    if prev_close > 0 and spot > prev_close:
        return (
            False,
            prev_close,
            35,
            notes,
            "Directional PE: spot above prior close — wait for bearish alignment",
        )
    if ema9 is not None and spot > float(ema9) + 15:
        return (
            False,
            float(ema9),
            40,
            notes,
            f"Directional PE: spot above 9 EMA {float(ema9):.0f}",
        )
    notes.append("Directional PE: spot vs prior close / EMA aligned")
    return True, trigger, 65, notes, None


def compute_strategy_entry(
    *,
    strategy_id: str,
    option_kind: str,
    quote: Dict[str, float],
    spot: float,
    strike: int,
    delta: float,
    intra: Dict[str, Any],
    prev_close: float = 0.0,
) -> EntryAnalysis:
    """
    Decide if structural entry is confirmed and compute LIMIT price (patient, not ask chase).
    """
    kind = (option_kind or "CE").upper()
    sid = strategy_id or "long_atm_directional"
    bid, ask, ltp, mid, spread_pct = _book(quote)
    ref_ltp = ltp or mid or ask or bid
    last_5m = intra.get("last_5m_close")
    if last_5m is not None:
        last_5m = float(last_5m)

    ready = False
    trigger: Optional[float] = None
    score = 0
    notes: List[str] = []
    block: Optional[str] = None

    if sid == "orb_15m_breakout":
        ready, trigger, score, notes, block = _analyze_orb(spot, kind, intra, last_5m)
    elif sid == "pdh_pdl_breakout":
        ready, trigger, score, notes, block = _analyze_pdh_pdl(spot, kind, intra, last_5m)
    elif sid == "ema_pullback_continuation":
        ready, trigger, score, notes, block = _analyze_ema_pullback(
            spot, kind, intra, prev_close
        )
    else:
        ready, trigger, score, notes, block = _analyze_directional(
            spot, kind, intra, prev_close
        )

    ready, trigger, score, notes, block, bb_style = _apply_bollinger_entry_gate(
        spot, kind, intra, ready, trigger, score, notes, block
    )

    fair = _structure_fair_premium(
        ref_ltp, spot, trigger if trigger is not None else spot, kind, delta
    )
    limit, style = _patient_buy_limit(quote, fair)
    if bb_style:
        style = f"{style}_{bb_style}" if ready else bb_style

    if ready:
        notes.append(
            f"LIMIT ₹{limit} ({style}) · fair ₹{fair:.2f} · book mid ₹{mid:.2f} "
            f"spread {spread_pct:.1f}%"
        )
    else:
        notes.append(
            f"Preview LIMIT ₹{limit} if setup confirms · fair ₹{fair:.2f} (do not market chase)"
        )

    return EntryAnalysis(
        entry_ready=ready,
        entry_limit_price=limit,
        fair_premium=round(fair, 2),
        entry_style=style if ready else "blocked_wait",
        spot_trigger=round(trigger, 2) if trigger is not None else None,
        confirmation_score=score,
        notes=notes,
        block_reason=block,
    )
