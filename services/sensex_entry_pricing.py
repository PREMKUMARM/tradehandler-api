"""
V2 — 5m Bollinger Bands entry pricing (patient LIMIT, not ask chase).

CE: enter at lower or middle band touch on 5m Sensex.
PE: enter at upper or middle band touch.
Blocks band-extension chase (CE at upper, PE at lower).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.sensex_constants import (
    is_past_sensex_entry_cutoff,
    sensex_entry_cutoff_message,
    sensex_is_bad_option_bar,
    sensex_is_gap_up_session,
)
from services.sensex_strategy_analysis import (
    FIXED_SL_INR,
    PREMIUM_BAND_HIGH,
    PREMIUM_BAND_LOW,
    STRATEGY_ID as TWENTY_RUPEES_ID,
)
from services.option_contract_indicators import contract_bb_is_active, contract_price_for_bb
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
    cap = fair * (1.0 + max_chase_pct / 100.0) if fair > 0 else (ask or ltp)

    if spread_pct >= WIDE_SPREAD_PCT and bid > 0:
        base = round_to_tick(bid + TICK)
        style = "bid_plus_tick_wide_spread"
    elif mid > 0:
        base = round_to_tick(mid)
        style = "mid_patient"
    elif bid > 0:
        base = round_to_tick(bid + TICK)
        style = "bid_plus_tick"
    else:
        base = round_to_tick(max(TICK, ltp * 0.995))
        style = "ltp_discount"

    limit = base
    if ask > 0:
        limit = min(limit, ask)
    if cap > 0:
        limit = min(limit, round_to_tick(cap))
    return round_to_tick(max(TICK, limit)), style


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

    if kind == "CE":
        trigger = float(or_h)
        confirmed = _candle_confirms_break(trigger, "CE", spot, last_5m)
        fresh = spot > trigger + ORB_BREAK_BUFFER
        inside = spot <= trigger
        if inside:
            return (
                False,
                trigger,
                15,
                notes,
                f"ORB CE: wait for close above OR high {trigger:.0f} (spot {spot:.0f})",
            )
        score = 85 if confirmed and fresh else (60 if fresh else 40)
        if confirmed:
            notes.append(f"ORB CE breakout confirmed (5m close > {trigger:.0f})")
        elif fresh:
            notes.append(f"ORB CE above OR high +{spot - trigger:.0f} pts — prefer 5m close confirm")
        return True, trigger, score, notes, None

    trigger = float(or_l)
    confirmed = _candle_confirms_break(trigger, "PE", spot, last_5m)
    fresh = spot < trigger - ORB_BREAK_BUFFER
    inside = spot >= trigger
    if inside:
        return (
            False,
            trigger,
            15,
            notes,
            f"ORB PE: wait for close below OR low {trigger:.0f} (spot {spot:.0f})",
        )
    score = 85 if confirmed and fresh else (60 if fresh else 40)
    if confirmed:
        notes.append(f"ORB PE breakdown confirmed (5m close < {trigger:.0f})")
    elif fresh:
        notes.append(f"ORB PE below OR low — prefer 5m close confirm")
    return True, trigger, score, notes, None


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

    px = contract_price_for_bb(spot, intra)
    bb = bollinger_zone(px, float(mid), float(upper), float(lower), kind)
    zone = bb["zone"]
    style_suffix = f"bb_{zone}"
    on_contract = contract_bb_is_active(intra)
    fmt = ".2f" if on_contract or px < 5000 else ".0f"

    if bb["extended"]:
        notes.append(
            f"BB {zone}: contract LTP extended — prefer middle ({mid:{fmt}}) not band chase"
        )
        if ready:
            return False, bb["trigger"], score, notes, bb["wait_msg"], style_suffix
        return ready, trigger, score, notes, block or bb["wait_msg"], style_suffix

    if bb["preferred"]:
        notes.append(
            f"BB {zone} touch (mid {mid:{fmt}}, L {lower:{fmt}}, U {upper:{fmt}}) — preferred entry zone"
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


def _analyze_green_bar_sentinel(
    spot: float,
    kind: str,
    intra: Dict[str, Any],
    last_5m: Optional[float],
) -> Tuple[bool, Optional[float], int, List[str], Optional[str]]:
    from services.sensex_oi_sentinel import reversal_confirmed_at_anchor

    notes: List[str] = []
    anchor = int(intra.get("anchor_strike") or 0)
    if anchor <= 0:
        return False, None, 20, notes, "2nd OI anchor strike unavailable — refresh option chain"

    if not intra.get("oi_baseline_ready"):
        return (
            False,
            float(anchor),
            25,
            notes,
            "OI baseline forming — wait until 9:16+ for green-bar ranking",
        )

    ready, score, msg = reversal_confirmed_at_anchor(
        spot=spot,
        anchor_strike=anchor,
        kind=kind,
        intra=intra,
        last_5m_close=last_5m,
    )
    notes.append(msg)
    trigger = float(anchor)
    if ready:
        notes.append(f"Green Bar Sentinel: reversal confirmed at 2nd OI anchor {anchor}")
        return True, trigger, score, notes, None
    return False, trigger, score, notes, msg


def _analyze_20rupees(
    quote: Dict[str, float],
    intra: Dict[str, Any],
    prev_close: float = 0.0,
) -> Tuple[bool, Optional[float], int, List[str], Optional[str], str]:
    """Entry when contract premium is in ₹17–₹23 band (not after session entry cutoff)."""
    bid, ask, ltp, _, _ = _book(quote)
    ltp = float(intra.get("contract_ltp") or intra.get("option_ltp") or ltp or 0)
    day_open = float(intra.get("day_open") or 0)
    notes: List[str] = []
    band = f"₹{PREMIUM_BAND_LOW:.0f}–{PREMIUM_BAND_HIGH:.0f}"

    if is_past_sensex_entry_cutoff():
        msg = sensex_entry_cutoff_message()
        notes.append(msg)
        return False, None, 0, notes, msg, "20rupees_cutoff"

    if sensex_is_gap_up_session(day_open, prev_close):
        gap_pct = (day_open - prev_close) / prev_close * 100.0 if prev_close > 0 else 0.0
        msg = f"Gap-up session (+{gap_pct:.2f}%) — skip entry"
        notes.append(msg)
        return False, None, 0, notes, msg, "20rupees_gap_up_skip"

    bar_open = float(intra.get("bar_open") or quote.get("open") or 0)
    bar_high = float(intra.get("bar_high") or quote.get("high") or ltp or 0)
    if sensex_is_bad_option_bar(bar_open, bar_high, ltp):
        msg = "Bad option tick (open/high > 3× LTP) — skip entry"
        notes.append(msg)
        return False, None, 0, notes, msg, "20rupees_bad_tick"

    if ltp <= 0:
        return (
            False,
            None,
            0,
            notes,
            "Option LTP unavailable — wait for chain quote",
            "20rupees_no_ltp",
        )

    notes.append(f"20rupees band {band} · LTP ₹{ltp:.2f}")

    if PREMIUM_BAND_LOW <= ltp <= PREMIUM_BAND_HIGH:
        center = (PREMIUM_BAND_LOW + PREMIUM_BAND_HIGH) / 2.0
        score = max(75, min(95, int(92 - abs(ltp - center) * 4)))
        notes.append(
            f"Premium in band — SL ₹{max(0.05, ltp - FIXED_SL_INR):.2f} · "
            f"target ₹{ltp + FIXED_SL_INR:.2f} (1:1)"
        )
        return True, round(ltp, 2), score, notes, None, "20rupees_in_band"

    if ltp > PREMIUM_BAND_HIGH:
        return (
            False,
            PREMIUM_BAND_HIGH,
            32,
            notes,
            f"Premium ₹{ltp:.2f} above {band} — wait for pullback into band",
            "20rupees_high",
        )

    return (
        False,
        PREMIUM_BAND_LOW,
        28,
        notes,
        f"Premium ₹{ltp:.2f} below {band} — wait for price to reach band",
        "20rupees_low",
    )


def _analyze_bb_5m(
    spot: float,
    kind: str,
    intra: Dict[str, Any],
) -> Tuple[bool, Optional[float], int, List[str], Optional[str], str]:
    """5m BB entry: CE at lower/middle, PE at upper/middle; block extension."""
    mid = intra.get("bb_middle")
    upper = intra.get("bb_upper")
    lower = intra.get("bb_lower")
    notes: List[str] = []
    if mid is None or upper is None or lower is None:
        return (
            False,
            None,
            0,
            notes,
            "5m Bollinger Bands not ready — need 20×5m bars on the option contract",
            "",
        )

    px = contract_price_for_bb(spot, intra)
    bb = bollinger_zone(px, float(mid), float(upper), float(lower), kind)
    zone = bb["zone"]
    style = f"bb5m_{zone}"

    bb_src = (intra.get("indicator_sources") or {}).get("bb_middle", "")
    sym = intra.get("bb_on_contract") or ""
    if bb_src:
        notes.append(f"BB source: {bb_src}" + (f" · {sym}" if sym else ""))

    notes.append(
        f"5m BB (contract) L {lower:.2f} M {mid:.2f} U {upper:.2f} · LTP {px:.2f} · {zone}"
    )

    if bb["extended"]:
        return False, float(bb["trigger"]), 28, notes, bb["wait_msg"], style

    if bb["preferred"]:
        notes.append("BB preferred touch — entry confirmed")
        return True, float(bb["trigger"]), 88, notes, None, style

    if zone == "between":
        notes.append("Between bands — patient entry toward middle band")
        return True, float(mid), 58, notes, None, style

    return False, float(bb["trigger"]), 38, notes, bb["wait_msg"], style


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
    Decide if 5m BB entry is confirmed and compute LIMIT price (patient, not ask chase).
    """
    kind = (option_kind or "CE").upper()
    sid = strategy_id or TWENTY_RUPEES_ID
    bid, ask, ltp, mid, spread_pct = _book(quote)
    ref_ltp = ltp or mid or ask or bid

    if sid == TWENTY_RUPEES_ID:
        ready, trigger, score, notes, block, tag = _analyze_20rupees(quote, intra, prev_close)
    elif sid == "bb_5m_mean_reversion":
        ready, trigger, score, notes, block, tag = _analyze_bb_5m(spot, kind, intra)
    else:
        ready, trigger, score, notes, block, tag = _analyze_bb_5m(spot, kind, intra)

    fair = _structure_fair_premium(
        ref_ltp, spot, trigger if trigger is not None else spot, kind, delta
    )
    limit, limit_style = _patient_buy_limit(quote, fair)
    if sid == TWENTY_RUPEES_ID:
        style = f"{limit_style}_{tag}" if ready and tag else (tag or limit_style)
    elif ready:
        style = f"{limit_style}_{tag}" if tag else limit_style
    else:
        style = tag or "blocked_wait"

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
        entry_style=style if ready else (tag or "blocked_wait"),
        spot_trigger=round(trigger, 2) if trigger is not None else None,
        confirmation_score=score,
        notes=notes,
        block_reason=block,
    )
