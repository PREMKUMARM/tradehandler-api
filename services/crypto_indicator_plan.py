"""BTCUSDT perp trade plan from live Binance data — 5m Bollinger Bands mean reversion."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from services.crypto_config import (
    DEFAULT_LEVERAGE,
    DEFAULT_QUANTITY_BTC,
    DEFAULT_REWARD_RATIO,
    DEFAULT_RISK_PCT,
    EXCHANGE,
    SYMBOL,
    compute_crypto_quantity,
)
from services.crypto_trail import ensure_min_rr_tp, get_crypto_trail_config
from services.crypto_ta import close_back_inside_long, close_back_inside_short
from services.crypto_live_indicators import recalculate_from_ticker
from services.kite_live_indicators import bollinger_zone
from utils.binance_order_utils import get_usdt_balance, round_price, round_quantity


def _adx_max() -> float:
    try:
        return max(10.0, min(40.0, float(os.getenv("CRYPTO_BB_ADX_MAX", "26") or 26)))
    except (TypeError, ValueError):
        return 26.0


def _rsi_long_max() -> float:
    try:
        return max(20.0, min(45.0, float(os.getenv("CRYPTO_BB_RSI_LONG_MAX", "38") or 38)))
    except (TypeError, ValueError):
        return 38.0


def _rsi_short_min() -> float:
    try:
        return max(55.0, min(80.0, float(os.getenv("CRYPTO_BB_RSI_SHORT_MIN", "62") or 62)))
    except (TypeError, ValueError):
        return 62.0


def _bb_bw_min_pct() -> float:
    try:
        return max(0.05, float(os.getenv("CRYPTO_BB_BW_MIN_PCT", "0.15") or 0.15))
    except (TypeError, ValueError):
        return 0.15


def _bb_bw_max_pct() -> float:
    try:
        return max(0.5, float(os.getenv("CRYPTO_BB_BW_MAX_PCT", "2.5") or 2.5))
    except (TypeError, ValueError):
        return 2.5


def _atr_ratio_max() -> float:
    try:
        return max(1.0, min(2.5, float(os.getenv("CRYPTO_ATR_RATIO_MAX", "1.5") or 1.5)))
    except (TypeError, ValueError):
        return 1.5


def _atr_sl_mult() -> float:
    try:
        return max(0.8, min(2.0, float(os.getenv("CRYPTO_BB_ATR_SL_MULT", "1.5") or 1.5)))
    except (TypeError, ValueError):
        return 1.5


def _sl_band_mult() -> float:
    """SL placed this many × full BB widths beyond the entry band edge (default 0.45)."""
    try:
        return max(0.25, min(0.8, float(os.getenv("CRYPTO_BB_SL_BAND_MULT", "0.45") or 0.45)))
    except (TypeError, ValueError):
        return 0.45


def _tp_stretch_mult() -> float:
    """TP beyond middle toward opposite band (0 = middle only, 0.5 = halfway to far band)."""
    try:
        return max(0.0, min(0.75, float(os.getenv("CRYPTO_BB_TP_STRETCH", "0.15") or 0.15)))
    except (TypeError, ValueError):
        return 0.35


def _min_tp_reward_usdt() -> float:
    try:
        return max(0.5, float(os.getenv("CRYPTO_MIN_TP_REWARD_USDT", "1.0") or 1.0))
    except (TypeError, ValueError):
        return 1.0


def _min_reward_risk_ratio() -> float:
    try:
        return max(1.0, float(os.getenv("CRYPTO_MIN_RR", "1.5") or 1.5))
    except (TypeError, ValueError):
        return 1.5


def passes_regime_filters(live: Dict[str, Any]) -> Tuple[bool, str]:
    """ADX ranging + bandwidth + ATR volatility gate (research: skip trends & squeezes)."""
    adx = live.get("adx14")
    bw = live.get("bb_bandwidth_pct")
    atr_ratio = live.get("atr_ratio")
    if adx is None or bw is None:
        return False, "ADX/bandwidth not ready — need more 5m bars"
    if float(adx) > _adx_max():
        return False, f"ADX {float(adx):.1f} > {_adx_max():.0f} — trending market, skip mean reversion"
    if float(bw) < _bb_bw_min_pct():
        return False, f"BB squeeze ({float(bw):.2f}% width) — wait for expansion"
    if float(bw) > _bb_bw_max_pct():
        return False, f"BB too wide ({float(bw):.2f}%) — volatility too high"
    if atr_ratio is not None and float(atr_ratio) > _atr_ratio_max():
        return False, f"ATR ratio {float(atr_ratio):.2f} — volatility expanding, wait"
    return True, f"Regime OK · ADX {float(adx):.1f} · BB width {float(bw):.2f}%"


def passes_rsi_filter(side: str, live: Dict[str, Any]) -> Tuple[bool, str]:
    rsi = live.get("rsi14")
    if rsi is None:
        return False, "RSI not ready"
    rsi_f = float(rsi)
    side_u = str(side or "").upper()
    if side_u == "LONG":
        cap = _rsi_long_max()
        if rsi_f <= cap:
            return True, f"RSI {rsi_f:.1f} oversold (≤ {cap:.0f})"
        return False, f"RSI {rsi_f:.1f} not oversold enough for LONG (need ≤ {cap:.0f})"
    cap = _rsi_short_min()
    if rsi_f >= cap:
        return True, f"RSI {rsi_f:.1f} overbought (≥ {cap:.0f})"
    return False, f"RSI {rsi_f:.1f} not overbought enough for SHORT (need ≥ {cap:.0f})"


def passes_trend_slope(side: str, live: Dict[str, Any]) -> Tuple[bool, str]:
    slope = live.get("bb_middle_slope")
    if slope is None:
        return True, "Middle-band slope neutral"
    s = float(slope)
    side_u = str(side or "").upper()
    if side_u == "LONG" and s < -50:
        return False, f"BB middle falling ({s:+.0f}) — avoid LONG fade"
    if side_u == "SHORT" and s > 50:
        return False, f"BB middle rising ({s:+.0f}) — avoid SHORT fade"
    return True, f"BB middle slope {s:+.0f} OK for {side_u}"


def passes_volume_filter(live: Dict[str, Any]) -> Tuple[bool, str]:
    """Pierce bar should not be a high-volume breakout (mean reversion on exhaustion)."""
    prev = live.get("prev_5m_bar")
    vol_avg = live.get("volume_avg20")
    if not isinstance(prev, dict) or vol_avg is None or float(vol_avg) <= 0:
        return True, "Volume filter skipped"
    prev_vol = float(prev.get("volume") or 0)
    ratio = prev_vol / float(vol_avg)
    cap = float(os.getenv("CRYPTO_BB_PIERCE_VOL_MAX", "1.6") or 1.6)
    if ratio <= cap:
        return True, f"Pierce vol {ratio:.2f}× avg (≤ {cap:.2f}×)"
    return False, f"Pierce vol spike {ratio:.2f}× avg — likely breakout not fade"


def _signal_bars(live: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    prev = live.get("prev_5m_bar")
    sig = live.get("last_5m_bar")
    if isinstance(prev, dict) and isinstance(sig, dict):
        return prev, sig
    return None, None


def estimate_gross_reward_usdt(
    entry: float, tp: float, side: str, qty: float
) -> float:
    side_u = str(side or "").upper()
    if side_u == "LONG":
        return max(0.0, (tp - entry) * qty)
    return max(0.0, (entry - tp) * qty)


def reward_risk_ratio(entry: float, sl: float, tp: float, side: str) -> float:
    side_u = str(side or "").upper()
    if side_u == "LONG":
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp
    if risk <= 0 or reward <= 0:
        return 0.0
    return reward / risk


def passes_min_tp_reward(
    entry: float, tp: float, side: str, qty: float
) -> Tuple[bool, str]:
    gross = estimate_gross_reward_usdt(entry, tp, side, qty)
    need = _min_tp_reward_usdt()
    if gross + 1e-9 >= need:
        return True, f"TP reward ~${gross:.2f} (min ${need:.2f})"
    return False, f"TP reward ${gross:.2f} below min ${need:.2f} after fees"


def passes_min_rr(entry: float, sl: float, tp: float, side: str) -> Tuple[bool, str]:
    rr = reward_risk_ratio(entry, sl, tp, side)
    need = _min_reward_risk_ratio()
    if rr + 1e-9 >= need:
        return True, f"R:R {rr:.2f} (min {need:.2f})"
    return False, f"R:R {rr:.2f} below min {need:.2f}"


def bb_reentry_reset_zone(live: Dict[str, Any]) -> bool:
    """True when price has reset toward BB middle (ready for next extreme-band entry)."""
    spot = float(live.get("btc_spot") or 0)
    mid = live.get("bb_middle")
    upper = live.get("bb_upper")
    lower = live.get("bb_lower")
    if mid is None or upper is None or lower is None or spot <= 0:
        return True
    bb = bollinger_zone(spot, float(mid), float(upper), float(lower), "CE")
    return str(bb.get("zone") or "") in ("middle", "between")


def _required_entry_zone(side: str) -> str:
    return "lower" if str(side or "").upper() == "LONG" else "upper"


def _daily_filter_enabled() -> bool:
    return os.getenv("CRYPTO_DAILY_CANDLE_FILTER", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def daily_candle_bias(live: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    Forming 1d candle bias: green day → LONG only, red day → SHORT only.
    Uses spot vs today's daily open (Binance UTC 1d kline).
    """
    spot = float(live.get("btc_spot") or 0)
    day_open = live.get("day_open")
    if day_open is None or spot <= 0:
        return None, "Daily candle not ready"
    day_open = float(day_open)
    if spot > day_open:
        return (
            "LONG",
            f"1d candle green (${spot:,.0f} > open ${day_open:,.0f}) — LONG/BUY only",
        )
    if spot < day_open:
        return (
            "SHORT",
            f"1d candle red (${spot:,.0f} < open ${day_open:,.0f}) — SHORT/SELL only",
        )
    return None, f"1d candle flat at open ${day_open:,.0f} — wait"


def side_aligns_with_daily_candle(side: str, live: Dict[str, Any]) -> Tuple[bool, str]:
    """Block entries that fight the forming daily candle direction."""
    if not _daily_filter_enabled():
        return True, "Daily candle filter off"
    allowed, msg = daily_candle_bias(live)
    if allowed is None:
        return False, msg
    side_u = str(side or "").upper()
    if side_u == allowed:
        return True, msg
    return False, f"{side_u} blocked — {msg}"


def _resolve_side(direction: str, live: Dict[str, Any]) -> str:
    d = (direction or "AUTO").upper()
    if d in ("LONG", "SHORT"):
        return d

    prev_bar, sig_bar = _signal_bars(live)
    mid = live.get("bb_middle")
    upper = live.get("bb_upper")
    lower = live.get("bb_lower")
    if prev_bar and sig_bar and mid and upper and lower:
        if close_back_inside_long(prev_bar, sig_bar, float(lower)):
            return "LONG"
        if close_back_inside_short(prev_bar, sig_bar, float(upper)):
            return "SHORT"

    spot = float(live.get("signal_spot") or live.get("btc_spot") or 0)
    if mid is not None and upper is not None and lower is not None and spot > 0:
        bb_ce = bollinger_zone(spot, float(mid), float(upper), float(lower), "CE")
        zone = str(bb_ce.get("zone") or "")
        if zone == "lower":
            return "LONG"
        if zone == "upper":
            return "SHORT"

    prev = float(live.get("prev_close") or spot)
    ema9 = float(live.get("ema9") or spot)
    if spot >= ema9 and spot >= prev:
        return "LONG"
    return "SHORT"


def _bb_entry_analysis(
    side: str, spot: float, live: Dict[str, Any]
) -> Tuple[bool, Optional[str], int, str, float, Optional[str]]:
    """
    BB mean reversion v3: close-back-inside + RSI + ADX regime (research-backed).
    Enter at close of confirmation bar after band pierce + re-entry.
    """
    mid = live.get("bb_middle")
    upper = live.get("bb_upper")
    lower = live.get("bb_lower")
    if mid is None or upper is None or lower is None:
        return (
            False,
            "5m Bollinger Bands not ready — need 20×5m bars on BTCUSDT",
            0,
            "blocked_wait",
            spot,
            None,
        )

    mid_f, upper_f, lower_f = float(mid), float(upper), float(lower)
    prev_bar, sig_bar = _signal_bars(live)
    if not prev_bar or not sig_bar:
        return False, "Need 2 completed 5m bars for reversal confirm", 0, "blocked_wait", spot, None

    side_u = str(side or "").upper()
    if side_u == "LONG":
        if not close_back_inside_long(prev_bar, sig_bar, lower_f):
            return (
                False,
                "LONG needs prev bar pierce lower BB + close back inside",
                35,
                "blocked_no_reversal",
                spot,
                "between",
            )
        bb_zone = "lower"
    else:
        if not close_back_inside_short(prev_bar, sig_bar, upper_f):
            return (
                False,
                "SHORT needs prev bar pierce upper BB + close back inside",
                35,
                "blocked_no_reversal",
                spot,
                "between",
            )
        bb_zone = "upper"

    ok_reg, reg_msg = passes_regime_filters(live)
    if not ok_reg:
        return False, reg_msg, 32, "blocked_regime", spot, bb_zone

    ok_rsi, rsi_msg = passes_rsi_filter(side_u, live)
    if not ok_rsi:
        return False, rsi_msg, 38, "blocked_rsi", spot, bb_zone

    ok_slope, slope_msg = passes_trend_slope(side_u, live)
    if not ok_slope:
        return False, slope_msg, 36, "blocked_slope", spot, bb_zone

    ok_vol, vol_msg = passes_volume_filter(live)
    if not ok_vol:
        return False, vol_msg, 34, "blocked_volume", spot, bb_zone

    entry_px = round_price(SYMBOL, float(sig_bar["close"]))
    score = 72
    if ok_reg:
        score += 6
    if ok_rsi:
        score += 6
    if ok_slope:
        score += 4
    if ok_vol:
        score += 4
    return True, None, min(score, 94), f"bb5m_{bb_zone}_reversal", entry_px, bb_zone


def _bb_exit_levels(
    side: str,
    entry_price: float,
    live: Dict[str, Any],
    *,
    bb_zone: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Extreme-band exits: SL beyond entry band; TP at middle (mean reversion) or scaled toward far band.
    """
    upper = float(live.get("bb_upper") or entry_price)
    lower = float(live.get("bb_lower") or entry_price)
    middle = float(live.get("bb_middle") or entry_price)
    width = max(upper - lower, entry_price * 0.002)
    sl_mult = _sl_band_mult()
    tp_stretch = _tp_stretch_mult()
    atr_mult = _atr_sl_mult()
    atr = float(live.get("atr14") or 0)
    zone = (bb_zone or "").lower()
    side_u = str(side or "").upper()

    if side_u == "LONG" and zone == "lower":
        sl_bb = lower - width * sl_mult
        sl_atr = entry_price - atr * atr_mult if atr else sl_bb
        # Tighter stop (closer to entry) — avoids oversized losses on failed reversals
        sl = round_price(SYMBOL, max(sl_bb, sl_atr))
        tp = round_price(SYMBOL, middle + (upper - middle) * tp_stretch)
    elif side_u == "SHORT" and zone == "upper":
        sl_bb = upper + width * sl_mult
        sl_atr = entry_price + atr * atr_mult if atr else sl_bb
        sl = round_price(SYMBOL, min(sl_bb, sl_atr))
        tp = round_price(SYMBOL, middle - (middle - lower) * tp_stretch)
    elif side_u == "LONG":
        sl = round_price(SYMBOL, lower - width * sl_mult)
        tp = round_price(SYMBOL, middle + (upper - middle) * tp_stretch)
    else:
        sl = round_price(SYMBOL, upper + width * sl_mult)
        tp = round_price(SYMBOL, middle - (middle - lower) * tp_stretch)

    min_rr = _min_reward_risk_ratio()
    tp = ensure_min_rr_tp(entry_price, sl, tp, side_u, min_rr=min_rr, symbol=SYMBOL)

    return entry_price, sl, tp


def refresh_exits_at_fill(plan: Dict[str, Any], *, fill_price: float) -> Dict[str, Any]:
    """Recompute SL/TP from live BB at fill — do not use stale levels from signal time."""
    live = recalculate_from_ticker()
    side = str(plan.get("side") or "LONG").upper()
    entry = float(fill_price or plan.get("entry_limit_price") or 0)
    if entry <= 0 or not live.get("connected"):
        return plan
    _, sl, tp = _bb_exit_levels(
        side,
        entry,
        live,
        bb_zone=str(plan.get("bb_zone") or ""),
    )
    out = dict(plan)
    out["entry_limit_price"] = round_price(SYMBOL, entry)
    out["entry_premium"] = out["entry_limit_price"]
    out["stop_loss_premium"] = sl
    out["target_premium"] = tp
    out["spot_stop_loss"] = sl
    out["spot_target"] = tp
    out["nifty_spot"] = entry
    out["indicators"] = {**(plan.get("indicators") or {}), **live}
    return out


def build_trade_plan(
    *,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    quantity_btc: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    live = recalculate_from_ticker()
    messages: List[str] = []
    if not live.get("connected"):
        return {}, ["Binance not connected — check API keys"]

    side = _resolve_side(direction, live)
    spot = float(live.get("signal_spot") or live.get("btc_spot") or 0)
    entry_ready, block, score, entry_style, entry_limit, bb_zone = _bb_entry_analysis(side, spot, live)

    if entry_ready:
        ok_day, day_msg = side_aligns_with_daily_candle(side, live)
        if not ok_day:
            entry_ready = False
            block = day_msg
            score = min(int(score or 0), 40)
        else:
            messages.append(day_msg)

    spot_entry, stop_loss_price, target_price = _bb_exit_levels(
        side,
        float(entry_limit or spot),
        live,
        bb_zone=bb_zone,
    )

    risk_pct = float(risk_percentage or DEFAULT_RISK_PCT)
    rr = float(reward_percentage or DEFAULT_REWARD_RATIO) if reward_percentage else DEFAULT_REWARD_RATIO

    if not entry_ready:
        entry_limit = round_price(SYMBOL, spot * 0.9995 if side == "LONG" else spot * 1.0005)

    sl_dist = abs(spot_entry - stop_loss_price)
    risk_inr = 0.0

    if quantity_btc is None:
        from services.paper_trading import is_paper_mode_for_segment

        paper = is_paper_mode_for_segment("crypto")
        if paper:
            from services.paper_funds import get_available_balance

            usdt = float(get_available_balance("crypto") or 0)
            messages.append(f"Paper fund: ${usdt:,.2f} USDT available")
        else:
            try:
                usdt = float(get_usdt_balance() or 0)
            except Exception:
                usdt = 0.0
        qty, size_msgs = compute_crypto_quantity(usdt, spot, paper=paper, symbol=SYMBOL)
        messages.extend(size_msgs)
        if qty <= 0 and not paper:
            block = block or (size_msgs[-1] if size_msgs else "Insufficient margin for min lot")
            entry_ready = False
            score = min(int(score or 0), 38)
    else:
        from services.paper_trading import is_paper_mode_for_segment

        qty = round_quantity(SYMBOL, float(quantity_btc))
        if not is_paper_mode_for_segment("crypto"):
            try:
                usdt = float(get_usdt_balance() or 0)
            except Exception:
                usdt = 0.0
            need_margin = (qty * spot) / max(1, DEFAULT_LEVERAGE)
            if usdt + 0.01 < need_margin:
                block = (
                    block
                    or f"Insufficient margin for {qty} BTC: need ${need_margin:,.2f} @ {DEFAULT_LEVERAGE}x, "
                    f"have ${usdt:,.2f} USDT"
                )
                entry_ready = False
                score = min(int(score or 0), 38)
                messages.append(block)
    notional = qty * spot
    risk_inr = notional * (risk_pct / 100.0) * DEFAULT_LEVERAGE

    if entry_ready and qty > 0:
        ok_rw, rw_msg = passes_min_tp_reward(float(entry_limit or spot), target_price, side, qty)
        ok_rr, rr_msg = passes_min_rr(float(entry_limit or spot), stop_loss_price, target_price, side)
        if not ok_rw:
            entry_ready = False
            block = rw_msg
            score = min(int(score or 0), 40)
        elif not ok_rr:
            entry_ready = False
            block = rr_msg
            score = min(int(score or 0), 40)
        else:
            messages.extend([rw_msg, rr_msg])

    bb_lo = live.get("bb_lower")
    bb_mid = live.get("bb_middle")
    bb_up = live.get("bb_upper")
    if bb_lo is not None and bb_mid is not None and bb_up is not None:
        messages.append(
            f"5m BB L ${float(bb_lo):,.2f} M ${float(bb_mid):,.2f} U ${float(bb_up):,.2f} · zone {bb_zone or '—'}"
        )
    if entry_ready and sl_dist > 0:
        rr = reward_risk_ratio(float(entry_limit or spot), stop_loss_price, target_price, side)
        messages.append(
            f"SL ${stop_loss_price:,.2f} ({sl_dist:,.0f} pts) · "
            f"TP ${target_price:,.2f} (min R:R {rr:.2f}) · "
            f"trail SL after {get_crypto_trail_config().activation_r:.2f}R profit"
        )

    plan: Dict[str, Any] = {
        "tradingsymbol": SYMBOL,
        "exchange": EXCHANGE,
        "side": side,
        "option_type": side,
        "leverage": DEFAULT_LEVERAGE,
        "strike": 0,
        "expiry": "PERP",
        "quantity": qty,
        "num_lots": 1,
        "lot_size": qty,
        "product": "USDT-M",
        "entry_order_type": "LIMIT",
        "exit_order_type": "STOP_MARKET",
        "entry_limit_price": entry_limit,
        "entry_premium": entry_limit,
        "stop_loss_premium": stop_loss_price,
        "target_premium": target_price,
        "spot_stop_loss": stop_loss_price,
        "spot_target": target_price,
        "nifty_spot": spot_entry,
        "risk_inr": round(risk_inr, 2),
        "reward_inr": round(risk_inr * rr, 2),
        "reward_ratio": rr,
        "entry_ready": entry_ready,
        "entry_block_reason": block,
        "entry_style": entry_style,
        "entry_confirmation_score": score,
        "strategy_id": "bb_5m_mean_reversion_v3",
        "strategy_name": f"BTCUSDT {side} BB 5m reversal {DEFAULT_LEVERAGE}x",
        "bb_lower": bb_lo,
        "bb_middle": bb_mid,
        "bb_upper": bb_up,
        "bb_zone": bb_zone,
        "day_open": live.get("day_open"),
        "day_candle_green": live.get("day_candle_green"),
        "daily_bias": daily_candle_bias(live)[0],
        "trail_enabled": get_crypto_trail_config().enabled,
        "min_rr": _min_reward_risk_ratio(),
        "indicators": live,
    }
    messages.append(
        f"{SYMBOL} {side} {DEFAULT_LEVERAGE}x · LIMIT ${entry_limit:,.2f} · "
        f"SL ${stop_loss_price:,.2f} TP ${target_price:,.2f} · qty {qty} BTC"
    )
    if block:
        messages.append(block)
    return plan, messages


def refresh_plan_at_execution(plan: Dict[str, Any]) -> Dict[str, Any]:
    direction = plan.get("side") or plan.get("option_type") or "AUTO"
    fresh, _ = build_trade_plan(
        direction=str(direction),
        quantity_btc=None,
    )
    if not fresh:
        return plan
    out = dict(plan)
    out.update(fresh)
    out["indicators"] = {**(plan.get("indicators") or {}), **(fresh.get("indicators") or {})}
    return out
