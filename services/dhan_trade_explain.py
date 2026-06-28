"""Human-readable entry/exit explanations for Dhan trade charts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.entry_quality import (
    day_direction_kind,
    entry_bar_quality_ok,
    entry_day_aligned_ok,
    entry_intraday_context_ok,
    entry_momentum_required,
    entry_scan_warmup_minutes,
    exit_model,
)
from services.sensex_constants import sensex_premium_in_band


def _time_only(datetime_ist: str) -> str:
    if not datetime_ist:
        return ""
    parts = str(datetime_ist).split(" ")
    return parts[1][:5] if len(parts) > 1 else datetime_ist[11:16]


def _bar_minutes(datetime_ist: str) -> int:
    hm = _time_only(datetime_ist)
    if not hm or ":" not in hm:
        return 0
    h, m = hm.split(":", 1)
    return int(h) * 60 + int(m)


def _find_row(rows: List[Dict[str, Any]], bar_index: Optional[int], datetime_ist: str = "") -> Optional[Dict[str, Any]]:
    if bar_index is not None:
        for row in rows:
            if int(row.get("bar_index", -1)) == int(bar_index):
                return row
        if 0 <= int(bar_index) < len(rows):
            return rows[int(bar_index)]
    if datetime_ist:
        for row in rows:
            if str(row.get("datetime_ist") or "").endswith(_time_only(datetime_ist)):
                return row
    return None


def _eval_entry_bar(
    row: Dict[str, Any],
    prev_row: Optional[Dict[str, Any]],
    rows_upto: List[Dict[str, Any]],
    *,
    kind: str,
    band: tuple[float, float],
    index_open: float,
    prev_close: float,
    scan_start_min: int,
    segment: str,
) -> tuple[bool, List[str], Optional[str]]:
    """Return (ok, passed_reasons, first_fail_reason)."""
    band_lo, band_hi = band
    close = float(row.get("close") or 0)
    passed: List[str] = []
    hm = _time_only(str(row.get("datetime_ist") or ""))
    bar_min = _bar_minutes(str(row.get("datetime_ist") or ""))

    if not sensex_premium_in_band(close, band_lo, band_hi):
        return False, [], f"Close ₹{close:.2f} outside entry band ₹{band_lo}–₹{band_hi}"

    passed.append(f"Close ₹{close:.2f} inside entry band ₹{band_lo}–₹{band_hi}")

    warmup = entry_scan_warmup_minutes()
    if warmup > 0 and bar_min < scan_start_min + warmup:
        return False, passed, f"Scan warmup — entries blocked until {warmup} min after scan start"

    spot = float(row.get("spot") or 0)
    spot_prev = float(prev_row.get("spot") or 0) if prev_row else 0.0
    effective_kind = kind.upper()
    if kind.upper() == "AUTO":
        effective_kind = day_direction_kind(index_open, spot) or ""
        if not effective_kind:
            return False, passed, "AUTO: index flat vs day open — no leg"
        day_pts = spot - index_open if index_open > 0 else 0
        passed.append(f"AUTO → {effective_kind} (spot {spot:.0f}, day move {day_pts:+.0f} pts vs open)")

    if not entry_day_aligned_ok(kind=effective_kind, index_open=index_open, spot=spot):
        return False, passed, f"{effective_kind} conflicts with day direction (spot vs index open)"

    if entry_day_aligned_ok(kind=effective_kind, index_open=index_open, spot=spot) and index_open > 0:
        passed.append(f"Day-aligned {effective_kind} (index {'up' if spot > index_open else 'down'} from open)")

    session_low = min(float(r.get("spot") or spot) for r in rows_upto) if rows_upto else spot
    session_high = max(float(r.get("spot") or spot) for r in rows_upto) if rows_upto else spot
    ok_ctx, why_ctx = entry_intraday_context_ok(
        kind=effective_kind,
        index_open=index_open,
        spot=spot,
        spot_prev=spot_prev,
        bar_minutes=bar_min,
        scan_start_minutes=scan_start_min,
        session_low_so_far=session_low,
        session_high_so_far=session_high,
        segment=segment,
    )
    if not ok_ctx:
        labels = {
            "scan_warmup": "Scan warmup period",
            "weak_day_trend": "Day move too small for this leg",
            "chase_exhaustion": "Chase filter — extended day, weak bar follow-through",
            "capitulation_bar": "Capitulation bar — move too sharp vs day trend",
            "bounce_from_low": "PE bounce-from-low filter",
        }
        return False, passed, labels.get(why_ctx, why_ctx or "Intraday context filter")

    if why_ctx == "":
        passed.append("Passed intraday filters (day move, chase, capitulation)")

    ok_bar, why_bar = entry_bar_quality_ok(
        kind=effective_kind,
        bar_open=float(row.get("open") or 0),
        bar_close=close,
        spot=spot,
        prev_close=prev_close,
        spot_prev=spot_prev,
    )
    if not ok_bar:
        labels = {
            "momentum": "Option bar not bullish (close must be > open)" if effective_kind == "CE" else "Option bar not bearish",
            "index_momentum": "Index did not move with leg on this bar",
            "direction": "Index vs prior close misaligned",
        }
        return False, passed, labels.get(why_bar, why_bar or "Bar quality filter")

    if entry_momentum_required():
        o, c = float(row.get("open") or 0), close
        passed.append(f"Momentum bar: close {'>' if c > o else '≤'} open (₹{o:.2f} → ₹{c:.2f})")
    if spot_prev > 0:
        passed.append(f"Index {spot:.0f} vs prior bar {spot_prev:.0f} ({spot - spot_prev:+.0f} pts)")

    return True, passed, None


def explain_trade_on_rows(
    rows: List[Dict[str, Any]],
    trade: Dict[str, Any],
    *,
    band: tuple[float, float],
    sl_inr: float,
    scan_window: Optional[Dict[str, str]],
    index_open: float = 0.0,
    segment: str = "sensex",
    kind: str = "CE",
    strike_source: str = "",
) -> Dict[str, Any]:
    """Build entry/exit decision narrative from OHLC rows + trade overlay."""
    scan_start = scan_window.get("start", "14:00") if scan_window else "14:00"
    scan_end = scan_window.get("end", "14:45") if scan_window else "14:45"
    scan_start_min = int(scan_start.split(":")[0]) * 60 + int(scan_start.split(":")[1])

    entry_row = _find_row(
        rows,
        trade.get("entry_bar_index"),
        str(trade.get("entry_datetime_ist") or ""),
    )
    exit_row = _find_row(
        rows,
        trade.get("exit_bar_index"),
        str(trade.get("exit_datetime_ist") or ""),
    )

    entry = float(trade.get("entry") or 0)
    exit_px = float(trade.get("exit") or 0)
    target = float(trade.get("target") or 0)
    r_unit = max(0.05, entry - sl_inr) if entry > sl_inr else 0
    if not target and entry and r_unit:
        target = round(entry + r_unit, 2)

    out: Dict[str, Any] = {"entry": None, "exit": None}

    # --- Entry explanation ---
    rejected: List[Dict[str, str]] = []
    if entry_row:
        entry_idx = rows.index(entry_row)
        prev_row = rows[entry_idx - 1] if entry_idx > 0 else None
        for i, row in enumerate(rows):
            if row is entry_row:
                break
            hm = _time_only(str(row.get("datetime_ist") or ""))
            if hm < scan_start or hm >= scan_end:
                continue
            close = float(row.get("close") or 0)
            if not sensex_premium_in_band(close, band[0], band[1]):
                continue
            prev = rows[i - 1] if i > 0 else None
            ok, _, fail = _eval_entry_bar(
                row,
                prev,
                rows[: i + 1],
                kind=kind,
                band=band,
                index_open=index_open,
                prev_close=index_open,
                scan_start_min=scan_start_min,
                segment=segment,
            )
            if not ok and fail:
                rejected.append({"time": hm, "close": f"{close:.2f}", "why": fail})

        ok_entry, passed, fail = _eval_entry_bar(
            entry_row,
            prev_row,
            rows[: entry_idx + 1],
            kind=kind,
            band=band,
            index_open=index_open,
            prev_close=index_open,
            scan_start_min=scan_start_min,
            segment=segment,
        )
        entry_reasons = list(passed)
        if strike_source:
            entry_reasons.append(f"Contract: {strike_source} strike {entry_row.get('strike')}")
        entry_reasons.append(f"Entry price = bar close ₹{entry:.2f} (filled at band close)")
        entry_reasons.append(f"First qualifying bar in scan window after filters")

        out["entry"] = {
            "time": _time_only(str(entry_row.get("datetime_ist") or "")),
            "price": entry,
            "passed": entry_reasons,
            "rejected_earlier": rejected[:6],
            "summary": (
                f"Entered at { _time_only(str(entry_row.get('datetime_ist') or '')) } because this was the "
                f"first scan-window bar where premium closed in ₹{band[0]}–{band[1]} and all entry filters passed."
            ),
        }

    # --- Exit explanation ---
    if exit_row and entry > 0:
        exit_reason = str(trade.get("exit_reason") or "")
        hm = _time_only(str(exit_row.get("datetime_ist") or ""))
        high = float(exit_row.get("high") or 0)
        low = float(exit_row.get("low") or 0)
        close = float(exit_row.get("close") or 0)
        reasons: List[str] = []

        if exit_reason == "target_t1":
            reasons.append(f"Exit model: {exit_model()} — full exit at 1R target (no trail)")
            reasons.append(f"1R = entry ₹{entry:.2f} − SL ₹{sl_inr:.2f} = ₹{r_unit:.2f} → target ₹{target:.2f}")
            if high >= target:
                reasons.append(f"Bar high ₹{high:.2f} touched/exceeded target ₹{target:.2f}")
            reasons.append(f"Exit booked at ₹{exit_px:.2f} on this bar")
        elif exit_reason == "stop_loss":
            reasons.append(f"Bar low ₹{low:.2f} hit fixed SL ₹{sl_inr:.2f}")
        elif exit_reason.startswith("trail"):
            reasons.append(f"Stepped trail stop triggered ({exit_reason})")
        elif exit_reason in ("eod", "eod_trail"):
            reasons.append("Position closed at end of session")
        else:
            reasons.append(f"Exit reason: {exit_reason}")

        out["exit"] = {
            "time": hm,
            "price": exit_px,
            "reason": exit_reason,
            "passed": reasons,
            "summary": (
                f"Exited at {hm} @ ₹{exit_px:.2f} — {exit_reason.replace('_', ' ')}"
                + (f" (target was ₹{target:.2f})" if exit_reason == "target_t1" else "")
            ),
        }

    return out
