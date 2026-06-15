"""
Sensex wizard — indicator-based LIMIT entry + GTT OCO exit (no MARKET orders).
"""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from agent.config import get_agent_config
from agent.tools.kite_tools import place_gtt_tool, place_order_tool
from schemas.sensex_trading import ChecklistStepStatus
from services.checklist_step_utils import parse_checklist_steps
from utils.margin_utils import parse_equity_margins
from services.sensex_strategy_analysis import analyze_fno_strategies
from utils.kite_utils import get_kite_instance, get_access_token
from utils.logger import log_error, log_info, log_warning
from services.sensex_constants import (
    is_past_sensex_entry_cutoff,
    resolve_sensex_bfo_product,
    sensex_entry_cutoff_message,
)
from services.sensex_run_params import SensexRunParams

IST = ZoneInfo("Asia/Kolkata")

STEP_TITLES = [
    "Confirm session, connectivity, and buying power",
    "Define the trade hypothesis on Sensex",
    "Check India VIX and event risk",
    "Strategy analysis",
    "Read the Sensex option chain",
    "Pick expiry with purpose",
    "Select strike using delta and liquidity",
    "Estimate premium, Greeks, and max loss",
    "Plan exit before entry",
    "Size the position to your risk rule",
    "Choose product and order type",
    "Final order ticket review",
]

# Steps that must be marked done in the wizard before place (0-indexed).
REQUIRED_MARKED_STEPS = [0, 1, 2, 9]
# Steps validated at execution time on the server during market hours.
MARKET_EXECUTION_STEPS = [4, 5, 6, 7, 8, 10, 11]


def _ist_clock() -> Tuple[int, bool]:
    now = datetime.now(IST)
    day = now.weekday()
    minutes = now.hour * 60 + now.minute
    return minutes, 1 <= day <= 5


def is_market_session_open() -> bool:
    minutes, is_weekday = _ist_clock()
    if not is_weekday:
        return False
    return 9 * 60 + 15 <= minutes < 15 * 60 + 30


def allow_offhours_sensex_place() -> bool:
    """When true, allow confirm/place for integration testing outside market hours."""
    return os.getenv("V2_ALLOW_OFFHOURS_PLACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _resolve_can_place(
    trade_plan: Optional[Dict[str, Any]],
    validation: Optional[Dict[str, Any]],
    market_open: bool,
) -> bool:
    if not trade_plan:
        return False
    if trade_plan.get("entry_ready") is False:
        return False
    if market_open:
        return bool(validation and validation.get("is_good_trade"))
    if allow_offhours_sensex_place():
        return True
    return False


def _resolve_v2_capital(margin: float = 0.0) -> float:
    from services.paper_funds import resolve_capital_for_segment

    cfg = get_agent_config()
    return resolve_capital_for_segment(
        "sensex",
        margin_fallback=margin,
        cfg_capital=float(cfg.trading_capital or 100000),
    )


def _resolve_run_params(
    run_params: Optional[Dict[str, Any]] = None,
    *,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    num_lots: Optional[int] = None,
) -> SensexRunParams:
    merged = dict(run_params or {})
    if risk_percentage is not None:
        merged.setdefault("risk_pct", risk_percentage)
    if num_lots is not None:
        merged["num_lots"] = num_lots
    return SensexRunParams.from_mapping(merged, direction=direction)


def normalize_sensex_trade_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Move flat run-parameter fields into run_params for REST/WS callers."""
    out = dict(kwargs)
    rp_fields = (
        "capital",
        "risk_pct",
        "risk_percentage",
        "sl_inr",
        "entry_band_low",
        "entry_band_high",
        "min_target_low",
        "min_target_high",
        "entry_scan_start_ist",
        "entry_scan_end_ist",
    )
    run_params: Dict[str, Any] = {}
    if isinstance(out.get("run_params"), dict):
        run_params.update(out["run_params"])
    for key in rp_fields:
        if key in out and out[key] is not None:
            run_params[key] = out.pop(key)
    if run_params:
        out["run_params"] = SensexRunParams.from_mapping(
            run_params,
            direction=str(out.get("direction") or "AUTO"),
            num_lots=out.get("num_lots"),
        ).to_dict()
    return out


def _resolve_sensex_capital(margin: float, rp: SensexRunParams) -> float:
    if rp.capital >= 10_000:
        return rp.capital
    return _resolve_v2_capital(margin)


def _check_kite_and_margin() -> Tuple[bool, float, str]:
    try:
        token = get_access_token()
        if not token or len(token) < 10:
            return False, 0.0, "Kite not connected — connect via profile menu"
        kite = get_kite_instance(skip_validation=False)
        margins = kite.margins()
        equity = margins.get("equity", {})
        available, _, _ = parse_equity_margins(equity)
        if available <= 0:
            return False, 0.0, "No available margin reported"
        return True, float(available), "Kite connected"
    except Exception as exc:
        return False, 0.0, f"Kite check failed: {exc}"


def _resolve_completed_steps(
    completed_steps: Optional[List[bool]], auto_execute: bool
) -> List[bool]:
    if auto_execute:
        return [True] * len(STEP_TITLES)
    if completed_steps and len(completed_steps) == len(STEP_TITLES):
        return list(completed_steps)
    return [False] * len(STEP_TITLES)


def fetch_live_checklist(
    direction: str,
    market_open: bool,
    risk_pct: float = 1.0,
    reward_pct: float = 2.0,
    num_lots: int = 1,
    capital: float = 0.0,
    only_steps: Optional[List[int]] = None,
    run_params: Optional[SensexRunParams] = None,
) -> Optional[Dict[str, Any]]:
    """Run checklist steps on live Kite data; None if not connected."""
    kite_ok, margin, _ = _check_kite_and_margin()
    rp = run_params or SensexRunParams.from_mapping({"risk_pct": risk_pct, "num_lots": num_lots}, direction=direction)
    if capital <= 0:
        capital = _resolve_sensex_capital(margin, rp)
    if not kite_ok or margin <= 0:
        return None
    from services.sensex_realtime_checklist import run_realtime_checklist

    return run_realtime_checklist(
        direction,
        margin,
        market_open,
        rp.risk_pct,
        reward_pct,
        rp.num_lots,
        capital,
        only_steps=only_steps,
        run_params=rp,
    )


def validate_checklist(
    completed_steps: List[bool],
    direction: str,
    market_open: bool,
    auto_execute: bool = False,
    risk_pct: float = 1.0,
    reward_pct: float = 2.0,
    num_lots: int = 1,
    capital: float = 0.0,
    run_params: Optional[SensexRunParams] = None,
) -> Tuple[List[ChecklistStepStatus], List[int], bool, Optional[Dict[str, Any]]]:
    """All steps validated against live Kite ticker + quotes when connected."""
    rp = run_params or SensexRunParams.from_mapping(
        {"risk_pct": risk_pct, "num_lots": num_lots},
        direction=direction,
    )
    live = fetch_live_checklist(
        rp.direction,
        market_open,
        rp.risk_pct,
        reward_pct,
        rp.num_lots,
        capital,
        run_params=rp,
    )

    if live:
        statuses = parse_checklist_steps(live["step_statuses"], ChecklistStepStatus)
        missing: List[int] = []
        if not auto_execute:
            for i in REQUIRED_MARKED_STEPS:
                marked = bool(completed_steps[i]) if i < len(completed_steps) else False
                if not marked:
                    missing.append(i)
                if i < len(statuses):
                    statuses[i] = statuses[i].model_copy(
                        update={"completed": statuses[i].server_ok and marked}
                    )
            for i, st in enumerate(statuses):
                if i not in REQUIRED_MARKED_STEPS:
                    statuses[i] = st.model_copy(update={"completed": st.server_ok})
        checklist_ready = live["checklist_ready"] and len(missing) == 0
        return statuses, missing, checklist_ready, live

    statuses: List[ChecklistStepStatus] = []
    missing: List[int] = []
    for i, title in enumerate(STEP_TITLES):
        marked = bool(completed_steps[i]) if i < len(completed_steps) else False
        server_ok = False
        msg = "Connect Kite for live checklist"
        if i in REQUIRED_MARKED_STEPS and not marked:
            missing.append(i)
        statuses.append(
            ChecklistStepStatus(
                index=i,
                title=title,
                completed=marked and server_ok,
                server_ok=server_ok,
                message=msg,
            )
        )
    return statuses, missing, False, None


def _step(
    index: int,
    title: str,
    server_ok: bool,
    message: str,
    output: Optional[str] = None,
) -> ChecklistStepStatus:
    return ChecklistStepStatus(
        index=index,
        title=title,
        completed=server_ok,
        server_ok=server_ok,
        message=message,
        output=output,
    )


def _validate_checklist_auto(
    direction: str,
    market_open: bool,
    plan: Optional[Dict[str, Any]] = None,
    plan_error: Optional[str] = None,
    resolved_direction: Optional[str] = None,
    margin: float = 0.0,
    strategy_analysis: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ChecklistStepStatus], List[int], bool]:
    """Build per-step outputs for auto-execute preview UI."""
    statuses: List[ChecklistStepStatus] = []
    missing: List[int] = []
    kite_ok, margin_val, kite_msg = _check_kite_and_margin()
    if margin <= 0:
        margin = margin_val
    dir_ok = direction.upper() in ("CE", "PE", "AUTO")
    opt = resolved_direction or (plan or {}).get("option_type") or direction.upper()

    for i, title in enumerate(STEP_TITLES):
        if i == 0:
            ok = kite_ok and margin > 0
            out = f"Available margin: ₹{margin:,.2f}" if ok else None
            if not ok:
                missing.append(i)
            statuses.append(_step(i, title, ok, kite_msg, out))
        elif i == 1:
            ok = dir_ok and (plan is not None or direction.upper() in ("CE", "PE"))
            if direction.upper() == "AUTO" and plan:
                msg = f"Direction AUTO → highest OI {opt}"
                out = f"Sensex {plan.get('nifty_spot')} vs prior close → {opt}"
            elif direction.upper() in ("CE", "PE"):
                msg = f"Direction: {direction.upper()}"
                out = f"Hypothesis: buy Sensex {direction.upper()}"
            else:
                msg = "Could not resolve direction"
                out = plan_error
                ok = False
                missing.append(i)
            statuses.append(_step(i, title, ok, msg, out))
        elif i == 2:
            vix = (strategy_analysis or {}).get("context", {}).get("vix")
            ok = plan is not None
            out = (
                f"VIX {vix:.1f} — sized for current IV" if vix else "VIX/event check passed"
            ) if ok else plan_error
            statuses.append(_step(i, title, ok, "VIX & event risk reviewed", out))
        elif i == 3:
            ok = strategy_analysis is not None and plan is not None
            sa = strategy_analysis or {}
            out = sa.get("output_summary") or plan_error
            msg = (
                f"Best: {sa.get('selected_name')} ({sa.get('selected_score')}/100)"
                if ok
                else "Strategy analysis failed"
            )
            statuses.append(_step(i, title, ok, msg, out))
        elif i == 4:
            ok = plan is not None
            out = (
                f"Sensex spot {plan.get('nifty_spot')} · focus ATM/near-ATM {opt} liquidity"
                if plan
                else plan_error
            )
            statuses.append(_step(i, title, ok, "Chain read (spot + OI context)", out))
        elif i == 5:
            ok = plan is not None
            out = f"Expiry: {plan.get('expiry')}" if plan else plan_error
            statuses.append(_step(i, title, ok, "Nearest liquid weekly/monthly expiry", out))
        elif i == 6:
            ok = plan is not None
            strat = (plan or {}).get("strategy_name", "")
            out = (
                f"Strike {plan.get('strike')} {opt} · ~delta 0.5 ATM · via {strat}"
                if plan
                else plan_error
            )
            statuses.append(_step(i, title, ok, "ATM strike selected", out))
        elif i == 7:
            ok = plan is not None
            out = (
                f"Entry ~₹{plan.get('entry_premium')} · max loss ~₹{plan.get('risk_inr')}"
                if plan
                else plan_error
            )
            statuses.append(_step(i, title, ok, "Premium & max loss estimated", out))
        elif i == 8:
            ok = plan is not None
            out = (
                f"SL premium ₹{plan.get('stop_loss_premium')} · Target ₹{plan.get('target_premium')} · "
                f"Sensex SL {plan.get('spot_stop_loss')} · Tgt {plan.get('spot_target')}"
                if plan
                else plan_error
            )
            statuses.append(_step(i, title, ok, "Exit plan (GTT OCO)", out))
        elif i == 9:
            ok = plan is not None
            lots = plan.get("num_lots") if plan else 0
            lot_sz = plan.get("lot_size") if plan else 75
            qty = plan.get("quantity") if plan else 0
            out = (
                f"{lots} lot(s) × {lot_sz} = {qty} qty · risk ₹{plan.get('risk_inr')}"
                if plan
                else plan_error
            )
            statuses.append(_step(i, title, ok, "Position sized to risk rule", out))
        elif i == 10:
            ok = plan is not None
            prod = resolve_sensex_bfo_product(plan) if plan else "NRML"
            out = f"{prod} · LIMIT entry + GTT OCO exit" if plan else plan_error
            statuses.append(_step(i, title, ok, "NRML + GTT (required by Zerodha)", out))
        elif i == 11:
            has_plan = plan is not None
            ok = has_plan
            sym = plan.get("tradingsymbol") if plan else "—"
            msg = (
                "Ticket ready — confirm to place"
                if has_plan and market_open
                else ("Market closed — review only" if has_plan else "Plan incomplete")
            )
            out = f"BUY {plan.get('quantity')} {sym} @ ~₹{plan.get('entry_premium')}" if plan else plan_error
            statuses.append(_step(i, title, ok, msg, out))

    if not kite_ok or margin <= 0:
        if 0 not in missing:
            missing.append(0)
    checklist_ready = plan is not None and dir_ok and 0 not in missing
    return statuses, missing, checklist_ready


def _resolve_direction(direction: str, nifty_ltp: float, prev_close: float) -> str:
    d = (direction or "AUTO").upper()
    if d in ("CE", "PE"):
        return d
    if prev_close > 0:
        return "CE" if nifty_ltp >= prev_close else "PE"
    return "CE"


def build_trade_plan(
    direction: str,
    risk_pct: float,
    reward_pct: float,
    num_lots: int,
    capital: float,
    strategy_analysis: Optional[Dict[str, Any]] = None,
    run_params: Optional[SensexRunParams] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Plan from live indicators (OR/PDH/PDL/EMA/VIX) + live option quote — LIMIT entry, GTT exit."""
    from services.sensex_indicator_plan import build_indicator_trade_plan

    rp = run_params or SensexRunParams.from_mapping(
        {"risk_pct": risk_pct, "num_lots": num_lots, "capital": capital},
        direction=direction,
    )
    plan, messages = build_indicator_trade_plan(
        direction=direction,
        risk_pct=rp.risk_pct,
        reward_pct=reward_pct,
        num_lots=rp.num_lots,
        capital=capital,
        strategy_analysis=strategy_analysis,
        run_params=rp,
    )
    if plan and plan.get("indicators"):
        plan["indicators"]["risk_pct"] = risk_pct
    if strategy_analysis and plan:
        messages.insert(
            0,
            f"Strategy: {strategy_analysis.get('selected_name')} "
            f"(score {strategy_analysis.get('selected_score')})",
        )
    return plan, messages


def _validate_trade_plan(plan: Dict[str, Any], capital: float, risk_pct: float, reward_pct: float) -> Dict[str, Any]:
    entry = plan["entry_premium"]
    sl = plan["stop_loss_premium"]
    tgt = plan["target_premium"]
    qty = plan["quantity"]
    if entry <= sl:
        return {"is_good_trade": False, "error": "Stop-loss premium must be below entry for long option"}
    if tgt <= entry:
        return {"is_good_trade": False, "error": "Target premium must be above entry for long option"}
    risk_amt = (entry - sl) * qty
    reward_amt = (tgt - entry) * qty
    max_risk = capital * risk_pct / 100.0
    from services.premium_exit_policy import entry_initial_rr, entry_validation_skips_reward

    ratio = reward_pct / risk_pct if risk_pct > 0 else 2.0
    initial_rr = entry_initial_rr()
    min_reward = risk_amt * initial_rr
    reward_ok = reward_amt >= min_reward
    risk_ok = risk_amt <= max_risk
    skip_reward_gate = entry_validation_skips_reward()
    validation = {
        "is_good_trade": risk_ok and (reward_ok or skip_reward_gate),
        "risk_amount": round(risk_amt, 2),
        "reward_amount": round(reward_amt, 2),
        "max_risk_amount": round(max_risk, 2),
        "min_required_reward": round(min_reward, 2),
        "risk_within_limit": risk_ok,
        "reward_meets_requirement": reward_ok,
        "risk_pct_used": round(risk_pct, 2),
        "reward_pct_used": round(reward_pct, 2),
        "capital_used": round(capital, 2),
        "reward_risk_ratio_required": round(initial_rr, 2),
        "strategy_reward_ratio": round(ratio, 2),
        "trail_extends_reward": skip_reward_gate,
    }
    validation["failure_reasons"] = _validation_failure_reasons(validation)
    validation["summary"] = (
        "Trade passes risk/reward rules"
        if validation["is_good_trade"]
        else " · ".join(validation["failure_reasons"])
    )
    return validation


def _validation_failure_reasons(validation: Dict[str, Any]) -> List[str]:
    """Human-readable reasons when is_good_trade is false."""
    if validation.get("is_good_trade"):
        return []
    if validation.get("error"):
        return [str(validation["error"])]

    reasons: List[str] = []
    risk_amt = float(validation.get("risk_amount") or 0)
    max_risk = float(validation.get("max_risk_amount") or 0)
    reward_amt = float(validation.get("reward_amount") or 0)
    min_reward = float(validation.get("min_required_reward") or 0)
    risk_pct = float(validation.get("risk_pct_used") or 0)
    reward_pct = float(validation.get("reward_pct_used") or 0)
    capital = float(validation.get("capital_used") or 0)
    rr = float(validation.get("reward_risk_ratio_required") or 2.0)

    if not validation.get("risk_within_limit"):
        reasons.append(
            f"Max loss ₹{risk_amt:,.0f} exceeds your {risk_pct:g}% risk cap "
            f"(₹{max_risk:,.0f} on ₹{capital:,.0f} margin)"
        )
        reasons.append(
            "Fix: reduce lots in Settings, pick a cheaper strike, or tighten SL premium"
        )
    if not validation.get("reward_meets_requirement") and not validation.get("trail_extends_reward"):
        reasons.append(
            f"Target profit ₹{reward_amt:,.0f} is below required ₹{min_reward:,.0f} "
            f"({rr:g}× risk for {reward_pct:g}% reward / {risk_pct:g}% risk rule)"
        )
        reasons.append("Fix: widen target premium or improve entry (lower premium / better LIMIT)")
    return reasons


def preview_trade(
    completed_steps: Optional[List[bool]] = None,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    num_lots: int = 1,
    auto_execute: bool = False,
    run_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = get_agent_config()
    rp = _resolve_run_params(
        run_params,
        direction=direction,
        risk_percentage=risk_percentage,
        num_lots=num_lots,
    )
    risk_pct = float(rp.risk_pct or cfg.risk_per_trade_pct or 1.0)
    reward_pct = float(reward_percentage or cfg.reward_per_trade_pct or 2.0)
    market_open = is_market_session_open()
    steps = _resolve_completed_steps(completed_steps, auto_execute)

    messages: List[str] = []
    _, margin, _ = _check_kite_and_margin()
    capital = _resolve_sensex_capital(margin, rp)

    trade_plan = None
    validation = None
    can_place = False
    plan_error: Optional[str] = None
    resolved_dir: Optional[str] = None

    strategy_analysis: Optional[Dict[str, Any]] = None

    live = fetch_live_checklist(
        rp.direction,
        market_open,
        risk_pct,
        reward_pct,
        rp.num_lots,
        capital,
        run_params=rp,
    )

    if auto_execute and live:
        statuses = parse_checklist_steps(live["step_statuses"], ChecklistStepStatus)
        missing = live["missing_steps"]
        checklist_ready = live["checklist_ready"]
        trade_plan = live.get("trade_plan")
        validation = live.get("validation")
        strategy_analysis = live.get("strategy_analysis")
        can_place = _resolve_can_place(trade_plan, validation, market_open)
        if trade_plan:
            messages.append(
                f"Live Sensex {live.get('nifty_spot')} via {live.get('data_source', 'quote')}"
            )
            if not validation or not validation.get("is_good_trade"):
                reasons = (validation or {}).get("failure_reasons") or []
                if reasons:
                    messages.extend(reasons)
                else:
                    messages.append("Risk/reward validation failed — adjust size or levels")
            elif not market_open and allow_offhours_sensex_place():
                messages.append(
                    "Test mode: off-hours place enabled (V2_ALLOW_OFFHOURS_PLACE)"
                )
            elif not market_open:
                messages.append("Preview only — confirm and place when market is open")
        else:
            messages.append("Could not build trade plan from live data")
    elif auto_execute:
        messages.append("Kite not connected — cannot run live checklist")
        statuses, missing, checklist_ready = [], list(range(len(STEP_TITLES))), False
    else:
        statuses, missing, checklist_ready, live = validate_checklist(
            steps,
            rp.direction,
            market_open,
            auto_execute=False,
            risk_pct=risk_pct,
            reward_pct=reward_pct,
            num_lots=rp.num_lots,
            capital=capital,
            run_params=rp,
        )
        if live:
            trade_plan = live.get("trade_plan")
            validation = live.get("validation")
            strategy_analysis = live.get("strategy_analysis")
            can_place = _resolve_can_place(trade_plan, validation, market_open)
            messages.append(
                f"Live Sensex {live.get('nifty_spot')} via {live.get('data_source', 'quote')}"
            )
        if not checklist_ready:
            messages.append(f"Complete checklist steps: {[i + 1 for i in missing]}")
        if not market_open and not allow_offhours_sensex_place():
            messages.append("Live orders only during market hours (9:15 AM–3:30 PM IST, Mon–Fri)")
        elif not market_open and allow_offhours_sensex_place():
            messages.append("Test mode: off-hours place enabled (V2_ALLOW_OFFHOURS_PLACE)")

    paper_mode = False
    try:
        from services.paper_trading import is_paper_mode_for_segment

        paper_mode = is_paper_mode_for_segment("sensex")
        if paper_mode:
            messages.append("Paper mode ON (Sensex50) — orders go to paper ledger when checklist is complete")
    except Exception:
        pass

    preview_core = {
        "can_place": can_place and not paper_mode,
        "checklist_ready": checklist_ready,
        "paper_trading_mode": paper_mode,
        "trade_plan": trade_plan,
    }
    from services.watch_execute import resolve_can_execute

    can_execute = resolve_can_execute(
        preview_core,
        trade_plan,
        offhours_allowed=allow_offhours_sensex_place(),
    )
    from services.trail_ops import get_exit_policy_summary

    exit_policy = get_exit_policy_summary(
        trade_plan.get("strategy_id") if trade_plan else None,
        quantity=int((trade_plan or {}).get("quantity") or 0) or None,
    )

    return {
        "can_place": can_place and not paper_mode,
        "can_execute": can_execute,
        "checklist_ready": checklist_ready,
        "missing_steps": missing,
        "step_statuses": [s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in statuses],
        "trade_plan": trade_plan,
        "validation": validation,
        "exit_policy": exit_policy,
        "messages": messages,
        "market_open": market_open,
        "allow_test_place": allow_offhours_sensex_place(),
        "paper_trading_mode": paper_mode,
        "strategy_analysis": strategy_analysis,
        "paper_funds": _paper_funds_payload(paper_mode),
    }


def _paper_funds_payload(paper_mode: bool) -> Optional[Dict[str, Any]]:
    if not paper_mode:
        return None
    try:
        from services.paper_funds import get_fund_snapshot

        return get_fund_snapshot("sensex")
    except Exception:
        return None


def get_strategy_analysis(
    direction: str = "AUTO",
    run_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Standalone strategy analysis for wizard step 4 (uses steps 1–3 context)."""
    _, margin, _ = _check_kite_and_margin()
    rp = _resolve_run_params(run_params, direction=direction)
    capital = _resolve_sensex_capital(margin, rp)
    return analyze_fno_strategies(
        direction_pref=rp.direction,
        margin=capital,
        run_params=rp,
    )


def get_checklist_analyze(
    step: int,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    num_lots: int = 1,
    run_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Realtime analysis for one checklist step and its prerequisite steps."""
    from services.sensex_realtime_checklist import step_indices_for_analysis

    cfg = get_agent_config()
    rp = _resolve_run_params(
        run_params,
        direction=direction,
        risk_percentage=risk_percentage,
        num_lots=num_lots,
    )
    risk_pct = float(rp.risk_pct or cfg.risk_per_trade_pct or 1.0)
    reward_pct = float(reward_percentage or cfg.reward_per_trade_pct or 2.0)
    market_open = is_market_session_open()
    indices = step_indices_for_analysis(step)
    _, margin, kite_msg = _check_kite_and_margin()
    capital = _resolve_sensex_capital(margin, rp)
    live = fetch_live_checklist(
        rp.direction,
        market_open,
        risk_pct,
        reward_pct,
        rp.num_lots,
        capital,
        only_steps=indices,
        run_params=rp,
    )
    if not live:
        return {
            "connected": False,
            "message": kite_msg,
            "focus_step": step,
            "analyzed_steps": indices,
            "step_statuses": [],
            "checklist_ready": False,
            "missing_steps": indices,
            "market_open": market_open,
            "allow_test_place": allow_offhours_sensex_place(),
        }
    validation = live.get("validation")
    trade_plan = live.get("trade_plan")
    from services.watch_skip_utils import dump_checklist_steps

    return {
        "connected": True,
        "message": kite_msg,
        "focus_step": step,
        "analyzed_steps": indices,
        "step_statuses": dump_checklist_steps(live["step_statuses"], ChecklistStepStatus),
        "checklist_ready": live["checklist_ready"],
        "missing_steps": live["missing_steps"],
        "trade_plan": trade_plan,
        "validation": validation,
        "strategy_analysis": live.get("strategy_analysis"),
        "can_place": _resolve_can_place(trade_plan, validation, market_open),
        "market_open": market_open,
        "allow_test_place": allow_offhours_sensex_place(),
        "nifty_spot": live.get("nifty_spot"),
        "data_source": live.get("data_source"),
    }


def get_checklist_live(
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    num_lots: int = 1,
    run_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Refresh all 12 checklist steps from live Kite data (for wizard step navigation)."""
    cfg = get_agent_config()
    rp = _resolve_run_params(
        run_params,
        direction=direction,
        risk_percentage=risk_percentage,
        num_lots=num_lots,
    )
    risk_pct = float(rp.risk_pct or cfg.risk_per_trade_pct or 1.0)
    reward_pct = float(reward_percentage or cfg.reward_per_trade_pct or 2.0)
    market_open = is_market_session_open()
    _, margin, kite_msg = _check_kite_and_margin()
    capital = _resolve_sensex_capital(margin, rp)
    live = fetch_live_checklist(
        rp.direction,
        market_open,
        risk_pct,
        reward_pct,
        rp.num_lots,
        capital,
        run_params=rp,
    )
    if not live:
        return {
            "connected": False,
            "message": kite_msg,
            "step_statuses": [],
            "checklist_ready": False,
            "missing_steps": list(range(len(STEP_TITLES))),
            "market_open": market_open,
            "allow_test_place": allow_offhours_sensex_place(),
        }
    validation = live.get("validation")
    trade_plan = live.get("trade_plan")
    from services.watch_skip_utils import dump_checklist_steps

    return {
        "connected": True,
        "message": kite_msg,
        "step_statuses": dump_checklist_steps(live["step_statuses"], ChecklistStepStatus),
        "checklist_ready": live["checklist_ready"],
        "missing_steps": live["missing_steps"],
        "trade_plan": trade_plan,
        "validation": validation,
        "strategy_analysis": live.get("strategy_analysis"),
        "can_place": _resolve_can_place(trade_plan, validation, market_open),
        "market_open": market_open,
        "allow_test_place": allow_offhours_sensex_place(),
        "nifty_spot": live.get("nifty_spot"),
        "data_source": live.get("data_source"),
    }


def place_gtt_for_plan(
    plan: Dict[str, Any],
    *,
    fill_price: Optional[float] = None,
    entry_order_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Place OCO GTT exit (SL + target) for a long NFO option."""
    from services.sensex_indicator_plan import gtt_triggers_from_plan, refresh_plan_at_execution

    result: Dict[str, Any] = {
        "gtt_trigger_id": None,
        "errors": [],
        "messages": [],
        "ok": False,
        "trade_plan": None,
    }
    working = dict(plan)
    if fill_price is not None and fill_price > 0:
        working["entry_limit_price"] = float(fill_price)
        working["entry_premium"] = float(fill_price)

    plan = refresh_plan_at_execution(working)
    if fill_price is not None and fill_price > 0:
        plan["entry_limit_price"] = float(fill_price)
        plan["entry_premium"] = float(fill_price)

    from services.premium_exit_policy import enforce_plan_exits

    entry_limit = float(plan.get("entry_limit_price") or plan.get("entry_premium") or 0)
    plan = enforce_plan_exits(plan, entry=entry_limit)

    symbol = plan.get("tradingsymbol")
    if not symbol:
        result["errors"].append("No tradingsymbol in plan")
        return result

    qty = int(plan.get("quantity") or plan.get("num_lots") or 1)
    sl_prem = float(plan["stop_loss_premium"])
    tgt_prem = float(plan["target_premium"])
    result["trade_plan"] = plan

    product = resolve_sensex_bfo_product(plan)
    gtt_plan = dict(plan)
    try:
        from services.momentum_trail import get_momentum_trail_config, gtt_tp_cap_for_trail

        if get_momentum_trail_config().enabled and entry_limit > 0:
            gtt_plan["target_premium"] = gtt_tp_cap_for_trail(entry_limit, tgt_prem)
    except Exception:
        pass
    sl_trigger, tp_trigger, last_price = gtt_triggers_from_plan(gtt_plan)
    gtt_tp = float(gtt_plan.get("target_premium") or tgt_prem)
    if fill_price is not None and fill_price > 0:
        last_price = fill_price

    exchange = str(plan.get("exchange") or "BFO")
    gtt = place_gtt_tool.invoke(
        {
            "tradingsymbol": symbol,
            "exchange": exchange,
            "trigger_type": "two-leg",
            "trigger_prices": [sl_trigger, tp_trigger],
            "last_price": last_price,
            "stop_loss_price": sl_prem,
            "target_price": gtt_tp,
            "quantity": qty,
            "transaction_type": "SELL",
            "product": product,
        }
    )

    if gtt.get("status") == "success":
        tid = gtt.get("trigger_id")
        if isinstance(tid, dict):
            tid = tid.get("trigger_id", tid)
        result["gtt_trigger_id"] = str(tid)
        result["ok"] = True
        result["messages"].append(f"GTT OCO trigger {result['gtt_trigger_id']}")
        if entry_order_id:
            try:
                from services.exit_trail_register import register_from_trade_plan

                register_from_trade_plan(
                    result.get("trade_plan") or plan,
                    entry_order_id=str(entry_order_id),
                    gtt_trigger_id=result["gtt_trigger_id"],
                    segment="sensex",
                    fill_price=fill_price,
                    paper=False,
                )
            except Exception as exc:
                log_warning(f"Exit trail register failed: {exc}")
    else:
        result["errors"].append(gtt.get("error") or "GTT placement failed")

    return result


def place_trade(
    completed_steps: Optional[List[bool]] = None,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    num_lots: int = 1,
    confirm: bool = False,
    auto_execute: bool = False,
    trade_plan_snapshot: Optional[Dict[str, Any]] = None,
    defer_gtt_until_fill: bool = False,
    run_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    preview = preview_trade(
        completed_steps,
        direction,
        risk_percentage,
        reward_percentage,
        num_lots,
        auto_execute=auto_execute,
        run_params=run_params,
    )
    result = {
        **preview,
        "placed": False,
        "entry_order_id": None,
        "gtt_trigger_id": None,
        "errors": [],
        "entry_paper": False,
    }

    if not confirm:
        result["errors"].append("Set confirm=true to place orders")
        return result

    if auto_execute:
        result["messages"] = ["Auto-executing checklist…"] + list(result.get("messages", []))

    plan = trade_plan_snapshot or preview.get("trade_plan")
    if not plan:
        result["errors"].append("Cannot place — no trade plan (re-run Place trade preview)")
        return result

    market_open = is_market_session_open()
    offhours_test = allow_offhours_sensex_place()
    from services.watch_execute import resolve_can_execute

    can_execute = resolve_can_execute(
        preview,
        plan,
        offhours_allowed=offhours_test and not market_open,
    ) or (
        confirm
        and not auto_execute
        and bool(preview.get("checklist_ready"))
        and bool(plan)
    )
    if not can_execute:
        from services.watch_skip_utils import can_execute_block_errors

        result["errors"].extend(can_execute_block_errors(preview, plan, segment="sensex"))
        return result

    if trade_plan_snapshot:
        result["trade_plan"] = plan
        result["messages"] = list(result.get("messages", [])) + [
            "Using trade plan from confirm dialog"
        ]

    # UI allows off-hours test; risk gate normally blocks outside 9:15–15:30 IST.
    skip_session = offhours_test and not market_open

    from services.sensex_indicator_plan import gtt_triggers_from_plan, refresh_plan_at_execution

    plan = refresh_plan_at_execution(plan)
    result["trade_plan"] = plan

    symbol = plan["tradingsymbol"]
    qty = int(plan["quantity"])
    entry_limit = float(plan.get("entry_limit_price") or plan.get("entry_premium"))
    sl_prem = float(plan["stop_loss_premium"])
    tgt_prem = float(plan["target_premium"])

    try:
        from services.paper_trading import is_paper_mode_for_segment

        if is_paper_mode_for_segment("sensex"):
            from services.paper_order_guard import paper_entry_levels_valid

            ok, msg = paper_entry_levels_valid(entry_limit, sl_prem, tgt_prem)
            if not ok:
                result["errors"].append(msg)
                return result
    except Exception:
        pass

    if not market_open and not skip_session:
        result["errors"].append(
            "Market is closed. Enable V2_ALLOW_OFFHOURS_PLACE for test bypass or place during 9:15 AM–3:30 PM IST."
        )
        return result

    if market_open and is_past_sensex_entry_cutoff() and not skip_session:
        result["errors"].append(sensex_entry_cutoff_message())
        return result

    if skip_session:
        result["messages"] = list(result.get("messages", [])) + [
            "Off-hours test: bypassing session gate (LIMIT entry may still be rejected when exchange closed)"
        ]

    entry_ready = plan.get("entry_ready", True)
    manual_confirm = confirm and preview.get("checklist_ready")
    if not entry_ready and not manual_confirm and not (skip_session and allow_offhours_sensex_place()):
        reason = plan.get("entry_block_reason") or "Entry setup not confirmed by indicators"
        result["errors"].append(reason)
        result["errors"].append(
            "Refresh preview after OR/PDH/EMA conditions align; limit is set to patient mid, not market chase"
        )
        return result
    if not entry_ready and manual_confirm:
        result["messages"] = list(result.get("messages", [])) + [
            "Manual confirm: placing despite entry_ready=false (user confirmed in UI)"
        ]

    try:
        from services.paper_trading import is_paper_mode_for_segment

        if is_paper_mode_for_segment("sensex"):
            result["messages"] = list(result.get("messages", [])) + [
                "Paper mode ON (Sensex50) — orders go to paper ledger, not Zerodha"
            ]

        ind = plan.get("indicators") or {}
        entry_style = plan.get("entry_style") or "indicator_limit"
        result["messages"] = list(result.get("messages", [])) + [
            (
                f"Entry {entry_style} LIMIT ₹{entry_limit} (fair ₹{plan.get('entry_fair_premium', entry_limit)}) · "
                f"{plan.get('num_lots')} lots × {plan.get('lot_size')} = {qty} qty · "
                f"Sensex SL {plan.get('spot_stop_loss')} TP {plan.get('spot_target')} · "
                f"GTT SL ₹{sl_prem} TP ₹{tgt_prem}"
            ),
            (
                f"Indicators @ place: spot {ind.get('nifty_spot')} OR {ind.get('or_low')}-{ind.get('or_high')} "
                f"PDH {ind.get('pdh')} PDL {ind.get('pdl')} EMA9 {ind.get('ema9')}"
            ),
        ]

        product = resolve_sensex_bfo_product(plan)
        if str(plan.get("product") or "").upper() == "MIS":
            log_warning("V2 place: MIS not valid with GTT on NFO — using NRML for entry and exit")
            result["messages"] = list(result.get("messages", [])) + [
                "Product: NRML (Zerodha GTT is not supported for MIS on NFO)"
            ]
            plan["product"] = product
            result["trade_plan"] = plan

        from services.sensex_indicator_plan import live_option_quote
        from utils.kite_order_utils import validate_buy_limit_price

        quote = live_option_quote(symbol, exchange="BFO")
        entry_limit, limit_ok, limit_msg = validate_buy_limit_price(entry_limit, quote=quote)
        plan["entry_limit_price"] = entry_limit
        result["trade_plan"] = plan
        if not limit_ok:
            result["errors"].append(limit_msg)
            return result
        if limit_msg:
            result["messages"] = list(result.get("messages", [])) + [limit_msg]

        exchange = str(plan.get("exchange") or "BFO")
        entry = place_order_tool.invoke(
            {
                "tradingsymbol": symbol,
                "exchange": exchange,
                "transaction_type": "BUY",
                "quantity": qty,
                "order_type": "LIMIT",
                "price": entry_limit,
                "product": product,
                "skip_session_check": skip_session,
                "segment": "sensex",
                "stoploss": sl_prem,
                "target": tgt_prem,
                "paper_trade_plan": plan,
            }
        )
        if entry.get("status") != "success":
            err = entry.get("error") or "Entry order failed"
            result["errors"].append(err)
            if not market_open:
                result["errors"].append(
                    "Zerodha rejects orders outside market hours — test during 9:15 AM–3:30 PM IST Mon–Fri"
                )
            return result

        entry_id = str(entry.get("order_id"))
        result["entry_order_id"] = entry_id
        result["entry_paper"] = bool(entry.get("paper"))
        log_info(
            f"V2 LIMIT entry {entry_id} {symbol} @ {entry_limit} qty={qty} "
            f"paper={result['entry_paper']}"
        )

        venue = "paper" if result["entry_paper"] else "Zerodha"
        if defer_gtt_until_fill:
            result["placed"] = True
            result["gtt_deferred"] = True
            result["trade_plan"] = plan
            result["messages"] = list(preview.get("messages", [])) + [
                f"Entry order {entry_id} on {venue}",
                "GTT OCO will attach after entry fills",
            ]
        else:
            gtt_result = place_gtt_for_plan(plan, entry_order_id=entry_id)
            result["trade_plan"] = gtt_result.get("trade_plan") or plan
            if gtt_result.get("gtt_trigger_id"):
                result["gtt_trigger_id"] = gtt_result["gtt_trigger_id"]
                result["placed"] = True
                result["messages"] = list(preview.get("messages", [])) + [
                    f"Entry order {entry_id} on {venue}",
                    *gtt_result.get("messages", []),
                ]
            else:
                result["errors"].extend(gtt_result.get("errors") or [])
                result["placed"] = False
                result["messages"] = list(preview.get("messages", [])) + [
                    f"Entry order {entry_id} placed on Zerodha; exit GTT failed — set SL/target manually in Kite",
                ]

    except Exception as exc:
        log_error(f"V2 place_trade error: {exc}")
        result["errors"].append(str(exc))

    log_info(
        "[V2_ORDER_RESULT]",
        symbol=plan.get("tradingsymbol") if plan else None,
        placed=bool(result.get("placed")),
        entry_order_id=result.get("entry_order_id"),
        gtt_trigger_id=result.get("gtt_trigger_id"),
        entry_paper=bool(result.get("entry_paper")),
        errors=result.get("errors"),
    )

    return result
