"""Crypto (Binance BTCUSDT perp) wizard — checklist, preview, place."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.crypto_config import (
    DEFAULT_LEVERAGE,
    MIN_LIVE_USDT_BALANCE,
    SYMBOL,
    allow_offhours_crypto_place,
    is_crypto_session_open,
)
from services.crypto_indicator_plan import (
    STRATEGY_PENDING_REASON,
    build_trade_plan,
    is_strategy_configured,
    refresh_plan_at_execution,
)
from services.crypto_live_indicators import recalculate_from_ticker
from utils.binance_order_utils import (
    get_binance_credentials,
    get_usdt_balance,
    place_limit_order,
    place_stop_market,
    set_margin_type,
)
from utils.logger import log_error

STEP_TITLES = [
    "Binance API & USDT balance",
    "Market session (24/7)",
    "Direction (LONG / SHORT / AUTO)",
    "Entry rules",
    "Exit & stop loss",
    "Position size & leverage",
    "Risk / reward",
    "Final order review",
]

REQUIRED_MARKED_STEPS = [0, 2, 7]
PENDING_MSG = "Awaiting strategy configuration"


def _step_statuses_from_live(
    plan: Dict[str, Any],
    live: Dict[str, Any],
    balance: float,
    *,
    paper_mode: bool = False,
) -> List[Dict[str, Any]]:
    side = (plan.get("side") or "AUTO").upper()
    strategy_ok = is_strategy_configured()
    min_bal = MIN_LIVE_USDT_BALANCE if not paper_mode else 0.0
    qty = float(plan.get("quantity") or 0)
    sl = float(plan.get("stop_loss_premium") or 0)
    entry_px = float(plan.get("entry_limit_price") or 0)

    checks: List[tuple[bool, str]] = [
        (
            balance >= min_bal or (paper_mode and bool(live.get("connected"))),
            "Pass" if balance >= min_bal or paper_mode else "Insufficient USDT balance",
        ),
        (
            is_crypto_session_open(),
            "Pass" if is_crypto_session_open() else "Market closed",
        ),
        (
            side in ("LONG", "SHORT", "AUTO"),
            "Pass" if side in ("LONG", "SHORT", "AUTO") else "Pick LONG, SHORT, or AUTO",
        ),
        (
            strategy_ok and bool(plan.get("entry_ready")),
            PENDING_MSG if not strategy_ok else (plan.get("entry_block_reason") or "Entry not ready"),
        ),
        (
            strategy_ok and sl > 0,
            PENDING_MSG if not strategy_ok else "Stop loss not set",
        ),
        (
            qty > 0,
            "Pass" if qty > 0 else "Position size is zero — check USDT margin",
        ),
        (
            strategy_ok and sl > 0 and entry_px > 0 and float(plan.get("target_premium") or 0) > 0,
            PENDING_MSG if not strategy_ok else "Risk / reward not validated",
        ),
        (
            False,
            PENDING_MSG,
        ),
    ]

    out: List[Dict[str, Any]] = []
    for i, title in enumerate(STEP_TITLES):
        ok, msg = checks[i] if i < len(checks) else (False, PENDING_MSG)
        if i == 7:
            ok = all(c[0] for c in checks[:7])
            msg = "Pass" if ok else (PENDING_MSG if not strategy_ok else "Complete prior steps")
        out.append(
            {
                "index": i,
                "title": title,
                "completed": ok,
                "server_ok": ok,
                "message": msg if not ok else "Pass",
                "output": "",
            }
        )
    return out


def get_checklist_live(
    *,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    quantity_btc: Optional[float] = None,
) -> Dict[str, Any]:
    paper_mode = False
    try:
        from services.paper_trading import is_paper_mode_for_segment

        paper_mode = is_paper_mode_for_segment("crypto")
    except Exception:
        pass

    connected = False
    balance = 0.0
    msg = ""
    if paper_mode:
        try:
            from services.paper_funds import get_available_balance

            balance = float(get_available_balance("crypto") or 0)
            connected = True
            msg = f"Paper mode · ${balance:,.2f} USDT available (public Binance quotes)"
        except Exception as exc:
            balance = 0.0
            connected = True
            msg = f"Paper mode · fund lookup: {exc}"
    else:
        try:
            get_binance_credentials()
            balance = get_usdt_balance()
            connected = True
            msg = f"Binance connected · USDT available ${balance:,.2f}"
        except Exception as exc:
            balance = 0.0
            connected = False
            msg = str(exc)

    plan, messages = build_trade_plan(
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        quantity_btc=quantity_btc,
    )
    live = plan.get("indicators") or recalculate_from_ticker()
    if not plan and live.get("connected"):
        connected = True
    market_open = is_crypto_session_open()
    steps = _step_statuses_from_live(plan, live, balance, paper_mode=paper_mode)

    from services.checklist_step_utils import apply_market_closed_gate

    steps = apply_market_closed_gate(
        steps,
        market_open=market_open,
        allow_offhours=allow_offhours_crypto_place(),
        gated_indices=list(range(2, len(STEP_TITLES))),
        closed_message="Market closed — preview only until session",
    )
    missing = [s["index"] for s in steps if not s["server_ok"]]
    ready = len(missing) == 0 and (market_open or allow_offhours_crypto_place())

    if paper_mode:
        messages.append("Paper mode ON (Crypto) — simulated Binance fills in paper ledger")

    if not is_strategy_configured():
        messages.append(STRATEGY_PENDING_REASON)

    from services.watch_execute import resolve_can_execute

    preview_stub = {
        "checklist_ready": ready,
        "can_place": ready and connected and bool(plan) and not paper_mode,
        "paper_trading_mode": paper_mode,
        "trade_plan": plan or None,
    }
    can_execute = resolve_can_execute(preview_stub, plan)

    return {
        "connected": connected,
        "message": msg,
        "checklist_ready": ready,
        "missing_steps": missing,
        "step_statuses": steps,
        "trade_plan": plan or None,
        "can_place": ready and connected and bool(plan) and not paper_mode,
        "can_execute": can_execute,
        "market_open": market_open,
        "allow_test_place": allow_offhours_crypto_place(),
        "paper_trading_mode": paper_mode,
        "messages": messages,
        "binance_balance_usdt": balance,
        "leverage": DEFAULT_LEVERAGE,
        "strategy_configured": is_strategy_configured(),
    }


def get_strategy_analysis(direction: str = "AUTO") -> Dict[str, Any]:
    return {
        "selected_id": "pending",
        "selected_name": "Not configured",
        "selected_score": 0,
        "selected_fit": "wait",
        "selected_option_kind": direction.upper() if direction else "AUTO",
        "strategies": [],
        "output_summary": STRATEGY_PENDING_REASON,
    }


def get_checklist_analyze(
    *,
    step: int,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    quantity_btc: Optional[float] = None,
) -> Dict[str, Any]:
    data = get_checklist_live(
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        quantity_btc=quantity_btc,
    )
    try:
        idx = int(step)
    except Exception:
        idx = 0
    rows = list(data.get("step_statuses") or [])
    focus = next((r for r in rows if int(r.get("index", -1)) == idx), None)
    if focus:
        data["messages"] = list(data.get("messages", [])) + [
            f"Analyze step {idx}: {focus.get('title')} — {focus.get('message')}"
        ]
    return data


def preview_trade(
    completed_steps: Optional[List[bool]] = None,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    quantity_btc: Optional[float] = None,
    auto_execute: bool = False,
) -> Dict[str, Any]:
    live_data = get_checklist_live(
        direction=direction,
        risk_percentage=risk_percentage,
        reward_percentage=reward_percentage,
        quantity_btc=quantity_btc,
    )
    plan = live_data.get("trade_plan") or {}
    can_place = bool(live_data.get("can_place"))
    if completed_steps and not auto_execute:
        for idx in REQUIRED_MARKED_STEPS:
            if idx < len(completed_steps) and not completed_steps[idx]:
                can_place = False
    from services.watch_execute import resolve_can_execute

    can_execute = resolve_can_execute(live_data, plan)
    return {
        **live_data,
        "can_execute": can_execute,
        "validation": {
            "is_good_trade": can_execute,
            "risk_amount": plan.get("risk_inr"),
            "reward_amount": plan.get("reward_inr"),
        },
        "strategy_analysis": get_strategy_analysis(direction),
    }


def place_trade(
    completed_steps: Optional[List[bool]] = None,
    direction: str = "AUTO",
    risk_percentage: Optional[float] = None,
    reward_percentage: Optional[float] = None,
    quantity_btc: Optional[float] = None,
    confirm: bool = False,
    auto_execute: bool = False,
    trade_plan_snapshot: Optional[Dict[str, Any]] = None,
    defer_exits_until_fill: bool = True,
) -> Dict[str, Any]:
    preview = preview_trade(
        completed_steps,
        direction,
        risk_percentage,
        reward_percentage,
        quantity_btc,
        auto_execute=auto_execute,
    )
    result = {
        **preview,
        "placed": False,
        "entry_order_id": None,
        "sl_order_id": None,
        "tp_order_id": None,
        "errors": [],
        "entry_paper": False,
    }

    if not confirm:
        result["errors"].append("Set confirm=true to place orders")
        return result

    if not is_strategy_configured():
        result["errors"].append(STRATEGY_PENDING_REASON)
        return result

    plan = trade_plan_snapshot or preview.get("trade_plan")
    if not plan:
        result["errors"].append("No trade plan")
        return result

    plan = refresh_plan_at_execution(plan)
    result["trade_plan"] = plan

    if not preview.get("can_execute") and not (confirm and preview.get("checklist_ready")):
        result["errors"].append("Checklist not ready")
        return result

    side = str(plan.get("side") or "LONG").upper()
    order_side = "BUY" if side == "LONG" else "SELL"
    exit_side = "SELL" if side == "LONG" else "BUY"
    qty = float(plan.get("quantity") or 0)
    entry_limit = float(plan.get("entry_limit_price") or 0)
    sl_px = float(plan.get("stop_loss_premium") or 0)
    tp_px = float(plan.get("target_premium") or 0)

    if qty <= 0:
        result["errors"].append("Position size is zero — insufficient USDT for min BTCUSDT lot at current leverage")
        return result

    if sl_px <= 0:
        result["errors"].append("Stop loss not configured")
        return result

    try:
        from services.paper_trading import is_paper_mode_for_segment, paper_place_order

        paper_mode = is_paper_mode_for_segment("crypto")
        if paper_mode:
            from services.paper_trade_detail import slim_trade_plan_for_paper

            payload = {
                "tradingsymbol": SYMBOL,
                "exchange": "BINANCE",
                "transaction_type": order_side,
                "quantity": qty,
                "order_type": "LIMIT",
                "price": entry_limit,
                "product": "ISOLATED",
                "segment": "crypto",
                "leverage": DEFAULT_LEVERAGE,
                "stoploss": sl_px,
                "target": tp_px,
                "paper_fill_price": entry_limit,
                "paper_trade_plan": slim_trade_plan_for_paper(plan),
            }
            entry_id = paper_place_order(payload)
            result["entry_order_id"] = entry_id
            result["entry_paper"] = True
            result["placed"] = True
            result["messages"] = list(preview.get("messages", [])) + [
                f"[PAPER] Entry {order_side} LIMIT {qty} BTC @ ${entry_limit:,.2f} ({DEFAULT_LEVERAGE}x)",
                f"Paper SL ${sl_px:,.2f} · TP ${tp_px:,.2f} (monitor)",
            ]
            from services.risk_gate import record_order_placed

            record_order_placed(qty * entry_limit)
            return result

        set_margin_type(SYMBOL, "ISOLATED")
        try:
            live_usdt = float(get_usdt_balance() or 0)
            need_margin = (qty * entry_limit) / max(1, DEFAULT_LEVERAGE)
            if live_usdt + 0.01 < need_margin:
                result["errors"].append(
                    f"Insufficient margin: need ${need_margin:,.2f} USDT @ {DEFAULT_LEVERAGE}x "
                    f"for {qty} BTC, have ${live_usdt:,.2f}"
                )
                return result
        except Exception as exc:
            result["errors"].append(f"Binance balance check failed: {exc}")
            return result

        entry = place_limit_order(
            symbol=SYMBOL,
            side=order_side,
            quantity=qty,
            price=entry_limit,
            leverage=DEFAULT_LEVERAGE,
        )
        if not entry.get("ok"):
            result["errors"].append(entry.get("error") or "Entry failed")
            return result

        entry_id = entry.get("order_id")
        result["entry_order_id"] = entry_id
        result["placed"] = True
        result["messages"] = list(preview.get("messages", [])) + [
            f"Entry {order_side} LIMIT {qty} BTC @ ${entry_limit:,.2f} ({DEFAULT_LEVERAGE}x)",
        ]

        if not defer_exits_until_fill:
            sl = place_stop_market(
                symbol=SYMBOL, side=exit_side, quantity=qty, stop_price=sl_px
            )
            if sl.get("ok"):
                result["sl_order_id"] = sl.get("order_id")
                result["messages"].append(f"SL STOP_MARKET @ ${sl_px:,.2f}")
            if tp_px > 0:
                tp = place_limit_order(
                    symbol=SYMBOL,
                    side=exit_side,
                    quantity=qty,
                    price=tp_px,
                    reduce_only=True,
                )
                if tp.get("ok"):
                    result["tp_order_id"] = tp.get("order_id")
                    result["messages"].append(f"TP LIMIT @ ${tp_px:,.2f}")
        else:
            result["exits_deferred"] = True
            result["messages"].append("SL/TP will attach after entry fills (watch)")

        from services.risk_gate import record_order_placed

        record_order_placed(qty * entry_limit)
    except Exception as exc:
        log_error(f"[Crypto] place_trade: {exc}")
        result["errors"].append(str(exc))

    return result
