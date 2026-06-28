"""All segments: 1:1 entry R:R and momentum trail after first target."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.commodity_indicator_plan import size_from_risk
from services.momentum_trail import (
    compute_trailed_levels,
    get_momentum_trail_config,
    gtt_tp_cap_for_trail,
    should_activate_trail,
)
from services.option_contract_indicators import resolve_long_buy_exit_levels
from services.premium_exit_policy import default_reward_pct, entry_initial_rr


def _sample_plan(*, entry: float, sl: float, qty: int = 75, exchange: str = "NFO") -> dict:
    tgt = entry + (entry - sl)
    return {
        "tradingsymbol": "TEST-OPT",
        "exchange": exchange,
        "product": "NRML",
        "entry_premium": entry,
        "entry_limit_price": entry,
        "stop_loss_premium": sl,
        "target_premium": tgt,
        "quantity": qty,
        "num_lots": 1,
        "strategy_id": "bb_5m_mean_reversion",
        "indicators": {"risk_pct": 1.0, "reward_pct": 1.0},
    }


@pytest.mark.parametrize(
    "module_path,validate_fn",
    [
        ("services.v2_trade_service", "_validate_trade_plan"),
        ("services.commodity_trade_service", "_validate_trade_plan"),
        ("services.sensex_trade_service", "_validate_trade_plan"),
    ],
)
def test_all_segments_validate_one_to_one_plan(module_path, validate_fn):
    import importlib

    mod = importlib.import_module(module_path)
    validate = getattr(mod, validate_fn)
    plan = _sample_plan(entry=20.0, sl=10.0, qty=50)
    out = validate(plan, capital=1_000_000, risk_pct=1.0, reward_pct=1.0)
    assert out["is_good_trade"] is True
    assert out["reward_risk_ratio_required"] == 1.0
    assert out["reward_amount"] == pytest.approx(out["risk_amount"], rel=0.01)
    assert out["trail_extends_reward"] is True


def test_default_reward_pct_matches_entry_rr():
    assert entry_initial_rr() == 1.0
    assert default_reward_pct(1.0) == 1.0
    assert default_reward_pct(2.0) == 2.0
    assert default_reward_pct(1.0, reward_percentage=3.0) == 3.0


def test_bb_exits_one_to_one_when_reward_ratio_one():
    intra = {"bb_lower": 15.0, "bb_middle": 20.0, "bb_upper": 25.0, "bb_on_contract": "TEST"}
    sl, tgt, _, _, _, _ = resolve_long_buy_exit_levels(
        strategy_id="bb_5m_mean_reversion",
        entry_premium=20.0,
        option_kind="CE",
        intra_bb=intra,
        underlying_spot=24000.0,
        underlying_sl=23900.0,
        underlying_tgt=24100.0,
        strike=24000,
        reward_ratio=1.0,
    )
    risk = 20.0 - sl
    reward = tgt - 20.0
    assert risk > 0
    assert reward == pytest.approx(risk, rel=0.05)


def test_20rupees_strategy_one_to_one():
    sl, tgt, _, _, _, note = resolve_long_buy_exit_levels(
        strategy_id="20rupees_strategy",
        entry_premium=20.0,
        option_kind="CE",
        intra_bb={},
        underlying_spot=82000.0,
        underlying_sl=0.0,
        underlying_tgt=0.0,
        strike=82000,
        reward_ratio=1.0,
    )
    assert sl == pytest.approx(9.0)
    assert tgt == pytest.approx(20.0 + (20.0 - sl))
    assert "1:1" in note


def test_trail_activates_at_first_target():
    entry, sl, tgt = 20.0, 10.0, 30.0
    assert should_activate_trail(30.0, entry, tgt, trail_active=False) is True
    assert should_activate_trail(29.9, entry, tgt, trail_active=False) is False
    cfg = get_momentum_trail_config()
    new_sl, new_tp, peak, active, note = compute_trailed_levels(
        entry=entry,
        peak=30.0,
        ltp=30.0,
        current_sl=sl,
        current_tp=tgt,
        trail_active=False,
        cfg=cfg,
        initial_risk_unit=entry - sl,
        initial_target=tgt,
    )
    assert active is True
    assert new_sl > entry
    assert new_tp > tgt
    assert "1:1 target" in note


def test_gtt_tp_widened_for_trail_all_segments():
    entry, tgt = 20.0, 30.0
    cap = gtt_tp_cap_for_trail(entry, tgt)
    assert cap > tgt


@pytest.mark.parametrize(
    "place_fn_path",
    [
        "services.v2_trade_service.place_gtt_for_plan",
        "services.commodity_trade_service.place_gtt_for_plan",
        "services.sensex_trade_service.place_gtt_for_plan",
    ],
)
def test_place_sl_exit_after_fill(place_fn_path):
    import importlib

    mod_name, fn_name = place_fn_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    place_sl = getattr(mod, fn_name)
    plan = _sample_plan(entry=20.0, sl=10.0)
    captured = {}

    def _invoke(payload):
        captured.update(payload)
        return {"status": "success", "order_id": "SL123"}

    segment = "nifty50"
    if "commodity" in mod_name:
        segment = "commodity"
    elif "sensex" in mod_name:
        segment = "sensex"

    with patch("services.sl_exit_service.place_order_tool") as mock_tool:
        mock_tool.invoke = _invoke
        with patch(
            "services.v2_indicator_plan.refresh_plan_at_execution",
            side_effect=lambda p, **kw: dict(p),
        ):
            with patch("services.exit_trail_register.register_from_trade_plan"):
                result = place_sl(plan, entry_order_id="E1")
    assert result.get("sl_order_id") == "SL123"
    assert captured["order_type"] == "SL-M"
    assert captured["transaction_type"] == "SELL"


def test_register_trail_stores_one_r_initial_target():
    with patch("services.momentum_trail.get_momentum_trail_config") as mock_cfg:
        mock_cfg.return_value.enabled = True
        with patch("database.connection.get_database") as mock_db:
            conn = MagicMock()
            mock_db.return_value.get_connection.return_value = conn
            from services.exit_trail_store import register_exit_trail

            register_exit_trail(
                segment="nifty50",
                entry_order_id="OID1",
                tradingsymbol="NIFTY-TEST",
                exchange="NFO",
                product="NRML",
                quantity=75,
                entry_price=20.0,
                stop_loss=10.0,
                target=30.0,
                gtt_trigger_id="G1",
                strategy_id="bb_5m_mean_reversion",
            )
            args = conn.execute.call_args[0][1]
            initial_target = args[12]
            assert initial_target == pytest.approx(30.0)


def test_commodity_sizing_not_reduced_for_one_to_one():
    qty, _, risk_inr, _ = size_from_risk(
        capital=500_000,
        risk_pct=1.0,
        reward_pct=3.0,
        entry_premium=100.0,
        sl_premium=90.0,
        target_premium=110.0,
        lot_size=100,
        max_qty_cap=10,
    )
    qty_loose, _, _, _ = size_from_risk(
        capital=500_000,
        risk_pct=1.0,
        reward_pct=1.0,
        entry_premium=100.0,
        sl_premium=90.0,
        target_premium=110.0,
        lot_size=100,
        max_qty_cap=10,
    )
    assert qty == qty_loose
    assert qty >= 1
    assert risk_inr > 0
