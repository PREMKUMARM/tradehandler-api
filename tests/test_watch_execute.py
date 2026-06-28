"""Tests for watch execute gates and strategy watch evaluation."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.watch_execute import is_paper_trading, resolve_can_execute


class TestResolveCanExecute:
    PLAN = {
        "tradingsymbol": "TEST24MAY100CE",
        "entry_limit_price": 100,
        "entry_premium": 100,
        "stop_loss_premium": 80,
        "target_premium": 120,
        "entry_ready": True,
    }

    def test_live_can_place(self):
        preview = {
            "can_place": True,
            "checklist_ready": True,
            "paper_trading_mode": False,
        }
        assert resolve_can_execute(preview, self.PLAN) is True

    def test_paper_requires_checklist(self):
        preview = {
            "can_place": False,
            "checklist_ready": False,
            "paper_trading_mode": True,
        }
        assert resolve_can_execute(preview, self.PLAN) is False

    def test_paper_checklist_complete(self):
        preview = {
            "can_place": False,
            "checklist_ready": True,
            "paper_trading_mode": True,
        }
        assert resolve_can_execute(preview, self.PLAN) is True

    def test_offhours_bypass(self):
        preview = {
            "can_place": False,
            "checklist_ready": False,
            "paper_trading_mode": False,
        }
        assert resolve_can_execute(preview, self.PLAN, offhours_allowed=True) is True

    def test_no_plan(self):
        preview = {"can_place": True, "checklist_ready": True}
        assert resolve_can_execute(preview, None) is False

    @patch("services.paper_trading.is_paper_mode", return_value=True)
    def test_paper_from_env(self, _mock):
        preview = {
            "can_place": False,
            "checklist_ready": True,
            "paper_trading_mode": False,
        }
        assert is_paper_trading(preview) is True
        assert resolve_can_execute(preview, self.PLAN) is True


class TestV2WatchEvaluate:
    def test_autonomous_when_checklist_complete_paper(self):
        from services.v2_strategy_watch import V2StrategyWatch

        watch = V2StrategyWatch()
        watch._armed = True
        watch._cfg.mode = "autonomous"
        watch._cfg.auto_place_on_signal = True
        watch._cfg.auto_execute_checklist = True
        watch._eval_count = 1
        watch._last_checklist_ready = False

        preview = {
            "checklist_ready": True,
            "can_place": False,
            "can_execute": True,
            "paper_trading_mode": True,
            "validation": {"is_good_trade": True},
            "trade_plan": {
                "tradingsymbol": "NIFTY24MAY100CE",
                "entry_ready": True,
                "entry_confirmation_score": 80,
                "entry_limit_price": 50,
                "stop_loss_premium": 40,
                "target_premium": 60,
                "quantity": 25,
            },
        }

        with patch.object(watch, "_persist"):
            with patch("services.v2_strategy_watch.v2_trade_service.preview_trade", return_value=preview):
                with patch(
                    "services.v2_strategy_watch.v2_trade_service.allow_offhours_v2_place",
                    return_value=False,
                ):
                    with patch(
                        "services.v2_order_guard.autonomous_place_allowed",
                        return_value=(True, "OK"),
                    ):
                        fire, auto, _, plan, can_exec = watch._evaluate_sync()

        assert fire is True
        assert auto is True
        assert can_exec is True
        assert plan["tradingsymbol"] == "NIFTY24MAY100CE"

    def test_no_autonomous_when_checklist_incomplete(self):
        from services.v2_strategy_watch import V2StrategyWatch

        watch = V2StrategyWatch()
        watch._armed = True
        watch._cfg.mode = "autonomous"
        watch._cfg.auto_place_on_signal = True
        watch._eval_count = 2
        watch._last_checklist_ready = False

        preview = {
            "checklist_ready": False,
            "can_place": False,
            "paper_trading_mode": True,
            "trade_plan": {"tradingsymbol": "X", "entry_ready": True},
        }

        with patch.object(watch, "_persist"):
            with patch("services.v2_strategy_watch.v2_trade_service.preview_trade", return_value=preview):
                fire, auto, _, _, can_exec = watch._evaluate_sync()

        assert fire is False
        assert auto is False
        assert can_exec is False


class TestCommodityWatchStatus:
    def test_pdl_break_in_progress_detail(self):
        from services.commodity_watch_status import describe_autonomous_setup

        plan = {
            "entry_ready": True,
            "entry_confirmation_score": 55,
            "strategy_name": "PDH / PDL breakout",
            "option_type": "PE",
            "indicators": {"pdl": 8746.0, "last_5m_close": 8860.0},
        }
        out = describe_autonomous_setup(plan, min_score=65)
        assert out["setup_phase"] == "in_progress"
        assert out["autonomous_eligible"] is False
        assert "PDL break in progress" in out["setup_detail"]
        assert "55/65" in out["setup_detail"]

    def test_confirmed_when_score_meets_minimum(self):
        from services.commodity_watch_status import describe_autonomous_setup

        plan = {"entry_ready": True, "entry_confirmation_score": 80}
        out = describe_autonomous_setup(plan, min_score=65)
        assert out["setup_phase"] == "confirmed"
        assert out["autonomous_eligible"] is True


class TestCommodityWatchEvaluate:
    def test_autonomous_retries_without_rising_edge(self):
        from services.commodity_strategy_watch import CommodityStrategyWatch

        watch = CommodityStrategyWatch()
        watch._armed = True
        watch._cfg.mode = "autonomous"
        watch._cfg.auto_place_on_signal = True
        watch._eval_count = 5
        watch._last_checklist_ready = True
        watch._signal_fired_today = True

        preview = {
            "checklist_ready": True,
            "can_place": True,
            "can_execute": True,
            "paper_trading_mode": False,
            "trade_plan": {
                "tradingsymbol": "CRUDEOILM26JUN8850PE",
                "entry_ready": True,
                "entry_confirmation_score": 72,
                "entry_limit_price": 120,
                "quantity": 1,
                "lot_size": 10,
            },
        }

        with patch.object(watch, "_persist"):
            with patch(
                "services.commodity_strategy_watch.is_commodity_new_trading_allowed",
                return_value=True,
            ):
                with patch(
                    "services.commodity_strategy_watch.commodity_trade_service.preview_trade",
                    return_value=preview,
                ):
                    with patch(
                        "services.commodity_strategy_watch.commodity_trade_service.allow_offhours_commodity_place",
                        return_value=False,
                    ):
                        with patch(
                            "services.commodity_strategy_watch.autonomous_place_allowed",
                            return_value=(True, "ok"),
                        ):
                            fire, auto, _, _, can_exec = watch._evaluate_sync()

        assert fire is False
        assert auto is True
        assert can_exec is True


class TestCommodityTryAutoPlace:
    def test_allows_second_trade_when_count_below_max(self):
        from services.commodity_strategy_watch import CommodityStrategyWatch

        watch = CommodityStrategyWatch()
        watch._armed = True
        watch._cfg.mode = "autonomous"
        watch._cfg.auto_place_on_signal = True
        watch._placed_today = True
        watch._placed_count_today = 1
        watch._placed_symbol_today = "CRUDEOILM26JUN8750PE"
        watch._placed_symbols_today = ["CRUDEOILM26JUN8750PE"]
        plan = {
            "tradingsymbol": "CRUDEOILM26JUN8550PE",
            "entry_ready": True,
            "entry_confirmation_score": 80,
            "entry_limit_price": 470,
            "entry_fair_premium": 470,
            "quantity": 1,
            "lot_size": 10,
        }
        preview = {"checklist_ready": True, "can_place": True}

        with patch.object(watch, "_persist"):
            with patch(
                "services.commodity_strategy_watch.is_commodity_new_trading_allowed",
                return_value=True,
            ):
                with patch(
                    "services.commodity_strategy_watch.autonomous_place_allowed",
                    return_value=(True, "OK"),
                ) as guard:
                    with patch(
                        "services.commodity_strategy_watch.commodity_trade_service.place_trade",
                        return_value={
                            "placed": True,
                            "entry_order_id": "1",
                            "gtt_deferred": True,
                            "trade_plan": plan,
                        },
                    ):
                        with patch(
                            "services.commodity_indicator_plan.refresh_plan_at_execution",
                            side_effect=lambda p, **kw: dict(p),
                        ):
                            with patch(
                                "services.risk_gate.check_order_allowed",
                                return_value=(True, "ok"),
                            ):
                                with patch(
                                    "services.risk_gate.is_kill_switch_active",
                                    return_value=False,
                                ):
                                    with patch(
                                        "services.paper_trading.is_paper_mode",
                                        return_value=False,
                                    ):
                                        with patch(
                                            "services.commodity_strategy_watch.broadcast_agent_update",
                                            return_value=None,
                                        ):
                                            with patch(
                                                "services.commodity_strategy_watch.push_service.send_to_user",
                                                return_value=None,
                                            ):
                                                import asyncio

                                                asyncio.run(watch._try_auto_place(preview, plan))

        guard.assert_called_once()
        assert guard.call_args.kwargs["placed_today"] is False
        assert watch._placed_count_today == 2


class TestCommodityOrderGuard:
    def test_allows_same_symbol_when_no_open_position(self):
        from services.commodity_order_guard import autonomous_place_allowed

        plan = {
            "tradingsymbol": "CRUDEOILM26JUN8600PE",
            "entry_ready": True,
            "entry_confirmation_score": 80,
            "entry_limit_price": 470,
            "entry_fair_premium": 470,
            "entry_style": "mid_patient",
            "indicators": {"option_bid": 468, "option_ask": 472, "option_ltp": 470},
        }
        with patch(
            "services.commodity_config.is_commodity_new_trading_allowed",
            return_value=True,
        ):
            with patch(
                "services.trading_agents.guard_agent.has_any_exchange_position",
                return_value=(False, ""),
            ):
                with patch(
                    "services.trading_agents.guard_agent.has_pending_exchange_order",
                    return_value=(False, ""),
                ):
                    with patch(
                        "services.trading_agents.guard_agent.has_exchange_position",
                        return_value=(False, ""),
                    ):
                        with patch(
                            "services.paper_trading.is_paper_mode_for_segment",
                            return_value=False,
                        ):
                            ok, msg = autonomous_place_allowed(plan, placed_today=False)
        assert ok is True
        assert msg == "OK"

    def test_blocks_same_symbol_with_open_position(self):
        from services.commodity_order_guard import autonomous_place_allowed

        plan = {
            "tradingsymbol": "CRUDEOILM26JUN8600PE",
            "entry_ready": True,
            "entry_confirmation_score": 80,
            "entry_limit_price": 470,
            "entry_fair_premium": 470,
            "entry_style": "mid_patient",
            "indicators": {"option_bid": 468, "option_ask": 472, "option_ltp": 470},
        }
        with patch(
            "services.commodity_config.is_commodity_new_trading_allowed",
            return_value=True,
        ):
            with patch(
                "services.trading_agents.guard_agent.has_any_exchange_position",
                return_value=(False, ""),
            ):
                with patch(
                    "services.trading_agents.guard_agent.has_pending_exchange_order",
                    return_value=(False, ""),
                ):
                    with patch(
                        "services.trading_agents.guard_agent.has_exchange_position",
                        return_value=(True, "Open position on CRUDEOILM26JUN8600PE (qty=1)"),
                    ):
                        with patch(
                            "services.paper_trading.is_paper_mode_for_segment",
                            return_value=False,
                        ):
                            ok, msg = autonomous_place_allowed(plan, placed_today=False)
        assert ok is False
        assert "Open position" in msg


class TestCommodityDeferredGtt:
    def test_place_trade_defers_gtt_by_default(self):
        from services import commodity_trade_service

        plan = {
            "tradingsymbol": "CRUDEOILM26JUN8600PE",
            "entry_ready": True,
            "entry_limit_price": 470.0,
            "entry_premium": 470.0,
            "stop_loss_premium": 450.0,
            "target_premium": 510.0,
            "num_lots": 1,
            "lot_size": 10,
            "product": "NRML",
            "indicators": {},
        }
        preview = {
            "checklist_ready": True,
            "can_place": True,
            "trade_plan": plan,
            "messages": [],
        }
        with patch.object(commodity_trade_service, "preview_trade", return_value=preview):
            with patch.object(commodity_trade_service, "is_mcx_session_open", return_value=True):
                with patch(
                    "services.commodity_config.is_commodity_new_trading_allowed",
                    return_value=True,
                ):
                    with patch.object(commodity_trade_service, "has_pending_mcx_order", return_value=(False, "")):
                        with patch.object(commodity_trade_service, "has_mcx_position", return_value=(False, "")):
                            with patch(
                                "services.commodity_trade_service.place_order_tool"
                            ) as order_tool:
                                order_tool.invoke = MagicMock(
                                    return_value={"status": "success", "order_id": "E1"}
                                )
                                with patch(
                                    "services.commodity_trade_service.place_gtt_for_plan",
                                ) as gtt_fn:
                                    with patch(
                                        "services.commodity_indicator_plan.refresh_plan_at_execution",
                                        side_effect=lambda p, **kw: dict(p),
                                    ):
                                        result = commodity_trade_service.place_trade(
                                            confirm=True,
                                            trade_plan_snapshot=plan,
                                        )
        gtt_fn.assert_not_called()
        assert result["placed"] is True
        assert result["gtt_deferred"] is True
        assert result["entry_order_id"] == "E1"
        assert result.get("gtt_trigger_id") is None

    def test_on_entry_filled_places_sl(self):
        from services.commodity_strategy_watch import CommodityStrategyWatch

        watch = CommodityStrategyWatch()
        plan = {
            "tradingsymbol": "CRUDEOILM26JUN8600PE",
            "entry_limit_price": 476.1,
            "stop_loss_premium": 460.0,
            "target_premium": 500.0,
            "num_lots": 1,
        }
        watch._register_pending(
            entry_id="E1",
            sym="CRUDEOILM26JUN8600PE",
            trade_plan=plan,
            gtt_trigger_id=None,
            gtt_deferred=True,
        )
        with patch.object(watch, "_order_fill_price", return_value=476.1):
            with patch(
                "services.commodity_strategy_watch.commodity_trade_service.place_gtt_for_plan",
                return_value={
                    "sl_order_id": "SL321",
                    "gtt_trigger_id": "SL321",
                    "trade_plan": {
                        "stop_loss_premium": 460.0,
                        "target_premium": 500.0,
                    },
                },
            ):
                with patch.object(watch, "_persist"):
                    import asyncio

                    asyncio.run(watch._on_entry_filled("E1"))

        assert not watch._pending_entries
        assert watch._pending_entry_order_id is None
        assert watch._events[0].kind == "auto_sl_placed"
        assert "SL321" in watch._events[0].message
