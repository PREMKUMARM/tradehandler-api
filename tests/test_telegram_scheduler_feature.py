"""
Automated tests for Telegram scheduler: catalog, REST surface, indicator reports, strategy overview.

Run from `tradehandler-api`:
  pip install pytest httpx
  python -m pytest tests/test_telegram_scheduler_feature.py -v

Note: Import `api.v1.routes.telegram_scheduler` via the package loads `api.v1` which registers
all routers and is slow. Tests load the route module file in isolation.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# conftest adds parent; support running this file directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_telegram_scheduler_routes():
    """Load route module without importing `api.v1` package (avoids full router tree)."""
    p = _ROOT / "api/v1/routes/telegram_scheduler.py"
    spec = importlib.util.spec_from_file_location("_telegram_scheduler_routes", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_catalog_payload_structure():
    from services.scheduler_catalog import (
        STRATEGY_DEFINITIONS,
        INDICATOR_DEFINITIONS,
        KITE_INTERVALS,
        get_catalog_payload,
    )

    payload = get_catalog_payload()
    assert payload["version"]
    assert isinstance(payload["strategies"], list)
    assert isinstance(payload["indicators"], list)
    assert isinstance(payload["intervals"], list)

    for s in STRATEGY_DEFINITIONS:
        assert "id" in s and "name" in s and "description" in s
    for ind in INDICATOR_DEFINITIONS:
        assert "id" in ind and "name" in ind
    for iv in KITE_INTERVALS:
        assert "id" in iv and "label" in iv

    ids = [x["id"] for x in KITE_INTERVALS]
    assert len(ids) == len(set(ids)), "duplicate interval ids"


def test_operation_type_labels_complete():
    from services.telegram_scheduler import OperationType

    mod = _load_telegram_scheduler_routes()
    labels = mod.OPERATION_TYPE_LABELS
    for op in OperationType:
        assert op.value in labels
        assert len(labels[op.value].strip()) > 0


def test_build_indicator_report_text_all_known_indicators():
    from services.scheduler_indicator_service import build_indicator_report_text

    base = 24500.0
    candles = []
    for i in range(80):
        c = base + i * 2.5 + (i % 5) * 0.5
        candles.append(
            {
                "open": c - 1,
                "high": c + 3,
                "low": c - 4,
                "close": c,
                "volume": 10000 + i * 100,
            }
        )

    ids = ["rsi_14", "ema_9_21", "vwap_distance", "bollinger_20", "pivots_daily"]
    text = build_indicator_report_text(
        "NIFTY-50", "NSE:NIFTY 50", "5minute", candles, ids
    )
    assert "Instrument: NIFTY-50" in text
    assert "Bars: 80" in text
    assert "RSI" in text
    assert "EMA" in text
    assert "VWAP" in text or "vwap" in text.lower()
    assert "Bollinger" in text
    assert "Pivots" in text or "pivot" in text.lower()


def test_build_indicator_report_empty_candles():
    from services.scheduler_indicator_service import build_indicator_report_text

    t = build_indicator_report_text("X", "NSE:FOO", "day", [], ["rsi_14"])
    assert "No historical candles" in t


def test_build_indicator_report_unknown_indicator():
    from services.scheduler_indicator_service import build_indicator_report_text

    candles = [{"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100}] * 30
    t = build_indicator_report_text("X", "NSE:FOO", "minute", candles, ["not_a_real_id"])
    assert "Unknown indicator" in t


def test_fetch_candles_sync_uses_kite_historical():
    from services.scheduler_indicator_service import fetch_candles_sync

    kite = MagicMock()
    kite.historical_data.return_value = [
        {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100}
    ]
    out = fetch_candles_sync(kite, 12345, "5minute")
    assert len(out) == 1
    kite.historical_data.assert_called_once()


@pytest.fixture
def scheduler_api_app():
    """Minimal FastAPI app with only telegram-scheduler routes (no full main.py)."""
    from fastapi import FastAPI

    ts_routes = _load_telegram_scheduler_routes()
    app = FastAPI()
    app.include_router(ts_routes.router, prefix="/api/v1")
    return app


def test_api_catalog_and_operation_types(scheduler_api_app):
    from starlette.testclient import TestClient

    client = TestClient(scheduler_api_app)

    r = client.get("/api/v1/telegram-scheduler/catalog")
    assert r.status_code == 200
    body = r.json()
    assert body.get("success") is True
    data = body["data"]
    assert "strategies" in data and "indicators" in data and "intervals" in data
    assert data.get("version")

    r2 = client.get("/api/v1/telegram-scheduler/operation-types")
    assert r2.status_code == 200
    j = r2.json()
    assert j.get("success") is True
    values = {x["value"] for x in j["data"]}
    assert "indicator_report" in values
    assert "strategy_overview" in values
    labels = {x["value"]: x["label"] for x in j["data"]}
    assert "Kite" in labels["indicator_report"] or "indicator" in labels["indicator_report"].lower()


def test_send_strategy_overview_notification_content():
    from services.telegram_scheduler import (
        TelegramScheduler,
        ScheduledTask,
        ScheduleType,
        OperationType,
    )

    captured = []

    async def capture(n):
        captured.append(n)

    sched = TelegramScheduler()
    sched._notify_telegram = capture  # type: ignore[method-assign]

    task = ScheduledTask(
        id="test-so-1",
        name="SO Test",
        description="d",
        schedule_type=ScheduleType.DAILY,
        operation_type=OperationType.STRATEGY_OVERVIEW,
        schedule_config={"time": "09:00"},
        operation_config={"strategy_ids": ["nifty50_options"]},
    )

    async def run():
        await sched._send_strategy_overview(task, {"strategy_ids": ["nifty50_options"]})

    asyncio.run(run())

    assert len(captured) == 1
    msg = captured[0].message
    assert "nifty50_options" in msg or "Nifty" in msg
    assert "Strategy" in msg or "strategy" in msg.lower()


def test_send_strategy_overview_all_strategies_when_empty_filter():
    from services.telegram_scheduler import (
        TelegramScheduler,
        ScheduledTask,
        ScheduleType,
        OperationType,
    )
    from services.scheduler_catalog import STRATEGY_DEFINITIONS

    captured = []

    async def capture(n):
        captured.append(n)

    sched = TelegramScheduler()
    sched._notify_telegram = capture  # type: ignore[method-assign]

    task = ScheduledTask(
        id="test-so-2",
        name="SO All",
        description="d",
        schedule_type=ScheduleType.DAILY,
        operation_type=OperationType.STRATEGY_OVERVIEW,
        schedule_config={"time": "09:00"},
        operation_config={},
    )

    async def run():
        await sched._send_strategy_overview(task, {})

    asyncio.run(run())

    assert len(captured) == 1
    msg = captured[0].message
    # Every catalog strategy should appear when filter is empty
    for s in STRATEGY_DEFINITIONS:
        assert s["id"] in msg or s["name"][:4] in msg
