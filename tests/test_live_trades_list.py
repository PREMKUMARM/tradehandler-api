"""Unit tests for live Kite order grouping."""
from services.live_trades_list import _group_live_trades, _premium_pnl


def _buy(oid, sym, px, qty=1, ex="MCX", ts="2026-05-29 20:21:39"):
    return {
        "order_id": oid,
        "exchange": ex,
        "tradingsymbol": sym,
        "product": "NRML",
        "transaction_type": "BUY",
        "order_type": "LIMIT",
        "status": "COMPLETE",
        "quantity": qty,
        "filled_quantity": qty,
        "average_price": px,
        "exchange_timestamp": ts,
    }


def _sell(oid, sym, px, qty=1, ex="MCX", ts="2026-05-29 20:22:13"):
    return {
        "order_id": oid,
        "exchange": ex,
        "tradingsymbol": sym,
        "product": "NRML",
        "transaction_type": "SELL",
        "order_type": "LIMIT",
        "status": "COMPLETE",
        "quantity": qty,
        "filled_quantity": qty,
        "average_price": px,
        "exchange_timestamp": ts,
    }


def test_premium_pnl_mcx_multiplier():
    pnl = _premium_pnl(418.9, 429.0, 1, "MCX", "CRUDEOILM26JUN8350PE", buy=True)
    assert pnl == 101.0


def test_group_closed_round_trip():
    sym = "CRUDEOILM26JUN8350PE"
    orders = [
        _buy("entry1", sym, 418.9),
        _sell("exit1", sym, 429.0),
    ]
    rows = _group_live_trades(orders, seg_filter="commodity", audit_sl_tp={}, gtt_by_sym={}, positions={})
    assert len(rows) == 1
    assert rows[0]["status"] == "closed"
    assert rows[0]["pnl"] == 101.0
    assert rows[0]["exit_order_id"] == "exit1"


def _open_buy(oid, sym, limit_px, qty=1, ex="MCX", ts="2026-05-29 21:25:56", filled=0):
    return {
        "order_id": oid,
        "exchange": ex,
        "tradingsymbol": sym,
        "product": "NRML",
        "transaction_type": "BUY",
        "order_type": "LIMIT",
        "status": "OPEN",
        "quantity": qty,
        "filled_quantity": filled,
        "price": limit_px,
        "average_price": 0,
        "exchange_timestamp": ts,
    }


def test_group_pending_open_limit():
    sym = "CRUDEOILM26JUN8300PE"
    orders = [_open_buy("pending1", sym, 404.1)]
    rows = _group_live_trades(
        orders,
        seg_filter="commodity",
        audit_sl_tp={"pending1": (398.4, 421.2)},
        gtt_by_sym={},
        positions={},
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["entry_price"] == 404.1
    assert rows[0]["stoploss"] == 398.4
    assert rows[0]["target"] == 421.2
    assert rows[0]["order_id"] == "pending1"


def test_group_open_position():
    sym = "CRUDEOILM26JUN8350PE"
    orders = [_buy("entry1", sym, 418.9)]
    positions = {
        sym: {"quantity": 1, "last_price": 448, "pnl": 291.0},
    }
    rows = _group_live_trades(
        orders,
        seg_filter="commodity",
        audit_sl_tp={"entry1": (409.35, 429.0)},
        gtt_by_sym={},
        positions=positions,
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "open"
    assert rows[0]["stoploss"] == 409.35
    assert rows[0]["target"] == 429.0
    assert rows[0]["pnl"] == 291.0
