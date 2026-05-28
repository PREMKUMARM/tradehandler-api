"""Tests for daily P&L sync into trade_limits."""
from unittest.mock import MagicMock, patch

from services import pnl_sync
from utils.trade_limits import trade_limits


def test_sync_sums_all_exchanges_not_segment_filtered():
  trade_limits.limits["pnl_inr_today"] = 0.0
  mock_kite = MagicMock()
  mock_kite.positions.return_value = {
      "net": [
          {"exchange": "NFO", "pnl": 100.0},
          {"exchange": "MCX", "pnl": -50.0},
      ]
  }
  pnl_sync._last_sync_at = None
  with patch("utils.kite_utils.get_kite_instance", return_value=mock_kite):
      result = pnl_sync.sync_daily_pnl_from_kite(force=True)
  assert result["ok"] is True
  assert result["pnl_inr_today"] == 50.0
  assert float(trade_limits.limits["pnl_inr_today"]) == 50.0
