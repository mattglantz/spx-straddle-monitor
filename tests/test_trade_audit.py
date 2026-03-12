"""
Unit tests for trade audit functions.
"""

import pytest
from trade_audit import _calc_trailing_stop, close_trade_at_price
from bot_config import CFG
import trade_status as ts


class TestTrailingStop:
    def test_no_change_below_trigger(self):
        new_stop, changed = _calc_trailing_stop(5800.0, 5794.0, 5.0, long=True)
        assert changed == False
        assert new_stop == 5794.0

    def test_breakeven_at_first_level(self):
        # Default first level: +10 pts -> stop to entry
        new_stop, changed = _calc_trailing_stop(5800.0, 5794.0, 10.0, long=True)
        assert changed == True
        assert new_stop == 5800.0  # entry = breakeven

    def test_progressive_trail_long(self):
        # +15 pts -> stop to entry + 5
        new_stop, changed = _calc_trailing_stop(5800.0, 5794.0, 15.0, long=True)
        assert changed == True
        assert new_stop == 5805.0

    def test_short_breakeven(self):
        # Short: entry 5800, stop 5806, float +10 -> stop to entry
        new_stop, changed = _calc_trailing_stop(5800.0, 5806.0, 10.0, long=False)
        assert changed == True
        assert new_stop == 5800.0

    def test_short_progressive(self):
        # Short: +15 pts -> stop to entry - 5
        new_stop, changed = _calc_trailing_stop(5800.0, 5806.0, 15.0, long=False)
        assert changed == True
        assert new_stop == 5795.0

    def test_stop_never_moves_backward_long(self):
        # If stop is already at 5805, don't move it back to 5800
        new_stop, changed = _calc_trailing_stop(5800.0, 5805.0, 10.0, long=True)
        assert changed == False
        assert new_stop == 5805.0

    def test_atr_scaled_levels(self):
        # With ATR=10, base = 4.0
        # First level: 4.0 float -> breakeven
        new_stop, changed = _calc_trailing_stop(5800.0, 5794.0, 4.0, long=True, atr=10.0)
        assert changed == True
        assert new_stop == 5800.0  # breakeven

    def test_atr_scaled_second_level(self):
        # ATR=10, base=4.0, second level: 6.0 float -> lock 2.0
        new_stop, changed = _calc_trailing_stop(5800.0, 5794.0, 6.0, long=True, atr=10.0)
        assert changed == True
        assert new_stop == 5802.0  # entry + 0.5*base = 5800 + 2


class TestCloseTrade:
    """Test the shared close_trade_at_price function."""

    class MockJournal:
        def __init__(self):
            self.updates = []
        def update_trade(self, trade_id, status, pnl, **kwargs):
            self.updates.append({"id": trade_id, "status": status, "pnl": pnl})

    def test_long_win(self):
        j = self.MockJournal()
        trade = {"id": 1, "price": 5800, "stop": 5794, "verdict": "BULLISH", "contracts": 1}
        outcome, pnl, price = close_trade_at_price(trade, 5815.0, j)
        assert outcome == ts.WIN
        assert pnl == 15.0

    def test_long_loss_capped(self):
        j = self.MockJournal()
        trade = {"id": 1, "price": 5800, "stop": 5794, "verdict": "BULLISH", "contracts": 1}
        outcome, pnl, price = close_trade_at_price(trade, 5780.0, j)
        assert outcome == ts.STOP
        assert pnl == -6.0
        assert price == 5794.0

    def test_short_win(self):
        j = self.MockJournal()
        trade = {"id": 1, "price": 5800, "stop": 5806, "verdict": "BEARISH", "contracts": 1}
        outcome, pnl, price = close_trade_at_price(trade, 5785.0, j)
        assert outcome == ts.WIN
        assert pnl == 15.0
