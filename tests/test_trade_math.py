"""
Unit tests for P&L math and trade state transitions.

These replicate the exact formulas from market_bot_v26.py without importing it
(that module has heavy side effects: win32gui, IBKR, anthropic, dotenv).
"""
import pytest


# =====================================================================
# Pure math functions extracted from market_bot_v26.py
# =====================================================================

def calc_pnl(entry, exit_price, is_long):
    """Basic P&L: lines 1676, 1752, 1800."""
    return (exit_price - entry) if is_long else (entry - exit_price)


def calc_time_exit_pnl(entry, stop, curr_price, is_long):
    """Time-exit with stop capping and P&L-based classification."""
    pnl = (curr_price - entry) if is_long else (entry - curr_price)
    stop_pnl = (stop - entry) if is_long else (entry - stop)
    outcome = "TIME-EXIT"
    if pnl < 0 and stop_pnl < 0 and pnl < stop_pnl:
        pnl = stop_pnl
        outcome = "STOP"
    elif pnl > 0:
        outcome = "WIN"
    elif pnl == 0:
        outcome = "BREAKEVEN"
    return pnl, outcome


def calc_reversal_pnl(t_entry, t_stop, curr_price, t_is_long):
    """Reversal closure with stop capping: lines 5057-5070."""
    close_price = curr_price
    if t_is_long:
        pnl = curr_price - t_entry
        if pnl < 0 and t_stop > 0:
            stop_loss = t_stop - t_entry
            if pnl < stop_loss:
                pnl = stop_loss
                close_price = t_stop
    else:
        pnl = t_entry - curr_price
        if pnl < 0 and t_stop > 0:
            stop_loss = t_entry - t_stop
            if pnl < stop_loss:
                pnl = stop_loss
                close_price = t_stop
    return pnl, close_price


def classify_stop_status(pnl_pts):
    """Stop-out status classification: lines 1744, 1800, 5071."""
    if pnl_pts < 0:
        return "LOSS"
    if pnl_pts > 0:
        return "WIN"
    return "BREAKEVEN"


def calc_win_rate(closed_trades):
    """Win rate from get_today_stats."""
    wins = sum(1 for t in closed_trades if t["status"] == "WIN")
    losses = sum(1 for t in closed_trades if t["status"] in ("LOSS", "TIME-EXIT", "STOP"))
    return (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0


def time_exit_close_price(outcome, curr_price, stop):
    """Which price to show in Telegram message."""
    if outcome == "STOP":
        return stop
    else:  # TIME-EXIT, WIN, BREAKEVEN — all use market price
        return curr_price


# =====================================================================
# Tests
# =====================================================================

class TestLongPnL:
    def test_long_win(self):
        # Entry 5800, target 5815 → +15 pts
        assert calc_pnl(5800, 5815, is_long=True) == 15.0

    def test_long_loss(self):
        # Entry 5800, stop 5790 → -10 pts
        assert calc_pnl(5800, 5790, is_long=True) == -10.0

    def test_long_breakeven(self):
        assert calc_pnl(5800, 5800, is_long=True) == 0.0

    def test_long_fractional(self):
        # Entry 5800.25, target 5812.50 → +12.25
        assert calc_pnl(5800.25, 5812.50, is_long=True) == pytest.approx(12.25)


class TestShortPnL:
    def test_short_win(self):
        # Entry 5800, target 5785 → +15 pts (price went down)
        assert calc_pnl(5800, 5785, is_long=False) == 15.0

    def test_short_loss(self):
        # Entry 5800, stop 5810 → -10 pts (price went up)
        assert calc_pnl(5800, 5810, is_long=False) == -10.0

    def test_short_breakeven(self):
        assert calc_pnl(5800, 5800, is_long=False) == 0.0

    def test_short_fractional(self):
        # Entry 5800.75, target 5788.50 → +12.25
        assert calc_pnl(5800.75, 5788.50, is_long=False) == pytest.approx(12.25)


class TestTimeExitCapping:
    def test_long_capped_at_stop(self):
        # Long: entry 5800, stop 5794, curr 5780 (ran way past stop)
        # Raw P&L: -20, stop_pnl: -6. Capped to -6.
        pnl, outcome = calc_time_exit_pnl(5800, 5794, 5780, is_long=True)
        assert pnl == -6.0
        assert outcome == "STOP"

    def test_short_capped_at_stop(self):
        # Short: entry 5800, stop 5806, curr 5838.50 (ran way past stop)
        # Raw P&L: -38.50, stop_pnl: -6. Capped to -6.
        pnl, outcome = calc_time_exit_pnl(5800, 5806, 5838.50, is_long=False)
        assert pnl == -6.0
        assert outcome == "STOP"

    def test_long_within_stop_not_capped(self):
        # Long: entry 5800, stop 5794, curr 5797 (loss but within stop)
        # Raw P&L: -3, stop_pnl: -6. -3 > -6, so NOT capped.
        pnl, outcome = calc_time_exit_pnl(5800, 5794, 5797, is_long=True)
        assert pnl == -3.0
        assert outcome == "TIME-EXIT"

    def test_short_within_stop_not_capped(self):
        # Short: entry 5800, stop 5806, curr 5803 (loss but within stop)
        # Raw P&L: -3, stop_pnl: -6. -3 > -6, so NOT capped.
        pnl, outcome = calc_time_exit_pnl(5800, 5806, 5803, is_long=False)
        assert pnl == -3.0
        assert outcome == "TIME-EXIT"

    def test_profitable_time_exit_classified_as_win(self):
        # Long: entry 5800, stop 5794, curr 5810 (profitable exit)
        pnl, outcome = calc_time_exit_pnl(5800, 5794, 5810, is_long=True)
        assert pnl == 10.0
        assert outcome == "WIN"

    def test_short_profitable_time_exit_classified_as_win(self):
        # Short: entry 5800, stop 5806, curr 5790
        pnl, outcome = calc_time_exit_pnl(5800, 5806, 5790, is_long=False)
        assert pnl == 10.0
        assert outcome == "WIN"

    def test_breakeven_time_exit_classified_as_breakeven(self):
        # Long: entry 5800, stop 5794, curr 5800 (flat at entry)
        pnl, outcome = calc_time_exit_pnl(5800, 5794, 5800, is_long=True)
        assert pnl == 0.0
        assert outcome == "BREAKEVEN"

    def test_reclassified_to_stop(self):
        # Verify the outcome changes when capped
        _, outcome = calc_time_exit_pnl(5800, 5794, 5770, is_long=True)
        assert outcome == "STOP"

    def test_exactly_at_stop_not_capped(self):
        # Long: entry 5800, stop 5794, curr 5794 (exactly at stop level)
        # Raw P&L: -6, stop_pnl: -6. pnl == stop_pnl, NOT < stop_pnl.
        pnl, outcome = calc_time_exit_pnl(5800, 5794, 5794, is_long=True)
        assert pnl == -6.0
        assert outcome == "TIME-EXIT"  # not capped because pnl is not LESS than stop_pnl

    def test_breakeven_stop_not_capped(self):
        # Long: entry 5800, stop moved to breakeven (5800), curr 5798
        # Raw P&L: -2, stop_pnl: 0. stop_pnl is NOT < 0, so cap doesn't fire.
        # Loss should be reported at actual market value, not capped to 0.
        pnl, outcome = calc_time_exit_pnl(5800, 5800, 5798, is_long=True)
        assert pnl == -2.0
        assert outcome == "TIME-EXIT"  # real loss, not masked as breakeven

    def test_short_breakeven_stop_not_capped(self):
        # Short: entry 5800, stop moved to breakeven (5800), curr 5803
        # Raw P&L: -3, stop_pnl: 0. stop_pnl is NOT < 0, so cap doesn't fire.
        pnl, outcome = calc_time_exit_pnl(5800, 5800, 5803, is_long=False)
        assert pnl == -3.0
        assert outcome == "TIME-EXIT"  # real loss, not masked as breakeven


class TestBreakevenClassification:
    def test_breakeven(self):
        assert classify_stop_status(0.0) == "BREAKEVEN"

    def test_loss(self):
        assert classify_stop_status(-5.0) == "LOSS"

    def test_win(self):
        assert classify_stop_status(3.0) == "WIN"

    def test_small_loss(self):
        assert classify_stop_status(-0.25) == "LOSS"


class TestReversalClosure:
    def test_long_reversal_capped(self):
        # Long: entry 5800, stop 5794, curr 5780 (price crashed)
        # Raw P&L: -20, stop_loss: -6. Capped to -6.
        pnl, close_price = calc_reversal_pnl(5800, 5794, 5780, t_is_long=True)
        assert pnl == -6.0
        assert close_price == 5794  # shows stop price, not market price

    def test_short_reversal_capped(self):
        # Short: entry 5800, stop 5806, curr 5830
        # Raw P&L: -30, stop_loss: -6. Capped to -6.
        pnl, close_price = calc_reversal_pnl(5800, 5806, 5830, t_is_long=False)
        assert pnl == -6.0
        assert close_price == 5806

    def test_long_reversal_not_capped(self):
        # Long: entry 5800, stop 5794, curr 5797 (small loss within stop)
        pnl, close_price = calc_reversal_pnl(5800, 5794, 5797, t_is_long=True)
        assert pnl == -3.0
        assert close_price == 5797  # market price (not capped)

    def test_profitable_reversal(self):
        # Long: entry 5800, stop 5794, curr 5810 (profitable)
        pnl, close_price = calc_reversal_pnl(5800, 5794, 5810, t_is_long=True)
        assert pnl == 10.0
        assert close_price == 5810

    def test_display_price_matches_pnl(self):
        """The bug we originally found: display price must be consistent with P&L."""
        entry, stop, curr = 6796.0, 6802.0, 6838.50  # original example from Telegram
        pnl, close_price = calc_reversal_pnl(entry, stop, curr, t_is_long=False)
        # P&L should be capped at stop: entry - stop = 6796 - 6802 = -6
        assert pnl == -6.0
        assert close_price == stop  # must show stop price, not 6838.50
        # Verify the math adds up: entry - close_price == pnl
        assert entry - close_price == pnl


class TestWinRate:
    def test_all_wins(self):
        trades = [{"status": "WIN"}, {"status": "WIN"}, {"status": "WIN"}]
        assert calc_win_rate(trades) == 100.0

    def test_all_losses(self):
        trades = [{"status": "LOSS"}, {"status": "LOSS"}]
        assert calc_win_rate(trades) == 0.0

    def test_mixed(self):
        trades = [{"status": "WIN"}, {"status": "LOSS"}, {"status": "WIN"}, {"status": "LOSS"}]
        assert calc_win_rate(trades) == 50.0

    def test_empty(self):
        assert calc_win_rate([]) == 0

    def test_time_exit_counts_as_loss(self):
        trades = [{"status": "WIN"}, {"status": "TIME-EXIT"}]
        assert calc_win_rate(trades) == 50.0

    def test_stop_counts_as_loss(self):
        trades = [{"status": "WIN"}, {"status": "STOP"}]
        assert calc_win_rate(trades) == 50.0

    def test_breakeven_excluded_from_win_and_loss(self):
        # BREAKEVEN is in closed trades but not WIN or LOSS_STATUSES
        trades = [{"status": "WIN"}, {"status": "BREAKEVEN"}, {"status": "LOSS"}]
        # wins=1, losses=1, breakeven doesn't count → 50%
        assert calc_win_rate(trades) == 50.0

    def test_all_breakeven(self):
        trades = [{"status": "BREAKEVEN"}, {"status": "BREAKEVEN"}]
        # wins=0, losses=0 → 0 (division guard)
        assert calc_win_rate(trades) == 0


class TestDailyLossLimit:
    def test_limit_is_in_dollars(self):
        """MAX_DAILY_LOSS = -500.0 means $500 max loss."""
        max_daily_loss = -500.0  # from CFG (dollars)
        # Realized is in contract-weighted points, convert to dollars: pts * 50
        realized_pts = -10.0  # 10 pts loss
        loss_dollars = realized_pts * 50  # = -$500
        assert loss_dollars == max_daily_loss

    def test_under_limit_pauses(self):
        realized_pts = -12.0  # 12 pts loss = $600
        loss_dollars = realized_pts * 50
        max_daily_loss = -500.0
        assert loss_dollars <= max_daily_loss  # -$600 <= -$500, should trigger pause

    def test_over_limit_continues(self):
        realized_pts = -8.0  # 8 pts loss = $400
        loss_dollars = realized_pts * 50
        max_daily_loss = -500.0
        assert not (loss_dollars <= max_daily_loss)  # -$400 > -$500, should NOT pause

    def test_exactly_at_limit_pauses(self):
        realized_pts = -10.0  # = -$500
        loss_dollars = realized_pts * 50
        max_daily_loss = -500.0
        assert loss_dollars <= max_daily_loss  # boundary: pauses


class TestTimeExitDisplayPrice:
    def test_uncapped_shows_market_price(self):
        # Profitable time exit is now WIN, but display still uses market price
        assert time_exit_close_price("WIN", 5810.0, 5794.0) == 5810.0

    def test_loss_time_exit_shows_market_price(self):
        # Unprofitable time exit within stop keeps TIME-EXIT status, uses market price
        assert time_exit_close_price("TIME-EXIT", 5797.0, 5794.0) == 5797.0

    def test_capped_shows_stop_price(self):
        assert time_exit_close_price("STOP", 5770.0, 5794.0) == 5794.0


class TestEodClose:
    """EOD force-close at 4:00 PM uses the same time-exit math."""

    def test_long_profitable_eod(self):
        # Long: entry 5800, stop 5794, curr 5815 → +15 WIN
        pnl, outcome = calc_time_exit_pnl(5800, 5794, 5815, is_long=True)
        assert pnl == 15.0
        assert outcome == "WIN"
        assert time_exit_close_price(outcome, 5815, 5794) == 5815  # market price

    def test_short_loss_capped_eod(self):
        # Short: entry 5800, stop 5808, curr 5830 → raw -30, capped to -8
        pnl, outcome = calc_time_exit_pnl(5800, 5808, 5830, is_long=False)
        assert pnl == -8.0
        assert outcome == "STOP"
        assert time_exit_close_price(outcome, 5830, 5808) == 5808  # stop price

    def test_long_small_loss_eod(self):
        # Long: entry 5800, stop 5794, curr 5797 → -3, within stop, TIME-EXIT
        pnl, outcome = calc_time_exit_pnl(5800, 5794, 5797, is_long=True)
        assert pnl == -3.0
        assert outcome == "TIME-EXIT"
        assert time_exit_close_price(outcome, 5797, 5794) == 5797  # market price

    def test_eod_pnl_dollars(self):
        """EOD close P&L in dollars: pnl * $50 * contracts."""
        pnl = 7.5  # points
        cts = 2
        pnl_dollars = pnl * 50 * cts
        assert pnl_dollars == 750.0
