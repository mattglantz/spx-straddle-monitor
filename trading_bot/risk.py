"""
Risk management — the gatekeeper between strategy and execution.

Every trade decision must pass through risk management before an
order is placed. This module enforces:

1. Position sizing based on ATR and account equity
2. Daily loss circuit breaker
3. Max trades per day
4. No new trades near RTH close
5. Reduced size on economic event days
6. Post-loss cooldown period
7. Sanity checks (spread too wide, no data, etc.)
"""

import logging
from datetime import datetime, time, timedelta
from typing import Optional

from trading_bot.config import (
    RiskConfig, ES_POINT_VALUE, ET,
    RTH_CLOSE, NO_NEW_TRADES_BEFORE_CLOSE_MIN,
)
from trading_bot.signals import SignalSnapshot, Regime, is_event_day
from trading_bot.strategy import Strategy, TradeDecision, Action
from trading_bot.data import MarketData

log = logging.getLogger(__name__)


class RiskManager:
    """
    Validates trade decisions against risk rules.
    Can reduce size, block trades, or force exits.
    """

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def check_entry(self, decision: TradeDecision, strategy: Strategy,
                    data: MarketData, signal: SignalSnapshot) -> tuple[bool, int, str]:
        """
        Validate a proposed entry.
        Returns (allowed, adjusted_quantity, reason_if_blocked).
        """
        reasons = []

        # 1. Daily loss limit
        if strategy.daily_pnl <= -self.cfg.max_daily_loss_dollars:
            return False, 0, (
                f"Daily loss limit hit: ${strategy.daily_pnl:.2f} "
                f"(max -${self.cfg.max_daily_loss_dollars:.2f})"
            )

        # 2. Max trades per day
        if strategy.trade_count_today >= self.cfg.max_trades_per_day:
            return False, 0, (
                f"Max trades ({self.cfg.max_trades_per_day}) reached for today"
            )

        # 3. Post-loss cooldown
        if strategy.in_cooldown:
            return False, 0, (
                f"In cooldown after loss ({self.cfg.loss_cooldown_sec}s)"
            )

        # 4. No new trades near close
        now_et = datetime.now(ET)
        close_cutoff = datetime.combine(
            now_et.date(),
            time(RTH_CLOSE.hour, RTH_CLOSE.minute),
            tzinfo=ET,
        ) - timedelta(minutes=NO_NEW_TRADES_BEFORE_CLOSE_MIN)
        if data.is_rth() and now_et >= close_cutoff:
            return False, 0, (
                f"Too close to RTH close (cutoff {close_cutoff.strftime('%H:%M')})"
            )

        # 5. Volatile regime — block entry
        if signal.regime == Regime.VOLATILE:
            return False, 0, "Volatile regime — no new entries"

        # 6. ATR sanity — need valid ATR
        if data.atr <= 0:
            return False, 0, "ATR not available — insufficient bar data"

        # 7. Spread sanity — don't trade if spread is unusually wide
        if data.spread > data.atr * 0.5 and data.spread > 0:
            return False, 0, (
                f"Spread too wide: {data.spread:.2f} vs ATR {data.atr:.2f}"
            )

        # 8. Calculate position size
        qty = self._calculate_size(data, signal)
        if qty <= 0:
            return False, 0, "Position size calculated as 0"

        return True, qty, ""

    def _calculate_size(self, data: MarketData, signal: SignalSnapshot) -> int:
        """
        Determine position size.

        Base size = max_position_contracts, reduced by:
        - Event days: multiply by event_day_size_fraction
        - High VIX: scale down proportionally
        - Weak signal: reduce if composite barely above threshold
        """
        base_size = self.cfg.max_position_contracts

        # Event day reduction
        now_et = datetime.now(ET)
        if is_event_day(now_et):
            base_size = max(1, int(base_size * self.cfg.event_day_size_fraction))
            log.info(f"Event day: size reduced to {base_size}")

        # VIX-based reduction (linear scale from vix=20 to vix=26)
        if signal.vix > 20:
            vix_factor = max(0.5, 1.0 - (signal.vix - 20) / 12)
            base_size = max(1, int(base_size * vix_factor))

        return min(base_size, self.cfg.max_position_contracts)

    def should_force_exit(self, strategy: Strategy, data: MarketData) -> tuple[bool, str]:
        """
        Check if we need to force-exit an existing position.
        Called independently from strategy evaluation.
        """
        if strategy.position is None:
            return False, ""

        # Force flat if daily loss limit is being approached with open position
        unrealized = self._estimate_unrealized(strategy, data)
        total_pnl = strategy.daily_pnl + unrealized
        if total_pnl <= -self.cfg.max_daily_loss_dollars:
            return True, (
                f"Force exit: daily PnL with unrealized = ${total_pnl:.2f} "
                f"exceeds limit -${self.cfg.max_daily_loss_dollars:.2f}"
            )

        # Force flat at end of day (if configured for no overnight)
        now_et = datetime.now(ET)
        if data.is_rth():
            eod = datetime.combine(
                now_et.date(), time(15, 58), tzinfo=ET)
            if now_et >= eod:
                return True, "End-of-day flat: closing before 4:00 PM"

        return False, ""

    def _estimate_unrealized(self, strategy: Strategy, data: MarketData) -> float:
        """Estimate unrealized PnL on current position."""
        pos = strategy.position
        if pos is None:
            return 0.0
        price = data.last_price
        if price <= 0:
            return 0.0
        if pos.side.value == "long":
            pnl_pts = price - pos.entry_price
        else:
            pnl_pts = pos.entry_price - price
        return pnl_pts * ES_POINT_VALUE * pos.quantity
