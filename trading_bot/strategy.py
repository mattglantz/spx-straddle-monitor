"""
Strategy engine — translates signals into trade decisions.

The strategy adapts its behavior to the detected regime:

MEAN_REVERT mode:
  - Enter counter-trend when VWAP z-score is extreme and composite confirms
  - Tight stops (1.5 ATR), targets back toward VWAP (1.5 ATR)
  - Quick exits — don't overstay the fade

TRENDING mode:
  - Enter with-trend on pullbacks (wait for composite to confirm direction)
  - Wider stops (2.5 ATR), larger targets (4 ATR)
  - Trailing stop once in profit to ride the move

VOLATILE mode:
  - No new entries (or very small size via risk module)
  - Tighten stops on existing positions
"""

import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from trading_bot.config import RiskConfig, SignalConfig, ES_POINT_VALUE, ET
from trading_bot.signals import SignalSnapshot, Regime
from trading_bot.data import MarketData

log = logging.getLogger(__name__)


class Side(Enum):
    LONG = "long"
    SHORT = "short"


class Action(Enum):
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"
    ADJUST_STOP = "adjust_stop"
    HOLD = "hold"


@dataclass
class TradeDecision:
    action: Action
    side: Optional[Side] = None
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    reason: str = ""
    regime: Regime = Regime.MEAN_REVERT


@dataclass
class Position:
    """Tracks a live position."""
    side: Side
    entry_price: float
    stop_price: float
    target_price: float
    quantity: int
    entry_time: datetime
    regime_at_entry: Regime
    trail_stop: float = 0.0        # trailing stop level (0 = not active)
    highest_since_entry: float = 0.0  # for long trailing
    lowest_since_entry: float = float("inf")  # for short trailing
    order_id: Optional[int] = None

    @property
    def unrealized_pnl(self) -> float:
        """Estimated PnL based on last price. Caller must provide price."""
        return 0.0  # calculated externally

    def update_trail(self, price: float, trail_atr: float, atr: float):
        """Update trailing stop if in trend mode."""
        if trail_atr <= 0 or atr <= 0:
            return
        trail_dist = trail_atr * atr
        if self.side == Side.LONG:
            self.highest_since_entry = max(self.highest_since_entry, price)
            new_trail = self.highest_since_entry - trail_dist
            if new_trail > self.trail_stop:
                self.trail_stop = new_trail
                # Only tighten the real stop, never widen
                if self.trail_stop > self.stop_price:
                    self.stop_price = self.trail_stop
        else:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            new_trail = self.lowest_since_entry + trail_dist
            if self.trail_stop == 0 or new_trail < self.trail_stop:
                self.trail_stop = new_trail
                if self.trail_stop < self.stop_price:
                    self.stop_price = self.trail_stop


class Strategy:
    """
    Core decision engine. Given current data, signals, and position state,
    decides what to do.
    """

    def __init__(self, signal_cfg: SignalConfig, risk_cfg: RiskConfig):
        self.signal_cfg = signal_cfg
        self.risk_cfg = risk_cfg
        self.position: Optional[Position] = None
        self.trades_today: list[dict] = []
        self._last_loss_time: Optional[datetime] = None

    def evaluate(self, data: MarketData, signal: SignalSnapshot) -> TradeDecision:
        """Main decision loop. Called on each evaluation cycle."""

        # ── Manage existing position ────────────────────────
        if self.position is not None:
            return self._manage_position(data, signal)

        # ── Look for new entry ──────────────────────────────
        return self._evaluate_entry(data, signal)

    def _evaluate_entry(self, data: MarketData, signal: SignalSnapshot) -> TradeDecision:
        """Determine whether to enter a new trade."""
        composite = signal.composite
        regime = signal.regime
        atr = data.atr
        price = data.last_price

        # No entries in VOLATILE regime
        if regime == Regime.VOLATILE:
            return TradeDecision(Action.HOLD, reason="Volatile regime — standing aside")

        # Need sufficient data
        if atr <= 0 or price <= 0:
            return TradeDecision(Action.HOLD, reason="Insufficient data for ATR")

        # Signal must exceed threshold
        if abs(composite) < self.signal_cfg.entry_threshold:
            return TradeDecision(
                Action.HOLD,
                reason=f"Signal {composite:.0f} below threshold {self.signal_cfg.entry_threshold}"
            )

        # Determine direction
        if composite > 0:
            side = Side.LONG
            stop_mult, target_mult = self._get_stop_target(regime)
            stop = price - stop_mult * atr
            target = price + target_mult * atr
            return TradeDecision(
                action=Action.ENTER_LONG,
                side=side,
                entry_price=price,
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                reason=self._entry_reason(signal, side),
                regime=regime,
            )
        else:
            side = Side.SHORT
            stop_mult, target_mult = self._get_stop_target(regime)
            stop = price + stop_mult * atr
            target = price - target_mult * atr
            return TradeDecision(
                action=Action.ENTER_SHORT,
                side=side,
                entry_price=price,
                stop_price=round(stop, 2),
                target_price=round(target, 2),
                reason=self._entry_reason(signal, side),
                regime=regime,
            )

    def _manage_position(self, data: MarketData, signal: SignalSnapshot) -> TradeDecision:
        """Manage an existing position — trailing stops, exits, etc."""
        pos = self.position
        price = data.last_price
        atr = data.atr

        # Update trailing stop for trend trades
        if pos.regime_at_entry == Regime.TRENDING and atr > 0:
            pos.update_trail(price, self.risk_cfg.trend_trail_atr, atr)

        # Check stop hit
        if pos.side == Side.LONG and price <= pos.stop_price:
            return TradeDecision(
                Action.EXIT, side=pos.side,
                reason=f"Stop hit at {pos.stop_price:.2f} (price={price:.2f})"
            )
        if pos.side == Side.SHORT and price >= pos.stop_price:
            return TradeDecision(
                Action.EXIT, side=pos.side,
                reason=f"Stop hit at {pos.stop_price:.2f} (price={price:.2f})"
            )

        # Check target hit
        if pos.side == Side.LONG and price >= pos.target_price:
            return TradeDecision(
                Action.EXIT, side=pos.side,
                reason=f"Target hit at {pos.target_price:.2f} (price={price:.2f})"
            )
        if pos.side == Side.SHORT and price <= pos.target_price:
            return TradeDecision(
                Action.EXIT, side=pos.side,
                reason=f"Target hit at {pos.target_price:.2f} (price={price:.2f})"
            )

        # Signal reversal exit: if composite flips strongly against us
        if pos.side == Side.LONG and signal.composite < -self.signal_cfg.entry_threshold:
            return TradeDecision(
                Action.EXIT, side=pos.side,
                reason=f"Signal reversal: composite={signal.composite:.0f}"
            )
        if pos.side == Side.SHORT and signal.composite > self.signal_cfg.entry_threshold:
            return TradeDecision(
                Action.EXIT, side=pos.side,
                reason=f"Signal reversal: composite={signal.composite:.0f}"
            )

        # Regime shift: if regime changes to VOLATILE, exit
        if signal.regime == Regime.VOLATILE:
            return TradeDecision(
                Action.EXIT, side=pos.side,
                reason="Regime shifted to VOLATILE"
            )

        return TradeDecision(Action.HOLD, reason="Position managed, holding")

    def _get_stop_target(self, regime: Regime) -> tuple[float, float]:
        """Return (stop_atr_mult, target_atr_mult) for the regime."""
        if regime == Regime.TRENDING:
            return self.risk_cfg.trend_stop_atr, self.risk_cfg.trend_target_atr
        else:
            return self.risk_cfg.mean_revert_stop_atr, self.risk_cfg.mean_revert_target_atr

    def _entry_reason(self, signal: SignalSnapshot, side: Side) -> str:
        parts = [
            f"{side.value.upper()} entry",
            f"regime={signal.regime.value}",
            f"composite={signal.composite:.0f}",
            f"vwap_z={signal.vwap_zscore:.1f}",
            f"mom={signal.momentum_signal:.0f}",
        ]
        if signal.is_event_day:
            parts.append("EVENT_DAY(dampened)")
        return " | ".join(parts)

    def open_position(self, side: Side, entry_price: float, stop_price: float,
                      target_price: float, quantity: int, regime: Regime,
                      order_id: Optional[int] = None):
        """Record that we've entered a position."""
        now = datetime.now(ET)
        self.position = Position(
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            entry_time=now,
            regime_at_entry=regime,
            highest_since_entry=entry_price,
            lowest_since_entry=entry_price,
            order_id=order_id,
        )
        log.info(
            f"POSITION OPENED: {side.value} {quantity}x ES @ {entry_price:.2f} "
            f"stop={stop_price:.2f} target={target_price:.2f} regime={regime.value}"
        )

    def close_position(self, exit_price: float, reason: str) -> dict:
        """Record position close and return trade summary."""
        pos = self.position
        if pos is None:
            return {}

        if pos.side == Side.LONG:
            pnl_points = exit_price - pos.entry_price
        else:
            pnl_points = pos.entry_price - exit_price

        pnl_dollars = pnl_points * ES_POINT_VALUE * pos.quantity
        duration = (datetime.now(ET) - pos.entry_time).total_seconds()

        trade = {
            "side": pos.side.value,
            "entry": pos.entry_price,
            "exit": exit_price,
            "stop": pos.stop_price,
            "target": pos.target_price,
            "quantity": pos.quantity,
            "pnl_points": round(pnl_points, 2),
            "pnl_dollars": round(pnl_dollars, 2),
            "regime": pos.regime_at_entry.value,
            "duration_sec": round(duration),
            "reason": reason,
            "time": datetime.now(ET).isoformat(),
        }
        self.trades_today.append(trade)
        self.position = None

        is_loss = pnl_dollars < 0
        if is_loss:
            self._last_loss_time = datetime.now(ET)

        log.info(
            f"POSITION CLOSED: {trade['side']} {trade['quantity']}x ES "
            f"entry={trade['entry']:.2f} exit={trade['exit']:.2f} "
            f"PnL={trade['pnl_points']:+.2f}pts (${trade['pnl_dollars']:+.2f}) "
            f"reason={reason}"
        )
        return trade

    @property
    def daily_pnl(self) -> float:
        return sum(t["pnl_dollars"] for t in self.trades_today)

    @property
    def trade_count_today(self) -> int:
        return len(self.trades_today)

    @property
    def in_cooldown(self) -> bool:
        """Are we in a post-loss cooldown?"""
        if self._last_loss_time is None:
            return False
        elapsed = (datetime.now(ET) - self._last_loss_time).total_seconds()
        return elapsed < self.risk_cfg.loss_cooldown_sec

    def reset_daily(self):
        """Call at start of each new trading day."""
        self.trades_today.clear()
        self._last_loss_time = None
        log.info("Strategy daily state reset")
