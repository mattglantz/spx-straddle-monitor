"""
Real-time data collection and bar aggregation for ES futures.

Responsibilities:
- Subscribe to ES continuous futures tick data via ib_async
- Build 1-min and 5-min OHLCV bars from real-time ticks
- Calculate running VWAP (resets each RTH session)
- Track session levels: prior close, overnight high/low, opening range
- Calculate ATR for position sizing and signal normalization
- Provide VIX snapshot
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from collections import deque
from typing import Optional

from trading_bot.config import ET, RTH_OPEN, RTH_CLOSE, GLOBEX_OPEN

log = logging.getLogger(__name__)


@dataclass
class Bar:
    """A single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    vwap: float = 0.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0


@dataclass
class SessionLevels:
    """Key price levels tracked across the session."""
    prior_close: float = 0.0
    overnight_high: float = 0.0
    overnight_low: float = float("inf")
    rth_open: float = 0.0
    opening_range_high: float = 0.0   # first 15 min high
    opening_range_low: float = float("inf")  # first 15 min low
    day_high: float = 0.0
    day_low: float = float("inf")

    def round_numbers_near(self, price: float, radius: float = 25.0) -> list[float]:
        """Return nearby round-number levels (multiples of 25 and 50)."""
        base = int(price / 25) * 25
        levels = []
        for offset in range(-4, 5):
            lvl = base + offset * 25
            if abs(lvl - price) <= radius * 2:
                levels.append(float(lvl))
        return levels


class BarAggregator:
    """Builds fixed-period bars from tick data."""

    def __init__(self, period_minutes: int, max_bars: int = 200):
        self.period_minutes = period_minutes
        self.bars: deque[Bar] = deque(maxlen=max_bars)
        self._current_bar: Optional[Bar] = None
        self._current_bar_start: Optional[datetime] = None

    def _bar_start(self, ts: datetime) -> datetime:
        """Round down to the start of the current bar period."""
        minute = (ts.minute // self.period_minutes) * self.period_minutes
        return ts.replace(minute=minute, second=0, microsecond=0)

    def update(self, price: float, volume: int, ts: datetime) -> Optional[Bar]:
        """
        Feed a tick. Returns a completed Bar when a period closes, else None.
        """
        bar_start = self._bar_start(ts)

        # New bar period started — close the old one
        if self._current_bar is not None and bar_start != self._current_bar_start:
            completed = self._current_bar
            self.bars.append(completed)
            self._current_bar = None
            self._current_bar_start = None
            # Start the new bar with this tick
            self._open_bar(price, volume, ts, bar_start)
            return completed

        if self._current_bar is None:
            self._open_bar(price, volume, ts, bar_start)
            return None

        # Update existing bar
        self._current_bar.high = max(self._current_bar.high, price)
        self._current_bar.low = min(self._current_bar.low, price)
        self._current_bar.close = price
        self._current_bar.volume += volume
        return None

    def _open_bar(self, price: float, volume: int, ts: datetime, bar_start: datetime):
        self._current_bar = Bar(
            timestamp=bar_start,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
        )
        self._current_bar_start = bar_start

    @property
    def last(self) -> Optional[Bar]:
        return self.bars[-1] if self.bars else None

    @property
    def current(self) -> Optional[Bar]:
        return self._current_bar


class VWAPTracker:
    """Cumulative VWAP that resets at RTH open."""

    def __init__(self):
        self.cum_pv: float = 0.0   # cumulative (price * volume)
        self.cum_vol: float = 0.0
        self.value: float = 0.0
        self._last_reset_date: Optional[datetime] = None

    def update(self, price: float, volume: int, ts: datetime):
        now_et = ts.astimezone(ET) if ts.tzinfo else ts
        today = now_et.date()

        # Reset at RTH open each day
        if self._last_reset_date != today and now_et.time() >= RTH_OPEN:
            self.cum_pv = 0.0
            self.cum_vol = 0.0
            self._last_reset_date = today

        vol = max(volume, 1)  # ensure we always update
        self.cum_pv += price * vol
        self.cum_vol += vol
        self.value = self.cum_pv / self.cum_vol if self.cum_vol > 0 else price


class ATRCalculator:
    """True Range / Average True Range from completed bars."""

    @staticmethod
    def calculate(bars: deque[Bar], period: int = 14) -> float:
        if len(bars) < 2:
            return 0.0
        true_ranges = []
        bar_list = list(bars)
        for i in range(1, len(bar_list)):
            prev_close = bar_list[i - 1].close
            curr = bar_list[i]
            tr = max(
                curr.high - curr.low,
                abs(curr.high - prev_close),
                abs(curr.low - prev_close),
            )
            true_ranges.append(tr)
        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
        # Wilder smoothed ATR
        atr = sum(true_ranges[:period]) / period
        for tr in true_ranges[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr


class MarketData:
    """
    Aggregates all real-time data for the bot.

    Owns bar builders, VWAP, session levels, and provides
    a clean interface for signals and strategy to consume.
    """

    def __init__(self, fast_period: int = 1, slow_period: int = 5,
                 max_bars: int = 200, atr_period: int = 14):
        self.fast_bars = BarAggregator(fast_period, max_bars)
        self.slow_bars = BarAggregator(slow_period, max_bars)
        self.vwap = VWAPTracker()
        self.levels = SessionLevels()
        self.atr_period = atr_period

        # Latest prices
        self.last_price: float = 0.0
        self.bid: float = 0.0
        self.ask: float = 0.0
        self.vix: float = 0.0

        # Options-derived data (fed from monitor or separate subscription)
        self.atm_iv: float = 0.0
        self.put_call_skew: float = 0.0
        self.straddle_price: float = 0.0

        self._session_date: Optional[datetime] = None
        self._opening_range_end: Optional[datetime] = None

    def on_tick(self, price: float, volume: int, ts: datetime):
        """Process a new tick from ES futures."""
        if price <= 0:
            return

        self.last_price = price
        now_et = ts.astimezone(ET) if ts.tzinfo else ts
        today = now_et.date()

        # Session rollover (Globex boundary at 18:00 ET)
        if self._session_date != today and now_et.time() >= GLOBEX_OPEN:
            self._new_session(price, today, now_et)

        # Update session levels
        self._update_levels(price, now_et)

        # Feed bar aggregators
        self.fast_bars.update(price, volume, ts)
        self.slow_bars.update(price, volume, ts)

        # Feed VWAP
        self.vwap.update(price, volume, ts)

    def _new_session(self, price: float, today, now_et: datetime):
        """Reset for a new trading session."""
        # Prior close = last close price before new session
        if self.last_price > 0:
            self.levels.prior_close = self.last_price
        self.levels.overnight_high = price
        self.levels.overnight_low = price
        self.levels.rth_open = 0.0
        self.levels.opening_range_high = 0.0
        self.levels.opening_range_low = float("inf")
        self.levels.day_high = price
        self.levels.day_low = price
        self._session_date = today
        self._opening_range_end = None
        log.info(f"New session: {today}, prior close={self.levels.prior_close:.2f}")

    def _update_levels(self, price: float, now_et: datetime):
        """Track session highs, lows, opening range."""
        t = now_et.time()

        # Pre-market / overnight
        if t < RTH_OPEN or t >= RTH_CLOSE:
            self.levels.overnight_high = max(self.levels.overnight_high, price)
            self.levels.overnight_low = min(self.levels.overnight_low, price)

        # RTH
        if RTH_OPEN <= t < RTH_CLOSE:
            # Capture RTH open
            if self.levels.rth_open == 0.0:
                self.levels.rth_open = price
                self._opening_range_end = datetime.combine(
                    now_et.date(), time(9, 45), tzinfo=ET)

            # Opening range (first 15 min)
            if self._opening_range_end and now_et <= self._opening_range_end:
                self.levels.opening_range_high = max(
                    self.levels.opening_range_high, price)
                self.levels.opening_range_low = min(
                    self.levels.opening_range_low, price)

            # Day high/low
            self.levels.day_high = max(self.levels.day_high, price)
            self.levels.day_low = min(self.levels.day_low, price)

    @property
    def atr(self) -> float:
        """Current ATR from 1-min bars."""
        return ATRCalculator.calculate(self.fast_bars.bars, self.atr_period)

    @property
    def slow_atr(self) -> float:
        """ATR from 5-min bars."""
        return ATRCalculator.calculate(self.slow_bars.bars, self.atr_period)

    @property
    def realized_vol(self) -> float:
        """
        Annualized realized vol from recent 5-min bar closes.
        Uses close-to-close log returns.
        """
        bars = list(self.slow_bars.bars)
        if len(bars) < 10:
            return 0.0
        returns = []
        for i in range(1, len(bars)):
            if bars[i - 1].close > 0 and bars[i].close > 0:
                returns.append(math.log(bars[i].close / bars[i - 1].close))
        if len(returns) < 5:
            return 0.0
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        # Annualize: 5-min bars → ~78 per day (6.5h RTH), ~252 trading days
        bars_per_day = 78
        return math.sqrt(var * bars_per_day * 252)

    def is_rth(self, ts: Optional[datetime] = None) -> bool:
        """Check if we're in Regular Trading Hours."""
        if ts is None:
            ts = datetime.now(ET)
        now_et = ts.astimezone(ET) if ts.tzinfo else ts
        if now_et.weekday() >= 5:
            return False
        return RTH_OPEN <= now_et.time() < RTH_CLOSE

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0
