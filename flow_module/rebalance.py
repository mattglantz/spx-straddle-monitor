"""
Pension & Fund End-of-Month/Quarter Rebalancing Flow Tracker.

The logic:
- Large allocators (pensions, endowments, target-date funds) maintain
  fixed equity/bond ratios, typically rebalanced monthly or quarterly.
- If equities rally during the month → they become overweight →
  funds SELL equities (and ES) in the last 2-3 trading days.
- If equities drop → they're underweight → funds BUY equities.
- Quarter-end rebalancing is 3-4x the magnitude of month-end.
- The flow direction is predictable: it opposes the MTD return.

Signal output:
  +100 = strong buy pressure expected (equities fell hard, funds must buy)
  -100 = strong sell pressure expected (equities rallied hard, funds must sell)
     0 = no significant rebalancing expected
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from calendar import monthrange

from flow_module.config import RebalanceConfig, ET

log = logging.getLogger(__name__)


def _last_business_day(year: int, month: int) -> date:
    """Get the last business day of a given month."""
    last_day = date(year, month, monthrange(year, month)[1])
    while last_day.weekday() >= 5:  # Saturday=5, Sunday=6
        last_day -= timedelta(days=1)
    return last_day


def _first_business_day(year: int, month: int) -> date:
    """Get the first business day of a given month."""
    d = date(year, month, 1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _is_quarter_end(month: int) -> bool:
    return month in (3, 6, 9, 12)


def _trading_days_until_month_end(today: date) -> int:
    """Approximate trading days until last business day of the month."""
    last_bd = _last_business_day(today.year, today.month)
    if today > last_bd:
        return 0
    count = 0
    d = today
    while d <= last_bd:
        if d.weekday() < 5:
            count += 1
        d += timedelta(days=1)
    return max(count - 1, 0)  # don't count today


@dataclass
class RebalanceSignal:
    """Output of the rebalance flow tracker."""
    signal: float                # -100 to +100
    mtd_return_pct: float        # month-to-date ES return
    qtd_return_pct: float        # quarter-to-date ES return
    days_to_month_end: int
    is_rebalance_window: bool    # within last N trading days
    is_quarter_end: bool
    flow_direction: str          # "BUY", "SELL", or "NEUTRAL"
    estimated_magnitude: str     # "small", "moderate", "large"
    description: str


class RebalanceTracker:
    """
    Tracks month-to-date and quarter-to-date equity returns to
    predict end-of-period rebalancing flows.
    """

    def __init__(self, cfg: RebalanceConfig):
        self.cfg = cfg

    def evaluate(self, daily_closes: list[float], dates: list[date],
                 live_price: float) -> RebalanceSignal:
        """
        Compute rebalance signal from daily price history.

        Args:
            daily_closes: List of daily closing prices (chronological)
            dates: Corresponding dates
            live_price: Current live ES price
        """
        today = datetime.now(ET).date()

        # MTD return: from last business day of prior month to now
        mtd_return = self._calc_mtd_return(daily_closes, dates, live_price, today)
        qtd_return = self._calc_qtd_return(daily_closes, dates, live_price, today)

        days_to_end = _trading_days_until_month_end(today)
        is_window = days_to_end <= self.cfg.window_days
        is_quarter = _is_quarter_end(today.month)

        # Signal strength: based on MTD return magnitude and proximity to month-end
        if not is_window or abs(mtd_return) < self.cfg.min_mtd_return_pct:
            signal = 0.0
            direction = "NEUTRAL"
            magnitude = "none"
            desc = self._neutral_description(mtd_return, days_to_end)
        else:
            # Return opposes flow: if market rallied → sell; if dropped → buy
            base_return = mtd_return
            if is_quarter:
                # Quarter-end: use the larger of MTD and QTD
                if abs(qtd_return) > abs(mtd_return):
                    base_return = qtd_return

            # Scale: 1.5% return → moderate, 3%+ → large
            raw_signal = -base_return / 3.0 * 100  # negate: rally → sell
            raw_signal = max(min(raw_signal, 100.0), -100.0)

            # Amplify for quarter-end
            if is_quarter:
                raw_signal *= min(self.cfg.quarter_multiplier, 3.0)
                raw_signal = max(min(raw_signal, 100.0), -100.0)

            # Proximity boost: stronger signal on the actual last day
            if days_to_end == 0:
                raw_signal *= 1.3
            elif days_to_end == 1:
                raw_signal *= 1.1
            raw_signal = max(min(raw_signal, 100.0), -100.0)

            signal = round(raw_signal, 1)
            direction = "BUY" if signal > 0 else "SELL"
            magnitude = self._classify_magnitude(abs(signal))
            desc = self._window_description(
                signal, mtd_return, qtd_return, days_to_end, is_quarter, direction)

        return RebalanceSignal(
            signal=signal,
            mtd_return_pct=round(mtd_return, 2),
            qtd_return_pct=round(qtd_return, 2),
            days_to_month_end=days_to_end,
            is_rebalance_window=is_window,
            is_quarter_end=is_quarter,
            flow_direction=direction,
            estimated_magnitude=magnitude,
            description=desc,
        )

    def _calc_mtd_return(self, closes: list[float], dates: list[date],
                         live: float, today: date) -> float:
        """Month-to-date return in %."""
        first_of_month = _first_business_day(today.year, today.month)
        # Find close on or just before month start
        ref_price = None
        for i in range(len(dates) - 1, -1, -1):
            if dates[i] < first_of_month:
                ref_price = closes[i]
                break
        if ref_price is None or ref_price <= 0 or live <= 0:
            return 0.0
        return ((live - ref_price) / ref_price) * 100

    def _calc_qtd_return(self, closes: list[float], dates: list[date],
                         live: float, today: date) -> float:
        """Quarter-to-date return in %."""
        q_start_month = ((today.month - 1) // 3) * 3 + 1
        q_start = _first_business_day(today.year, q_start_month)
        ref_price = None
        for i in range(len(dates) - 1, -1, -1):
            if dates[i] < q_start:
                ref_price = closes[i]
                break
        if ref_price is None or ref_price <= 0 or live <= 0:
            return 0.0
        return ((live - ref_price) / ref_price) * 100

    def _classify_magnitude(self, abs_signal: float) -> str:
        if abs_signal >= 70:
            return "large"
        elif abs_signal >= 35:
            return "moderate"
        else:
            return "small"

    def _neutral_description(self, mtd_return: float, days_to_end: int) -> str:
        if days_to_end > 3:
            return (f"Not in rebalance window ({days_to_end} trading days to month-end). "
                    f"MTD return: {mtd_return:+.1f}%")
        return (f"In window but MTD return ({mtd_return:+.1f}%) below threshold "
                f"({self.cfg.min_mtd_return_pct}%). No significant flow expected.")

    def _window_description(self, signal: float, mtd: float, qtd: float,
                            days: int, is_quarter: bool, direction: str) -> str:
        period = "QUARTER-END" if is_quarter else "Month-end"
        return (
            f"{period} rebalance window active ({days} days left). "
            f"MTD {mtd:+.1f}% / QTD {qtd:+.1f}%. "
            f"Expect {direction} flow from pension/fund rebalancing. "
            f"Signal: {signal:+.0f}"
        )
