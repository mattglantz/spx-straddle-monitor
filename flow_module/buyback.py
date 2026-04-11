"""
Corporate Buyback Window Tracker.

The logic:
- S&P 500 companies buy back ~$5B/day of their own stock on average.
- This is the single largest consistent source of equity demand.
- BUT: companies enter "blackout periods" ~2 weeks before their
  quarterly earnings, and can't buy back shares.
- During aggregate blackout (Jan, Apr, Jul, Oct when most of the
  S&P 500 reports), buyback demand evaporates.
- When blackout lifts, pent-up demand floods back → strong bid.

Signal output:
  +100 = blackout just ended → buyback demand surging
  -100 = deep in blackout → major demand absent
     0 = normal buyback activity
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from flow_module.config import BuybackConfig, ET

log = logging.getLogger(__name__)


# Approximate S&P 500 earnings season windows
# Peak reporting: roughly 3rd week of Jan, Apr, Jul, Oct through mid next month
EARNINGS_SEASONS = [
    # (blackout_start_month, blackout_start_day, blackout_end_month, blackout_end_day)
    (1, 10, 2, 10),   # Q4 earnings: blackout mid-Jan to mid-Feb
    (4, 10, 5, 10),   # Q1 earnings: blackout mid-Apr to mid-May
    (7, 10, 8, 10),   # Q2 earnings: blackout mid-Jul to mid-Aug
    (10, 10, 11, 10), # Q3 earnings: blackout mid-Oct to mid-Nov
]


@dataclass
class BuybackSignal:
    """Output of the buyback window tracker."""
    signal: float                  # -100 to +100
    is_blackout: bool
    blackout_phase: str            # "pre_blackout", "blackout", "post_blackout", "open_window"
    days_in_phase: int
    next_phase_change: str         # description of what's next
    estimated_daily_flow_b: float  # estimated daily buyback in billions
    description: str


class BuybackTracker:
    """
    Tracks the corporate buyback window cycle and estimates
    demand impact on ES futures.
    """

    def __init__(self, cfg: BuybackConfig):
        self.cfg = cfg

    def evaluate(self) -> BuybackSignal:
        today = datetime.now(ET).date()

        blackout_start, blackout_end = self._current_or_next_blackout(today)

        if blackout_start is None or blackout_end is None:
            return BuybackSignal(
                signal=0, is_blackout=False, blackout_phase="open_window",
                days_in_phase=0, next_phase_change="Unknown",
                estimated_daily_flow_b=self.cfg.estimated_daily_buyback_b,
                description="Normal buyback window. Full demand active.",
            )

        # Determine which phase we're in
        if today < blackout_start:
            # Pre-blackout: approaching
            days_until = (blackout_start - today).days
            if days_until <= 5:
                phase = "pre_blackout"
                signal = -30.0  # mild negative — demand about to disappear
                daily_flow = self.cfg.estimated_daily_buyback_b * 1.2  # front-loading
                desc = (
                    f"Approaching blackout ({days_until} days). "
                    f"Companies may front-load buybacks. "
                    f"Demand drops off {blackout_start}."
                )
                next_change = f"Blackout begins {blackout_start}"
            else:
                phase = "open_window"
                signal = 15.0
                daily_flow = self.cfg.estimated_daily_buyback_b
                desc = (
                    f"Open buyback window. ~${daily_flow:.0f}B/day demand. "
                    f"Next blackout: {blackout_start} ({days_until} days)."
                )
                next_change = f"Blackout in {days_until} days"

        elif today <= blackout_end:
            # In blackout
            phase = "blackout"
            days_in = (today - blackout_start).days
            days_left = (blackout_end - today).days
            # Signal scales with depth into blackout (worst in middle)
            depth = days_in / max((blackout_end - blackout_start).days, 1)
            signal = -60.0 - 40.0 * min(depth, 1.0)  # -60 to -100
            daily_flow = self.cfg.estimated_daily_buyback_b * 0.15  # ~85% reduction
            desc = (
                f"BLACKOUT ACTIVE (day {days_in}, {days_left} days remaining). "
                f"~85% of buyback demand absent. "
                f"Only ~${daily_flow:.1f}B/day. Window reopens ~{blackout_end}."
            )
            next_change = f"Blackout ends ~{blackout_end}"

        else:
            # Post-blackout: demand surging back
            days_since = (today - blackout_end).days
            if days_since <= 10:
                phase = "post_blackout"
                # Strong positive — pent-up demand
                freshness = max(0.3, 1.0 - days_since / 10)
                signal = 80.0 * freshness
                daily_flow = self.cfg.estimated_daily_buyback_b * 1.5  # surge
                desc = (
                    f"Post-blackout SURGE ({days_since} days since blackout ended). "
                    f"Pent-up buyback demand: ~${daily_flow:.0f}B/day. "
                    f"This is historically one of the strongest demand periods."
                )
                next_change = "Surge normalizes in ~10 days"
            else:
                phase = "open_window"
                signal = 15.0
                daily_flow = self.cfg.estimated_daily_buyback_b
                desc = (
                    f"Normal buyback window. ~${daily_flow:.0f}B/day demand active."
                )
                # Find next blackout
                next_start, _ = self._next_blackout_after(today)
                if next_start:
                    days_until = (next_start - today).days
                    next_change = f"Next blackout: {next_start} ({days_until} days)"
                else:
                    next_change = "No upcoming blackout in calendar"

        return BuybackSignal(
            signal=round(signal, 1),
            is_blackout=phase == "blackout",
            blackout_phase=phase,
            days_in_phase=max((today - blackout_start).days, 0) if phase == "blackout" else 0,
            next_phase_change=next_change,
            estimated_daily_flow_b=round(daily_flow, 1),
            description=desc,
        )

    def _current_or_next_blackout(self, today: date):
        """Find the blackout window that today is in, or the next one."""
        year = today.year
        for yr in (year - 1, year, year + 1):
            for start_m, start_d, end_m, end_d in EARNINGS_SEASONS:
                end_yr = yr if end_m >= start_m else yr + 1
                try:
                    start = date(yr, start_m, start_d)
                    end = date(end_yr, end_m, end_d)
                except ValueError:
                    continue
                # If we're in the window or within 30 days pre/post
                if start - timedelta(days=30) <= today <= end + timedelta(days=15):
                    return start, end
        return None, None

    def _next_blackout_after(self, today: date):
        """Find the next blackout window starting after today."""
        year = today.year
        for yr in (year, year + 1):
            for start_m, start_d, end_m, end_d in EARNINGS_SEASONS:
                end_yr = yr if end_m >= start_m else yr + 1
                try:
                    start = date(yr, start_m, start_d)
                    end = date(end_yr, end_m, end_d)
                except ValueError:
                    continue
                if start > today:
                    return start, end
        return None, None
