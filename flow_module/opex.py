"""
Options Expiration (OpEx) Gamma Unwind Tracker.

The logic:
- Monthly OpEx (3rd Friday) and Quad Witching (Mar/Jun/Sep/Dec)
  represent massive options expiration events.
- Before OpEx: dealer gamma pins price near strikes with most
  open interest → reduced volatility, mean-reversion.
- At/after OpEx: gamma unwinds, hedges are removed, and the
  market is "released" → volatility spikes, directional moves.
- The Monday after OpEx is historically one of the most volatile
  days of the month.
- Quad Witching is 3-4x the notional of regular monthly OpEx.

Signal output:
  Doesn't predict direction — instead flags the volatility regime:
  "PIN" = pre-OpEx, expect mean-reversion / low vol
  "UNWIND" = at/post-OpEx, expect vol expansion / trending
  "NEUTRAL" = not near OpEx
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from flow_module.config import OpExConfig, MONTHLY_OPEX, QUAD_WITCHING, ET

log = logging.getLogger(__name__)


@dataclass
class OpExSignal:
    """Output of the OpEx tracker."""
    phase: str                  # "PIN", "UNWIND", or "NEUTRAL"
    days_to_opex: int           # trading days to next monthly OpEx
    next_opex_date: str
    is_quad_witching: bool
    vol_regime: str             # "compressed", "expanded", "normal"
    magnitude: float            # 0-100, how significant this OpEx is
    description: str


class OpExTracker:
    """
    Tracks proximity to options expiration events and predicts
    the volatility regime (pin vs unwind).
    """

    def __init__(self, cfg: OpExConfig):
        self.cfg = cfg
        self._opex_dates = sorted(
            datetime.strptime(d, "%Y-%m-%d").date() for d in MONTHLY_OPEX)
        self._quad_dates = set(
            datetime.strptime(d, "%Y-%m-%d").date() for d in QUAD_WITCHING)

    def evaluate(self) -> OpExSignal:
        today = datetime.now(ET).date()

        # Find next OpEx
        next_opex = None
        for d in self._opex_dates:
            if d >= today:
                next_opex = d
                break

        if next_opex is None:
            return OpExSignal(
                phase="NEUTRAL", days_to_opex=99,
                next_opex_date="unknown", is_quad_witching=False,
                vol_regime="normal", magnitude=0,
                description="No upcoming OpEx dates in calendar."
            )

        # Find most recent past OpEx (for post-OpEx detection)
        prev_opex = None
        for d in reversed(self._opex_dates):
            if d < today:
                prev_opex = d
                break

        days_to = self._trading_days_between(today, next_opex)
        is_quad = next_opex in self._quad_dates

        # Check if we're in post-OpEx unwind (Monday after)
        days_since_prev = (
            self._trading_days_between(prev_opex, today)
            if prev_opex else 99
        )
        prev_was_quad = prev_opex in self._quad_dates if prev_opex else False

        # Determine phase
        if days_since_prev <= self.cfg.post_opex_days:
            phase = "UNWIND"
            vol_regime = "expanded"
            base_magnitude = 70.0
            if prev_was_quad:
                base_magnitude = 95.0
            desc = self._unwind_description(prev_opex, prev_was_quad, days_since_prev)

        elif days_to <= self.cfg.pre_opex_days:
            phase = "PIN"
            vol_regime = "compressed"
            base_magnitude = 60.0
            if is_quad:
                base_magnitude = 85.0
            desc = self._pin_description(next_opex, is_quad, days_to)

        elif days_to == 0:
            # OpEx day itself
            phase = "UNWIND"
            vol_regime = "expanded"
            base_magnitude = 80.0 if not is_quad else 100.0
            desc = self._opex_day_description(next_opex, is_quad)

        else:
            phase = "NEUTRAL"
            vol_regime = "normal"
            base_magnitude = 0.0
            desc = f"Next OpEx: {next_opex} ({days_to} trading days away)"
            if is_quad:
                desc += " [QUAD WITCHING]"

        return OpExSignal(
            phase=phase,
            days_to_opex=days_to,
            next_opex_date=next_opex.strftime("%Y-%m-%d"),
            is_quad_witching=is_quad,
            vol_regime=vol_regime,
            magnitude=round(base_magnitude, 1),
            description=desc,
        )

    def _trading_days_between(self, start: date, end: date) -> int:
        """Count weekdays between two dates (exclusive of start)."""
        if start >= end:
            return 0
        count = 0
        d = start + timedelta(days=1)
        while d <= end:
            if d.weekday() < 5:
                count += 1
            d += timedelta(days=1)
        return count

    def _pin_description(self, opex: date, is_quad: bool, days: int) -> str:
        kind = "QUAD WITCHING" if is_quad else "Monthly OpEx"
        return (
            f"{kind} on {opex} ({days} trading day{'s' if days != 1 else ''} away). "
            f"Dealer gamma pinning likely. Expect compressed vol and mean-reversion. "
            f"Fade moves away from round strikes."
        )

    def _opex_day_description(self, opex: date, is_quad: bool) -> str:
        kind = "QUAD WITCHING" if is_quad else "Monthly OpEx"
        return (
            f"{kind} TODAY ({opex}). Massive gamma unwind in progress. "
            f"Expect increased volatility, especially into the close. "
            f"Directional moves after 2 PM ET are common."
        )

    def _unwind_description(self, prev_opex: date, was_quad: bool, days_since: int) -> str:
        kind = "Quad Witching" if was_quad else "Monthly OpEx"
        return (
            f"Post-{kind} unwind ({days_since} day{'s' if days_since != 1 else ''} since {prev_opex}). "
            f"Dealer hedges have been removed. "
            f"Expect trending behavior and vol expansion."
        )
