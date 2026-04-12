"""
Calendar Seasonality Tracker.

Pure math — no data feeds needed. These patterns have persisted
for decades because they're driven by structural flows:

1. TURN OF MONTH (TOM): Last trading day + first 3 trading days.
   401(k) contributions, pension inflows, and payroll-driven
   investments hit the market on a monthly cycle. Historically
   the strongest 4-day window of any month.

2. DAY OF WEEK: Monday weakness (weekend risk digestion),
   Wednesday/Thursday strength (mid-week positioning).

3. PRE-FOMC DRIFT: ES tends to drift up 20-30pts in the 24 hours
   before an FOMC announcement. Decades of data support this.
   Driven by short covering and positioning ahead of uncertainty.

4. PRE-HOLIDAY EFFECT: The 1-2 trading days before 3-day weekends
   and major holidays tend to be bullish (low volume, short
   covering, positive sentiment).

5. MONTHLY SEASONALITY: Some months are structurally stronger
   (Nov-Jan) or weaker (Sep-Oct) due to fiscal year cycles,
   tax-loss selling, and institutional allocation patterns.

Signal output:
  +100 = strong bullish seasonal tailwind (e.g., TOM + pre-holiday)
  -100 = strong bearish seasonal headwind (rarely this extreme)
     0 = no notable seasonal effect
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time
from calendar import monthrange

from flow_module.config import ET, ECONOMIC_EVENTS

log = logging.getLogger(__name__)

# ── US Market Holidays (dates markets are CLOSED) ───────────
# We care about the trading days BEFORE these
US_HOLIDAYS = [
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
]

# FOMC dates (announcement day — drift is the 24h BEFORE this)
FOMC_DATES = [ds for ds, name in ECONOMIC_EVENTS if "FOMC" in name]

# Monthly bias: historical average excess return by month
# Positive = bullish, Negative = bearish (based on S&P 500 since 1950)
MONTHLY_BIAS = {
    1: +0.3,    # January: mild positive (January effect, new allocations)
    2: -0.1,    # February: flat to slight negative
    3: +0.1,    # March: mild positive (quarter-end positioning)
    4: +0.4,    # April: strong (tax refund flows, post-tax-selling bounce)
    5: +0.1,    # May: "sell in May" but actually slightly positive
    6: +0.1,    # June: mild
    7: +0.3,    # July: solid (mid-year rebalancing)
    8: -0.1,    # August: weak (low volume, vacation)
    9: -0.5,    # September: worst month (fiscal year-end selling, mutual fund distributions)
    10: +0.2,   # October: historically volatile but net positive (bottoming month)
    11: +0.5,   # November: strong (Thanksgiving rally, year-end positioning begins)
    12: +0.5,   # December: strong (Santa Claus rally, window dressing, tax-loss harvesting done)
}

# Day-of-week bias (0=Monday, 4=Friday)
DOW_BIAS = {
    0: -0.3,    # Monday: weakest (weekend gap risk, negative news digestion)
    1: +0.1,    # Tuesday: slight recovery
    2: +0.2,    # Wednesday: positive (mid-week, often FOMC day)
    3: +0.2,    # Thursday: positive (positioning)
    4: +0.1,    # Friday: mixed (OpEx effects, weekend de-risking)
}


def _last_business_day(year: int, month: int) -> date:
    last_day = date(year, month, monthrange(year, month)[1])
    while last_day.weekday() >= 5:
        last_day -= timedelta(days=1)
    return last_day


def _first_business_day(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _next_business_day(d: date) -> date:
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _prev_business_day(d: date) -> date:
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


@dataclass
class SeasonalEffect:
    """A single seasonal factor."""
    name: str
    bias: float        # raw bias score
    active: bool
    description: str


@dataclass
class SeasonalitySignal:
    """Output of the seasonality tracker."""
    signal: float              # -100 to +100
    effects: list[SeasonalEffect]
    active_effects: list[str]  # names of currently active effects
    bias_direction: str        # "BULLISH", "BEARISH", "NEUTRAL"
    description: str


class SeasonalityTracker:
    """
    Evaluates calendar-based seasonal effects for the current day.
    """

    def __init__(self):
        self._holiday_dates = set(
            datetime.strptime(d, "%Y-%m-%d").date() for d in US_HOLIDAYS)
        self._fomc_dates = set(
            datetime.strptime(d, "%Y-%m-%d").date() for d in FOMC_DATES)

    def evaluate(self) -> SeasonalitySignal:
        today = datetime.now(ET).date()
        effects = []

        # 1. Turn of Month
        effects.append(self._check_tom(today))

        # 2. Day of Week
        effects.append(self._check_dow(today))

        # 3. Pre-FOMC Drift
        effects.append(self._check_fomc(today))

        # 4. Pre-Holiday
        effects.append(self._check_holiday(today))

        # 5. Monthly Seasonality
        effects.append(self._check_monthly(today))

        # Combine active effects
        active = [e for e in effects if e.active]
        active_names = [e.name for e in active]

        # Weighted sum (TOM and FOMC get extra weight)
        weights = {
            "Turn of Month": 2.5,
            "Day of Week": 1.0,
            "Pre-FOMC Drift": 3.0,
            "Pre-Holiday": 1.5,
            "Monthly Bias": 1.0,
        }

        total_weighted_bias = 0.0
        total_weight = 0.0
        for e in effects:
            w = weights.get(e.name, 1.0)
            if e.active:
                total_weighted_bias += e.bias * w
                total_weight += w

        # Always include monthly and DOW as baseline
        for e in effects:
            if e.name in ("Monthly Bias", "Day of Week"):
                if not e.active:
                    w = weights.get(e.name, 1.0)
                    total_weighted_bias += e.bias * w
                    total_weight += w

        if total_weight > 0:
            raw_signal = (total_weighted_bias / total_weight) * 100
        else:
            raw_signal = 0.0

        signal = max(min(round(raw_signal, 1), 100.0), -100.0)

        if signal > 10:
            direction = "BULLISH"
        elif signal < -10:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        desc = self._build_description(active, signal, direction, today)

        return SeasonalitySignal(
            signal=signal,
            effects=effects,
            active_effects=active_names,
            bias_direction=direction,
            description=desc,
        )

    def _check_tom(self, today: date) -> SeasonalEffect:
        """Turn of Month: last trading day + first 3 trading days."""
        last_bd = _last_business_day(today.year, today.month)

        # Check if today is the last business day of the month
        if today == last_bd:
            return SeasonalEffect(
                name="Turn of Month", bias=0.5, active=True,
                description="Last trading day of month. 401(k)/pension inflows arriving."
            )

        # Check if today is in the first 3 business days of the month
        first_bd = _first_business_day(today.year, today.month)
        day2 = _next_business_day(first_bd)
        day3 = _next_business_day(day2)

        if today in (first_bd, day2, day3):
            day_num = 1 if today == first_bd else (2 if today == day2 else 3)
            strength = 0.5 - (day_num - 1) * 0.1  # strongest on day 1, fades
            return SeasonalEffect(
                name="Turn of Month", bias=strength, active=True,
                description=f"TOM day {day_num}: monthly inflows still hitting the market."
            )

        return SeasonalEffect(
            name="Turn of Month", bias=0.0, active=False,
            description="Not in TOM window."
        )

    def _check_dow(self, today: date) -> SeasonalEffect:
        """Day of week effect."""
        dow = today.weekday()
        bias = DOW_BIAS.get(dow, 0.0)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday"]
        name = day_names[dow]

        if dow >= 5:
            return SeasonalEffect(
                name="Day of Week", bias=0.0, active=False,
                description=f"{name}: market closed."
            )

        direction = "positive" if bias > 0 else "negative" if bias < 0 else "neutral"
        return SeasonalEffect(
            name="Day of Week", bias=bias, active=abs(bias) >= 0.2,
            description=f"{name}: historically {direction} day ({bias:+.1f} bias)."
        )

    def _check_fomc(self, today: date) -> SeasonalEffect:
        """Pre-FOMC drift: day before and day of FOMC."""
        tomorrow = _next_business_day(today)
        day_after = _next_business_day(tomorrow)

        # Day before FOMC — the strongest drift day
        if tomorrow in self._fomc_dates:
            return SeasonalEffect(
                name="Pre-FOMC Drift", bias=0.6, active=True,
                description=(
                    f"FOMC announcement TOMORROW ({tomorrow}). "
                    "Pre-FOMC drift in effect: ES historically drifts up "
                    "20-30pts in the 24h before announcement. "
                    "Driven by short covering and vol compression."
                )
            )

        # FOMC day itself — drift continues into announcement
        if today in self._fomc_dates:
            return SeasonalEffect(
                name="Pre-FOMC Drift", bias=0.3, active=True,
                description=(
                    "FOMC announcement TODAY. Drift effect continues "
                    "into the announcement. Post-announcement: expect "
                    "increased volatility regardless of direction."
                )
            )

        # Two days before — mild anticipation
        if day_after in self._fomc_dates:
            return SeasonalEffect(
                name="Pre-FOMC Drift", bias=0.2, active=True,
                description=f"FOMC in 2 days ({day_after}). Early positioning may begin."
            )

        return SeasonalEffect(
            name="Pre-FOMC Drift", bias=0.0, active=False,
            description="No FOMC nearby."
        )

    def _check_holiday(self, today: date) -> SeasonalEffect:
        """Pre-holiday effect: day before a market holiday or 3-day weekend."""
        tomorrow = today + timedelta(days=1)

        # Check if tomorrow or the day after is a holiday / weekend creating 3+ day break
        days_off = 0
        check = tomorrow
        while check in self._holiday_dates or check.weekday() >= 5:
            days_off += 1
            check += timedelta(days=1)
            if days_off > 5:
                break

        if days_off >= 2:  # at least a 3-day weekend or holiday
            return SeasonalEffect(
                name="Pre-Holiday", bias=0.4, active=True,
                description=(
                    f"Pre-holiday effect: {days_off}-day break ahead. "
                    "Low volume, short covering, bullish bias typical."
                )
            )

        # Also check for regular 3-day weekends (Friday)
        if today.weekday() == 4:  # Friday
            # Normal weekend is 2 days, check for Monday holiday
            monday = today + timedelta(days=3)
            if monday in self._holiday_dates:
                return SeasonalEffect(
                    name="Pre-Holiday", bias=0.4, active=True,
                    description="3-day weekend (Monday holiday). Pre-holiday bullish bias."
                )

        return SeasonalEffect(
            name="Pre-Holiday", bias=0.0, active=False,
            description="No holiday nearby."
        )

    def _check_monthly(self, today: date) -> SeasonalEffect:
        """Monthly seasonality pattern."""
        month = today.month
        bias = MONTHLY_BIAS.get(month, 0.0)
        month_names = ["", "January", "February", "March", "April", "May",
                        "June", "July", "August", "September", "October",
                        "November", "December"]
        name = month_names[month]

        direction = "bullish" if bias > 0 else "bearish" if bias < 0 else "neutral"
        return SeasonalEffect(
            name="Monthly Bias", bias=bias, active=abs(bias) >= 0.3,
            description=(
                f"{name}: historically {direction} month ({bias:+.1f} avg excess return). "
                + self._monthly_context(month)
            )
        )

    def _monthly_context(self, month: int) -> str:
        """Extra context for why this month has its seasonal pattern."""
        reasons = {
            1: "New year allocations, January effect.",
            4: "Tax refund flows, post-tax-selling bounce.",
            7: "Mid-year rebalancing inflows.",
            9: "Fiscal year-end selling, mutual fund distributions.",
            10: "Historically volatile but tends to mark bottoms.",
            11: "Thanksgiving rally, year-end positioning begins.",
            12: "Santa Claus rally, tax-loss harvesting complete, window dressing.",
        }
        return reasons.get(month, "")

    def _build_description(self, active: list[SeasonalEffect], signal: float,
                           direction: str, today: date) -> str:
        if not active:
            return f"No strong seasonal effects today ({today})."

        parts = [e.description for e in active]
        return f"Seasonal bias: {direction} ({signal:+.0f}). " + " | ".join(parts)
