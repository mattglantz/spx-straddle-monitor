"""
CTA (Commodity Trading Advisor) Trend-Following Trigger Tracker.

The logic:
- Managed futures / CTA funds (~$300B AUM) use systematic trend-following.
- Most use variations of moving-average crossover rules on daily data.
- When ES crosses ABOVE a key MA → CTAs receive a BUY signal and
  mechanically buy ES futures in size over 1-3 days.
- When ES crosses BELOW → they SELL.
- The key MAs: 50-day, 100-day, 200-day.
- The 200-day is the "big one" — crossing it triggers the largest flows.
- When price approaches an MA, we know a wave of forced buying or
  selling is imminent. The flow WILL come — it's rules-based.

Signal output:
  +100 = price just crossed above a major MA → CTA buy flows incoming
  -100 = price just crossed below a major MA → CTA sell flows incoming
     0 = price is far from all MAs → no CTA flow expected
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from flow_module.config import CTAConfig, ET
from flow_module.data import MarketDataStore

log = logging.getLogger(__name__)


@dataclass
class MALevel:
    """State of price relative to a single moving average."""
    period: int
    ma_value: float
    price: float
    distance_pct: float      # positive = price above MA
    is_near: bool             # within proximity threshold
    crossed_above: bool       # just crossed above (bullish trigger)
    crossed_below: bool       # just crossed below (bearish trigger)
    days_since_cross: int     # how many days since last cross


@dataclass
class CTASignal:
    """Output of the CTA trigger tracker."""
    signal: float             # -100 to +100
    ma_levels: list[MALevel]
    nearest_trigger: str      # e.g., "200 DMA at 5320.5 (1.2% below)"
    flow_direction: str       # "BUY", "SELL", or "NEUTRAL"
    estimated_magnitude: str  # "small", "moderate", "large"
    description: str


class CTATracker:
    """
    Tracks ES price relative to key moving averages and detects
    crossover events that trigger CTA flows.
    """

    def __init__(self, cfg: CTAConfig):
        self.cfg = cfg

    def evaluate(self, store: MarketDataStore) -> CTASignal:
        """Compute CTA signal from daily bar history."""
        price = store.live_price if store.live_price > 0 else store.last_close
        if price <= 0 or len(store.daily_bars) < max(self.cfg.ma_periods) + 5:
            return CTASignal(
                signal=0, ma_levels=[], nearest_trigger="Insufficient data",
                flow_direction="NEUTRAL", estimated_magnitude="none",
                description="Need more historical data for MA calculation.",
            )

        closes = store.closes
        ma_levels = []
        total_signal = 0.0

        for period in self.cfg.ma_periods:
            ma_val = sum(closes[-period:]) / period

            # Distance from MA
            dist_pct = ((price - ma_val) / ma_val) * 100

            # Check for recent crossover
            crossed_above = False
            crossed_below = False
            days_since = 999

            # Look back 5 days for recent crossover
            if len(closes) > period + 5:
                recent_closes = closes[-(period + 5):]
                for i in range(5, 0, -1):
                    window = recent_closes[:len(recent_closes) - 5 + i]
                    prev_ma = sum(window[-period:]) / period
                    prev_close = window[-1]
                    curr_window = recent_closes[:len(recent_closes) - 5 + i + 1]
                    curr_ma = sum(curr_window[-period:]) / period
                    curr_close = curr_window[-1]

                    if prev_close < prev_ma and curr_close >= curr_ma:
                        crossed_above = True
                        days_since = 5 - i
                        break
                    elif prev_close > prev_ma and curr_close <= curr_ma:
                        crossed_below = True
                        days_since = 5 - i
                        break

            is_near = abs(dist_pct) < self.cfg.proximity_pct

            level = MALevel(
                period=period,
                ma_value=round(ma_val, 2),
                price=round(price, 2),
                distance_pct=round(dist_pct, 2),
                is_near=is_near,
                crossed_above=crossed_above,
                crossed_below=crossed_below,
                days_since_cross=days_since,
            )
            ma_levels.append(level)

            # Score this MA — weight by importance (200 > 100 > 50)
            weight = period / 200.0  # 200-day gets 1.0, 50-day gets 0.25
            if crossed_above and days_since <= 3:
                # Fresh cross above — strong buy signal, decays over days
                freshness = max(0.3, 1.0 - days_since * 0.25)
                total_signal += 100 * weight * freshness
            elif crossed_below and days_since <= 3:
                freshness = max(0.3, 1.0 - days_since * 0.25)
                total_signal -= 100 * weight * freshness
            elif is_near:
                # Approaching but not crossed — anticipation signal
                if dist_pct > 0:
                    total_signal += 20 * weight  # above, support
                else:
                    total_signal -= 20 * weight  # below, resistance

        total_signal = max(min(total_signal, 100.0), -100.0)

        # Find nearest trigger for display
        nearest = min(ma_levels, key=lambda m: abs(m.distance_pct))
        above_below = "above" if nearest.distance_pct > 0 else "below"
        nearest_str = (
            f"{nearest.period} DMA at {nearest.ma_value:.1f} "
            f"({abs(nearest.distance_pct):.1f}% {above_below})"
        )

        direction = "NEUTRAL"
        if total_signal > 10:
            direction = "BUY"
        elif total_signal < -10:
            direction = "SELL"

        magnitude = self._classify(abs(total_signal))
        desc = self._build_description(ma_levels, total_signal, direction)

        return CTASignal(
            signal=round(total_signal, 1),
            ma_levels=ma_levels,
            nearest_trigger=nearest_str,
            flow_direction=direction,
            estimated_magnitude=magnitude,
            description=desc,
        )

    def _classify(self, abs_sig: float) -> str:
        if abs_sig >= 60:
            return "large"
        elif abs_sig >= 25:
            return "moderate"
        elif abs_sig > 10:
            return "small"
        return "none"

    def _build_description(self, levels: list[MALevel], signal: float,
                           direction: str) -> str:
        parts = []

        # Report crossovers first
        for lvl in levels:
            if lvl.crossed_above and lvl.days_since_cross <= 3:
                parts.append(
                    f"ES crossed ABOVE {lvl.period} DMA ({lvl.ma_value:.0f}) "
                    f"{lvl.days_since_cross} day(s) ago → CTA BUY trigger")
            elif lvl.crossed_below and lvl.days_since_cross <= 3:
                parts.append(
                    f"ES crossed BELOW {lvl.period} DMA ({lvl.ma_value:.0f}) "
                    f"{lvl.days_since_cross} day(s) ago → CTA SELL trigger")

        # Report proximity
        for lvl in levels:
            if lvl.is_near and not lvl.crossed_above and not lvl.crossed_below:
                above_below = "above" if lvl.distance_pct > 0 else "below"
                parts.append(
                    f"ES is {abs(lvl.distance_pct):.1f}% {above_below} "
                    f"{lvl.period} DMA ({lvl.ma_value:.0f}) — approaching trigger")

        if not parts:
            parts.append("ES is away from all key moving averages. No CTA trigger imminent.")

        return " | ".join(parts)
