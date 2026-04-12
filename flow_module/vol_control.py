"""
Volatility Control / Risk Parity Flow Estimator.

The logic:
- Vol-targeting funds (hundreds of billions in AUM) maintain a target
  portfolio volatility, typically 10-12%.
- When realized vol RISES → they must reduce equity exposure → SELL ES.
- When realized vol FALLS → they can increase equity exposure → BUY ES.
- This creates the "vol compression rally" pattern: after a selloff,
  once vol starts declining, massive mechanical buying kicks in.
- The opposite: when a calm market suddenly gets volatile, these funds
  dump equities, accelerating the selloff.
- The signal is the RATE OF CHANGE of realized vol, not the level.

Signal output:
  +100 = realized vol dropping fast → vol-control funds buying heavily
  -100 = realized vol rising fast → vol-control funds selling heavily
     0 = vol stable → no significant flow
"""

import math
import logging
from dataclasses import dataclass
from datetime import datetime

from flow_module.config import VolControlConfig, ET
from flow_module.data import MarketDataStore

log = logging.getLogger(__name__)


@dataclass
class VolControlSignal:
    """Output of the vol-control flow estimator."""
    signal: float                # -100 to +100
    short_rv: float              # 1-month realized vol (annualized %)
    long_rv: float               # 3-month realized vol (annualized %)
    rv_roc: float                # rate of change of short RV (% pts per week)
    vix: float
    vix_rv_spread: float         # VIX - realized vol (term structure signal)
    flow_direction: str          # "BUY", "SELL", or "NEUTRAL"
    phase: str                   # "compression", "expansion", "stable"
    estimated_magnitude: str
    description: str


class VolControlTracker:
    """
    Estimates vol-targeting fund flows from realized vol dynamics.
    """

    def __init__(self, cfg: VolControlConfig):
        self.cfg = cfg

    def evaluate(self, store: MarketDataStore) -> VolControlSignal:
        """Compute vol-control signal from daily bar history + VIX."""
        closes = store.closes
        vix = store.live_vix if store.live_vix > 0 else store.vix_close

        if len(closes) < self.cfg.long_rv_window + 10:
            return VolControlSignal(
                signal=0, short_rv=0, long_rv=0, rv_roc=0,
                vix=vix, vix_rv_spread=0,
                flow_direction="NEUTRAL", phase="stable",
                estimated_magnitude="none",
                description="Insufficient data for vol calculation.",
            )

        # Calculate realized vol at two lookbacks
        short_rv = self._realized_vol(closes, self.cfg.short_rv_window)
        long_rv = self._realized_vol(closes, self.cfg.long_rv_window)

        # Rate of change of short-term RV over the last week
        # Compare current 21-day RV to what it was 5 days ago
        rv_roc = 0.0
        if len(closes) > self.cfg.short_rv_window + self.cfg.rv_roc_window + 2:
            prev_closes = closes[:-(self.cfg.rv_roc_window)]
            prev_rv = self._realized_vol(prev_closes, self.cfg.short_rv_window)
            if prev_rv > 0:
                rv_roc = short_rv - prev_rv  # change in vol pts

        # VIX - RV spread (forward vol premium)
        vix_rv_spread = vix - short_rv if vix > 0 else 0.0

        # Determine phase and signal
        signal = 0.0
        phase = "stable"

        if rv_roc <= self.cfg.rv_drop_threshold:
            # Vol compressing → funds buying back equity exposure
            phase = "compression"
            # Scale: -2% RV drop → moderate, -5%+ → strong
            raw = abs(rv_roc) / 5.0 * 100
            signal = min(raw, 100.0)
            # Boost if VIX is also elevated (more room to compress)
            if vix > self.cfg.vix_elevated:
                signal = min(signal * 1.3, 100.0)

        elif rv_roc >= self.cfg.rv_rise_threshold:
            # Vol expanding → funds forced to sell equity
            phase = "expansion"
            raw = rv_roc / 5.0 * 100
            signal = -min(raw, 100.0)
            # Boost if coming from very low vol (larger % change in allocation)
            if long_rv < 12:
                signal = max(signal * 1.3, -100.0)

        # Additional signal: short RV vs long RV divergence
        if short_rv > 0 and long_rv > 0:
            rv_ratio = short_rv / long_rv
            if rv_ratio > 1.5 and phase != "expansion":
                # Short-term vol spiked above long-term → selling pressure
                signal -= 20
            elif rv_ratio < 0.7 and phase != "compression":
                # Short-term vol collapsed below long-term → buying pressure
                signal += 20

        signal = max(min(round(signal, 1), 100.0), -100.0)

        direction = "NEUTRAL"
        if signal > 10:
            direction = "BUY"
        elif signal < -10:
            direction = "SELL"

        magnitude = self._classify(abs(signal))
        desc = self._build_description(
            phase, short_rv, long_rv, rv_roc, vix, vix_rv_spread, signal, direction)

        return VolControlSignal(
            signal=signal,
            short_rv=round(short_rv, 1),
            long_rv=round(long_rv, 1),
            rv_roc=round(rv_roc, 2),
            vix=round(vix, 1),
            vix_rv_spread=round(vix_rv_spread, 1),
            flow_direction=direction,
            phase=phase,
            estimated_magnitude=magnitude,
            description=desc,
        )

    def _realized_vol(self, closes: list[float], window: int) -> float:
        """Annualized realized vol from daily closes."""
        if len(closes) < window + 1:
            return 0.0
        recent = closes[-(window + 1):]
        returns = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0 and recent[i] > 0:
                returns.append(math.log(recent[i] / recent[i - 1]))
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(var * 252) * 100

    def _classify(self, abs_sig: float) -> str:
        if abs_sig >= 60:
            return "large"
        elif abs_sig >= 25:
            return "moderate"
        elif abs_sig > 10:
            return "small"
        return "none"

    def _build_description(self, phase: str, short_rv: float, long_rv: float,
                           rv_roc: float, vix: float, spread: float,
                           signal: float, direction: str) -> str:
        if phase == "compression":
            return (
                f"Vol COMPRESSION in progress. 1M RV: {short_rv:.1f}% "
                f"(dropping {abs(rv_roc):.1f}%pts/wk). 3M RV: {long_rv:.1f}%. "
                f"VIX: {vix:.1f}. Vol-control funds increasing equity exposure → "
                f"mechanical BUY flow. Signal: {signal:+.0f}"
            )
        elif phase == "expansion":
            return (
                f"Vol EXPANSION in progress. 1M RV: {short_rv:.1f}% "
                f"(rising {rv_roc:.1f}%pts/wk). 3M RV: {long_rv:.1f}%. "
                f"VIX: {vix:.1f}. Vol-control funds reducing equity exposure → "
                f"mechanical SELL flow. Signal: {signal:+.0f}"
            )
        else:
            return (
                f"Vol stable. 1M RV: {short_rv:.1f}%, 3M RV: {long_rv:.1f}%. "
                f"VIX: {vix:.1f} (VIX-RV spread: {spread:+.1f}). "
                f"No significant vol-control flow expected."
            )
