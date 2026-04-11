"""
Signal generation for the ES Futures Trading Bot.

This is where the "outside the box" thinking lives. Instead of
relying on lagging indicators (MA crossovers, RSI), we use:

1. OPTIONS-DERIVED regime detection — IV/RV ratio and put/call skew
   tell us whether dealers are long or short gamma, which predicts
   whether the market will mean-revert or trend.

2. VWAP z-score — institutional benchmark. Deviation from VWAP
   measured in ATR units gives us a normalized overbought/oversold.

3. Momentum confirmation — rate of change and higher-high/lower-low
   structure prevent us from fading strong trends.

4. Key level proximity — prior close, overnight range, opening range,
   round numbers act as magnets in calm markets and breakout points
   in volatile markets.

5. Composite scoring — regime determines the weighting. In mean-reversion
   mode we weight the fade signals. In trending mode we weight momentum.
"""

import math
import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from trading_bot.config import RegimeConfig, SignalConfig, ET, ECONOMIC_EVENTS
from trading_bot.data import MarketData

log = logging.getLogger(__name__)


class Regime(Enum):
    MEAN_REVERT = "mean_revert"
    TRENDING = "trending"
    VOLATILE = "volatile"    # stand aside or tiny size


@dataclass
class SignalSnapshot:
    """All computed signals at a point in time."""
    timestamp: datetime
    regime: Regime
    # Individual signals (-100 to +100, positive = bullish)
    vwap_signal: float = 0.0
    momentum_signal: float = 0.0
    level_signal: float = 0.0
    # Options-derived
    iv_rv_ratio: float = 0.0
    vix: float = 0.0
    skew: float = 0.0
    # Composite
    composite: float = 0.0
    # Context
    vwap_zscore: float = 0.0
    atr: float = 0.0
    is_event_day: bool = False


class RegimeDetector:
    """
    Determines market regime from options-derived data and realized vol.

    The key insight: when implied vol trades at a big premium to realized vol,
    option sellers (dealers) are collecting premium and hedging by buying dips
    and selling rips — this creates mean-reversion. When realized vol exceeds
    implied, the market is moving more than expected and tends to trend.
    """

    def __init__(self, cfg: RegimeConfig):
        self.cfg = cfg

    def detect(self, data: MarketData) -> Regime:
        vix = data.vix
        rv = data.realized_vol * 100  # convert to VIX-like percentage
        iv = data.atm_iv * 100 if data.atm_iv > 0 else vix
        skew = abs(data.put_call_skew)

        # Extreme volatility — stand aside
        if vix > self.cfg.vix_high:
            log.debug(f"Regime: VOLATILE (VIX={vix:.1f} > {self.cfg.vix_high})")
            return Regime.VOLATILE

        # Elevated skew with high VIX = negative gamma environment
        if skew > self.cfg.skew_elevated and vix > self.cfg.vix_low:
            log.debug(f"Regime: TRENDING (skew={skew:.1f}, VIX={vix:.1f})")
            return Regime.TRENDING

        # IV/RV ratio determines expected behavior
        iv_rv = iv / rv if rv > 0 else 2.0

        if iv_rv > self.cfg.iv_rv_mean_revert:
            log.debug(f"Regime: MEAN_REVERT (IV/RV={iv_rv:.2f})")
            return Regime.MEAN_REVERT
        elif iv_rv < self.cfg.iv_rv_trending:
            log.debug(f"Regime: TRENDING (IV/RV={iv_rv:.2f})")
            return Regime.TRENDING

        # Default: calm market = mean revert
        if vix < self.cfg.vix_low:
            return Regime.MEAN_REVERT

        return Regime.MEAN_REVERT


class VWAPSignal:
    """
    Generates a signal based on price deviation from VWAP.

    In mean-reversion mode: price far above VWAP = short signal (fade).
    In trending mode: price above VWAP = long signal (ride).
    """

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg

    def calculate(self, data: MarketData, regime: Regime) -> tuple[float, float]:
        """Returns (signal, z_score). Signal is -100 to +100."""
        vwap = data.vwap.value
        price = data.last_price
        atr = data.atr

        if vwap <= 0 or price <= 0 or atr <= 0:
            return 0.0, 0.0

        # Z-score: how many ATRs away from VWAP
        z = (price - vwap) / atr

        if regime == Regime.MEAN_REVERT:
            # Fade: the further from VWAP, the stronger the counter signal
            if abs(z) < self.cfg.vwap_fade_exit:
                return 0.0, z  # near VWAP, no signal
            # Negative z (below VWAP) → positive signal (buy)
            # Positive z (above VWAP) → negative signal (sell)
            strength = min(abs(z) / self.cfg.vwap_fade_entry, 1.0) * 100
            return -math.copysign(strength, z), z

        elif regime == Regime.TRENDING:
            # Trend: above VWAP = bullish, below = bearish
            strength = min(abs(z) / 3.0, 1.0) * 60  # cap at 60 for trending
            return math.copysign(strength, z), z

        return 0.0, z


class MomentumSignal:
    """
    Rate of change and structure analysis on slow bars.

    Measures whether the market is making directional progress
    (trending) or chopping (mean-reverting).
    """

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg

    def calculate(self, data: MarketData) -> float:
        """Returns signal from -100 to +100."""
        bars = list(data.slow_bars.bars)
        lookback = self.cfg.momentum_lookback

        if len(bars) < lookback:
            return 0.0

        recent = bars[-lookback:]

        # Rate of change
        start_price = recent[0].close
        end_price = recent[-1].close
        if start_price <= 0:
            return 0.0
        roc_pct = ((end_price - start_price) / start_price) * 100

        # Higher highs / lower lows count
        hh_count = 0
        ll_count = 0
        for i in range(1, len(recent)):
            if recent[i].high > recent[i - 1].high:
                hh_count += 1
            if recent[i].low < recent[i - 1].low:
                ll_count += 1

        # Net structure: positive = uptrend structure
        structure = (hh_count - ll_count) / (lookback - 1)  # -1 to 1

        # Combine ROC and structure
        roc_score = max(min(roc_pct / self.cfg.momentum_threshold, 1.0), -1.0) * 60
        structure_score = structure * 40

        return max(min(roc_score + structure_score, 100.0), -100.0)


class KeyLevelSignal:
    """
    Generates signals based on proximity to key price levels.

    In mean-reversion: levels act as magnets (fade away from them, expect return).
    In trending: levels act as breakout/breakdown triggers.
    """

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg

    def calculate(self, data: MarketData, regime: Regime) -> float:
        """Returns signal from -100 to +100."""
        price = data.last_price
        atr = data.atr
        levels = data.levels

        if price <= 0 or atr <= 0:
            return 0.0

        prox = self.cfg.level_proximity_atr

        # Collect all relevant levels
        key_levels = {
            "prior_close": levels.prior_close,
            "overnight_high": levels.overnight_high,
            "overnight_low": levels.overnight_low if levels.overnight_low < float("inf") else 0.0,
            "opening_range_high": levels.opening_range_high,
            "opening_range_low": levels.opening_range_low if levels.opening_range_low < float("inf") else 0.0,
            "day_high": levels.day_high,
            "day_low": levels.day_low if levels.day_low < float("inf") else 0.0,
        }
        # Add round numbers
        for i, rn in enumerate(levels.round_numbers_near(price)):
            key_levels[f"round_{i}"] = rn

        # Find nearest support and resistance
        nearest_above = float("inf")
        nearest_below = 0.0
        for name, level in key_levels.items():
            if level <= 0:
                continue
            if level > price:
                nearest_above = min(nearest_above, level)
            elif level < price:
                nearest_below = max(nearest_below, level)

        above_dist = (nearest_above - price) / atr if nearest_above < float("inf") else 99.0
        below_dist = (price - nearest_below) / atr if nearest_below > 0 else 99.0

        if regime == Regime.MEAN_REVERT:
            # Near resistance from below → expect rejection → slight short bias
            # Near support from above → expect bounce → slight long bias
            if above_dist < prox:
                return -30.0  # close to resistance, lean short
            if below_dist < prox:
                return 30.0   # close to support, lean long
            return 0.0

        elif regime == Regime.TRENDING:
            # Breaking above resistance = bullish. Breaking below support = bearish.
            # Price above all nearby levels = uptrend confirmed
            if above_dist > 3.0 and below_dist < prox:
                # Just above a level, trending up through it
                return 40.0
            if below_dist > 3.0 and above_dist < prox:
                # Just below a level, trending down through it
                return -40.0
            return 0.0

        return 0.0


def is_event_day(ts: datetime) -> bool:
    """Check if today has a major economic event."""
    today_str = ts.strftime("%Y-%m-%d")
    return any(ds == today_str for ds, _ in ECONOMIC_EVENTS)


def event_name_today(ts: datetime) -> str:
    today_str = ts.strftime("%Y-%m-%d")
    events = [name for ds, name in ECONOMIC_EVENTS if ds == today_str]
    return ", ".join(events) if events else ""


class SignalEngine:
    """
    Master signal generator. Computes regime, individual signals,
    and the weighted composite score.
    """

    def __init__(self, regime_cfg: RegimeConfig, signal_cfg: SignalConfig):
        self.regime_detector = RegimeDetector(regime_cfg)
        self.vwap_signal = VWAPSignal(signal_cfg)
        self.momentum_signal = MomentumSignal(signal_cfg)
        self.level_signal = KeyLevelSignal(signal_cfg)
        self.cfg = signal_cfg

    def evaluate(self, data: MarketData) -> SignalSnapshot:
        """Run all signals and produce a composite score."""
        now = datetime.now(ET)
        regime = self.regime_detector.detect(data)

        vwap_sig, vwap_z = self.vwap_signal.calculate(data, regime)
        mom_sig = self.momentum_signal.calculate(data)
        level_sig = self.level_signal.calculate(data, regime)

        # IV/RV context
        rv = data.realized_vol * 100
        iv = data.atm_iv * 100 if data.atm_iv > 0 else data.vix
        iv_rv = iv / rv if rv > 0 else 2.0

        event_day = is_event_day(now)

        # Regime-adaptive weighting
        if regime == Regime.MEAN_REVERT:
            # Heavy weight on VWAP fade, moderate on levels, low on momentum
            composite = (
                vwap_sig * 0.55 +
                level_sig * 0.30 +
                mom_sig * 0.15   # small weight — want to ensure we're not fading a runaway
            )
        elif regime == Regime.TRENDING:
            # Heavy weight on momentum, moderate VWAP (trend direction), levels for confirmation
            composite = (
                mom_sig * 0.50 +
                vwap_sig * 0.30 +
                level_sig * 0.20
            )
        else:
            # VOLATILE: signals are noisy, dampen everything
            composite = (vwap_sig * 0.2 + mom_sig * 0.2 + level_sig * 0.1) * 0.5

        # Dampen on event days (signals less reliable)
        if event_day:
            composite *= 0.6

        composite = max(min(composite, 100.0), -100.0)

        snap = SignalSnapshot(
            timestamp=now,
            regime=regime,
            vwap_signal=round(vwap_sig, 1),
            momentum_signal=round(mom_sig, 1),
            level_signal=round(level_sig, 1),
            iv_rv_ratio=round(iv_rv, 2),
            vix=data.vix,
            skew=data.put_call_skew,
            composite=round(composite, 1),
            vwap_zscore=round(vwap_z, 2),
            atr=round(data.atr, 2),
            is_event_day=event_day,
        )

        log.debug(
            f"Signal: regime={regime.value} composite={composite:.1f} "
            f"vwap={vwap_sig:.0f}(z={vwap_z:.1f}) mom={mom_sig:.0f} "
            f"lvl={level_sig:.0f} IV/RV={iv_rv:.1f} VIX={data.vix:.1f}"
        )
        return snap
