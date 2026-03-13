"""
Entry timing engine (v29 Feature #4) — Regime-aware entry quality scoring.

Instead of entering immediately when confidence > threshold, this module
scores the setup quality and optionally queues entries waiting for price
to pull back to a favorable level (VWAP, key level, anchored VWAP).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict

from bot_config import CFG, now_et

logger = logging.getLogger("entry_timing")


@dataclass
class PendingEntry:
    """An entry waiting for a pullback to a favorable level."""
    verdict: str
    confidence: int
    target: float
    stop: float
    contracts: int
    entry_zone_low: float   # lower bound of entry zone
    entry_zone_high: float  # upper bound of entry zone
    original_price: float   # price when signal was generated
    metrics: dict           # full metrics snapshot
    signals: dict           # signal attribution
    reasoning: str
    session: str
    created_at: datetime = field(default_factory=lambda: now_et().replace(tzinfo=None))
    bars_waited: int = 0
    max_wait_bars: int = 6
    setup_quality: float = 0.0


class EntryTimer:
    """
    Manages pending entries and evaluates setup quality.

    When ENTRY_TIMING_ENABLED, trades are not entered immediately.
    Instead, the setup quality is scored. If quality is below threshold,
    the entry is queued for up to N bars, waiting for price to pull back
    to VWAP or a key level before entering.
    """

    def __init__(self):
        self.pending: List[PendingEntry] = []

    def calc_setup_quality(self, metrics: dict, verdict: str, entry_price: float) -> float:
        """
        Score the entry setup quality from 0-100.

        Factors:
        - Distance to VWAP (closer = better entry)
        - Distance to nearest key level (support for longs, resistance for shorts)
        - Session phase alignment (opening drive, trend hours)
        - ATR position within day range (buy low in range, sell high)
        """
        score = 50.0  # baseline
        is_long = "BULL" in verdict.upper() or "LONG" in verdict.upper()

        # 1. VWAP distance (0-20 pts) — best entry is near or below VWAP for longs
        try:
            vwap_val = float(metrics.get("vwap_val", 0))
        except (ValueError, TypeError):
            vwap_val = 0
        if vwap_val > 0 and entry_price > 0:
            vwap_dist_pct = (entry_price - vwap_val) / vwap_val * 100
            if is_long:
                # Buying below VWAP = great, above = worse
                if vwap_dist_pct <= -0.1:
                    score += 20  # below VWAP — excellent
                elif vwap_dist_pct <= 0.05:
                    score += 15  # near VWAP
                elif vwap_dist_pct <= 0.15:
                    score += 5   # slightly above
                else:
                    score -= 10  # chasing
            else:
                # Selling above VWAP = great
                if vwap_dist_pct >= 0.1:
                    score += 20
                elif vwap_dist_pct >= -0.05:
                    score += 15
                elif vwap_dist_pct >= -0.15:
                    score += 5
                else:
                    score -= 10

        # 2. Key level proximity (0-15 pts) — are we near support/resistance?
        prior = metrics.get("prior", {})
        if prior:
            levels = []
            for k in ("prev_high", "prev_low", "prior_close"):
                v = prior.get(k, 0)
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    continue
                if v > 0:
                    levels.append(v)
            if levels and entry_price > 0:
                distances = [abs(entry_price - lv) for lv in levels]
                nearest_dist = min(distances)
                if nearest_dist <= 3.0:
                    score += 15  # right at a level — strong reaction zone
                elif nearest_dist <= 8.0:
                    score += 8   # near a level
                elif nearest_dist <= 15.0:
                    score += 3   # moderate proximity

        # 3. Session phase alignment (0-10 pts)
        try:
            hour = now_et().hour
            minute = now_et().minute
            time_val = hour + minute / 60
            if 9.5 <= time_val <= 10.5:
                score += 10  # opening drive — strong directional moves
            elif 13.0 <= time_val <= 15.5:
                score += 5   # afternoon trend
            elif 11.0 <= time_val <= 13.0:
                score -= 5   # lunch lull — choppy
        except Exception:
            pass

        # 4. Confluence level bonus (0-5 pts)
        confl = metrics.get("confluence_level", "NONE")
        if confl == "STRONG":
            score += 5
        elif confl == "MODERATE":
            score += 3

        return max(0, min(100, round(score, 1)))

    def should_enter_immediately(self, setup_quality: float) -> bool:
        """High-quality setups enter immediately."""
        return setup_quality >= CFG.ENTRY_TIMING_MIN_QUALITY

    def add_pending(self, verdict: str, confidence: int, target: float,
                    stop: float, contracts: int, entry_price: float,
                    metrics: dict, signals: dict, reasoning: str,
                    session: str, setup_quality: float) -> PendingEntry:
        """Queue an entry waiting for a pullback."""
        # Calculate entry zone: VWAP ± 2 pts for longs, or key level proximity
        is_long = "BULL" in verdict.upper() or "LONG" in verdict.upper()
        vwap = metrics.get("vwap_val", entry_price)

        if is_long:
            # For longs, want to buy at or below VWAP
            zone_high = min(entry_price, vwap + 2.0)
            zone_low = vwap - 5.0
        else:
            # For shorts, want to sell at or above VWAP
            zone_low = max(entry_price, vwap - 2.0)
            zone_high = vwap + 5.0

        pending = PendingEntry(
            verdict=verdict,
            confidence=confidence,
            target=target,
            stop=stop,
            contracts=contracts,
            entry_zone_low=round(zone_low * 4) / 4,
            entry_zone_high=round(zone_high * 4) / 4,
            original_price=entry_price,
            metrics=metrics,
            signals=signals,
            reasoning=reasoning,
            session=session,
            max_wait_bars=CFG.ENTRY_TIMING_MAX_WAIT_BARS,
            setup_quality=setup_quality,
        )
        # Replace any existing pending entry in same direction
        self.pending = [p for p in self.pending
                        if ("BULL" in p.verdict.upper()) != ("BULL" in verdict.upper())]
        self.pending.append(pending)
        logger.info(
            f"[ENTRY TIMER] Queued {verdict} entry — quality {setup_quality:.0f} "
            f"(threshold {CFG.ENTRY_TIMING_MIN_QUALITY}), zone [{zone_low:.2f}-{zone_high:.2f}], "
            f"max wait {CFG.ENTRY_TIMING_MAX_WAIT_BARS} bars"
        )
        return pending

    def check_pending(self, current_price: float) -> List[PendingEntry]:
        """
        Check if any pending entries should be triggered.
        Returns list of entries ready to execute.
        """
        ready = []
        still_pending = []

        for p in self.pending:
            p.bars_waited += 1

            # Check if price is in the entry zone
            in_zone = p.entry_zone_low <= current_price <= p.entry_zone_high

            if in_zone:
                logger.info(
                    f"[ENTRY TIMER] {p.verdict} triggered — price {current_price:.2f} "
                    f"in zone [{p.entry_zone_low:.2f}-{p.entry_zone_high:.2f}] "
                    f"after {p.bars_waited} bars"
                )
                # Adjust entry price to current price
                p.original_price = current_price
                ready.append(p)
            elif p.bars_waited >= p.max_wait_bars:
                # Expired — enter at current price anyway (signal still valid)
                logger.info(
                    f"[ENTRY TIMER] {p.verdict} expired after {p.bars_waited} bars — "
                    f"entering at market {current_price:.2f}"
                )
                p.original_price = current_price
                ready.append(p)
            else:
                still_pending.append(p)

        self.pending = still_pending
        return ready

    def cancel_conflicting(self, verdict: str):
        """Cancel pending entries that conflict with a new signal direction."""
        is_long = "BULL" in verdict.upper() or "LONG" in verdict.upper()
        before = len(self.pending)
        self.pending = [p for p in self.pending
                        if ("BULL" in p.verdict.upper()) == is_long]
        cancelled = before - len(self.pending)
        if cancelled:
            logger.info(f"[ENTRY TIMER] Cancelled {cancelled} conflicting pending entries")

    def clear(self):
        """Clear all pending entries."""
        self.pending.clear()
