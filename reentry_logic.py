"""
Re-entry logic (v29 Feature #12) — Smart re-entry after stop-outs.

After a stop-out, if the same directional signal persists AND price
returns to the original entry zone within a time window, allows
re-entry with a tighter stop. Max 1 re-entry per original signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List

from bot_config import CFG, now_et

logger = logging.getLogger("reentry_logic")


@dataclass
class StopoutRecord:
    """Records a stop-out for potential re-entry."""
    trade_id: int
    verdict: str           # BULLISH/BEARISH
    entry_price: float     # original entry
    stop_price: float      # original stop
    target_price: float    # original target
    stopout_time: datetime
    stopout_price: float
    original_stop_dist: float  # abs(entry - stop) in pts
    reentry_used: bool = False  # only 1 re-entry per stop-out


class ReEntryTracker:
    """
    Tracks recent stop-outs and evaluates re-entry conditions.

    Conditions for re-entry:
    1. Same directional verdict persists (Claude still sees the setup)
    2. Price returned to original entry zone (±REENTRY_ZONE_PTS)
    3. Within REENTRY_MAX_WAIT_MIN minutes of stop-out
    4. Tighter stop (REENTRY_STOP_FACTOR of original distance)
    5. Max 1 re-entry per original signal
    """

    def __init__(self):
        self.recent_stopouts: List[StopoutRecord] = []
        self._max_history = 10  # keep last N stop-outs

    def record_stopout(self, trade: dict, stopout_price: float):
        """Log a stop-out for potential re-entry tracking."""
        if not CFG.REENTRY_ENABLED:
            return

        entry = float(trade.get("price", 0))
        stop = float(trade.get("stop", 0))
        target = float(trade.get("target", 0))
        verdict = str(trade.get("verdict", "")).upper()

        if entry <= 0 or not verdict:
            return

        stop_dist = abs(entry - stop) if stop > 0 else 10.0

        record = StopoutRecord(
            trade_id=trade.get("id", 0),
            verdict=verdict,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            stopout_time=now_et().replace(tzinfo=None),
            stopout_price=stopout_price,
            original_stop_dist=stop_dist,
        )
        self.recent_stopouts.append(record)

        # Trim old records
        if len(self.recent_stopouts) > self._max_history:
            self.recent_stopouts = self.recent_stopouts[-self._max_history:]

        logger.info(
            f"[REENTRY] Recorded stop-out: {verdict} entry={entry:.2f} "
            f"stop={stop:.2f} at {stopout_price:.2f} | "
            f"Re-entry zone: [{entry - CFG.REENTRY_ZONE_PTS:.2f}, "
            f"{entry + CFG.REENTRY_ZONE_PTS:.2f}]"
        )

    def check_reentry(self, current_price: float, current_verdict: str,
                      current_confidence: int) -> Optional[dict]:
        """
        Check if re-entry conditions are met for any recent stop-out.

        Returns a dict with re-entry parameters if conditions met, else None.
        """
        if not CFG.REENTRY_ENABLED:
            return None
        if not self.recent_stopouts:
            return None

        now = now_et().replace(tzinfo=None)
        current_is_bull = any(k in current_verdict.upper() for k in ("BULL", "LONG", "BUY"))
        current_is_bear = any(k in current_verdict.upper() for k in ("BEAR", "SHORT", "SELL"))

        for record in reversed(self.recent_stopouts):
            if record.reentry_used:
                continue

            # Check time window
            elapsed_min = (now - record.stopout_time).total_seconds() / 60
            if elapsed_min > CFG.REENTRY_MAX_WAIT_MIN:
                continue

            # Check same direction
            record_is_bull = any(k in record.verdict for k in ("BULL", "LONG", "BUY"))
            if record_is_bull and not current_is_bull:
                continue
            if not record_is_bull and not current_is_bear:
                continue

            # Check price in entry zone
            zone_low = record.entry_price - CFG.REENTRY_ZONE_PTS
            zone_high = record.entry_price + CFG.REENTRY_ZONE_PTS
            if not (zone_low <= current_price <= zone_high):
                continue

            # All conditions met — calculate re-entry parameters
            record.reentry_used = True
            tighter_stop_dist = record.original_stop_dist * CFG.REENTRY_STOP_FACTOR

            if record_is_bull:
                new_stop = current_price - tighter_stop_dist
                new_target = record.target_price  # keep original target
            else:
                new_stop = current_price + tighter_stop_dist
                new_target = record.target_price

            # Snap to ES tick
            new_stop = round(new_stop * 4) / 4
            new_target = round(new_target * 4) / 4

            logger.info(
                f"[REENTRY] Re-entry triggered! {record.verdict} at {current_price:.2f} "
                f"(original entry {record.entry_price:.2f}, stopped {elapsed_min:.0f}m ago) | "
                f"Tighter stop: {new_stop:.2f} (was {record.stop_price:.2f})"
            )

            return {
                "verdict": record.verdict,
                "entry_price": current_price,
                "target": new_target,
                "stop": new_stop,
                "confidence": current_confidence,
                "original_trade_id": record.trade_id,
                "reason": (
                    f"Re-entry: same signal persisted {elapsed_min:.0f}m after stop-out, "
                    f"price returned to entry zone, tighter stop ({CFG.REENTRY_STOP_FACTOR:.0%})"
                ),
            }

        return None

    def cleanup_expired(self):
        """Remove stop-outs older than 2x the max wait window."""
        if not self.recent_stopouts:
            return
        now = now_et().replace(tzinfo=None)
        cutoff = timedelta(minutes=CFG.REENTRY_MAX_WAIT_MIN * 2)
        self.recent_stopouts = [
            r for r in self.recent_stopouts
            if (now - r.stopout_time) < cutoff
        ]
