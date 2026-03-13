"""
Session phase detection, economic calendar, and position sizing utilities.

Extracted from market_bot_v26.py.
"""

from __future__ import annotations

from datetime import datetime, time as dtime
from typing import Optional, Tuple

from bot_config import logger, now_et, CFG, HTTP, _cache_lock, ET


# =================================================================
# --- SESSION PHASE ---
# =================================================================

def get_session_phase() -> str:
    t = now_et().time()
    if t < dtime(4, 0):
        return "OVERNIGHT"
    elif t < dtime(9, 30):
        return "PRE-MARKET"
    elif t < dtime(10, 30):
        return "OPENING DRIVE"
    elif t < dtime(12, 0):
        return "MID-MORNING"
    elif t < dtime(13, 30):
        return "LUNCH CHOP"
    elif t < dtime(15, 50):
        return "AFTERNOON / POWER HOUR"
    elif t < dtime(16, 15):
        return "CLOSING IMBALANCE"
    elif t < dtime(18, 0):
        return "POST-CLOSE"
    else:
        return "GLOBEX OVERNIGHT"


# =================================================================
# --- ECONOMIC CALENDAR ---
# =================================================================

_econ_calendar_cache: list = []
_econ_calendar_fetched: Optional[datetime] = None


def _fetch_economic_calendar() -> list:
    """Fetch today's high-impact economic events. Caches for 4 hours."""
    global _econ_calendar_cache, _econ_calendar_fetched
    now = now_et()

    # Return cache if fresh (fetched today within last 4 hours)
    with _cache_lock:
        if _econ_calendar_fetched and (now - _econ_calendar_fetched).total_seconds() < 14400:
            return list(_econ_calendar_cache)

    events = []
    try:
        import requests
        today_str = now.strftime("%Y-%m-%d")
        url = f"https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for item in data:
                event_date = str(item.get("date", ""))[:10]
                country = item.get("country", "")
                impact = item.get("impact", "").lower()
                title = item.get("title", "")
                event_time_str = str(item.get("date", ""))

                if event_date != today_str:
                    continue
                if country != "USD":
                    continue
                if impact not in ("high", "medium"):
                    continue

                try:
                    from dateutil import parser as dtparser
                    event_dt = dtparser.parse(event_time_str)
                    if event_dt.tzinfo is None:
                        event_dt = event_dt.replace(tzinfo=ET)
                    else:
                        event_dt = event_dt.astimezone(ET)
                except Exception:
                    continue

                events.append({
                    "title": title,
                    "time": event_dt,
                    "impact": impact,
                })
                logger.info(f"Econ event: {title} @ {event_dt.strftime('%H:%M')} ET ({impact})")

    except Exception as e:
        logger.warning(f"Economic calendar fetch failed: {e}")

    with _cache_lock:
        _econ_calendar_cache = events
        _econ_calendar_fetched = now
    return events


def is_news_approaching() -> Tuple[bool, Optional[str], Optional[datetime], Optional[str]]:
    """Returns (is_approaching, event_label, event_datetime, impact) if a real economic event is within 60 minutes.
    impact is 'high' or 'medium'."""
    now = now_et()
    events = _fetch_economic_calendar()

    for ev in events:
        event_dt = ev["time"]
        minutes_until = (event_dt - now).total_seconds() / 60
        if 0 < minutes_until <= 60:
            impact_tag = "HIGH" if ev["impact"] == "high" else "MED"
            label = f"[{impact_tag}] {ev['title']} @ {event_dt.strftime('%H:%M')} ET"
            return True, label, event_dt, ev["impact"]

    return False, None, None, None


def is_news_blackout(
    pre_minutes: int = 6,
    post_minutes: int = 5,
) -> Tuple[bool, Optional[str], Optional[datetime], Optional[str]]:
    """Check if we're in a news blackout window.

    Returns (in_blackout, event_label, event_datetime, phase) where phase is
    'PRE' (before event) or 'POST' (after event).

    Blackout = pre_minutes before through post_minutes after any high/medium
    impact USD economic event.
    """
    now = now_et()
    events = _fetch_economic_calendar()

    for ev in events:
        event_dt = ev["time"]
        delta_seconds = (event_dt - now).total_seconds()
        minutes_until = delta_seconds / 60

        impact_tag = "HIGH" if ev["impact"] == "high" else "MED"
        label = f"[{impact_tag}] {ev['title']} @ {event_dt.strftime('%H:%M')} ET"

        # PRE-event window: 0 < minutes_until <= pre_minutes
        if 0 < minutes_until <= pre_minutes:
            return True, label, event_dt, "PRE"

        # POST-event window: event just happened, within post_minutes
        minutes_since = -minutes_until  # positive value = minutes past
        if 0 <= minutes_since <= post_minutes:
            return True, label, event_dt, "POST"

    return False, None, None, None


# =================================================================
# --- POSITION SIZING ---
# =================================================================

def position_suggestion(confidence: int, flat_threshold: int = 60,
                        win_rate: float = 0.0, avg_win: float = 0.0,
                        avg_loss: float = 0.0, total_trades: int = 0) -> Tuple[int, str]:
    """
    Map confidence % to a contract tier from CFG.POSITION_TIERS.

    Default tiers: 60%+ = 1 ct, 70%+ = 2 ct, 80%+ = 3 ct, 90%+ = 4 ct.

    v29: When KELLY_SIZING_ENABLED, applies half-Kelly fraction to scale
    the tier thresholds based on rolling edge. Requires min trades before
    activating. Negative edge reduces all tiers by 1.
    """
    if confidence < flat_threshold:
        return 0, f"NO TRADE (Conf {confidence}% < {flat_threshold}%)"

    tiers = list(CFG.POSITION_TIERS)

    # v29 Kelly criterion adjustment
    kelly_info = ""
    if (CFG.KELLY_SIZING_ENABLED
            and total_trades >= CFG.KELLY_MIN_TRADES
            and avg_win > 0 and avg_loss > 0):
        # Kelly fraction: f = (p*b - q) / b where p=win_rate, b=avg_win/avg_loss, q=1-p
        b = avg_win / avg_loss
        kelly_full = (win_rate * b - (1 - win_rate)) / b if b > 0 else 0
        kelly_f = kelly_full * CFG.KELLY_FRACTION  # half-Kelly

        if kelly_f <= 0:
            # Negative edge — reduce all tiers by 1 contract (floor at 1)
            tiers = [(t, max(1, c - 1)) for t, c in tiers]
            kelly_info = f" [Kelly={kelly_f:.2f}, reduced]"
        elif kelly_f < 0.5:
            # Small edge — cap at 2 contracts max
            tiers = [(t, min(c, 2)) for t, c in tiers]
            kelly_info = f" [Kelly={kelly_f:.2f}, capped@2]"
        elif kelly_f >= 0.8:
            # Strong edge — bump high tiers by 1 (cap at 5)
            tiers = [(t, min(5, c + 1) if c >= 3 else c) for t, c in tiers]
            kelly_info = f" [Kelly={kelly_f:.2f}, boosted]"
        else:
            kelly_info = f" [Kelly={kelly_f:.2f}]"

    contracts = 1  # fallback
    for threshold, cts in sorted(tiers, reverse=True):
        if confidence >= threshold:
            contracts = cts
            break
    return contracts, f"{contracts} ct ({confidence}% conf{kelly_info})"
