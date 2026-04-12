"""
Configuration for the ES Futures Trading Bot.

All tunable parameters in one place. Adjust these to change
bot behavior without touching strategy logic.
"""

from dataclasses import dataclass, field
from datetime import time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ── IBKR Connection ─────────────────────────────────────────
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7496          # 7496 = live TWS, 7497 = paper
IBKR_CLIENT_ID = 10       # separate from monitor (which uses 2)

# ── Instrument ──────────────────────────────────────────────
ES_SYMBOL = "ES"
ES_EXCHANGE = "CME"
ES_TICK_SIZE = 0.25        # minimum tick for ES
ES_POINT_VALUE = 50.0      # $50 per point for ES

# ── Session Times (Eastern) ─────────────────────────────────
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
# Globex opens Sun 6pm, closes Fri 5pm — we track overnight as 18:00–09:30
GLOBEX_OPEN = time(18, 0)
NO_NEW_TRADES_BEFORE_CLOSE_MIN = 15  # no new entries in last 15 min of RTH


@dataclass
class BarConfig:
    """How we build bars from ticks."""
    fast_period: int = 1       # 1-minute bars for entries
    slow_period: int = 5       # 5-minute bars for trend context
    lookback_bars: int = 200   # history to keep in memory


@dataclass
class RegimeConfig:
    """Thresholds for regime detection."""
    # VIX thresholds
    vix_low: float = 16.0          # below = calm / mean-reversion
    vix_high: float = 26.0         # above = volatile / stand aside
    # IV / Realized Vol ratio
    iv_rv_mean_revert: float = 1.2  # IV >> RV = vol overpriced = mean-revert
    iv_rv_trending: float = 0.85    # RV >> IV = market moving = trend
    # Put/Call skew (25-delta risk reversal, in vol pts)
    skew_elevated: float = 8.0      # above = dealers short puts = negative gamma
    # Realized vol lookback (in 5-min bars)
    rv_lookback: int = 60           # ~5 hours of 5-min bars


@dataclass
class SignalConfig:
    """Thresholds for signal generation."""
    # VWAP z-score thresholds (in ATR units)
    vwap_fade_entry: float = 1.8     # enter fade when |z| > this
    vwap_fade_exit: float = 0.3      # exit fade near VWAP
    # Momentum
    momentum_lookback: int = 10      # 5-min bars for momentum calc
    momentum_threshold: float = 0.4  # min ROC % to confirm trend
    # Key levels
    level_proximity_atr: float = 0.5  # how close = "at a level"
    # Composite signal thresholds
    entry_threshold: float = 55.0     # |composite| must exceed to enter
    exit_threshold: float = 15.0      # |composite| below = exit


@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_position_contracts: int = 2
    max_daily_loss_dollars: float = 1000.0
    max_trades_per_day: int = 8
    # Stop / Target in ATR multiples
    mean_revert_stop_atr: float = 1.5
    mean_revert_target_atr: float = 1.5
    trend_stop_atr: float = 2.5
    trend_target_atr: float = 4.0
    # Trailing stop (trend mode only): distance in ATR
    trend_trail_atr: float = 2.0
    # ATR period (1-min bars)
    atr_period: int = 14
    # Cooldown after loss (seconds)
    loss_cooldown_sec: int = 300
    # Event day sizing (fraction of normal)
    event_day_size_fraction: float = 0.5


@dataclass
class BotConfig:
    """Master config aggregating all sub-configs."""
    bars: BarConfig = field(default_factory=BarConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    # How often the strategy evaluates (seconds)
    eval_interval_sec: int = 5
    # Logging
    log_level: str = "INFO"
    # Paper mode guard — extra confirmation required to go live
    paper_mode: bool = True


# ── Economic Calendar (imported from monitor) ───────────────
ECONOMIC_EVENTS = [
    ("2025-01-29", "FOMC"), ("2025-03-19", "FOMC"),
    ("2025-05-07", "FOMC"), ("2025-06-18", "FOMC"),
    ("2025-07-30", "FOMC"), ("2025-09-17", "FOMC"),
    ("2025-10-29", "FOMC"), ("2025-12-10", "FOMC"),
    ("2026-01-28", "FOMC"), ("2026-03-18", "FOMC"),
    ("2026-05-06", "FOMC"), ("2026-06-17", "FOMC"),
    ("2026-07-29", "FOMC"), ("2026-09-16", "FOMC"),
    ("2026-10-28", "FOMC"), ("2026-12-09", "FOMC"),
    ("2025-01-15", "CPI"), ("2025-02-12", "CPI"), ("2025-03-12", "CPI"),
    ("2025-04-10", "CPI"), ("2025-05-13", "CPI"), ("2025-06-11", "CPI"),
    ("2025-07-10", "CPI"), ("2025-08-12", "CPI"), ("2025-09-10", "CPI"),
    ("2025-10-14", "CPI"), ("2025-11-12", "CPI"), ("2025-12-10", "CPI"),
    ("2026-01-14", "CPI"), ("2026-02-11", "CPI"), ("2026-03-11", "CPI"),
    ("2026-04-14", "CPI"), ("2026-05-12", "CPI"), ("2026-06-10", "CPI"),
    ("2026-07-14", "CPI"), ("2026-08-12", "CPI"), ("2026-09-15", "CPI"),
    ("2026-10-13", "CPI"), ("2026-11-10", "CPI"), ("2026-12-10", "CPI"),
    ("2025-01-10", "NFP"), ("2025-02-07", "NFP"), ("2025-03-07", "NFP"),
    ("2025-04-04", "NFP"), ("2025-05-02", "NFP"), ("2025-06-06", "NFP"),
    ("2025-07-03", "NFP"), ("2025-08-01", "NFP"), ("2025-09-05", "NFP"),
    ("2025-10-03", "NFP"), ("2025-11-07", "NFP"), ("2025-12-05", "NFP"),
    ("2026-01-09", "NFP"), ("2026-02-06", "NFP"), ("2026-03-06", "NFP"),
    ("2026-04-03", "NFP"), ("2026-05-01", "NFP"), ("2026-06-05", "NFP"),
    ("2026-07-02", "NFP"), ("2026-08-07", "NFP"), ("2026-09-04", "NFP"),
    ("2026-10-02", "NFP"), ("2026-11-06", "NFP"), ("2026-12-04", "NFP"),
]
