"""
Configuration for the Structural Flow Module.
"""

from dataclasses import dataclass, field
from datetime import time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ── IBKR Connection ─────────────────────────────────────────
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7496           # 7496 = live TWS, 7497 = paper
IBKR_CLIENT_ID = 20        # unique from monitor (2) and bot (10)

# ── Instrument ──────────────────────────────────────────────
ES_SYMBOL = "ES"
ES_EXCHANGE = "CME"
ES_POINT_VALUE = 50.0

# ── Dashboard ───────────────────────────────────────────────
DASH_PORT = 8060            # separate from monitor (8050)


@dataclass
class RebalanceConfig:
    """Pension / fund end-of-month rebalancing."""
    # How many trading days before month-end the window opens
    window_days: int = 3
    # Quarter-end is stronger than month-end
    quarter_multiplier: float = 3.0
    # MTD return threshold to consider flow significant (%)
    min_mtd_return_pct: float = 1.5
    # Estimated AUM that rebalances (trillions) — for display context
    estimated_rebalance_aum_t: float = 5.0


@dataclass
class OpExConfig:
    """Options expiration gamma unwind."""
    # Days before OpEx to flag (pre-pin behavior)
    pre_opex_days: int = 2
    # Days after OpEx to flag (unwind / vol release)
    post_opex_days: int = 1


@dataclass
class CTAConfig:
    """CTA trend-following trigger levels."""
    # Moving average periods to track
    ma_periods: list[int] = field(default_factory=lambda: [50, 100, 200])
    # How close price must be to MA to flag (% of price)
    proximity_pct: float = 1.0
    # Estimated CTA AUM (billions) — for display context
    estimated_cta_aum_b: float = 300.0


@dataclass
class VolControlConfig:
    """Volatility control / risk parity flows."""
    # Realized vol lookback periods (trading days)
    short_rv_window: int = 21       # 1 month
    long_rv_window: int = 63        # 3 months
    # RV rate-of-change lookback (days)
    rv_roc_window: int = 5
    # Threshold: RV drop rate that triggers "vol compression buy" (% per week)
    rv_drop_threshold: float = -2.0
    # Threshold: RV rise rate that triggers "vol expansion sell"
    rv_rise_threshold: float = 3.0
    # VIX threshold — elevated VIX makes vol-control selling more impactful
    vix_elevated: float = 22.0


@dataclass
class BuybackConfig:
    """Corporate buyback window tracking."""
    # Approximate earnings season months (when blackout starts)
    earnings_months: list[int] = field(default_factory=lambda: [1, 4, 7, 10])
    # Blackout starts this many days into the earnings month
    blackout_start_day: int = 10
    # Blackout ends this many days into the month after earnings
    blackout_end_day: int = 5
    # Estimated daily buyback flow (billions)
    estimated_daily_buyback_b: float = 5.0


@dataclass
class FlowConfig:
    """Master config."""
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    opex: OpExConfig = field(default_factory=OpExConfig)
    cta: CTAConfig = field(default_factory=CTAConfig)
    vol_control: VolControlConfig = field(default_factory=VolControlConfig)
    buyback: BuybackConfig = field(default_factory=BuybackConfig)
    # Historical data lookback (calendar days)
    history_lookback_days: int = 252


# ── OpEx / Economic Calendar ────────────────────────────────
MONTHLY_OPEX = [
    "2025-01-17", "2025-02-21", "2025-03-21", "2025-04-17",
    "2025-05-16", "2025-06-20", "2025-07-18", "2025-08-15",
    "2025-09-19", "2025-10-17", "2025-11-21", "2025-12-19",
    "2026-01-16", "2026-02-20", "2026-03-20", "2026-04-17",
    "2026-05-15", "2026-06-19", "2026-07-17", "2026-08-21",
    "2026-09-18", "2026-10-16", "2026-11-20", "2026-12-18",
]

QUAD_WITCHING = [
    "2025-03-21", "2025-06-20", "2025-09-19", "2025-12-19",
    "2026-03-20", "2026-06-19", "2026-09-18", "2026-12-18",
]

# ── Economic Events (FOMC, CPI, NFP for seasonality) ───────
ECONOMIC_EVENTS = [
    ("2025-01-29", "FOMC"), ("2025-03-19", "FOMC"),
    ("2025-05-07", "FOMC"), ("2025-06-18", "FOMC"),
    ("2025-07-30", "FOMC"), ("2025-09-17", "FOMC"),
    ("2025-10-29", "FOMC"), ("2025-12-10", "FOMC"),
    ("2026-01-28", "FOMC"), ("2026-03-18", "FOMC"),
    ("2026-05-06", "FOMC"), ("2026-06-17", "FOMC"),
    ("2026-07-29", "FOMC"), ("2026-09-16", "FOMC"),
    ("2026-10-28", "FOMC"), ("2026-12-09", "FOMC"),
]

# Month-end dates (last business day) — generated dynamically in rebalance module
