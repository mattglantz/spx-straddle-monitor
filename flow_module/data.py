"""
Historical and real-time data collection from IBKR.

Fetches daily bars for ES futures to power all flow calculations:
- Moving averages (CTA triggers)
- Month-to-date returns (rebalancing)
- Realized volatility (vol-control flows)

Also provides live price and VIX for the dashboard.
"""

import asyncio
import math
import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional

from ib_async import IB, ContFuture, Index

from flow_module.config import (
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, ES_SYMBOL, ES_EXCHANGE, ET,
)

log = logging.getLogger(__name__)


@dataclass
class DailyBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return 0.0 if math.isnan(val) or math.isinf(val) else float(val)
    except (TypeError, ValueError):
        return 0.0


class MarketDataStore:
    """
    Collects and stores historical daily bars and live prices.
    """

    def __init__(self):
        self.daily_bars: list[DailyBar] = []
        self.live_price: float = 0.0
        self.live_vix: float = 0.0
        self.vix_close: float = 0.0
        self.last_update: Optional[datetime] = None

    async def connect_and_fetch(self, ib: IB, lookback_days: int = 252):
        """Fetch historical daily bars for ES and current VIX."""
        # Qualify ES continuous futures
        es = ContFuture(ES_SYMBOL, ES_EXCHANGE)
        await ib.qualifyContractsAsync(es)

        # Fetch daily bars
        log.info(f"Fetching {lookback_days} days of ES daily bars...")
        bars = await ib.reqHistoricalDataAsync(
            es,
            endDateTime="",
            durationStr=f"{lookback_days} D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        self.daily_bars = []
        for bar in bars:
            d = bar.date if isinstance(bar.date, date) else datetime.strptime(
                str(bar.date)[:10], "%Y-%m-%d").date()
            self.daily_bars.append(DailyBar(
                date=d,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=int(bar.volume),
            ))

        self.daily_bars.sort(key=lambda b: b.date)
        log.info(f"Loaded {len(self.daily_bars)} daily bars "
                 f"({self.daily_bars[0].date} to {self.daily_bars[-1].date})")

        # Subscribe to live ES + VIX
        es_ticker = ib.reqMktData(es, "", False, False)
        vix_contract = Index("VIX", "CBOE")
        await ib.qualifyContractsAsync(vix_contract)
        vix_ticker = ib.reqMktData(vix_contract, "", False, False)

        return es_ticker, vix_ticker

    def update_live(self, es_ticker, vix_ticker):
        """Pull latest prices from ticker objects."""
        price = _safe_float(es_ticker.last) or _safe_float(es_ticker.close)
        if price > 0:
            self.live_price = price

        vix = _safe_float(vix_ticker.last) or _safe_float(vix_ticker.close)
        if vix > 0:
            self.live_vix = vix
        vix_close = _safe_float(vix_ticker.close)
        if vix_close > 0:
            self.vix_close = vix_close

        self.last_update = datetime.now(ET)

    def append_today(self, price: float):
        """Add/update today's bar in the series (for intraday calcs)."""
        today = datetime.now(ET).date()
        if self.daily_bars and self.daily_bars[-1].date == today:
            # Update today's bar
            bar = self.daily_bars[-1]
            bar.high = max(bar.high, price)
            bar.low = min(bar.low, price)
            bar.close = price
        elif price > 0:
            self.daily_bars.append(DailyBar(
                date=today,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0,
            ))

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.daily_bars]

    @property
    def last_close(self) -> float:
        return self.daily_bars[-1].close if self.daily_bars else 0.0

    def close_on_date(self, target: date) -> Optional[float]:
        """Get closing price on or just before a date."""
        for bar in reversed(self.daily_bars):
            if bar.date <= target:
                return bar.close
        return None

    def bars_since(self, start_date: date) -> list[DailyBar]:
        """Get all bars from start_date onward."""
        return [b for b in self.daily_bars if b.date >= start_date]

    def last_n_closes(self, n: int) -> list[float]:
        """Get the last N closing prices."""
        return [b.close for b in self.daily_bars[-n:]]

    def moving_average(self, period: int) -> float:
        """Simple moving average of last N closes."""
        closes = self.last_n_closes(period)
        return sum(closes) / len(closes) if closes else 0.0

    def realized_vol(self, window: int = 21) -> float:
        """Annualized realized volatility from daily closes."""
        closes = self.last_n_closes(window + 1)
        if len(closes) < 3:
            return 0.0
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                returns.append(math.log(closes[i] / closes[i - 1]))
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(var * 252) * 100  # annualized, in %
