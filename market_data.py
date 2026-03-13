"""
Consolidated market data fetcher — IBKR primary, yfinance fallback.

Extracted from market_bot_v26.py.
"""

from __future__ import annotations

import json
import threading
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import yfinance as yf

from bot_config import logger, CFG, now_et, _cache_lock
from ibkr_client import IBKRClient, IBKR_AVAILABLE

# Module-level cache for heavy historical data that doesn't change between cycles
_md_cache: Dict[str, Any] = {}
_md_cache_time: Optional[datetime] = None
_MD_CACHE_TTL = 300  # Refresh heavy data every 5 min


class MarketData:
    """
    Fetches all market data ONCE per cycle.
    IBKR (live) is primary source. Falls back to yfinance if IBKR unavailable.
    Heavy historical data (5m 60D, daily 1Y) is cached and reused across cycles.
    """

    def __init__(self, ibkr: IBKRClient = None):
        self.ibkr = ibkr
        self.fetch_time = now_et()
        self._gamma_detail = {}
        self.data_source = "IBKR" if (ibkr and ibkr.connected) else "yfinance"
        # Defaults for new data streams (overwritten by fetch if available)
        self.dxy = pd.Series(dtype=float)
        self.vix9d = pd.Series(dtype=float)
        self.tick_nyse = pd.DataFrame()
        self.rsp = pd.DataFrame()
        self.spy = pd.DataFrame()
        self._fetch_all()

    def _fetch_all(self):
        logger.info(f"Fetching market data via {self.data_source}...")

        if self.ibkr and self.ibkr.connected:
            self._fetch_ibkr()
        else:
            self._fetch_yfinance()

        logger.info(f"Market data snapshot complete ({self.data_source}).")
        self._check_data_freshness()

    def _check_data_freshness(self):
        """Warn if latest ES data is stale during market hours."""
        try:
            now = now_et()
            # Only check during market hours (weekdays 9:30-16:00 ET)
            if now.weekday() >= 5:
                return
            from datetime import time as dtime
            if not (dtime(9, 30) <= now.time() <= dtime(16, 0)):
                return
            if hasattr(self, 'es_today_1m') and not self.es_today_1m.empty:
                last_bar = self.es_today_1m.index[-1]
                if hasattr(last_bar, 'tz_localize'):
                    last_bar = pd.Timestamp(last_bar)
                    if last_bar.tzinfo is not None:
                        last_bar = last_bar.tz_convert("America/New_York").tz_localize(None)
                gap_min = (now.replace(tzinfo=None) - pd.Timestamp(last_bar)).total_seconds() / 60
                if gap_min > 30:
                    logger.warning(f"[STALE DATA] Latest ES bar is {gap_min:.0f} min old — data may be stale")
        except Exception as e:
            logger.debug(f"Data freshness check error: {e}")

    def _fetch_with_fallback(self, ibkr_fn, yf_fn, label: str):
        """Try IBKR first, fall back to yfinance. Returns the result or empty DataFrame/Series."""
        try:
            result = ibkr_fn()
            if isinstance(result, pd.DataFrame) and result.empty:
                raise ValueError("Empty DataFrame from IBKR")
            if isinstance(result, pd.Series) and result.empty:
                raise ValueError("Empty Series from IBKR")
            return result
        except Exception as e:
            logger.warning(f"IBKR {label} failed ({e}), falling back to yfinance")
            try:
                return yf_fn()
            except Exception as e2:
                logger.warning(f"yfinance {label} also failed: {e2}")
                return pd.DataFrame()

    def _fetch_ibkr(self):
        """Primary: IBKR live data -- faster, real-time, more history.
        Heavy historical data (5m 60D, daily 1Y) is cached to reduce IBKR API load.
        """
        global _md_cache, _md_cache_time
        ib = self.ibkr

        with _cache_lock:
            cache_stale = (
                _md_cache_time is None
                or (now_et() - _md_cache_time).total_seconds() > _MD_CACHE_TTL
                or not _md_cache
            )

        # ES Futures
        try:
            # Always fetch fresh: short-term data
            self.es_1m = ib.get_historical_bars("ES", "2 D", "1 min")
            self.es_15m = ib.get_historical_bars("ES", "5 D", "15 mins")
            self.es_1h = ib.get_historical_bars("ES", "20 D", "1 hour")

            # Cached: heavy historical data (refresh every 5 min)
            if cache_stale:
                self.es_5m = ib.get_historical_bars("ES", "60 D", "5 mins")
                self.es_daily = ib.get_historical_bars("ES", "1 Y", "1 day")
                with _cache_lock:
                    _md_cache["es_5m"] = self.es_5m
                    _md_cache["es_daily"] = self.es_daily
                    _md_cache_time = now_et()
                logger.info(f"  IBKR ES (fresh + cache refreshed): 1m={len(self.es_1m)} 5m={len(self.es_5m)} "
                            f"15m={len(self.es_15m)} 1h={len(self.es_1h)} D={len(self.es_daily)}")
            else:
                # Read cache under lock to prevent race with cache.clear()
                with _cache_lock:
                    self.es_5m = _md_cache.get("es_5m", pd.DataFrame())
                    self.es_daily = _md_cache.get("es_daily", pd.DataFrame())
                logger.info(f"  IBKR ES (fresh + cached): 1m={len(self.es_1m)} 5m={len(self.es_5m)} "
                            f"15m={len(self.es_15m)} 1h={len(self.es_1h)} D={len(self.es_daily)}")
        except Exception as e:
            logger.warning(f"  IBKR ES failed, falling back to yfinance: {e}")
            # Clear any partial IBKR data to prevent mixing sources
            self.es_1m = self.es_5m = self.es_15m = self.es_1h = self.es_daily = pd.DataFrame()
            with _cache_lock:
                _md_cache.clear()
                _md_cache_time = None
            self._fetch_es_yfinance()

        # NQ Futures
        self.nq_15m = self._fetch_with_fallback(
            lambda: ib.get_historical_bars("NQ", "2 D", "15 mins"),
            lambda: yf.Ticker("NQ=F").history(period="2d", interval="15m"),
            "NQ"
        )

        # VIX
        self.vix = self._fetch_with_fallback(
            lambda: ib.get_historical_bars("VIX", "2 D", "15 mins"),
            lambda: yf.Ticker("^VIX").history(period="2d", interval="15m"),
            "VIX"
        )

        # VIX9D (for term structure: VIX vs VIX9D)
        self.vix9d = self._fetch_with_fallback(
            lambda: ib.get_historical_bars("VIX9D", "2 D", "15 mins")["Close"],
            lambda: yf.Ticker("^VIX9D").history(period="2d", interval="15m")["Close"],
            "VIX9D"
        )

        # TNX (for fractal engine)
        self.tnx = self._fetch_with_fallback(
            lambda: ib.get_historical_bars("TNX", "2 D", "15 mins")["Close"],
            lambda: yf.Ticker("^TNX").history(period="2d", interval="15m")["Close"],
            "TNX"
        )

        # Breadth (Mag7)
        self.breadth_data = self._fetch_with_fallback(
            lambda: ib.get_breadth_data(CFG.BREADTH_TICKERS.split()),
            lambda: yf.download(
                CFG.BREADTH_TICKERS, period="1d", interval="1m", progress=False
            )["Close"],
            "breadth"
        )

        # DXY (already qualified in ibkr_client) — normalize to Series like TNX
        self.dxy = self._fetch_with_fallback(
            lambda: ib.get_historical_bars("DXY", "2 D", "15 mins")["Close"],
            lambda: yf.Ticker("DX-Y.NYB").history(period="2d", interval="15m")["Close"],
            "DXY"
        )

        # NYSE TICK (real institutional flow — replaces 8-stock proxy)
        self.tick_nyse = self._fetch_with_fallback(
            lambda: ib.get_historical_bars("TICK-NYSE", "1 D", "1 min"),
            lambda: pd.DataFrame(),  # no yfinance equivalent
            "TICK-NYSE"
        )

        # RSP / SPY for real breadth (equal-weight vs cap-weight divergence)
        self.rsp = self._fetch_with_fallback(
            lambda: ib.get_historical_bars("RSP", "1 D", "5 mins"),
            lambda: yf.Ticker("RSP").history(period="1d", interval="5m"),
            "RSP"
        )
        self.spy = self._fetch_with_fallback(
            lambda: ib.get_historical_bars("SPY", "1 D", "5 mins"),
            lambda: yf.Ticker("SPY").history(period="1d", interval="5m"),
            "SPY"
        )

    def _fetch_es_yfinance(self):
        """Fallback for ES if IBKR fails."""
        try:
            es = yf.Ticker("ES=F")
            self.es_1m = es.history(period="2d", interval="1m")
            self.es_5m = es.history(period="59d", interval="5m")
            self.es_15m = es.history(period="5d", interval="15m")
            self.es_1h = es.history(period="5d", interval="60m")
            self.es_daily = es.history(period="60d", interval="1d")
            self.data_source = "yfinance (IBKR fallback)"
        except Exception as e:
            logger.error(f"yfinance ES fallback failed: {e}")
            self.es_1m = self.es_5m = self.es_15m = self.es_1h = self.es_daily = pd.DataFrame()

    def _fetch_yfinance(self):
        """Full yfinance path (no IBKR)."""
        try:
            es = yf.Ticker("ES=F")
            self.es_1m = es.history(period="2d", interval="1m")
            self.es_5m = es.history(period="59d", interval="5m")
            self.es_15m = es.history(period="5d", interval="15m")
            self.es_1h = es.history(period="5d", interval="60m")
            self.es_daily = es.history(period="60d", interval="1d")
        except Exception as e:
            logger.error(f"ES data fetch failed: {e}")
            self.es_1m = self.es_5m = self.es_15m = self.es_1h = self.es_daily = pd.DataFrame()

        try:
            self.nq_15m = yf.Ticker("NQ=F").history(period="2d", interval="15m")
        except Exception as e:
            logger.warning(f"yfinance NQ fetch failed: {e}")
            self.nq_15m = pd.DataFrame()

        try:
            self.vix = yf.Ticker("^VIX").history(period="2d", interval="15m")
        except Exception as e:
            logger.warning(f"yfinance VIX fetch failed: {e}")
            self.vix = pd.DataFrame()

        try:
            self.vix9d = yf.Ticker("^VIX9D").history(period="2d", interval="15m")["Close"]
        except Exception as e:
            logger.warning(f"yfinance VIX9D fetch failed: {e}")
            self.vix9d = pd.Series(dtype=float)

        try:
            self.tnx = yf.Ticker("^TNX").history(period="2d", interval="15m")["Close"]
        except Exception as e:
            logger.warning(f"yfinance TNX fetch failed: {e}")
            self.tnx = pd.Series(dtype=float)

        try:
            self.breadth_data = yf.download(
                CFG.BREADTH_TICKERS, period="1d", interval="1m", progress=False
            )["Close"]
        except Exception as e:
            logger.warning(f"yfinance breadth fetch failed: {e}")
            self.breadth_data = pd.DataFrame()

        try:
            self.dxy = yf.Ticker("DX-Y.NYB").history(period="2d", interval="15m")["Close"]
        except Exception as e:
            logger.warning(f"yfinance DXY fetch failed: {e}")
            self.dxy = pd.Series(dtype=float)

        # TICK-NYSE not available via yfinance
        self.tick_nyse = pd.DataFrame()

        try:
            self.rsp = yf.Ticker("RSP").history(period="1d", interval="5m")
        except Exception as e:
            logger.warning(f"yfinance RSP fetch failed: {e}")
            self.rsp = pd.DataFrame()

        try:
            self.spy = yf.Ticker("SPY").history(period="1d", interval="5m")
        except Exception as e:
            logger.warning(f"yfinance SPY fetch failed: {e}")
            self.spy = pd.DataFrame()

    @property
    def current_price(self) -> float:
        """Get current ES price -- IBKR live tick if available, else latest bar."""
        if self.ibkr and self.ibkr.connected:
            try:
                price = self.ibkr.get_live_price("ES")
                if price > 0:
                    return price
            except Exception:
                pass
        if not self.es_1m.empty:
            return float(self.es_1m["Close"].iloc[-1])
        return 0.0

    @property
    def es_today_1m(self) -> pd.DataFrame:
        """Returns only today's 1-minute data."""
        if self.es_1m.empty:
            return pd.DataFrame()
        try:
            df = self.es_1m.copy()
            if df.index.tz is not None:
                df.index = df.index.tz_convert("America/New_York")
            today = now_et().date()
            return df[df.index.date == today]
        except Exception:
            return self.es_1m.tail(390)  # fallback: last ~1 session


# =================================================================
# --- FED FUNDS FUTURES (Feature #9) ---
# =================================================================

_fedwatch_cache: Dict[str, Any] = {}
_fedwatch_cache_time: float = 0.0


def fetch_fedwatch_probs() -> dict:
    """
    Fetch Fed funds rate probabilities from CME FedWatch via yfinance
    Fed funds futures proxy.

    Uses ZQ (30-Day Fed Fund futures) to derive implied cut/hike probabilities.
    Caches for FEDWATCH_CACHE_TTL seconds (default 4 hours).

    Returns dict with: cut_prob, hike_prob, hold_prob, prev_cut_prob, prev_hike_prob
    """
    global _fedwatch_cache, _fedwatch_cache_time

    if not CFG.FEDWATCH_ENABLED:
        return {}

    # Check cache
    if _fedwatch_cache and (_time.time() - _fedwatch_cache_time) < CFG.FEDWATCH_CACHE_TTL:
        return _fedwatch_cache

    try:
        # Current Fed funds rate (effective) — approximate from SOFR proxy
        current_rate = 4.33  # Updated manually or from config; approximate

        # Fetch front-month Fed fund futures (ZQ)
        # ZQ is priced as 100 - implied rate
        zq = yf.Ticker("ZQ=F")
        hist = zq.history(period="10d", interval="1d")

        if hist.empty or len(hist) < 2:
            logger.warning("FedWatch: insufficient ZQ data")
            return {}

        # Current implied rate
        current_close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-5]) if len(hist) >= 5 else float(hist["Close"].iloc[0])

        implied_rate = 100 - current_close
        prev_implied = 100 - prev_close

        # Derive probabilities (simplified — each 25bp step)
        rate_diff = current_rate - implied_rate  # positive = market expects cut
        prev_diff = current_rate - prev_implied

        # Convert to probabilities (each 25bp movement = ~100% of one cut/hike)
        cut_prob = max(0, min(100, rate_diff / 0.25 * 100))
        hike_prob = max(0, min(100, -rate_diff / 0.25 * 100))
        hold_prob = max(0, 100 - cut_prob - hike_prob)

        prev_cut = max(0, min(100, prev_diff / 0.25 * 100))
        prev_hike = max(0, min(100, -prev_diff / 0.25 * 100))

        result = {
            "cut_prob": round(cut_prob, 1),
            "hike_prob": round(hike_prob, 1),
            "hold_prob": round(hold_prob, 1),
            "prev_cut_prob": round(prev_cut, 1),
            "prev_hike_prob": round(prev_hike, 1),
            "implied_rate": round(implied_rate, 4),
            "current_rate": current_rate,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        _fedwatch_cache = result
        _fedwatch_cache_time = _time.time()
        logger.info(f"[FEDWATCH] Cut={cut_prob:.0f}% Hike={hike_prob:.0f}% Hold={hold_prob:.0f}% "
                    f"(implied {implied_rate:.3f}%)")
        return result

    except Exception as e:
        logger.warning(f"FedWatch fetch failed: {e}")
        return _fedwatch_cache if _fedwatch_cache else {}
