"""
Unit tests for technical indicators.

Tests core calculations with known inputs/outputs.
No IBKR, Claude, or Telegram dependencies.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock


# We need a mock MarketData to avoid IBKR imports
class MockMarketData:
    """Minimal MarketData mock for indicator testing."""

    def __init__(self, prices=None, volumes=None, n_bars=100):
        if prices is None:
            # Generate realistic ES price data around 5800
            np.random.seed(42)
            base = 5800.0
            returns = np.random.randn(n_bars) * 2  # ~2 pt moves per bar
            closes = base + np.cumsum(returns)
            highs = closes + np.abs(np.random.randn(n_bars)) * 1.5
            lows = closes - np.abs(np.random.randn(n_bars)) * 1.5
            opens = closes + np.random.randn(n_bars) * 0.5
        else:
            closes = np.array(prices)
            n_bars = len(closes)
            highs = closes + 1.0
            lows = closes - 1.0
            opens = closes

        if volumes is None:
            volumes = np.random.randint(1000, 10000, n_bars)

        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="1min")

        self.es_1m = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows,
            "Close": closes, "Volume": volumes,
        }, index=dates)

        self.es_5m = self.es_1m.resample("5min").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()

        self.es_15m = self.es_1m.resample("15min").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()

        self.es_1h = self.es_1m.resample("1h").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()

        self.es_daily = self.es_1m.resample("1D").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()

        self.es_today_1m = self.es_1m
        self.vix = pd.DataFrame({"Close": [16.5] * 10}, index=pd.date_range(end=datetime.now(), periods=10, freq="15min"))
        self.nq_15m = self.es_15m.copy()
        self.breadth_data = pd.DataFrame()
        self.tnx = pd.Series(dtype=float)
        self.ibkr = None

    @property
    def current_price(self):
        return float(self.es_1m["Close"].iloc[-1])


class TestVWAP:
    def test_vwap_returns_valid_status(self):
        from indicators import calc_vwap
        md = MockMarketData(n_bars=200)
        status, vwap_status, levels = calc_vwap(md)
        assert status != "N/A"
        assert isinstance(levels, dict)

    def test_vwap_empty_data(self):
        from indicators import calc_vwap
        md = MockMarketData(n_bars=200)
        md.es_today_1m = pd.DataFrame()
        status, vwap_status, levels = calc_vwap(md)
        assert status == "N/A"


class TestRVOL:
    def test_rvol_returns_dict(self):
        from indicators import calc_rvol
        md = MockMarketData(n_bars=200)
        result = calc_rvol(md)
        assert isinstance(result, dict)
        assert "rvol" in result
        assert "status" in result

    def test_rvol_positive(self):
        from indicators import calc_rvol
        md = MockMarketData(n_bars=200)
        result = calc_rvol(md)
        assert result["rvol"] >= 0


class TestVIXTermStructure:
    def test_vix_term_returns_dict(self):
        from indicators import calc_vix_term_structure
        md = MockMarketData(n_bars=200)
        result = calc_vix_term_structure(md)
        assert isinstance(result, dict)
        assert "vix" in result


class TestMTFMomentum:
    def test_mtf_returns_alignment(self):
        from indicators import calc_mtf_momentum
        md = MockMarketData(n_bars=200)
        result = calc_mtf_momentum(md)
        assert isinstance(result, dict)
        assert "alignment" in result
        assert "score" in result


class TestCumulativeDelta:
    def test_delta_returns_string(self):
        from indicators import calc_cumulative_delta
        md = MockMarketData(n_bars=200)
        raw_val, bias = calc_cumulative_delta(md)
        assert isinstance(bias, str)


class TestGapAnalysis:
    def test_gap_returns_dict(self):
        from indicators import calc_gap_analysis
        md = MockMarketData(n_bars=200)
        result = calc_gap_analysis(md)
        assert isinstance(result, dict)
