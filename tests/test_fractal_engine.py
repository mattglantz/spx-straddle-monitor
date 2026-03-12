"""
Unit tests for fractal_engine.py pure functions.

Tests cover: dtw_distance, _safe_corr, _weighted_corr, _vix_bucket,
_gap_bucket, _rvol_bucket, _classify_open_type, _filter_bars_by_session.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import time as dtime

# Import pure functions from fractal_engine
from fractal_engine import (
    dtw_distance,
    _safe_corr,
    _weighted_corr,
    _vix_bucket,
    _gap_bucket,
    _rvol_bucket,
    _classify_open_type,
    _filter_bars_by_session,
)


# =================================================================
# --- dtw_distance ---
# =================================================================

class TestDtwDistance:
    def test_identical_series(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert dtw_distance(s, s) == 0.0

    def test_shifted_series(self):
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        dist = dtw_distance(s1, s2)
        assert dist > 0

    def test_symmetry(self):
        s1 = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        s2 = np.array([2.0, 4.0, 1.0, 6.0, 3.0])
        assert abs(dtw_distance(s1, s2) - dtw_distance(s2, s1)) < 1e-10

    def test_different_lengths(self):
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = dtw_distance(s1, s2)
        assert dist >= 0

    def test_constant_series(self):
        s1 = np.full(10, 5.0)
        s2 = np.full(10, 5.0)
        assert dtw_distance(s1, s2) == 0.0


# =================================================================
# --- _safe_corr ---
# =================================================================

class TestSafeCorr:
    def test_perfect_positive(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert abs(_safe_corr(a, b) - 1.0) < 1e-10

    def test_perfect_negative(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        assert abs(_safe_corr(a, b) - (-1.0)) < 1e-10

    def test_zero_correlation(self):
        # Orthogonal signals — correlation near zero
        a = np.array([1.0, -1.0, 1.0, -1.0])
        b = np.array([1.0, 1.0, -1.0, -1.0])
        assert abs(_safe_corr(a, b)) < 0.1

    def test_constant_array_returns_zero(self):
        a = np.array([5.0, 5.0, 5.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        assert _safe_corr(a, b) == 0.0

    def test_both_constant_returns_zero(self):
        a = np.full(5, 3.0)
        b = np.full(5, 7.0)
        assert _safe_corr(a, b) == 0.0

    def test_too_short_returns_zero(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert _safe_corr(a, b) == 0.0


# =================================================================
# --- _weighted_corr ---
# =================================================================

class TestWeightedCorr:
    def test_uniform_weights_match_unweighted(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        w = np.ones(5)
        wc = _weighted_corr(a, b, w)
        uc = _safe_corr(a, b)
        assert abs(wc - uc) < 0.01

    def test_constant_returns_zero(self):
        a = np.full(5, 3.0)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.ones(5)
        assert _weighted_corr(a, b, w) == 0.0

    def test_too_short_returns_zero(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        w = np.array([1.0, 1.0])
        assert _weighted_corr(a, b, w) == 0.0


# =================================================================
# --- _vix_bucket ---
# =================================================================

class TestVixBucket:
    def test_zero_returns_negative(self):
        assert _vix_bucket(0) == -1

    def test_negative_returns_negative(self):
        assert _vix_bucket(-5) == -1

    def test_low_vix(self):
        assert _vix_bucket(12) == 0

    def test_normal_vix(self):
        assert _vix_bucket(17) == 1

    def test_elevated_vix(self):
        assert _vix_bucket(22) == 2

    def test_high_vix(self):
        assert _vix_bucket(28) == 3

    def test_crisis_vix(self):
        assert _vix_bucket(35) == 4

    def test_boundary_15(self):
        assert _vix_bucket(14.99) == 0
        assert _vix_bucket(15.0) == 1

    def test_boundary_25(self):
        assert _vix_bucket(24.99) == 2
        assert _vix_bucket(25.0) == 3


# =================================================================
# --- _gap_bucket ---
# =================================================================

class TestGapBucket:
    def test_large_gap_down(self):
        assert _gap_bucket(-0.5) == -2

    def test_small_gap_down(self):
        assert _gap_bucket(-0.1) == -1

    def test_flat_open(self):
        assert _gap_bucket(0.0) == 0

    def test_small_gap_up(self):
        assert _gap_bucket(0.1) == 1

    def test_large_gap_up(self):
        assert _gap_bucket(0.5) == 2

    def test_boundary_negative(self):
        assert _gap_bucket(-0.05) == 0
        assert _gap_bucket(-0.0501) == -1


# =================================================================
# --- _rvol_bucket ---
# =================================================================

class TestRvolBucket:
    def test_zero_volume(self):
        assert _rvol_bucket(0, 1000000) == "VERY_LOW"

    def test_zero_avg_volume(self):
        assert _rvol_bucket(500000, 0) == "NORMAL"

    def test_very_low(self):
        assert _rvol_bucket(400000, 1000000) == "VERY_LOW"

    def test_low(self):
        assert _rvol_bucket(700000, 1000000) == "LOW"

    def test_normal(self):
        assert _rvol_bucket(1000000, 1000000) == "NORMAL"

    def test_high(self):
        assert _rvol_bucket(1700000, 1000000) == "HIGH"

    def test_very_high(self):
        assert _rvol_bucket(2500000, 1000000) == "VERY_HIGH"


# =================================================================
# --- _classify_open_type ---
# =================================================================

class TestClassifyOpenType:
    def test_none_returns_auction(self):
        assert _classify_open_type(None) == "AUCTION"

    def test_empty_returns_auction(self):
        assert _classify_open_type("") == "AUCTION"

    def test_drive_up(self):
        assert _classify_open_type("Drive Up") == "DRIVE_UP"

    def test_drive_buy(self):
        assert _classify_open_type("Drive Buy") == "DRIVE_UP"

    def test_drive_bullish(self):
        assert _classify_open_type("Bullish Drive") == "DRIVE_UP"

    def test_drive_down(self):
        assert _classify_open_type("Drive Down") == "DRIVE_DOWN"

    def test_drive_sell(self):
        assert _classify_open_type("Drive Sell") == "DRIVE_DOWN"

    def test_drive_bearish(self):
        assert _classify_open_type("Bearish Drive") == "DRIVE_DOWN"

    def test_generic_drive(self):
        assert _classify_open_type("Drive") == "DRIVE_UP"

    def test_auction(self):
        assert _classify_open_type("Auction Rotation") == "AUCTION"

    def test_case_insensitive(self):
        assert _classify_open_type("DRIVE UP") == "DRIVE_UP"
        assert _classify_open_type("drive down") == "DRIVE_DOWN"


# =================================================================
# --- _filter_bars_by_session ---
# =================================================================

class TestFilterBarsBySession:
    def _make_df(self, hours):
        """Create a DataFrame with DatetimeIndex at specified hours."""
        import pandas as pd
        dates = pd.date_range("2025-01-02", periods=len(hours), freq="h")
        # Override hours
        new_dates = []
        for i, h in enumerate(hours):
            new_dates.append(dates[0].replace(hour=h, minute=0))
        idx = pd.DatetimeIndex(new_dates)
        return pd.DataFrame({"Close": range(len(hours)), "Volume": 100}, index=idx)

    def test_none_session_returns_original(self):
        df = self._make_df([9, 10, 11, 12])
        result = _filter_bars_by_session(df, None)
        assert len(result) == len(df)

    def test_day_session_filters(self):
        # DAY = 3AM-4PM
        df = self._make_df([2, 3, 9, 15, 16, 20])
        result = _filter_bars_by_session(df, "DAY")
        # 3AM (included), 9AM, 3PM are in range; 2AM (excluded), 4PM (excluded), 8PM (excluded)
        assert len(result) == 3

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame()
        result = _filter_bars_by_session(df, "DAY")
        assert result.empty

    def test_unknown_session_returns_original(self):
        df = self._make_df([9, 10])
        result = _filter_bars_by_session(df, "UNKNOWN")
        assert len(result) == len(df)
