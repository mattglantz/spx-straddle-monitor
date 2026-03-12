"""
Unit tests for session utilities.
"""

import pytest
from session_utils import position_suggestion


class TestPositionSuggestion:
    def test_below_threshold_zero(self):
        contracts, desc = position_suggestion(55, flat_threshold=60)
        assert contracts == 0
        assert "NO TRADE" in desc

    def test_at_threshold_one_contract(self):
        contracts, desc = position_suggestion(60, flat_threshold=60)
        assert contracts == 1

    def test_70_confidence(self):
        contracts, _ = position_suggestion(70, flat_threshold=60)
        assert contracts == 2

    def test_80_confidence(self):
        contracts, _ = position_suggestion(80, flat_threshold=60)
        assert contracts == 3

    def test_90_confidence(self):
        contracts, _ = position_suggestion(90, flat_threshold=60)
        assert contracts == 4

    def test_100_confidence(self):
        contracts, _ = position_suggestion(100, flat_threshold=60)
        assert contracts == 4
