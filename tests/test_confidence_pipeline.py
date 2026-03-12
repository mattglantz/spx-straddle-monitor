"""
Integration tests for the confidence pipeline.

Tests the full flow: AccuracyTracker → calc_regime_adjustment →
calc_signal_confluence → apply_confidence_pipeline.

Uses an in-memory SQLite database with the real trades table schema.
No IBKR, Claude, Telegram, or win32 dependencies.
"""

import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo

import pytest

import trade_status as ts
from confidence_engine import (
    AccuracyTracker,
    apply_confidence_pipeline,
    calc_regime_adjustment,
    calc_signal_confluence,
)

ET = ZoneInfo("America/New_York")


# ─── Helpers ───────────────────────────────────────────────────────

class MockJournal:
    """In-memory SQLite journal matching the real trades table schema."""

    def __init__(self):
        self._db = sqlite3.connect(":memory:")
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(f"""
            CREATE TABLE trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                price       REAL    NOT NULL,
                verdict     TEXT    NOT NULL,
                confidence  INTEGER NOT NULL DEFAULT 0,
                target      REAL,
                stop        REAL,
                status      TEXT    NOT NULL DEFAULT '{ts.OPEN}',
                pnl         REAL    NOT NULL DEFAULT 0.0,
                contracts   INTEGER NOT NULL DEFAULT 1,
                reasoning   TEXT,
                session     TEXT,
                fractal_recorded INTEGER DEFAULT NULL,
                oca_group   TEXT    DEFAULT NULL
            )
        """)
        self._db.commit()

    def _conn(self):
        """Return the same in-memory connection (`:memory:` is per-connection)."""
        return self._db

    def seed_trades(self, trades: list):
        """Bulk insert test trades.

        Each dict should have: timestamp, verdict, status, pnl.
        Optional: confidence, session, price, target, stop.
        """
        for t in trades:
            self._db.execute(
                "INSERT INTO trades "
                "(timestamp, price, verdict, confidence, target, stop, status, pnl, session) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    t["timestamp"],
                    t.get("price", 5800.0),
                    t["verdict"],
                    t.get("confidence", 70),
                    t.get("target", 5815.0),
                    t.get("stop", 5794.0),
                    t["status"],
                    t["pnl"],
                    t.get("session", "ACTIVE"),
                ),
            )
        self._db.commit()


def make_now_fn(dt: datetime):
    """Return a callable that returns the given datetime (timezone-aware ET)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    return lambda: dt


def _make_trade(status, pnl, minutes_ago=0, base_time=None, verdict="BULLISH"):
    """Helper to create a single trade dict for seeding."""
    if base_time is None:
        base_time = datetime(2026, 3, 4, 10, 0)
    ts_val = base_time - timedelta(minutes=minutes_ago)
    return {
        "timestamp": ts_val.strftime("%Y-%m-%d %H:%M"),
        "verdict": verdict,
        "status": status,
        "pnl": pnl,
    }


def mock_position_fn(confidence, flat_threshold=60):
    """Pure-math position suggestion matching the bot's formula.

    Uses defaults: ACCOUNT_SIZE=50000, MAX_RISK_PCT=2, ES_CONTRACT_RISK=250.
    """
    if confidence < flat_threshold:
        return 0, f"NO TRADE (Conf {confidence}% < {flat_threshold}%)"
    scale = (confidence - flat_threshold) / (100 - flat_threshold)
    risk = 50_000 * 0.02 * (0.5 + 0.5 * scale)
    contracts = int(risk / 250)
    return contracts, f"{contracts} ct (Risk ${risk:.0f} at {confidence}% conf)"


# ─── Fake projection for confluence tests ──────────────────────────

@dataclass
class FakeProjection:
    direction: str = "BULLISH"
    confidence: int = 75


# ─── Test 1: Empty history ─────────────────────────────────────────

class TestEmptyHistory:
    def test_no_penalty_on_empty_history(self):
        journal = MockJournal()
        now = datetime(2026, 3, 4, 10, 0, tzinfo=ET)
        tracker = AccuracyTracker(journal, now_fn=make_now_fn(now))

        acc = tracker.get_recent_accuracy()
        assert acc["total"] == 0
        assert acc["confidence_adjustment"] == 0
        assert acc["win_rate"] == 50

    def test_adjust_confidence_unchanged(self):
        journal = MockJournal()
        now = datetime(2026, 3, 4, 10, 0, tzinfo=ET)
        tracker = AccuracyTracker(journal, now_fn=make_now_fn(now))

        assert tracker.adjust_confidence(75) == 75
        assert tracker.adjust_confidence(50) == 50


# ─── Tests 2-5: Streak penalties and decay ─────────────────────────

class TestStreakPenalties:
    """Tests that losing streaks apply correct penalties and decay with staleness."""

    BASE_TIME = datetime(2026, 3, 4, 10, 0)

    def _seed_streak(self, journal, n_losses, n_wins_before, base_time=None):
        """Seed n_wins_before older wins followed by n_losses recent losses.

        Inserted in chronological order so that higher IDs = more recent trades,
        matching the ``ORDER BY id DESC`` query in AccuracyTracker.
        """
        if base_time is None:
            base_time = self.BASE_TIME
        trades = []
        total = n_wins_before + n_losses
        # Older wins first (lower IDs)
        for i in range(n_wins_before):
            offset = (total - i) * 30  # furthest back
            trades.append(_make_trade(ts.WIN, 12.0, minutes_ago=offset, base_time=base_time))
        # Recent losses last (higher IDs)
        for i in range(n_losses):
            offset = (n_losses - 1 - i) * 30  # most recent = 0 minutes ago
            trades.append(_make_trade(ts.LOSS, -8.0, minutes_ago=offset, base_time=base_time))
        journal.seed_trades(trades)

    def test_3_loss_streak_penalty_minus_5(self):
        """3 consecutive losses with 10+ total trades → -5 penalty."""
        journal = MockJournal()
        self._seed_streak(journal, n_losses=3, n_wins_before=8)

        now = make_now_fn(datetime(2026, 3, 4, 10, 5, tzinfo=ET))
        tracker = AccuracyTracker(journal, now_fn=now)
        acc = tracker.get_recent_accuracy()

        assert acc["confidence_adjustment"] == -5
        assert acc["streak"] == 3
        assert acc["streak_type"] == "LOSING"

    def test_4_loss_streak_penalty_minus_7(self):
        """4 consecutive losses with 10+ total trades → -7 penalty."""
        journal = MockJournal()
        self._seed_streak(journal, n_losses=4, n_wins_before=7)

        now = make_now_fn(datetime(2026, 3, 4, 10, 5, tzinfo=ET))
        tracker = AccuracyTracker(journal, now_fn=now)
        acc = tracker.get_recent_accuracy()

        assert acc["confidence_adjustment"] == -7
        assert acc["streak"] == 4

    def test_streak_decayed_to_zero_after_3_hours(self):
        """4-loss streak, but 3+ hours stale → penalty decayed to 0 (v26.2: lowered from 4h)."""
        journal = MockJournal()
        # Trades happened at 10:00
        self._seed_streak(journal, n_losses=4, n_wins_before=7)

        # Now it's 13:05 — 3+ hours later → full reset
        now = make_now_fn(datetime(2026, 3, 4, 13, 5, tzinfo=ET))
        tracker = AccuracyTracker(journal, now_fn=now)
        acc = tracker.get_recent_accuracy()

        assert acc["confidence_adjustment"] == 0
        assert acc["hours_since_last_trade"] >= 3.0

    def test_streak_halved_after_90_minutes(self):
        """4-loss streak, 2 hours stale → penalty halved: int(-7/2) = -3 (v26.2: lowered from 2h)."""
        journal = MockJournal()
        self._seed_streak(journal, n_losses=4, n_wins_before=7)

        # 2 hours after last trade (between 1.5h and 3h → halved)
        now = make_now_fn(datetime(2026, 3, 4, 12, 5, tzinfo=ET))
        tracker = AccuracyTracker(journal, now_fn=now)
        acc = tracker.get_recent_accuracy()

        assert acc["confidence_adjustment"] == -3  # int(-7 / 2)
        assert 1.5 <= acc["hours_since_last_trade"] <= 3.0


# ─── Test 6: Regime adjustment ─────────────────────────────────────

class TestRegimeAdjustment:
    def test_high_vix_no_double_squeeze(self):
        """VIX=22 → threshold=57, but confidence_mod should NOT penalty VIX again."""
        regime = calc_regime_adjustment(
            vix_term={"vix": 22, "structure": "CONTANGO"},
            rvol={"rvol": 1.0},
            day_type={"trend_probability": 50},
            mtf={"score": 0},
        )
        assert regime["flat_threshold"] == 62
        assert regime["regime"] == "ELEVATED VOLATILITY"
        # VIX between 20-27 should NOT add a confidence penalty
        # (the threshold already handles it — no double squeeze)
        assert regime["confidence_mod"] == 0

    def test_low_vix_reward(self):
        """VIX=12 → threshold=55, confidence_mod=+3."""
        regime = calc_regime_adjustment(
            vix_term={"vix": 12, "structure": "CONTANGO"},
            rvol={"rvol": 1.0},
            day_type={"trend_probability": 50},
            mtf={"score": 0},
        )
        assert regime["flat_threshold"] == 55
        assert regime["confidence_mod"] == 3

    def test_penalty_capped_at_minus_15(self):
        """Multiple negative factors shouldn't exceed -15 total."""
        regime = calc_regime_adjustment(
            vix_term={"vix": 30, "structure": "STEEP BACKWARDATION"},
            rvol={"rvol": 0.2},  # -5
            day_type={"trend_probability": 50},
            mtf={"score": 0},
        )
        # backwardation: -3, low rvol: -5, = -8
        # But if more factors pile up, cap at -15
        assert regime["confidence_mod"] >= -15


# ─── Test 7: Signal confluence ─────────────────────────────────────

class TestSignalConfluence:
    def test_3_bullish_signals_gives_floor(self):
        """3+ agreeing bullish signals → conviction_floor >= 63."""
        metrics = {
            "fractal": {"projection": FakeProjection("BULLISH", 75)},
            "flow_data": {"flow_bias": "BULLISH"},
            "liq_sweeps": {"active_sweep": "BEAR TRAP"},  # bullish signal
            "mtf_momentum": {"alignment": "FULL BULLISH"},
            "tick_proxy": {"extreme": ""},
            "cum_delta_bias": "",
            "opening_type": {"bias": "NEUTRAL"},
        }
        result = calc_signal_confluence(metrics)

        assert result["direction"] == "BULLISH"
        assert result["conviction_floor"] >= 63
        assert result["confluence_level"] in ("WEAK", "MODERATE", "STRONG")
        assert len(result["confirming"]) >= 3

    def test_no_signals_no_floor(self):
        """No directional signals → conviction_floor == 0."""
        metrics = {
            "fractal": {},
            "flow_data": {},
            "liq_sweeps": {"active_sweep": "NONE"},
            "mtf_momentum": {"alignment": "MIXED / CONFLICTED"},
            "tick_proxy": {"extreme": ""},
            "cum_delta_bias": "",
            "opening_type": {"bias": "NEUTRAL"},
        }
        result = calc_signal_confluence(metrics)
        assert result["conviction_floor"] == 0
        assert result["confluence_level"] == "NONE"

    def test_confluence_protects_from_flat(self):
        """With 3+ bullish signals, a BULLISH verdict below threshold survives."""
        # Setup: raw confidence 55, threshold 60, but 3+ bullish signals
        journal = MockJournal()
        now = make_now_fn(datetime(2026, 3, 4, 10, 0, tzinfo=ET))
        tracker = AccuracyTracker(journal, now_fn=now)

        metrics = {
            "fractal": {"projection": FakeProjection("BULLISH", 75)},
            "flow_data": {"flow_bias": "BULLISH"},
            "liq_sweeps": {"active_sweep": "BEAR TRAP"},
            "mtf_momentum": {"alignment": "FULL BULLISH"},
            "tick_proxy": {"extreme": ""},
            "cum_delta_bias": "",
            "opening_type": {"bias": "NEUTRAL"},
        }
        regime = {"confidence_mod": 0, "flat_threshold": 60}
        data = {"confidence": 55, "verdict": "BULLISH"}

        data, verdict, conf, contracts, pos_str, confluence, _decomp = apply_confidence_pipeline(
            data, metrics, tracker, regime, md=None,
            position_suggestion_fn=mock_position_fn,
        )

        # Verdict should NOT be forced FLAT — confluence floor protects it
        assert verdict == "BULLISH"
        assert conf >= 60  # at least threshold (lifted by floor or threshold lift)


# ─── Test 8: Confluence floor + high VIX threshold ─────────────────

class TestConfluenceThresholdLift:
    def test_floor_lifted_to_threshold_in_high_vix(self):
        """When confluence floor < flat_threshold, lift to threshold so trade isn't FLATted."""
        journal = MockJournal()
        now = make_now_fn(datetime(2026, 3, 4, 10, 0, tzinfo=ET))
        tracker = AccuracyTracker(journal, now_fn=now)

        # 3 signals → floor=63, but VIX>27 → threshold=62
        # Actually let's make threshold=65 to be above 63 floor
        metrics = {
            "fractal": {"projection": FakeProjection("BULLISH", 70)},
            "flow_data": {"flow_bias": "BULLISH"},
            "liq_sweeps": {"active_sweep": "BEAR TRAP"},
            "mtf_momentum": {"alignment": "MOSTLY BULLISH"},
            "tick_proxy": {"extreme": ""},
            "cum_delta_bias": "",
            "opening_type": {"bias": "NEUTRAL"},
        }
        # Simulate scenario: floor=63 from 3 signals, but threshold=65
        regime = {"confidence_mod": -10, "flat_threshold": 65}
        data = {"confidence": 60, "verdict": "BULLISH"}

        data, verdict, conf, contracts, pos_str, confluence, _decomp = apply_confidence_pipeline(
            data, metrics, tracker, regime, md=None,
            position_suggestion_fn=mock_position_fn,
        )

        # Conf after accuracy (no penalty, empty journal) = 60
        # After regime -10 = 50
        # Confluence floor should kick in (floor >= 63 for 3+ signals)
        # Then threshold lift: 63 < 65 → lift to 65
        assert verdict == "BULLISH"
        assert conf >= 65  # Lifted to at least threshold


# ─── Test 9: Full end-to-end pipeline ──────────────────────────────

class TestFullPipeline:
    def test_end_to_end_bullish_trade(self):
        """Full pipeline: trade history + regime + signals → final verdict + contracts."""
        journal = MockJournal()
        base = datetime(2026, 3, 4, 9, 30)
        # 15 trades: 10 wins, 5 losses (67% win rate → +5 bonus)
        trades = []
        for i in range(10):
            trades.append(_make_trade(ts.WIN, 12.0, minutes_ago=(i + 6) * 30, base_time=base))
        for i in range(5):
            trades.append(_make_trade(ts.LOSS, -8.0, minutes_ago=(i + 16) * 30, base_time=base))
        journal.seed_trades(trades)

        now = make_now_fn(datetime(2026, 3, 4, 9, 35, tzinfo=ET))
        tracker = AccuracyTracker(journal, now_fn=now)

        # Regime: normal VIX
        regime = calc_regime_adjustment(
            vix_term={"vix": 16, "structure": "CONTANGO"},
            rvol={"rvol": 1.2},
            day_type={"trend_probability": 55},
            mtf={"score": 60},
        )

        # Metrics with moderate confluence
        metrics = {
            "fractal": {"projection": FakeProjection("BULLISH", 72)},
            "flow_data": {"flow_bias": "BULLISH"},
            "liq_sweeps": {"active_sweep": "NONE"},
            "mtf_momentum": {"alignment": "MOSTLY BULLISH"},
            "tick_proxy": {"extreme": ""},
            "cum_delta_bias": "",
            "opening_type": {"bias": "NEUTRAL"},
        }

        # Fake Claude response
        data = {
            "confidence": 72,
            "verdict": "BULLISH",
            "target": 5830.0,
            "invalidation": 5790.0,
            "setup": "Strong bullish fractal with MTF alignment",
        }

        data, verdict, conf, contracts, pos_str, confluence, _decomp = apply_confidence_pipeline(
            data, metrics, tracker, regime, md=None,
            position_suggestion_fn=mock_position_fn,
        )

        # With 67% win rate + 10 trades, should get +5 accuracy bonus
        # Regime: VIX 16 (normal), RVOL 1.2 (normal), MTF 60 → +3
        # Raw 72 + accuracy +5 + regime +3 = 80
        assert verdict == "BULLISH"
        assert conf > 72  # Should have positive adjustments
        assert contracts > 0  # Should suggest some position
        assert "ct" in pos_str

    def test_end_to_end_forced_flat(self):
        """Low confidence + no confluence → forced FLAT."""
        journal = MockJournal()
        now = make_now_fn(datetime(2026, 3, 4, 10, 0, tzinfo=ET))
        tracker = AccuracyTracker(journal, now_fn=now)

        regime = {"confidence_mod": -5, "flat_threshold": 60}
        metrics = {
            "fractal": {},
            "flow_data": {},
            "liq_sweeps": {"active_sweep": "NONE"},
            "mtf_momentum": {"alignment": "MIXED / CONFLICTED"},
            "tick_proxy": {"extreme": ""},
            "cum_delta_bias": "",
            "opening_type": {"bias": "NEUTRAL"},
        }
        data = {"confidence": 55, "verdict": "BEARISH"}

        data, verdict, conf, contracts, pos_str, confluence, _decomp = apply_confidence_pipeline(
            data, metrics, tracker, regime, md=None,
            position_suggestion_fn=mock_position_fn,
        )

        # 55 - 5 regime = 50, below threshold 60, no confluence → FLAT
        assert verdict == "FLAT"
        assert contracts == 0
        assert conf < 60

    def test_position_sizing_scales_correctly(self):
        """Contracts scale from 50% at threshold to 100% at confidence=100."""
        # At threshold exactly → 50% of max risk
        contracts_at_threshold, _ = mock_position_fn(60, flat_threshold=60)

        # At 100% confidence → 100% of max risk
        contracts_at_max, _ = mock_position_fn(100, flat_threshold=60)

        # At midpoint → 75% of max risk
        contracts_at_mid, _ = mock_position_fn(80, flat_threshold=60)

        assert contracts_at_threshold > 0
        assert contracts_at_max > contracts_at_threshold
        assert contracts_at_mid > contracts_at_threshold
        assert contracts_at_max >= contracts_at_mid

    def test_below_threshold_zero_contracts(self):
        """Below threshold → 0 contracts."""
        contracts, pos_str = mock_position_fn(55, flat_threshold=60)
        assert contracts == 0
        assert "NO TRADE" in pos_str
