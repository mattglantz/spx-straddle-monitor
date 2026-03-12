"""
Unit tests for the trading journal.
Uses in-memory SQLite to avoid file I/O.
"""

import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch

import trade_status as ts


class TestJournal:
    def _make_journal(self):
        """Create a journal with in-memory DB."""
        from journal import Journal
        j = Journal(db_path=Path(":memory:"))
        # Override _conn to use same in-memory connection
        j._mem_db = sqlite3.connect(":memory:")
        j._mem_db.row_factory = sqlite3.Row
        j._mem_db.execute("PRAGMA journal_mode=WAL")
        j._mem_db.execute(f"""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                verdict TEXT NOT NULL,
                confidence INTEGER NOT NULL DEFAULT 0,
                target REAL, stop REAL,
                status TEXT NOT NULL DEFAULT '{ts.OPEN}',
                pnl REAL NOT NULL DEFAULT 0.0,
                contracts INTEGER NOT NULL DEFAULT 1,
                reasoning TEXT, session TEXT,
                fractal_recorded INTEGER DEFAULT NULL,
                oca_group TEXT DEFAULT NULL,
                signals TEXT DEFAULT NULL,
                partial_closed INTEGER DEFAULT 0,
                stop_updated_at TEXT DEFAULT NULL
            )
        """)
        j._mem_db.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0
            )
        """)
        j._mem_db.execute("""
            CREATE TABLE IF NOT EXISTS trade_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id    INTEGER NOT NULL,
                event_type  TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                data        TEXT NOT NULL
            )
        """)
        j._mem_db.commit()

        from contextlib import contextmanager
        @contextmanager
        def mem_conn():
            yield j._mem_db
        j._conn = mem_conn
        return j

    def test_add_trade(self):
        j = self._make_journal()
        tid = j.add_trade(5800.0, "BULLISH", 75, 5815.0, 5794.0, contracts=2)
        assert tid > 0

    def test_get_open_trades(self):
        j = self._make_journal()
        j.add_trade(5800.0, "BULLISH", 75, 5815.0, 5794.0)
        open_trades = j.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["verdict"] == "BULLISH"

    def test_update_trade(self):
        j = self._make_journal()
        tid = j.add_trade(5800.0, "BULLISH", 75, 5815.0, 5794.0)
        j.update_trade(tid, ts.WIN, 15.0)
        open_trades = j.get_open_trades()
        assert len(open_trades) == 0  # No longer open

    def test_add_skipped_trade(self):
        j = self._make_journal()
        tid = j.add_skipped_trade(5800.0, "BEARISH", 65, 5785.0, 5810.0, reason="R:R < 1.2")
        assert tid > 0

    def test_get_today_stats_empty(self):
        j = self._make_journal()
        stats = j.get_today_stats()
        assert stats["total"] == 0
        assert stats["wins"] == 0

    def test_signal_performance_empty(self):
        j = self._make_journal()
        result = j.get_signal_performance()
        assert result == []

    def test_trade_with_signals(self):
        j = self._make_journal()
        signals = {"fractal_dir": "BULLISH", "mtf": "ALIGNED", "flow": "BULLISH"}
        tid = j.add_trade(5800.0, "BULLISH", 75, 5815.0, 5794.0, signals=signals)
        assert tid > 0


class TestTradeStatus:
    def test_is_long(self):
        assert ts.is_long("BULLISH") == True
        assert ts.is_long("Bullish") == True
        assert ts.is_long("BEARISH") == False
        assert ts.is_long("LONG") == True
        assert ts.is_long("BUY") == True

    def test_is_short(self):
        assert ts.is_short("BEARISH") == True
        assert ts.is_short("SHORT") == True
        assert ts.is_short("SELL") == True
        assert ts.is_short("BULLISH") == False

    def test_status_sets(self):
        assert ts.WIN in ts.CLOSED_STATUSES
        assert ts.LOSS in ts.CLOSED_STATUSES
        assert ts.OPEN not in ts.CLOSED_STATUSES
        assert ts.SKIPPED not in ts.CLOSED_STATUSES
