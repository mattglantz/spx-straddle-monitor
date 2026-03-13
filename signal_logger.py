"""
Signal Decomposition Logger — structured per-cycle signal snapshot.

Logs every signal component and pipeline transformation for post-hoc analysis.
One row per analysis cycle in cycle_signals table (trading_journal.db).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from bot_config import CFG, logger, now_et


class SignalLogger:
    """Logs all signal values and pipeline stages per analysis cycle."""

    def __init__(self, db_path=None):
        self.db_path = db_path or CFG.DB_FILE
        self._init_table()

    def _init_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cycle_signals (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id            TEXT NOT NULL,
                    timestamp           TEXT NOT NULL,
                    price               REAL NOT NULL,
                    -- Fractal
                    fractal_dir         TEXT,
                    fractal_conf        INTEGER,
                    fractal_matches     INTEGER,
                    -- MTF
                    mtf_score           INTEGER,
                    mtf_alignment       TEXT,
                    -- TICK
                    tick_proxy          REAL,
                    tick_signal         TEXT,
                    -- Delta
                    cum_delta_bias      TEXT,
                    -- Volume
                    rvol                REAL,
                    rvol_status         TEXT,
                    -- Volatility
                    vix                 REAL,
                    vix_structure       TEXT,
                    -- GEX / Flow
                    gex_regime          TEXT,
                    flow_bias           TEXT,
                    -- Opening/Day type
                    opening_type        TEXT,
                    day_type            TEXT,
                    gap_pct             REAL,
                    -- Correlation / Skew
                    cross_corr_regime   TEXT,
                    iv_skew_signal      TEXT,
                    vol_shift           TEXT,
                    -- Divergence
                    divergence_score    INTEGER,
                    divergence_severity TEXT,
                    -- Pipeline stages
                    raw_verdict         TEXT,
                    raw_confidence      INTEGER,
                    accuracy_adj        INTEGER,
                    regime_adj          INTEGER,
                    confluence_level    TEXT,
                    floor_applied       INTEGER DEFAULT 0,
                    forced_flat         INTEGER DEFAULT 0,
                    mtf_conflict        INTEGER DEFAULT 0,
                    final_verdict       TEXT,
                    final_confidence    INTEGER,
                    contracts           INTEGER,
                    -- Trade linkage
                    trade_taken         INTEGER DEFAULT 0,
                    trade_id            INTEGER DEFAULT NULL,
                    -- Restructuring metrics (v2.3)
                    confluence_override   INTEGER DEFAULT 0,
                    confluence_bull_score  REAL,
                    confluence_bear_score  REAL,
                    confluence_signals     TEXT,
                    fractal_in_confluence  INTEGER DEFAULT 0,
                    fractal_agrees_claude  INTEGER DEFAULT NULL,
                    correlated_stack_count INTEGER DEFAULT 0,
                    correlated_stack_dir   TEXT,
                    -- v29 additions
                    vix9d_ratio            REAL,
                    vol_scale_ratio        REAL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cycle_signals_ts "
                "ON cycle_signals(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cycle_signals_cycle "
                "ON cycle_signals(cycle_id)"
            )
            # Add new columns to existing table if needed
            for col, coltype in [
                ("confluence_override", "INTEGER DEFAULT 0"),
                ("confluence_bull_score", "REAL"),
                ("confluence_bear_score", "REAL"),
                ("confluence_signals", "TEXT"),
                ("fractal_in_confluence", "INTEGER DEFAULT 0"),
                ("fractal_agrees_claude", "INTEGER DEFAULT NULL"),
                ("correlated_stack_count", "INTEGER DEFAULT 0"),
                ("vix9d_ratio", "REAL"),
                ("vol_scale_ratio", "REAL"),
                ("correlated_stack_dir", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE cycle_signals ADD COLUMN {col} {coltype}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            conn.commit()

    def log_cycle(
        self,
        cycle_id: str,
        price: float,
        metrics: dict,
        raw_verdict: str,
        raw_confidence: int,
        accuracy_adj: int,
        regime_adj: int,
        confluence: dict,
        floor_applied: bool,
        forced_flat: bool,
        mtf_conflict: bool,
        final_verdict: str,
        final_confidence: int,
        contracts: int,
        trade_taken: bool = False,
        trade_id: int = None,
        confluence_override: bool = False,
    ):
        """Extract all signal values from metrics dict and log one row."""
        ts = now_et().strftime("%Y-%m-%d %H:%M:%S")

        # Extract signal values safely
        fractal = metrics.get("fractal", {})
        proj = fractal.get("projection", None)
        mtf = metrics.get("mtf_momentum", {})
        tick = metrics.get("tick_proxy", {})
        rvol_d = metrics.get("rvol", {})
        vix_d = metrics.get("vix_term", {})
        gex_d = metrics.get("gex_regime", {})
        flow_d = metrics.get("flow_data", {})
        open_d = metrics.get("opening_type", {})
        day_d = metrics.get("day_type", {})
        gap_d = metrics.get("gap", {})
        corr_d = metrics.get("cross_corr", {})
        skew_d = metrics.get("iv_skew", {})
        vol_d = metrics.get("vol_shift", {})
        div_d = metrics.get("divergence", {})
        cum_delta_bias = metrics.get("cum_delta_bias", "")

        # --- Compute restructuring metrics ---
        # 1. Did fractal vote in confluence? (conf >= 65)
        fractal_dir = getattr(proj, "direction", "") if proj else ""
        fractal_conf_val = getattr(proj, "confidence", 0) if proj else 0
        fractal_in_confl = int(fractal_conf_val >= 65 and
                               any(k in fractal_dir for k in ("BULL", "BEAR")))

        # 2. Does fractal agree with Claude's raw verdict?
        fractal_agrees = None  # None = no fractal or no directional verdict
        if fractal_in_confl and raw_verdict not in ("FLAT", "NEUTRAL"):
            raw_bull = any(k in raw_verdict.upper() for k in ("BULL", "LONG", "BUY"))
            raw_bear = any(k in raw_verdict.upper() for k in ("BEAR", "SHORT", "SELL"))
            frac_bull = "BULL" in fractal_dir
            frac_bear = "BEAR" in fractal_dir
            fractal_agrees = int((raw_bull and frac_bull) or (raw_bear and frac_bear))

        # 3. Correlated signal stacking: MTF + cum_delta + TICK + delta_levels
        #    These all derive from price/volume data and aren't independent.
        _corr_signals = []
        _mtf_al = mtf.get("alignment", "")
        if "BULL" in _mtf_al:
            _corr_signals.append("BULL")
        elif "BEAR" in _mtf_al:
            _corr_signals.append("BEAR")
        _cd_bias = cum_delta_bias if isinstance(cum_delta_bias, str) else str(cum_delta_bias)
        if "BUYER" in _cd_bias and "RISING" in _cd_bias:
            _corr_signals.append("BULL")
        elif "SELLER" in _cd_bias and "FALLING" in _cd_bias:
            _corr_signals.append("BEAR")
        _tick_sig = tick.get("signal", "") or tick.get("extreme", "")
        if "BULL" in str(_tick_sig).upper():
            _corr_signals.append("BULL")
        elif "BEAR" in str(_tick_sig).upper():
            _corr_signals.append("BEAR")
        dal = metrics.get("delta_levels", {})
        _dal_bias = dal.get("net_bias", "")
        if _dal_bias == "BULLISH":
            _corr_signals.append("BULL")
        elif _dal_bias == "BEARISH":
            _corr_signals.append("BEAR")

        bull_corr = _corr_signals.count("BULL")
        bear_corr = _corr_signals.count("BEAR")
        corr_stack_count = max(bull_corr, bear_corr)
        corr_stack_dir = "BULLISH" if bull_corr > bear_corr else "BEARISH" if bear_corr > bull_corr else "MIXED"

        # Confluence signal details
        confl_signals = json.dumps(confluence.get("confirming", []))

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO cycle_signals (
                        cycle_id, timestamp, price,
                        fractal_dir, fractal_conf, fractal_matches,
                        mtf_score, mtf_alignment,
                        tick_proxy, tick_signal,
                        cum_delta_bias,
                        rvol, rvol_status,
                        vix, vix_structure,
                        gex_regime, flow_bias,
                        opening_type, day_type, gap_pct,
                        cross_corr_regime, iv_skew_signal, vol_shift,
                        divergence_score, divergence_severity,
                        raw_verdict, raw_confidence,
                        accuracy_adj, regime_adj,
                        confluence_level, floor_applied, forced_flat, mtf_conflict,
                        final_verdict, final_confidence, contracts,
                        trade_taken, trade_id,
                        confluence_override, confluence_bull_score, confluence_bear_score,
                        confluence_signals, fractal_in_confluence,
                        fractal_agrees_claude,
                        correlated_stack_count, correlated_stack_dir,
                        vix9d_ratio, vol_scale_ratio
                    ) VALUES (
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?,
                        ?, ?,
                        ?, ?
                    )
                """, (
                    cycle_id, ts, price,
                    getattr(proj, "direction", None) if proj else None,
                    getattr(proj, "confidence", None) if proj else None,
                    getattr(proj, "match_count", None) if proj else None,
                    mtf.get("score"),
                    mtf.get("alignment"),
                    tick.get("tick_proxy"),
                    tick.get("signal"),
                    cum_delta_bias if isinstance(cum_delta_bias, str) else str(cum_delta_bias),
                    rvol_d.get("rvol"),
                    rvol_d.get("status"),
                    vix_d.get("vix"),
                    vix_d.get("structure"),
                    gex_d.get("regime"),
                    flow_d.get("flow_bias"),
                    open_d.get("type"),
                    day_d.get("day_type"),
                    gap_d.get("gap_pct"),
                    corr_d.get("regime"),
                    skew_d.get("skew_signal"),
                    vol_d.get("shift") if isinstance(vol_d.get("shift"), str) else str(vol_d.get("shift", "")),
                    div_d.get("score"),
                    div_d.get("severity"),
                    raw_verdict,
                    raw_confidence,
                    accuracy_adj,
                    regime_adj,
                    confluence.get("confluence_level", "NONE"),
                    int(floor_applied),
                    int(forced_flat),
                    int(mtf_conflict),
                    final_verdict,
                    final_confidence,
                    contracts,
                    int(trade_taken),
                    trade_id,
                    int(confluence_override),
                    confluence.get("bull_score", 0),
                    confluence.get("bear_score", 0),
                    confl_signals,
                    fractal_in_confl,
                    fractal_agrees,
                    corr_stack_count,
                    corr_stack_dir,
                    metrics.get("vix_term", {}).get("ratio"),
                    metrics.get("vol_scaling", {}).get("ratio"),
                ))
                conn.commit()
        except Exception:
            logger.exception("SignalLogger.log_cycle failed")

    def query(self, where: str = "", params: tuple = (), limit: int = 100) -> list:
        """Generic query: SELECT * FROM cycle_signals WHERE {where} LIMIT {limit}."""
        sql = "SELECT * FROM cycle_signals"
        if where:
            sql += f" WHERE {where}"
        sql += f" ORDER BY id DESC LIMIT {limit}"
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            logger.exception("SignalLogger.query failed")
            return []

    def get_signal_summary(self, hours: int = 24) -> dict:
        """Aggregate stats for the last N hours."""
        cutoff = now_et().strftime("%Y-%m-%d %H:%M:%S")
        # Approximate: just use last N rows and filter by time
        rows = self.query(
            where="timestamp >= datetime(?, ?)",
            params=(cutoff, f"-{hours} hours"),
            limit=500,
        )
        if not rows:
            return {"cycles": 0}

        total = len(rows)
        verdicts = {}
        for r in rows:
            v = r.get("final_verdict", "UNKNOWN")
            verdicts[v] = verdicts.get(v, 0) + 1

        trades = sum(1 for r in rows if r.get("trade_taken"))
        avg_conf = sum(r.get("final_confidence", 0) for r in rows) / total if total else 0

        return {
            "cycles": total,
            "verdicts": verdicts,
            "trades_taken": trades,
            "avg_confidence": round(avg_conf, 1),
            "avg_accuracy_adj": round(
                sum(r.get("accuracy_adj", 0) for r in rows) / total, 1
            ) if total else 0,
            "avg_regime_adj": round(
                sum(r.get("regime_adj", 0) for r in rows) / total, 1
            ) if total else 0,
        }

    def format_telegram_report(self, hours: int = 24) -> str:
        """Format summary for Telegram /signals command."""
        s = self.get_signal_summary(hours)
        if s["cycles"] == 0:
            return "No signal data in the last 24 hours."

        lines = [
            f"*Signal Summary ({hours}h)*",
            f"Cycles: {s['cycles']} | Trades: {s['trades_taken']}",
            f"Avg confidence: {s['avg_confidence']}%",
            f"Avg accuracy adj: {s['avg_accuracy_adj']:+.1f}",
            f"Avg regime adj: {s['avg_regime_adj']:+.1f}",
            "",
            "Verdicts:",
        ]
        for v, count in sorted(s["verdicts"].items(), key=lambda x: -x[1]):
            pct = count / s["cycles"] * 100
            lines.append(f"  {v}: {count} ({pct:.0f}%)")

        return "\n".join(lines)
