"""
Weekly strategy report (v29 Feature #18) — PDF generation.

Generates a PDF report with:
  - Weekly P&L table and summary
  - Signal performance breakdown
  - Regime summary
  - Top lessons from trade reviews
  - Session stats

Saves to logs/reports/. Triggered via /report command or auto-generated
on Friday after close.

Requires: fpdf2 >= 2.7.0
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from bot_config import CFG, now_et

logger = logging.getLogger("weekly_report")

REPORTS_DIR = Path("logs/reports")


def _ensure_reports_dir():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_weekly_report(journal, days: int = 7) -> Optional[Path]:
    """
    Generate a PDF weekly strategy report.

    Args:
        journal: Journal instance for DB queries
        days: Number of days to cover (default 7)

    Returns:
        Path to generated PDF, or None on failure.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.error("fpdf2 not installed — run: pip install fpdf2")
        return None

    _ensure_reports_dir()

    now = now_et()
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    filename = f"weekly_report_{end_date}.pdf"
    filepath = REPORTS_DIR / filename

    # --- Gather data ---
    trades = _get_trades(start_date, end_date)
    daily_stats = _get_daily_stats(trades)
    signal_perf = _get_signal_performance(trades)
    regime_summary = _get_regime_summary(start_date, end_date)
    lessons = _get_lessons(start_date, end_date)

    # --- Summary stats ---
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["status"] == "WIN")
    losses = sum(1 for t in trades if t["status"] in ("LOSS", "STOPPED", "STOP"))
    total_pnl_pts = sum(float(t.get("pnl", 0)) for t in trades
                        if t["status"] not in ("OPEN", "FLOATING"))
    total_pnl_d = total_pnl_pts * CFG.POINT_VALUE
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    avg_win = 0
    avg_loss = 0
    if wins > 0:
        avg_win = sum(float(t["pnl"]) for t in trades if t["status"] == "WIN") / wins
    if losses > 0:
        avg_loss = sum(float(t["pnl"]) for t in trades
                       if t["status"] in ("LOSS", "STOPPED", "STOP")) / losses

    # --- Build PDF ---
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "ES Trading Bot - Weekly Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"{start_date} to {end_date}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # --- Summary Section ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Performance Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    summary_data = [
        ("Total Trades", str(total_trades)),
        ("Wins / Losses", f"{wins} / {losses}"),
        ("Win Rate", f"{win_rate:.1f}%"),
        ("Total P&L", f"${total_pnl_d:+,.0f} ({total_pnl_pts:+.1f} pts)"),
        ("Avg Win", f"{avg_win:+.1f} pts (${avg_win * CFG.POINT_VALUE:+,.0f})"),
        ("Avg Loss", f"{avg_loss:+.1f} pts (${avg_loss * CFG.POINT_VALUE:+,.0f})"),
        ("Expectancy", f"{avg_win * (win_rate/100) + avg_loss * (1 - win_rate/100):.1f} pts/trade"
         if (wins + losses) > 0 else "N/A"),
    ]

    col_w = 55
    val_w = 80
    for label, value in summary_data:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 6, label, border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(val_w, 6, value, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # --- Daily Breakdown Table ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Daily Breakdown", new_x="LMARGIN", new_y="NEXT")

    headers = ["Date", "Trades", "Wins", "Losses", "P&L (pts)", "P&L ($)"]
    col_widths = [30, 20, 20, 20, 30, 30]

    # Header row
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(50, 50, 80)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(0, 0, 0)
    for day in sorted(daily_stats.keys()):
        ds = daily_stats[day]
        row = [
            day,
            str(ds["total"]),
            str(ds["wins"]),
            str(ds["losses"]),
            f"{ds['pnl_pts']:+.1f}",
            f"${ds['pnl_d']:+,.0f}",
        ]
        for i, val in enumerate(row):
            pdf.cell(col_widths[i], 6, val, border=1, align="C")
        pdf.ln()

    pdf.ln(6)

    # --- Signal Performance ---
    if signal_perf:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Signal Performance", new_x="LMARGIN", new_y="NEXT")

        sig_headers = ["Signal Combo", "Trades", "Win Rate", "Avg P&L"]
        sig_widths = [60, 25, 30, 35]

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(50, 50, 80)
        pdf.set_text_color(255, 255, 255)
        for i, h in enumerate(sig_headers):
            pdf.cell(sig_widths[i], 7, h, border=1, fill=True, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 0)
        for sp in signal_perf[:10]:
            row = [
                sp["combo"][:30],
                str(sp["total"]),
                f"{sp['win_rate']:.0f}%",
                f"{sp['avg_pnl']:+.1f} pts",
            ]
            for i, val in enumerate(row):
                pdf.cell(sig_widths[i], 6, val, border=1, align="C")
            pdf.ln()

        pdf.ln(6)

    # --- Regime Summary ---
    if regime_summary:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Regime Distribution", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for regime, count in sorted(regime_summary.items(), key=lambda x: -x[1]):
            pct = count / sum(regime_summary.values()) * 100
            pdf.cell(0, 6, f"  {regime}: {count} cycles ({pct:.0f}%)",
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

    # --- Lessons Learned ---
    if lessons:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Key Lessons", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for i, lesson in enumerate(lessons[:8], 1):
            text = f"{i}. [{lesson['date']}] {lesson['text']}"
            pdf.multi_cell(0, 5, text[:200], new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # --- Footer ---
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 6, f"Generated {now.strftime('%Y-%m-%d %H:%M ET')} | ES Trading Bot v29",
             new_x="LMARGIN", new_y="NEXT", align="C")

    # Save
    try:
        pdf.output(str(filepath))
        logger.info(f"[WEEKLY REPORT] Saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"[WEEKLY REPORT] PDF save failed: {e}")
        return None


# ── Data helpers ──────────────────────────────────────────────────

def _get_trades(start_date: str, end_date: str) -> list:
    """Get all trades in date range from journal DB."""
    try:
        conn = sqlite3.connect(CFG.DB_FILE, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trades WHERE timestamp >= ? AND timestamp <= ?",
            (start_date, end_date + " 23:59:59")
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"_get_trades failed: {e}")
        return []


def _get_daily_stats(trades: list) -> dict:
    """Group trades by date and compute daily stats."""
    daily = {}
    for t in trades:
        if t["status"] in ("OPEN", "FLOATING"):
            continue
        date = t["timestamp"][:10] if t.get("timestamp") else "unknown"
        if date not in daily:
            daily[date] = {"total": 0, "wins": 0, "losses": 0, "pnl_pts": 0, "pnl_d": 0}
        d = daily[date]
        d["total"] += 1
        pnl = float(t.get("pnl", 0))
        cts = int(t.get("contracts", 1) or 1)
        if t["status"] == "WIN":
            d["wins"] += 1
        elif t["status"] in ("LOSS", "STOPPED", "STOP"):
            d["losses"] += 1
        d["pnl_pts"] += pnl
        d["pnl_d"] += pnl * CFG.POINT_VALUE * cts
    return daily


def _get_signal_performance(trades: list) -> list:
    """Compute win rate by signal combo from trade signals JSON."""
    combos = {}
    for t in trades:
        if t["status"] in ("OPEN", "FLOATING"):
            continue
        signals_raw = t.get("signals", "{}")
        try:
            if isinstance(signals_raw, str):
                signals = eval(signals_raw) if signals_raw.startswith("{") else {}
            else:
                signals = signals_raw or {}
        except Exception:
            signals = {}

        # Build combo string from active signals
        active = []
        for k, v in sorted(signals.items()):
            if v and str(v) not in ("N/A", "NONE", "0", "0.0", "1.0"):
                active.append(k)
        combo = "+".join(active[:4]) if active else "unknown"

        if combo not in combos:
            combos[combo] = {"total": 0, "wins": 0, "pnl_sum": 0}
        combos[combo]["total"] += 1
        pnl = float(t.get("pnl", 0))
        combos[combo]["pnl_sum"] += pnl
        if t["status"] == "WIN":
            combos[combo]["wins"] += 1

    result = []
    for combo, stats in combos.items():
        if stats["total"] >= 2:
            result.append({
                "combo": combo,
                "total": stats["total"],
                "win_rate": stats["wins"] / stats["total"] * 100,
                "avg_pnl": stats["pnl_sum"] / stats["total"],
            })
    return sorted(result, key=lambda x: -x["win_rate"])


def _get_regime_summary(start_date: str, end_date: str) -> dict:
    """Count regime occurrences from cycle_signals."""
    try:
        db_path = Path("signal_logger.db")
        if not db_path.exists():
            return {}
        conn = sqlite3.connect(str(db_path), timeout=5)
        rows = conn.execute(
            "SELECT final_verdict FROM cycle_signals "
            "WHERE timestamp >= ? AND timestamp <= ?",
            (start_date, end_date + " 23:59:59")
        ).fetchall()
        conn.close()
        counts = {}
        for r in rows:
            v = r[0] or "UNKNOWN"
            counts[v] = counts.get(v, 0) + 1
        return counts
    except Exception:
        return {}


def _get_lessons(start_date: str, end_date: str) -> list:
    """Get lessons from trade reviews."""
    try:
        conn = sqlite3.connect(CFG.DB_FILE, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT timestamp, review FROM trade_reviews "
            "WHERE timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp DESC LIMIT 10",
            (start_date, end_date + " 23:59:59")
        ).fetchall()
        conn.close()
        lessons = []
        for r in rows:
            review_text = r["review"] or ""
            # Extract key lesson (first sentence or line)
            lines = review_text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and len(line) > 20 and "lesson" in line.lower():
                    lessons.append({"date": r["timestamp"][:10], "text": line})
                    break
            else:
                if lines and lines[0].strip():
                    lessons.append({"date": r["timestamp"][:10], "text": lines[0][:200]})
        return lessons
    except Exception:
        return []
