"""
Live trading dashboard (v29 Feature #16) — Real-time web UI.

Dash app on port 8051 showing P&L, signals, open trades, key levels,
and trade history. Reads from trading_journal.db (read-only) and
cycle_memory.json. Auto-refreshes every 5 seconds.

Launch: python dashboard.py (standalone) or auto-started by market_bot
when DASHBOARD_ENABLED=True.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import dash
from dash import html, dcc, dash_table, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from bot_config import CFG, logger

# =================================================================
# --- APP SETUP ---
# =================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="ES Trading Bot Dashboard",
    update_title=None,
)

DB_PATH = CFG.DB_FILE
CYCLE_MEMORY_FILE = Path("logs/cycle_memory.json")
SIGNAL_WEIGHTS_FILE = Path("signal_weights.json")


def _read_db(query: str, params: tuple = ()) -> list:
    """Read-only query against trading journal."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _read_cycle_memory() -> dict:
    """Read cycle memory JSON."""
    try:
        if CYCLE_MEMORY_FILE.exists():
            data = json.loads(CYCLE_MEMORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            return {"cycles": data, "key_events": []}
    except Exception:
        pass
    return {"cycles": [], "key_events": []}


def _read_signal_weights() -> dict:
    """Read signal weights JSON."""
    try:
        if SIGNAL_WEIGHTS_FILE.exists():
            data = json.loads(SIGNAL_WEIGHTS_FILE.read_text(encoding="utf-8"))
            return data.get("weights", {})
    except Exception:
        pass
    return {}


# =================================================================
# --- LAYOUT ---
# =================================================================

app.layout = dbc.Container([
    # Auto-refresh interval
    dcc.Interval(id="refresh", interval=5000, n_intervals=0),

    # Header
    dbc.Row([
        dbc.Col(html.H2("ES Trading Bot", className="text-info mb-0"), width=6),
        dbc.Col(html.Div(id="header-status", className="text-end mt-2"), width=6),
    ], className="my-3"),

    # P&L + Signals Row
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Today's P&L", className="bg-dark"),
            dbc.CardBody(id="pnl-card"),
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Current Signals", className="bg-dark"),
            dbc.CardBody(id="signals-card"),
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Signal Weights", className="bg-dark"),
            dbc.CardBody(id="weights-card"),
        ]), width=4),
    ], className="mb-3"),

    # Open Trades
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Open Trades", className="bg-dark"),
            dbc.CardBody(id="open-trades-card"),
        ]), width=12),
    ], className="mb-3"),

    # Trade History
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Today's Trade History", className="bg-dark"),
            dbc.CardBody(id="history-card"),
        ]), width=12),
    ], className="mb-3"),

    # Session Events
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Session Events", className="bg-dark"),
            dbc.CardBody(id="events-card"),
        ]), width=12),
    ]),

], fluid=True, className="bg-dark text-light")


# =================================================================
# --- CALLBACKS ---
# =================================================================

@callback(
    Output("header-status", "children"),
    Output("pnl-card", "children"),
    Output("signals-card", "children"),
    Output("weights-card", "children"),
    Output("open-trades-card", "children"),
    Output("history-card", "children"),
    Output("events-card", "children"),
    Input("refresh", "n_intervals"),
)
def update_dashboard(_):
    today = datetime.now().strftime("%Y-%m-%d")

    # --- Header Status ---
    header = html.Span([
        html.Span(f"v{_get_version()} ", className="text-muted"),
        html.Span(f"| {datetime.now().strftime('%H:%M:%S')} ", className="text-muted"),
    ])

    # --- P&L Card ---
    stats = _get_today_stats(today)
    pnl_color = "text-success" if stats["net_pnl"] >= 0 else "text-danger"
    pnl_card = [
        html.H3(f"${stats['net_pnl']:+,.0f}", className=pnl_color),
        html.P([
            html.Span(f"Realized: ${stats['realized']:+,.0f} | "),
            html.Span(f"Floating: ${stats['floating']:+,.0f}"),
        ], className="text-muted mb-1"),
        html.P([
            html.Span(f"Trades: {stats['total']} | "),
            html.Span(f"W: {stats['wins']} L: {stats['losses']} | "),
            html.Span(f"WR: {stats['win_rate']:.0f}%"),
        ], className="mb-0"),
    ]

    # --- Signals Card ---
    memory = _read_cycle_memory()
    cycles = memory.get("cycles", [])
    if cycles:
        last = cycles[-1]
        verdict = last.get("verdict", "N/A")
        v_color = "text-success" if "BULL" in verdict else "text-danger" if "BEAR" in verdict else "text-muted"
        signals_card = [
            html.H4(verdict, className=v_color),
            html.P(f"Confidence: {last.get('confidence', 0)}%", className="mb-1"),
            html.P(f"Fractal: {last.get('fractal', 'N/A')}", className="mb-1"),
            html.P(f"MTF: {last.get('mtf', 'N/A')}", className="mb-1"),
            html.P(f"GEX: {last.get('gex', 'N/A')}", className="mb-0"),
        ]
    else:
        signals_card = [html.P("No cycle data yet", className="text-muted")]

    # --- Signal Weights Card ---
    weights = _read_signal_weights()
    if weights:
        weight_items = []
        for name, wt in sorted(weights.items()):
            wt_color = "text-success" if wt > 1.1 else "text-danger" if wt < 0.9 else "text-muted"
            weight_items.append(
                html.Div(f"{name}: {wt:.2f}", className=f"{wt_color} small")
            )
        weights_card = weight_items or [html.P("Default weights", className="text-muted")]
    else:
        weights_card = [html.P("Default weights (1.0)", className="text-muted")]

    # --- Open Trades ---
    open_trades = _read_db(
        "SELECT * FROM trades WHERE status IN ('OPEN', 'FLOATING') ORDER BY id"
    )
    if open_trades:
        rows = []
        for t in open_trades:
            pnl = float(t.get("pnl", 0))
            pnl_d = pnl * CFG.POINT_VALUE * int(t.get("contracts", 1) or 1)
            rows.append({
                "ID": t["id"],
                "Verdict": t["verdict"],
                "Entry": f"{float(t['price']):.2f}",
                "Target": f"{float(t.get('target', 0)):.2f}",
                "Stop": f"{float(t.get('stop', 0)):.2f}",
                "Cts": t.get("contracts", 1),
                "Float P&L": f"${pnl_d:+,.0f}",
                "Conf": f"{t.get('confidence', 0)}%",
            })
        open_card = dash_table.DataTable(
            data=rows,
            columns=[{"name": c, "id": c} for c in rows[0].keys()],
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": "#1a1a2e", "color": "#e0e0e0",
                         "border": "1px solid #333", "padding": "8px"},
            style_header={"backgroundColor": "#16213e", "fontWeight": "bold"},
        )
    else:
        open_card = html.P("No open trades", className="text-muted")

    # --- Trade History ---
    closed = _read_db(
        "SELECT * FROM trades WHERE status NOT IN ('OPEN', 'FLOATING') "
        f"AND timestamp LIKE '{today}%' ORDER BY id DESC LIMIT 20"
    )
    if closed:
        rows = []
        for t in closed:
            pnl = float(t.get("pnl", 0))
            cts = int(t.get("contracts", 1) or 1)
            pnl_d = pnl * CFG.POINT_VALUE * cts
            rows.append({
                "ID": t["id"],
                "Time": t["timestamp"][-5:] if t.get("timestamp") else "",
                "Verdict": t["verdict"],
                "Entry": f"{float(t['price']):.2f}",
                "P&L": f"${pnl_d:+,.0f} ({pnl:+.1f} pts)",
                "Status": t["status"],
                "Conf": f"{t.get('confidence', 0)}%",
            })
        history_card = dash_table.DataTable(
            data=rows,
            columns=[{"name": c, "id": c} for c in rows[0].keys()],
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": "#1a1a2e", "color": "#e0e0e0",
                         "border": "1px solid #333", "padding": "8px"},
            style_header={"backgroundColor": "#16213e", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"filter_query": "{Status} = WIN"},
                 "backgroundColor": "#0d3320", "color": "#4ade80"},
                {"if": {"filter_query": "{Status} contains LOSS"},
                 "backgroundColor": "#3b1219", "color": "#f87171"},
            ],
        )
    else:
        history_card = html.P("No trades today", className="text-muted")

    # --- Session Events ---
    events = memory.get("key_events", [])
    today_events = [e for e in events if e.get("date") == today]
    if today_events:
        event_items = []
        for e in reversed(today_events[-15:]):
            event_items.append(
                html.Div(
                    f"[{e['time']}] {e['type']}: {e['description']}"
                    + (f" @ {e['price']}" if e.get('price') else ""),
                    className="small text-muted mb-1"
                )
            )
        events_card = event_items
    else:
        events_card = [html.P("No events yet today", className="text-muted")]

    return header, pnl_card, signals_card, weights_card, open_card, history_card, events_card


def _get_today_stats(today: str) -> dict:
    """Get today's trading statistics."""
    trades = _read_db(
        "SELECT status, pnl, contracts FROM trades "
        f"WHERE timestamp LIKE '{today}%'"
    )
    wins = sum(1 for t in trades if t["status"] == "WIN")
    losses = sum(1 for t in trades if t["status"] in ("LOSS", "STOPPED", "STOP"))
    realized = sum(
        float(t["pnl"]) * CFG.POINT_VALUE * int(t.get("contracts", 1) or 1)
        for t in trades if t["status"] not in ("OPEN", "FLOATING")
    )
    floating = sum(
        float(t["pnl"]) * CFG.POINT_VALUE * int(t.get("contracts", 1) or 1)
        for t in trades if t["status"] in ("OPEN", "FLOATING")
    )
    total = wins + losses
    return {
        "wins": wins, "losses": losses, "total": total,
        "win_rate": (wins / total * 100) if total > 0 else 0,
        "realized": realized, "floating": floating,
        "net_pnl": realized + floating,
    }


def _get_version() -> str:
    """Try to read version from market_bot_v26.py."""
    try:
        bot = Path("market_bot_v26.py")
        if bot.exists():
            text = bot.read_text(encoding="utf-8")[:500]
            for line in text.split("\n"):
                if "v2" in line.lower() and ("version" in line.lower() or "v2" in line):
                    import re
                    m = re.search(r"v?(\d+\.\d+)", line)
                    if m:
                        return m.group(1)
    except Exception:
        pass
    return "29.0"


# =================================================================
# --- STANDALONE ENTRY ---
# =================================================================

if __name__ == "__main__":
    print(f"Dashboard running on http://localhost:{CFG.DASHBOARD_PORT}")
    app.run(debug=False, port=CFG.DASHBOARD_PORT, host="0.0.0.0")
