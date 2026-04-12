"""
Dash dashboard for the Structural Flow Module.

Displays all five flow signals, the composite view, and
a visual calendar of upcoming flow events.
"""

import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional

from flow_module.config import ET, DASH_PORT
from flow_module.flows import FlowSnapshot

# ── Shared state for dashboard callbacks ────────────────────
_current_snapshot: Optional[FlowSnapshot] = None


def set_snapshot(snap: FlowSnapshot):
    global _current_snapshot
    _current_snapshot = snap


def _signal_color(signal: float) -> str:
    """Color based on signal strength and direction."""
    if signal > 40:
        return "#00c853"     # strong green
    elif signal > 15:
        return "#66bb6a"     # mild green
    elif signal < -40:
        return "#ff1744"     # strong red
    elif signal < -15:
        return "#ef5350"     # mild red
    return "#78909c"         # grey / neutral


def _signal_badge(signal: float, direction: str) -> dbc.Badge:
    """Create a colored badge for a signal."""
    if direction == "BUY":
        color = "success"
    elif direction == "SELL":
        color = "danger"
    else:
        color = "secondary"
    return dbc.Badge(f"{signal:+.0f} {direction}", color=color,
                     className="ms-2", style={"fontSize": "1rem"})


def _phase_badge(phase: str) -> dbc.Badge:
    """Badge for OpEx phase."""
    colors = {
        "PIN": "info",
        "UNWIND": "warning",
        "NEUTRAL": "secondary",
    }
    return dbc.Badge(phase, color=colors.get(phase, "secondary"),
                     className="ms-2", style={"fontSize": "1rem"})


def _gauge_figure(value: float, title: str) -> go.Figure:
    """Create a gauge chart for a signal value (-100 to +100)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "", "font": {"size": 28, "color": "white"}},
        gauge={
            "axis": {"range": [-100, 100], "tickwidth": 1,
                     "tickcolor": "#444", "dtick": 50},
            "bar": {"color": _signal_color(value), "thickness": 0.3},
            "bgcolor": "#1e1e1e",
            "borderwidth": 0,
            "steps": [
                {"range": [-100, -40], "color": "rgba(255,23,68,0.15)"},
                {"range": [-40, -15], "color": "rgba(239,83,80,0.1)"},
                {"range": [-15, 15], "color": "rgba(120,144,156,0.08)"},
                {"range": [15, 40], "color": "rgba(102,187,106,0.1)"},
                {"range": [40, 100], "color": "rgba(0,200,83,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": value,
            },
        },
        title={"text": title, "font": {"size": 14, "color": "#aaa"}},
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        font={"color": "white"},
    )
    return fig


def _flow_card(title: str, signal: float, direction: str,
               description: str, extra_info: str = "") -> dbc.Card:
    """Build a card for one flow signal."""
    return dbc.Card([
        dbc.CardHeader([
            html.Span(title, style={"fontWeight": "bold", "fontSize": "1.1rem"}),
            _signal_badge(signal, direction),
        ], style={"backgroundColor": "#161b22", "borderBottom": "1px solid #30363d"}),
        dbc.CardBody([
            dcc.Graph(
                figure=_gauge_figure(signal, ""),
                config={"displayModeBar": False},
                style={"height": "180px"},
            ),
            html.P(description, style={"fontSize": "0.85rem", "color": "#c9d1d9",
                                        "marginTop": "8px"}),
            html.P(extra_info, style={"fontSize": "0.8rem", "color": "#8b949e"})
            if extra_info else html.Span(),
        ]),
    ], style={
        "backgroundColor": "#0d1117",
        "border": "1px solid #30363d",
        "marginBottom": "12px",
    })


def _opex_card(snap: FlowSnapshot) -> dbc.Card:
    """Special card for OpEx (non-directional)."""
    opex = snap.opex
    return dbc.Card([
        dbc.CardHeader([
            html.Span("OpEx Gamma", style={"fontWeight": "bold", "fontSize": "1.1rem"}),
            _phase_badge(opex.phase),
        ], style={"backgroundColor": "#161b22", "borderBottom": "1px solid #30363d"}),
        dbc.CardBody([
            html.Div([
                html.Span("Next OpEx: ", style={"color": "#8b949e"}),
                html.Span(opex.next_opex_date,
                          style={"fontWeight": "bold", "color": "white"}),
                html.Span(f" ({opex.days_to_opex}d)",
                          style={"color": "#8b949e"}),
                html.Span(" QUAD", style={"color": "#f0ad4e", "fontWeight": "bold"})
                if opex.is_quad_witching else html.Span(),
            ]),
            html.Div([
                html.Span("Vol Regime: ", style={"color": "#8b949e"}),
                html.Span(opex.vol_regime.upper(),
                          style={"fontWeight": "bold",
                                 "color": {"compressed": "#42a5f5",
                                           "expanded": "#ffa726",
                                           "normal": "#78909c"}.get(opex.vol_regime, "white")}),
            ], style={"marginTop": "8px"}),
            html.P(opex.description,
                   style={"fontSize": "0.85rem", "color": "#c9d1d9", "marginTop": "12px"}),
        ]),
    ], style={
        "backgroundColor": "#0d1117",
        "border": "1px solid #30363d",
        "marginBottom": "12px",
    })


def build_layout() -> html.Div:
    """Build the complete dashboard layout."""
    return html.Div([
        # Auto-refresh
        dcc.Interval(id="flow-refresh", interval=10_000, n_intervals=0),

        # Header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("ES Structural Flow Monitor",
                            style={"color": "white", "marginBottom": "0"}),
                    html.P(id="flow-headline",
                           style={"color": "#8b949e", "fontSize": "1rem",
                                  "marginTop": "4px"}),
                ], width=8),
                dbc.Col([
                    html.Div(id="flow-timestamp",
                             style={"textAlign": "right", "color": "#8b949e",
                                    "paddingTop": "12px"}),
                ], width=4),
            ], className="my-3"),

            # Net composite gauge (big)
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                id="net-gauge",
                                config={"displayModeBar": False},
                                style={"height": "220px"},
                            ),
                        ]),
                    ], style={
                        "backgroundColor": "#0d1117",
                        "border": "1px solid #30363d",
                    }),
                ], width=12),
            ], className="mb-3"),

            # Flow cards
            html.Div(id="flow-cards"),

            # Detail tables (CTA levels, GEX strikes, seasonal effects)
            html.Div(id="detail-tables"),

        ], fluid=True, style={
            "backgroundColor": "#010409",
            "minHeight": "100vh",
            "padding": "16px",
        }),
    ])


def register_callbacks(app: dash.Dash):
    """Register all dashboard callbacks."""

    @app.callback(
        [
            Output("flow-headline", "children"),
            Output("flow-timestamp", "children"),
            Output("net-gauge", "figure"),
            Output("flow-cards", "children"),
            Output("detail-tables", "children"),
        ],
        Input("flow-refresh", "n_intervals"),
    )
    def update_dashboard(_n):
        snap = _current_snapshot

        if snap is None:
            empty_fig = _gauge_figure(0, "NET STRUCTURAL FLOW")
            return (
                "Waiting for data...",
                "",
                empty_fig,
                html.Div("Loading flow data...",
                         style={"color": "#8b949e", "textAlign": "center"}),
                html.Div(),
            )

        # Headline
        headline = snap.headline

        # Timestamp
        ts = snap.timestamp.strftime("%Y-%m-%d %H:%M:%S ET")

        # Net gauge
        net_fig = _gauge_figure(snap.net_signal, "NET STRUCTURAL FLOW")

        # Flow cards — build rows
        gex = snap.gex
        seasonal = snap.seasonality

        cards = html.Div([
            # Row 1: GEX + OpEx (the two gamma/vol signals)
            dbc.Row([
                dbc.Col([
                    _gex_card(snap) if gex else _flow_card(
                        "GEX (Gamma Exposure)", 0, "NEUTRAL",
                        "Waiting for options chain data...", ""),
                ], md=6, sm=12),
                dbc.Col([
                    _opex_card(snap),
                ], md=6, sm=12),
            ]),
            # Row 2: CTA + Vol Control
            dbc.Row([
                dbc.Col([
                    _flow_card(
                        "CTA Trend-Following",
                        snap.cta.signal,
                        snap.cta.flow_direction,
                        snap.cta.description,
                        f"Nearest trigger: {snap.cta.nearest_trigger}"
                    ),
                ], md=6, sm=12),
                dbc.Col([
                    _flow_card(
                        "Vol-Control / Risk Parity",
                        snap.vol_control.signal,
                        snap.vol_control.flow_direction,
                        snap.vol_control.description,
                        f"1M RV: {snap.vol_control.short_rv:.1f}% | "
                        f"3M RV: {snap.vol_control.long_rv:.1f}% | "
                        f"RV RoC: {snap.vol_control.rv_roc:+.1f}%/wk"
                    ),
                ], md=6, sm=12),
            ]),
            # Row 3: Rebalance + Buyback
            dbc.Row([
                dbc.Col([
                    _flow_card(
                        "Pension Rebalancing",
                        snap.rebalance.signal,
                        snap.rebalance.flow_direction,
                        snap.rebalance.description,
                        f"MTD: {snap.rebalance.mtd_return_pct:+.1f}% | "
                        f"QTD: {snap.rebalance.qtd_return_pct:+.1f}% | "
                        f"Days to EOM: {snap.rebalance.days_to_month_end}"
                    ),
                ], md=6, sm=12),
                dbc.Col([
                    _flow_card(
                        "Corporate Buybacks",
                        snap.buyback.signal,
                        "BUY" if snap.buyback.signal > 10 else
                        ("SELL" if snap.buyback.signal < -10 else "NEUTRAL"),
                        snap.buyback.description,
                        f"Est. daily flow: ${snap.buyback.estimated_daily_flow_b:.1f}B | "
                        f"Phase: {snap.buyback.blackout_phase}"
                    ),
                ], md=6, sm=12),
            ]),
            # Row 4: Seasonality + Summary
            dbc.Row([
                dbc.Col([
                    _seasonality_card(snap) if seasonal else _flow_card(
                        "Seasonality", 0, "NEUTRAL", "Loading...", ""),
                ], md=6, sm=12),
                dbc.Col([
                    _summary_card(snap),
                ], md=6, sm=12),
            ]),
        ])

        # Detail tables
        detail = html.Div([
            _build_cta_table(snap),
            _build_gex_table(snap),
            _build_seasonal_table(snap),
        ])

        return headline, ts, net_fig, cards, detail


def _gex_card(snap: FlowSnapshot) -> dbc.Card:
    """Card for GEX gamma exposure."""
    gex = snap.gex
    if gex is None:
        return _flow_card("GEX (Gamma Exposure)", 0, "NEUTRAL",
                          "Waiting for options data...", "")

    regime_color = {
        "POSITIVE": "#00c853", "NEGATIVE": "#ff1744", "NEUTRAL": "#ffc107",
        "UNKNOWN": "#78909c",
    }.get(gex.gamma_regime, "#78909c")

    return dbc.Card([
        dbc.CardHeader([
            html.Span("GEX (Gamma Exposure)",
                       style={"fontWeight": "bold", "fontSize": "1.1rem"}),
            _signal_badge(gex.signal,
                          "BUY" if gex.signal > 10 else
                          "SELL" if gex.signal < -10 else "NEUTRAL"),
        ], style={"backgroundColor": "#161b22",
                  "borderBottom": "1px solid #30363d"}),
        dbc.CardBody([
            html.Div([
                html.Span("Gamma Flip: ", style={"color": "#8b949e"}),
                html.Span(f"{gex.gamma_flip_level:.0f}",
                          style={"fontWeight": "bold", "fontSize": "1.3rem",
                                 "color": "white"}),
                html.Span(f"  ({gex.distance_to_flip:+.0f}pts)",
                          style={"color": regime_color, "fontWeight": "bold"}),
            ]),
            html.Div([
                html.Span("Regime: ", style={"color": "#8b949e"}),
                html.Span(gex.gamma_regime,
                          style={"fontWeight": "bold", "color": regime_color}),
            ], style={"marginTop": "6px"}),
            html.P(gex.description,
                   style={"fontSize": "0.85rem", "color": "#c9d1d9",
                          "marginTop": "10px"}),
            html.P("STALE DATA — waiting for refresh",
                   style={"fontSize": "0.8rem", "color": "#f0ad4e"})
            if gex.stale else html.Span(),
        ]),
    ], style={
        "backgroundColor": "#0d1117",
        "border": "1px solid #30363d",
        "marginBottom": "12px",
    })


def _seasonality_card(snap: FlowSnapshot) -> dbc.Card:
    """Card for calendar seasonality."""
    seasonal = snap.seasonality
    if seasonal is None:
        return _flow_card("Seasonality", 0, "NEUTRAL", "Loading...", "")

    return _flow_card(
        "Seasonality",
        seasonal.signal,
        seasonal.bias_direction,
        seasonal.description,
        f"Active: {', '.join(seasonal.active_effects)}"
        if seasonal.active_effects else "No strong seasonal effects today"
    )


def _summary_card(snap: FlowSnapshot) -> dbc.Card:
    """Summary of all active flows and net signal."""
    rows = []
    flow_items = [
        ("GEX", snap.gex.signal if snap.gex and not snap.gex.stale else 0,
         snap.gex.gamma_regime if snap.gex and not snap.gex.stale else "N/A"),
        ("Rebalance", snap.rebalance.signal, snap.rebalance.flow_direction),
        ("CTA", snap.cta.signal, snap.cta.flow_direction),
        ("Vol-Control", snap.vol_control.signal, snap.vol_control.flow_direction),
        ("Buyback", snap.buyback.signal,
         "BUY" if snap.buyback.signal > 10 else "SELL" if snap.buyback.signal < -10 else "NEUTRAL"),
        ("OpEx", snap.opex.magnitude * (1 if snap.opex.phase != "PIN" else -1),
         snap.opex.phase),
        ("Seasonal", snap.seasonality.signal if snap.seasonality else 0,
         snap.seasonality.bias_direction if snap.seasonality else "N/A"),
    ]

    for name, sig, direction in flow_items:
        color = _signal_color(sig)
        rows.append(html.Tr([
            html.Td(name, style={"color": "#c9d1d9"}),
            html.Td(f"{sig:+.0f}", style={"color": color, "fontWeight": "bold",
                                            "textAlign": "right"}),
            html.Td(direction, style={"color": color, "textAlign": "center"}),
        ]))

    # Net row
    net_color = _signal_color(snap.net_signal)
    rows.append(html.Tr([
        html.Td("NET", style={"color": "white", "fontWeight": "bold",
                               "borderTop": "1px solid #30363d"}),
        html.Td(f"{snap.net_signal:+.0f}", style={
            "color": net_color, "fontWeight": "bold",
            "textAlign": "right", "fontSize": "1.2rem",
            "borderTop": "1px solid #30363d"}),
        html.Td(snap.net_direction, style={
            "color": net_color, "fontWeight": "bold",
            "textAlign": "center", "borderTop": "1px solid #30363d"}),
    ]))

    return dbc.Card([
        dbc.CardHeader(
            html.Span("Flow Summary",
                       style={"fontWeight": "bold", "fontSize": "1.1rem"}),
            style={"backgroundColor": "#161b22",
                   "borderBottom": "1px solid #30363d"}),
        dbc.CardBody([
            html.Table([html.Tbody(rows)],
                       style={"width": "100%"}),
            html.P(f"Conviction: {snap.conviction.upper()}",
                   style={"textAlign": "center", "marginTop": "12px",
                          "fontSize": "1rem", "fontWeight": "bold",
                          "color": {"high": "#00c853", "moderate": "#ffc107",
                                    "low": "#78909c", "none": "#555"}.get(
                              snap.conviction, "#555")}),
        ]),
    ], style={
        "backgroundColor": "#0d1117",
        "border": "1px solid #30363d",
        "marginBottom": "12px",
    })


def _build_cta_table(snap: FlowSnapshot) -> html.Div:
    """Build a detail table showing MA levels for CTA tracking."""
    if not snap.cta.ma_levels:
        return html.Div()

    rows = []
    for lvl in snap.cta.ma_levels:
        status = ""
        color = "#78909c"
        if lvl.crossed_above and lvl.days_since_cross <= 3:
            status = f"CROSSED ABOVE {lvl.days_since_cross}d ago"
            color = "#00c853"
        elif lvl.crossed_below and lvl.days_since_cross <= 3:
            status = f"CROSSED BELOW {lvl.days_since_cross}d ago"
            color = "#ff1744"
        elif lvl.is_near:
            status = "APPROACHING"
            color = "#ffc107"
        else:
            above_below = "above" if lvl.distance_pct > 0 else "below"
            status = f"{abs(lvl.distance_pct):.1f}% {above_below}"

        rows.append(html.Tr([
            html.Td(f"{lvl.period} DMA", style={"color": "#c9d1d9",
                                                  "fontWeight": "bold"}),
            html.Td(f"{lvl.ma_value:.1f}", style={"color": "white",
                                                    "textAlign": "right"}),
            html.Td(f"{lvl.distance_pct:+.1f}%", style={
                "color": "#00c853" if lvl.distance_pct > 0 else "#ff1744",
                "textAlign": "right"}),
            html.Td(status, style={"color": color, "textAlign": "center",
                                    "fontWeight": "bold" if color != "#78909c" else "normal"}),
        ]))

    return dbc.Card([
        dbc.CardHeader(
            html.Span("CTA Moving Average Levels",
                       style={"fontWeight": "bold", "fontSize": "1rem"}),
            style={"backgroundColor": "#161b22",
                   "borderBottom": "1px solid #30363d"}),
        dbc.CardBody([
            html.Table([
                html.Thead(html.Tr([
                    html.Th("MA", style={"color": "#8b949e"}),
                    html.Th("Level", style={"color": "#8b949e", "textAlign": "right"}),
                    html.Th("Distance", style={"color": "#8b949e", "textAlign": "right"}),
                    html.Th("Status", style={"color": "#8b949e", "textAlign": "center"}),
                ])),
                html.Tbody(rows),
            ], style={"width": "100%"}),
        ]),
    ], style={
        "backgroundColor": "#0d1117",
        "border": "1px solid #30363d",
        "marginTop": "12px",
    })


def _build_gex_table(snap: FlowSnapshot) -> html.Div:
    """Detail table showing top GEX strikes."""
    if snap.gex is None or not snap.gex.top_strikes:
        return html.Div()

    rows = []
    for s in snap.gex.top_strikes:
        net_color = "#00c853" if s.net_gex > 0 else "#ff1744"
        rows.append(html.Tr([
            html.Td(f"{s.strike:.0f}", style={"color": "white", "fontWeight": "bold"}),
            html.Td(f"{s.call_oi:,}", style={"color": "#66bb6a", "textAlign": "right"}),
            html.Td(f"{s.put_oi:,}", style={"color": "#ef5350", "textAlign": "right"}),
            html.Td(f"{s.call_gex:,.0f}", style={"color": "#66bb6a", "textAlign": "right"}),
            html.Td(f"{s.put_gex:,.0f}", style={"color": "#ef5350", "textAlign": "right"}),
            html.Td(f"{s.net_gex:+,.0f}", style={"color": net_color,
                                                    "textAlign": "right",
                                                    "fontWeight": "bold"}),
        ]))

    return dbc.Card([
        dbc.CardHeader([
            html.Span("GEX by Strike (Top 5)",
                       style={"fontWeight": "bold", "fontSize": "1rem"}),
            html.Span(f"  Flip: {snap.gex.gamma_flip_level:.0f}",
                       style={"color": "#ffc107", "fontSize": "0.9rem",
                              "marginLeft": "12px"}),
        ], style={"backgroundColor": "#161b22",
                  "borderBottom": "1px solid #30363d"}),
        dbc.CardBody([
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Strike", style={"color": "#8b949e"}),
                    html.Th("Call OI", style={"color": "#8b949e", "textAlign": "right"}),
                    html.Th("Put OI", style={"color": "#8b949e", "textAlign": "right"}),
                    html.Th("Call GEX", style={"color": "#8b949e", "textAlign": "right"}),
                    html.Th("Put GEX", style={"color": "#8b949e", "textAlign": "right"}),
                    html.Th("Net GEX", style={"color": "#8b949e", "textAlign": "right"}),
                ])),
                html.Tbody(rows),
            ], style={"width": "100%"}),
        ]),
    ], style={
        "backgroundColor": "#0d1117",
        "border": "1px solid #30363d",
        "marginTop": "12px",
    })


def _build_seasonal_table(snap: FlowSnapshot) -> html.Div:
    """Detail table showing all seasonal effects."""
    if snap.seasonality is None:
        return html.Div()

    rows = []
    for e in snap.seasonality.effects:
        color = "#78909c"
        if e.active and e.bias > 0:
            color = "#00c853"
        elif e.active and e.bias < 0:
            color = "#ff1744"

        rows.append(html.Tr([
            html.Td(e.name, style={"color": "#c9d1d9", "fontWeight": "bold"}),
            html.Td(f"{e.bias:+.1f}", style={"color": color, "textAlign": "right",
                                               "fontWeight": "bold"}),
            html.Td("ACTIVE" if e.active else "inactive",
                     style={"color": color if e.active else "#555",
                            "textAlign": "center",
                            "fontWeight": "bold" if e.active else "normal"}),
            html.Td(e.description, style={"color": "#8b949e",
                                           "fontSize": "0.8rem"}),
        ]))

    return dbc.Card([
        dbc.CardHeader(
            html.Span("Seasonal Effects",
                       style={"fontWeight": "bold", "fontSize": "1rem"}),
            style={"backgroundColor": "#161b22",
                   "borderBottom": "1px solid #30363d"}),
        dbc.CardBody([
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Effect", style={"color": "#8b949e"}),
                    html.Th("Bias", style={"color": "#8b949e", "textAlign": "right"}),
                    html.Th("Status", style={"color": "#8b949e", "textAlign": "center"}),
                    html.Th("Detail", style={"color": "#8b949e"}),
                ])),
                html.Tbody(rows),
            ], style={"width": "100%"}),
        ]),
    ], style={
        "backgroundColor": "#0d1117",
        "border": "1px solid #30363d",
        "marginTop": "12px",
    })


def create_app() -> dash.Dash:
    """Create and configure the Dash app."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="ES Flow Monitor",
    )
    app.layout = build_layout()
    register_callbacks(app)
    return app
