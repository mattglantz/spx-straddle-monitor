import asyncio
import threading

# MUST create an event loop before importing ib_async / eventkit
# (required on Python 3.10+ where get_event_loop() no longer auto-creates one)
asyncio.set_event_loop(asyncio.new_event_loop())

import pandas as pd
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, time, timedelta
import math
from zoneinfo import ZoneInfo
import logging
import webbrowser
from ib_async import IB, Index, Option, ContFuture

# ============================================================
# CONFIGURATION
# ============================================================
PORT = 7496              # 7497 for paper trading
CLIENT_ID = 2
UNDERLYING = 'SPX'
EXCHANGE = 'CBOE'
STRIKE_STEP = 5
NUM_EXPIRIES = 9         # 0DTE through 8DTE coverage
THEME = dbc.themes.CYBORG

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

BACKEND_INTERVAL_SEC = 120
UI_INTERVAL_MS = 120000
EXPIRY_REFRESH_MINUTES = 60
RECONNECT_DELAY_SEC = 10

ET = ZoneInfo("America/New_York")

# ============================================================
# ECONOMIC CALENDAR
# ============================================================
ECONOMIC_EVENTS = [
    ("2025-01-29", "FOMC Decision"), ("2025-03-19", "FOMC Decision"),
    ("2025-05-07", "FOMC Decision"), ("2025-06-18", "FOMC Decision"),
    ("2025-07-30", "FOMC Decision"), ("2025-09-17", "FOMC Decision"),
    ("2025-10-29", "FOMC Decision"), ("2025-12-10", "FOMC Decision"),
    ("2026-01-28", "FOMC Decision"), ("2026-03-18", "FOMC Decision"),
    ("2026-05-06", "FOMC Decision"), ("2026-06-17", "FOMC Decision"),
    ("2026-07-29", "FOMC Decision"), ("2026-09-16", "FOMC Decision"),
    ("2026-10-28", "FOMC Decision"), ("2026-12-09", "FOMC Decision"),
    ("2025-01-15", "CPI"), ("2025-02-12", "CPI"), ("2025-03-12", "CPI"),
    ("2025-04-10", "CPI"), ("2025-05-13", "CPI"), ("2025-06-11", "CPI"),
    ("2025-07-10", "CPI"), ("2025-08-12", "CPI"), ("2025-09-10", "CPI"),
    ("2025-10-14", "CPI"), ("2025-11-12", "CPI"), ("2025-12-10", "CPI"),
    ("2026-01-14", "CPI"), ("2026-02-11", "CPI"), ("2026-03-11", "CPI"),
    ("2026-04-14", "CPI"), ("2026-05-12", "CPI"), ("2026-06-10", "CPI"),
    ("2026-07-14", "CPI"), ("2026-08-12", "CPI"), ("2026-09-15", "CPI"),
    ("2026-10-13", "CPI"), ("2026-11-10", "CPI"), ("2026-12-10", "CPI"),
    ("2025-01-10", "NFP"), ("2025-02-07", "NFP"), ("2025-03-07", "NFP"),
    ("2025-04-04", "NFP"), ("2025-05-02", "NFP"), ("2025-06-06", "NFP"),
    ("2025-07-03", "NFP"), ("2025-08-01", "NFP"), ("2025-09-05", "NFP"),
    ("2025-10-03", "NFP"), ("2025-11-07", "NFP"), ("2025-12-05", "NFP"),
    ("2026-01-09", "NFP"), ("2026-02-06", "NFP"), ("2026-03-06", "NFP"),
    ("2026-04-03", "NFP"), ("2026-05-01", "NFP"), ("2026-06-05", "NFP"),
    ("2026-07-02", "NFP"), ("2026-08-07", "NFP"), ("2026-09-04", "NFP"),
    ("2026-10-02", "NFP"), ("2026-11-06", "NFP"), ("2026-12-04", "NFP"),
    ("2025-01-30", "GDP"), ("2025-03-27", "GDP"), ("2025-04-30", "GDP"),
    ("2025-06-26", "GDP"), ("2025-07-30", "GDP"), ("2025-09-25", "GDP"),
    ("2025-10-29", "GDP"), ("2025-12-23", "GDP"),
    ("2026-01-29", "GDP"), ("2026-03-26", "GDP"), ("2026-04-29", "GDP"),
    ("2026-06-25", "GDP"), ("2026-07-29", "GDP"), ("2026-09-24", "GDP"),
    ("2026-10-28", "GDP"), ("2026-12-22", "GDP"),
    ("2025-01-17", "Monthly OpEx"), ("2025-02-21", "Monthly OpEx"),
    ("2025-03-21", "Quad Witching"), ("2025-04-17", "Monthly OpEx"),
    ("2025-05-16", "Monthly OpEx"), ("2025-06-20", "Quad Witching"),
    ("2025-07-18", "Monthly OpEx"), ("2025-08-15", "Monthly OpEx"),
    ("2025-09-19", "Quad Witching"), ("2025-10-17", "Monthly OpEx"),
    ("2025-11-21", "Monthly OpEx"), ("2025-12-19", "Quad Witching"),
    ("2026-01-16", "Monthly OpEx"), ("2026-02-20", "Monthly OpEx"),
    ("2026-03-20", "Quad Witching"), ("2026-04-17", "Monthly OpEx"),
    ("2026-05-15", "Monthly OpEx"), ("2026-06-19", "Quad Witching"),
    ("2026-07-17", "Monthly OpEx"), ("2026-08-21", "Monthly OpEx"),
    ("2026-09-18", "Quad Witching"), ("2026-10-16", "Monthly OpEx"),
    ("2026-11-20", "Monthly OpEx"), ("2026-12-18", "Quad Witching"),
]


def get_events_on_date(expiry_str):
    exp_date = datetime.strptime(expiry_str, '%Y%m%d').date()
    exp_str = exp_date.strftime('%Y-%m-%d')
    return [name for ds, name in ECONOMIC_EVENTS if ds == exp_str]


def get_upcoming_events():
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    today_str = today.strftime('%Y-%m-%d')
    tomorrow_str = tomorrow.strftime('%Y-%m-%d')
    today_evts = [name for ds, name in ECONOMIC_EVENTS if ds == today_str]
    tomorrow_evts = [name for ds, name in ECONOMIC_EVENTS if ds == tomorrow_str]
    return today_evts, tomorrow_evts


# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
logging.getLogger('ib_async').setLevel(logging.ERROR)

# ============================================================
# SHARED STATE
# ============================================================
shared_state = {
    "df": pd.DataFrame(),
    "spot_price": 0.0,
    "active_strike": 0,
    "status": "Initializing...",
    "mode": "Starting",
    "last_update": datetime.now().strftime("%H:%M:%S"),
    "session_open_spot": None,
    "session_date": None,
    "open_straddles": {},
    "prior_close": None,
    "vix_price": 0.0,
    "vix_prior_close": None,
    "straddle_history": [],
}
state_lock = threading.Lock()

# ============================================================
# HELPERS
# ============================================================

def safe_float(val):
    if val is None:
        return 0.0
    try:
        return 0.0 if math.isnan(val) or math.isinf(val) else float(val)
    except (TypeError, ValueError):
        return 0.0


def get_nearest_strike(price, step=STRIKE_STEP):
    if price is None or price <= 0:
        return 0
    return step * round(price / step)


def calc_dte(expiry_str):
    exp_date = datetime.strptime(expiry_str, '%Y%m%d').date()
    return max((exp_date - datetime.now().date()).days, 0)


def calc_wing_offset(spot, annual_vol, dte, strike_step=STRIKE_STEP):
    """Estimate strike offset for ~25-delta options.
    25-delta ≈ 0.6745 std deviations OTM in Black-Scholes."""
    dte_adj = max(dte, 0.5)  # half-day floor for 0DTE
    offset = spot * annual_vol * 0.6745 * math.sqrt(dte_adj / 365)
    return strike_step * max(1, round(offset / strike_step))


def get_greeks(ticker):
    for g in (ticker.modelGreeks, ticker.lastGreeks, ticker.bidGreeks, ticker.askGreeks):
        if g is not None:
            iv = safe_float(g.impliedVol)
            if iv > 0:
                return iv, safe_float(g.gamma), safe_float(g.theta)
    return 0.0, 0.0, 0.0


def safe_bid_ask(ticker):
    bid = safe_float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0.0
    ask = safe_float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0.0
    mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0.0
    return bid, ask, mid


def is_regular_session():
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


# ============================================================
# EXPIRY FETCHING
# ============================================================

async def get_next_n_spxw_expiries(ib, contract, n=NUM_EXPIRIES):
    log.info("Fetching option chains...")
    if not contract.conId:
        await ib.qualifyContractsAsync(contract)
    chains = await ib.reqSecDefOptParamsAsync(
        contract.symbol, '', contract.secType, contract.conId
    )
    spxw_chains = [c for c in chains if c.tradingClass == 'SPXW' and c.exchange == 'SMART']
    if not spxw_chains:
        spxw_chains = [c for c in chains if c.exchange == 'SMART']
    all_expirations = set()
    for chain in spxw_chains:
        all_expirations.update(chain.expirations)
    sorted_exp = sorted(all_expirations)
    today_str = datetime.now().strftime('%Y%m%d')
    future_exps = [e for e in sorted_exp
                    if e >= today_str and datetime.strptime(e, '%Y%m%d').weekday() < 5][:n]
    trading_class = spxw_chains[0].tradingClass if spxw_chains else 'SPX'
    log.info(f"Expiries ({len(future_exps)}): {future_exps}  class={trading_class}")
    return [(exp, trading_class) for exp in future_exps]


# ============================================================
# SUBSCRIPTION MANAGER
# ============================================================

class StraddleSubscription:
    def __init__(self):
        self.active_tickers = []
        self.tickers_map = {}
        self.wing_tickers_map = {}
        self.strike = 0

    async def update(self, ib, target_strike, expiries_info, spot=0, est_annual_vol=0.15):
        if target_strike == self.strike or target_strike <= 0:
            return False
        log.info(f"Strike change: {self.strike} -> {target_strike}")
        self._cancel_all(ib)
        contracts_by_expiry = {}
        wing_contracts_by_expiry = {}
        to_qualify = []
        for expiry, t_class in expiries_info:
            # ATM straddle contracts
            call = Option(UNDERLYING, expiry, target_strike, 'C', 'SMART', tradingClass=t_class)
            put = Option(UNDERLYING, expiry, target_strike, 'P', 'SMART', tradingClass=t_class)
            contracts_by_expiry[expiry] = (call, put)
            to_qualify.extend([call, put])

            # OTM wing contracts for 25-delta skew
            if spot > 0 and est_annual_vol > 0:
                dte = calc_dte(expiry)
                offset = calc_wing_offset(spot, est_annual_vol, dte)
                put_wing_strike = target_strike - offset
                call_wing_strike = target_strike + offset
                otm_put = Option(UNDERLYING, expiry, put_wing_strike, 'P', 'SMART', tradingClass=t_class)
                otm_call = Option(UNDERLYING, expiry, call_wing_strike, 'C', 'SMART', tradingClass=t_class)
                wing_contracts_by_expiry[expiry] = (otm_put, otm_call, put_wing_strike, call_wing_strike)
                to_qualify.extend([otm_put, otm_call])

        await ib.qualifyContractsAsync(*to_qualify)
        gticks = '100,101,104,106'

        for expiry, (call_c, put_c) in contracts_by_expiry.items():
            if call_c.conId <= 0 or put_c.conId <= 0:
                log.warning(f"Skipping {expiry}: ATM qualification failed")
                continue
            c_t = ib.reqMktData(call_c, gticks, False, False)
            p_t = ib.reqMktData(put_c, gticks, False, False)
            self.active_tickers.extend([c_t, p_t])
            self.tickers_map[expiry] = {'call': c_t, 'put': p_t}

        for expiry, (otm_put_c, otm_call_c, ps, cs) in wing_contracts_by_expiry.items():
            if otm_put_c.conId <= 0 or otm_call_c.conId <= 0:
                log.warning(f"Skipping wings {expiry}: qualification failed (P:{ps} C:{cs})")
                continue
            p_t = ib.reqMktData(otm_put_c, gticks, False, False)
            c_t = ib.reqMktData(otm_call_c, gticks, False, False)
            self.active_tickers.extend([p_t, c_t])
            self.wing_tickers_map[expiry] = {
                'otm_put': p_t, 'otm_call': c_t,
                'put_strike': ps, 'call_strike': cs,
            }
            log.info(f"Wings {expiry}: P@{ps} / C@{cs}")

        self.strike = target_strike
        await asyncio.sleep(1)
        return True

    def _cancel_all(self, ib):
        for t in self.active_tickers:
            try:
                ib.cancelMktData(t.contract)
            except Exception:
                pass
        self.active_tickers.clear()
        self.tickers_map.clear()
        self.wing_tickers_map.clear()
        self.strike = 0


# ============================================================
# SPOT PRICE
# ============================================================

def compute_spot(spx_ticker, es_ticker):
    spx_last = safe_float(spx_ticker.last)
    spx_close = safe_float(spx_ticker.close)
    es_last = safe_float(es_ticker.last)
    es_close = safe_float(es_ticker.close)
    if is_regular_session():
        spot = spx_last if spx_last > 0 else spx_close
        return spot, "Live SPX"
    if spx_close > 0 and es_last > 0 and es_close > 0:
        delta = es_last - es_close
        return spx_close + delta, f"Implied (ES {delta:+.2f})"
    if spx_close > 0:
        return spx_close, "SPX Close (waiting for ES)"
    return 0.0, "No Data"


# ============================================================
# BUILD TABLE ROW
# ============================================================

def build_row(expiry, strike, call_ticker, put_ticker, cur_spot, wing_data=None, open_straddle=0):
    cb, ca, cm = safe_bid_ask(call_ticker)
    pb, pa, pm = safe_bid_ask(put_ticker)
    straddle_price = cm + pm

    c_iv, c_gamma, c_theta = get_greeks(call_ticker)
    p_iv, p_gamma, p_theta = get_greeks(put_ticker)

    avg_iv = (c_iv + p_iv) / 2 if (c_iv > 0 and p_iv > 0) else (c_iv or p_iv)
    straddle_gamma = c_gamma + p_gamma
    straddle_theta = c_theta + p_theta

    dte = calc_dte(expiry)
    exp_fmt = f"{expiry[:4]}-{expiry[4:6]}-{expiry[6:]}"

    # Implied move % (straddle as pct of spot)
    impl_move_pct = (straddle_price / cur_spot * 100) if (cur_spot > 0 and straddle_price > 0) else 0
    # Implied move in SPX points (85% expected move rule)
    impl_move_pts = straddle_price * 0.85 if straddle_price > 0 else 0

    # Breakeven points
    if straddle_price > 0 and strike > 0:
        breakevens = f"{strike - straddle_price:.0f} / {strike + straddle_price:.0f}"
    else:
        breakevens = "-"

    # Straddle decay from session open
    decay_dollars = 0.0
    decay_pct = 0.0
    if open_straddle > 0 and straddle_price > 0:
        decay_dollars = open_straddle - straddle_price
        decay_pct = (decay_dollars / open_straddle) * 100

    # P/C Skew: 25-delta risk reversal (OTM put IV - OTM call IV)
    # Positive = puts vol richer than calls (normal equity skew)
    skew = 0
    if wing_data:
        wing_put_iv = get_greeks(wing_data['otm_put'])[0]
        wing_call_iv = get_greeks(wing_data['otm_call'])[0]
        if wing_put_iv > 0 and wing_call_iv > 0:
            skew = (wing_put_iv - wing_call_iv) * 100

    # Events on this expiry date
    events = get_events_on_date(expiry)
    event_str = " | ".join(events) if events else ""

    return {
        "DTE": dte,
        "Expiry": exp_fmt,
        "Strike": strike,
        "Call Bid/Ask": f"{cb:.2f} / {ca:.2f}",
        "Put Bid/Ask": f"{pb:.2f} / {pa:.2f}",
        "Straddle Price": round(straddle_price, 2),
        "Decay $": round(decay_dollars, 2),
        "Decay %": round(decay_pct, 2),
        "Breakevens": breakevens,
        "Impl Move %": round(impl_move_pct, 2),
        "Impl Move Pts": round(impl_move_pts, 1),
        "P/C Skew": round(skew, 2),
        "IV": avg_iv,
        "Gamma": round(straddle_gamma, 4),
        "Theta": round(straddle_theta, 2),
        "Events": event_str,
    }


# ============================================================
# MAIN IBKR LOOP (auto-reconnect)
# ============================================================

async def ib_loop():
    while True:
        ib = IB()
        try:
            log.info(f"Connecting to IBKR on port {PORT}...")
            await ib.connectAsync('127.0.0.1', PORT, clientId=CLIENT_ID)
            log.info("Connected to IBKR.")
            ib.reqMarketDataType(4)

            spx = Index(UNDERLYING, EXCHANGE)
            await ib.qualifyContractsAsync(spx)
            spx_ticker = ib.reqMktData(spx, '', False, False)

            es = ContFuture('ES', 'CME')
            await ib.qualifyContractsAsync(es)
            es_ticker = ib.reqMktData(es, '', False, False)

            vix = Index('VIX', 'CBOE')
            await ib.qualifyContractsAsync(vix)
            vix_ticker = ib.reqMktData(vix, '', False, False)

            log.info("Waiting for initial SPX data...")
            for attempt in range(120):
                if safe_float(spx_ticker.last) > 0 or safe_float(spx_ticker.close) > 0:
                    break
                if attempt % 10 == 9:
                    log.warning(f"Still waiting for SPX data ({attempt+1}s).")
                await asyncio.sleep(1)
            log.info("SPX data received. Entering main loop.")

            expiries_info = await get_next_n_spxw_expiries(ib, spx, n=NUM_EXPIRIES)
            last_expiry_refresh = datetime.now()
            subs = StraddleSubscription()

            while ib.isConnected():
                try:
                    now = datetime.now()
                    today = now.date()

                    # Day rollover
                    with state_lock:
                        if shared_state["session_date"] != today:
                            log.info(f"New session day: {today}")
                            shared_state["session_date"] = today
                            shared_state["session_open_spot"] = None
                            shared_state["open_straddles"] = {}
                            shared_state["prior_close"] = None
                            shared_state["straddle_history"] = []

                    # Refresh expiries
                    if (now - last_expiry_refresh) > timedelta(minutes=EXPIRY_REFRESH_MINUTES):
                        log.info("Refreshing expiry list...")
                        expiries_info = await get_next_n_spxw_expiries(ib, spx, n=NUM_EXPIRIES)
                        last_expiry_refresh = now
                        subs._cancel_all(ib)

                    # Spot price
                    cur_spot, pricing_mode = compute_spot(spx_ticker, es_ticker)
                    target_strike = get_nearest_strike(cur_spot)

                    # Capture prior close + session open
                    spx_close = safe_float(spx_ticker.close)
                    with state_lock:
                        if shared_state["prior_close"] is None and spx_close > 0:
                            shared_state["prior_close"] = spx_close
                            log.info(f"Prior close: {spx_close:.2f}")
                        if shared_state["session_open_spot"] is None and cur_spot > 0 and is_regular_session():
                            shared_state["session_open_spot"] = cur_spot
                            log.info(f"Session open spot: {cur_spot:.2f}")

                    # VIX for wing strike estimation
                    cur_vix_for_wings = safe_float(vix_ticker.last) or safe_float(vix_ticker.close)
                    est_vol = cur_vix_for_wings / 100 if cur_vix_for_wings > 0 else 0.15

                    # Re-subscribe if strike moved (includes ATM + 25-delta wings)
                    changed = await subs.update(ib, target_strike, expiries_info,
                                                spot=cur_spot, est_annual_vol=est_vol)
                    if changed:
                        with state_lock:
                            shared_state["status"] = f"Switched to {target_strike}"

                    # Build table rows
                    table_data = []
                    for expiry, _ in expiries_info:
                        if expiry not in subs.tickers_map:
                            continue
                        data = subs.tickers_map[expiry]
                        wing_data = subs.wing_tickers_map.get(expiry)

                        with state_lock:
                            if (expiry not in shared_state["open_straddles"]
                                    and is_regular_session()):
                                # Defer capture until we compute straddle price below
                                pass
                            open_straddle = shared_state["open_straddles"].get(expiry, 0)

                        row = build_row(
                            expiry, subs.strike,
                            data['call'], data['put'],
                            cur_spot,
                            wing_data=wing_data,
                            open_straddle=open_straddle,
                        )
                        table_data.append(row)

                        with state_lock:
                            if (expiry not in shared_state["open_straddles"]
                                    and row["Straddle Price"] > 0
                                    and is_regular_session()):
                                shared_state["open_straddles"][expiry] = row["Straddle Price"]
                                log.info(f"Open straddle {expiry}: ${row['Straddle Price']:.2f}")

                    table_data.sort(key=lambda r: r["DTE"])

                    # Collect straddle history for intraday decay chart (0-1 DTE)
                    if is_regular_session():
                        ts = datetime.now(ET).strftime("%H:%M")
                        with state_lock:
                            for row in table_data:
                                if row["DTE"] <= 1 and row["Straddle Price"] > 0:
                                    shared_state["straddle_history"].append({
                                        "timestamp": ts,
                                        "dte": row["DTE"],
                                        "price": row["Straddle Price"],
                                        "expiry": row["Expiry"],
                                    })

                    # VIX price
                    vix_last = safe_float(vix_ticker.last)
                    vix_close = safe_float(vix_ticker.close)
                    cur_vix = vix_last if vix_last > 0 else vix_close

                    with state_lock:
                        if shared_state["vix_prior_close"] is None and vix_close > 0:
                            shared_state["vix_prior_close"] = vix_close
                        shared_state["df"] = pd.DataFrame(table_data)
                        shared_state["spot_price"] = cur_spot
                        shared_state["active_strike"] = subs.strike
                        shared_state["status"] = "Live"
                        shared_state["mode"] = pricing_mode
                        shared_state["last_update"] = now.strftime("%H:%M:%S")
                        shared_state["vix_price"] = cur_vix

                    await asyncio.sleep(BACKEND_INTERVAL_SEC)

                except Exception as loop_e:
                    log.error(f"Loop error: {loop_e}", exc_info=True)
                    await asyncio.sleep(5)

        except Exception as e:
            log.error(f"Connection lost: {e}")
            with state_lock:
                shared_state["status"] = "Reconnecting..."
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

        log.info(f"Retrying in {RECONNECT_DELAY_SEC}s...")
        await asyncio.sleep(RECONNECT_DELAY_SEC)


def start_ib_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ib_loop())


# ============================================================
# CHART BUILDERS
# ============================================================

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(color="#8b949e", size=13, family="Consolas, monospace"),
    margin=dict(l=60, r=20, t=50, b=45),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=13)),
    height=420,
    title_font=dict(size=16, color="#e6edf3"),
)

TRACE_COLORS = [
    '#3fb950', '#f47067', '#58a6ff', '#d29922', '#bc8cff',
    '#7ee787', '#f0883e', '#56d4dd', '#db61a2', '#79c0ff',
]


def build_term_structure_chart(df):
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, title="Vol Term Structure (waiting for data)")
        return fig
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df["DTE"], y=df["IV"],
        mode='lines+markers', name='IV',
        line=dict(color='#d29922', width=3), marker=dict(size=10),
    ), secondary_y=False)
    bar_colors = ['#58a6ff'] * len(df)
    for i, evt in enumerate(df["Events"]):
        if evt:
            bar_colors[i] = '#f0883e'
    fig.add_trace(go.Bar(
        x=df["DTE"], y=df["Impl Move %"],
        name='Impl Move %',
        marker=dict(color=bar_colors, opacity=0.6),
        text=[f"{v:.1f}%" for v in df["Impl Move %"]],
        textposition='outside',
        textfont=dict(size=13, color='#8b949e'),
    ), secondary_y=True)
    for _, row in df.iterrows():
        if row.get("Events"):
            short = row["Events"].split("|")[0].strip().split("(")[0].strip()
            fig.add_annotation(
                x=row["DTE"], y=row["IV"],
                text=f"! {short}",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor="#f0883e",
                font=dict(size=12, color="#f0883e"),
                ax=0, ay=-30,
            )
    fig.update_layout(
        **CHART_LAYOUT,
        title="Vol Term Structure (0-8 DTE)",
        xaxis_title="Days to Expiration",
        xaxis=dict(tickmode='array', tickvals=df["DTE"].tolist()),
    )
    fig.update_yaxes(title_text="IV", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="Implied Move %", ticksuffix="%", secondary_y=True)
    return fig


def build_straddle_history_chart(history):
    if not history:
        fig = go.Figure()
        fig.update_layout(**CHART_LAYOUT, title="Intraday Straddle Decay (waiting for data)")
        return fig
    df_hist = pd.DataFrame(history)
    fig = go.Figure()
    for dte_val in sorted(df_hist["dte"].unique()):
        subset = df_hist[df_hist["dte"] == dte_val]
        label = f"{int(dte_val)}DTE ({subset.iloc[0]['expiry']})"
        color = '#f47067' if dte_val == 0 else '#58a6ff'
        fig.add_trace(go.Scatter(
            x=subset["timestamp"], y=subset["price"],
            mode='lines+markers', name=label,
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Intraday Straddle Decay (0-1 DTE)",
        xaxis_title="Time (ET)",
        yaxis_title="Straddle Price ($)",
        yaxis=dict(tickprefix="$"),
    )
    return fig


# ============================================================
# DASH UI
# ============================================================

# Google Font: JetBrains Mono for the terminal/trading desk feel
GOOGLE_FONT = "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap"

app = dash.Dash(__name__, external_stylesheets=[THEME, GOOGLE_FONT])
app.title = "SPX Straddle Monitor"

# Inject global CSS for larger base font and custom scrollbar
app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
    body {
        font-family: 'IBM Plex Sans', sans-serif !important;
        background: #0a0e14 !important;
    }
    .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 15px !important;
        padding: 10px 14px !important;
        line-height: 1.4 !important;
    }
    .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 12px 14px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    .card { border-radius: 10px !important; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>'''

# Card styling helper
CARD_STYLE = {
    "borderRadius": "10px",
    "border": "1px solid #21262d",
    "background": "linear-gradient(135deg, #0d1117 0%, #161b22 100%)",
}
CARD_LABEL = {"fontSize": "12px", "color": "#8b949e", "fontWeight": "600",
              "letterSpacing": "0.8px", "textTransform": "uppercase",
              "fontFamily": "IBM Plex Sans, sans-serif", "marginBottom": "4px"}
CARD_VALUE = {"fontFamily": "JetBrains Mono, monospace", "fontWeight": "700",
              "marginBottom": "0", "letterSpacing": "-0.5px"}

TABLE_STYLE = {
    'style_header': {
        'backgroundColor': '#161b22', 'color': '#8b949e',
        'fontWeight': '600', 'border': '1px solid #21262d',
        'borderBottom': '2px solid #30363d',
    },
    'style_data': {
        'backgroundColor': '#0d1117', 'color': '#e6edf3',
        'border': '1px solid #21262d',
    },
    'style_cell': {
        'minWidth': '70px',
    },
    'style_data_conditional': [
        {'if': {'row_index': 'odd'}, 'backgroundColor': '#111820'},
        # Straddle price - bold green accent
        {'if': {'column_id': 'Straddle Price'},
         'backgroundColor': '#0a2618', 'color': '#3fb950', 'fontWeight': '700',
         'borderLeft': '3px solid #238636', 'borderRight': '3px solid #238636'},
        # Decay positive (seller profit) = green
        {'if': {'column_id': 'Decay $', 'filter_query': '{Decay $} > 0'},
         'color': '#3fb950', 'fontWeight': '700'},
        {'if': {'column_id': 'Decay %', 'filter_query': '{Decay %} > 0'},
         'color': '#3fb950'},
        # Decay negative (straddle expansion) = red
        {'if': {'column_id': 'Decay $', 'filter_query': '{Decay $} < 0'},
         'color': '#f47067', 'fontWeight': '700'},
        {'if': {'column_id': 'Decay %', 'filter_query': '{Decay %} < 0'},
         'color': '#f47067'},
        # P/C Skew colors
        {'if': {'column_id': 'P/C Skew', 'filter_query': '{P/C Skew} > 1'},
         'color': '#f0883e'},
        {'if': {'column_id': 'P/C Skew', 'filter_query': '{P/C Skew} < -1'},
         'color': '#58a6ff'},
        # Elevated IV
        {'if': {'column_id': 'IV', 'filter_query': '{IV} > 0.20'},
         'backgroundColor': '#2a1800', 'color': '#f0883e', 'fontWeight': '700'},
        # Events
        {'if': {'column_id': 'Events', 'filter_query': '{Events} ne ""'},
         'backgroundColor': '#1c1100', 'color': '#d29922', 'fontWeight': '600'},
        # 0 DTE row highlight
        {'if': {'column_id': 'DTE', 'filter_query': '{DTE} = 0'},
         'backgroundColor': '#280020', 'color': '#f778ba', 'fontWeight': '700'},
    ],
}

COLUMNS = [
    {'name': 'DTE', 'id': 'DTE', 'type': 'numeric'},
    {'name': 'Expiry', 'id': 'Expiry'},
    {'name': 'Strike', 'id': 'Strike', 'type': 'numeric'},
    {'name': 'Call Bid/Ask', 'id': 'Call Bid/Ask'},
    {'name': 'Put Bid/Ask', 'id': 'Put Bid/Ask'},
    {'name': 'Straddle', 'id': 'Straddle Price', 'type': 'numeric',
     'format': dash_table.FormatTemplate.money(2)},
    {'name': 'Decay $', 'id': 'Decay $', 'type': 'numeric',
     'format': {'specifier': '+.2f'}},
    {'name': 'Decay %', 'id': 'Decay %', 'type': 'numeric',
     'format': {'specifier': '+.1f'}},
    {'name': 'Impl Pts', 'id': 'Impl Move Pts', 'type': 'numeric',
     'format': {'specifier': '.1f'}},
    {'name': 'P/C Skew', 'id': 'P/C Skew', 'type': 'numeric',
     'format': {'specifier': '+.2f'}},
    {'name': 'IV', 'id': 'IV', 'type': 'numeric',
     'format': {'specifier': '.1%'}},
    {'name': 'Events', 'id': 'Events'},
]

app.layout = dbc.Container([
    dcc.Interval(id='update-interval', interval=UI_INTERVAL_MS, n_intervals=0),

    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("SPX", style={"color": "#3fb950", "fontWeight": "700"}),
                html.Span(" STRADDLE MONITOR", style={"color": "#8b949e", "fontWeight": "400"}),
            ], style={"fontSize": "28px", "fontFamily": "IBM Plex Sans, sans-serif",
                       "textAlign": "center", "letterSpacing": "2px", "padding": "8px 0"}),
        ], width=12),
    ]),

    # Divider
    html.Hr(style={"borderColor": "#21262d", "margin": "4px 0 16px 0"}),

    # Info Cards Row
    dbc.Row([
        # Spot Price + Day Change
        dbc.Col(html.Div([
            html.Div("SPX SPOT", style=CARD_LABEL),
            html.Div(id="spot-display", style={**CARD_VALUE, "fontSize": "36px", "color": "#3fb950"}),
            html.Div(id="session-move-display",
                     style={**CARD_VALUE, "fontSize": "18px", "marginTop": "4px"}),
            html.Div(id="mode-display", style={"fontSize": "12px", "color": "#484f58",
                                                 "fontFamily": "JetBrains Mono, monospace"}),
            html.Div(id="session-open-display",
                     style={"fontSize": "12px", "color": "#484f58",
                            "fontFamily": "JetBrains Mono, monospace"}),
        ], style={**CARD_STYLE, "padding": "16px 20px"}), width=3),

        # Strike
        dbc.Col(html.Div([
            html.Div("ACTIVE STRIKE", style=CARD_LABEL),
            html.Div(id="strike-display", style={**CARD_VALUE, "fontSize": "32px", "color": "#58a6ff"}),
        ], style={**CARD_STYLE, "padding": "16px 20px"}), width=2),

        # VIX
        dbc.Col(html.Div([
            html.Div("VIX", style=CARD_LABEL),
            html.Div(id="vix-display",
                     style={**CARD_VALUE, "fontSize": "32px"}),
            html.Div(id="vix-change-display",
                     style={**CARD_VALUE, "fontSize": "18px", "marginTop": "4px"}),
            html.Div(id="vol-premium-display", style={"marginTop": "6px"}),
        ], style={**CARD_STYLE, "padding": "16px 20px"}), width=3),

        # Status
        dbc.Col(html.Div([
            html.Div("STATUS", style=CARD_LABEL),
            html.Div(id="status-display",
                     style={**CARD_VALUE, "fontSize": "20px", "color": "#d29922"}),
            html.Div(id="time-display",
                     style={"fontSize": "12px", "color": "#484f58",
                            "fontFamily": "JetBrains Mono, monospace"}),
        ], style={**CARD_STYLE, "padding": "16px 20px"}), width=2),

        # Events
        dbc.Col(html.Div([
            html.Div("EVENTS", style=CARD_LABEL),
            html.Div(id="today-events-display",
                     style={**CARD_VALUE, "fontSize": "16px", "color": "#d29922"}),
        ], style={**CARD_STYLE, "padding": "16px 20px"}), width=2),
    ], className="mb-4 g-3"),

    # Data Table
    dbc.Row([dbc.Col([
        html.Div([
            dash_table.DataTable(
                id='live-table',
                columns=COLUMNS,
                style_table={'overflowX': 'auto', 'borderRadius': '8px',
                             'border': '1px solid #21262d'},
                sort_action='native',
                tooltip_header={
                    'P/C Skew': '25Δ Risk Reversal: OTM put IV minus OTM call IV (vol pts). Positive = puts richer',
                    'Decay $': 'Straddle price change from session open (positive = decay / seller profit)',
                    'Decay %': 'Percentage decay from session-open straddle price',
                    'Impl Pts': 'Expected move in SPX points (straddle × 0.85)',
                },
                tooltip_delay=0, tooltip_duration=None,
                **TABLE_STYLE,
            ),
        ]),
    ], width=12)], className="mb-4"),

    # Charts
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Graph(id='term-structure-chart', config={'displayModeBar': False}),
        ], style={**CARD_STYLE, "padding": "8px", "overflow": "hidden"}), width=6),
        dbc.Col(html.Div([
            dcc.Graph(id='straddle-history-chart', config={'displayModeBar': False}),
        ], style={**CARD_STYLE, "padding": "8px", "overflow": "hidden"}), width=6),
    ], className="mb-4"),

], fluid=True, style={"backgroundColor": "#0a0e14", "minHeight": "100vh",
                       "padding": "16px 24px"})


# ============================================================
# CALLBACKS
# ============================================================

@app.callback(
    [
        Output('live-table', 'data'),
        Output('spot-display', 'children'),
        Output('mode-display', 'children'),
        Output('strike-display', 'children'),
        Output('session-move-display', 'children'),
        Output('session-open-display', 'children'),
        Output('vix-display', 'children'),
        Output('vix-change-display', 'children'),
        Output('vol-premium-display', 'children'),
        Output('status-display', 'children'),
        Output('time-display', 'children'),
        Output('today-events-display', 'children'),
        Output('term-structure-chart', 'figure'),
        Output('straddle-history-chart', 'figure'),
    ],
    [Input('update-interval', 'n_intervals')],
)
def update_dashboard(n):
    with state_lock:
        df = shared_state["df"].copy()
        spot = shared_state["spot_price"]
        mode = shared_state["mode"]
        strike = shared_state["active_strike"]
        status = shared_state["status"]
        last_update = shared_state["last_update"]
        session_open = shared_state["session_open_spot"]
        prior_close = shared_state["prior_close"]
        vix = shared_state["vix_price"]
        vix_prior_close = shared_state["vix_prior_close"]
        straddle_history = list(shared_state["straddle_history"])

    # Session move: current spot vs prior close
    if prior_close and prior_close > 0 and spot > 0:
        mv = spot - prior_close
        mv_pct = (mv / prior_close) * 100
        color = "#3fb950" if mv >= 0 else "#f47067"
        session_move = html.Span(f"{mv:+.2f}  ({mv_pct:+.2f}%)",
                                 style={"color": color, "fontFamily": "JetBrains Mono, monospace",
                                        "fontSize": "28px", "fontWeight": "700"})
    else:
        session_move = html.Span("-", style={"color": "#484f58", "fontSize": "28px",
                                              "fontFamily": "JetBrains Mono, monospace"})

    # Session open display
    if session_open and session_open > 0:
        session_open_display = html.Span(
            f"Open: {session_open:,.2f}",
            style={"color": "#484f58", "fontFamily": "JetBrains Mono, monospace",
                   "fontSize": "12px"})
    else:
        session_open_display = ""

    # VIX display with color coding
    if vix > 0:
        if vix >= 30:
            vix_color = "#f47067"
        elif vix >= 20:
            vix_color = "#f0883e"
        else:
            vix_color = "#3fb950"
        vix_display = html.Span(f"{vix:.2f}", style={"color": vix_color})
    else:
        vix_display = html.Span("-", style={"color": "#484f58"})

    # VIX day change
    if vix_prior_close and vix_prior_close > 0 and vix > 0:
        vix_mv = vix - vix_prior_close
        vix_mv_pct = (vix_mv / vix_prior_close) * 100
        vix_chg_color = "#f47067" if vix_mv >= 0 else "#3fb950"  # red up, green down
        vix_change_display = html.Span(
            f"{vix_mv:+.2f}  ({vix_mv_pct:+.2f}%)",
            style={"color": vix_chg_color, "fontFamily": "JetBrains Mono, monospace",
                   "fontSize": "18px", "fontWeight": "700"})
    else:
        vix_change_display = html.Span("-", style={"color": "#484f58",
                                                     "fontFamily": "JetBrains Mono, monospace"})

    # Vol premium: 0DTE ATM IV vs VIX
    vol_premium_display = ""
    if not df.empty and vix > 0:
        dte0_rows = df[df["DTE"] == 0]
        if not dte0_rows.empty:
            atm_iv = dte0_rows.iloc[0]["IV"]
            if atm_iv > 0:
                atm_iv_pct = atm_iv * 100
                premium = atm_iv_pct - vix
                ratio = atm_iv_pct / vix
                if premium > 5:
                    prem_color, prem_label = "#3fb950", "RICH"
                elif premium > 0:
                    prem_color, prem_label = "#d29922", "FAIR+"
                elif premium > -5:
                    prem_color, prem_label = "#8b949e", "FAIR"
                else:
                    prem_color, prem_label = "#f47067", "CHEAP"
                vol_premium_display = html.Div([
                    html.Span("VOL PREM ", style={"fontSize": "10px", "color": "#484f58",
                                                    "textTransform": "uppercase",
                                                    "letterSpacing": "0.5px"}),
                    html.Span(f"{ratio:.2f}x ",
                              style={"fontSize": "16px", "color": prem_color,
                                     "fontWeight": "700",
                                     "fontFamily": "JetBrains Mono, monospace"}),
                    html.Span(prem_label,
                              style={"fontSize": "12px", "color": prem_color,
                                     "fontWeight": "700"}),
                ], style={"marginTop": "4px"})

    today_evts, tomorrow_evts = get_upcoming_events()
    event_parts = []
    if today_evts:
        event_parts.append(html.Div(f"Today: {' / '.join(today_evts)}"))
    if tomorrow_evts:
        event_parts.append(html.Div(f"Tmrw: {' / '.join(tomorrow_evts)}",
                                    style={"color": "#8b949e", "fontSize": "14px"}))
    today_events_text = event_parts if event_parts else "No events"

    fig_term = build_term_structure_chart(df)
    fig_history = build_straddle_history_chart(straddle_history)

    return (
        df.to_dict('records'),
        f"{spot:,.2f}",
        mode,
        str(strike),
        session_move,
        session_open_display,
        vix_display,
        vix_change_display,
        vol_premium_display,
        status,
        f"Updated: {last_update}",
        today_events_text,
        fig_term,
        fig_history,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    log.info("Starting IBKR data thread...")
    t = threading.Thread(target=start_ib_thread, daemon=True)
    t.start()

    url = "http://127.0.0.1:8050"
    log.info(f"Opening browser: {url}")
    webbrowser.open(url)

    app.run(debug=False, port=8050)
