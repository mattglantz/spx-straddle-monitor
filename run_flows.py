#!/usr/bin/env python3
"""
ES Structural Flow Monitor — Entry Point

Connects to IBKR, fetches historical data, and launches a live
dashboard showing all five structural flow signals.

Usage:
    python run_flows.py               # Default (paper port 7497)
    python run_flows.py --port 7496   # Live TWS port
    python run_flows.py --no-browser  # Don't auto-open browser

Dashboard runs at http://127.0.0.1:8060
"""

import asyncio
import argparse
import logging
import threading
import webbrowser
from datetime import datetime

# Ensure event loop exists before ib_async import
asyncio.set_event_loop(asyncio.new_event_loop())

from ib_async import IB

from flow_module.config import (
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, DASH_PORT, FlowConfig, ET,
)
from flow_module.data import MarketDataStore
from flow_module.flows import FlowAggregator
from flow_module.dashboard import create_app, set_snapshot
import flow_module.config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("ib_async").setLevel(logging.ERROR)

log = logging.getLogger("run_flows")

# Refresh interval for flow calculations (seconds)
REFRESH_INTERVAL = 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Structural Flow Monitor"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help=f"IBKR API port (default: {IBKR_PORT})"
    )
    parser.add_argument(
        "--client-id", type=int, default=None,
        help="IBKR client ID"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open browser"
    )
    parser.add_argument(
        "--refresh", type=int, default=REFRESH_INTERVAL,
        help=f"Refresh interval in seconds (default: {REFRESH_INTERVAL})"
    )
    return parser.parse_args()


async def flow_loop(args):
    """Main async loop: connect, fetch history, run flow analysis."""
    port = args.port or IBKR_PORT
    client_id = args.client_id or IBKR_CLIENT_ID
    refresh = args.refresh

    config = FlowConfig()
    store = MarketDataStore()
    aggregator = FlowAggregator(config)

    while True:
        ib = IB()
        try:
            log.info(f"Connecting to IBKR {IBKR_HOST}:{port} clientId={client_id}...")
            await ib.connectAsync(IBKR_HOST, port, clientId=client_id)
            log.info("Connected to IBKR.")

            ib.reqMarketDataType(4)  # delayed/frozen OK for daily analysis

            # Fetch historical data
            es_ticker, vix_ticker = await store.connect_and_fetch(
                ib, lookback_days=config.history_lookback_days)

            # Wait for live data
            log.info("Waiting for live data...")
            for attempt in range(30):
                store.update_live(es_ticker, vix_ticker)
                if store.live_price > 0:
                    break
                if attempt % 10 == 9:
                    log.warning(f"Still waiting ({attempt+1}s)...")
                await asyncio.sleep(1)

            log.info(f"Live ES: {store.live_price:.2f}, VIX: {store.live_vix:.1f}")
            log.info(f"Historical bars: {len(store.daily_bars)}")
            log.info("Entering flow analysis loop...")

            # Main loop
            while ib.isConnected():
                # Update live prices
                store.update_live(es_ticker, vix_ticker)
                if store.live_price > 0:
                    store.append_today(store.live_price)

                # Run flow analysis
                snapshot = aggregator.evaluate(store)
                set_snapshot(snapshot)

                # Print summary to console
                log.info(f"\n{'='*60}")
                log.info(f"  {snapshot.headline}")
                log.info(f"  Rebalance: {snapshot.rebalance.signal:+.0f} "
                         f"({snapshot.rebalance.flow_direction})")
                log.info(f"  OpEx:      {snapshot.opex.phase} "
                         f"(next: {snapshot.opex.next_opex_date})")
                log.info(f"  CTA:       {snapshot.cta.signal:+.0f} "
                         f"({snapshot.cta.flow_direction})")
                log.info(f"  Vol-Ctrl:  {snapshot.vol_control.signal:+.0f} "
                         f"({snapshot.vol_control.flow_direction})")
                log.info(f"  Buyback:   {snapshot.buyback.signal:+.0f} "
                         f"({snapshot.buyback.blackout_phase})")
                log.info(f"  NET:       {snapshot.net_signal:+.0f} "
                         f"({snapshot.net_direction}, {snapshot.conviction})")
                log.info(f"{'='*60}")

                # Sleep with periodic live-price updates
                for _ in range(refresh * 2):
                    store.update_live(es_ticker, vix_ticker)
                    if store.live_price > 0:
                        store.append_today(store.live_price)
                    await asyncio.sleep(0.5)

        except Exception as e:
            log.error(f"Connection error: {e}", exc_info=True)
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

        log.info("Reconnecting in 10s...")
        await asyncio.sleep(10)


def start_flow_thread(args):
    """Run the flow loop in a background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(flow_loop(args))


def main():
    args = parse_args()

    if args.port is not None:
        cfg.IBKR_PORT = args.port
    if args.client_id is not None:
        cfg.IBKR_CLIENT_ID = args.client_id

    log.info("=" * 60)
    log.info("  ES STRUCTURAL FLOW MONITOR")
    log.info("=" * 60)
    log.info(f"  Tracking: Rebalancing | OpEx | CTA | Vol-Control | Buybacks")
    log.info(f"  Dashboard: http://127.0.0.1:{DASH_PORT}")
    log.info("=" * 60)

    # Start IBKR data thread
    data_thread = threading.Thread(target=start_flow_thread, args=(args,),
                                   daemon=True)
    data_thread.start()

    # Create and launch Dash app
    app = create_app()

    if not args.no_browser:
        webbrowser.open(f"http://127.0.0.1:{DASH_PORT}")

    app.run(host="0.0.0.0", port=DASH_PORT, debug=False)


if __name__ == "__main__":
    main()
