#!/usr/bin/env python3
"""
ES Futures Gamma-Regime Adaptive Trading Bot — Entry Point

Usage:
    python run_bot.py               # Run in paper mode (default)
    python run_bot.py --live        # Run in LIVE mode (real orders!)
    python run_bot.py --port 7496   # Override IBKR port
    python run_bot.py --status 30   # Print status every 30 seconds

Requires:
    - IBKR TWS or Gateway running with API enabled
    - Python 3.10+
    - pip install ib_async
"""

import asyncio
import argparse
import logging
import signal
import sys

# Ensure event loop exists before ib_async import
asyncio.set_event_loop(asyncio.new_event_loop())

from trading_bot.config import BotConfig, IBKR_PORT
from trading_bot.bot import TradingBot
import trading_bot.config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("ib_async").setLevel(logging.ERROR)

log = logging.getLogger("run_bot")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ES Futures Gamma-Regime Adaptive Trading Bot"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable LIVE trading (default is paper mode)"
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
        "--max-contracts", type=int, default=None,
        help="Maximum position size in contracts"
    )
    parser.add_argument(
        "--max-daily-loss", type=float, default=None,
        help="Maximum daily loss in dollars before circuit breaker"
    )
    parser.add_argument(
        "--status", type=int, default=60,
        help="Print status every N seconds (default: 60)"
    )
    return parser.parse_args()


async def run_with_status(bot: TradingBot, status_interval: int):
    """Run the bot with periodic status printing."""
    async def status_printer():
        while bot._running:
            await asyncio.sleep(status_interval)
            if bot.market_data.last_price > 0:
                log.info("\n" + bot.print_status())
    status_task = asyncio.create_task(status_printer())
    try:
        await bot.run()
    finally:
        status_task.cancel()


def main():
    args = parse_args()

    # Build config from arguments
    config = BotConfig()

    if args.live:
        config.paper_mode = False
        log.warning("=" * 60)
        log.warning("  LIVE TRADING MODE — REAL ORDERS WILL BE PLACED")
        log.warning("=" * 60)
        confirm = input("Type 'YES' to confirm live trading: ")
        if confirm.strip() != "YES":
            log.info("Live trading not confirmed. Exiting.")
            sys.exit(0)

    if args.port is not None:
        cfg.IBKR_PORT = args.port
    if args.client_id is not None:
        cfg.IBKR_CLIENT_ID = args.client_id
    if args.max_contracts is not None:
        config.risk.max_position_contracts = args.max_contracts
    if args.max_daily_loss is not None:
        config.risk.max_daily_loss_dollars = args.max_daily_loss

    bot = TradingBot(config)

    # Graceful shutdown on Ctrl+C
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def handle_signal(sig, frame):
        log.info(f"Signal {sig} received, shutting down...")
        loop.create_task(bot.shutdown())

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(run_with_status(bot, args.status))
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — shutting down...")
        loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()
        log.info("Done.")


if __name__ == "__main__":
    main()
