"""
Main bot orchestrator — ties everything together.

Flow:
1. Connect to IBKR
2. Subscribe to ES futures real-time data + VIX
3. On each tick: update bars, VWAP, session levels
4. Every N seconds: run signals → strategy → risk → execution pipeline
5. Monitor fills, manage trailing stops
6. Log everything, track daily PnL
7. Auto-reconnect on disconnection

Can optionally read options-derived data (IV, skew) from the
existing straddle monitor's shared state, or collect it independently.
"""

import asyncio
import logging
import math
import json
from datetime import datetime, time
from pathlib import Path
from typing import Optional

from ib_async import IB, Index, ContFuture

from trading_bot.config import (
    ES_SYMBOL, ES_EXCHANGE, ES_POINT_VALUE, ET,
    BotConfig,
)
import trading_bot.config as cfg
from trading_bot.data import MarketData
from trading_bot.signals import SignalEngine, SignalSnapshot, Regime, event_name_today
from trading_bot.strategy import Strategy, Action, Side
from trading_bot.execution import ExecutionManager, OrderSet
from trading_bot.risk import RiskManager

log = logging.getLogger(__name__)

TRADE_LOG_PATH = Path(__file__).parent.parent / "trade_log.json"


def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return 0.0 if math.isnan(val) or math.isinf(val) else float(val)
    except (TypeError, ValueError):
        return 0.0


class TradingBot:
    """
    The main ES futures trading bot.
    """

    def __init__(self, config: Optional[BotConfig] = None):
        self.cfg = config or BotConfig()
        self.ib: Optional[IB] = None
        self.market_data = MarketData(
            fast_period=self.cfg.bars.fast_period,
            slow_period=self.cfg.bars.slow_period,
            max_bars=self.cfg.bars.lookback_bars,
            atr_period=self.cfg.risk.atr_period,
        )
        self.signal_engine = SignalEngine(self.cfg.regime, self.cfg.signals)
        self.strategy = Strategy(self.cfg.signals, self.cfg.risk)
        self.risk_manager = RiskManager(self.cfg.risk)
        self.execution: Optional[ExecutionManager] = None

        self._running = False
        self._session_date = None
        self._last_eval_time: Optional[datetime] = None

    async def run(self):
        """Main entry point. Connects and runs forever with auto-reconnect."""
        self._running = True
        log.info("=" * 60)
        log.info("ES FUTURES GAMMA-REGIME ADAPTIVE BOT")
        log.info("=" * 60)
        log.info(f"Paper mode: {self.cfg.paper_mode}")
        log.info(f"Max position: {self.cfg.risk.max_position_contracts} contracts")
        log.info(f"Max daily loss: ${self.cfg.risk.max_daily_loss_dollars}")
        log.info(f"Eval interval: {self.cfg.eval_interval_sec}s")
        log.info("=" * 60)

        while self._running:
            try:
                await self._connect_and_trade()
            except Exception as e:
                log.error(f"Connection error: {e}", exc_info=True)
            log.info("Reconnecting in 10s...")
            await asyncio.sleep(10)

    async def _connect_and_trade(self):
        """Single connection session."""
        self.ib = IB()
        try:
            log.info(f"Connecting to IBKR {cfg.IBKR_HOST}:{cfg.IBKR_PORT} clientId={cfg.IBKR_CLIENT_ID}...")
            await self.ib.connectAsync(cfg.IBKR_HOST, cfg.IBKR_PORT, clientId=cfg.IBKR_CLIENT_ID)
            log.info("Connected to IBKR.")

            self.ib.reqMarketDataType(1)  # live data for trading
            self.execution = ExecutionManager(self.ib)

            # Subscribe to ES futures
            es = ContFuture(ES_SYMBOL, ES_EXCHANGE)
            await self.ib.qualifyContractsAsync(es)
            es_ticker = self.ib.reqMktData(es, '', False, False)
            log.info(f"Subscribed to ES: {es}")

            # Subscribe to VIX
            vix = Index('VIX', 'CBOE')
            await self.ib.qualifyContractsAsync(vix)
            vix_ticker = self.ib.reqMktData(vix, '', False, False)
            log.info("Subscribed to VIX")

            # Subscribe to SPX for IV data (ATM options)
            spx = Index('SPX', 'CBOE')
            await self.ib.qualifyContractsAsync(spx)

            # Wait for initial ES data
            log.info("Waiting for ES data...")
            for attempt in range(60):
                if _safe_float(es_ticker.last) > 0 or _safe_float(es_ticker.close) > 0:
                    break
                if attempt % 10 == 9:
                    log.warning(f"Still waiting for ES data ({attempt+1}s)")
                await asyncio.sleep(1)

            initial_price = _safe_float(es_ticker.last) or _safe_float(es_ticker.close)
            if initial_price <= 0:
                log.error("Could not get initial ES price. Aborting session.")
                return
            log.info(f"Initial ES price: {initial_price:.2f}")

            # Qualify ES contract for execution
            await self.execution.qualify_contract()

            # Main loop
            log.info("Entering main trading loop...")
            await self._main_loop(es_ticker, vix_ticker)

        finally:
            # Clean disconnect
            if self.strategy.position is not None:
                log.warning("Disconnecting with open position! Consider flatting first.")
            try:
                self.ib.disconnect()
            except Exception:
                pass

    async def _main_loop(self, es_ticker, vix_ticker):
        """Core loop: ingest data, evaluate signals, make decisions."""
        while self.ib.isConnected() and self._running:
            try:
                now = datetime.now(ET)

                # Daily reset
                today = now.date()
                if self._session_date != today:
                    self._session_date = today
                    self.strategy.reset_daily()
                    event = event_name_today(now)
                    if event:
                        log.info(f"EVENT DAY: {event}")

                # Ingest latest tick data
                self._ingest_tick(es_ticker, vix_ticker, now)

                # Only evaluate on interval
                should_eval = (
                    self._last_eval_time is None or
                    (now - self._last_eval_time).total_seconds() >= self.cfg.eval_interval_sec
                )

                if should_eval:
                    self._last_eval_time = now
                    await self._evaluate_cycle(now)

                # Check fills on active orders every tick
                if self.execution and self.execution.has_active_bracket:
                    await self._check_order_status()

                await asyncio.sleep(0.25)  # 4 Hz tick processing

            except Exception as e:
                log.error(f"Loop error: {e}", exc_info=True)
                await asyncio.sleep(1)

    def _ingest_tick(self, es_ticker, vix_ticker, now: datetime):
        """Feed latest market data into our data structures."""
        price = _safe_float(es_ticker.last) or _safe_float(es_ticker.close)
        bid = _safe_float(es_ticker.bid)
        ask = _safe_float(es_ticker.ask)
        volume = max(int(_safe_float(es_ticker.volume)), 0)

        if price > 0:
            self.market_data.on_tick(price, volume, now)
            self.market_data.bid = bid
            self.market_data.ask = ask

        # VIX
        vix_val = _safe_float(vix_ticker.last) or _safe_float(vix_ticker.close)
        if vix_val > 0:
            self.market_data.vix = vix_val

    async def _evaluate_cycle(self, now: datetime):
        """Run the full signal → strategy → risk → execution pipeline."""

        # Need minimum data
        if len(self.market_data.fast_bars.bars) < 20:
            return  # not enough bars yet
        if self.market_data.last_price <= 0:
            return

        # Generate signals
        signal = self.signal_engine.evaluate(self.market_data)

        # Check if risk manager wants to force-exit
        force_exit, reason = self.risk_manager.should_force_exit(
            self.strategy, self.market_data)
        if force_exit and self.strategy.position is not None:
            log.warning(f"FORCE EXIT: {reason}")
            await self._execute_exit(reason)
            return

        # Strategy decision
        decision = self.strategy.evaluate(self.market_data, signal)

        if decision.action == Action.HOLD:
            return

        # Entry
        if decision.action in (Action.ENTER_LONG, Action.ENTER_SHORT):
            allowed, qty, block_reason = self.risk_manager.check_entry(
                decision, self.strategy, self.market_data, signal)

            if not allowed:
                log.info(f"Entry BLOCKED by risk: {block_reason}")
                return

            log.info(
                f"ENTRY SIGNAL: {decision.action.value} {qty}x ES "
                f"@ {decision.entry_price:.2f} stop={decision.stop_price:.2f} "
                f"target={decision.target_price:.2f}"
            )
            log.info(f"Reason: {decision.reason}")

            if self.cfg.paper_mode:
                log.info("[PAPER] Would place order — logging only")
                self._log_paper_trade(decision, qty, signal)
                # Simulate fill for paper mode tracking
                side = Side.LONG if decision.action == Action.ENTER_LONG else Side.SHORT
                self.strategy.open_position(
                    side=side,
                    entry_price=decision.entry_price,
                    stop_price=decision.stop_price,
                    target_price=decision.target_price,
                    quantity=qty,
                    regime=decision.regime,
                )
                return

            # Live execution
            if decision.action == Action.ENTER_LONG:
                await self.execution.enter_long(
                    decision.entry_price, decision.stop_price,
                    decision.target_price, qty)
            else:
                await self.execution.enter_short(
                    decision.entry_price, decision.stop_price,
                    decision.target_price, qty)

            side = Side.LONG if decision.action == Action.ENTER_LONG else Side.SHORT
            self.strategy.open_position(
                side=side,
                entry_price=decision.entry_price,
                stop_price=decision.stop_price,
                target_price=decision.target_price,
                quantity=qty,
                regime=decision.regime,
            )

        # Exit
        elif decision.action == Action.EXIT:
            await self._execute_exit(decision.reason)

    async def _execute_exit(self, reason: str):
        """Execute a position exit."""
        price = self.market_data.last_price

        if self.cfg.paper_mode:
            log.info(f"[PAPER] Exit @ {price:.2f} — {reason}")
            trade = self.strategy.close_position(price, reason)
            if trade:
                self._save_trade(trade)
            return

        # Live: flatten via market order
        if self.execution:
            await self.execution.flat_position()
        trade = self.strategy.close_position(price, reason)
        if trade:
            self._save_trade(trade)

    async def _check_order_status(self):
        """Monitor active bracket orders for fills."""
        if not self.execution:
            return
        fills = self.execution.check_fills()

        # Entry filled — update position with actual fill price
        if fills["entry_filled"] and self.strategy.position:
            actual_entry = fills["entry_price"]
            if actual_entry > 0 and actual_entry != self.strategy.position.entry_price:
                log.info(
                    f"Entry filled at {actual_entry:.2f} "
                    f"(expected {self.strategy.position.entry_price:.2f})"
                )
                self.strategy.position.entry_price = actual_entry

        # Stop or target filled — close position
        if fills["stop_filled"]:
            exit_price = fills["exit_price"]
            trade = self.strategy.close_position(exit_price, "Stop filled by broker")
            if trade:
                self._save_trade(trade)
            self.execution.active_orders = OrderSet()

        elif fills["target_filled"]:
            exit_price = fills["exit_price"]
            trade = self.strategy.close_position(exit_price, "Target filled by broker")
            if trade:
                self._save_trade(trade)
            self.execution.active_orders = OrderSet()

    def _log_paper_trade(self, decision, qty: int, signal: SignalSnapshot):
        """Log paper trade details for analysis."""
        log.info(
            f"[PAPER TRADE] {decision.action.value} {qty}x ES "
            f"entry={decision.entry_price:.2f} stop={decision.stop_price:.2f} "
            f"target={decision.target_price:.2f} "
            f"regime={signal.regime.value} VIX={signal.vix:.1f} "
            f"IV/RV={signal.iv_rv_ratio:.2f} composite={signal.composite:.0f}"
        )

    def _save_trade(self, trade: dict):
        """Append trade to JSON log file."""
        try:
            trades = []
            if TRADE_LOG_PATH.exists():
                trades = json.loads(TRADE_LOG_PATH.read_text())
            trades.append(trade)
            TRADE_LOG_PATH.write_text(json.dumps(trades, indent=2))
            log.info(f"Trade saved to {TRADE_LOG_PATH}")
        except Exception as e:
            log.error(f"Failed to save trade log: {e}")

    def print_status(self):
        """Print current bot status summary."""
        md = self.market_data
        pos = self.strategy.position
        lines = [
            f"ES: {md.last_price:.2f}  VWAP: {md.vwap.value:.2f}  "
            f"ATR: {md.atr:.2f}  VIX: {md.vix:.1f}",
            f"Bars: {len(md.fast_bars.bars)} (1m) / {len(md.slow_bars.bars)} (5m)",
            f"Levels: prior_close={md.levels.prior_close:.2f} "
            f"ON_H={md.levels.overnight_high:.2f} ON_L={md.levels.overnight_low:.2f}",
            f"Daily PnL: ${self.strategy.daily_pnl:.2f}  "
            f"Trades: {self.strategy.trade_count_today}",
        ]
        if pos:
            lines.append(
                f"POSITION: {pos.side.value} {pos.quantity}x "
                f"entry={pos.entry_price:.2f} stop={pos.stop_price:.2f} "
                f"target={pos.target_price:.2f}"
            )
        else:
            lines.append("POSITION: flat")
        return "\n".join(lines)

    async def shutdown(self):
        """Graceful shutdown."""
        log.info("Shutting down bot...")
        self._running = False
        if self.strategy.position is not None and not self.cfg.paper_mode:
            log.info("Flattening position before shutdown...")
            if self.execution:
                await self.execution.flat_position()
            self.strategy.close_position(
                self.market_data.last_price, "Bot shutdown")
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
        log.info("Bot shutdown complete.")
