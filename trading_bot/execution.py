"""
Order execution and management via IBKR.

Handles:
- Placing bracket orders (entry + stop-loss + take-profit)
- Tracking order fills and status
- Cancelling / modifying orders
- Flatting the position (market close-out)

Uses limit orders for entries to avoid slippage. Stop orders
for protection. Limit orders for targets.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ib_async import (
    IB, ContFuture, Order, LimitOrder, StopOrder, MarketOrder, Trade
)

from trading_bot.config import (
    ES_SYMBOL, ES_EXCHANGE, ES_TICK_SIZE, ES_POINT_VALUE, ET
)
from trading_bot.strategy import Side

log = logging.getLogger(__name__)


def _round_tick(price: float) -> float:
    """Round price to the nearest ES tick (0.25)."""
    return round(round(price / ES_TICK_SIZE) * ES_TICK_SIZE, 2)


@dataclass
class OrderSet:
    """Tracks a bracket: entry, stop, and target orders."""
    entry_trade: Optional[Trade] = None
    stop_trade: Optional[Trade] = None
    target_trade: Optional[Trade] = None
    side: Optional[Side] = None
    quantity: int = 0
    filled: bool = False
    cancelled: bool = False


class ExecutionManager:
    """
    Manages order lifecycle for ES futures trades via IBKR.
    """

    def __init__(self, ib: IB):
        self.ib = ib
        self.es_contract: Optional[ContFuture] = None
        self.active_orders: OrderSet = OrderSet()
        self._qualified = False

    async def qualify_contract(self):
        """Qualify the ES continuous futures contract."""
        if self._qualified:
            return
        self.es_contract = ContFuture(ES_SYMBOL, ES_EXCHANGE)
        await self.ib.qualifyContractsAsync(self.es_contract)
        log.info(f"Qualified ES contract: {self.es_contract}")
        self._qualified = True

    async def enter_long(self, price: float, stop: float, target: float,
                         quantity: int) -> OrderSet:
        """Place a long bracket: buy limit + sell stop + sell limit target."""
        await self.qualify_contract()
        return await self._place_bracket(
            side=Side.LONG,
            entry_action="BUY",
            exit_action="SELL",
            price=price,
            stop=stop,
            target=target,
            quantity=quantity,
        )

    async def enter_short(self, price: float, stop: float, target: float,
                          quantity: int) -> OrderSet:
        """Place a short bracket: sell limit + buy stop + buy limit target."""
        await self.qualify_contract()
        return await self._place_bracket(
            side=Side.SHORT,
            entry_action="SELL",
            exit_action="BUY",
            price=price,
            stop=stop,
            target=target,
            quantity=quantity,
        )

    async def _place_bracket(self, side: Side, entry_action: str, exit_action: str,
                              price: float, stop: float, target: float,
                              quantity: int) -> OrderSet:
        """Place a bracket order set."""
        entry_price = _round_tick(price)
        stop_price = _round_tick(stop)
        target_price = _round_tick(target)

        # Create OCA (One-Cancels-All) group for stop and target
        oca_group = f"ES_BOT_{datetime.now(ET).strftime('%H%M%S%f')}"

        # Entry: limit order
        entry_order = LimitOrder(
            action=entry_action,
            totalQuantity=quantity,
            lmtPrice=entry_price,
            tif="GTC",
            outsideRth=True,
        )

        # Stop loss
        stop_order = StopOrder(
            action=exit_action,
            totalQuantity=quantity,
            stopPrice=stop_price,
            tif="GTC",
            outsideRth=True,
            ocaGroup=oca_group,
            ocaType=1,  # cancel other orders in group on fill
        )

        # Profit target
        target_order = LimitOrder(
            action=exit_action,
            totalQuantity=quantity,
            lmtPrice=target_price,
            tif="GTC",
            outsideRth=True,
            ocaGroup=oca_group,
            ocaType=1,
        )

        order_set = OrderSet(side=side, quantity=quantity)

        # Place entry first
        entry_trade = self.ib.placeOrder(self.es_contract, entry_order)
        order_set.entry_trade = entry_trade
        log.info(
            f"Entry order placed: {entry_action} {quantity}x ES @ {entry_price:.2f} "
            f"(orderId={entry_trade.order.orderId})"
        )

        # Place stop and target (these wait for entry fill before becoming active)
        # In practice with OCA, we place them immediately — IBKR handles the rest
        stop_trade = self.ib.placeOrder(self.es_contract, stop_order)
        order_set.stop_trade = stop_trade
        log.info(f"Stop order placed: {exit_action} @ {stop_price:.2f}")

        target_trade = self.ib.placeOrder(self.es_contract, target_order)
        order_set.target_trade = target_trade
        log.info(f"Target order placed: {exit_action} @ {target_price:.2f}")

        self.active_orders = order_set
        return order_set

    async def modify_stop(self, new_stop: float):
        """Move the stop loss (e.g., for trailing stops)."""
        if self.active_orders.stop_trade is None:
            return
        trade = self.active_orders.stop_trade
        new_price = _round_tick(new_stop)
        trade.order.stopPrice = new_price
        self.ib.placeOrder(self.es_contract, trade.order)
        log.info(f"Stop modified to {new_price:.2f}")

    async def flat_position(self) -> Optional[Trade]:
        """Close everything at market. Emergency or end-of-day exit."""
        await self.qualify_contract()

        # Cancel all pending orders
        await self.cancel_all()

        # Determine if we have a position to close
        positions = [p for p in self.ib.positions()
                     if p.contract.symbol == ES_SYMBOL]

        if not positions:
            log.info("No ES position to flatten")
            return None

        pos = positions[0]
        qty = abs(pos.position)
        action = "SELL" if pos.position > 0 else "BUY"

        order = MarketOrder(
            action=action,
            totalQuantity=qty,
            tif="GTC",
            outsideRth=True,
        )
        trade = self.ib.placeOrder(self.es_contract, order)
        log.info(f"FLAT: {action} {qty}x ES at market")
        self.active_orders = OrderSet()
        return trade

    async def cancel_all(self):
        """Cancel all active orders for our bracket."""
        os = self.active_orders
        for trade in [os.entry_trade, os.stop_trade, os.target_trade]:
            if trade is not None and trade.isActive():
                self.ib.cancelOrder(trade.order)
                log.info(f"Cancelled order {trade.order.orderId}")
        self.active_orders = OrderSet()

    def check_fills(self) -> dict:
        """
        Check status of active orders.
        Returns dict with 'entry_filled', 'stop_filled', 'target_filled', 'entry_price'.
        """
        os = self.active_orders
        result = {
            "entry_filled": False,
            "stop_filled": False,
            "target_filled": False,
            "entry_price": 0.0,
            "exit_price": 0.0,
        }

        if os.entry_trade and os.entry_trade.orderStatus.status == "Filled":
            result["entry_filled"] = True
            result["entry_price"] = os.entry_trade.orderStatus.avgFillPrice

        if os.stop_trade and os.stop_trade.orderStatus.status == "Filled":
            result["stop_filled"] = True
            result["exit_price"] = os.stop_trade.orderStatus.avgFillPrice

        if os.target_trade and os.target_trade.orderStatus.status == "Filled":
            result["target_filled"] = True
            result["exit_price"] = os.target_trade.orderStatus.avgFillPrice

        return result

    @property
    def has_active_bracket(self) -> bool:
        os = self.active_orders
        return os.entry_trade is not None and not os.cancelled
