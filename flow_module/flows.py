"""
Master Flow Aggregator.

Combines all five structural flow trackers into a single
composite view with net directional bias and actionable signals.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from flow_module.config import FlowConfig, ET
from flow_module.data import MarketDataStore
from flow_module.rebalance import RebalanceTracker, RebalanceSignal
from flow_module.opex import OpExTracker, OpExSignal
from flow_module.cta import CTATracker, CTASignal
from flow_module.vol_control import VolControlTracker, VolControlSignal
from flow_module.buyback import BuybackTracker, BuybackSignal

log = logging.getLogger(__name__)


@dataclass
class FlowSnapshot:
    """Complete flow state at a point in time."""
    timestamp: datetime
    es_price: float
    vix: float

    # Individual flow signals
    rebalance: RebalanceSignal
    opex: OpExSignal
    cta: CTASignal
    vol_control: VolControlSignal
    buyback: BuybackSignal

    # Composite
    net_signal: float          # -100 to +100 weighted composite
    net_direction: str         # "BUY", "SELL", "NEUTRAL"
    conviction: str            # "high", "moderate", "low", "none"
    active_flows: list[str]    # list of flows currently active
    headline: str              # one-line summary for dashboard


class FlowAggregator:
    """
    Runs all flow trackers and produces a weighted composite signal.

    Weighting reflects the estimated AUM and market impact of each flow:
    - Rebalancing: 30% (trillions in AUM, very predictable timing)
    - Vol Control:  25% (hundreds of billions, mechanical)
    - CTA:          20% (hundreds of billions, trigger-based)
    - Buyback:      15% (consistent but spread out)
    - OpEx:         10% (affects vol regime more than direction)
    """

    WEIGHTS = {
        "rebalance": 0.30,
        "vol_control": 0.25,
        "cta": 0.20,
        "buyback": 0.15,
        "opex": 0.10,
    }

    def __init__(self, cfg: FlowConfig):
        self.cfg = cfg
        self.rebalance = RebalanceTracker(cfg.rebalance)
        self.opex = OpExTracker(cfg.opex)
        self.cta = CTATracker(cfg.cta)
        self.vol_control = VolControlTracker(cfg.vol_control)
        self.buyback = BuybackTracker(cfg.buyback)

    def evaluate(self, store: MarketDataStore) -> FlowSnapshot:
        """Run all trackers and produce composite snapshot."""
        now = datetime.now(ET)
        price = store.live_price if store.live_price > 0 else store.last_close
        vix = store.live_vix

        # Run each tracker
        closes = store.closes
        dates = [b.date for b in store.daily_bars]

        rebal_sig = self.rebalance.evaluate(closes, dates, price)
        opex_sig = self.opex.evaluate()
        cta_sig = self.cta.evaluate(store)
        vol_sig = self.vol_control.evaluate(store)
        buyback_sig = self.buyback.evaluate()

        # Compute weighted composite
        # OpEx is non-directional (PIN/UNWIND), so we only use it as
        # a vol-regime modifier, not a directional signal
        directional_signals = {
            "rebalance": rebal_sig.signal,
            "vol_control": vol_sig.signal,
            "cta": cta_sig.signal,
            "buyback": buyback_sig.signal,
        }

        weighted_sum = sum(
            sig * self.WEIGHTS[name]
            for name, sig in directional_signals.items()
        )

        # OpEx modifier: during UNWIND phase, amplify signals (trends run further)
        # During PIN phase, dampen signals (mean-reversion dominates)
        if opex_sig.phase == "UNWIND":
            weighted_sum *= 1.2
        elif opex_sig.phase == "PIN":
            weighted_sum *= 0.7

        net_signal = max(min(round(weighted_sum, 1), 100.0), -100.0)

        # Determine direction and conviction
        if net_signal > 15:
            net_direction = "BUY"
        elif net_signal < -15:
            net_direction = "SELL"
        else:
            net_direction = "NEUTRAL"

        conviction = self._classify_conviction(net_signal, directional_signals)

        # Which flows are active (non-zero signal)?
        active = []
        if abs(rebal_sig.signal) > 10:
            active.append(f"Rebalance ({rebal_sig.flow_direction})")
        if opex_sig.phase != "NEUTRAL":
            active.append(f"OpEx ({opex_sig.phase})")
        if abs(cta_sig.signal) > 10:
            active.append(f"CTA ({cta_sig.flow_direction})")
        if abs(vol_sig.signal) > 10:
            active.append(f"Vol-Control ({vol_sig.flow_direction})")
        if abs(buyback_sig.signal) > 10:
            active.append(f"Buyback ({buyback_sig.blackout_phase})")

        headline = self._build_headline(
            net_signal, net_direction, conviction, active, price, vix)

        snapshot = FlowSnapshot(
            timestamp=now,
            es_price=price,
            vix=vix,
            rebalance=rebal_sig,
            opex=opex_sig,
            cta=cta_sig,
            vol_control=vol_sig,
            buyback=buyback_sig,
            net_signal=net_signal,
            net_direction=net_direction,
            conviction=conviction,
            active_flows=active,
            headline=headline,
        )

        log.info(
            f"Flow Snapshot: net={net_signal:+.0f} ({net_direction}) "
            f"conviction={conviction} | "
            f"rebal={rebal_sig.signal:+.0f} cta={cta_sig.signal:+.0f} "
            f"vol={vol_sig.signal:+.0f} buyback={buyback_sig.signal:+.0f} "
            f"opex={opex_sig.phase}"
        )

        return snapshot

    def _classify_conviction(self, net: float, signals: dict) -> str:
        """
        Conviction is high when multiple flows agree in direction.
        """
        abs_net = abs(net)
        if abs_net < 15:
            return "none"

        # Count how many signals agree with net direction
        agreeing = sum(
            1 for sig in signals.values()
            if (sig > 10 and net > 0) or (sig < -10 and net < 0)
        )

        if abs_net >= 50 and agreeing >= 3:
            return "high"
        elif abs_net >= 30 and agreeing >= 2:
            return "moderate"
        elif abs_net >= 15:
            return "low"
        return "none"

    def _build_headline(self, net: float, direction: str, conviction: str,
                        active: list[str], price: float, vix: float) -> str:
        if not active:
            return f"ES {price:.0f} | VIX {vix:.1f} | No significant structural flows active"

        flow_list = ", ".join(active)
        return (
            f"ES {price:.0f} | VIX {vix:.1f} | "
            f"Net: {direction} {abs(net):.0f} ({conviction}) | "
            f"Active: {flow_list}"
        )
