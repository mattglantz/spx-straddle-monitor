"""
Gamma Exposure (GEX) Estimator.

Fetches SPX options chain data from IBKR, calculates net dealer
gamma at each strike, and finds the "gamma flip" level.

The core insight:
- Customers are net LONG calls and LONG puts (buying protection/upside).
- Dealers (market makers) are the other side: SHORT calls, SHORT puts.
- When dealer is short a call → they have POSITIVE gamma at that strike.
  (Price up → delta increases → dealer buys underlying to hedge → stabilizing.)
- When dealer is short a put → they have NEGATIVE gamma at that strike.
  (Price down → delta decreases → dealer sells underlying to hedge → destabilizing.)
- Net GEX = sum of all call gamma×OI (positive) minus put gamma×OI (negative).
- The GAMMA FLIP LEVEL is where net GEX crosses zero.
  Above it → positive gamma → mean-reversion.
  Below it → negative gamma → acceleration / trending.

This gives us a hard number instead of a fuzzy regime estimate.

Signal output:
  +100 = price well above gamma flip (deep positive gamma territory)
  -100 = price well below gamma flip (deep negative gamma territory)
     0 = price at the flip level (transition zone)
"""

import asyncio
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from ib_async import IB, Index, Option

from flow_module.config import ET

log = logging.getLogger(__name__)

SPX_SYMBOL = "SPX"
SPX_EXCHANGE = "CBOE"
SPX_MULTIPLIER = 100
STRIKE_STEP = 25        # scan every $25 for efficiency
STRIKE_RANGE = 200      # ±200 points from ATM (16 strikes per side)


@dataclass
class StrikeGEX:
    """GEX data at a single strike."""
    strike: float
    call_oi: int = 0
    put_oi: int = 0
    call_gamma: float = 0.0
    put_gamma: float = 0.0
    call_gex: float = 0.0   # positive (dealer long gamma)
    put_gex: float = 0.0    # negative (dealer short gamma)
    net_gex: float = 0.0


@dataclass
class GEXSignal:
    """Output of the GEX estimator."""
    signal: float              # -100 to +100
    gamma_flip_level: float    # the exact price where GEX crosses zero
    spot_price: float
    distance_to_flip: float    # points: positive = above flip, negative = below
    distance_to_flip_pct: float
    total_gex: float           # net GEX across all strikes (in $ gamma)
    gamma_regime: str          # "POSITIVE", "NEGATIVE", "NEUTRAL"
    top_strikes: list[StrikeGEX]  # strikes with highest absolute GEX
    description: str
    stale: bool = False        # True if data is old


@dataclass
class GEXData:
    """Internal state for GEX calculations."""
    strike_data: list[StrikeGEX] = field(default_factory=list)
    spot: float = 0.0
    last_fetch: Optional[datetime] = None
    expiry_used: str = ""


class GEXEstimator:
    """
    Fetches SPX options data and calculates dealer gamma exposure.

    Designed to run periodically (every 10-15 min) since fetching
    the full chain is resource-intensive.
    """

    def __init__(self):
        self._data = GEXData()
        self._signal: Optional[GEXSignal] = None

    async def fetch_and_calculate(self, ib: IB, spot: float):
        """
        Fetch options chain from IBKR and compute GEX.
        This is the expensive call — run it every 10-15 minutes.
        """
        if spot <= 0:
            log.warning("GEX: no spot price, skipping fetch")
            return

        self._data.spot = spot

        try:
            # Get available expiries
            spx = Index(SPX_SYMBOL, SPX_EXCHANGE)
            await ib.qualifyContractsAsync(spx)

            chains = await ib.reqSecDefOptParamsAsync(
                spx.symbol, "", spx.secType, spx.conId)
            spxw_chains = [c for c in chains
                           if c.tradingClass == "SPXW" and c.exchange == "SMART"]
            if not spxw_chains:
                spxw_chains = [c for c in chains if c.exchange == "SMART"]

            if not spxw_chains:
                log.warning("GEX: no option chains found")
                return

            # Find nearest expiry (0DTE or 1DTE — where gamma is most concentrated)
            today_str = datetime.now(ET).strftime("%Y%m%d")
            all_expiries = set()
            for chain in spxw_chains:
                all_expiries.update(chain.expirations)
            sorted_exp = sorted(e for e in all_expiries if e >= today_str)

            if not sorted_exp:
                log.warning("GEX: no future expiries found")
                return

            # Use the nearest 2 expiries (0DTE + next day have most gamma)
            target_expiries = sorted_exp[:2]
            trading_class = spxw_chains[0].tradingClass

            # Generate strikes to scan
            atm = STRIKE_STEP * round(spot / STRIKE_STEP)
            strikes = [atm + i * STRIKE_STEP
                       for i in range(-STRIKE_RANGE // STRIKE_STEP,
                                       STRIKE_RANGE // STRIKE_STEP + 1)]

            log.info(f"GEX: scanning {len(strikes)} strikes × "
                     f"{len(target_expiries)} expiries around {atm}")

            # Build and qualify all option contracts
            contracts = []
            contract_map = {}  # (expiry, strike, right) → contract
            for expiry in target_expiries:
                for strike in strikes:
                    for right in ("C", "P"):
                        opt = Option(SPX_SYMBOL, expiry, strike, right,
                                     "SMART", tradingClass=trading_class)
                        contracts.append(opt)
                        contract_map[(expiry, strike, right)] = opt

            # Qualify in batches to avoid overwhelming IBKR
            batch_size = 50
            for i in range(0, len(contracts), batch_size):
                batch = contracts[i:i + batch_size]
                await ib.qualifyContractsAsync(*batch)
                await asyncio.sleep(0.1)

            # Request snapshot data for each qualified contract
            # Snapshot mode: get one reading then auto-cancel
            tickers = {}
            valid_contracts = [c for c in contracts if c.conId > 0]
            log.info(f"GEX: requesting snapshots for {len(valid_contracts)} contracts")

            for i in range(0, len(valid_contracts), batch_size):
                batch = valid_contracts[i:i + batch_size]
                batch_tickers = []
                for c in batch:
                    t = ib.reqMktData(c, "100,101,106", snapshot=True)
                    key = (c.lastTradeDateOrContractMonth, c.strike, c.right)
                    tickers[key] = t
                    batch_tickers.append(t)
                # Wait for snapshot data to arrive
                await asyncio.sleep(2)

            # Extract gamma and OI, compute GEX per strike
            gex_by_strike: dict[float, StrikeGEX] = {}

            for (expiry, strike, right), ticker in tickers.items():
                if strike not in gex_by_strike:
                    gex_by_strike[strike] = StrikeGEX(strike=strike)

                entry = gex_by_strike[strike]

                # Extract gamma from Greeks
                gamma = 0.0
                for g in (ticker.modelGreeks, ticker.lastGreeks,
                          ticker.bidGreeks, ticker.askGreeks):
                    if g is not None:
                        gval = _safe_float(getattr(g, "gamma", 0))
                        if gval > 0:
                            gamma = gval
                            break

                # Extract open interest
                oi = int(_safe_float(getattr(ticker, "callOpenInterest", 0))
                         if right == "C"
                         else _safe_float(getattr(ticker, "putOpenInterest", 0)))
                # Fallback: generic OI field
                if oi <= 0:
                    oi = int(_safe_float(getattr(ticker, "openInterest", 0)))

                if right == "C":
                    entry.call_gamma = max(entry.call_gamma, gamma)
                    entry.call_oi += oi
                else:
                    entry.put_gamma = max(entry.put_gamma, gamma)
                    entry.put_oi += oi

            # Calculate GEX per strike
            # GEX = OI × Gamma × Multiplier × Spot × 0.01
            # Call GEX is positive (dealer long gamma when short calls)
            # Put GEX is negative (dealer short gamma when short puts)
            for entry in gex_by_strike.values():
                entry.call_gex = (entry.call_oi * entry.call_gamma *
                                  SPX_MULTIPLIER * spot * 0.01)
                entry.put_gex = -(entry.put_oi * entry.put_gamma *
                                  SPX_MULTIPLIER * spot * 0.01)
                entry.net_gex = entry.call_gex + entry.put_gex

            # Snapshot subscriptions auto-cancel — no need to cancel manually

            strike_list = sorted(gex_by_strike.values(), key=lambda s: s.strike)
            self._data.strike_data = strike_list
            self._data.last_fetch = datetime.now(ET)
            self._data.expiry_used = ",".join(target_expiries)

            # Compute the signal
            self._signal = self._compute_signal(strike_list, spot)

            log.info(f"GEX: computed. Flip={self._signal.gamma_flip_level:.0f} "
                     f"regime={self._signal.gamma_regime} "
                     f"signal={self._signal.signal:+.0f}")

        except Exception as e:
            log.error(f"GEX fetch error: {e}", exc_info=True)

    def get_signal(self, current_price: float) -> GEXSignal:
        """
        Get the current GEX signal. Uses cached data from last fetch,
        but updates the signal based on current price vs flip level.
        """
        if self._signal is None or not self._data.strike_data:
            return GEXSignal(
                signal=0, gamma_flip_level=0, spot_price=current_price,
                distance_to_flip=0, distance_to_flip_pct=0,
                total_gex=0, gamma_regime="UNKNOWN",
                top_strikes=[], description="GEX data not yet available.",
                stale=True,
            )

        # Check staleness (>20 min old)
        stale = False
        if self._data.last_fetch:
            age = (datetime.now(ET) - self._data.last_fetch).total_seconds()
            stale = age > 1200

        # Recompute signal with current price against cached flip level
        flip = self._signal.gamma_flip_level
        if flip > 0 and current_price > 0:
            dist = current_price - flip
            dist_pct = (dist / flip) * 100

            # Signal scales with distance from flip
            # ±1% from flip → ±50 signal, ±2% → ±100
            raw_signal = (dist_pct / 2.0) * 100
            signal = max(min(round(raw_signal, 1), 100.0), -100.0)

            regime = "POSITIVE" if dist > 0 else "NEGATIVE"
            if abs(dist_pct) < 0.2:
                regime = "NEUTRAL"

            return GEXSignal(
                signal=signal,
                gamma_flip_level=flip,
                spot_price=current_price,
                distance_to_flip=round(dist, 1),
                distance_to_flip_pct=round(dist_pct, 2),
                total_gex=self._signal.total_gex,
                gamma_regime=regime,
                top_strikes=self._signal.top_strikes,
                description=self._build_description(
                    signal, flip, current_price, dist, regime, stale),
                stale=stale,
            )

        return self._signal

    def _compute_signal(self, strikes: list[StrikeGEX], spot: float) -> GEXSignal:
        """Calculate gamma flip level and signal from strike data."""
        if not strikes:
            return GEXSignal(
                signal=0, gamma_flip_level=spot, spot_price=spot,
                distance_to_flip=0, distance_to_flip_pct=0,
                total_gex=0, gamma_regime="UNKNOWN",
                top_strikes=[], description="No strike data.",
            )

        total_gex = sum(s.net_gex for s in strikes)

        # Find gamma flip: where cumulative GEX from below crosses zero
        # Walk from lowest strike upward, accumulating GEX
        flip_level = spot  # default to ATM if we can't find a cross
        cumulative = 0.0
        for i, s in enumerate(strikes):
            prev_cum = cumulative
            cumulative += s.net_gex
            # Check for zero crossing
            if prev_cum < 0 and cumulative >= 0:
                # Interpolate
                if cumulative != prev_cum:
                    frac = abs(prev_cum) / (abs(prev_cum) + abs(cumulative))
                    if i > 0:
                        flip_level = strikes[i - 1].strike + frac * STRIKE_STEP
                    else:
                        flip_level = s.strike
                else:
                    flip_level = s.strike
                break
            elif prev_cum >= 0 and cumulative < 0:
                if cumulative != prev_cum:
                    frac = abs(prev_cum) / (abs(prev_cum) + abs(cumulative))
                    if i > 0:
                        flip_level = strikes[i - 1].strike + frac * STRIKE_STEP
                    else:
                        flip_level = s.strike
                break

        # Distance from flip
        dist = spot - flip_level
        dist_pct = (dist / flip_level) * 100 if flip_level > 0 else 0

        raw_signal = (dist_pct / 2.0) * 100
        signal = max(min(round(raw_signal, 1), 100.0), -100.0)

        regime = "POSITIVE" if dist > 0 else "NEGATIVE"
        if abs(dist_pct) < 0.2:
            regime = "NEUTRAL"

        # Top strikes by absolute GEX
        top = sorted(strikes, key=lambda s: abs(s.net_gex), reverse=True)[:5]

        return GEXSignal(
            signal=signal,
            gamma_flip_level=round(flip_level, 1),
            spot_price=spot,
            distance_to_flip=round(dist, 1),
            distance_to_flip_pct=round(dist_pct, 2),
            total_gex=round(total_gex, 0),
            gamma_regime=regime,
            top_strikes=top,
            description=self._build_description(
                signal, flip_level, spot, dist, regime, False),
        )

    def _build_description(self, signal: float, flip: float, spot: float,
                           dist: float, regime: str, stale: bool) -> str:
        stale_note = " [STALE DATA]" if stale else ""
        if regime == "POSITIVE":
            return (
                f"POSITIVE GAMMA: ES {spot:.0f} is {abs(dist):.0f}pts ABOVE "
                f"gamma flip at {flip:.0f}. Dealers buy dips / sell rips. "
                f"Mean-reversion favored.{stale_note}"
            )
        elif regime == "NEGATIVE":
            return (
                f"NEGATIVE GAMMA: ES {spot:.0f} is {abs(dist):.0f}pts BELOW "
                f"gamma flip at {flip:.0f}. Dealers chase moves. "
                f"Trend acceleration / momentum favored.{stale_note}"
            )
        else:
            return (
                f"AT GAMMA FLIP: ES {spot:.0f} near flip level {flip:.0f}. "
                f"Transition zone — regime could go either way.{stale_note}"
            )


def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return 0.0 if math.isnan(val) or math.isinf(val) else float(val)
    except (TypeError, ValueError):
        return 0.0
