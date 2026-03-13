"""
advanced_features.py — 4 advanced features for Market Bot v28.2

Features:
  1. calc_gex_regime     — Dealer Gamma Exposure model (long/short gamma)
  2. FlowScanner         — Unusual options flow detection
  3. calc_vol_regime_shift — Intraday realized vol regime shift detection
  4. calc_divergence_score — Cross-signal divergence detector
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import math
from bot_config import now_et


def _safe_float(val, default=0.0):
    """Convert to float, handling NaN/None/empty safely."""
    try:
        v = float(val or default)
        return default if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return default

logger = logging.getLogger("MarketBot")



def calc_gex_regime(md) -> dict:
    """
    Dealer Gamma Exposure (GEX) model.

    When dealers are LONG GAMMA (positive GEX):
      - They sell rips and buy dips (hedging)
      - Market pins, ranges compress, mean-reversion works
      - SELL iron condors, fade breakouts

    When dealers are SHORT GAMMA (negative GEX):
      - They chase momentum (forced to buy into rallies, sell into drops)
      - Moves accelerate, ranges expand, breakouts work
      - BUY breakouts, avoid fading moves, wider stops

    GEX flip point = the price level where GEX changes sign.

    Calculation: GEX ≈ Σ(Call_OI × Call_Gamma - Put_OI × Put_Gamma) × 100 × Spot
    Simplified (no greeks): Use OI distribution as proxy.
    Dealers are generally short puts (long gamma on puts) and short calls (short gamma on calls).
    Net GEX > 0 when put OI > call OI below spot (typical).
    """
    try:
        es_price = md.current_price
        if es_price <= 0:
            return _gex_default()

        # Get OI data from gamma detail (already computed by calc_gamma_levels)
        gamma_detail = getattr(md, '_gamma_detail', {})
        call_oi = gamma_detail.get("call_oi", 0)
        put_oi = gamma_detail.get("put_oi", 0)
        call_wall = gamma_detail.get("call_wall", 0)
        put_wall = gamma_detail.get("put_wall", 0)

        # Try to get fuller chain from IBKR for per-strike calculation
        chain_data = None
        if md.ibkr and md.ibkr.connected:
            try:
                chain_data = md.ibkr.get_spx_options_chain(strike_range=75)
            except Exception:
                pass

        if chain_data and not chain_data.get("calls", pd.DataFrame()).empty:
            calls_df = chain_data["calls"]
            puts_df = chain_data["puts"]

            # Per-strike GEX approximation
            # Dealers are short calls above spot, long puts below spot (market-making)
            # GEX at each strike ≈ OI × gamma × 100 × spot
            # Without greeks, we approximate gamma as higher for ATM, lower for OTM
            # Simple model: gamma ≈ 1 / (1 + |strike - spot| / 50)

            gex_by_strike = {}
            all_strikes = set()

            has_gamma = "gamma" in calls_df.columns

            for _, row in calls_df.iterrows():
                s = float(row["strike"])
                all_strikes.add(s)
                oi = float(row.get("openInterest", 0) or row.get("volume", 0) or 0)
                # Use real gamma from IBKR greeks when available
                if has_gamma and float(row.get("gamma", 0)) > 0:
                    gamma = float(row["gamma"])
                else:
                    gamma = 1.0 / (1.0 + abs(s - es_price) / 50.0)
                # Dealers are typically short calls → negative gamma for them
                gex_by_strike[s] = gex_by_strike.get(s, 0) - oi * gamma

            has_gamma_p = "gamma" in puts_df.columns

            for _, row in puts_df.iterrows():
                s = float(row["strike"])
                all_strikes.add(s)
                oi = float(row.get("openInterest", 0) or row.get("volume", 0) or 0)
                if has_gamma_p and float(row.get("gamma", 0)) > 0:
                    gamma = float(row["gamma"])
                else:
                    gamma = 1.0 / (1.0 + abs(s - es_price) / 50.0)
                # Dealers are typically short puts → but put gamma is negative direction
                # So dealer short put = positive gamma exposure for them
                gex_by_strike[s] = gex_by_strike.get(s, 0) + oi * gamma

            # Total GEX
            total_gex = sum(gex_by_strike.values())

            # GEX flip point: find closest sign change to spot price
            sorted_strikes = sorted(gex_by_strike.keys())
            flip_level = es_price
            flip_candidates = []
            cumulative = 0
            for s in sorted_strikes:
                prev_cum = cumulative
                cumulative += gex_by_strike[s]
                if prev_cum * cumulative < 0:  # Sign change
                    flip_candidates.append(s)
            if flip_candidates:
                # Pick the flip point closest to current price
                flip_level = min(flip_candidates, key=lambda s: abs(s - es_price))

            # Regime classification
            if total_gex > 0:
                # Positive GEX = dealers are long gamma = mean-reversion regime
                magnitude = "STRONG" if total_gex > put_oi * 0.5 else "MODERATE"
                regime = f"LONG GAMMA ({magnitude})"
                signal = (
                    "Dealers hedging = market pins. Fade breakouts, sell premium. "
                    "Expect ranges to compress. Iron condors and mean-reversion favored."
                )
                playbook = "MEAN-REVERT"
            else:
                magnitude = "STRONG" if abs(total_gex) > call_oi * 0.5 else "MODERATE"
                regime = f"SHORT GAMMA ({magnitude})"
                signal = (
                    "Dealers chasing = moves accelerate. Trade breakouts, buy premium. "
                    "Expect wider ranges. Straddles and momentum plays favored."
                )
                playbook = "MOMENTUM"

            result = {
                "regime": regime,
                "total_gex": round(total_gex, 0),
                "flip_level": round(flip_level, 0),
                "distance_to_flip": round(flip_level - es_price, 1),
                "playbook": playbook,
                "signal": signal,
                "call_oi": call_oi,
                "put_oi": put_oi,
            }
            logger.info(
                f"GEX: {regime} | Flip: {flip_level:.0f} ({flip_level - es_price:+.0f}) | "
                f"Playbook: {playbook}"
            )
            return result

        # ─── Fallback: estimate from aggregate OI ───
        if call_oi > 0 or put_oi > 0:
            # Simple heuristic: if put OI > call OI, dealers are likely long gamma
            if put_oi > call_oi * 1.2:
                regime = "LONG GAMMA (EST)"
                playbook = "MEAN-REVERT"
                signal = "Put OI dominant → likely dealer long gamma. Fade moves."
            elif call_oi > put_oi * 1.2:
                regime = "SHORT GAMMA (EST)"
                playbook = "MOMENTUM"
                signal = "Call OI dominant → likely dealer short gamma. Ride moves."
            else:
                regime = "NEUTRAL GAMMA"
                playbook = "MIXED"
                signal = "Balanced OI. No strong gamma tilt."

            flip_est = (call_wall + put_wall) / 2 if call_wall and put_wall else es_price
            return {
                "regime": regime, "total_gex": 0, "flip_level": round(flip_est, 0),
                "distance_to_flip": round(flip_est - es_price, 1),
                "playbook": playbook, "signal": signal,
                "call_oi": call_oi, "put_oi": put_oi,
            }

        return _gex_default()

    except Exception as e:
        logger.warning(f"GEX calc failed: {e}")
        return _gex_default()


def _gex_default() -> dict:
    return {
        "regime": "UNKNOWN", "total_gex": 0, "flip_level": 0,
        "distance_to_flip": 0, "playbook": "MIXED",
        "signal": "Insufficient data for GEX calculation.",
        "call_oi": 0, "put_oi": 0,
    }


# =====================================================================
# 3. FLOW SCANNER — Unusual Options Activity
# =====================================================================

class FlowScanner:
    """
    Detects unusual SPX options flow with moneyness-weighted bias and
    volume-normalized confidence.

    Upgrades from raw P/C ratio:
    - ITM/ATM flow weighted 3x vs far OTM (directional vs lottery tickets)
    - P/C ratio normalized against rolling session average
    - Minimum total volume threshold suppresses false signals on dead days
    - Cycle-over-cycle delta tracking catches live sweeps
    """

    # Moneyness weight: how much directional signal a strike carries.
    # ITM/ATM options are more likely directional; far OTM are hedges/lotto.
    _MONEYNESS_WEIGHTS = {
        "DEEP_ITM": 3.0,   # > 2% ITM
        "ITM": 2.5,        # 0-2% ITM
        "ATM": 2.0,        # within 0.5% of spot
        "NEAR_OTM": 1.0,   # 0-3% OTM
        "FAR_OTM": 0.3,    # > 3% OTM (likely hedges)
    }

    def __init__(self):
        self.prev_calls = {}   # strike → {"vol": x, "oi": y}
        self.prev_puts = {}
        self.alerts = []
        self.cumulative_call_vol = 0
        self.cumulative_put_vol = 0
        # Rolling P/C ratio history for normalization
        self._pc_history = []     # last N raw P/C ratios
        self._vol_history = []    # last N total volumes
        self._MAX_HISTORY = 30    # ~30 cycles ≈ 5 hours at 10min intervals

    def _classify_moneyness(self, strike: float, spot: float,
                            is_call: bool) -> tuple:
        """Return (label, weight) for a strike relative to spot."""
        pct_dist = (strike - spot) / spot * 100  # positive = above spot

        if is_call:
            # Call: ITM when strike < spot
            if pct_dist < -2.0:
                return "DEEP_ITM", self._MONEYNESS_WEIGHTS["DEEP_ITM"]
            elif pct_dist < 0:
                return "ITM", self._MONEYNESS_WEIGHTS["ITM"]
            elif abs(pct_dist) <= 0.5:
                return "ATM", self._MONEYNESS_WEIGHTS["ATM"]
            elif pct_dist <= 3.0:
                return "NEAR_OTM", self._MONEYNESS_WEIGHTS["NEAR_OTM"]
            else:
                return "FAR_OTM", self._MONEYNESS_WEIGHTS["FAR_OTM"]
        else:
            # Put: ITM when strike > spot
            if pct_dist > 2.0:
                return "DEEP_ITM", self._MONEYNESS_WEIGHTS["DEEP_ITM"]
            elif pct_dist > 0:
                return "ITM", self._MONEYNESS_WEIGHTS["ITM"]
            elif abs(pct_dist) <= 0.5:
                return "ATM", self._MONEYNESS_WEIGHTS["ATM"]
            elif pct_dist >= -3.0:
                return "NEAR_OTM", self._MONEYNESS_WEIGHTS["NEAR_OTM"]
            else:
                return "FAR_OTM", self._MONEYNESS_WEIGHTS["FAR_OTM"]

    def scan(self, md, threshold_vol_oi: float = 3.0,
             threshold_notional: float = 500_000) -> dict:
        """
        Scan for unusual options activity with moneyness weighting.

        Returns dict with flow bias, alerts, and confidence level.
        """
        try:
            if not (md.ibkr and md.ibkr.connected):
                return self._empty()

            chain = md.ibkr.get_spx_options_chain(strike_range=75)
            calls_df = chain.get("calls", pd.DataFrame())
            puts_df = chain.get("puts", pd.DataFrame())

            if calls_df.empty and puts_df.empty:
                return self._empty()

            es_price = md.current_price
            alerts = []

            # DTE multiplier — near-expiry flow is more directional
            expiry_str = chain.get("expiry", "")
            dte_mult = 1.0
            if expiry_str and expiry_str != "N/A":
                try:
                    expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()
                    dte = (expiry_date - datetime.now().date()).days
                    if dte <= 1:
                        dte_mult = 1.5    # 0DTE/1DTE — strongest directional signal
                    elif dte <= 3:
                        dte_mult = 1.2    # Weekly
                    elif dte <= 7:
                        dte_mult = 1.0    # Baseline
                    elif dte <= 30:
                        dte_mult = 0.7    # Monthly — more hedging noise
                    else:
                        dte_mult = 0.4    # LEAPS — least directional
                except ValueError:
                    dte_mult = 1.0

            # Raw volume totals (for backward compat)
            total_call_vol = 0
            total_put_vol = 0
            total_call_notional = 0
            total_put_notional = 0

            # Moneyness-weighted volume totals (for improved bias)
            weighted_call_vol = 0.0
            weighted_put_vol = 0.0

            # Scan calls
            curr_calls = {}
            for _, row in calls_df.iterrows():
                strike = float(row["strike"])
                vol = _safe_float(row.get("volume", 0))
                oi = _safe_float(row.get("openInterest", 0))
                mid = _safe_float(row.get("lastPrice", 0))
                curr_calls[strike] = {"vol": vol, "oi": oi, "mid": mid}
                total_call_vol += vol
                notional = vol * mid * 100  # SPX multiplier
                total_call_notional += notional

                # Moneyness weight × DTE multiplier
                _label, weight = self._classify_moneyness(strike, es_price, is_call=True)
                weighted_call_vol += vol * weight * dte_mult

                # Check volume/OI ratio for unusual activity
                if oi > 0 and vol > threshold_vol_oi * oi and notional > threshold_notional:
                    prev = self.prev_calls.get(strike, {})
                    delta_vol = vol - prev.get("vol", 0)

                    alerts.append({
                        "type": "CALL",
                        "strike": strike,
                        "volume": int(vol),
                        "oi": int(oi),
                        "vol_oi_ratio": round(vol / oi, 1),
                        "notional": round(notional / 1_000_000, 2),
                        "delta_vol": int(max(delta_vol, 0)),
                        "distance": round(strike - es_price, 1),
                        "side": "ABOVE" if strike > es_price else "BELOW",
                        "moneyness": _label,
                        "weight": weight,
                    })

            # Scan puts
            curr_puts = {}
            for _, row in puts_df.iterrows():
                strike = float(row["strike"])
                vol = _safe_float(row.get("volume", 0))
                oi = _safe_float(row.get("openInterest", 0))
                mid = _safe_float(row.get("lastPrice", 0))
                curr_puts[strike] = {"vol": vol, "oi": oi, "mid": mid}
                total_put_vol += vol
                notional = vol * mid * 100
                total_put_notional += notional

                _label, weight = self._classify_moneyness(strike, es_price, is_call=False)
                weighted_put_vol += vol * weight * dte_mult

                if oi > 0 and vol > threshold_vol_oi * oi and notional > threshold_notional:
                    prev = self.prev_puts.get(strike, {})
                    delta_vol = vol - prev.get("vol", 0)

                    alerts.append({
                        "type": "PUT",
                        "strike": strike,
                        "volume": int(vol),
                        "oi": int(oi),
                        "vol_oi_ratio": round(vol / oi, 1),
                        "notional": round(notional / 1_000_000, 2),
                        "delta_vol": int(max(delta_vol, 0)),
                        "distance": round(es_price - strike, 1),
                        "side": "ABOVE" if strike > es_price else "BELOW",
                        "moneyness": _label,
                        "weight": weight,
                    })

            # Update state for next cycle
            self.prev_calls = curr_calls
            self.prev_puts = curr_puts
            self.cumulative_call_vol = total_call_vol
            self.cumulative_put_vol = total_put_vol

            # Sort by weighted notional: notional × moneyness_weight
            # This surfaces ITM/ATM sweeps over far OTM lottery tickets
            for a in alerts:
                a["weighted_notional"] = round(a["notional"] * a["weight"], 2)
            alerts.sort(key=lambda x: x["weighted_notional"], reverse=True)

            # --- Moneyness-weighted P/C ratio ---
            if weighted_call_vol > 0:
                weighted_pc = weighted_put_vol / weighted_call_vol
            else:
                weighted_pc = 1.0

            # Raw P/C for reference
            raw_pc = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0

            # --- Volume normalization ---
            # Track rolling averages to know if today's flow is meaningful
            total_vol = total_call_vol + total_put_vol
            self._pc_history.append(raw_pc)
            self._vol_history.append(total_vol)
            if len(self._pc_history) > self._MAX_HISTORY:
                self._pc_history = self._pc_history[-self._MAX_HISTORY:]
                self._vol_history = self._vol_history[-self._MAX_HISTORY:]

            avg_vol = sum(self._vol_history) / len(self._vol_history) if self._vol_history else total_vol
            vol_ratio = total_vol / avg_vol if avg_vol > 0 else 1.0

            # --- Flow confidence: is this signal trustworthy? ---
            # Low volume days get "LOW" confidence regardless of P/C ratio
            if vol_ratio < 0.5:
                flow_confidence = "LOW"
            elif vol_ratio < 0.8:
                flow_confidence = "MODERATE"
            else:
                flow_confidence = "HIGH"

            # --- Flow bias using weighted P/C ---
            # Use the moneyness-weighted ratio for classification
            if flow_confidence == "LOW":
                # Suppress bias on dead volume days
                flow_bias = "BALANCED (low volume)"
            elif weighted_pc > 1.5:
                flow_bias = "HEAVY PUT FLOW (bearish)"
            elif weighted_pc > 1.1:
                flow_bias = "PUT-LEANING"
            elif weighted_pc < 0.65:
                flow_bias = "HEAVY CALL FLOW (bullish)"
            elif weighted_pc < 0.9:
                flow_bias = "CALL-LEANING"
            else:
                flow_bias = "BALANCED"

            # Build signal string
            top3 = alerts[:3]
            signal_parts = []
            for a in top3:
                signal_parts.append(
                    f"{a['type']} {a['strike']:.0f} {a['moneyness']} — "
                    f"{a['volume']:,} vol "
                    f"({a['vol_oi_ratio']:.1f}x OI) ${a['notional']:.1f}M"
                )

            signal = " | ".join(signal_parts) if signal_parts else "No unusual flow"

            result = {
                "alerts": alerts[:5],
                "total_alerts": len(alerts),
                "flow_bias": flow_bias,
                "pc_ratio": round(raw_pc, 2),
                "weighted_pc_ratio": round(weighted_pc, 2),
                "flow_confidence": flow_confidence,
                "vol_vs_avg": round(vol_ratio, 2),
                "total_call_vol": int(total_call_vol),
                "total_put_vol": int(total_put_vol),
                "call_notional_m": round(total_call_notional / 1_000_000, 1),
                "put_notional_m": round(total_put_notional / 1_000_000, 1),
                "signal": signal,
                "dte_mult": round(dte_mult, 2),
            }

            if alerts:
                logger.info(
                    f"Flow: {len(alerts)} unusual prints | {flow_bias} "
                    f"[{flow_confidence}] DTE×{dte_mult} | "
                    f"wP/C {weighted_pc:.2f} (raw {raw_pc:.2f}) | "
                    f"vol {vol_ratio:.1f}x avg | "
                    f"Top: {signal_parts[0] if signal_parts else 'N/A'}"
                )
            return result

        except Exception as e:
            logger.warning(f"Flow scanner failed: {e}")
            return self._empty()

    def _empty(self) -> dict:
        return {
            "alerts": [], "total_alerts": 0, "flow_bias": "N/A",
            "pc_ratio": 0, "weighted_pc_ratio": 0, "flow_confidence": "N/A",
            "vol_vs_avg": 0, "total_call_vol": 0, "total_put_vol": 0,
            "call_notional_m": 0, "put_notional_m": 0,
            "signal": "No flow data available",
        }


# =====================================================================
# 4. INTRADAY VOL REGIME SHIFT DETECTION
# =====================================================================

def calc_vol_regime_shift(md) -> dict:
    """
    Track realized vol in rolling 30-min windows.
    Flag when vol suddenly expands (breakout incoming) or compresses (range-bound).

    Uses 1-min bars to calculate standard deviation of returns in windows.
    Compares current window to prior window and to session average.
    """
    try:
        bars_1m = md.es_1m
        if bars_1m.empty or len(bars_1m) < 60:
            return _vol_shift_default()

        # Calculate 1-min returns
        closes = bars_1m["Close"].dropna()
        returns = closes.pct_change().dropna()

        if len(returns) < 60:
            return _vol_shift_default()

        # Skip first 15 min of session — opening volatility is noise, not signal
        if hasattr(bars_1m.index, 'hour'):
            from datetime import time as dtime
            rth_bars = bars_1m[bars_1m.index.time >= dtime(9, 45)]
            if len(rth_bars) > 0 and len(bars_1m) - len(rth_bars) < 15:
                # We're in the first 15 min of RTH — skip vol shift detection
                return _vol_shift_default()

        # 30-min windows (30 bars of 1-min data)
        window = 30
        current_window = returns.iloc[-window:]
        prior_window = returns.iloc[-2 * window:-window] if len(returns) >= 2 * window else returns.iloc[:window]

        # Realized vol (annualized std of returns)
        current_vol = float(current_window.std() * np.sqrt(252 * 390) * 100)  # 390 min/day
        prior_vol = float(prior_window.std() * np.sqrt(252 * 390) * 100)
        session_vol = float(returns.std() * np.sqrt(252 * 390) * 100)

        # Vol ratio (current vs prior)
        vol_ratio = current_vol / prior_vol if prior_vol > 0 else 1.0

        # Vol vs session average
        vol_vs_session = current_vol / session_vol if session_vol > 0 else 1.0

        # Classify
        shift = "STABLE"
        signal = "Vol steady. No regime change."
        alert = False

        if vol_ratio > 2.5:
            shift = "🔴 EXTREME EXPANSION"
            signal = (
                f"Vol exploded {vol_ratio:.1f}x in 30 min! "
                "Prior signals may be invalidated. Tighten stops or flatten. "
                "Wait for vol to settle before new entries."
            )
            alert = True
        elif vol_ratio > 1.8:
            shift = "⚠️ VOL EXPANSION"
            signal = (
                f"Vol expanded {vol_ratio:.1f}x. "
                "Breakout regime — momentum plays favored, wider stops needed."
            )
            alert = True
        elif vol_ratio < 0.4:
            shift = "VOL COMPRESSION"
            signal = (
                f"Vol collapsed to {vol_ratio:.1f}x prior. "
                "Range-bound — sell premium, tighter stops. Coiling for breakout?"
            )
        elif vol_ratio < 0.6:
            shift = "VOL DECLINING"
            signal = "Vol fading. Range-bound conditions developing."

        # Expected range from current vol
        es_price = md.current_price
        daily_move = es_price * (current_vol / 100) / np.sqrt(252)
        hourly_move = daily_move / np.sqrt(6.5)  # 6.5 hr trading day

        result = {
            "shift": shift,
            "current_vol": round(current_vol, 1),
            "prior_vol": round(prior_vol, 1),
            "session_vol": round(session_vol, 1),
            "vol_ratio": round(vol_ratio, 2),
            "vol_vs_session": round(vol_vs_session, 2),
            "signal": signal,
            "alert": alert,
            "expected_hourly_range": round(hourly_move, 1),
        }

        if alert:
            logger.info(f"⚡ VOL SHIFT: {shift} | Ratio {vol_ratio:.1f}x | "
                         f"Current {current_vol:.1f}% vs Prior {prior_vol:.1f}%")

        return result

    except Exception as e:
        logger.warning(f"Vol regime shift calc failed: {e}")
        return _vol_shift_default()


def _vol_shift_default() -> dict:
    return {
        "shift": "N/A", "current_vol": 0, "prior_vol": 0,
        "session_vol": 0, "vol_ratio": 1.0, "vol_vs_session": 1.0,
        "signal": "Insufficient data.", "alert": False,
        "expected_hourly_range": 0,
    }



# =====================================================================
# 4. CROSS-SIGNAL DIVERGENCE DETECTOR
# =====================================================================

def calc_divergence_score(md, metrics: dict) -> dict:
    """
    Cross-check all indicators against price action.
    When many indicators disagree with the current price trend,
    a reversal is likely. High divergence = danger zone.

    Checks:
    1. Price direction (1h trend) vs each indicator's direction
    2. Counts agreements and disagreements
    3. Scores severity and identifies which indicators diverge
    """
    try:
        es_1h = md.es_1h
        if es_1h.empty or len(es_1h) < 5:
            return _divergence_default()

        # Determine price trend (last 3 1h bars) — use percentage, not fixed points
        recent = es_1h["Close"].iloc[-3:]
        price_change = float(recent.iloc[-1] - recent.iloc[0])
        price_pct = (price_change / float(recent.iloc[0]) * 100) if float(recent.iloc[0]) > 0 else 0
        price_dir = "UP" if price_pct > 0.03 else "DOWN" if price_pct < -0.03 else "FLAT"

        # Collect each indicator's directional signal
        signals = {}

        # Fractal
        fractal = metrics.get("fractal", {})
        proj = fractal.get("projection", None)
        if proj:
            f_dir = getattr(proj, "direction", "") if not isinstance(proj, dict) else proj.get("direction", "")
            if "BULL" in str(f_dir).upper():
                signals["Fractal"] = "UP"
            elif "BEAR" in str(f_dir).upper():
                signals["Fractal"] = "DOWN"
            else:
                signals["Fractal"] = "FLAT"

        # MTF Momentum
        mtf = metrics.get("mtf_momentum", {})
        alignment = mtf.get("alignment", "")
        if "BULL" in alignment.upper():
            signals["MTF"] = "UP"
        elif "BEAR" in alignment.upper():
            signals["MTF"] = "DOWN"
        else:
            signals["MTF"] = "FLAT"

        # Cumulative Delta
        delta_bias = metrics.get("cum_delta_bias", "")
        if "BUY" in delta_bias.upper() or "POSITIVE" in delta_bias.upper():
            signals["Delta"] = "UP"
        elif "SELL" in delta_bias.upper() or "NEGATIVE" in delta_bias.upper():
            signals["Delta"] = "DOWN"
        else:
            signals["Delta"] = "FLAT"

        # TICK Proxy
        tick = metrics.get("tick_proxy", {})
        tick_ext = tick.get("extreme", "")
        if "POSITIVE" in tick_ext.upper() or "BULLISH" in tick_ext.upper():
            signals["TICK"] = "UP"
        elif "NEGATIVE" in tick_ext.upper() or "BEARISH" in tick_ext.upper():
            signals["TICK"] = "DOWN"
        else:
            signals["TICK"] = "FLAT"

        # Breadth
        breadth = metrics.get("breadth", "")
        if "STRONG BUYING" in breadth.upper() or "POSITIVE" in breadth.upper():
            signals["Breadth"] = "UP"
        elif "STRONG SELLING" in breadth.upper() or "NEGATIVE" in breadth.upper():
            signals["Breadth"] = "DOWN"
        else:
            signals["Breadth"] = "FLAT"

        # Cross-asset correlation
        corr = metrics.get("cross_corr", {})
        corr_regime = corr.get("regime", "")
        if "RISK-ON" in corr_regime.upper():
            signals["Correlation"] = "UP"
        elif "RISK-OFF" in corr_regime.upper():
            signals["Correlation"] = "DOWN"
        else:
            signals["Correlation"] = "FLAT"

        # VIX direction (inverse to ES)
        vix_term = metrics.get("vix_term", {})
        vix_signal = vix_term.get("signal", "")
        if "BACKWARDATION" in vix_signal.upper() or "FEAR" in vix_signal.upper():
            signals["VIX"] = "DOWN"  # VIX fear = bearish for ES
        elif "CONTANGO" in vix_signal.upper():
            signals["VIX"] = "UP"  # VIX calm = bullish for ES
        else:
            signals["VIX"] = "FLAT"

        # GEX regime
        gex = metrics.get("gex_regime", {})
        gex_playbook = gex.get("playbook", "")
        if gex_playbook == "MEAN-REVERT" and price_dir == "UP":
            signals["GEX"] = "DOWN"  # Long gamma + price up = expect pull back
        elif gex_playbook == "MEAN-REVERT" and price_dir == "DOWN":
            signals["GEX"] = "UP"
        elif gex_playbook == "MOMENTUM":
            signals["GEX"] = price_dir  # Short gamma confirms momentum
        else:
            signals["GEX"] = "FLAT"

        # Count divergences
        if price_dir == "FLAT":
            return {
                "score": 0, "max_score": len(signals),
                "divergent": [], "aligned": list(signals.keys()),
                "price_direction": "FLAT",
                "severity": "N/A (price flat)",
                "signal": "Price flat — divergence detection not applicable.",
                "alert": False,
            }

        # Weight major signals higher: MTF/TICK count 2x, others 1x
        _weights = {"MTF": 2, "TICK": 2, "FLOW": 1.5, "VIX": 1, "GEX": 1, "RVOL": 0.5}
        divergent = []
        aligned = []
        div_score = 0.0
        total_weight = 0.0
        for name, sig_dir in signals.items():
            if sig_dir == "FLAT":
                continue
            w = _weights.get(name, 1)
            total_weight += w
            if sig_dir != price_dir:
                divergent.append(name)
                div_score += w
            else:
                aligned.append(name)

        active_signals = len(divergent) + len(aligned)
        div_pct = div_score / total_weight * 100 if total_weight > 0 else 0

        # Severity
        alert = False
        if div_pct >= 75:
            severity = "🔴 EXTREME DIVERGENCE"
            signal = (
                f"Price trending {price_dir} but {len(divergent)}/{active_signals} indicators disagree: "
                f"{', '.join(divergent)}. "
                "High probability reversal — tighten stops or flatten."
            )
            alert = True
        elif div_pct >= 50:
            severity = "⚠️ HIGH DIVERGENCE"
            signal = (
                f"{len(divergent)}/{active_signals} indicators diverge from price ({price_dir}): "
                f"{', '.join(divergent)}. "
                "Caution — reduce size."
            )
            alert = True
        elif div_pct >= 30:
            severity = "MODERATE DIVERGENCE"
            signal = f"Some divergence: {', '.join(divergent)}. Monitor."
        else:
            severity = "LOW (ALIGNED)"
            signal = f"Indicators mostly agree with price trend ({price_dir})."

        result = {
            "score": div_score,
            "max_score": active_signals,
            "divergence_pct": round(div_pct, 0),
            "divergent": divergent,
            "aligned": aligned,
            "price_direction": price_dir,
            "severity": severity,
            "signal": signal,
            "alert": alert,
        }

        if alert:
            logger.info(f"🚨 DIVERGENCE: {severity} — {div_score}/{active_signals} indicators vs price {price_dir}")

        return result

    except Exception as e:
        logger.warning(f"Divergence calc failed: {e}")
        return _divergence_default()


def _divergence_default() -> dict:
    return {
        "score": 0, "max_score": 0, "divergence_pct": 0,
        "divergent": [], "aligned": [], "price_direction": "N/A",
        "severity": "N/A", "signal": "Insufficient data.", "alert": False,
    }
