"""
Typed structures for market metrics.

Replaces the untyped metrics dict with structured dataclasses.
Catches typos at development time and enables IDE autocomplete.

Usage:
    These types are used for documentation and optional type checking.
    The metrics dict is still used at runtime for backward compatibility,
    but these types define the expected shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VWAPLevels:
    upper_2: float = 0.0
    upper_1: float = 0.0
    lower_1: float = 0.0
    lower_2: float = 0.0


@dataclass
class GapAnalysis:
    gap_size: float = 0.0
    gap_pct: float = 0.0
    fill_status: str = "N/A"
    summary: str = "N/A"


@dataclass
class OpeningType:
    type: str = "N/A"
    bias: str = "N/A"
    description: str = ""


@dataclass
class RVOLData:
    rvol: float = 1.0
    status: str = "NORMAL"


@dataclass
class VIXTermStructure:
    vix: float = 0.0
    vix9d: float = 0.0
    structure: str = "N/A"
    signal: str = "N/A"


@dataclass
class MTFMomentum:
    alignment: str = "N/A"
    score: int = 0
    signal: str = "N/A"
    timeframes: Dict[str, str] = field(default_factory=dict)


@dataclass
class VPOCMigration:
    migration: str = "N/A"
    trend: str = "N/A"
    signal: str = "N/A"
    naked_pocs: List[Dict] = field(default_factory=list)


@dataclass
class TickProxy:
    tick_proxy: float = 50.0
    extreme: str = "N/A"
    signal: str = "N/A"
    cumulative: float = 0.0
    bullish_extremes: int = 0
    bearish_extremes: int = 0


@dataclass
class GEXRegime:
    regime: str = "N/A"
    flip_level: float = 0.0
    distance_to_flip: float = 0.0
    playbook: str = "MIXED"
    signal: str = "N/A"


@dataclass
class FlowData:
    flow_bias: str = "N/A"
    bias: str = "N/A"
    pc_ratio: float = 0.0
    total_call_vol: int = 0
    total_put_vol: int = 0
    call_notional_m: float = 0.0
    put_notional_m: float = 0.0
    signal: str = "N/A"
    alerts: List[Dict] = field(default_factory=list)


@dataclass
class LiquiditySweeps:
    active_sweep: str = "NONE"
    signal: str = "No sweeps"
    cluster_signal: str = "N/A"
    swing_highs: List[Dict] = field(default_factory=list)
    swing_lows: List[Dict] = field(default_factory=list)


@dataclass
class VolShift:
    shift: str = "N/A"
    current_vol: float = 0.0
    prior_vol: float = 0.0
    vol_ratio: float = 1.0
    expected_hourly_range: float = 0.0
    signal: str = "N/A"
    alert: bool = False


@dataclass
class DivergenceData:
    severity: str = "N/A"
    score: int = 0
    max_score: int = 0
    price_direction: str = "N/A"
    divergent: List[str] = field(default_factory=list)
    aligned: List[str] = field(default_factory=list)
    signal: str = "N/A"
    alert: bool = False


@dataclass
class RegimeData:
    regime: str = "N/A"
    confidence_mod: int = 0
    flat_threshold: int = 60
    adjustments: List[str] = field(default_factory=list)
    summary: str = "Standard regime."


@dataclass
class PriorDay:
    prev_high: float = 0.0
    prev_low: float = 0.0
    prev_close: float = 0.0
    prev_range: float = 0.0


@dataclass
class AnchoredVWAPs:
    weekly_vwap: float = 0.0
    monthly_vwap: float = 0.0
    swing_vwap: float = 0.0
    swing_label: str = "Swing"
    convergence: str = "N/A"
    convergence_level: int = 0
    signal: str = "N/A"


@dataclass
class FractalResult:
    """Result from the fractal engine analysis."""
    projection: Any = None  # ForwardProjection
    prompt_text: str = ""
    match_count: int = 0
    top_matches: List[Dict] = field(default_factory=list)
    total_days_scanned: int = 0


@dataclass
class GammaDetail:
    net_gamma: str = ""
    source: str = ""
    distance_to_call: str = ""
    distance_to_put: str = ""
    call_wall_2: float = 0.0
    put_wall_2: float = 0.0


@dataclass
class StructureData:
    fractal: str = "N/A"
    vsa: str = "N/A"
    wick: str = "N/A"
    atr: float = 0.0


@dataclass
class IBData:
    ib_high: float = 0.0
    ib_low: float = 0.0
    ib_status: str = "N/A"


@dataclass
class CycleMetrics:
    """
    Full metrics payload for one analysis cycle.

    This is a typed representation of the metrics dict.
    All fields have sensible defaults, so you can create a partial instance.
    The existing dict-based metrics are still used at runtime for backward
    compatibility — this class serves as documentation and for new code.
    """
    # Session context
    session: str = "N/A"
    data_source: str = "yfinance"

    # Core indicators
    vwap_val: float = 0.0
    vwap_status: str = "N/A"
    vwap_levels: VWAPLevels = field(default_factory=VWAPLevels)

    gap: GapAnalysis = field(default_factory=GapAnalysis)
    opening_type: OpeningType = field(default_factory=OpeningType)
    rvol: RVOLData = field(default_factory=RVOLData)
    vix_term: VIXTermStructure = field(default_factory=VIXTermStructure)
    mtf_momentum: MTFMomentum = field(default_factory=MTFMomentum)
    vpoc_migration: VPOCMigration = field(default_factory=VPOCMigration)
    tick_proxy: TickProxy = field(default_factory=TickProxy)

    # Advanced features
    gex_regime: GEXRegime = field(default_factory=GEXRegime)
    flow_data: FlowData = field(default_factory=FlowData)
    liq_sweeps: LiquiditySweeps = field(default_factory=LiquiditySweeps)
    vol_shift: VolShift = field(default_factory=VolShift)
    divergence: DivergenceData = field(default_factory=DivergenceData)

    # Regime
    regime: RegimeData = field(default_factory=RegimeData)

    # Levels
    prior: PriorDay = field(default_factory=PriorDay)
    anchored_vwaps: AnchoredVWAPs = field(default_factory=AnchoredVWAPs)
    vpoc: Dict = field(default_factory=dict)
    ib: IBData = field(default_factory=IBData)

    # Gamma
    g_call: float = 0.0
    g_put: float = 0.0
    gamma_detail: GammaDetail = field(default_factory=GammaDetail)

    # Internals
    breadth: str = "N/A"
    cum_delta_bias: str = "N/A"
    cum_delta_val: float = 0.0
    structure: StructureData = field(default_factory=StructureData)

    # Fractal
    fractal: FractalResult = field(default_factory=FractalResult)

    # Time-of-day context
    tod_context: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> 'CycleMetrics':
        """
        Create a CycleMetrics from the traditional metrics dict.
        Unknown keys are silently ignored for backward compatibility.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert back to a plain dict for backward compatibility."""
        from dataclasses import asdict
        return asdict(self)
