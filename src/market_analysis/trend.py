# src/market_analysis/trend.py

from typing import Dict, Optional, Union, List, Tuple, Any
from enum import Enum
import pandas as pd
import numpy as np
import talib
from dataclasses import dataclass, field
from datetime import datetime
import logging
from .base import MarketAnalyzer, MarketRegime, AnalysisConfig, AnalysisMetrics

class TrendStrength(Enum):
    """Trend strength classification"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TrendMetrics(AnalysisMetrics):
    """Container for trend analysis metrics"""

    # Initialize parent class fields
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.95
    regime: MarketRegime = MarketRegime.UNKNOWN

    # Core trend measures
    strength: TrendStrength = TrendStrength.MODERATE
    momentum: float = 0.0

    # Technical indicators
    adx: float = 0.0
    dmi_plus: float = 0.0
    dmi_minus: float = 0.0

    # Price action metrics
    price_slope: float = 0.0
    price_velocity: float = 0.0
    price_acceleration: float = 0.0

    # Structure metrics
    support_level: float = 0.0
    resistance_level: float = 0.0
    trend_age: int = 1

    # Confirmation metrics
    volume_trend_correlation: float = 0.0
    swing_magnitude: float = 0.0

    def validate(self) -> bool:
        """Validate metric values"""
        return all([
            isinstance(self.regime, MarketRegime),
            isinstance(self.strength, TrendStrength),
            isinstance(self.momentum, (int, float)),
            all(isinstance(x, (int, float)) for x in [
                self.adx, self.dmi_plus, self.dmi_minus,
                self.price_slope, self.price_velocity,
                self.price_acceleration, self.support_level,
                self.resistance_level, self.trend_age,
                self.volume_trend_correlation, self.swing_magnitude
            ])
        ])

[REST OF FILE CONTENT REMAINS THE SAME]