# src/market_analysis/volatility.py

from typing import Dict, Optional, Union, List, Tuple, Any
from enum import Enum
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore  # Add this for percentile calculation
import talib
from dataclasses import dataclass
from datetime import datetime
import logging
from .base import MarketAnalyzer, AnalysisConfig, AnalysisMetrics, MarketRegime

class VolatilityRegime(Enum):
    """Volatility regime classification"""
    EXTREMELY_LOW = "extremely_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREMELY_HIGH = "extremely_high"

@dataclass
class VolatilityMetrics(AnalysisMetrics):
    """Container for volatility analysis metrics"""

    # Core volatility measures
    historical_volatility: float = 0.0
    implied_volatility: Optional[float] = None
    normalized_atr: float = 0.0
    parkinson_volatility: float = 0.0

    # Relative measures
    volatility_zscore: float = 0.0
    regime: VolatilityRegime = VolatilityRegime.NORMAL
    percentile_rank: float = 50.0

    # Volume-based measures
    volume_volatility: float = 0.0
    volume_zscore: float = 0.0

    # Trend-relative measures
    trend_relative_volatility: float = 0.0
    volatility_trend: float = 0.0

    def validate(self) -> bool:
        """Validate metric values"""
        return all([
            isinstance(self.historical_volatility, (int, float)),
            isinstance(self.normalized_atr, (int, float)),
            isinstance(self.volatility_zscore, (int, float)),
            isinstance(self.regime, VolatilityRegime)
        ])

[REST OF FILE REMAINS THE SAME]