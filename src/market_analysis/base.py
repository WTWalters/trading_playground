# src/market_analysis/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pydantic import BaseModel

class MarketRegime(Enum):
    """Market regime classification enum"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class AnalysisConfig(BaseModel):
    """
    Configuration settings for market analysis

    Attributes:
        volatility_window: Period for volatility calculations
        trend_strength_threshold: Minimum threshold for trend confirmation
        volatility_threshold: Threshold for volatility regime changes
        outlier_std_threshold: Standard deviations for outlier detection
        minimum_data_points: Minimum required data points for analysis
    """
    volatility_window: int = 20
    trend_strength_threshold: float = 0.1
    volatility_threshold: float = 0.02
    outlier_std_threshold: float = 3.0
    minimum_data_points: int = 20

@dataclass
class VolatilityMetrics:
    """
    Container for volatility-related metrics

    Attributes:
        historical_volatility: Historical volatility measure
        normalized_atr: Normalized Average True Range
        volatility_regime: Current volatility regime classification
        zscore: Standard score of current volatility
    """
    historical_volatility: float
    normalized_atr: float
    volatility_regime: str
    zscore: float

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get attribute value safely

        Args:
            key: Attribute name
            default: Default value if attribute doesn't exist

        Returns:
            Attribute value or default
        """
        return getattr(self, key, default)

class MarketAnalyzer(ABC):
    """
    Abstract base class for market analysis components

    Provides common functionality and interface for all market analyzers
    """

    def __init__(self, config: AnalysisConfig):
        """
        Initialize analyzer with configuration

        Args:
            config: Analysis configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ) -> Dict[str, Union[float, str, pd.Series]]:
        """
        Run analysis on market data

        Args:
            data: Market data DataFrame
            additional_metrics: Optional metrics from other analyzers

        Returns:
            Dictionary containing analysis results
        """
        pass

    def _validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data structure and sufficiency

        Args:
            data: Input DataFrame to validate

        Returns:
            bool: True if data is valid
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Missing required columns: {required_columns}")
            return False

        if len(data) < max(2, self.config.minimum_data_points):
            self.logger.error(f"Insufficient data points: {len(data)}")
            return False

        return True

class AnalysisOrchestrator:
    """
    Coordinates execution of multiple market analyzers

    Handles the sequencing and data flow between different analysis components
    """

    def __init__(
        self,
        analyzers: Dict[str, MarketAnalyzer],
        config: AnalysisConfig
    ):
        """
        Initialize orchestrator with analyzers

        Args:
            analyzers: Dictionary of analyzer instances
            config: Analysis configuration
        """
        self.analyzers = analyzers
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def run_analysis(
        self,
        data: pd.DataFrame,
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Run specified analyses on market data

        Args:
            data: Market data DataFrame
            analysis_types: Optional list of analysis types to run

        Returns:
            Dictionary containing results from all analyzers
        """
        try:
            results = {}
            analysis_types = analysis_types or list(self.analyzers.keys())

            for analysis_type in analysis_types:
                if analysis_type not in self.analyzers:
                    self.logger.warning(f"Unknown analyzer: {analysis_type}")
                    continue

                analyzer = self.analyzers[analysis_type]
                results[analysis_type] = await analyzer.analyze(
                    data,
                    additional_metrics=results
                )

            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {}
