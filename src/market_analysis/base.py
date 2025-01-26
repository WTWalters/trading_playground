# src/market_analysis/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, TypeVar, Protocol
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pydantic import BaseModel, Field

# Type variables for generic types
T = TypeVar('T')
MetricsType = TypeVar('MetricsType', bound='BaseMetrics')

class MarketRegime(Enum):
    """
    Market regime classification

    Defines different market states:
    - TRENDING_UP: Clear upward trend
    - TRENDING_DOWN: Clear downward trend
    - RANGING: Sideways movement
    - VOLATILE: High volatility
    - UNKNOWN: Undefined state
    """
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class AnalysisConfig(BaseModel):
    """
    Configuration settings for market analysis

    Attributes:
        volatility_window: Period for volatility calculations (default: 10)
        trend_strength_threshold: Minimum threshold for trend confirmation (default: 0.1)
        volatility_threshold: Threshold for volatility regime changes (default: 0.02)
        outlier_std_threshold: Standard deviations for outlier detection (default: 3.0)
        minimum_data_points: Minimum required data points (default: 10)
    """
    volatility_window: int = Field(default=10, ge=5, le=50)
    trend_strength_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    volatility_threshold: float = Field(default=0.02, ge=0.0, le=0.1)
    outlier_std_threshold: float = Field(default=3.0, ge=1.0, le=5.0)
    minimum_data_points: int = Field(default=10, ge=5, le=100)

    class Config:
        """Pydantic config"""
        validate_assignment = True
        arbitrary_types_allowed = True

class BaseMetrics(Protocol):
    """Protocol defining interface for metrics classes"""
    def get(self, key: str, default: Any = None) -> Any:
        """Get metric value safely"""
        ...

@dataclass
class VolatilityMetrics:
    """
    Container for volatility-related metrics

    Attributes:
        historical_volatility: Rolling volatility measure
        normalized_atr: ATR normalized by price
        volatility_regime: Current volatility classification
        zscore: Standardized volatility score
    """
    historical_volatility: float
    normalized_atr: float
    volatility_regime: str
    zscore: float

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get attribute value safely with default

        Args:
            key: Attribute name to retrieve
            default: Default value if attribute doesn't exist

        Returns:
            Attribute value or default
        """
        return getattr(self, key, default)

    def validate(self) -> bool:
        """
        Validate metric values

        Returns:
            bool: True if all metrics are valid
        """
        return all([
            isinstance(self.historical_volatility, (int, float)),
            isinstance(self.normalized_atr, (int, float)),
            isinstance(self.volatility_regime, str),
            isinstance(self.zscore, (int, float))
        ])

class MarketAnalyzer(ABC):
    """
    Abstract base class for market analyzers

    Provides:
    - Common initialization
    - Data validation
    - Abstract analysis interface
    - Logging functionality
    """

    def __init__(self, config: AnalysisConfig):
        """
        Initialize analyzer with configuration

        Args:
            config: Analysis parameters and settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ) -> Dict[str, Union[float, str, pd.Series, BaseMetrics]]:
        """
        Analyze market data

        Args:
            data: OHLCV DataFrame
            additional_metrics: Optional metrics from other analyzers

        Returns:
            Dictionary containing analysis results
        """
        pass

    def _validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data structure and sufficiency

        Args:
            data: OHLCV DataFrame to validate

        Returns:
            bool: True if data is valid

        Checks:
        - Required columns present
        - Sufficient data points
        - No missing values
        """
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']

            # Check required columns
            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                self.logger.error(f"Missing columns: {missing}")
                return False

            # Check data points
            if len(data) < self.config.minimum_data_points:
                self.logger.error(
                    f"Insufficient data points: {len(data)} "
                    f"(minimum: {self.config.minimum_data_points})"
                )
                return False

            # Check for NaN values
            if data[required_columns].isna().any().any():
                self.logger.error("Data contains missing values")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False

class AnalysisOrchestrator:
    """
    Coordinates multiple market analyzers

    Features:
    - Analyzer management
    - Sequential execution
    - Results aggregation
    - Error handling
    """

    def __init__(
        self,
        analyzers: Dict[str, MarketAnalyzer],
        config: AnalysisConfig
    ):
        """
        Initialize orchestrator

        Args:
            analyzers: Dictionary of analyzer instances
            config: Analysis configuration
        """
        self.analyzers = analyzers
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    async def run_analysis(
        self,
        data: pd.DataFrame,
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Run specified market analyses

        Args:
            data: OHLCV DataFrame
            analysis_types: Optional list of analyses to run

        Returns:
            Dictionary containing results from all analyzers
        """
        try:
            results = {}
            analysis_types = analysis_types or list(self.analyzers.keys())

            # Run each analyzer sequentially
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
            self.logger.error(f"Analysis orchestration failed: {str(e)}")
            return {}

    def get_available_analyzers(self) -> List[str]:
        """Get list of available analyzers"""
        return list(self.analyzers.keys())

    def validate_config(self) -> bool:
        """Validate configuration settings"""
        return all(
            isinstance(analyzer, MarketAnalyzer)
            for analyzer in self.analyzers.values()
        )
