# src/market_analysis/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, TypeVar, Protocol
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pydantic import BaseModel, Field, validator

# Type variables for generic types
T = TypeVar('T')
MetricsType = TypeVar('MetricsType', bound='BaseMetrics')

class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> 'MarketRegime':
        """Convert string to MarketRegime enum"""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN

class AnalysisConfig(BaseModel):
    """Configuration settings for market analysis"""

    # Time windows
    volatility_window: int = Field(default=20, ge=5, le=100)
    trend_window: int = Field(default=20, ge=5, le=100)
    momentum_window: int = Field(default=14, ge=5, le=50)

    # Thresholds
    trend_strength_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    volatility_threshold: float = Field(default=0.02, ge=0.0, le=0.1)
    momentum_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    outlier_std_threshold: float = Field(default=3.0, ge=1.0, le=5.0)

    # Data requirements
    minimum_data_points: int = Field(default=30, ge=10, le=200)

    # Analysis parameters
    use_log_returns: bool = Field(default=True)
    volatility_scaling: bool = Field(default=True)

    @validator('trend_window')
    def validate_trend_window(cls, v, values):
        """Ensure trend window is appropriate"""
        if 'volatility_window' in values and v < values['volatility_window']:
            raise ValueError("Trend window should be >= volatility window")
        return v

    class Config:
        """Pydantic model configuration"""
        validate_assignment = True
        arbitrary_types_allowed = True
        extra = "allow"

class BaseMetrics(Protocol):
    """Protocol for metrics classes"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        ...

    def validate(self) -> bool:
        """Validate metric values"""
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """Get metric value with default"""
        ...

@dataclass
class AnalysisMetrics:
    """Base class for analysis metrics"""

    timestamp: datetime
    metrics: Dict[str, Any]
    confidence: float
    regime: Optional[MarketRegime] = MarketRegime.UNKNOWN

    def __post_init__(self):
        """Convert any metric values to proper types"""
        if isinstance(self.regime, str):
            self.regime = MarketRegime.from_string(self.regime)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'timestamp': self.timestamp,
            'metrics': self.metrics,
            'regime': self.regime.value,
            'confidence': self.confidence
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to metrics"""
        if key in self.metrics:
            return self.metrics[key]
        return getattr(self, key, None)

    def get(self, key: str, default: Any = None) -> Any:
        """Get metric value with default"""
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

class MarketAnalyzer(ABC):
    """Abstract base class for market analyzers"""

    def __init__(self, config: AnalysisConfig):
        """Initialize analyzer"""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._last_analysis: Optional[AnalysisMetrics] = None
        self._analysis_history: List[AnalysisMetrics] = []

    @abstractmethod
    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ) -> AnalysisMetrics:
        """
        Analyze market data

        Args:
            data: OHLCV DataFrame
            additional_metrics: Optional metrics from other analyzers

        Returns:
            Analysis metrics

        Raises:
            ValueError: If data validation fails
        """
        if not self._validate_input(data):
            raise ValueError("Invalid input data")

        # Store analysis timestamp
        timestamp = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now()

        return AnalysisMetrics(
            timestamp=timestamp,
            metrics={},
            regime=MarketRegime.UNKNOWN,
            confidence=0.0
        )

    def _create_metrics(
        self,
        timestamp: datetime,
        metrics_dict: Dict[str, Any],
        regime: MarketRegime = MarketRegime.UNKNOWN,
        confidence: float = 0.95
    ) -> AnalysisMetrics:
        """Helper to create metrics with proper structure"""
        return AnalysisMetrics(
            timestamp=timestamp,
            metrics=metrics_dict,
            regime=regime,
            confidence=confidence
        )

    def _validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']

            # Check columns
            if not all(col in data.columns for col in required_columns):
                missing = set(required_columns) - set(data.columns)
                self.logger.error(f"Missing columns: {missing}")
                return False

            # Check for NaN values
            if data[required_columns].isna().any().any():
                self.logger.error("Data contains missing values")
                return False

            # Special case for analysis of single data point
            if len(data) == 1:
                return True

            # Check data points for time series analysis
            if len(data) < self.config.minimum_data_points:
                self.logger.error(
                    f"Insufficient data points: {len(data)} "
                    f"(minimum: {self.config.minimum_data_points})"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False

    def get_last_analysis(self) -> Optional[AnalysisMetrics]:
        """Get most recent analysis results"""
        return self._last_analysis

    def get_analysis_history(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[AnalysisMetrics]:
        """
        Get historical analysis results

        Args:
            start: Optional start datetime
            end: Optional end datetime

        Returns:
            List of analysis metrics
        """
        history = self._analysis_history

        if start:
            history = [m for m in history if m.timestamp >= start]
        if end:
            history = [m for m in history if m.timestamp <= end]

        return history

class AnalysisOrchestrator:
    """Coordinates multiple market analyzers"""

    def __init__(
        self,
        analyzers: Dict[str, MarketAnalyzer],
        config: AnalysisConfig
    ):
        """Initialize orchestrator"""
        self.analyzers = analyzers
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._last_results: Dict[str, AnalysisMetrics] = {}

    async def run_analysis(
        self,
        data: pd.DataFrame,
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, AnalysisMetrics]:
        """
        Run market analysis

        Args:
            data: OHLCV DataFrame
            analysis_types: Optional list of analyses to run

        Returns:
            Dictionary of analysis results
        """
        try:
            results = {}
            analysis_types = analysis_types or list(self.analyzers.keys())

            # Sequential analysis
            for analysis_type in analysis_types:
                if analysis_type not in self.analyzers:
                    self.logger.warning(f"Unknown analyzer: {analysis_type}")
                    continue

                analyzer = self.analyzers[analysis_type]
                results[analysis_type] = await analyzer.analyze(
                    data,
                    additional_metrics=results
                )

            self._last_results = results
            return results

        except Exception as e:
            self.logger.error(f"Analysis orchestration failed: {str(e)}")
            return {}

    def get_available_analyzers(self) -> List[str]:
        """Get list of available analyzers"""
        return list(self.analyzers.keys())

    def get_last_results(self) -> Dict[str, AnalysisMetrics]:
        """Get most recent analysis results"""
        return self._last_results

    def validate_setup(self) -> bool:
        """Validate analyzer setup"""
        return all(
            isinstance(analyzer, MarketAnalyzer)
            for analyzer in self.analyzers.values()
        )
