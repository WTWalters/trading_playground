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
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class AnalysisConfig(BaseModel):
    volatility_window: int = 20
    trend_strength_threshold: float = 0.1
    volatility_threshold: float = 0.02
    outlier_std_threshold: float = 3.0
    minimum_data_points: int = 20

@dataclass
class VolatilityMetrics:
    historical_volatility: float
    normalized_atr: float
    volatility_regime: str
    zscore: float

class MarketAnalyzer(ABC):
    """Abstract base class for market analysis components"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ) -> Dict[str, Union[float, str, pd.Series]]:
        """Run analysis on market data"""
        pass

    def _validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data structure"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Missing required columns: {required_columns}")
            return False

        if len(data) < self.config.minimum_data_points:
            self.logger.error(f"Insufficient data points: {len(data)}")
            return False

        return True

class AnalysisOrchestrator:
    """Coordinates multiple analysis components"""

    def __init__(
        self,
        analyzers: Dict[str, MarketAnalyzer],
        config: AnalysisConfig
    ):
        self.analyzers = analyzers
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def run_analysis(
        self,
        data: pd.DataFrame,
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """Run specified analyses on market data"""
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
