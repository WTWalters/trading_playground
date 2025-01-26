# src/market_analysis/volatility.py

from typing import Dict, Optional, Union, List, Tuple, Any
from enum import Enum
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import talib
from dataclasses import dataclass, field
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
    
    # Initialize parent class fields
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.95
    regime: VolatilityRegime = VolatilityRegime.NORMAL

    # Core volatility measures
    historical_volatility: float = 0.0
    implied_volatility: Optional[float] = None
    normalized_atr: float = 0.0
    parkinson_volatility: float = 0.0

    # Relative measures
    volatility_zscore: float = 0.0
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

class VolatilityAnalyzer(MarketAnalyzer):
    """Advanced market volatility analysis system"""

    def __init__(self, config: AnalysisConfig):
        """Initialize analyzer with configuration"""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self._volatility_history: List[VolatilityMetrics] = []
        self._last_analysis: Optional[VolatilityMetrics] = None

    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ) -> Dict[str, Union[float, str, pd.Series, VolatilityMetrics]]:
        """Perform comprehensive volatility analysis"""
        try:
            if not self._validate_input(data):
                return self._get_empty_results()

            timestamp = data.index[-1]

            # Calculate core volatility measures
            hist_vol = self._calculate_historical_volatility(data)
            norm_atr = self._calculate_normalized_atr(data)
            park_vol = self._calculate_parkinson_volatility(data)

            # Calculate volume-based measures
            vol_metrics = self._analyze_volume_volatility(data)

            # Calculate relative measures
            zscore, percentile = self._calculate_relative_measures(hist_vol)

            # Determine regime
            regime = self._classify_regime(zscore)

            # Calculate trend-relative measures
            trend_vol = self._calculate_trend_relative_volatility(
                data,
                hist_vol,
                additional_metrics
            )

            # Create metrics dictionary
            metrics_dict = {
                'historical_volatility': float(hist_vol.iloc[-1]),
                'implied_volatility': None,
                'normalized_atr': float(norm_atr.iloc[-1]),
                'parkinson_volatility': float(park_vol.iloc[-1]),
                'volatility_zscore': zscore,
                'volatility_regime': regime.value,
                'percentile_rank': percentile,
                'volume_volatility': vol_metrics['volume_volatility'],
                'volume_zscore': vol_metrics['volume_zscore'],
                'trend_relative_volatility': trend_vol['relative_vol'],
                'volatility_trend': trend_vol['vol_trend']
            }

            # Create metrics object
            metrics = VolatilityMetrics(
                timestamp=timestamp,
                metrics=metrics_dict,
                regime=regime,
                confidence=0.95,
                historical_volatility=float(hist_vol.iloc[-1]),
                normalized_atr=float(norm_atr.iloc[-1]),
                parkinson_volatility=float(park_vol.iloc[-1]),
                volatility_zscore=zscore,
                percentile_rank=percentile,
                volume_volatility=vol_metrics['volume_volatility'],
                volume_zscore=vol_metrics['volume_zscore'],
                trend_relative_volatility=trend_vol['relative_vol'],
                volatility_trend=trend_vol['vol_trend']
            )

            # Update history
            self._update_history(metrics)

            return {
                'metrics': metrics,
                'historical_series': {
                    'historical_volatility': hist_vol,
                    'normalized_atr': norm_atr,
                    'parkinson_volatility': park_vol
                },
                'signals': self._generate_signals(metrics),
                'regimes': self._get_regime_changes(),
                'regime': regime
            }

        except Exception as e:
            self.logger.error(f"Volatility analysis failed: {str(e)}")
            return self._get_empty_results()

    def _calculate_historical_volatility(
        self,
        data: pd.DataFrame,
        method: str = 'log'
    ) -> pd.Series:
        """Calculate historical volatility"""
        try:
            if method == 'log':
                returns = np.log(data['close'] / data['close'].shift(1))
            else:
                returns = data['close'].pct_change()

            volatility = returns.rolling(
                window=self.config.volatility_window
            ).std() * np.sqrt(252)

            return volatility.fillna(0)

        except Exception as e:
            self.logger.error(f"Historical volatility calculation failed: {str(e)}")
            return pd.Series(index=data.index)

    def _calculate_normalized_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate normalized Average True Range"""
        try:
            atr = pd.Series(
                talib.ATR(
                    data['high'].values,
                    data['low'].values,
                    data['close'].values,
                    timeperiod=self.config.volatility_window
                ),
                index=data.index
            )
            return (atr / data['close']).fillna(0)

        except Exception as e:
            self.logger.error(f"ATR calculation failed: {str(e)}")
            return pd.Series(index=data.index)

    def _calculate_parkinson_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility (using high-low range)"""
        try:
            log_hl = np.log(data['high'] / data['low'])
            park_vol = np.sqrt(
                log_hl.rolling(self.config.volatility_window).apply(
                    lambda x: sum(x * x) / (4 * len(x) * np.log(2))
                )
            ) * np.sqrt(252)
            return park_vol.fillna(0)

        except Exception as e:
            self.logger.error(f"Parkinson volatility calculation failed: {str(e)}")
            return pd.Series(index=data.index)

    def _analyze_volume_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume-based volatility measures"""
        try:
            vol_std = data['volume'].rolling(
                self.config.volatility_window
            ).std()
            vol_mean = data['volume'].rolling(
                self.config.volatility_window
            ).mean()

            current_vol_std = float(vol_std.iloc[-1])
            current_vol_mean = float(vol_mean.iloc[-1])

            volume_volatility = (
                current_vol_std / current_vol_mean
                if current_vol_mean != 0 else 0
            )

            volume_zscore = (
                (data['volume'].iloc[-1] - current_vol_mean) / current_vol_std
                if current_vol_std != 0 else 0
            )

            return {
                'volume_volatility': volume_volatility,
                'volume_zscore': volume_zscore
            }

        except Exception as e:
            self.logger.error(f"Volume volatility analysis failed: {str(e)}")
            return {'volume_volatility': 0.0, 'volume_zscore': 0.0}

    def _calculate_relative_measures(
        self,
        volatility: pd.Series
    ) -> Tuple[float, float]:
        """Calculate relative volatility measures"""
        try:
            current_vol = volatility.iloc[-1]
            historical_mean = volatility.mean()
            historical_std = volatility.std()

            zscore = (
                (current_vol - historical_mean) / historical_std
                if historical_std != 0 else 0
            )

            percentile = float(
                percentileofscore(volatility.dropna(), current_vol)
                if len(volatility.dropna()) > 0 else 50
            )

            return zscore, percentile

        except Exception as e:
            self.logger.error(f"Relative measures calculation failed: {str(e)}")
            return 0.0, 50.0

    def _classify_regime(self, zscore: float) -> VolatilityRegime:
        """Classify volatility regime based on z-score"""
        if zscore < -2:
            return VolatilityRegime.EXTREMELY_LOW
        elif zscore < -1:
            return VolatilityRegime.LOW
        elif zscore < 1:
            return VolatilityRegime.NORMAL
        elif zscore < 2:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREMELY_HIGH

    def _calculate_trend_relative_volatility(
        self,
        data: pd.DataFrame,
        volatility: pd.Series,
        additional_metrics: Optional[Dict]
    ) -> Dict[str, float]:
        """Calculate volatility relative to trend"""
        try:
            trend_strength = (
                additional_metrics.get('trend', {})
                .get('strength', 1.0)
                if additional_metrics else 1.0
            )

            vol_trend = (
                volatility.iloc[-1] / volatility.iloc[-self.config.volatility_window]
                if len(volatility) >= self.config.volatility_window else 1.0
            )

            return {
                'relative_vol': float(volatility.iloc[-1] / trend_strength),
                'vol_trend': vol_trend
            }

        except Exception as e:
            self.logger.error(f"Trend-relative calculation failed: {str(e)}")
            return {'relative_vol': 0.0, 'vol_trend': 1.0}

    def _generate_signals(self, metrics: VolatilityMetrics) -> Dict[str, bool]:
        """Generate trading signals based on volatility"""
        return {
            'volatility_breakout': metrics.volatility_zscore > 2.0,
            'volatility_compression': metrics.volatility_zscore < -1.0,
            'volume_spike': metrics.volume_zscore > 2.0,
            'regime_change': self._detect_regime_change()
        }

    def _detect_regime_change(self) -> bool:
        """Detect change in volatility regime"""
        if len(self._volatility_history) < 2:
            return False
        return (
            self._volatility_history[-1].regime !=
            self._volatility_history[-2].regime
        )

    def _update_history(self, metrics: VolatilityMetrics) -> None:
        """Update volatility metrics history"""
        self._volatility_history.append(metrics)
        self._last_analysis = metrics

        # Maintain history length
        max_history = 1000
        if len(self._volatility_history) > max_history:
            self._volatility_history = self._volatility_history[-max_history:]

    def _get_regime_changes(self) -> List[Dict[str, Any]]:
        """Get history of regime changes"""
        changes = []
        for i in range(1, len(self._volatility_history)):
            if self._volatility_history[i].regime != self._volatility_history[i-1].regime:
                changes.append({
                    'timestamp': self._volatility_history[i].timestamp,
                    'from_regime': self._volatility_history[i-1].regime,
                    'to_regime': self._volatility_history[i].regime,
                    'zscore': self._volatility_history[i].volatility_zscore
                })
        return changes

    def _get_empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'metrics': None,
            'historical_series': {},
            'signals': {},
            'regimes': []
        }