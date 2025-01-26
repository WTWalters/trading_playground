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

class TrendDirection(Enum):
    """Trend direction classification"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"

@dataclass
class TrendMetrics(AnalysisMetrics):
    """Container for trend analysis metrics"""
    
    # Parent class fields with defaults
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.95
    regime: MarketRegime = MarketRegime.UNKNOWN

    # Core trend measures
    direction: TrendDirection = TrendDirection.SIDEWAYS
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
            isinstance(self.direction, TrendDirection),
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

class TrendAnalyzer(MarketAnalyzer):
    """Advanced market trend analysis system"""

    def __init__(self, config: AnalysisConfig):
        """Initialize analyzer with configuration"""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self._trend_history: List[TrendMetrics] = []
        self._last_analysis: Optional[TrendMetrics] = None

    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ) -> Dict[str, Union[MarketRegime, pd.Series, float, TrendMetrics]]:
        """Perform comprehensive trend analysis"""
        try:
            if not self._validate_input(data):
                return self._get_empty_results()

            timestamp = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now()

            # Calculate core trend indicators
            adx_metrics = self._calculate_directional_movement(data)
            price_metrics = self._calculate_price_metrics(data)
            structure_metrics = self._identify_price_structure(data)
            confirmation_metrics = self._analyze_trend_confirmation(
                data, additional_metrics
            )

            # Determine trend characteristics
            direction = self._classify_direction(
                adx_metrics, price_metrics
            )
            strength = self._classify_strength(adx_metrics['adx'])
            regime = self._classify_regime(
                direction,
                adx_metrics,
                confirmation_metrics
            )

            # Create metrics object
            metrics = TrendMetrics(
                timestamp=timestamp,
                metrics=price_metrics,
                regime=regime,
                direction=direction,
                strength=strength,
                momentum=price_metrics['momentum'],
                adx=adx_metrics['adx'],
                dmi_plus=adx_metrics['dmi_plus'],
                dmi_minus=adx_metrics['dmi_minus'],
                price_slope=price_metrics['slope'],
                price_velocity=price_metrics['velocity'],
                price_acceleration=price_metrics['acceleration'],
                support_level=structure_metrics['support'],
                resistance_level=structure_metrics['resistance'],
                trend_age=self._calculate_trend_age(regime),
                volume_trend_correlation=confirmation_metrics['volume_correlation'],
                swing_magnitude=confirmation_metrics['swing_magnitude']
            )

            # Update history
            self._update_history(metrics)

            return {
                'metrics': metrics,
                'signals': self._generate_signals(metrics),
                'levels': structure_metrics,
                'indicators': {
                    'adx': adx_metrics['adx_series'],
                    'dmi_plus': adx_metrics['dmi_plus_series'],
                    'dmi_minus': adx_metrics['dmi_minus_series']
                },
                'regime': regime
            }

        except Exception as e:
            self.logger.error(f"Trend analysis failed: {str(e)}")
            return self._get_empty_results()

    def _calculate_directional_movement(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Union[float, pd.Series]]:
        """Calculate directional movement indicators"""
        try:
            # Calculate ADX and DMI
            period = min(14, len(data) - 1)  # Ensure period doesn't exceed data length
            adx = pd.Series(
                talib.ADX(
                    data['high'].values,
                    data['low'].values,
                    data['close'].values,
                    timeperiod=period
                ),
                index=data.index
            )

            dmi_plus = pd.Series(
                talib.PLUS_DI(
                    data['high'].values,
                    data['low'].values,
                    data['close'].values,
                    timeperiod=period
                ),
                index=data.index
            )

            dmi_minus = pd.Series(
                talib.MINUS_DI(
                    data['high'].values,
                    data['low'].values,
                    data['close'].values,
                    timeperiod=period
                ),
                index=data.index
            )

            return {
                'adx': float(adx.iloc[-1]),
                'dmi_plus': float(dmi_plus.iloc[-1]),
                'dmi_minus': float(dmi_minus.iloc[-1]),
                'adx_series': adx,
                'dmi_plus_series': dmi_plus,
                'dmi_minus_series': dmi_minus
            }

        except Exception as e:
            self.logger.error(f"Directional movement calculation failed: {str(e)}")
            return {
                'adx': 0.0, 'dmi_plus': 0.0, 'dmi_minus': 0.0,
                'adx_series': pd.Series(), 'dmi_plus_series': pd.Series(),
                'dmi_minus_series': pd.Series()
            }

    def _calculate_price_metrics(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate price-based metrics"""
        try:
            close = data['close']
            period = min(self.config.trend_window, len(data) - 1)

            # Calculate slope (momentum)
            slope = (close - close.shift(period)) / period

            # Calculate velocity (rate of change)
            velocity = close.pct_change(period)

            # Calculate acceleration
            acceleration = velocity.diff()

            # Calculate momentum
            mom_period = min(self.config.momentum_window, len(data) - 1)
            momentum = talib.MOM(close.values, timeperiod=mom_period)

            return {
                'slope': float(slope.iloc[-1]),
                'velocity': float(velocity.iloc[-1]),
                'acceleration': float(acceleration.iloc[-1]),
                'momentum': float(momentum[-1] if len(momentum) > 0 else 0.0)
            }

        except Exception as e:
            self.logger.error(f"Price metrics calculation failed: {str(e)}")
            return {
                'slope': 0.0,
                'velocity': 0.0,
                'acceleration': 0.0,
                'momentum': 0.0
            }

    def _identify_price_structure(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Identify key price levels"""
        try:
            period = min(self.config.trend_window, len(data) - 1)
            window_data = data.tail(period)

            # Calculate pivot points
            pivot = (window_data['high'] + window_data['low'] + window_data['close']) / 3

            # Find support and resistance
            support = float(window_data['low'].min())
            resistance = float(window_data['high'].max())

            # Calculate intermediate levels
            r1 = 2 * pivot - window_data['low']
            s1 = 2 * pivot - window_data['high']

            return {
                'pivot': float(pivot.iloc[-1]),
                'support': support,
                'resistance': resistance,
                'r1': float(r1.iloc[-1]),
                's1': float(s1.iloc[-1])
            }

        except Exception as e:
            self.logger.error(f"Structure identification failed: {str(e)}")
            return {
                'pivot': 0.0,
                'support': 0.0,
                'resistance': 0.0,
                'r1': 0.0,
                's1': 0.0
            }

    def _analyze_trend_confirmation(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict]
    ) -> Dict[str, float]:
        """Analyze trend confirmation factors"""
        try:
            # Calculate volume-price correlation
            min_periods = min(self.config.trend_window, len(data) - 1)
            volume_correlation = data['volume'].corr(
                data['close'],
                min_periods=min_periods
            )

            # Calculate swing magnitude
            swing = (data['high'] - data['low']).mean()

            return {
                'volume_correlation': float(volume_correlation if not pd.isna(volume_correlation) else 0.0),
                'swing_magnitude': float(swing)
            }

        except Exception as e:
            self.logger.error(f"Trend confirmation analysis failed: {str(e)}")
            return {
                'volume_correlation': 0.0,
                'swing_magnitude': 0.0
            }

    def _classify_direction(
        self,
        adx_metrics: Dict[str, float],
        price_metrics: Dict[str, float]
    ) -> TrendDirection:
        """Classify trend direction"""
        if adx_metrics['adx'] > 20:
            if adx_metrics['dmi_plus'] > adx_metrics['dmi_minus']:
                return TrendDirection.UP
            else:
                return TrendDirection.DOWN
        else:
            return TrendDirection.SIDEWAYS

    def _classify_regime(
        self,
        direction: TrendDirection,
        adx_metrics: Dict[str, float],
        confirmation_metrics: Dict[str, float]
    ) -> MarketRegime:
        """Classify market regime"""
        if direction == TrendDirection.UP and adx_metrics['adx'] > 20:
            return MarketRegime.TRENDING_UP
        elif direction == TrendDirection.DOWN and adx_metrics['adx'] > 20:
            return MarketRegime.TRENDING_DOWN
        elif confirmation_metrics['swing_magnitude'] > self.config.volatility_threshold:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.RANGING

    def _classify_strength(self, adx: float) -> TrendStrength:
        """Classify trend strength"""
        if adx < 15:
            return TrendStrength.VERY_WEAK
        elif adx < 25:
            return TrendStrength.WEAK
        elif adx < 35:
            return TrendStrength.MODERATE
        elif adx < 45:
            return TrendStrength.STRONG
        else:
            return TrendStrength.VERY_STRONG

    def _calculate_trend_age(self, current_regime: MarketRegime) -> int:
        """Calculate age of current trend"""
        if not self._trend_history:
            return 1

        age = 1
        for metrics in reversed(self._trend_history[:-1]):
            if metrics.regime == current_regime:
                age += 1
            else:
                break
        return age

    def _generate_signals(self, metrics: TrendMetrics) -> Dict[str, bool]:
        """Generate trading signals"""
        return {
            'trend_change': self._detect_trend_change(),
            'breakout': metrics.swing_magnitude > self.config.volatility_threshold,
            'momentum_shift': abs(metrics.momentum) > self.config.momentum_threshold,
            'volume_confirmation': metrics.volume_trend_correlation > 0.7
        }

    def _detect_trend_change(self) -> bool:
        """Detect trend regime change"""
        if len(self._trend_history) < 2:
            return False
        return (
            self._trend_history[-1].regime !=
            self._trend_history[-2].regime
        )

    def _update_history(self, metrics: TrendMetrics) -> None:
        """Update trend history"""
        self._trend_history.append(metrics)
        self._last_analysis = metrics

        # Maintain history length
        max_history = 1000
        if len(self._trend_history) > max_history:
            self._trend_history = self._trend_history[-max_history:]

    def _get_empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'metrics': None,
            'signals': {},
            'levels': {},
            'indicators': {},
            'regime': MarketRegime.UNKNOWN
        }