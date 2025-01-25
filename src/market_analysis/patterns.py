# src/market_analysis/patterns.py

from typing import Dict, Optional, Union, List
import pandas as pd
import numpy as np
import talib
from .base import MarketAnalyzer, AnalysisConfig

class PatternAnalyzer(MarketAnalyzer):
    """
    Detects and analyzes candlestick patterns and measures their effectiveness
    in different market conditions.
    """

    def __init__(self, config: AnalysisConfig):
        """
        Initialize pattern analyzer with configuration and pattern detection parameters.

        Args:
            config: Analysis configuration object containing window sizes and thresholds
        """
        super().__init__(config)
        self.config.minimum_data_points = 5  # Override minimum data points for patterns

        # Define pattern detection parameters
        self.pattern_params = {
            'DOJI': {'body_ratio': 0.1},  # Maximum body/range ratio for doji
            'ENGULFING': {'body_ratio': 1.5}  # Minimum ratio for engulfing
        }

        # Initialize TA-Lib pattern functions
        self.pattern_functions = {
            'DOJI': talib.CDLDOJI,
            'ENGULFING': talib.CDLENGULFING,
            'HAMMER': talib.CDLHAMMER,
            'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
            'MORNING_STAR': talib.CDLMORNINGSTAR,
            'EVENING_STAR': talib.CDLEVENINGSTAR
        }

    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ) -> Dict[str, Union[Dict, List, pd.Series]]:
        """
        Detect and analyze candlestick patterns in the given market data.

        Args:
            data: DataFrame with OHLCV data
            additional_metrics: Optional dictionary of metrics from other analyzers

        Returns:
            Dictionary containing:
                - patterns: Dictionary of detected patterns
                - recent_patterns: List of recently detected patterns
                - success_rates: Pattern success rates if trend data is available
        """
        if not self._validate_input(data):
            return {}

        try:
            # Convert data to proper type and calculate basic measurements
            ohlc = data[['open', 'high', 'low', 'close']].astype(float)
            body_range = abs(ohlc['close'] - ohlc['open'])
            total_range = ohlc['high'] - ohlc['low']

            patterns = {}
            recent_patterns = []
            success_rates = {}

            # Detect standard patterns using TA-Lib
            for name, func in self.pattern_functions.items():
                pattern = pd.Series(
                    func(
                        ohlc['open'].values,
                        ohlc['high'].values,
                        ohlc['low'].values,
                        ohlc['close'].values
                    ),
                    index=data.index
                )
                patterns[name] = pattern

                # Check for recent pattern occurrences
                recent_signals = pattern.tail(5)
                for idx, value in recent_signals.items():
                    if value != 0:
                        recent_patterns.append({
                            'pattern': name,
                            'date': idx,
                            'signal': 'bullish' if value > 0 else 'bearish'
                        })

            # Add custom pattern detection
            is_doji = body_range / total_range < self.pattern_params['DOJI']['body_ratio']
            patterns['CUSTOM_DOJI'] = pd.Series(np.where(is_doji, 1, 0), index=data.index)

            # Calculate success rates if trend data is available
            if additional_metrics and 'trend_analysis' in additional_metrics:
                for name, pattern in patterns.items():
                    success_rates[name] = self._calculate_pattern_success(
                        pattern,
                        ohlc['close'],
                        lookforward=10
                    )

            return {
                'patterns': patterns,
                'recent_patterns': recent_patterns,
                'success_rates': success_rates
            }

        except Exception as e:
            self.logger.error(f"Pattern detection failed: {str(e)}")
            return {}

    def _calculate_pattern_success(
        self,
        pattern: pd.Series,
        prices: pd.Series,
        lookforward: int = 10
    ) -> Dict[str, float]:
        """
        Calculate success rates for pattern signals.

        Args:
            pattern: Series of pattern signals
            prices: Series of close prices
            lookforward: Number of bars to look forward for success/failure

        Returns:
            Dictionary containing:
                - bullish_rate: Success rate of bullish signals
                - bearish_rate: Success rate of bearish signals
                - total_signals: Total number of pattern occurrences
        """
        try:
            # Identify bullish and bearish signals
            bullish_signals = pattern > 0
            bearish_signals = pattern < 0

            # Calculate forward returns
            forward_returns = prices.shift(-lookforward) / prices - 1

            # Calculate success rates
            bullish_success = (
                (forward_returns[bullish_signals] > 0).sum() /
                bullish_signals.sum() if bullish_signals.any() else 0
            )
            bearish_success = (
                (forward_returns[bearish_signals] < 0).sum() /
                bearish_signals.sum() if bearish_signals.any() else 0
            )

            return {
                'bullish_rate': float(bullish_success),
                'bearish_rate': float(bearish_success),
                'total_signals': int((bullish_signals | bearish_signals).sum())
            }

        except Exception as e:
            self.logger.error(f"Success rate calculation failed: {str(e)}")
            return {
                'bullish_rate': 0.0,
                'bearish_rate': 0.0,
                'total_signals': 0
            }
