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
        self.config.minimum_data_points = 20  # Increased minimum data points for reliable pattern detection

        # Define pattern detection parameters with more sensitive thresholds
        self.pattern_params = {
            'DOJI': {'body_ratio': 0.1},  # Maximum body/range ratio for doji
            'ENGULFING': {'body_ratio': 1.5},  # Minimum ratio for engulfing
            'HAMMER': {'body_ratio': 0.3, 'shadow_ratio': 2.0},  # Body to lower shadow ratio
            'SHOOTING_STAR': {'body_ratio': 0.3, 'shadow_ratio': 2.0}  # Body to upper shadow ratio
        }

        # Initialize TA-Lib pattern functions with optional parameters
        self.pattern_functions = {
            'DOJI': talib.CDLDOJI,
            'ENGULFING': talib.CDLENGULFING,
            'HAMMER': talib.CDLHAMMER,
            'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
            'MORNING_STAR': lambda o, h, l, c: talib.CDLMORNINGSTAR(o, h, l, c, penetration=0.3),
            'EVENING_STAR': lambda o, h, l, c: talib.CDLEVENINGSTAR(o, h, l, c, penetration=0.3)
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
            # Check for NaN values in critical columns
            if data[['open', 'high', 'low', 'close']].isna().any().any():
                return {}

            # Convert data to proper type and calculate basic measurements
            ohlc = data[['open', 'high', 'low', 'close']].astype(float)
            body_range = abs(ohlc['close'] - ohlc['open'])
            total_range = ohlc['high'] - ohlc['low']
            upper_shadow = ohlc['high'] - ohlc[['open', 'close']].max(axis=1)
            lower_shadow = ohlc[['open', 'close']].min(axis=1) - ohlc['low']

            patterns = {}
            recent_patterns = []
            success_rates = {}

            # Custom pattern detection
            # Doji pattern (very small body compared to shadows)
            is_doji = body_range / total_range < self.pattern_params['DOJI']['body_ratio']
            patterns['CUSTOM_DOJI'] = pd.Series(np.where(is_doji, 1, 0), index=data.index)

            # Bullish engulfing pattern
            prev_body_range = body_range.shift(1)
            is_bullish_engulfing = (
                (ohlc['open'] < ohlc['close']) &  # Current candle is bullish
                (ohlc['open'].shift(1) > ohlc['close'].shift(1)) &  # Previous candle is bearish
                (body_range > prev_body_range * self.pattern_params['ENGULFING']['body_ratio'])
            )
            patterns['CUSTOM_ENGULFING'] = pd.Series(np.where(is_bullish_engulfing, 1, 0), index=data.index)

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

            # Process recent patterns
            for name, pattern in patterns.items():
                recent_signals = pattern.tail(5)
                for idx, value in recent_signals.items():
                    if value != 0:
                        signal_type = 'bullish' if value > 0 else 'bearish'
                        if name.startswith('CUSTOM_'):
                            signal_type = 'bullish' if name == 'CUSTOM_ENGULFING' else 'neutral'
                        recent_patterns.append({
                            'pattern': name.replace('CUSTOM_', ''),
                            'date': idx,
                            'signal': signal_type
                        })

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

            # Calculate success rates with minimum signal threshold
            min_signals = 3  # Minimum number of signals required for valid rate
            bullish_success = (
                (forward_returns[bullish_signals] > 0).sum() /
                bullish_signals.sum() if bullish_signals.sum() >= min_signals else 0
            )
            bearish_success = (
                (forward_returns[bearish_signals] < 0).sum() /
                bearish_signals.sum() if bearish_signals.sum() >= min_signals else 0
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
