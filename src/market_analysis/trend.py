# src/market_analysis/trend.py

from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
import talib
from .base import MarketAnalyzer, MarketRegime, AnalysisConfig

class TrendAnalyzer(MarketAnalyzer):
    """Analyzes market trends using multiple indicators and classifies market regimes"""

    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ) -> Dict[str, Union[MarketRegime, pd.Series, float]]:
        """
        Calculate trend metrics and determine market regime

        Args:
            data: DataFrame with OHLCV data
            additional_metrics: Optional dictionary of metrics from other analyzers

        Returns:
            Dictionary containing:
                - regime: Current market regime classification
                - adx: Average Directional Index series
                - price_slope: Price momentum/slope series
                - ema_short: Short-term EMA
                - ema_long: Long-term EMA
                - trend_strength: Current ADX value
                - trend_momentum: Current price slope
        """
        if not self._validate_input(data):
            return {}

        try:
            # Comprehensive NaN check for required columns
            required_cols = ['high', 'low', 'close']
            if (data[required_cols].isna().any().any() or
                data[required_cols].isnull().any().any()):
                self.logger.error("NaN or null values found in required columns")
                return {}

            # Calculate EMAs for trend direction
            try:
                ema_short = pd.Series(
                    talib.EMA(data['close'].values.astype(float), timeperiod=20),
                    index=data.index
                )
                ema_long = pd.Series(
                    talib.EMA(data['close'].values.astype(float), timeperiod=50),
                    index=data.index
                )
            except Exception as e:
                self.logger.error(f"EMA calculation failed: {str(e)}")
                return {}

            # Calculate ADX for trend strength
            try:
                adx = pd.Series(
                    talib.ADX(
                        data['high'].values.astype(float),
                        data['low'].values.astype(float),
                        data['close'].values.astype(float),
                        timeperiod=14
                    ),
                    index=data.index
                )
            except Exception as e:
                self.logger.error(f"ADX calculation failed: {str(e)}")
                return {}

            # Calculate slopes for trend momentum
            slope_window = self.config.volatility_window
            try:
                price_slope = (
                    data['close'].astype(float).diff(slope_window) /
                    data['close'].astype(float).shift(slope_window)
                )
            except Exception as e:
                self.logger.error(f"Slope calculation failed: {str(e)}")
                return {}

            # Get current values, handling potential NaN
            try:
                current_slope = float(price_slope.iloc[-1])
                current_adx = float(adx.iloc[-1])
            except (IndexError, ValueError) as e:
                self.logger.error(f"Failed to get current values: {str(e)}")
                return {}

            # Determine trend regime with more nuanced thresholds
            if pd.isna(current_adx) or pd.isna(current_slope):
                regime = MarketRegime.UNKNOWN
            elif current_adx > 25:  # Strong trend
                if current_slope > self.config.trend_strength_threshold:
                    regime = MarketRegime.TRENDING_UP
                elif current_slope < -self.config.trend_strength_threshold:
                    regime = MarketRegime.TRENDING_DOWN
                else:
                    regime = MarketRegime.RANGING
            else:  # Weak trend
                regime = MarketRegime.RANGING

            # Check volatility override
            if additional_metrics and 'volatility_analysis' in additional_metrics:
                vol_metrics = additional_metrics['volatility_analysis'].get('metrics')
                if vol_metrics and vol_metrics.volatility_regime == 'high_volatility':
                    regime = MarketRegime.VOLATILE

            # Ensure all series are free of NaN values
            adx = adx.fillna(0)
            price_slope = price_slope.fillna(0)
            ema_short = ema_short.fillna(method='ffill').fillna(0)
            ema_long = ema_long.fillna(method='ffill').fillna(0)

            return {
                'regime': regime,
                'adx': adx,
                'price_slope': price_slope,
                'ema_short': ema_short,
                'ema_long': ema_long,
                'trend_strength': float(current_adx),
                'trend_momentum': float(current_slope)
            }

        except ValueError as e:
            self.logger.error(f"Trend calculation error: {str(e)}")
            return {}
        except KeyError as e:
            self.logger.error(f"Missing required data column: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error in trend analysis: {str(e)}")
            return {}
