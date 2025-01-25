# src/market_analysis/volatility.py

from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
import talib
from .base import MarketAnalyzer, VolatilityMetrics, AnalysisConfig

class VolatilityAnalyzer(MarketAnalyzer):
    """
    Analyzes market volatility using multiple metrics including historical volatility,
    Average True Range (ATR), and volatility regimes.
    """

    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ) -> Dict[str, Union[float, str, pd.Series]]:
        """
        Calculate volatility metrics for the given market data.

        Args:
            data: DataFrame with OHLCV data (open, high, low, close, volume)
            additional_metrics: Optional dictionary of metrics from other analyzers

        Returns:
            Dictionary containing:
                - metrics: VolatilityMetrics object with current readings
                - historical_volatility: Full time series of historical volatility
                - normalized_atr: Full time series of normalized ATR
                - volatility_zscore: Current volatility z-score
        """
        if not self._validate_input(data):
            return {}

        try:
            # Check for NaN values in critical columns
            if data['close'].isna().any():
                return {}

            # Calculate returns and historical volatility
            returns = np.log(data['close'] / data['close'].shift(1))
            hist_vol = (returns.rolling(self.config.volatility_window).std() *
                       np.sqrt(252)) / 100  # Annualized and converted to decimal

            # Calculate ATR and Normalized ATR
            atr = pd.Series(
                talib.ATR(
                    data['high'].values,
                    data['low'].values,
                    data['close'].values,
                    timeperiod=self.config.volatility_window
                ),
                index=data.index
            )
            # Handle NaN values in ATR calculation
            norm_atr = (atr / data['close']).fillna(0)

            # Calculate current metrics and z-score
            current_vol = float(hist_vol.iloc[-1] or 0)
            current_norm_atr = float(norm_atr.iloc[-1] or 0)

            # Calculate z-score for regime determination
            vol_mean = hist_vol.mean()
            vol_std = hist_vol.std()
            # Adjust the volatility threshold calculation
            zscore = float((current_vol - vol_mean) / vol_std if vol_std != 0 else 0)
            # Make the regime detection more sensitive
            if abs(zscore) > self.config.outlier_std_threshold:
                regime = "high_volatility" if zscore > 0 else "low_volatility"
            else:
                regime = "normal_volatility"

            # Create metrics object with current readings
            metrics = VolatilityMetrics(
                historical_volatility=current_vol,
                normalized_atr=current_norm_atr,
                volatility_regime=regime,
                zscore=zscore
            )

            return {
                'metrics': metrics,
                'historical_volatility': hist_vol,
                'normalized_atr': norm_atr,
                'volatility_zscore': zscore
            }

        except ValueError as e:
            self.logger.error(f"Volatility calculation error: {str(e)}")
            return {}
        except KeyError as e:
            self.logger.error(f"Missing required data column: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error in volatility analysis: {str(e)}")
            return {}
