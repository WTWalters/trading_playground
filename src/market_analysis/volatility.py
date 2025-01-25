# src/market_analysis/volatility.py

from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
import talib
from .base import MarketAnalyzer, VolatilityMetrics, AnalysisConfig

class VolatilityAnalyzer(MarketAnalyzer):
    """
    Analyzes market volatility using multiple metrics

    Implements various volatility measures including:
    - Historical volatility
    - Average True Range (ATR)
    - Volatility regimes
    - Z-score based outlier detection
    """

    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ) -> Dict[str, Union[float, str, pd.Series]]:
        """
        Calculate comprehensive volatility metrics

        Args:
            data: OHLCV DataFrame
            additional_metrics: Optional metrics from other analyzers

        Returns:
            Dictionary containing:
            - metrics: VolatilityMetrics object
            - historical_volatility: Time series
            - normalized_atr: Time series
            - volatility_zscore: Current z-score
        """
        if not self._validate_input(data):
            return self._empty_result()

        try:
            # Validate data quality
            if data['close'].isna().any():
                return self._empty_result()

            # Calculate log returns and historical volatility
            returns = np.log(data['close'] / data['close'].shift(1))
            hist_vol = (returns.rolling(self.config.volatility_window).std() *
                       np.sqrt(252)) / 100  # Annualized

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
            norm_atr = (atr / data['close']).fillna(0)

            # Calculate current metrics
            current_vol = float(hist_vol.iloc[-1] or 0)
            current_norm_atr = float(norm_atr.iloc[-1] or 0)

            # Calculate z-score and regime
            vol_mean = hist_vol.mean()
            vol_std = hist_vol.std()
            zscore = float((current_vol - vol_mean) / vol_std if vol_std != 0 else 0)

            # Determine volatility regime
            regime = self._classify_regime(zscore)

            # Create metrics object
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

        except Exception as e:
            self.logger.error(f"Volatility analysis failed: {str(e)}")
            return self._empty_result()

    def _classify_regime(self, zscore: float) -> str:
        """
        Classify volatility regime based on z-score

        Args:
            zscore: Current volatility z-score

        Returns:
            String indicating volatility regime
        """
        if abs(zscore) > self.config.outlier_std_threshold:
            return "high_volatility" if zscore > 0 else "low_volatility"
        return "normal_volatility"

    def _empty_result(self) -> Dict:
        """
        Create empty result structure

        Returns:
            Dictionary with default/empty values
        """
        return {
            'metrics': VolatilityMetrics(
                historical_volatility=0.0,
                normalized_atr=0.0,
                volatility_regime="unknown",
                zscore=0.0
            ),
            'historical_volatility': pd.Series(),
            'normalized_atr': pd.Series(),
            'volatility_zscore': 0.0
        }
