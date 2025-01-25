# src/market_analysis/volatility.py

from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
import talib
from .base import MarketAnalyzer, VolatilityMetrics, AnalysisConfig

class VolatilityAnalyzer(MarketAnalyzer):
    """
    Analyzes market volatility using multiple metrics

    Features:
    - Historical volatility calculation
    - Average True Range (ATR) analysis
    - Volatility regime classification
    - Z-score based outlier detection
    - Normalized volatility measures
    """

    async def analyze(
        self,
        data: pd.DataFrame,
        additional_metrics: Optional[Dict] = None
    ) -> Dict[str, Union[float, str, pd.Series]]:
        """
        Calculate comprehensive volatility metrics

        Args:
            data: OHLCV DataFrame with market data
            additional_metrics: Optional metrics from other analyzers

        Returns:
            Dictionary containing:
            - metrics: VolatilityMetrics instance with current readings
            - historical_volatility: Full time series of volatility
            - normalized_atr: Time series of normalized ATR
            - volatility_zscore: Current volatility z-score

        Raises:
            ValueError: If data validation fails
        """
        # Validate input data
        if not self._validate_input(data):
            return {}

        try:
            # Check for data quality
            if data['close'].isna().any():
                self.logger.warning("NaN values detected in close prices")
                return {}

            # Calculate returns and historical volatility
            returns = np.log(data['close'] / data['close'].shift(1))
            hist_vol = (returns.rolling(self.config.volatility_window).std() *
                       np.sqrt(252)) / 100  # Annualize and convert to decimal

            # Calculate ATR and normalize it
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

            # Get current metrics
            current_vol = float(hist_vol.iloc[-1] or 0)
            current_norm_atr = float(norm_atr.iloc[-1] or 0)

            # Calculate z-score for regime determination
            vol_mean = hist_vol.mean()
            vol_std = hist_vol.std()
            zscore = float((current_vol - vol_mean) / vol_std if vol_std != 0 else 0)

            # Classify volatility regime
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
            return {}  # Return empty dict on error

    def _classify_regime(self, zscore: float) -> str:
        """
        Classify current volatility regime based on z-score

        Args:
            zscore: Standard score of current volatility

        Returns:
            String indicating volatility regime:
            - "high_volatility": Above threshold
            - "low_volatility": Below negative threshold
            - "normal_volatility": Within thresholds
        """
        if abs(zscore) > self.config.outlier_std_threshold:
            return "high_volatility" if zscore > 0 else "low_volatility"
        return "normal_volatility"
